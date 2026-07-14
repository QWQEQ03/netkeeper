"""Safe, injectable OpenAI-compatible translation client; offline by default."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import yaml

from netkeeper_sim.api import API_REGISTRY, ApiCall, ApiRequest, execute, export_json_schema, validate_request
from netkeeper_sim.api.models import OptimizationRequest
from netkeeper_sim.schemas import JointAction, NetworkSnapshot
from netkeeper_sim.simulator import UnifiedNetworkEnvironment

PROMPT_VERSION = "netkeeper-translation-v1"
RESULT_SCHEMA = {"type": "object", "additionalProperties": False, "oneOf": [
    {"properties": {"status": {"const": "accepted"}, "calls": {"type": "array", "minItems": 1}, "need_optimization": {"type": "boolean"}}, "required": ["status", "calls", "need_optimization"]},
    {"properties": {"status": {"const": "rejected"}, "calls": {"type": "array", "maxItems": 0}, "error": {"type": "object"}}, "required": ["status", "calls", "error"]},
]}


class TranslationFailure(Exception):
    def __init__(self, code: str, message: str): self.code, self.message = code, message; super().__init__(message)


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 1024
    structured_output_mode: str = "json_schema"
    max_retries: int = 2
    rate_limit_per_second: float = 2.0
    cache_directory: str = ".netkeeper_cache/deepseek"
    save_raw_response: bool = False


def load_config(path: str | Path | None = None, environ: Mapping[str, str] | None = None) -> DeepSeekConfig:
    raw: dict[str, Any] = {}
    if path and Path(path).is_file(): raw.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    env = dict(environ or os.environ)
    mapping = {"DEEPSEEK_BASE_URL": "base_url", "DEEPSEEK_MODEL": "model", "DEEPSEEK_TIMEOUT": "timeout_seconds", "DEEPSEEK_TEMPERATURE": "temperature", "DEEPSEEK_MAX_TOKENS": "max_tokens", "DEEPSEEK_STRUCTURED_OUTPUT_MODE": "structured_output_mode"}
    for key, field in mapping.items():
        if key in env: raw[field] = env[key]
    for key in ("timeout_seconds", "temperature"): 
        if key in raw: raw[key] = float(raw[key])
    for key in ("max_tokens", "max_retries"):
        if key in raw: raw[key] = int(raw[key])
    return DeepSeekConfig(**{k: v for k, v in raw.items() if k in DeepSeekConfig.__dataclass_fields__})


class Transport(Protocol):
    def __call__(self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]: ...


def _http_transport(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response: return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise TranslationFailure(f"HTTP_{exc.code}", body[:240]) from exc
    except (urllib.error.URLError, TimeoutError) as exc: raise TranslationFailure("NETWORK_ERROR", str(exc)) from exc


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig | None = None, *, transport: Transport | None = None, environ: Mapping[str, str] | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.config, self.transport, self.environ, self.sleep = config or load_config(environ=environ), transport or _http_transport, dict(environ or os.environ), sleep
        self._lock, self._last = threading.Lock(), 0.0

    def complete(self, messages: list[dict[str, str]], *, online: bool, request_id: str, response_schema: Mapping[str, Any] | None = None) -> tuple[Mapping[str, Any], dict[str, Any]]:
        if not online: raise TranslationFailure("ONLINE_DISABLED", "online calls require --online")
        key = self.environ.get("DEEPSEEK_API_KEY", "")
        if not key: raise TranslationFailure("MISSING_API_KEY", "DEEPSEEK_API_KEY is required for online translation")
        payload: dict[str, Any] = {"model": self.config.model, "messages": messages, "temperature": self.config.temperature, "max_tokens": self.config.max_tokens}
        if response_schema and self.config.structured_output_mode != "off": payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "translation_result", "strict": True, "schema": response_schema}}
        for attempt in range(self.config.max_retries + 1):
            with self._lock:
                wait = max(0.0, 1 / self.config.rate_limit_per_second - (time.monotonic() - self._last))
                if wait: self.sleep(wait)
                self._last = time.monotonic()
            started = time.monotonic()
            try:
                response = self.transport(self.config.base_url.rstrip("/") + "/chat/completions", {"Authorization": "Bearer " + key, "Content-Type": "application/json"}, payload, self.config.timeout_seconds)
                return response, {"request_id": request_id, "retries": attempt, "elapsed_ms": round((time.monotonic() - started) * 1000, 3), "token_usage": response.get("usage", {}), "status": "ok"}
            except TranslationFailure as exc:
                retryable = exc.code in {"NETWORK_ERROR", "HTTP_429", "HTTP_500", "HTTP_502", "HTTP_503", "HTTP_504"}
                if not retryable or attempt == self.config.max_retries: raise
                self.sleep((2 ** attempt) * 0.1 + random.Random(f"{request_id}:{attempt}").random() * 0.05)
        raise AssertionError("unreachable")


def _hash(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
def _plain(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple): return [_plain(v) for v in value]
    if isinstance(value, list): return [_plain(v) for v in value]
    return value


class JsonCache:
    def __init__(self, directory: str | Path): self.directory = Path(directory)
    def get(self, key: str) -> Mapping[str, Any] | None:
        path = self.directory / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    def put(self, key: str, value: Mapping[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True); (self.directory / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


class PromptBuilder:
    def __init__(self, few_shot_path: str | Path | None = None) -> None:
        self.few_shot_path = Path(few_shot_path) if few_shot_path else None

    def context(self, snapshot: NetworkSnapshot, *, scenario_id: str = "") -> dict[str, Any]:
        return {"topology_id": snapshot.topology.topology_id, "scenario_id": scenario_id, "snapshot_id": snapshot.snapshot_id,
                "nodes": [n.node_id for n in snapshot.topology.nodes],
                "links": [{"link_id": l.link_id, "endpoints": [l.source, l.target], "state": snapshot.configuration.link_states.get(l.link_id, "up")} for l in snapshot.topology.links],
                "policy_ids": [p.policy_id for p in snapshot.policies],
                "bgp_targets": [{"router_id": r.router_id, "prefix": r.prefix, "next_hop": r.next_hop} for r in snapshot.configuration.bgp.routes],
                "od_pairs": [{"src": d.source, "dst": d.destination} for d in snapshot.traffic.demands if d.destination is not None]}
    def build(self, text: str, snapshot: NetworkSnapshot, *, mode: str, scenario_id: str = "", feedback: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
        system = "You translate network requests to JSON only. Use only whitelist APIs. Never output code. If entities are absent, parameters insufficient, calls conflict, or operation unsupported, return rejected with empty calls."
        api = [{"name": d.name, "category": d.category, "description": d.description, "arguments_schema": _plain(d.arguments_schema)} for d in API_REGISTRY.values()]
        body: dict[str, Any] = {"prompt_version": PROMPT_VERSION, "instruction": text, "network_context": self.context(snapshot, scenario_id=scenario_id), "apis": api, "api_request_schema": export_json_schema(), "translation_result_schema": RESULT_SCHEMA}
        messages = [{"role": "system", "content": system}]
        if mode in {"few_shot", "full"} and self.few_shot_path and self.few_shot_path.is_file():
            examples = [json.loads(line) for line in self.few_shot_path.read_text(encoding="utf-8").splitlines() if line.strip()][:6]
            messages.append({"role": "user", "content": "Fixed train-only examples: " + json.dumps(examples, ensure_ascii=False)})
        if feedback: body["validation_feedback"] = {k: feedback[k] for k in ("code", "call_index", "api", "message") if k in feedback}
        messages.append({"role": "user", "content": json.dumps(body, ensure_ascii=False)})
        return messages


def parse_response(response: Mapping[str, Any]) -> dict[str, Any]:
    try: content = response["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError) as exc: raise TranslationFailure("MALFORMED_PROVIDER_RESPONSE", "missing choices/message") from exc
    message = response["choices"][0]["message"]
    if not content and message.get("tool_calls"):
        calls = message["tool_calls"]
        if len(calls) != 1: raise TranslationFailure("JSON_FALLBACK_REJECTED", "exactly one tool call is required")
        content = calls[0].get("function", {}).get("arguments", "")
    if not isinstance(content, str) or not content: raise TranslationFailure("EMPTY_RESPONSE", "provider returned no JSON content")
    if content.strip() != content or content.startswith("```"): raise TranslationFailure("JSON_FALLBACK_REJECTED", "response must be exactly one JSON object")
    try: value = json.loads(content)
    except json.JSONDecodeError as exc: raise TranslationFailure("INVALID_JSON", "response is not one complete JSON object") from exc
    if not isinstance(value, dict): raise TranslationFailure("INVALID_JSON", "response must be an object")
    _validate_result(value); return value


def _validate_result(value: Mapping[str, Any]) -> None:
    allowed = {"status", "calls", "need_optimization", "error"}
    if set(value) - allowed or value.get("status") not in {"accepted", "rejected"} or not isinstance(value.get("calls"), list): raise TranslationFailure("TRANSLATION_SCHEMA_INVALID", "invalid TranslationResult envelope")
    if value["status"] == "accepted":
        if not value["calls"] or type(value.get("need_optimization")) is not bool or "error" in value: raise TranslationFailure("TRANSLATION_SCHEMA_INVALID", "invalid accepted result")
    else:
        if value["calls"] or not isinstance(value.get("error"), Mapping) or not isinstance(value["error"].get("code"), str): raise TranslationFailure("TRANSLATION_SCHEMA_INVALID", "invalid rejected result")


@dataclass(frozen=True)
class TranslationRunResult:
    status: str
    translation: Mapping[str, Any] | None
    error: Mapping[str, Any] | None
    attempts: int
    cache_hit: bool
    validation: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


class RecordingDispatcher:
    def __init__(self): self.requests: list[OptimizationRequest] = []
    def dispatch(self, snapshot: NetworkSnapshot, request: OptimizationRequest) -> JointAction:
        self.requests.append(request); return JointAction((), requested_by="llm", snapshot_id=snapshot.snapshot_id)


class Translator:
    def __init__(self, client: DeepSeekClient, builder: PromptBuilder, cache: JsonCache | None = None) -> None: self.client, self.builder, self.cache = client, builder, cache or JsonCache(client.config.cache_directory)
    def translate(self, text: str, snapshot: NetworkSnapshot, *, request_id: str, mode: str = "prompt_only", online: bool = False, scenario_id: str = "", env: UnifiedNetworkEnvironment | None = None, run_executor: bool = False, dispatcher: RecordingDispatcher | None = None) -> TranslationRunResult:
        if mode not in {"prompt_only", "few_shot", "full"}: raise ValueError("unknown mode")
        attempts, feedback = 0, None
        while attempts < (2 if mode == "full" else 1):
            messages = self.builder.build(text, snapshot, mode=mode, scenario_id=scenario_id, feedback=feedback)
            key = _hash({"provider": "deepseek", "model": self.client.config.model, "mode": mode, "prompt_version": PROMPT_VERSION, "input": text, "context": self.builder.context(snapshot, scenario_id=scenario_id), "schema": RESULT_SCHEMA, "feedback": feedback})
            cached = self.cache.get(key); attempts += 1
            try:
                if cached:
                    result = dict(cached["translation"])
                    _meta = cached.get("meta", {"status": "cache", "elapsed_ms": 0.0, "token_usage": {}})
                else:
                    response, _meta = self.client.complete(messages, online=online, request_id=request_id, response_schema=RESULT_SCHEMA if mode == "full" else None)
                    result = parse_response(response)
                    # Default cache deliberately stores validated structured
                    # output only, never provider raw text or credentials.
                    self.cache.put(key, {"translation": result, "meta": _meta})
                if result["status"] == "rejected": return TranslationRunResult("rejected", result, None, attempts, bool(cached), metadata=_meta)
                req = ApiRequest("v1", request_id, tuple(ApiCall.from_dict(v) for v in result["calls"]), snapshot.snapshot_id, result["need_optimization"], not run_executor)
                checked = validate_request(req, snapshot)
                if checked.valid:
                    execution = None
                    if run_executor:
                        if env is None: raise ValueError("env required when run_executor")
                        response_exec = execute(env, snapshot, req, dispatcher=dispatcher)
                        execution = response_exec.to_dict()
                    return TranslationRunResult("accepted", result, None, attempts, bool(cached), checked.to_dict(), execution, _meta)
                err = checked.errors[0]
                if mode != "full" or feedback is not None: return TranslationRunResult("failed", result, {"code": err.code, "call_index": err.call_index, "api": err.api, "message": err.message}, attempts, bool(cached), checked.to_dict(), metadata=_meta)
                feedback = {"code": err.code, "call_index": err.call_index, "api": err.api, "message": err.message}
            except TranslationFailure as exc:
                if mode != "full" or feedback is not None: return TranslationRunResult("failed", None, {"code": exc.code, "message": exc.message}, attempts, bool(cached) if 'cached' in locals() else False)
                feedback = {"code": exc.code, "message": exc.message}
        raise AssertionError("unreachable")


def rewrite_record(record: Mapping[str, Any], client: DeepSeekClient, *, online: bool, cache: JsonCache | None = None) -> dict[str, Any]:
    """Optional derived rewrite; never sees expected calls or replaces source text."""
    if not record.get("rewrite_selected"): return {"intent_id": record["intent_id"], "status": "not_selected"}
    source = str(record["original_text"]); source_hash = _hash(source); cache = cache or JsonCache(client.config.cache_directory)
    key = _hash({"stage": "rewrite", "model": client.config.model, "source": source_hash})
    cached = cache.get(key)
    if cached: return dict(cached)
    messages = [{"role": "system", "content": "Lightly rewrite this sentence. Preserve every entity and number exactly. Output one JSON object: {\"rewrite\": string}."}, {"role": "user", "content": source}]
    response, meta = client.complete(messages, online=online, request_id=str(record["intent_id"]), response_schema={"type": "object", "properties": {"rewrite": {"type": "string"}}, "required": ["rewrite"], "additionalProperties": False})
    value = parse_response({"choices": [{"message": {"content": response["choices"][0]["message"]["content"]}}]}) if False else json.loads(response["choices"][0]["message"]["content"])
    rewritten = value.get("rewrite") if isinstance(value, dict) else None
    entities = re.findall(r"R\d+|L:[^\s，。]+|\d+(?:\.\d+)?", source)
    faithful = isinstance(rewritten, str) and all(token in rewritten for token in entities)
    result = {"intent_id": record["intent_id"], "status": "rewritten" if faithful else "needs_review", "original_text": source, "rewrite_text": rewritten, "model": client.config.model, "config": {"temperature": client.config.temperature}, "source_hash": source_hash, "response_hash": _hash(response), "manual_review": not faithful, "meta": meta}
    cache.put(key, result); return result
