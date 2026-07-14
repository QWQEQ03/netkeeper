"""Stable identifiers and schema-wide constants."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Iterable


SCHEMA_VERSION = "netkeeper-sim.schema.v1"


def canonical_text(value: object) -> str:
    """Return the canonical string used in deterministic manifests."""
    return unicodedata.normalize("NFC", str(value)).strip()


def natural_key(value: object) -> tuple[int, int | str]:
    text = canonical_text(value)
    if re.fullmatch(r"[+-]?\d+", text):
        return (0, int(text))
    return (1, text)


def stable_hash(value: Any, length: int = 8) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def slug(value: object) -> str:
    text = canonical_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unnamed"


def topology_id(normalized_name: str, manifest: Any) -> str:
    return f"zoo:{slug(normalized_name)}:{stable_hash(manifest)}"


def link_id(source: str, target: str, ordinal: int) -> str:
    left, right = sorted((source, target), key=natural_key)
    return f"L:{left}--{right}:{ordinal}"


def directed_arc_id(link: str, source: str, target: str) -> str:
    return f"A:{link}:{source}->{target}"


def configuration_version_id(version: int) -> str:
    if version < 0:
        raise ValueError("configuration version must be non-negative")
    return f"cv{version}"


def snapshot_id(
    topology: str,
    topology_state_version: int,
    configuration_version: int,
    step: int,
) -> str:
    if min(topology_state_version, configuration_version, step) < 0:
        raise ValueError("snapshot versions and step must be non-negative")
    return f"{topology}:tv{topology_state_version}:cv{configuration_version}:t{step}"


def canonical_tuple(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted((canonical_text(value) for value in values), key=natural_key))
