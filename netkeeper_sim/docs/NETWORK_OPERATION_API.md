# NetKeeper Network Operation API v1

`netkeeper_sim.api` is the safe boundary between structured intent and the immutable unified simulation schemas.  It has no natural-language parser, HTTP service, dynamic import, reflection dispatch, `eval`, or `exec`.  Calls are accepted only when their exact name is present in the static `API_REGISTRY` mapping.

## Execution model

1. Phase 1 parses the request and validates JSON shape, registry membership, types/ranges, topology entities/states, and whole-batch conflicts.  It produces an immutable `ExecutionPlan`; an error returns a stable `ApiResponse` and does not call the environment.
2. Phase 2 maps policy/traffic/failure calls to immutable `Event` objects and explicit configuration calls to `AtomicAction` objects in one `JointAction`.  It calls `UnifiedNetworkEnvironment.step()` once.  Routing, BGP selection, traffic, policy consistency, MLU, Traffic Shift, rewards, and configuration diff remain solely in that environment.

The executor verifies that the supplied snapshot is the environment current snapshot and that `expected_snapshot_id`, when supplied, matches it.  A mismatch is `STALE_SNAPSHOT` with zero effects.  API Event IDs are deterministic: `API:{request_id}:{call_index:04d}`.

If the environment reports a planned operation error, its mutable `current_snapshot` pointer is restored to the original immutable snapshot and the response is `ENVIRONMENT_REJECTED`.  Inputs, historical snapshots, dataset files, scenario events, and topology objects are never mutated.

`get_network_state` alone is a true read: it does not advance the step or consume scenario events.  In a valid mixed batch it must appear before writes and its response state is the post-commit bounded summary.  `dry_run` performs Phase 1 and mapping only; it never dispatches an optimizer or calls `step()`.

## Request and response

```json
{
  "api_version": "v1",
  "request_id": "intent-000001",
  "expected_snapshot_id": "optional-snapshot-id",
  "dry_run": false,
  "calls": [
    {"api": "add_reachable_policy", "arguments": {"src": "R1", "dst": "R8"}},
    {"api": "set_ospf_weight", "arguments": {"link_id": "L:R1--R2:0", "weight": 20}}
  ]
}
```

```json
{
  "success": true,
  "request_id": "intent-000001",
  "applied_calls": [0, 1],
  "before_snapshot_id": "...",
  "after_snapshot_id": "...",
  "configuration_diff": {"ospf_weights.L:R1--R2:0": [1, 20]},
  "event_diff": {"policy_count_before": 6, "policy_count_after": 7},
  "metrics": {"maximum_link_utilization": 0.42},
  "optimization_status": "not_requested",
  "errors": []
}
```

The Draft 2020-12 request schema is returned by `netkeeper_sim.api.export_json_schema()`; the stable response envelope schema is `export_response_json_schema()`.  The request has `additionalProperties: false` at request, call, and argument levels.  The runtime validator applies the same contract plus graph/state validation; this is the schema loader used by the CLI.

## API whitelist

| Category | APIs | Range / mapping |
|---|---|---|
| Policy | `add_reachable_policy`, `add_forward_policy`, `add_avoid_policy`, `add_isolation_policy`, `remove_policy` | Events. Forward uses `pass_node`/`avoid_node`, optional `all_path` or `any_path`; isolation has four distinct OD endpoints. |
| Traffic | `set_traffic_demand`, `scale_traffic_demand`, `set_traffic_hotspot` | Events with an in-memory immutable `TrafficMatrix`; demand is finite `>=0`, scale finite `>0`. |
| Topology | `set_link_state`, `set_node_state` | Existing `link_id`/`node_id`, state `up` or `down`; Event failure/recovery. |
| OSPF | `set_ospf_weight` | Atomic Action; integer `[1, 64]`. |
| BGP | `set_bgp_local_pref`, `set_bgp_as_path_length`, `set_bgp_med` | Existing `(router_id,prefix,next_hop)` target; integer `[1,64]`. |
| Performance | `set_link_bandwidth`, `set_link_capacity`, `set_queue_length` | Atomic Action. Bandwidth/capacity must respect physical/current bounds; queue integer `[0,1000000]`. |
| Control | `get_network_state`, `optimize_network` | Bounded read; or dispatcher request with objectives from `policy_consistency`, `mlu`, `traffic_shift`, `config_change`. |

All nodes, links, OD demands, policy IDs, and BGP targets are validated against the supplied snapshot.  Nodes/links that are down cannot be configured.  `remove_policy` requires its stable policy ID; automatically created IDs are `P:api:{request_id}:{call_index}`.

## Ordering and conflicts

Valid source order is read calls, high-level policy/traffic/topology Events, explicit configuration Actions, then one final optimizer request.  The executor asserts the validator plan retains that source order; it does not silently reorder calls.

Rejected conflicts include duplicate parameter targets, repeated set/scale on one OD, add/remove misuse, Forward-pass and Forward-avoid on the same OD/node, a policy against an isolation OD pair, overlapping isolation endpoints, a link-up under a down endpoint, and configuring a down object.  `need_optimization=true` is an implicit final optimizer request; it is rejected together with explicit `optimize_network`.

## Errors

Errors always have `error_type`, `code`, `message`, `call_index`, `api`, and `details`.  Main codes are: `UNSUPPORTED_VERSION`, `UNSUPPORTED_API`, `INVALID_REQUEST`, `INVALID_TYPE`, `MISSING_ARGUMENT`, `ADDITIONAL_PROPERTY`, `OUT_OF_RANGE`, `UNKNOWN_NODE`, `UNKNOWN_LINK`, `OD_NOT_FOUND`, `UNKNOWN_BGP_ROUTE`, `POLICY_NOT_FOUND`, `INVALID_ENDPOINTS`, `INVALID_WAYPOINT`, `OVERLAPPING_ISOLATION`, `OBJECT_DOWN`, `POLICY_CONFLICT`, `TRAFFIC_OPERATION_CONFLICT`, `DUPLICATE_OPERATION`, `ORDER_VIOLATION`, `STALE_SNAPSHOT`, `MAPPING_FAILED`, `ENVIRONMENT_REJECTED`, and `OPTIMIZER_UNAVAILABLE`.

## CLI

Run from `netkeeper_sim/`:

```bash
python3 -m netkeeper_sim.api.cli list
python3 -m netkeeper_sim.api.cli export-schema --output /tmp/netkeeper-api-schema.json
python3 -m netkeeper_sim.api.cli export-schema --kind response --output /tmp/netkeeper-api-response-schema.json
python3 -m netkeeper_sim.api.cli execute \
  --dataset-root ../data/netkeeper_lite --scenario-file scenarios/test.jsonl --index 0 \
  --seed 7 --request request.json --response response.json
python3 -m netkeeper_sim.api.cli execute \
  --dataset-root ../data/netkeeper_lite --scenario-file scenarios/test.jsonl --scenario-id S:test:00000 \
  --request request.json --response response.json --dry-run
```

The CLI always calls `execute`; it cannot bypass the registry or validator.  It returns exit status 2 for a validly parsed request that the API rejects, while still writing the machine-readable response JSON.

## Later blocks

Fourth-block tooling can enumerate `API_REGISTRY` for name/category/description and serialize `export_json_schema()` directly into LLM tool definitions.  It must submit parsed JSON to this API, never source code.

Fifth-block code may inject an object implementing `dispatch(snapshot, OptimizationRequest) -> JointAction` into `execute`.  No legacy trainer is imported here.  Without that dispatcher the result is `OPTIMIZER_UNAVAILABLE`; with `dry_run`, the optimization request remains `pending` without dispatch.
