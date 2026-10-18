"""Validated state-dependent TD3 exploration rules and runtime features.

The rule language is intentionally small and data-only.  JSON is parsed with
duplicate-key rejection, statically type checked once, and compiled into an
immutable tree.  Decision-time evaluation never uses ``eval`` or ``exec``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from experiment_config import COMMUNICATION_RANGE_M
from llm_explorer import ExplorerValidationError, NUM_UAV, _json_clone
from replay_auxiliary import TASK_TYPE_ENCODING


STATE_RULE_SCHEMA_VERSION = "td3-state-bias-rule-v1"
STATE_RULE_LANGUAGE_VERSION = "td3-state-bias-rule-language-v1"
STATE_RULE_FEATURE_CATALOG_VERSION = "td3-state-bias-runtime-features-v1"
STATE_RULE_MAX_COMPONENT_NODES = 40
STATE_RULE_MAX_TOTAL_NODES = 100
STATE_RULE_MAX_DEPTH = 8

NUMBER = "number"
BOOLEAN = "boolean"


def _field(value_type: str, meaning: str, unit: str, validity: str, identity: str,
           aggregation: str = "one UAV at one movement-decision boundary") -> dict[str, str]:
    return {
        "type": value_type,
        "physical_meaning": meaning,
        "unit_and_normalization": unit,
        "validity": validity,
        "target_or_link_identity": identity,
        "aggregation_and_scope": aggregation,
    }


# Numeric values paired with a validity flag are set to zero when unavailable.
# Zero therefore remains distinguishable from unknown without admitting nulls
# into arithmetic expressions.
RUNTIME_FEATURE_CATALOG: dict[str, dict[str, str]] = {
    "uav_id": _field(NUMBER, "UAV identifier", "integer-valued ID 0..15", "always", "evaluated UAV"),
    "time_s": _field(NUMBER, "episode time at this policy observation", "seconds, not normalized", "always", "episode clock"),
    "remaining_time_fraction": _field(NUMBER, "fraction of episode time remaining", "dimensionless [0,1]", "always", "episode clock"),
    "actor_speed": _field(NUMBER, "original Actor horizontal-speed scalar", "normalized action [-1,1]; physical speed is 5*(value+1) m/s", "always", "evaluated UAV"),
    "actor_heading": _field(NUMBER, "original Actor absolute-heading scalar", "periodic normalized action [-1,1); radians are pi*value", "always", "evaluated UAV"),
    "actor_vertical": _field(NUMBER, "original Actor vertical-speed scalar", "normalized action [-1,1]; physical vertical speed is 2*value m/s", "always", "evaluated UAV"),
    "movement_mask": _field(BOOLEAN, "whether TD3 owns movement now", "Boolean", "always", "evaluated UAV"),
    "has_fov_task": _field(BOOLEAN, "current assigned visual/FOV service role", "Boolean", "always", "evaluated UAV and assigned discovered RoI"),
    "has_com_task": _field(BOOLEAN, "current assigned COM service role", "Boolean", "always", "evaluated UAV and observable SR"),
    "has_search_task": _field(BOOLEAN, "current Search role", "Boolean", "always", "evaluated UAV"),
    "is_hovering": _field(BOOLEAN, "current Hovering-only role", "Boolean", "always", "evaluated UAV"),
    "uav_x_m": _field(NUMBER, "current UAV x coordinate", "meters", "always", "evaluated UAV"),
    "uav_y_m": _field(NUMBER, "current UAV y coordinate", "meters", "always", "evaluated UAV"),
    "uav_z_m": _field(NUMBER, "current UAV altitude", "meters", "always", "evaluated UAV"),
    "fov_target_valid": _field(BOOLEAN, "assigned discovered RoI geometry is available", "Boolean", "always", "assigned FOV RoI"),
    "fov_relative_x_m": _field(NUMBER, "RoI x minus UAV x", "meters", "when fov_target_valid", "assigned FOV RoI"),
    "fov_relative_y_m": _field(NUMBER, "RoI y minus UAV y", "meters", "when fov_target_valid", "assigned FOV RoI"),
    "fov_relative_z_m": _field(NUMBER, "RoI z minus UAV z", "meters", "when fov_target_valid", "assigned FOV RoI"),
    "fov_horizontal_distance_m": _field(NUMBER, "horizontal UAV-to-RoI distance", "meters", "when fov_target_valid", "assigned FOV RoI"),
    "fov_target_bearing_rad": _field(NUMBER, "absolute bearing from UAV to RoI", "radians [-pi,pi]", "when fov_target_valid", "assigned FOV RoI"),
    "fov_c9_satisfied": _field(BOOLEAN, "canonical C9 range condition", "Boolean", "when fov_target_valid", "assigned FOV RoI"),
    "fov_c9_margin_m": _field(NUMBER, "b1*h-d; nonnegative satisfies C9", "meters", "when fov_target_valid", "assigned FOV RoI"),
    "fov_c10_geometry_valid": _field(BOOLEAN, "canonical finite C10 edge geometry exists", "Boolean", "always", "assigned FOV RoI"),
    "fov_c10_satisfied": _field(BOOLEAN, "canonical C10 margin is nonnegative", "Boolean", "when fov_c10_geometry_valid", "assigned FOV RoI"),
    "fov_c10_margin_m": _field(NUMBER, "min(d_L,d_R)-RoI radius", "meters", "when fov_c10_geometry_valid", "assigned FOV RoI"),
    "fov_capture_valid": _field(BOOLEAN, "actual projected camera footprint is valid now", "Boolean", "when fov_target_valid", "assigned FOV RoI"),
    "fov_coverage_ratio": _field(NUMBER, "actual circular-RoI footprint intersection ratio", "dimensionless ratio [0,1]", "when fov_target_valid", "assigned FOV RoI"),
    "fov_image_quantity": _field(NUMBER, "RoI area divided by footprint area", "dimensionless, raw unclipped I", "when fov_target_valid", "assigned FOV RoI"),
    "com_target_valid": _field(BOOLEAN, "assigned observable SR geometry is available", "Boolean", "always", "assigned COM SR"),
    "com_relative_x_m": _field(NUMBER, "SR x minus UAV x", "meters", "when com_target_valid", "assigned COM SR"),
    "com_relative_y_m": _field(NUMBER, "SR y minus UAV y", "meters", "when com_target_valid", "assigned COM SR"),
    "com_relative_z_m": _field(NUMBER, "SR z minus UAV z", "meters", "when com_target_valid", "assigned COM SR"),
    "com_distance_3d_m": _field(NUMBER, "current UAV-to-SR distance", "meters", "when com_target_valid", "assigned COM SR"),
    "com_target_bearing_rad": _field(NUMBER, "absolute bearing from UAV to SR", "radians [-pi,pi]", "when com_target_valid", "assigned COM SR"),
    "com_in_range": _field(BOOLEAN, "inclusive 400 m S2U eligibility test", "Boolean", "when com_target_valid", "assigned COM SR-to-UAV link"),
    "com_reference_capacity_valid": _field(BOOLEAN, "current-geometry S2U expected-capacity value is available", "Boolean", "always", "assigned COM SR-to-UAV link"),
    "com_reference_capacity_mbps": _field(NUMBER, "current-geometry conditional expected S2U capacity", "Mbps at the configured reference COM bandwidth; not a fixed maximum and not realized shared-FDMA throughput", "when com_reference_capacity_valid", "assigned COM SR-to-UAV link"),
    "com_sr_queue_valid": _field(BOOLEAN, "assigned SR FIFO summary is observable", "Boolean", "always", "assigned COM SR"),
    "com_sr_queue_empty": _field(BOOLEAN, "assigned observable SR FIFO has no active packet", "Boolean", "when com_sr_queue_valid", "assigned COM SR"),
    "com_sr_backlog_bits": _field(NUMBER, "remaining bits in assigned SR FIFO", "bits, not normalized", "when com_sr_queue_valid", "assigned COM SR FIFO", "sum over active packets in that FIFO at this boundary"),
    "com_sr_packet_count": _field(NUMBER, "active packet count in assigned SR FIFO", "integer-valued packets", "when com_sr_queue_valid", "assigned COM SR FIFO", "count over active packets in that FIFO"),
    "com_sr_hol_deadline_valid": _field(BOOLEAN, "assigned SR head-of-line deadline exists", "Boolean", "always", "assigned COM SR FIFO"),
    "com_sr_hol_remaining_deadline_s": _field(NUMBER, "assigned SR HOL absolute deadline minus current time", "seconds; may be nonpositive", "when com_sr_hol_deadline_valid", "assigned COM SR FIFO"),
    "uav_queue_valid": _field(BOOLEAN, "local aggregate UAV FIFO summary is available", "Boolean", "always", "evaluated UAV FIFO"),
    "uav_queue_empty": _field(BOOLEAN, "local aggregate UAV FIFO is empty", "Boolean", "when uav_queue_valid", "evaluated UAV FIFO"),
    "uav_backlog_bits": _field(NUMBER, "local aggregate remaining packet bits", "bits, not normalized", "when uav_queue_valid", "evaluated UAV FIFO", "sum over active FOV and COM packets"),
    "uav_fov_backlog_bits": _field(NUMBER, "local FOV remaining packet bits", "bits, not normalized", "when uav_queue_valid", "evaluated UAV FIFO", "sum over active FOV packets"),
    "uav_com_backlog_bits": _field(NUMBER, "local COM remaining packet bits", "bits, not normalized", "when uav_queue_valid", "evaluated UAV FIFO", "sum over active COM packets"),
    "uav_hol_valid": _field(BOOLEAN, "local aggregate FIFO has a HOL packet", "Boolean", "always", "evaluated UAV FIFO"),
    "uav_hol_remaining_deadline_s": _field(NUMBER, "local HOL absolute deadline minus current time", "seconds; may be nonpositive", "when uav_hol_valid", "evaluated UAV FIFO"),
    "u2g_link_valid": _field(BOOLEAN, "current UAV-to-GS expected-link fields are available", "Boolean", "always", "evaluated UAV to GS"),
    "u2g_distance_3d_m": _field(NUMBER, "current UAV-to-GS distance", "meters", "when u2g_link_valid", "evaluated UAV to GS"),
    "u2g_in_range": _field(BOOLEAN, "inclusive 400 m U2G eligibility test", "Boolean", "when u2g_link_valid", "evaluated UAV to GS"),
    "u2g_reference_capacity_mbps": _field(NUMBER, "current-geometry conditional expected U2G capacity", "Mbps at full 10 MHz reference bandwidth with installed LoS state and expected fading; not scheduled throughput or a fixed maximum", "when u2g_link_valid", "evaluated UAV to GS"),
    "best_u2u_neighbor_valid": _field(BOOLEAN, "an in-range directed U2U neighbor with valid expected capacity exists", "Boolean", "always", "evaluated UAV to selected neighbor"),
    "best_u2u_neighbor_id": _field(NUMBER, "identity of maximum-reference-capacity in-range neighbor; lowest ID breaks ties", "integer-valued UAV ID", "when best_u2u_neighbor_valid", "selected U2U neighbor"),
    "best_u2u_relative_x_m": _field(NUMBER, "selected neighbor x minus evaluated UAV x", "meters", "when best_u2u_neighbor_valid", "selected U2U neighbor"),
    "best_u2u_relative_y_m": _field(NUMBER, "selected neighbor y minus evaluated UAV y", "meters", "when best_u2u_neighbor_valid", "selected U2U neighbor"),
    "best_u2u_relative_z_m": _field(NUMBER, "selected neighbor z minus evaluated UAV z", "meters", "when best_u2u_neighbor_valid", "selected U2U neighbor"),
    "best_u2u_distance_3d_m": _field(NUMBER, "distance to selected neighbor", "meters", "when best_u2u_neighbor_valid", "selected U2U neighbor"),
    "best_u2u_reference_capacity_mbps": _field(NUMBER, "maximum current-geometry directed U2U expected capacity among in-range neighbors", "Mbps at full 10 MHz reference bandwidth with expected Rician fading; not scheduled shared-FDMA throughput or a fixed maximum", "when best_u2u_neighbor_valid", "evaluated UAV to selected neighbor", "maximum over in-range valid directed neighbors; identity is retained"),
}

FIELD_TYPES = {name: spec["type"] for name, spec in RUNTIME_FEATURE_CATALOG.items()}


def runtime_feature_catalog() -> dict[str, Any]:
    return {
        "schema_version": STATE_RULE_FEATURE_CATALOG_VERSION,
        "unavailable_numeric_encoding": "zero accompanied by a false validity flag; rules must branch on the flag",
        "relative_position_convention": "target position minus evaluated UAV position",
        "fields": [
            {"name": name, **_json_clone(spec)}
            for name, spec in RUNTIME_FEATURE_CATALOG.items()
        ],
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExplorerValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _Node:
    kind: str
    value_type: str
    value: Any
    args: tuple["_Node", ...] = ()


@dataclass(frozen=True)
class _NodeInfo:
    node: _Node
    count: int
    depth: int


_ARITY = {
    "add": 2, "sub": 2, "mul": 2, "safe_div": 2, "neg": 1,
    "abs": 1, "tanh": 1, "min": 2, "max": 2, "clip": 3,
    "sin": 1, "cos": 1, "wrap_angle": 1, "atan2": 2,
    "lt": 2, "le": 2, "gt": 2, "ge": 2, "eq": 2,
    "and": 2, "or": 2, "not": 1, "if": 3,
}
_NUMERIC_OPS = frozenset({
    "add", "sub", "mul", "safe_div", "neg", "abs", "tanh", "min", "max",
    "clip", "sin", "cos", "wrap_angle", "atan2",
})
_ORDERING_OPS = frozenset({"lt", "le", "gt", "ge"})
_LOGIC_OPS = frozenset({"and", "or", "not"})


def _compile_expression(value: Any, *, path: str) -> _NodeInfo:
    if isinstance(value, bool):
        raise ExplorerValidationError(f"{path}: Boolean constants are not expression nodes")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ExplorerValidationError(f"{path}: numeric constant must be finite")
        return _NodeInfo(_Node("constant", NUMBER, number), 1, 1)
    if not isinstance(value, dict):
        raise ExplorerValidationError(f"{path}: node must be a finite number or object")

    keys = set(value)
    if keys == {"field"}:
        field = value["field"]
        if not isinstance(field, str) or field not in FIELD_TYPES:
            raise ExplorerValidationError(f"{path}: unknown formal field {field!r}")
        return _NodeInfo(_Node("field", FIELD_TYPES[field], field), 1, 1)
    if keys != {"op", "args"}:
        raise ExplorerValidationError(
            f"{path}: object keys must be exactly ['field'] or ['args','op']"
        )
    op = value["op"]
    args_value = value["args"]
    if not isinstance(op, str) or op not in _ARITY:
        raise ExplorerValidationError(f"{path}: unsupported operation {op!r}")
    if not isinstance(args_value, list) or len(args_value) != _ARITY[op]:
        raise ExplorerValidationError(
            f"{path}: {op} requires exactly {_ARITY[op]} arguments"
        )
    children = tuple(
        _compile_expression(arg, path=f"{path}.{op}.args[{index}]")
        for index, arg in enumerate(args_value)
    )
    types = tuple(child.node.value_type for child in children)
    if op in _NUMERIC_OPS and any(kind != NUMBER for kind in types):
        raise ExplorerValidationError(f"{path}: {op} accepts numeric arguments only")
    if op in _ORDERING_OPS and types != (NUMBER, NUMBER):
        raise ExplorerValidationError(f"{path}: {op} accepts numeric arguments only")
    if op == "eq" and (len(set(types)) != 1 or types[0] not in {NUMBER, BOOLEAN}):
        raise ExplorerValidationError(f"{path}: eq arguments must have the same type")
    if op in _LOGIC_OPS and any(kind != BOOLEAN for kind in types):
        raise ExplorerValidationError(f"{path}: {op} accepts Boolean arguments only")
    if op == "if":
        if types[0] != BOOLEAN or types[1] != types[2]:
            raise ExplorerValidationError(
                f"{path}: if requires Boolean condition and same-typed branches"
            )
        result_type = types[1]
    elif op in _ORDERING_OPS or op == "eq" or op in _LOGIC_OPS:
        result_type = BOOLEAN
    else:
        result_type = NUMBER
    count = 1 + sum(child.count for child in children)
    depth = 1 + max(child.depth for child in children)
    if depth > STATE_RULE_MAX_DEPTH:
        raise ExplorerValidationError(
            f"{path}: expression depth {depth} exceeds {STATE_RULE_MAX_DEPTH}; root depth is 1"
        )
    if op == "clip":
        lower, upper = children[1].node, children[2].node
        if lower.kind == upper.kind == "constant" and lower.value > upper.value:
            raise ExplorerValidationError(f"{path}: clip lower must be <= upper")
    return _NodeInfo(
        _Node("op", result_type, op, tuple(child.node for child in children)),
        count,
        depth,
    )


@dataclass(frozen=True)
class CompiledStateRule:
    rule: dict[str, Any]
    components: tuple[_Node, _Node, _Node]
    component_node_counts: tuple[int, int, int]
    component_depths: tuple[int, int, int]

    def evaluate(self, features: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
        trace = {"conditions_evaluated": 0, "conditions_true": 0, "short_circuits": 0}
        result = np.asarray(
            [_evaluate_node(node, features, trace) for node in self.components],
            dtype=np.float64,
        )
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ExplorerValidationError("rule final action components must be finite numbers")
        return result, trace


def parse_and_compile_state_rule(content: str) -> CompiledStateRule:
    try:
        value = json.loads(str(content), object_pairs_hook=_reject_duplicate_pairs)
    except (TypeError, json.JSONDecodeError, ExplorerValidationError) as exc:
        raise ExplorerValidationError(
            f"invalid state-rule JSON: {type(exc).__name__}: {exc}"
        ) from exc
    return compile_state_rule(value)


def compile_state_rule(value: Any) -> CompiledStateRule:
    if not isinstance(value, dict):
        raise ExplorerValidationError("state-rule root must be an object")
    required = {"schema_version", "speed", "heading", "vertical"}
    if set(value) != required:
        raise ExplorerValidationError(
            "state-rule root keys must be exactly schema_version, speed, heading, vertical"
        )
    if value["schema_version"] != STATE_RULE_SCHEMA_VERSION:
        raise ExplorerValidationError(
            f"state-rule schema_version must be {STATE_RULE_SCHEMA_VERSION!r}"
        )
    infos = tuple(
        _compile_expression(value[name], path=name)
        for name in ("speed", "heading", "vertical")
    )
    for name, info in zip(("speed", "heading", "vertical"), infos):
        if info.node.value_type != NUMBER:
            raise ExplorerValidationError(f"{name}: final component must be numeric")
        if info.count > STATE_RULE_MAX_COMPONENT_NODES:
            raise ExplorerValidationError(
                f"{name}: node count {info.count} exceeds {STATE_RULE_MAX_COMPONENT_NODES}"
            )
    total = sum(info.count for info in infos)
    if total > STATE_RULE_MAX_TOTAL_NODES:
        raise ExplorerValidationError(
            f"total node count {total} exceeds {STATE_RULE_MAX_TOTAL_NODES}"
        )
    return CompiledStateRule(
        rule=_json_clone(value),
        components=tuple(info.node for info in infos),
        component_node_counts=tuple(info.count for info in infos),
        component_depths=tuple(info.depth for info in infos),
    )


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ExplorerValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExplorerValidationError(f"{context} must be finite")
    return result


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ExplorerValidationError(f"{context} must be Boolean")
    return bool(value)


def _evaluate_node(node: _Node, features: Mapping[str, Any], trace: dict[str, int]) -> Any:
    if node.kind == "constant":
        return node.value
    if node.kind == "field":
        if node.value not in features:
            raise ExplorerValidationError(f"runtime field {node.value!r} is missing")
        if node.value_type == BOOLEAN:
            return _boolean(features[node.value], context=f"field {node.value}")
        return _number(features[node.value], context=f"field {node.value}")

    op = node.value
    if op == "if":
        condition = _boolean(_evaluate_node(node.args[0], features, trace), context="if condition")
        trace["conditions_evaluated"] += 1
        trace["conditions_true"] += int(condition)
        return _evaluate_node(node.args[1] if condition else node.args[2], features, trace)
    if op == "and":
        left = _boolean(_evaluate_node(node.args[0], features, trace), context="and left")
        trace["conditions_evaluated"] += 1
        trace["conditions_true"] += int(left)
        if not left:
            trace["short_circuits"] += 1
            return False
        return _boolean(_evaluate_node(node.args[1], features, trace), context="and right")
    if op == "or":
        left = _boolean(_evaluate_node(node.args[0], features, trace), context="or left")
        trace["conditions_evaluated"] += 1
        trace["conditions_true"] += int(left)
        if left:
            trace["short_circuits"] += 1
            return True
        return _boolean(_evaluate_node(node.args[1], features, trace), context="or right")
    if op == "not":
        return not _boolean(_evaluate_node(node.args[0], features, trace), context="not argument")

    values = tuple(_evaluate_node(arg, features, trace) for arg in node.args)
    if op == "eq":
        return values[0] == values[1]
    if op in _ORDERING_OPS:
        left, right = map(lambda item: _number(item, context=f"{op} argument"), values)
        return {"lt": left < right, "le": left <= right, "gt": left > right, "ge": left >= right}[op]
    numbers = tuple(_number(item, context=f"{op} argument") for item in values)
    if op == "add": result = numbers[0] + numbers[1]
    elif op == "sub": result = numbers[0] - numbers[1]
    elif op == "mul": result = numbers[0] * numbers[1]
    elif op == "safe_div": result = 0.0 if abs(numbers[1]) <= 1e-8 else numbers[0] / numbers[1]
    elif op == "neg": result = -numbers[0]
    elif op == "abs": result = abs(numbers[0])
    elif op == "tanh": result = math.tanh(numbers[0])
    elif op == "min": result = min(numbers)
    elif op == "max": result = max(numbers)
    elif op == "clip":
        if numbers[1] > numbers[2]:
            raise ExplorerValidationError("runtime clip lower must be <= upper")
        result = min(max(numbers[0], numbers[1]), numbers[2])
    elif op == "sin": result = math.sin(numbers[0])
    elif op == "cos": result = math.cos(numbers[0])
    elif op == "atan2": result = 0.0 if numbers == (0.0, 0.0) else math.atan2(numbers[0], numbers[1])
    elif op == "wrap_angle": result = (numbers[0] + math.pi) % (2.0 * math.pi) - math.pi
    else:  # pragma: no cover - compile-time whitelist makes this unreachable
        raise ExplorerValidationError(f"unsupported compiled operation {op!r}")
    return _number(result, context=f"{op} result")


def state_rule_output_schema() -> dict[str, Any]:
    return {
        "schema_version": STATE_RULE_SCHEMA_VERSION,
        "root_keys_exactly": ["schema_version", "speed", "heading", "vertical"],
        "node_forms": ["finite JSON number", '{"field":"formal_name"}', '{"op":"name","args":[...]}'],
        "limits": {
            "component_nodes": STATE_RULE_MAX_COMPONENT_NODES,
            "total_nodes": STATE_RULE_MAX_TOTAL_NODES,
            "maximum_depth": STATE_RULE_MAX_DEPTH,
            "depth_definition": "the root expression has depth 1",
        },
    }


def _scalar(snapshot: Mapping[str, Any], name: str, index: Any, default: float = 0.0) -> float:
    try:
        value = float(np.asarray(snapshot[name])[index])
    except (KeyError, IndexError, TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _flag(snapshot: Mapping[str, Any], name: str, index: Any, default: bool = False) -> bool:
    try:
        return bool(np.asarray(snapshot[name])[index])
    except (KeyError, IndexError, TypeError, ValueError):
        return bool(default)


def build_runtime_feature_rows(
    env: Any,
    snapshot: Mapping[str, Any],
    movement_mask: Any,
    actor_action: Any,
    *,
    episode_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Build read-only decision-aligned rule inputs from an existing snapshot."""

    mask = np.asarray(movement_mask, dtype=bool).reshape(NUM_UAV)
    actors = np.asarray(actor_action, dtype=np.float64).reshape(NUM_UAV, 3)
    if not np.isfinite(actors).all():
        raise ExplorerValidationError("original Actor action must be finite")
    time_s = _scalar(snapshot, "snapshot_time_s", 0)
    duration = float(episode_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ExplorerValidationError("episode_seconds must be finite and positive")
    remaining = min(1.0, max(0.0, (duration - time_s) / duration))
    positions = np.asarray(
        [env.uav_dict[uav_id].get_position() for uav_id in range(NUM_UAV)],
        dtype=np.float64,
    )
    if positions.shape != (NUM_UAV, 3) or not np.isfinite(positions).all():
        raise ExplorerValidationError("current UAV positions are invalid")

    rows: list[dict[str, Any]] = []
    for uav_id in range(NUM_UAV):
        task_codes = [
            int(code)
            for valid, code in zip(
                np.asarray(snapshot.get("task_pair_valid", np.zeros((NUM_UAV, 2), bool)))[uav_id],
                np.asarray(snapshot.get("task_type", np.zeros((NUM_UAV, 2), int)))[uav_id],
            )
            if bool(valid)
        ]
        has_fov = TASK_TYPE_ENCODING["FOV"] in task_codes
        has_com = TASK_TYPE_ENCODING["COM"] in task_codes
        has_search = TASK_TYPE_ENCODING["Search"] in task_codes
        is_hovering = bool(task_codes) and all(
            code == TASK_TYPE_ENCODING["Hovering"] for code in task_codes
        )
        row: dict[str, Any] = {
            name: (False if spec["type"] == BOOLEAN else 0.0)
            for name, spec in RUNTIME_FEATURE_CATALOG.items()
        }
        row.update({
            "uav_id": float(uav_id), "time_s": time_s,
            "remaining_time_fraction": remaining,
            "actor_speed": float(actors[uav_id, 0]),
            "actor_heading": float(actors[uav_id, 1]),
            "actor_vertical": float(actors[uav_id, 2]),
            "movement_mask": bool(mask[uav_id]),
            "has_fov_task": has_fov, "has_com_task": has_com,
            "has_search_task": has_search, "is_hovering": is_hovering,
            "uav_x_m": float(positions[uav_id, 0]),
            "uav_y_m": float(positions[uav_id, 1]),
            "uav_z_m": float(positions[uav_id, 2]),
        })

        fov_valid = has_fov and _flag(snapshot, "vs_pair_valid", uav_id)
        row["fov_target_valid"] = fov_valid
        if fov_valid:
            roi_id = int(_scalar(snapshot, "vs_roi_id", uav_id, -1.0))
            roi_rows = np.flatnonzero(
                np.asarray(snapshot.get("roi_observable", ()), dtype=bool)
                & (np.asarray(snapshot.get("roi_id", ()), dtype=int) == roi_id)
            )
            if roi_rows.size:
                target = np.asarray(snapshot["roi_position_m"], dtype=float)[int(roi_rows[0])]
                relative = target - positions[uav_id]
                row.update({
                    "fov_relative_x_m": float(relative[0]),
                    "fov_relative_y_m": float(relative[1]),
                    "fov_relative_z_m": float(relative[2]),
                    "fov_horizontal_distance_m": _scalar(snapshot, "vs_horizontal_distance_m", uav_id),
                    "fov_target_bearing_rad": math.atan2(float(relative[1]), float(relative[0])) if np.any(relative[:2]) else 0.0,
                    "fov_c9_margin_m": _scalar(snapshot, "vs_c9_margin_m", uav_id),
                    "fov_c10_geometry_valid": _flag(snapshot, "vs_c10_geometry_valid", uav_id),
                    "fov_c10_margin_m": _scalar(snapshot, "vs_c10_margin_m", uav_id),
                    "fov_capture_valid": _flag(snapshot, "vs_capture_valid", uav_id),
                    "fov_coverage_ratio": _scalar(snapshot, "vs_coverage_ratio", uav_id),
                    "fov_image_quantity": _scalar(snapshot, "vs_image_quantity", uav_id),
                })
                row["fov_c9_satisfied"] = row["fov_c9_margin_m"] >= 0.0
                row["fov_c10_satisfied"] = bool(row["fov_c10_geometry_valid"] and row["fov_c10_margin_m"] >= 0.0)
            else:
                row["fov_target_valid"] = False

        com_sr_id = None
        valid_tasks = np.asarray(snapshot.get("task_pair_valid", np.zeros((NUM_UAV, 2), bool)))[uav_id]
        task_types = np.asarray(snapshot.get("task_type", np.zeros((NUM_UAV, 2), int)))[uav_id]
        targets = np.asarray(snapshot.get("task_target_id", np.full((NUM_UAV, 2), -1, int)))[uav_id]
        target_valid = np.asarray(snapshot.get("task_target_valid", np.zeros((NUM_UAV, 2), bool)))[uav_id]
        for valid, code, target, visible in zip(valid_tasks, task_types, targets, target_valid):
            if bool(valid) and int(code) == TASK_TYPE_ENCODING["COM"] and bool(visible):
                com_sr_id = int(target)
                break
        com_valid = (
            has_com and com_sr_id is not None and 0 <= com_sr_id < len(np.asarray(snapshot.get("sr_observable", ())))
            and _flag(snapshot, "sr_observable", com_sr_id)
        )
        row["com_target_valid"] = com_valid
        if com_valid:
            target = np.asarray(snapshot["sr_position_m"], dtype=float)[com_sr_id]
            relative = target - positions[uav_id]
            distance = float(np.linalg.norm(relative))
            capacity_valid = _flag(
                snapshot, "s2u_link_valid", (com_sr_id, uav_id)
            )
            row.update({
                "com_relative_x_m": float(relative[0]),
                "com_relative_y_m": float(relative[1]),
                "com_relative_z_m": float(relative[2]),
                "com_distance_3d_m": distance,
                "com_target_bearing_rad": math.atan2(float(relative[1]), float(relative[0])) if np.any(relative[:2]) else 0.0,
                "com_in_range": distance <= float(COMMUNICATION_RANGE_M),
                "com_reference_capacity_valid": capacity_valid,
                "com_reference_capacity_mbps": (
                    _scalar(
                        snapshot,
                        "s2u_reference_capacity_mbps",
                        (com_sr_id, uav_id),
                    )
                    if capacity_valid
                    else 0.0
                ),
                "com_sr_queue_valid": _flag(snapshot, "sr_queue_valid", com_sr_id),
                "com_sr_queue_empty": _flag(snapshot, "sr_queue_empty", com_sr_id),
                "com_sr_backlog_bits": _scalar(snapshot, "sr_backlog_bits", com_sr_id),
                "com_sr_packet_count": _scalar(snapshot, "sr_packet_count", com_sr_id),
                "com_sr_hol_deadline_valid": _flag(snapshot, "sr_hol_deadline_valid", com_sr_id),
                "com_sr_hol_remaining_deadline_s": _scalar(snapshot, "sr_hol_remaining_deadline_s", com_sr_id),
            })

        row.update({
            "uav_queue_valid": _flag(snapshot, "uav_queue_valid", uav_id),
            "uav_queue_empty": _flag(snapshot, "uav_queue_empty", uav_id),
            "uav_backlog_bits": _scalar(snapshot, "uav_backlog_bits", uav_id),
            "uav_fov_backlog_bits": _scalar(snapshot, "uav_vs_backlog_bits", uav_id),
            "uav_com_backlog_bits": _scalar(snapshot, "uav_com_backlog_bits", uav_id),
            "uav_hol_valid": _flag(snapshot, "uav_hol_valid", uav_id),
            "uav_hol_remaining_deadline_s": _scalar(snapshot, "uav_hol_remaining_deadline_s", uav_id),
            "u2g_link_valid": _flag(snapshot, "u2g_link_valid", uav_id),
            "u2g_distance_3d_m": _scalar(snapshot, "u2g_distance_m", uav_id),
            "u2g_in_range": _flag(snapshot, "u2g_in_range", uav_id),
            "u2g_reference_capacity_mbps": _scalar(snapshot, "u2g_reference_capacity_mbps", uav_id),
        })
        neighbor_valid = np.asarray(snapshot.get("u2u_link_valid", np.zeros((NUM_UAV, NUM_UAV), bool)), dtype=bool)[uav_id]
        neighbor_range = np.asarray(snapshot.get("u2u_in_range", np.zeros((NUM_UAV, NUM_UAV), bool)), dtype=bool)[uav_id]
        capacities = np.asarray(snapshot.get("u2u_reference_capacity_mbps", np.zeros((NUM_UAV, NUM_UAV))), dtype=float)[uav_id]
        candidates = np.flatnonzero(neighbor_valid & neighbor_range & np.isfinite(capacities) & (capacities > 0.0))
        if candidates.size:
            best_capacity = float(np.max(capacities[candidates]))
            best = int(min(int(candidate) for candidate in candidates if capacities[candidate] == best_capacity))
            relative = positions[best] - positions[uav_id]
            row.update({
                "best_u2u_neighbor_valid": True,
                "best_u2u_neighbor_id": float(best),
                "best_u2u_relative_x_m": float(relative[0]),
                "best_u2u_relative_y_m": float(relative[1]),
                "best_u2u_relative_z_m": float(relative[2]),
                "best_u2u_distance_3d_m": _scalar(snapshot, "u2u_distance_m", (uav_id, best)),
                "best_u2u_reference_capacity_mbps": best_capacity,
            })
        rows.append(row)
    return rows


def evaluate_rule_for_joint_action(
    compiled: CompiledStateRule,
    feature_rows: list[dict[str, Any]],
    movement_mask: Any,
    *,
    sigma_episode_start: float,
    weight: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Evaluate only TD3-controlled UAVs, clip once, and return diagnostics."""

    sigma = _number(sigma_episode_start, context="sigma_episode_start")
    if sigma < 0.0:
        raise ExplorerValidationError("sigma_episode_start must be non-negative")
    resolved_weight = _number(weight, context="state-rule weight")
    mask = np.asarray(movement_mask, dtype=bool).reshape(NUM_UAV)
    if len(feature_rows) != NUM_UAV:
        raise ExplorerValidationError("runtime feature rows must contain 16 UAVs")
    effective = np.zeros((NUM_UAV, 3), dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []
    for uav_id in range(NUM_UAV):
        item: dict[str, Any] = {
            "uav_id": uav_id, "movement_mask": bool(mask[uav_id]),
            "raw_bias": [0.0, 0.0, 0.0], "bounded_bias": [0.0, 0.0, 0.0],
            "effective_bias": [0.0, 0.0, 0.0], "clipped_components": 0,
            "runtime_failure": None, "trace": {"conditions_evaluated": 0, "conditions_true": 0, "short_circuits": 0},
            "features": _json_clone(feature_rows[uav_id]),
        }
        if mask[uav_id]:
            try:
                raw, trace = compiled.evaluate(feature_rows[uav_id])
                bounded = np.clip(raw, -sigma, sigma)
                applied = resolved_weight * bounded
                if not np.isfinite(applied).all():
                    raise ExplorerValidationError("effective bias is non-finite")
                effective[uav_id] = applied.astype(np.float32)
                item.update(
                    raw_bias=raw.tolist(), bounded_bias=bounded.tolist(),
                    effective_bias=applied.tolist(),
                    clipped_components=int(np.count_nonzero(raw != bounded)), trace=trace,
                )
            except (ArithmeticError, ExplorerValidationError, TypeError, ValueError) as exc:
                item["runtime_failure"] = f"{type(exc).__name__}: {exc}"
        diagnostics.append(item)
    return effective.reshape(-1), diagnostics
