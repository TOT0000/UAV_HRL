"""Shared current-only interface, candidate schema, and prompt construction."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from experiment_config import (
    GROUND_STATION_POSITION_M,
    NUM_UAV,
    ROI_COUNT_MAX,
    S2U_COMMUNICATION_RANGE_M,
    TASK_POTENTIAL_NORMALIZATION_EPSILON,
)
from llm_baseline import load_fixed_samples
from replay_auxiliary import SNAPSHOT_FIELD_SPECS


CANDIDATE_SCHEMA_VERSION = "uav-hrl-llm-shared-feature-candidate-v2"
OBS_INTERFACE_VERSION = "uav-hrl-llm-current-observation-v1"
PROMPT_VERSION = "uav-hrl-llm-design-prompt-v7"
DESIGN_RUN_SCHEMA_VERSION = "uav-hrl-llm-design-run-v4"
APPROVED_ARTIFACT_SCHEMA_VERSION = "uav-hrl-approved-shared-feature-design-v2"
ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "llm_candidate_schema.json"
PROMPT_TEMPLATE_PATH = ROOT / "prompts" / "llm_design_prompt.txt"
OBS_KEYS = ("state", "movement_mask") + tuple(SNAPSHOT_FIELD_SPECS)


def candidate_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def model_candidate_schema() -> dict[str, Any]:
    """Schema exposed to models; host-only artifact metadata is deliberately absent."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "uav-hrl-llm-model-candidate-v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["features", "code"],
        "properties": {
            "features": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "description", "reward_weight"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string", "minLength": 1},
                        "reward_weight": {"type": "number"},
                    },
                },
            },
            "code": {"type": "string", "minLength": 1},
        },
    }


def allowed_source_fields(constants: dict[str, Any]) -> set[str]:
    return {f"obs.{key}" for key in OBS_KEYS} | {
        f"constants.{key}" for key in constants
    }


def build_obs_arrays(fixed_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expose only action-preceding policy information to candidate code."""
    result = {
        "state": np.asarray(fixed_arrays["state"]),
        "movement_mask": np.asarray(fixed_arrays["current_movement_mask"]),
    }
    for name in SNAPSHOT_FIELD_SPECS:
        result[name] = np.asarray(fixed_arrays[f"current_{name}"])
    return result


def _require(mapping: dict, key: str, source: str):
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"required constant {key!r} is absent from {source}")
    return value


def build_constants(fixed_metadata: dict[str, Any]) -> dict[str, Any]:
    contract = fixed_metadata.get("compatibility_contract") or {}
    checkpoint = contract.get("source_checkpoint_contract") or {}
    environment = contract.get("source_training_environment_contract") or {}
    visual = checkpoint.get("visual_sensing_configuration") or {}
    channel = checkpoint.get("channel_configuration") or {}
    deadlines = checkpoint.get("production_task_deadline_seconds") or {}
    camera = visual.get("vs_camera") or visual.get("camera") or {}
    task_config = checkpoint.get("task_potential_configuration") or {}
    constants = {
        "num_uav": {
            "value": int(_require(checkpoint, "num_uav", "checkpoint contract")),
            "dtype": "int",
            "unit": None,
            "meaning": "number of UAV rows; UAV row index equals UAV id",
            "source": "fixed sample source_checkpoint_contract.num_uav",
        },
        "max_roi_rows": {
            "value": int(ROI_COUNT_MAX),
            "dtype": "int",
            "unit": None,
            "meaning": "padded RoI/SR row count",
            "source": "experiment_config.ROI_COUNT_MAX",
        },
        "environment_width_m": {
            "value": float(
                _require(
                    environment,
                    "training_environment_width_m",
                    "training environment contract",
                )
            ),
            "dtype": "float",
            "unit": "m",
            "meaning": "training-area width",
            "source": "fixed sample source_training_environment_contract",
        },
        "environment_height_m": {
            "value": float(
                _require(
                    environment,
                    "training_environment_height_m",
                    "training environment contract",
                )
            ),
            "dtype": "float",
            "unit": "m",
            "meaning": "training-area height",
            "source": "fixed sample source_training_environment_contract",
        },
        "episode_seconds": {
            "value": float(_require(contract, "episode_seconds", "compatibility contract")),
            "dtype": "float",
            "unit": "s",
            "meaning": "episode duration",
            "source": "fixed sample compatibility_contract.episode_seconds",
        },
        "movement_interval_seconds": {
            "value": 1.0,
            "dtype": "float",
            "unit": "s",
            "meaning": "one centralized movement decision interval",
            "source": "formal environment contract in HRL_task_aware",
        },
        "routing_slot_seconds": {
            "value": float(_require(channel, "routing_slot_seconds", "channel contract")),
            "dtype": "float",
            "unit": "s",
            "meaning": "routing/service slot duration",
            "source": "fixed sample channel_configuration.routing_slot_seconds",
        },
        "routing_slots_per_movement": {
            "value": int(
                round(1.0 / float(_require(channel, "routing_slot_seconds", "channel contract")))
            ),
            "dtype": "int",
            "unit": "slots",
            "meaning": "routing slots in each movement interval",
            "source": "derived from verified movement and routing intervals",
        },
        "vs_deadline_seconds": {
            "value": float(_require(deadlines, "FOV", "task deadline contract")),
            "dtype": "float",
            "unit": "s",
            "meaning": "visual packet end-to-end deadline from generation",
            "source": "fixed sample production_task_deadline_seconds.FOV",
        },
        "com_deadline_seconds": {
            "value": float(_require(deadlines, "COM", "task deadline contract")),
            "dtype": "float",
            "unit": "s",
            "meaning": "communication packet end-to-end deadline from generation",
            "source": "fixed sample production_task_deadline_seconds.COM",
        },
        "communication_range_m": {
            "value": float(
                checkpoint.get("maximum_3d_communication_distance_m", S2U_COMMUNICATION_RANGE_M)
            ),
            "dtype": "float",
            "unit": "m",
            "meaning": "inclusive S2U/U2U/U2G 3-D range",
            "source": "fixed sample checkpoint communication contract",
        },
        "ground_station_position_m": {
            "value": [float(value) for value in checkpoint.get(
                "ground_station_position_m", GROUND_STATION_POSITION_M
            )],
            "dtype": "float[3]",
            "unit": "m",
            "meaning": "fixed GS xyz position",
            "source": "fixed sample checkpoint contract",
        },
        "vs_camera_f_m": {
            "value": float(_require(camera, "f_m", "visual sensing camera")),
            "dtype": "float",
            "unit": "m",
            "meaning": "VS camera focal length",
            "source": "fixed sample visual_sensing_configuration.vs_camera",
        },
        "vs_camera_image_width_m": {
            "value": float(_require(camera, "image_width_m", "visual sensing camera")),
            "dtype": "float",
            "unit": "m",
            "meaning": "VS camera sensor width in the tilt plane",
            "source": "fixed sample visual_sensing_configuration.vs_camera",
        },
        "vs_camera_image_length_m": {
            "value": float(_require(camera, "image_length_m", "visual sensing camera")),
            "dtype": "float",
            "unit": "m",
            "meaning": "VS camera cross-track sensor length",
            "source": "fixed sample visual_sensing_configuration.vs_camera",
        },
        "vs_b1": {
            "value": float(_require(visual, "b1", "visual sensing configuration")),
            "dtype": "float",
            "unit": "ratio",
            "meaning": "2*f/image_width used by C9 and d_L",
            "source": "fixed sample visual_sensing_configuration.b1",
        },
        "vs_b2": {
            "value": float(_require(visual, "b2", "visual sensing configuration")),
            "dtype": "float",
            "unit": "ratio",
            "meaning": "2*f/image_length used by d_R",
            "source": "fixed sample visual_sensing_configuration.b2",
        },
        "default_roi_radius_m": {
            "value": float(_require(visual, "default_roi_radius_m", "visual sensing configuration")),
            "dtype": "float",
            "unit": "m",
            "meaning": "formal scenario RoI radius used when an object has no override",
            "source": "fixed sample visual_sensing_configuration",
        },
        "constraint_epsilon": {
            "value": float(TASK_POTENTIAL_NORMALIZATION_EPSILON),
            "dtype": "float",
            "unit": None,
            "meaning": "denominator epsilon in C9/C10/COM penalties",
            "source": "experiment_config.TASK_POTENTIAL_NORMALIZATION_EPSILON",
        },
        "task_type_encoding": {
            "value": (contract.get("replay_auxiliary") or {}).get(
                "task_type_encoding"
            ),
            "dtype": "mapping[str,int]",
            "unit": None,
            "meaning": "numeric task codes in obs.task_type",
            "source": "fixed sample replay auxiliary metadata",
        },
        "packet_type_encoding": {
            "value": (contract.get("replay_auxiliary") or {}).get(
                "packet_type_encoding"
            ),
            "dtype": "mapping[str,int]",
            "unit": None,
            "meaning": "numeric HOL packet type codes",
            "source": "fixed sample replay auxiliary metadata",
        },
        "constraint_penalty_weights": {
            "value": task_config.get(
                "constraint_penalty_weights",
                {"c9": 1.0, "c10": 1.0, "com_range": 1.0},
            ),
            "dtype": "mapping[str,float]",
            "unit": None,
            "meaning": "existing immediate movement reward penalty weights",
            "source": "fixed sample task_potential_configuration",
        },
    }
    for name, item in constants.items():
        if item["value"] is None:
            raise ValueError(f"constant {name} has no verified value")
    return constants


def runtime_constants(constants_metadata: dict[str, Any]) -> dict[str, Any]:
    return {name: item["value"] for name, item in constants_metadata.items()}


def runtime_diagnostic_contract(
    fixed_metadata: dict[str, Any], constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build worker-safe diagnostics from the authoritative saved interface.

    The worker receives only structural metadata.  It never receives provenance,
    credentials, or a second independently maintained state-layout definition.
    """

    checkpoint = fixed_metadata["compatibility_contract"][
        "source_checkpoint_contract"
    ]
    schema = checkpoint.get("movement_state_feature_schema") or {}
    features = schema.get("features") or []
    state_dimension = int(schema.get("dimension", -1))
    if state_dimension <= 0 or len(features) != state_dimension:
        raise ValueError("fixed artifact lacks a complete authoritative state schema")
    num_uav = int(constants_metadata["num_uav"]["value"])
    uav_indices = [
        int(item["index"])
        for item in features
        if isinstance(item, dict) and str(item.get("name", "")).startswith("uav_")
    ]
    if not uav_indices or len(uav_indices) % num_uav:
        raise ValueError("authoritative state schema has inconsistent UAV blocks")
    local_dimension = len(uav_indices) // num_uav
    uav_start = min(uav_indices)
    uav_stop = max(uav_indices) + 1
    if uav_start != 0 or uav_stop != num_uav * local_dimension:
        raise ValueError("authoritative state schema UAV blocks are not contiguous")
    return {
        "observation_interface_version": OBS_INTERFACE_VERSION,
        "state": {
            "shape": [state_dimension],
            "dtype": "float32",
            "uav_block": {
                "start": uav_start,
                "stop_exclusive": uav_stop,
                "num_uav": num_uav,
                "features_per_uav": local_dimension,
                "layout_source": (
                    "saved source_checkpoint_contract.movement_state_feature_schema"
                ),
            },
        },
        "movement_mask": {
            "shape": [num_uav],
            "dtype": "bool",
            "semantics": (
                "true exactly where centralized movement control owns the UAV"
            ),
        },
    }


def _state_schema_lines(fixed_metadata: dict[str, Any]) -> list[str]:
    checkpoint = fixed_metadata["compatibility_contract"]["source_checkpoint_contract"]
    schema = checkpoint.get("movement_state_feature_schema") or {}
    features = schema.get("features") or []
    dimension = int(schema.get("dimension", -1))
    if dimension <= 0 or len(features) != dimension:
        raise ValueError("fixed artifact lacks a complete authoritative state schema")

    num_uav = int(checkpoint.get("num_uav", -1))
    uav_features: dict[int, list[dict[str, Any]]] = {}
    coverage_features = []
    global_features = []
    for item in features:
        name = str(item.get("name", ""))
        prefix, separator, _ = name.partition(".")
        if separator and prefix.startswith("uav_") and prefix[4:].isdigit():
            uav_features.setdefault(int(prefix[4:]), []).append(item)
        elif name.startswith("coverage_macro[") and name.endswith("]"):
            coverage_features.append(item)
        else:
            global_features.append(item)
    if num_uav <= 0 or set(uav_features) != set(range(num_uav)):
        raise ValueError("authoritative state schema has inconsistent UAV rows")
    local_dimensions = {len(items) for items in uav_features.values()}
    if len(local_dimensions) != 1:
        raise ValueError("authoritative state schema has inconsistent UAV block widths")
    local_dimension = local_dimensions.pop()
    uav_stop = num_uav * local_dimension
    uav_indices = sorted(
        int(item["index"]) for items in uav_features.values() for item in items
    )
    if uav_indices != list(range(uav_stop)):
        raise ValueError("authoritative state schema UAV blocks are not contiguous")

    coverage_indices = sorted(int(item["index"]) for item in coverage_features)
    if coverage_indices:
        coverage_start = coverage_indices[0]
        coverage_stop = coverage_indices[-1] + 1
        if coverage_indices != list(range(coverage_start, coverage_stop)):
            raise ValueError("authoritative coverage state block is not contiguous")
        coordinates = []
        for item in coverage_features:
            raw = str(item["name"])[len("coverage_macro[") : -1]
            try:
                row, column = (int(value) for value in raw.split(",", 1))
            except (TypeError, ValueError) as exc:
                raise ValueError("authoritative coverage feature name is invalid") from exc
            coordinates.append((row, column))
        row_count = max(row for row, _ in coordinates) + 1
        column_count = max(column for _, column in coordinates) + 1
        if set(coordinates) != {
            (row, column)
            for row in range(row_count)
            for column in range(column_count)
        }:
            raise ValueError("authoritative coverage state grid is incomplete")
    else:
        coverage_start = coverage_stop = row_count = column_count = 0

    first_uav = sorted(uav_features[0], key=lambda item: int(item["index"]))
    lines = [
        f"Original state obs['state']: shape ({dimension},), dtype float32, unchanged ordering/scales.",
        (
            f"Indices 0..{uav_stop - 1} are {num_uav} UAV blocks of "
            f"{local_dimension} values. For UAV id k, base={local_dimension}*k:"
        ),
    ]
    for item in first_uav:
        local_name = str(item["name"]).split(".", 1)[1]
        local_offset = int(item["index"])
        lines.append(
            f"  base+{local_offset}: {local_name}; range [{item['minimum']},{item['maximum']}]; "
            f"{item['normalization']}"
        )
    if coverage_features:
        lines.append(
            f"Indices {coverage_start}..{coverage_stop - 1} are "
            f"coverage_macro[row,column], a {row_count}x{column_count} row-major "
            "map of visited-cell fractions in [0,1]."
        )
    for item in sorted(global_features, key=lambda value: int(value["index"])):
        lines.append(
            f"Index {item['index']}: {item['name']}; {item['normalization']}."
        )
    return lines


def render_environment_interface(
    fixed_metadata: dict[str, Any],
    constants_metadata: dict[str, Any],
    *,
    include_system_semantics: bool = True,
) -> str:
    num_uav = int(constants_metadata["num_uav"]["value"])
    max_task_slots = int(SNAPSHOT_FIELD_SPECS["task_type"]["shape"][1])
    lines = [
        f"Interface version: {OBS_INTERFACE_VERSION}",
        "Timing: every obs value is current-only and available before the movement action. No action, next-state, post-action delivery/energy/penalty, lambda, checkpoint, episode, scenario, or source identity is exposed.",
        "",
        *_state_schema_lines(fixed_metadata),
        f"obs['movement_mask']: shape ({num_uav},), bool; true exactly where centralized movement control owns the UAV. An all-false mask is legal.",
        "",
        "Current auxiliary fields (all arrays are per one observation):",
    ]
    for name, spec in SNAPSHOT_FIELD_SPECS.items():
        lines.append(
            f"- obs['{name}']: shape {tuple(spec['shape'])}, dtype {np.dtype(spec['dtype']).name}, "
            f"unit {spec['unit'] or 'none'}, validity {spec['mask'] or 'always/own valid flag'}; {spec['semantics']}."
        )
    lines.extend(
        (
            "",
            "Axis and missing-data rules:",
            f"- UAV arrays use UAV id as row index 0..{num_uav - 1}. task_type/task_target_id axes are [uav_id, assignment_slot], with at most {max_task_slots} typed assignments.",
            "- RoI and SR arrays are compact padded rows. roi_id[row] and sr_id[row] give object IDs; never use an object ID as a compact row. Match IDs explicitly under roi_observable/sr_observable and mapping-valid flags.",
            "- U2U matrices are [sender_uav_id, receiver_uav_id]. U2G vectors are [sender_uav_id]. S2U matrices are [compact_sr_row, receiver_uav_id].",
            "- Invalid/padded IDs are -1; numeric padding is zero. Empty queues, no discovered RoI, no service target, and all-false masks are legal, not errors.",
            "- Reference capacities are current expected physical capacities at the stated reference bandwidth, not scheduled or realized service and not delivered data.",
        )
    )
    if include_system_semantics:
        lines.extend(
            (
                "",
                "Visual sensing and existing reward semantics:",
                "- For assigned VS geometry, h is relative altitude and d is horizontal UAV-RoI distance. b1=2*f/image_width and b2=2*f/image_length.",
                "- C9 uses G=clip(b1*h/(d+epsilon),0,1), violation=1-G, averaged over currently assigned VS pairs. model_range_valid is d<=b1*h with valid positive geometry.",
                "- After C9 is model-range-valid, d_L=(h^2+d^2)/(b1*h+d) and d_R=(h^2+d^2)/sqrt(b2^2*h^2+(1+b2^2)*d^2). C10 violation=1-clip(min(d_L,d_R)/(RoI_radius+epsilon),0,1), averaged only over finite C9-valid pairs.",
                "- vs_geometry_valid, vs_c10_geometry_valid, and vs_capture_valid are distinct. The exact C9 boundary may be model-range-valid while capture-invalid because of a horizontal corner ray.",
                "- Packet generation requires vs_capture_valid, but incomplete C10 coverage is not a separate hard gate. image_quantity is RoI area divided by the oblique camera-footprint area and its raw value may exceed 1. A valid capture creates physical_bits=packet_max_bits*clip(image_quantity,0,1). On timely GS delivery, useful VS bits equal those frozen physical bits times frozen capture coverage; useful COM bits equal timely physical bits.",
                "- COM violation=1-clip(communication_range_m/(assigned_S2U_distance_3d+epsilon),0,1), averaged over all assigned COM pairs, including out-of-range pairs.",
                "- The three stored baseline penalties are per-task-type means and each has weight 1; they are post-action reward components and therefore are not present in obs.",
            )
        )
    lines.extend(
        (
            "",
            "constants contains these verified fixed entries (candidate code accesses constants[name] to get the value):",
        )
    )
    for name, item in constants_metadata.items():
        lines.append(
            f"- constants['{name}']: {item['dtype']}, unit {item['unit'] or 'none'}, value={json.dumps(item['value'], separators=(',', ':'))}; {item['meaning']}; source={item['source']}."
        )
    return "\n".join(lines)


def minimal_model_candidate_example() -> dict[str, Any]:
    """Short executable interface example, not a recommended feature design."""

    return {
        "features": [
            {
                "name": "controlled_uav_nonempty_queue_fraction",
                "description": (
                    "Fraction of controlled UAVs with valid queue summaries that are "
                    "nonempty; fixed [0,1] normalization, with 0 for an empty valid set."
                ),
                "reward_weight": 0.0,
            }
        ],
        "code": (
            "def compute_extra_state(obs, constants):\n"
            "    applicable = obs[\"movement_mask\"] & obs[\"uav_queue_valid\"]\n"
            "    count = int(np.count_nonzero(applicable))\n"
            "    value = 0.0\n"
            "    if count > 0:\n"
            "        value = float(np.count_nonzero(applicable & (~obs[\"uav_queue_empty\"]))) / float(count)\n"
            "    return [value]\n"
        ),
    }


def format_schema_and_example(schema: dict[str, Any] | None = None) -> str:
    schema = model_candidate_schema() if schema is None else schema
    example = minimal_model_candidate_example()
    return (
        "Model submission schema (Draft 2020-12; no additional fields):\n"
        + json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n\nParseable executable example (reward_weight=0 makes it interface guidance, not a suggested reward design):\n"
        + json.dumps(example, indent=2, ensure_ascii=False, allow_nan=False)
    )


SUPPORTED_OPERATIONS = """Supported code subset:
- No import statements. `np` is already provided.
- Literal arithmetic, comparisons, Boolean expressions, indexing with literal obs/constants keys, local variables, if/for constructs, and returns.
- Safe built-ins: abs, bool, enumerate, float, int, len, list, max, min, range, sum, tuple, zip.
- NumPy: abs, all, any, arange, array, asarray, bool_, clip, concatenate, count_nonzero, exp, float32, float64, int32, int64, isfinite, log, log1p, maximum, mean, minimum, ones, sqrt, stack, sum, where, zeros, and linalg.norm.
- Indexed writes may fill arrays independently created inside the function by supported allocating operations such as np.zeros, np.ones, np.arange, and np.array. Safe local aliases and slices of those arrays remain writable.
- Augmented assignment may update a proven immutable numeric local or an independently created local NumPy array. It may not update an input-backed or uncertain target.
- Do not store input-backed or otherwise mutable references in Python containers for later nested mutation; that container-reference pattern is conservatively rejected.
- obs, constants, and aliases, slices, or np.asarray views that may share their data must remain unchanged. If ownership cannot be confirmed across every control-flow path, indexed writes are rejected.
- A feature that directly copies one original-state scalar, or two features with the same statically resolved output expression, is rejected. Derived quantities such as relative distances, normalized deadlines, and queue loads remain allowed.
- The same operation whitelist still applies: do not use array/object methods, imports, file/network/process access, eval/exec, reflection, globals, randomness, or time."""


def baseline_lines(baseline_report: dict[str, Any]) -> str:
    params = baseline_report["parameters"]["lambda_values_mbit_per_joule"]
    estimates = baseline_report["lipschitz"]["estimates_by_lambda_mbit_per_joule"]
    lines = []
    for value in params:
        key = format(float(value), ".17g")
        result = estimates.get(key)
        if result is None or result.get("l_hat") is None:
            raise ValueError(f"baseline lambda {key} has no usable estimate")
        lines.append(f"- lambda={key} Mbit/J: L_hat={result['l_hat']:.17g}")
    return "\n".join(lines)


def movement_and_energy_spec(constants_metadata: dict[str, Any]) -> str:
    width = constants_metadata["environment_width_m"]["value"]
    height = constants_metadata["environment_height_m"]["value"]
    return (
        "The actor emits one [speed_scalar, heading_scalar, vertical_scalar] block "
        "per UAV. Speed and vertical scalars are clamped to [-1,1]; heading wraps "
        "periodically to [-1,1). They decode as v_xy=5*(speed_scalar+1) m/s, "
        "theta=pi*heading_scalar rad, and v_z=2*vertical_scalar m/s, with "
        "[v_x,v_y]=v_xy*[cos(theta),sin(theta)]. Blocks outside movement_mask are "
        "replaced by hover [-1,0,0]; Search control may then replace its UAV commands. "
        "The command is held over four 0.25 s substeps. Each proposal uses p'=p+v*dt, "
        f"clips x to [0,{width:g}] m and y to [0,{height:g}] m and altitude to the "
        "UAV's configured AGL bounds; UAV 0 is additionally projected inside the "
        "configured hard 3-D GS gateway sphere. Energy is charged from actual projected "
        "displacement: E=P(v)*dt. For speed V=||v||, direction q=v/V (zero when "
        "V=0), climb sine s=v_z/V (zero when V=0), and per-rotor thrust "
        "T=||m*(0,0,-g)-0.5*rho*V^2*S_FP*q||/n_r, the canonical power is "
        "P=n_r*(P_profile+P_induced+P_climb+P_parasite), where "
        "P_profile=(delta/8)*(T/(c_T*rho*A)+3*V^2)*sqrt(T*rho*c_s^2*A/c_T), "
        "P_induced=(1+c_f)*T*sqrt(sqrt(T^2/(4*rho^2*A^2)+V^4/4)-V^2/2), "
        "P_climb=m*g*V*s/n_r, and P_parasite=0.5*d_0*V^3*rho*c_s*A. It uses "
        "n_r=4, rho=1.293 kg/m^3, "
        "S_FP=0.01 m^2, g=9.8 m/s^2, m=2 kg, delta=0.012, c_T=0.302, "
        "c_s=0.0955, c_f=0.131, rotor area A=0.0314 m^2, and d_0=0.834."
    )


def visual_sensing_spec(constants_metadata: dict[str, Any]) -> str:
    return (
        "Camera b1=2*f/image_width and b2=2*f/image_length use the constants below. "
        "C9 proximity is G=clip(b1*h/(d+epsilon),0,1); model-range validity requires "
        "finite positive geometry and d<=b1*h. C9 penalty is mean(1-G) over assigned "
        "VS pairs. For C9-valid pairs, d_L=(h^2+d^2)/(b1*h+d) and "
        "d_R=(h^2+d^2)/sqrt(b2^2*h^2+(1+b2^2)*d^2); C10 penalty is "
        "mean(1-clip(min(d_L,d_R)/(RoI_radius+epsilon),0,1)) over finite eligible "
        "pairs. Capture validity is a distinct polygon/corner-ray geometry check; C10 "
        "incomplete coverage is not an additional packet-generation gate. Coverage is "
        "the camera-footprint/RoI intersection area divided by pi*radius^2. The camera "
        "footprint area is the area of the same canonical oblique footprint polygon, and "
        "image_quantity=RoI area/camera footprint area; this raw ratio may exceed 1. A valid capture "
        "creates physical_bits=packet_max_bits*clip(image_quantity,0,1), including partial "
        "coverage. Timely useful VS bits at GS equal the frozen physical bits times the "
        "frozen capture coverage. geometry-valid, C10-valid, and capture-valid flags in "
        "obs are intentionally distinct."
    )


def communication_spec(constants_metadata: dict[str, Any]) -> str:
    distance = constants_metadata["communication_range_m"]["value"]
    return (
        f"S2U, U2U, and U2G links use inclusive finite 3-D distance <= {distance:g} m. "
        "A2G (S2U/U2G) path loss is free-space loss plus the currently sampled LoS or "
        "NLoS excess loss; U2U uses the directed altitude-dependent A2A loss. Expected "
        "capacities are E[B*log2(1+SNR*G)]/1e6 Mbit/s, with Rician fading for LoS/U2U "
        "and Rayleigh fading for NLoS. Runtime service uses 50 fading blocks per 0.25 s "
        "slot and equal FDMA across active S2U/U2U/U2G links in one shared 10 MHz pool. "
        "The obs reference capacities are current expected physical capacities at their "
        "documented reference bandwidth; they are not the allocated-bandwidth block "
        "service or delivered bits. COM range penalty is mean(1-clip(R_com/(d_3D+epsilon),"
        "0,1)) over every assigned COM UAV-SR pair, whether or not upload traffic is active."
    )


def render_prompt(
    *,
    fixed_metadata: dict[str, Any],
    baseline_report: dict[str, Any],
    constants_metadata: dict[str, Any],
    beta: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    round_request: str,
) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{MOVEMENT_AND_ENERGY_SPEC}}": movement_and_energy_spec(constants_metadata),
        "{{VISUAL_SENSING_SPEC}}": visual_sensing_spec(constants_metadata),
        "{{COMMUNICATION_SPEC}}": communication_spec(constants_metadata),
        "{{INPUT_FIELD_TABLE}}": render_environment_interface(
            fixed_metadata, constants_metadata, include_system_semantics=False
        ),
        "{{BETA}}": format(float(beta), ".17g"),
        "{{SUPPORTED_OPERATIONS}}": SUPPORTED_OPERATIONS,
        "{{MINIMAL_EXECUTABLE_EXAMPLE}}": json.dumps(
            minimal_model_candidate_example(), indent=2, ensure_ascii=False, allow_nan=False
        ),
        "{{BASELINE_AND_ACCEPTANCE_RULE}}": (
            "Evaluated baseline estimates:\n"
            + baseline_lines(baseline_report)
            + "\nAcceptance: "
            f"margin(lambda)=max({absolute_tolerance:.17g}, "
            f"{relative_tolerance:.17g}*abs(L_baseline(lambda))); require "
            "L_candidate < L_baseline - margin for every lambda."
        ),
        "{{ROUND_CONTEXT}}": round_request,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    if re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", template):
        raise RuntimeError("prompt contains an unfilled placeholder")
    return template


def estimate_token_budget(
    prompt: str, *, context_length: int, max_output_tokens: int
) -> dict[str, Any]:
    """Conservative tokenizer-free estimate, explicitly not an exact count."""
    characters = len(prompt)
    utf8_bytes = len(prompt.encode("utf-8"))
    words = len(prompt.split())
    # English, Python, and JSON punctuation vary materially by tokenizer. These
    # bounds intentionally expose that uncertainty instead of claiming a count.
    lower = int(math.ceil(characters / 4.5))
    central = int(math.ceil(characters / 3.5))
    upper = int(math.ceil(characters / 2.5))
    chat_template_overhead_upper = 128
    required_upper = upper + chat_template_overhead_upper + int(max_output_tokens)
    return {
        "method": (
            "tokenizer unavailable; character-based English/JSON/Python estimate "
            "reported as an uncertainty interval, not an exact token count"
        ),
        "exact": False,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "whitespace_words": words,
        "estimated_prompt_tokens_lower": lower,
        "estimated_prompt_tokens_central": central,
        "estimated_prompt_tokens_upper": upper,
        "chat_template_overhead_upper": chat_template_overhead_upper,
        "reserved_output_tokens": int(max_output_tokens),
        "output_reservation_scope": (
            "the server's completion budget; reasoning and final-answer tokens, "
            "when the model exposes reasoning, must both fit within this reservation"
        ),
        "client_context_budget": int(context_length),
        "estimated_total_upper": required_upper,
        "fits_client_budget": required_upper <= int(context_length),
    }


def load_design_inputs(fixed_directory: str | Path):
    directory = Path(fixed_directory).resolve()
    arrays, fixed_metadata = load_fixed_samples(directory)
    report_path = directory / "baseline_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"baseline report is missing: {report_path}")
    baseline = json.loads(report_path.read_text(encoding="utf-8"))
    reference = baseline.get("fixed_samples") or {}
    if reference.get("sample_content_sha256") != fixed_metadata.get(
        "sample_content_sha256"
    ):
        raise ValueError("baseline report and fixed sample content hashes disagree")
    fixed_pair = fixed_metadata.get("pair_contract") or {}
    lipschitz = baseline.get("lipschitz") or {}
    for key in (
        "all_pair_count",
        "primary_pair_count",
        "zero_distance_pair_count",
        "near_zero_distance_pair_count",
        "primary_pair_set_sha256",
    ):
        if fixed_pair.get(key) != lipschitz.get(key):
            raise ValueError(f"baseline report and fixed sample pair contract disagree: {key}")
    lambdas = tuple(
        float(value)
        for value in baseline["parameters"]["lambda_values_mbit_per_joule"]
    )
    saved_lambdas = tuple(
        float(value)
        for value in fixed_pair.get("baseline_lambda_values_mbit_per_joule", ())
    )
    if lambdas != saved_lambdas:
        raise ValueError("baseline lambda list disagrees with fixed sample metadata")
    constants_metadata = build_constants(fixed_metadata)
    return arrays, fixed_metadata, baseline, constants_metadata
