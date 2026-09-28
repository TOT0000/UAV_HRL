"""Shared current-only interface, candidate schema, and prompt construction."""

from __future__ import annotations

import json
import math
from pathlib import Path
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
PROMPT_VERSION = "uav-hrl-llm-design-prompt-v2"
DESIGN_RUN_SCHEMA_VERSION = "uav-hrl-llm-design-run-v2"
APPROVED_ARTIFACT_SCHEMA_VERSION = "uav-hrl-approved-shared-feature-design-v2"
ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "llm_candidate_schema.json"
PROMPT_TEMPLATE_PATH = ROOT / "prompts" / "llm_design_prompt.txt"
OBS_KEYS = ("state", "movement_mask") + tuple(SNAPSHOT_FIELD_SPECS)


def candidate_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


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


def _state_schema_lines(fixed_metadata: dict[str, Any]) -> list[str]:
    checkpoint = fixed_metadata["compatibility_contract"]["source_checkpoint_contract"]
    schema = checkpoint.get("movement_state_feature_schema") or {}
    features = schema.get("features") or []
    if int(schema.get("dimension", -1)) != 531 or len(features) != 531:
        raise ValueError("fixed artifact lacks the authoritative 531-D state schema")
    lines = [
        "Original state obs['state']: shape (531,), dtype float32, unchanged ordering/scales.",
        "Indices 0..271 are 16 UAV blocks of 17 values. For UAV id k, base=17*k:",
    ]
    for item in features[:17]:
        local_name = str(item["name"]).split(".", 1)[1]
        lines.append(
            f"  base+{item['index']}: {local_name}; range [{item['minimum']},{item['maximum']}]; "
            f"{item['normalization']}"
        )
    lines.extend(
        (
            "Indices 272..527 are coverage_macro[row,column], a 16x16 row-major map of visited-cell fractions in [0,1].",
            f"Index 528: {features[528]['name']}; {features[528]['normalization']}.",
            f"Index 529: {features[529]['name']}; {features[529]['normalization']}.",
            f"Index 530: {features[530]['name']}; {features[530]['normalization']}.",
        )
    )
    return lines


def render_environment_interface(
    fixed_metadata: dict[str, Any], constants_metadata: dict[str, Any]
) -> str:
    lines = [
        f"Interface version: {OBS_INTERFACE_VERSION}",
        "Timing: every obs value is current-only and available before the movement action. No action, next-state, post-action delivery/energy/penalty, lambda, checkpoint, episode, scenario, or source identity is exposed.",
        "",
        *_state_schema_lines(fixed_metadata),
        "obs['movement_mask']: shape (16,), bool; true exactly where centralized movement control owns the UAV. An all-false mask is legal.",
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
            "- UAV arrays use UAV id as row index 0..15. task_type/task_target_id axes are [uav_id, assignment_slot], with at most two typed assignments.",
            "- RoI and SR arrays are compact padded rows. roi_id[row] and sr_id[row] give object IDs; never use an object ID as a compact row. Match IDs explicitly under roi_observable/sr_observable and mapping-valid flags.",
            "- U2U matrices are [sender_uav_id, receiver_uav_id]. U2G vectors are [sender_uav_id]. S2U matrices are [compact_sr_row, receiver_uav_id].",
            "- Invalid/padded IDs are -1; numeric padding is zero. Empty queues, no discovered RoI, no service target, and all-false masks are legal, not errors.",
            "- Reference capacities are current expected physical capacities at the stated reference bandwidth, not scheduled or realized service and not delivered data.",
            "",
            "Visual sensing and existing reward semantics:",
            "- For assigned VS geometry, h is relative altitude and d is horizontal UAV-RoI distance. b1=2*f/image_width and b2=2*f/image_length.",
            "- C9 uses G=clip(b1*h/(d+epsilon),0,1), violation=1-G, averaged over currently assigned VS pairs. model_range_valid is d<=b1*h with valid positive geometry.",
            "- After C9 is model-range-valid, d_L=(h^2+d^2)/(b1*h+d) and d_R=(h^2+d^2)/sqrt(b2^2*h^2+(1+b2^2)*d^2). C10 violation=1-clip(min(d_L,d_R)/(RoI_radius+epsilon),0,1), averaged only over finite C9-valid pairs.",
            "- vs_geometry_valid, vs_c10_geometry_valid, and vs_capture_valid are distinct. The exact C9 boundary may be model-range-valid while capture-invalid because of a horizontal corner ray.",
            "- Packet generation requires vs_capture_valid, but incomplete C10 coverage is not a separate hard gate. When capture is valid, physical size uses min(max(image_quantity,0),1); useful delivered VS bits additionally use frozen capture coverage.",
            "- COM violation=1-clip(communication_range_3d/(assigned_S2U_distance_3d+epsilon),0,1), averaged over all assigned COM pairs, including out-of-range pairs.",
            "- The three stored baseline penalties are per-task-type means and each has weight 1; they are post-action reward components and therefore are not present in obs.",
            "",
            "constants contains these verified fixed entries (candidate code accesses constants[name] to get the value):",
        )
    )
    for name, item in constants_metadata.items():
        lines.append(
            f"- constants['{name}']: {item['dtype']}, unit {item['unit'] or 'none'}, value={json.dumps(item['value'], separators=(',', ':'))}; {item['meaning']}; source={item['source']}."
        )
    return "\n".join(lines)


def format_schema_and_example(schema: dict[str, Any]) -> str:
    example = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_name": "format_only_control_fraction_example",
        "reward_input_mode": "current_only",
        "features": [
            {
                "index": 0,
                "name": "controlled_uav_fraction",
                "dtype": "float32",
                "description": "Example of mask-aware normalization: the fraction of UAVs currently controlled by TD3. Its zero reward weight makes this an interface example, not a recommended reward design.",
                "range": {"minimum": 0.0, "maximum": 1.0},
                "source_fields": ["obs.movement_mask", "constants.num_uav"],
                "formula": "count_nonzero(movement_mask) / max(num_uav, 1), clipped to [0,1]",
                "missing_data_rule": "An all-false movement mask is a valid empty controlled set and returns 0.",
                "reward_weight": 0.0,
            }
        ],
        "code": (
            "def compute_extra_state(obs, constants):\n"
            "    denominator = max(float(constants[\"num_uav\"]), 1.0)\n"
            "    controlled_fraction = np.clip(\n"
            "        float(np.count_nonzero(obs[\"movement_mask\"])) / denominator,\n"
            "        0.0,\n"
            "        1.0,\n"
            "    )\n"
            "    return np.asarray([controlled_fraction], dtype=np.float32)\n"
        ),
    }
    return (
        "JSON Schema (Draft 2020-12; no additional fields):\n"
        + json.dumps(schema, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n\nParseable interface example (reward_weight=0 makes it formatting guidance, not a suggested reward design):\n"
        + json.dumps(example, indent=2, ensure_ascii=False, allow_nan=False)
    )


SUPPORTED_OPERATIONS = """Supported code subset:
- No import statements. `np` is already provided.
- Literal arithmetic, comparisons, Boolean expressions, indexing with literal obs/constants keys, local variables, if/for constructs, and returns.
- Safe built-ins: abs, bool, enumerate, float, int, len, list, max, min, range, sum, tuple, zip.
- NumPy: abs, all, any, arange, array, asarray, bool_, clip, concatenate, count_nonzero, exp, float32, float64, int32, int64, isfinite, log, log1p, maximum, mean, minimum, ones, sqrt, stack, sum, where, zeros, and linalg.norm.
- A feature that directly copies one original-state scalar, or two features with the same statically resolved output expression, is rejected. Derived quantities such as relative distances, normalized deadlines, and queue loads remain allowed.
- Do not use array/object methods, imports, file/network/process access, eval/exec, reflection, globals, mutation of obs/constants, randomness, or time."""


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
        "{{ENVIRONMENT_INTERFACE}}": render_environment_interface(
            fixed_metadata, constants_metadata
        ),
        "{{BETA}}": format(float(beta), ".17g"),
        "{{SUPPORTED_OPERATIONS}}": SUPPORTED_OPERATIONS,
        "{{BASELINE_RESULTS_BY_LAMBDA}}": baseline_lines(baseline_report),
        "{{IMPROVEMENT_TOLERANCE}}": (
            f"margin(lambda)=max({absolute_tolerance:.17g}, "
            f"{relative_tolerance:.17g}*abs(L_baseline(lambda))); require "
            "L_candidate < L_baseline - margin for every lambda."
        ),
        "{{OUTPUT_JSON_SPEC_AND_EXAMPLE}}": format_schema_and_example(
            candidate_schema()
        ),
        "{{ROUND_REQUEST}}": round_request,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    if "{{" in template or "}}" in template:
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
