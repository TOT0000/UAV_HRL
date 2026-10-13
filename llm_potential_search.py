"""Train-and-evaluate every LLM potential candidate without proxy screening.

This workflow is intentionally independent from the Lipschitz, reviewer-agent,
and complete-episode ordering searches.  It generates one candidate per model
request, validates only executable/runtime contracts, trains all four accepted
candidates from scratch, evaluates their episode-400 checkpoints on the exact
baseline manifest, and feeds objective results into the next round.
"""

from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np

from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    parse_candidate_json_envelope,
)
from llm_design import (
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    _effective_context_budget,
    _git_sha,
    _slug,
    _write_json,
    allocate_design_directory,
    model_inventory_summary,
)
from llm_design_contract import (
    SUPPORTED_OPERATIONS,
    build_constants,
    communication_spec,
    estimate_token_budget,
    minimal_model_candidate_example,
    movement_and_energy_spec,
    render_environment_interface,
    visual_sensing_spec,
    build_obs_arrays,
)
from llm_episode_search import (
    EpisodeDataset,
    EpisodeRecord,
    _call_model,
    _make_client,
    _role_config,
    load_complete_episode_dataset,
)
from llm_potential import (
    DEFAULT_BETA,
    MergeableMoments,
    candidate_fingerprint,
    save_potential_artifact,
    summarize_component_records,
    validate_potential_candidate,
)
from llm_numeric_operations import NUMERIC_OPERATION_RULES_VERSION
from experiment_config import DEFAULT_TRAINING_SEED, FORMAL_EXPERIMENT_DEFAULTS
from llm_episode_training import WINDOWS_PORTABLE_PATH_LIMIT
from scenario_manifest import ScenarioManifest


SEARCH_RUN_SCHEMA_VERSION = "uav-hrl-llm-potential-search-run-v1"
SEARCH_PROMPT_VERSION = "uav-hrl-llm-potential-search-prompt-v1"
LEGACY_PROMPT_VERSION = "uav-hrl-llm-potential-search-prompt-v0"
PROMPT_MIGRATION_ID = "compact-training-feedback-v2"
SEARCH_METHOD_ID = "td3_dinkelbach_llm_potential"
DEFAULT_OUTPUT_ROOT = Path("results") / "llm_potential_searches"
DEFAULT_RUNTIME_ROOT = Path.home() / "uav_hrl_rt"
DEFAULT_ROUNDS = 5
DEFAULT_REPAIRS_PER_CANDIDATE = 5
DEFAULT_TRAIN_EPISODES = 400
DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_EVALUATION_ROI_COUNT = 8
DEFAULT_EPISODE_SECONDS = 60
DEFAULT_VALIDATION_EPISODES = 4
ROUND_DIRECTIONS = ("add", "remove", "reweight", "redesign")

ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = ROOT / "prompts"
COMMON_TEMPLATE = PROMPT_ROOT / "llm_potential_search_common.txt"
INITIAL_TEMPLATE = PROMPT_ROOT / "llm_potential_search_initial.txt"
FEEDBACK_TEMPLATE = PROMPT_ROOT / "llm_potential_search_feedback.txt"
DIRECTION_TEMPLATES = {
    "add": PROMPT_ROOT / "llm_potential_search_direction_add.txt",
    "remove": PROMPT_ROOT / "llm_potential_search_direction_remove.txt",
    "reweight": PROMPT_ROOT / "llm_potential_search_direction_reweight.txt",
    "redesign": PROMPT_ROOT / "llm_potential_search_direction_redesign.txt",
}
VALIDATION_REPAIR_TEMPLATE = PROMPT_ROOT / "llm_potential_search_validation_repair.txt"
DUPLICATE_REPAIR_TEMPLATE = PROMPT_ROOT / "llm_potential_search_duplicate_repair.txt"
_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


class PotentialSearchError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _replace_template(template: str, replacements: dict[str, Any]) -> str:
    names = list(dict.fromkeys(_PLACEHOLDER.findall(template)))
    missing = [name for name in names if name not in replacements]
    if missing:
        raise PotentialSearchError(
            "training-search prompt is missing replacements for: "
            + ", ".join(missing)
        )
    return _PLACEHOLDER.sub(
        lambda match: str(replacements[match.group(1)]), template
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(message: str) -> None:
    print(f"[llm-potential-search] {message}", flush=True)


def _candidate_id(round_number: int, slot_index: int) -> str:
    return f"r{int(round_number):02d}_c{int(slot_index):02d}"


def _short_runtime_output_root(
    runtime_root: str | Path,
    search_output: str | Path,
    *,
    kind: str,
    round_number: int,
    slot_index: int,
) -> Path:
    """Use a stable shallow root with checkpoint/evaluation path headroom."""

    leaves = {
        "training": "t",
        "evaluation": "e",
        "artifact": "a",
        "model": "m",
        "candidate": "c",
    }
    if kind not in leaves:
        raise ValueError(f"unsupported runtime output kind: {kind}")
    digest = hashlib.sha256(
        (
            f"{Path(search_output).resolve()}|{kind}|round={int(round_number)}|"
            f"slot={int(slot_index)}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (Path(runtime_root).resolve() / leaves[kind] / digest).resolve()


def _direction(round_number: int, slot_index: int) -> str:
    return "initial" if int(round_number) == 1 else ROUND_DIRECTIONS[slot_index - 1]


def _substitute_candidate_id(candidate: dict[str, Any], expected: str) -> dict[str, Any]:
    required = {"candidate_id", "design_summary", "components", "code"}
    if not isinstance(candidate, dict) or set(candidate) != required:
        raise CandidateError(
            f"candidate fields must be exactly {sorted(required)}"
        )
    if candidate.get("candidate_id") != expected:
        raise CandidateError(
            f"candidate_id must be {expected!r}, received {candidate.get('candidate_id')!r}"
        )
    if not isinstance(candidate.get("design_summary"), str) or not candidate[
        "design_summary"
    ].strip():
        raise CandidateError("design_summary must be a non-empty string")
    return copy.deepcopy(candidate)


def parse_single_candidate(content: str, *, expected_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, metadata = parse_candidate_json_envelope(content)
    return _substitute_candidate_id(value, expected_id), metadata


def bounded_validation_dataset(
    sources: Iterable[str | Path], *, episode_limit: int = DEFAULT_VALIDATION_EPISODES
) -> EpisodeDataset:
    """Load a small executable-validation probe, never a performance screen."""

    complete = load_complete_episode_dataset(sources)
    count = min(int(episode_limit), len(complete.episodes))
    if count <= 0:
        raise PotentialSearchError("validation sources contain no complete episodes")
    selected = complete.episodes[:count]
    source_rows = np.concatenate([item.rows for item in selected])
    arrays = {name: np.asarray(value[source_rows]).copy() for name, value in complete.arrays.items()}
    records = []
    offset = 0
    for index, item in enumerate(selected):
        length = int(item.rows.size)
        records.append(
            EpisodeRecord(
                index=index,
                source_index=item.source_index,
                source_id=item.source_id,
                episode_id=item.episode_id,
                scenario_id=item.scenario_id,
                rows=np.arange(offset, offset + length, dtype=np.int64),
            )
        )
        offset += length
    provenance = {
        **copy.deepcopy(complete.provenance),
        "schema_version": "uav-hrl-training-search-validation-probe-v1",
        "episode_count": count,
        "transition_count": int(offset),
        "source_episode_count": len(complete.episodes),
        "purpose": "executable_shape_finite_range_interface_validation_only",
        "performance_screening_performed": False,
        "lipschitz_evaluation_performed": False,
        "episode_ordering_evaluation_performed": False,
    }
    return EpisodeDataset(arrays, tuple(records), complete.fixed_metadata, provenance)


def _find_symbol(tree: ast.AST, qualified_name: str) -> ast.AST:
    names = qualified_name.split(".")
    nodes = list(getattr(tree, "body", ()))
    current = None
    for name in names:
        current = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ),
            None,
        )
        if current is None:
            raise PotentialSearchError(f"source symbol not found: {qualified_name}")
        nodes = list(getattr(current, "body", ()))
    return current


def _source_symbol(relative_path: str, qualified_name: str) -> dict[str, Any]:
    path = ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    node = _find_symbol(ast.parse(source), qualified_name)
    lines = source.splitlines()
    start = int(node.lineno)
    end = int(node.end_lineno)
    text = "\n".join(lines[start - 1 : end])
    return {
        "source_file": relative_path,
        "symbol": qualified_name,
        "line_start": start,
        "line_end": end,
        "sha256": _sha256_text(text),
        "code": text,
    }


def _source_fragment(
    relative_path: str, *, label: str, start_marker: str, end_marker: str
) -> dict[str, Any]:
    path = ROOT / relative_path
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if start_marker in line), None)
    if start is None:
        raise PotentialSearchError(f"source marker not found: {relative_path}: {start_marker}")
    end = next(
        (i for i in range(start, len(lines)) if end_marker in lines[i]), None
    )
    if end is None:
        raise PotentialSearchError(f"source marker not found: {relative_path}: {end_marker}")
    text = "\n".join(lines[start : end + 1])
    return {
        "source_file": relative_path,
        "symbol": label,
        "line_start": start + 1,
        "line_end": end + 1,
        "sha256": _sha256_text(text),
        "code": text,
    }


def build_environment_source_bundle() -> dict[str, Any]:
    """Extract compact, versioned excerpts from the current implementation."""

    categories = {
        "observation": [
            _source_symbol("centralized_movement.py", "get_global_movement_state"),
            _source_symbol("centralized_movement.py", "movement_mask_from_state"),
            _source_symbol("replay_auxiliary.py", "capture_replay_snapshot"),
        ],
        "movement_and_energy": [
            _source_symbol("centralized_movement.py", "project_joint_action"),
            _source_symbol("centralized_movement.py", "decode_joint_velocity_commands"),
            _source_symbol("centralized_movement.py", "build_velocity_substep_proposals"),
            _source_symbol("centralized_movement.py", "apply_joint_movement_proposals"),
            _source_symbol("Energy_model.py", "EnergyConsumptionModel.propulsion_power"),
        ],
        "visual_sensing": [
            _source_symbol("visual_sensing.py", "vs_geometry"),
            _source_symbol("Packet_scheduler_v1.py", "fov_physical_packet_size_bits"),
            _source_symbol("Packet_scheduler_v1.py", "PacketEngine.inject_packets"),
        ],
        "communication_and_service": [
            _source_symbol("Packet_scheduler_v1.py", "PacketEngine.active_s2u_links"),
            _source_symbol("Packet_scheduler_v1.py", "PacketEngine._block_service_profile"),
            _source_fragment(
                "Packet_scheduler_v1.py",
                label="PacketEngine.serve_active_links timely GS-delivery branch",
                start_marker="timely_delivery = False",
                end_marker='reason="delivered"',
            ),
        ],
        "delivery_and_base_reward": [
            _source_symbol("HRL_task_aware.py", "_interval_reward"),
            _source_fragment(
                "HRL_task_aware.py",
                label="train interval base+extra reward application",
                start_marker="interval_reward = _interval_reward(",
                end_marker="episode_final_movement_reward += combined_movement_reward",
            ),
        ],
    }
    rendered = {}
    all_items = []
    for category, items in categories.items():
        all_items.extend(items)
        rendered[category] = "\n\n".join(
            "# Source: {source_file}:{line_start}-{line_end}; symbol: {symbol}; sha256: {sha256}\n{code}".format(
                **item
            )
            for item in items
        )
    call_relationships = [
        "HRL_task_aware.train -> get_global_movement_state -> movement_mask_from_state -> project_joint_action -> decode_joint_velocity_commands -> build_velocity_substep_proposals -> apply_joint_movement_proposals.",
        "HRL_task_aware.train captures capture_replay_snapshot before the movement action; PacketEngine.inject_packets and service execute after movement/link updates in each 0.25 s slot.",
        "PacketEngine.inject_packets -> visual_sensing.vs_geometry for VS capture; PacketEngine.serve_active_links classifies final GS completion and timely useful bits.",
        "HRL_task_aware.train computes current and next potential components from their own snapshots, applies the host-owned discounted difference once, and keeps the TD3 state unchanged.",
    ]
    identity = [
        {
            key: value
            for key, value in item.items()
            if key != "code"
        }
        for item in all_items
    ]
    content_identity = {
        "call_relationships": call_relationships,
        "excerpts": {
            category: [
                {
                    "source_file": item["source_file"],
                    "symbol": item["symbol"],
                    "code_sha256": _sha256_text(item["code"]),
                }
                for item in items
            ]
            for category, items in categories.items()
        },
    }
    return {
        "schema_version": "uav-hrl-llm-potential-search-source-context-v1",
        "git_sha": _git_sha(),
        "call_relationships": call_relationships,
        "excerpts": categories,
        "rendered": rendered,
        "content_sha256": _sha256_json(content_identity),
        "bundle_sha256": _sha256_json(
            {"git_sha": _git_sha(), "call_relationships": call_relationships, "identity": identity}
        ),
    }


def _source_bundle_content_sha256(bundle: dict[str, Any]) -> str:
    """Hash executable excerpt content while ignoring Git SHA and source line drift."""

    excerpts = bundle.get("excerpts") or {}
    identity = {
        "call_relationships": list(bundle.get("call_relationships") or []),
        "excerpts": {
            category: [
                {
                    "source_file": item.get("source_file"),
                    "symbol": item.get("symbol"),
                    "code_sha256": _sha256_text(str(item.get("code", ""))),
                }
                for item in items
            ]
            for category, items in excerpts.items()
        },
    }
    return _sha256_json(identity)


def _minimal_example(candidate_id: str) -> str:
    return json.dumps(
        {
            "candidate_id": candidate_id,
            "design_summary": "Format-only example; not a design recommendation.",
            "components": [
                {
                    "name": "format_only_zero",
                    "description": "Constant zero normalized to [0,1]; format test only.",
                    "weight": 0.0,
                }
            ],
            "code": (
                "def compute_potential_components(obs, constants):\n"
                "    return np.asarray([0.0], dtype=np.float32)"
            ),
        },
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def render_common_prompt(
    *,
    candidate_id: str,
    dataset: EpisodeDataset,
    constants: dict[str, Any],
    beta: float,
    source_bundle: dict[str, Any],
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
    evaluation_roi_count: int = DEFAULT_EVALUATION_ROI_COUNT,
) -> str:
    relationships = "\n".join(
        f"- {item}" for item in source_bundle["call_relationships"]
    )
    rendered = source_bundle["rendered"]
    return _replace_template(
        COMMON_TEMPLATE.read_text(encoding="utf-8"),
        {
            "MOVEMENT_AND_ENERGY_SPEC": movement_and_energy_spec(constants),
            "MOVEMENT_AND_ENERGY_CODE": relationships + "\n\n" + rendered["movement_and_energy"],
            "VISUAL_SENSING_SPEC": visual_sensing_spec(constants),
            "VISUAL_SENSING_CODE": rendered["visual_sensing"],
            "COMMUNICATION_SPEC": communication_spec(constants),
            "COMMUNICATION_AND_SERVICE_CODE": rendered["communication_and_service"],
            "INPUT_FIELD_TABLE": render_environment_interface(
                dataset.fixed_metadata,
                constants,
                include_system_semantics=False,
            ),
            "OBSERVATION_CODE": rendered["observation"],
            "BASE_REWARD_SPEC": (
                "B_timely_Mbit is interval timely useful delivery in Mbit. The three "
                "penalties are the actual post-action C9, C10, and assigned-COM range "
                "means with unit coefficients. Lambda is the per-episode Dinkelbach "
                "value actually used by training."
            ),
            "DELIVERY_AND_BASE_REWARD_CODE": rendered["delivery_and_base_reward"],
            "BETA": format(float(beta), ".17g"),
            "SUPPORTED_OPERATIONS": SUPPORTED_OPERATIONS,
            "CANDIDATE_ID": candidate_id,
            "MINIMAL_EXECUTABLE_EXAMPLE": _minimal_example(candidate_id),
            "TRAIN_EPISODES": int(train_episodes),
            "EVALUATION_EPISODES": int(evaluation_episodes),
            "EVALUATION_ROI_COUNT": int(evaluation_roi_count),
            "GAMMA": format(
                float(FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["gamma"]),
                ".17g",
            ),
            "TERMINATION_AND_BOOTSTRAP_SPEC": (
                "The current implementation treats the 60-second horizon as a true "
                "terminal transition: TD3 stores not_done=0 and the host uses zero "
                "effective next potential. A future bootstrap-preserving truncation "
                "must retain not_done=1 and use its actual pre-reset next observation."
            ),
            "TRAINING_AND_EVALUATION_SETTINGS": (
                f"Train each candidate independently from scratch for {int(train_episodes)} "
                f"episodes with the shared training seed and scenario manifest. Evaluate "
                f"checkpoint episode {int(train_episodes)} for {int(evaluation_episodes)} "
                f"episodes on the exact fixed-{int(evaluation_roi_count)}-RoI baseline "
                "manifest, with learning and exploration disabled."
            ),
            "EVALUATION_AREA_SPEC": "1000 × 1000 m",
            "EPISODE_SECONDS": DEFAULT_EPISODE_SECONDS,
        },
    )


def _render_stage(template: Path, replacements: dict[str, Any]) -> str:
    return _replace_template(template.read_text(encoding="utf-8"), replacements)


def _short_design(record: dict[str, Any]) -> dict[str, Any]:
    submission = record.get("submission") or {}
    features = submission.get("components") or []
    return {
        "candidate_id": submission.get("candidate_id", record.get("candidate_id")),
        "design_summary": str(submission.get("design_summary", ""))[:600],
        "components": [
            {
                "name": item.get("name"),
                "weight": item.get("weight"),
                "description": str(item.get("description", ""))[:240],
            }
            for item in features
        ],
        "code_formula_excerpt": str(submission.get("code", ""))[:2400],
        "status": record.get("status"),
    }


def _best_candidate_feedback(best: dict[str, Any]) -> dict[str, Any]:
    """Serialize one complete design without duplicating bulky run records."""

    return {
        "candidate_id": best["candidate_id"],
        "round": best["round"],
        "complete_candidate": copy.deepcopy(best["submission"]),
        "evaluation": _evaluation_feedback(best.get("evaluation")),
        "compact_measured_training_summary": _compact_historical_training(best),
    }


def _evaluation_feedback(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    keys = (
        "status",
        "checkpoint_episode",
        "episode_count",
        "mean_episode_energy_efficiency_mbit_per_j",
        "std_episode_energy_efficiency_mbit_per_j",
        "baseline_mean_episode_energy_efficiency_mbit_per_j",
        "baseline_absolute_gap_mbit_per_j",
        "baseline_percent_gap",
        "candidate_metrics",
    )
    return {key: result.get(key) for key in keys}


def _load_ordered_training_rows(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    run_directory = (record.get("training") or {}).get("run_directory")
    if not run_directory:
        return None
    path = Path(run_directory) / "potential_training_episode_metrics.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    episodes = [int(row.get("episode", -1)) for row in rows]
    if episodes != list(range(1, len(rows) + 1)):
        return None
    return rows


def _decode_exploration(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _exploration_parameter_summary(
    values: list[dict[str, Any] | None], *, ordered: bool
) -> dict[str, Any]:
    missing_marker = {"missing": True}
    keys = sorted(
        {
            key
            for value in values
            if isinstance(value, dict)
            for key in value
        }
    )
    if not keys:
        return {
            "status": "missing",
            "reason": "no usable exploration records",
            "episode_order_available": bool(ordered),
        }
    parameters: dict[str, Any] = {}
    for key in keys:
        sequence = [
            value[key] if isinstance(value, dict) and key in value else missing_marker
            for value in values
        ]
        encoded = [_canonical_json(item) for item in sequence]
        if len(set(encoded)) == 1:
            parameters[key] = {"value": sequence[0]}
            continue
        item: dict[str, Any] = {
            "start": sequence[0] if ordered else {"unavailable": "episode order not saved"},
            "end": sequence[-1] if ordered else {"unavailable": "episode order not saved"},
            "missing_count": sum(value == missing_marker for value in sequence),
        }
        numeric = [
            float(value)
            for value in sequence
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
        ]
        if numeric:
            if ordered:
                monotonic_up = all(
                    left <= right for left, right in zip(numeric, numeric[1:])
                )
                monotonic_down = all(
                    left >= right for left, right in zip(numeric, numeric[1:])
                )
                item["monotonic"] = (
                    "nondecreasing"
                    if monotonic_up
                    else "nonincreasing"
                    if monotonic_down
                    else "no"
                )
            else:
                item["trend"] = "unknown_without_episode_order"
                monotonic_up = monotonic_down = False
            if (
                not ordered
                or not (monotonic_up or monotonic_down)
                or len(numeric) != len(sequence)
            ):
                item["min"] = min(numeric)
                item["max"] = max(numeric)
        else:
            distinct = []
            for value, serialized in zip(sequence, encoded):
                if serialized not in {_canonical_json(item) for item in distinct}:
                    distinct.append(value)
            if len(distinct) <= 6:
                item["distinct_values"] = distinct
            else:
                item["distinct_value_count"] = len(distinct)
        parameters[key] = item
    return {
        "status": "summarized",
        "episode_order_available": bool(ordered),
        "parameters": parameters,
    }


def _compact_training_blocks(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = (record.get("training") or {}).get("summaries") or []
    ordered_rows = _load_ordered_training_rows(record)
    compact_blocks = []
    exploration_blocks = []
    retained = (
        "episode_range",
        "episode_count",
        "step_count",
        "energy_efficiency_mbit_per_j",
        "timely_useful_delivery_mbit",
        "movement_energy_j",
        "vs_delay_violation_probability",
        "com_delay_violation_probability",
        "raw_reward_sum",
        "raw_step_reward_absolute_mean",
        "shaping_sum",
        "training_reward_sum",
        "actor_updates",
        "critic_updates",
        "dinkelbach_lambda_range",
        "potential_components",
    )
    for block in summaries:
        episode_range = list(block.get("episode_range") or [])
        values: list[dict[str, Any] | None]
        ordered = False
        if len(episode_range) == 2 and ordered_rows is not None:
            start, end = map(int, episode_range)
            selected = ordered_rows[start - 1 : end]
            if [int(row.get("episode", -1)) for row in selected] == list(
                range(start, end + 1)
            ):
                values = [_decode_exploration(row.get("exploration")) for row in selected]
                ordered = True
            else:
                values = []
        else:
            values = []
        if not values:
            values = [_decode_exploration(item) for item in block.get("exploration") or []]
        compact_blocks.append(
            {key: copy.deepcopy(block.get(key)) for key in retained if key in block}
        )
        exploration_blocks.append(
            {
                "episode_range": episode_range,
                "source": (
                    "ordered potential_training_episode_metrics.jsonl"
                    if ordered
                    else "legacy distinct exploration list; episode order unavailable"
                ),
                "summary": _exploration_parameter_summary(values, ordered=ordered),
            }
        )
    return compact_blocks, exploration_blocks


def _compact_historical_training(record: dict[str, Any]) -> dict[str, Any]:
    """Compact every saved episode into one traceable measured summary."""

    rows = _load_ordered_training_rows(record)
    if not rows:
        blocks, _exploration = _compact_training_blocks(record)
        return {
            "source": "saved_training_blocks",
            "scope": "all saved blocks retained because episode rows were unavailable",
            "blocks": blocks,
        }

    def distribution(field: str) -> dict[str, float]:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
        }

    def violation(prefix: str) -> dict[str, Any]:
        eligible = sum(int(row[f"{prefix}_eligible_packets"]) for row in rows)
        violated = sum(int(row[f"{prefix}_violation_packets"]) for row in rows)
        return {
            "eligible_packets": eligible,
            "violated_packets": violated,
            "value": float(violated / eligible) if eligible else None,
            "missing": eligible == 0,
            "denominator": "pooled eligible end-to-end packets",
        }

    raw_steps = MergeableMoments()
    for row in rows:
        raw_steps.merge(MergeableMoments.from_dict(row["raw_step_reward"]))
    names = list(rows[0].get("component_names") or ())
    return {
        "source": "potential_training_episode_metrics.jsonl",
        "scope": {
            "first_episode": int(rows[0]["episode"]),
            "last_episode": int(rows[-1]["episode"]),
            "episode_count": len(rows),
            "step_count": sum(int(row["transition_count"]) for row in rows),
        },
        "energy_efficiency_mbit_per_j": distribution(
            "energy_efficiency_mbit_per_j"
        ),
        "timely_useful_delivery_mbit": distribution("timely_mbits"),
        "movement_energy_j": distribution("movement_energy_j"),
        "vs_delay_violation_probability": violation("fov"),
        "com_delay_violation_probability": violation("com"),
        "raw_reward_sum": distribution("raw_reward_sum"),
        "raw_step_reward": raw_steps.to_dict(),
        "shaping_sum": distribution("shaping_sum"),
        "training_reward_sum": distribution("training_reward_sum"),
        "actor_updates": sum(int(row["actor_updates"]) for row in rows),
        "critic_updates": sum(int(row["critic_updates"]) for row in rows),
        "dinkelbach_lambda_range": [
            min(float(row["dinkelbach_lambda_used"]) for row in rows),
            max(float(row["dinkelbach_lambda_used"]) for row in rows),
        ],
        "potential_components": summarize_component_records(
            rows, component_names=names
        ),
        "reduction": (
            "all episode rows merged from count, sum, sum-of-squares, and "
            "sum-of-absolute-values; no standard deviations were averaged"
        ),
    }


def _compact_fairness(record: dict[str, Any]) -> dict[str, Any]:
    fairness = record.get("training_fairness") or {}
    schedule = fairness.get("exploration_schedule_configuration")
    return {
        "status": fairness.get("status", "missing"),
        "training_seed": fairness.get("seed"),
        "training_episodes": fairness.get("episodes"),
        "fresh_initialization": fairness.get("fresh_initialization"),
        "training_scenario_prefix_matches_baseline": fairness.get(
            "training_scenario_prefix_matches_baseline"
        ),
        "episode_400_is_stop_not_decay_horizon": fairness.get(
            "episode_400_is_stop_not_decay_horizon"
        ),
        "exploration_schedule_configuration": copy.deepcopy(schedule),
    }


def _latest_round_feedback(
    round_record: dict[str, Any], *, best_candidate_id: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    training_candidates = []
    exploration_by_candidate: dict[str, list[dict[str, Any]]] = {}
    fairness_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in round_record["candidate_order"]:
        record = round_record["candidates"][candidate_id]
        if candidate_id == best_candidate_id:
            design = {
                "candidate_id": candidate_id,
                "design_reference": "shared historical-best candidate above",
            }
        else:
            design = _short_design(record)
            design.pop("code_formula_excerpt", None)
        results.append(
            {
                **design,
                "requested_direction": record.get("requested_direction"),
                "actual_modification_summary": record.get("actual_modification_summary"),
                "evaluation": _evaluation_feedback(record.get("evaluation")),
                "failure": record.get("failure"),
            }
        )
        blocks, exploration = _compact_training_blocks(record)
        training_candidates.append(
            {
                "candidate_id": candidate_id,
                "status": (record.get("training") or {}).get("status"),
                "blocks": blocks,
                "failure": record.get("failure"),
            }
        )
        exploration_by_candidate[candidate_id] = exploration
        fairness_by_candidate[candidate_id] = _compact_fairness(record)

    exploration_values = list(exploration_by_candidate.values())
    shared_exploration = (
        exploration_values[0]
        if exploration_values
        and all(value == exploration_values[0] for value in exploration_values[1:])
        else None
    )
    fairness_values = list(fairness_by_candidate.values())
    shared_fairness = (
        fairness_values[0]
        if fairness_values
        and all(value == fairness_values[0] for value in fairness_values[1:])
        else None
    )
    if shared_exploration is None:
        for item in training_candidates:
            item["exploration_by_block"] = exploration_by_candidate[item["candidate_id"]]
    if shared_fairness is None:
        for item in training_candidates:
            item["fairness"] = fairness_by_candidate[item["candidate_id"]]
    training = {
        "reward_history_semantics": (
            "raw reward uses the episode lambda; weighted potential and shaping "
            "components are separate mergeable measurements"
        ),
        "shared_fairness_and_training_contract": shared_fairness,
        "shared_exploration_by_block": shared_exploration,
        "candidates": training_candidates,
    }
    return results, training


def _historical_summaries(state: dict[str, Any], *, exclude_round: int) -> list[dict[str, Any]]:
    best_id = (state.get("best_candidate") or {}).get("candidate_id")
    summaries = []
    for round_record in state.get("rounds", []):
        if int(round_record["round"]) >= int(exclude_round):
            continue
        for candidate_id in round_record["candidate_order"]:
            if candidate_id == best_id:
                continue
            record = round_record["candidates"][candidate_id]
            evaluation = record.get("evaluation") or {}
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "round": round_record["round"],
                    "design": _short_design(record),
                    "mean_episode_ee_mbit_per_j": evaluation.get(
                        "mean_episode_energy_efficiency_mbit_per_j"
                    ),
                    "baseline_percent_gap": evaluation.get("baseline_percent_gap"),
                    "compact_measured_training_summary": (
                        _compact_historical_training(record)
                    ),
                    "status": record.get("status"),
                    "failure": record.get("failure"),
                }
            )
    return summaries


def _generation_stage_prompt(
    *, state: dict[str, Any], current: dict[str, Any], candidate_id: str
) -> str:
    """Render the non-COMMON part for generation and bounded repair prompts."""

    return _generation_stage_prompt_parts(
        state=state, current=current, candidate_id=candidate_id
    )[0]


def _generation_stage_prompt_parts(
    *, state: dict[str, Any], current: dict[str, Any], candidate_id: str
) -> tuple[str, dict[str, int]]:
    """Render one stage and report its semantically meaningful section sizes."""

    if int(current["round"]) == 1:
        summaries = [
            {
                **_short_design(current["candidates"][item]),
                "training_status": "not_trained",
                "note": "summary is only for design diversity",
            }
            for item in current["candidate_order"]
            if current["candidates"][item].get("status") == "validated"
        ]
        summary_text = json.dumps(summaries, indent=2, ensure_ascii=False)
        stage = _render_stage(
            INITIAL_TEMPLATE,
            {
                "CANDIDATE_ID": candidate_id,
                "CURRENT_ROUND_CANDIDATE_SUMMARIES": summary_text,
            },
        ).strip()
        return stage, {
            "current_round_untrained_design_summaries": len(summary_text),
            "generation_direction": 0,
            "stage_scaffolding": len(stage) - len(summary_text),
        }

    latest = state["rounds"][-1]
    best_id = current["reference_candidate"]["candidate_id"]
    latest_results, latest_training = _latest_round_feedback(
        latest, best_candidate_id=best_id
    )
    best = _best_candidate_feedback(current["reference_candidate"])
    best_text = json.dumps(best, indent=2, ensure_ascii=False)
    baseline_text = json.dumps(
        {
            "checkpoint_episode": state["baseline_preflight"]["checkpoint_episode"],
            "metrics": state["baseline_preflight"]["metrics"],
        },
        indent=2,
        ensure_ascii=False,
    )
    latest_results_text = json.dumps(latest_results, indent=2, ensure_ascii=False)
    latest_training_text = json.dumps(latest_training, indent=2, ensure_ascii=False)
    historical_text = json.dumps(
        _historical_summaries(state, exclude_round=int(latest["round"])),
        indent=2,
        ensure_ascii=False,
    )
    current_round_text = json.dumps(
        [
            {
                **_short_design(current["candidates"][item]),
                "training_status": "not_trained",
            }
            for item in current["candidate_order"]
            if current["candidates"][item].get("status") == "validated"
        ],
        indent=2,
        ensure_ascii=False,
    )
    settings = state.get("settings") or {}
    train_episodes = int(settings.get("train_episodes", DEFAULT_TRAIN_EPISODES))
    block_size = int(settings.get("summary_block_size", 50))
    block_definition = ", ".join(
        f"{start}–{min(start + block_size - 1, train_episodes)}"
        for start in range(1, train_episodes + 1, block_size)
    )
    feedback = _render_stage(
        FEEDBACK_TEMPLATE,
        {
            "CANDIDATE_ID": candidate_id,
            "BEST_CANDIDATE": best_text,
            "BASELINE_EVALUATION_RESULTS": baseline_text,
            "LATEST_ROUND_CANDIDATE_RESULTS": latest_results_text,
            "LATEST_ROUND_TRAINING_SUMMARIES": latest_training_text,
            "TRAINING_BLOCK_DEFINITION": block_definition,
            "HISTORICAL_CANDIDATE_SUMMARIES": historical_text,
            "CURRENT_ROUND_CANDIDATE_SUMMARIES": current_round_text,
        },
    )
    direction = current["candidates"][candidate_id]["requested_direction"]
    direction_text = DIRECTION_TEMPLATES[direction].read_text(encoding="utf-8").strip()
    stage = (
        feedback.strip()
        + "\n\n"
        + direction_text
    )
    dynamic_total = sum(
        map(
            len,
            (
                best_text,
                baseline_text,
                latest_results_text,
                latest_training_text,
                historical_text,
                current_round_text,
                direction_text,
            ),
        )
    )
    return stage, {
        "historical_best_candidate": len(best_text),
        "baseline_evaluation": len(baseline_text),
        "latest_round_candidate_evaluations": len(latest_results_text),
        "training_block_summaries": len(latest_training_text),
        "historical_candidate_summaries": len(historical_text),
        "generation_direction": len(direction_text),
        "stage_scaffolding": len(stage) - dynamic_total,
    }


def _base_generation_prompt(
    *,
    state: dict[str, Any],
    current: dict[str, Any],
    candidate_id: str,
    common: str,
    stage: str | None = None,
) -> str:
    if stage is None:
        stage = _generation_stage_prompt(
            state=state, current=current, candidate_id=candidate_id
        )
    return common.rstrip() + "\n\n" + stage + "\n"


def _repair_prompt(
    *, common: str, record: dict[str, Any], duplicate: bool
) -> str:
    generation_stage = record.get(
        "active_generation_stage_request", record["generation_stage_request"]
    )
    if duplicate:
        stage = _render_stage(
            DUPLICATE_REPAIR_TEMPLATE,
            {
                "CANDIDATE_ID": record["candidate_id"],
                "GENERATION_REQUEST": generation_stage,
                "DUPLICATE_CANDIDATE": json.dumps(
                    record.get("last_submission"), indent=2, ensure_ascii=False
                ),
                "DUPLICATE_DIAGNOSTICS": json.dumps(
                    record.get("last_validation"), indent=2, ensure_ascii=False
                ),
            },
        )
    else:
        failed_response = _load_failed_response_material(record)
        stage = _render_stage(
            VALIDATION_REPAIR_TEMPLATE,
            {
                "CANDIDATE_ID": record["candidate_id"],
                "GENERATION_REQUEST": generation_stage,
                "FAILED_RESPONSE_MATERIAL": json.dumps(
                    failed_response, indent=2, ensure_ascii=False
                ),
                "VALIDATION_FEEDBACK": json.dumps(
                    record.get("last_validation"), indent=2, ensure_ascii=False
                ),
                "UNRESOLVED_CHECKS": json.dumps(
                    (record.get("last_validation") or {}).get(
                        "unresolved_checks", []
                    ),
                    indent=2,
                    ensure_ascii=False,
                ),
            },
        )
    return common.rstrip() + "\n\n" + stage.strip() + "\n"


def _load_failed_response_material(record: dict[str, Any]) -> dict[str, Any]:
    """Reload the latest failed response from its persisted candidate version."""

    versions = record.get("versions") or []
    if not versions:
        raise PotentialSearchError("repair prompt requires a saved failed version")
    latest = versions[-1]
    directory = Path(latest["directory"])
    response_kind = latest.get("response_kind")
    if response_kind is None:
        raw_path = directory / "raw_response.txt"
        try:
            legacy_raw = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PotentialSearchError(
                f"saved raw model response is missing: {raw_path}"
            ) from exc
        try:
            legacy_parsed, _metadata = parse_candidate_json_envelope(legacy_raw)
        except CandidateError:
            return {
                "response_kind": "raw_unparsed_model_response",
                "raw_response": legacy_raw,
            }
        return {
            "response_kind": "parsed_candidate",
            "parsed_candidate": legacy_parsed,
        }
    if response_kind == "parsed_json":
        path = directory / "submission.json"
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PotentialSearchError(
                f"saved parsed candidate is missing or invalid: {path}"
            ) from exc
        return {
            "response_kind": "parsed_candidate",
            "parsed_candidate": parsed,
        }
    if response_kind != "raw_unparsed":
        raise PotentialSearchError(f"unsupported saved response kind: {response_kind!r}")
    path = directory / "raw_response.txt"
    try:
        raw_response = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PotentialSearchError(
            f"saved raw model response is missing: {path}"
        ) from exc
    return {
        "response_kind": "raw_unparsed_model_response",
        "raw_response": raw_response,
    }


def _exception_details(exc: BaseException) -> dict[str, Any]:
    details: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        location = {
            name: getattr(cause, name)
            for name in ("lineno", "colno", "pos")
            if getattr(cause, name, None) is not None
        }
        if location:
            details["location"] = location
            details["location_error_type"] = type(cause).__name__
            break
        cause = cause.__cause__ or cause.__context__
    return details


def _code_ast(candidate: dict[str, Any]) -> str:
    return ast.dump(
        ast.parse(str(candidate["code"]), mode="exec"),
        annotate_fields=True,
        include_attributes=False,
    )


def _returned_feature_expressions(candidate: dict[str, Any]) -> list[str] | None:
    """Return exact direct return expressions when statically unambiguous."""

    tree = ast.parse(str(candidate["code"]), mode="exec")
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "compute_potential_components"
        ),
        None,
    )
    if function is None:
        return None
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        return None
    value = returns[0].value
    if isinstance(value, (ast.List, ast.Tuple)):
        elements = value.elts
    elif (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in {"array", "asarray"}
        and value.args
        and isinstance(value.args[0], (ast.List, ast.Tuple))
    ):
        elements = value.args[0].elts
    else:
        return None
    return [
        ast.dump(item, annotate_fields=True, include_attributes=False)
        for item in elements
    ]


def validate_direction(
    direction: str,
    candidate: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if direction == "initial":
        return {
            "status": "passed",
            "requested_direction": direction,
            "reliable_checks": ["not_applicable_first_round"],
        }
    if reference is None:
        raise PotentialSearchError("revision direction lacks its fixed reference candidate")
    old_weights = [float(item["weight"]) for item in reference["components"]]
    new_weights = [float(item["weight"]) for item in candidate["components"]]
    old_expr = _returned_feature_expressions(reference)
    new_expr = _returned_feature_expressions(candidate)
    issues = []
    checks = []
    exception = None
    if direction == "add":
        checks.append("feature_count_increased")
        if len(new_weights) <= len(old_weights):
            issues.append("add direction must increase feature count")
        if old_expr is not None and new_expr is not None:
            checks.append("retained_formula_and_weight_ordered_subsequence")
            cursor = 0
            for expression, weight in zip(new_expr, new_weights):
                if cursor < len(old_expr) and expression == old_expr[cursor] and weight == old_weights[cursor]:
                    cursor += 1
            if cursor != len(old_expr):
                issues.append("reliably extracted original formulas and weights were not all retained")
    elif direction == "remove":
        if len(old_weights) == 1:
            exception = "single_feature_reference_uses_simpler_replacement"
            checks.append("single_feature_replacement_is_substantive")
            if candidate_fingerprint(candidate) == candidate_fingerprint(reference):
                issues.append("single-feature replacement did not change computation or weight")
        else:
            checks.append("feature_count_decreased")
            if len(new_weights) >= len(old_weights):
                issues.append("remove direction must decrease feature count")
            if old_expr is not None and new_expr is not None:
                checks.append("remaining_formula_and_weight_ordered_subsequence")
                cursor = 0
                for expression, weight in zip(old_expr, old_weights):
                    if cursor < len(new_expr) and expression == new_expr[cursor] and weight == new_weights[cursor]:
                        cursor += 1
                if cursor != len(new_expr):
                    issues.append("reliably extracted remaining formulas or weights changed")
    elif direction == "reweight":
        checks.extend(("code_ast_unchanged", "feature_count_unchanged", "weight_changed"))
        if _code_ast(candidate) != _code_ast(reference):
            issues.append("reweight direction changed feature computation")
        if len(new_weights) != len(old_weights):
            issues.append("reweight direction changed feature count")
        if new_weights == old_weights:
            issues.append("reweight direction did not change any weight")
    elif direction == "redesign":
        checks.append("substantive_fingerprint_changed")
        if candidate_fingerprint(candidate) == candidate_fingerprint(reference):
            issues.append("redesign changed only non-executable metadata")
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return {
        "status": "failed" if issues else "passed",
        "requested_direction": direction,
        "reliable_checks": checks,
        "formula_comparison_available": old_expr is not None and new_expr is not None,
        "single_feature_remove_exception": exception,
        "issues": issues,
    }


def modification_summary(
    direction: str,
    candidate: dict[str, Any],
    reference: dict[str, Any] | None,
    direction_report: dict[str, Any],
) -> dict[str, Any]:
    if reference is None:
        return {
            "requested_direction": direction,
            "component_count": len(candidate["components"]),
            "weights": [float(item["weight"]) for item in candidate["components"]],
            "basis": None,
        }
    old = [float(item["weight"]) for item in reference["components"]]
    new = [float(item["weight"]) for item in candidate["components"]]
    return {
        "requested_direction": direction,
        "basis_candidate_id": reference.get("candidate_name"),
        "feature_count_before": len(old),
        "feature_count_after": len(new),
        "feature_names_before": [
            str(item["name"]) for item in reference["components"]
        ],
        "feature_names_after": [
            str(item["name"]) for item in candidate["components"]
        ],
        "weights_before": old,
        "weights_after": new,
        "normalized_code_ast_changed": _code_ast(candidate) != _code_ast(reference),
        "changed_weight_indices": [
            index
            for index, (left, right) in enumerate(zip(old, new))
            if left != right
        ],
        "formula_comparison_available": direction_report.get(
            "formula_comparison_available"
        ),
        "single_feature_remove_exception": direction_report.get(
            "single_feature_remove_exception"
        ),
    }


def _all_validated_records(state: dict[str, Any], current: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for round_record in state.get("rounds", []):
        yield from round_record["candidates"].values()
    yield from current["candidates"].values()


def _duplicate_diagnostics(
    state: dict[str, Any], current: dict[str, Any], candidate_id: str, fingerprint: str
) -> dict[str, Any] | None:
    for record in _all_validated_records(state, current):
        if record.get("candidate_id") == candidate_id:
            continue
        if record.get("fingerprint") == fingerprint:
            return {
                "status": "failed",
                "stage": "duplicate",
                "matching_candidate_id": record.get("candidate_id"),
                "fingerprint": fingerprint,
                "semantics": (
                    "normalized Python AST plus feature order/count and fixed reward weights; "
                    "display names, descriptions, comments, and formatting are ignored; this "
                    "does not claim general mathematical equivalence"
                ),
            }
    return None


def _save_candidate_version(
    directory: Path,
    *,
    submission: Any | None,
    normalized: dict[str, Any] | None,
    validation: dict[str, Any],
    raw_content: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "raw_response.txt").write_text(raw_content, encoding="utf-8")
    _write_json(directory / "submission.json", submission)
    _write_json(directory / "validation_report.json", validation)
    if normalized is not None:
        _write_json(directory / "candidate.json", normalized)
        (directory / "candidate.py").write_text(normalized["code"], encoding="utf-8")


def _training_fairness_provenance(
    *,
    training: dict[str, Any],
    baseline_run: str | Path,
    expected_seed: int,
    expected_episodes: int,
) -> dict[str, Any]:
    run = Path(training["run_directory"]).resolve()
    baseline = Path(baseline_run).resolve()
    resolved = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    baseline_resolved = json.loads(
        (baseline / "resolved_config.json").read_text(encoding="utf-8")
    )
    candidate_manifest = ScenarioManifest.load(run / "scenario_manifest.json")
    baseline_manifest = ScenarioManifest.load(baseline / "scenario_manifest.json")
    if int(resolved.get("seed", -1)) != int(expected_seed):
        raise PotentialSearchError("candidate training seed differs from the search contract")
    if int(resolved.get("episodes", -1)) != int(expected_episodes):
        raise PotentialSearchError(
            f"candidate training horizon differs from episode {int(expected_episodes)}"
        )
    if candidate_manifest.episode_count != int(expected_episodes):
        raise PotentialSearchError("candidate training manifest has the wrong length")
    if candidate_manifest.episodes != baseline_manifest.episodes[: int(expected_episodes)]:
        raise PotentialSearchError(
            "candidate training scenarios do not equal the baseline run's first "
            f"{int(expected_episodes)} scenarios"
        )
    invariant_fields = (
        "assignment_strategy",
        "assignment_rounds",
        "routing_policy",
        "reward_mode",
        "task_observation_mode",
        "task_potential_enabled",
        "movement_hyperparameters",
        "effective_routing_agent_configuration",
        "exploration_schedule_configuration",
        "training_environment_width_m",
        "training_environment_height_m",
        "roi_count",
    )
    mismatches = {
        field: {
            "candidate": resolved.get(field),
            "baseline": baseline_resolved.get(field),
        }
        for field in invariant_fields
        if resolved.get(field) != baseline_resolved.get(field)
    }
    if mismatches:
        raise PotentialSearchError(f"candidate/baseline fairness settings differ: {mismatches}")
    exploration = resolved["exploration_schedule_configuration"]
    if int(exploration["movement_exploration_decay_episodes"]) != 1000 or int(
        exploration["routing_epsilon_decay_episodes"]
    ) != 1000:
        raise PotentialSearchError("episode-400 training compressed the exploration schedule")
    prefix_hash = _sha256_json(candidate_manifest.episodes)
    return {
        "schema_version": "uav-hrl-llm-potential-search-fairness-v1",
        "status": "passed",
        "seed": int(expected_seed),
        "episodes": int(expected_episodes),
        "fresh_initialization": True,
        "candidate_run": str(run),
        "baseline_run": str(baseline),
        "training_scenario_prefix_matches_baseline": True,
        "training_scenario_prefix_sha256": prefix_hash,
        "candidate_training_manifest_hash": candidate_manifest.content_hash,
        "baseline_full_training_manifest_hash": baseline_manifest.content_hash,
        "invariant_fields": list(invariant_fields),
        "exploration_schedule_configuration": exploration,
        "episode_400_is_stop_not_decay_horizon": True,
    }


def _new_round(state: dict[str, Any], round_number: int) -> dict[str, Any]:
    best = state.get("best_candidate")
    reference = (
        {
            key: copy.deepcopy(best[key])
            for key in (
                "round",
                "candidate_id",
                "candidate",
                "submission",
                "evaluation",
                "mean_episode_energy_efficiency_mbit_per_j",
            )
        }
        if round_number > 1 and best is not None
        else None
    )
    if round_number > 1 and reference is None:
        raise PotentialSearchError("later round requires a completed historical best candidate")
    candidate_order = [_candidate_id(round_number, index) for index in range(1, 5)]
    return {
        "round": int(round_number),
        "status": "running",
        "phase": "generate",
        "candidate_order": candidate_order,
        "generation_index": 0,
        "work_index": 0,
        "reference_candidate": reference,
        "reference_candidate_id": (reference or {}).get("candidate_id"),
        "candidates": {
            candidate_id: {
                "candidate_id": candidate_id,
                "requested_direction": _direction(round_number, index),
                "base_candidate_id": (reference or {}).get("candidate_id"),
                "status": "pending_generation",
                "versions": [],
                "corrections_used": 0,
                "transport_failures": [],
            }
            for index, candidate_id in enumerate(candidate_order, start=1)
        },
    }


def _result(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state["status"],
        "output_directory": state["output_directory"],
        "search_round": state["search_round"],
        "model_calls": state["model_calls"],
        "best_candidate": state.get("best_candidate"),
        "stop_reason": state.get("stop_reason"),
    }


def _atomic_temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _validate_atomic_write_paths(
    path: Path, *, portable_path_limit: int
) -> dict[str, Any]:
    target = path.resolve()
    temporary = _atomic_temporary_path(target)
    lengths = {
        "target": len(str(target)),
        "temporary": len(str(temporary)),
    }
    longest_name = max(lengths, key=lengths.get)
    if lengths[longest_name] > int(portable_path_limit):
        longest = target if longest_name == "target" else temporary
        raise PotentialSearchError(
            "atomic output exceeds the portable Windows path limit: "
            f"{lengths[longest_name]} > {int(portable_path_limit)}: {longest}"
        )
    return {
        "portable_path_limit": int(portable_path_limit),
        "target_path": str(target),
        "target_path_length": lengths["target"],
        "temporary_path": str(temporary),
        "temporary_path_length": lengths["temporary"],
    }


def _atomic_write_text(
    path: Path, text: str, *, portable_path_limit: int | None = None
) -> None:
    if portable_path_limit is not None:
        _validate_atomic_write_paths(
            path, portable_path_limit=portable_path_limit
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _archive_cached_prompt(
    directory: Path, *, kind: str, text: str
) -> tuple[Path, str]:
    """Save an immutable, short-path prompt backup and return its full hash."""

    if kind not in {"request", "stage"}:
        raise PotentialSearchError(f"unsupported cached prompt backup kind: {kind}")
    digest = _sha256_text(text)
    stem = "rq" if kind == "request" else "st"
    for collision_index in range(1000):
        suffix = "" if collision_index == 0 else f"-{collision_index}"
        path = directory / f"{stem}-{digest[:16]}{suffix}.txt"
        _validate_atomic_write_paths(
            path, portable_path_limit=WINDOWS_PORTABLE_PATH_LIMIT
        )
        if path.exists():
            if path.read_text(encoding="utf-8") == text:
                return path, digest
            continue
        _atomic_write_text(
            path, text, portable_path_limit=WINDOWS_PORTABLE_PATH_LIMIT
        )
        if path.read_text(encoding="utf-8") != text:
            raise PotentialSearchError(
                f"cached prompt backup verification failed: {path}"
            )
        return path, digest
    raise PotentialSearchError(
        f"cached prompt backup name collisions exhausted for {kind}: {directory}"
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _save_state(output: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _now()
    _atomic_write_json(output / "state.json", state)


def _write_completed_search_artifacts(output: Path, state: dict[str, Any]) -> None:
    best = state.get("best_candidate")
    if not isinstance(best, dict):
        raise PotentialSearchError("completed search has no historical best candidate")
    training_run = Path(best["training"]["run_directory"]).resolve()
    checkpoint_episode = int(best["evaluation"]["checkpoint_episode"])
    full_checkpoint = (
        training_run / "checkpoints" / "full" / f"ep_{checkpoint_episode:04d}"
    )
    if not full_checkpoint.is_dir():
        raise PotentialSearchError(
            "historical best candidate lacks its complete resume checkpoint: "
            f"{full_checkpoint}"
        )
    _atomic_write_json(output / "best_candidate.json", best)
    _atomic_write_json(
        output / "all_results.json",
        {
            "schema_version": "uav-hrl-llm-potential-search-results-v1",
            "method_id": SEARCH_METHOD_ID,
            "round_count": len(state.get("rounds") or ()),
            "rounds": state.get("rounds") or [],
        },
    )
    _atomic_write_json(
        output / "manual_continuation.json",
        {
            "schema_version": "uav-hrl-llm-potential-continuation-v1",
            "candidate_id": best["candidate_id"],
            "training_run_directory": str(training_run),
            "complete_resume_checkpoint": str(full_checkpoint),
            "complete_resume_checkpoint_exists": True,
            "powershell_command_template": (
                "& $Python .\\run_experiment.py resume "
                f"'{training_run}' --target-episodes 1500"
            ),
            "automatic_continuation_started": False,
        },
    )


def _migration_directory(output: Path) -> Path:
    return output / "migrations" / PROMPT_MIGRATION_ID


def _ensure_migration_backup(output: Path) -> Path:
    migration = _migration_directory(output)
    migration.mkdir(parents=True, exist_ok=True)
    backup = migration / "state.before.json"
    if not backup.exists():
        source = output / "state.json"
        _atomic_write_text(backup, source.read_text(encoding="utf-8"))
    source_context = output / "prompt_source_context" / "environment_source_context.json"
    source_backup = migration / "environment_source_context.before.json"
    if source_context.is_file() and not source_backup.exists():
        _atomic_write_text(
            source_backup, source_context.read_text(encoding="utf-8")
        )
    return migration


def _validate_saved_search_contract(settings: dict[str, Any]) -> None:
    expected = {
        "evaluation_area_m": [1000.0, 1000.0],
        "episode_seconds": DEFAULT_EPISODE_SECONDS,
        "candidate_count_per_round": 4,
        "candidate_generation_calls_per_request": 1,
    }
    changed = {
        key: {"saved": settings.get(key), "required": value}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if changed:
        raise PotentialSearchError(
            f"resume training/evaluation contract changed: {changed}"
        )
    for key in ("train_episodes", "evaluation_episodes", "evaluation_roi_count", "summary_block_size"):
        if int(settings.get(key, 0)) <= 0:
            raise PotentialSearchError(f"saved {key} must be positive")


def _prepare_resume_compatibility(
    *,
    output: Path,
    state: dict[str, Any],
    source_bundle: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any] | None:
    """Allow prompt-only revision drift after content-level contract checks."""

    _validate_saved_search_contract(state["settings"])
    saved_constants_path = output / "constants.json"
    if not saved_constants_path.is_file():
        raise PotentialSearchError("resume constants contract record is missing")
    saved_constants = json.loads(saved_constants_path.read_text(encoding="utf-8"))
    if saved_constants != constants:
        raise PotentialSearchError("resume obs/constants contract changed")

    saved_prompt_version = state.get("prompt_version")
    if saved_prompt_version not in {LEGACY_PROMPT_VERSION, SEARCH_PROMPT_VERSION}:
        raise PotentialSearchError(
            f"unsupported saved training-search prompt version: {saved_prompt_version!r}"
        )
    reasons = []
    if saved_prompt_version != SEARCH_PROMPT_VERSION:
        reasons.append("prompt_feedback_format_changed")

    saved_bundle_hash = state.get("environment_source_bundle_sha256")
    if not saved_bundle_hash:
        raise PotentialSearchError("resume source bundle hash is missing")
    current_bundle_hash = source_bundle["bundle_sha256"]
    saved_source_path = (
        output / "prompt_source_context" / "environment_source_context.json"
    )
    saved_source = None
    if saved_bundle_hash not in (None, current_bundle_hash):
        try:
            saved_source = json.loads(saved_source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PotentialSearchError(
                "resume source bundle changed and the saved bundle cannot be verified"
            ) from exc
        saved_content_hash = _source_bundle_content_sha256(saved_source)
        current_content_hash = _source_bundle_content_sha256(source_bundle)
        if saved_content_hash != current_content_hash:
            raise PotentialSearchError(
                "resume environment source content changed: "
                f"saved={saved_content_hash}, current={current_content_hash}"
            )
        reasons.append("git_revision_or_source_locations_changed_with_identical_excerpt_content")

    if not reasons:
        state["environment_source_bundle_sha256"] = current_bundle_hash
        state["environment_source_bundle_content_sha256"] = (
            source_bundle.get("content_sha256")
            or _source_bundle_content_sha256(source_bundle)
        )
        return None

    migration = _ensure_migration_backup(output)
    record = {
        "migration_id": PROMPT_MIGRATION_ID,
        "migrated_at_utc": _now(),
        "reasons": reasons,
        "from_prompt_version": saved_prompt_version,
        "to_prompt_version": SEARCH_PROMPT_VERSION,
        "from_git_sha": (saved_source or {}).get("git_sha", state.get("git_sha")),
        "to_git_sha": source_bundle.get("git_sha"),
        "from_source_bundle_sha256": saved_bundle_hash,
        "to_source_bundle_sha256": current_bundle_hash,
        "verified_source_content_sha256": _source_bundle_content_sha256(source_bundle),
        "state_backup": str((migration / "state.before.json").resolve()),
    }
    existing = state.setdefault("compatibility_migrations", [])
    signature = (
        record["from_prompt_version"],
        record["to_prompt_version"],
        record["from_source_bundle_sha256"],
        record["to_source_bundle_sha256"],
    )
    if not any(
        (
            item.get("from_prompt_version"),
            item.get("to_prompt_version"),
            item.get("from_source_bundle_sha256"),
            item.get("to_source_bundle_sha256"),
        )
        == signature
        for item in existing
    ):
        existing.append(record)
        _atomic_write_json(migration / "migration.json", record)
    state["prompt_version"] = SEARCH_PROMPT_VERSION
    state["environment_source_bundle_sha256"] = current_bundle_hash
    state["environment_source_bundle_content_sha256"] = (
        source_bundle.get("content_sha256")
        or _source_bundle_content_sha256(source_bundle)
    )
    return record


def _prepare_generation_request(
    *,
    output: Path,
    state: dict[str, Any],
    current: dict[str, Any],
    candidate_id: str,
    common: str,
) -> tuple[str, dict[str, int]]:
    record = current["candidates"][candidate_id]
    stage, stage_sections = _generation_stage_prompt_parts(
        state=state, current=current, candidate_id=candidate_id
    )
    request = _base_generation_prompt(
        state=state,
        current=current,
        candidate_id=candidate_id,
        common=common,
        stage=stage,
    )
    sections = {"environment_interface_source": len(common), **stage_sections}
    sections["request_scaffolding"] = len(request) - sum(sections.values())
    metadata = {
        "prompt_version": SEARCH_PROMPT_VERSION,
        "common_contract_sha256": _sha256_text(common.rstrip()),
        "generation_stage_sha256": _sha256_text(stage),
        "generation_request_sha256": _sha256_text(request),
        "sections": sections,
    }
    sent = bool(record.get("model_call_records") or record.get("versions"))
    old_request = record.get("generation_request")
    old_stage = record.get("generation_stage_request")
    old_metadata = record.get("generation_prompt_metadata") or {}
    if old_request and old_stage:
        suffix = "\n\n" + old_stage + "\n"
        if not old_request.endswith(suffix):
            raise PotentialSearchError(
                "saved generation request cannot be separated into COMMON and stage"
            )
        old_common = old_request[: -len(suffix)]

        def normalize_lines(value: str) -> str:
            return re.sub(
                r"(?m)^(# Source: [^\r\n]+?):\d+-\d+(; symbol:)",
                r"\1:<source-lines>\2",
                value.rstrip(),
            )

        if normalize_lines(old_common) != normalize_lines(common):
            raise PotentialSearchError(
                "resume environment/interface COMMON prompt content changed"
            )
    already_current = (
        old_metadata.get("prompt_version") == SEARCH_PROMPT_VERSION
        and old_metadata.get("generation_request_sha256")
        == metadata["generation_request_sha256"]
    )
    if already_current:
        record["active_generation_stage_request"] = old_stage
        record["active_generation_prompt_metadata"] = old_metadata
        return old_request, dict(old_metadata.get("sections") or sections)

    if old_request and old_request != request:
        migration = _ensure_migration_backup(output)
        archive = migration / "p" / candidate_id
        request_path, old_hash = _archive_cached_prompt(
            archive, kind="request", text=old_request
        )
        stage_path = None
        old_stage_hash = None
        if old_stage is not None:
            stage_path, old_stage_hash = _archive_cached_prompt(
                archive, kind="stage", text=old_stage
            )
        migration_record = {
            "migrated_at_utc": _now(),
            "from_prompt_version": old_metadata.get(
                "prompt_version", LEGACY_PROMPT_VERSION
            ),
            "to_prompt_version": SEARCH_PROMPT_VERSION,
            "old_generation_request_sha256": old_hash,
            "old_generation_stage_sha256": old_stage_hash,
            "new_generation_request_sha256": metadata[
                "generation_request_sha256"
            ],
            "old_request_archive": str(request_path.resolve()),
            "old_stage_archive": (
                str(stage_path.resolve()) if stage_path is not None else None
            ),
            "reason": "unsent cached request rebuilt with compact training feedback",
        }
        migrations = record.setdefault("generation_request_migrations", [])
        if not any(
            item.get("old_generation_request_sha256") == old_hash
            and item.get("new_generation_request_sha256")
            == metadata["generation_request_sha256"]
            for item in migrations
        ):
            migrations.append(migration_record)

    if sent:
        record["active_generation_stage_request"] = stage
        record["active_generation_prompt_metadata"] = metadata
    else:
        record["generation_stage_request"] = stage
        record["generation_request"] = request
        record["generation_prompt_metadata"] = metadata
        record["active_generation_stage_request"] = stage
        record["active_generation_prompt_metadata"] = metadata
    return request, sections


def _repair_prompt_extra_sections(
    record: dict[str, Any], *, duplicate: bool
) -> dict[str, int]:
    if duplicate:
        response = json.dumps(
            record.get("last_submission"), indent=2, ensure_ascii=False
        )
    else:
        response = json.dumps(
            _load_failed_response_material(record), indent=2, ensure_ascii=False
        )
    feedback = json.dumps(
        record.get("last_validation"), indent=2, ensure_ascii=False
    )
    return {
        "repair_failed_response": len(response),
        "repair_error_feedback": len(feedback),
    }


def _save_prompt_budget_report(
    *,
    output: Path,
    candidate_id: str,
    record: dict[str, Any],
    prompt: str,
    budget: dict[str, Any],
    sections: dict[str, int],
    repair: bool,
) -> dict[str, Any]:
    prompt_hash = _sha256_text(prompt)
    directory = output / "prompt_diagnostics" / candidate_id
    prompt_path = directory / f"pending-{prompt_hash[:16]}.txt"
    report_path = directory / f"pending-{prompt_hash[:16]}.json"
    if not prompt_path.exists():
        _atomic_write_text(prompt_path, prompt)
    report = {
        "schema_version": "uav-hrl-llm-potential-search-prompt-budget-v1",
        "prompt_version": SEARCH_PROMPT_VERSION,
        "candidate_id": candidate_id,
        "request_kind": "repair" if repair else "generation",
        "prompt_sha256": prompt_hash,
        "prompt_path": str(prompt_path.resolve()),
        "sections_characters": sections,
        "section_characters_total": sum(sections.values()),
        "token_counting_or_estimation_method": budget.get("method"),
        "token_count_exact": budget.get("exact"),
        "full_request_budget": copy.deepcopy(budget),
        "includes_full_request_and_output_reservation": True,
    }
    _atomic_write_json(report_path, report)
    record["last_prompt_budget_report"] = report
    return report


def _resolve_baseline_evaluation(
    requested: str | Path,
) -> tuple[Path, Path]:
    """Resolve one explicit paper-evaluation metadata file and its saved manifest."""

    requested = Path(requested).resolve()
    metadata_candidates: list[Path] = []
    if requested.is_file():
        metadata_candidates = [requested]
    elif requested.is_dir():
        direct = requested / "paper_evaluation_metadata.json"
        parent = requested.parent / "paper_evaluation_metadata.json"
        if direct.is_file():
            metadata_candidates = [direct]
        elif parent.is_file():
            metadata_candidates = [parent]
        else:
            metadata_candidates = sorted(
                requested.rglob("paper_evaluation_metadata.json")
            )
    if not metadata_candidates:
        raise PotentialSearchError(
            f"no paper_evaluation_metadata.json found under {requested}"
        )
    if len(metadata_candidates) != 1:
        choices = ", ".join(str(path.parent) for path in metadata_candidates)
        raise PotentialSearchError(
            "multiple baseline evaluations found; pass one explicit evaluation "
            f"directory: {choices}"
        )
    metadata_path = metadata_candidates[0]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PotentialSearchError(
            f"baseline evaluation metadata is invalid: {metadata_path}"
        ) from exc
    points = metadata.get("points") or []
    if len(points) != 1:
        raise PotentialSearchError(
            "baseline evaluation must contain exactly one fixed-RoI point"
        )
    manifest_value = points[0].get("scenario_manifest_path")
    if not manifest_value:
        raise PotentialSearchError(
            "baseline evaluation point does not record scenario_manifest_path"
        )
    manifest = Path(manifest_value).resolve()
    if not manifest.is_file():
        raise PotentialSearchError(
            f"baseline evaluation manifest is missing: {manifest}"
        )
    return metadata_path.parent, manifest


def run_potential_search(
    *,
    validation_sources: Iterable[str | Path] | None = None,
    baseline_run: str | Path | None = None,
    baseline_evaluation: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_output_tokens: int = 16384,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    reasoning_effort: str | None = None,
    timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    total_timeout: float = DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    progress_interval: float = DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    worker_timeout: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    beta: float = DEFAULT_BETA,
    train_episodes: int | None = None,
    evaluation_episodes: int | None = None,
    summary_block_size: int | None = None,
    max_rounds: int | None = None,
    max_repairs_per_candidate: int | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
    client: Any | None = None,
    training_runner: Callable[..., dict[str, Any]] | None = None,
    evaluation_runner: Callable[..., dict[str, Any]] | None = None,
    baseline_validator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or resume the independent four-candidate trained-feedback search."""

    if resume is not None and dry_run:
        raise ValueError("--dry-run cannot be combined with --resume")
    if resume is None:
        if float(beta) != DEFAULT_BETA:
            raise ValueError("potential shaping beta is fixed at 1")
        train_episodes = (
            DEFAULT_TRAIN_EPISODES if train_episodes is None else int(train_episodes)
        )
        evaluation_episodes = (
            DEFAULT_EVALUATION_EPISODES
            if evaluation_episodes is None
            else int(evaluation_episodes)
        )
        summary_block_size = (
            50 if summary_block_size is None else int(summary_block_size)
        )
        if min(train_episodes, evaluation_episodes, summary_block_size) <= 0:
            raise ValueError("training, evaluation, and summary budgets must be positive")
        if not validation_sources or not provider or not model:
            raise ValueError(
                "new training searches require validation sources, provider, and model"
            )
        baseline_manifest = None
        if baseline_evaluation is not None and not dry_run:
            baseline_evaluation, baseline_manifest = _resolve_baseline_evaluation(
                baseline_evaluation
            )
        max_rounds = DEFAULT_ROUNDS if max_rounds is None else int(max_rounds)
        max_repairs_per_candidate = (
            DEFAULT_REPAIRS_PER_CANDIDATE
            if max_repairs_per_candidate is None
            else int(max_repairs_per_candidate)
        )
        if max_rounds <= 0 or max_repairs_per_candidate < 0:
            raise ValueError("rounds must be positive and repair budget non-negative")
        config = _role_config(
            provider=provider,
            model=model,
            base_url=base_url,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            connect_timeout=connect_timeout,
            total_timeout=total_timeout,
            progress_interval=progress_interval,
        )
        output = allocate_design_directory(
            _slug(model), output_root=output_root, output_dir=output_dir
        )
        state = {
            "schema_version": SEARCH_RUN_SCHEMA_VERSION,
            "prompt_version": SEARCH_PROMPT_VERSION,
            "method_id": SEARCH_METHOD_ID,
            "status": "running",
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "git_sha": _git_sha(),
            "numeric_operation_rules_version": NUMERIC_OPERATION_RULES_VERSION,
            "output_directory": str(output),
            "search_round": 1,
            "model_calls": 0,
            "rounds": [],
            "current_round": None,
            "best_candidate": None,
            "stop_reason": None,
            "settings": {
                "validation_sources": [
                    str(Path(value).resolve()) for value in validation_sources
                ],
                "validation_episode_limit": DEFAULT_VALIDATION_EPISODES,
                "model": config,
                "beta": float(beta),
                "worker_timeout": float(worker_timeout),
                "max_rounds": int(max_rounds),
                "max_repairs_per_candidate": int(max_repairs_per_candidate),
                "baseline_run": (
                    str(Path(baseline_run).resolve()) if baseline_run else None
                ),
                "baseline_evaluation": (
                    str(Path(baseline_evaluation).resolve())
                    if baseline_evaluation
                    else None
                ),
                "evaluation_manifest": (
                    str(Path(baseline_manifest).resolve())
                    if baseline_manifest is not None
                    else None
                ),
                "training_seed": int(DEFAULT_TRAINING_SEED),
                "train_episodes": int(train_episodes),
                "evaluation_episodes": int(evaluation_episodes),
                "evaluation_roi_count": DEFAULT_EVALUATION_ROI_COUNT,
                "summary_block_size": int(summary_block_size),
                "td3_gamma": float(
                    FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["gamma"]
                ),
                "evaluation_area_m": [1000.0, 1000.0],
                "episode_seconds": DEFAULT_EPISODE_SECONDS,
                "candidate_count_per_round": 4,
                "candidate_generation_calls_per_request": 1,
                "runtime_root": str(Path(runtime_root).resolve()),
                "proxy_screening": {
                    "lipschitz": False,
                    "episode_ordering": False,
                    "reviewer_model": False,
                },
            },
        }
        _save_state(output, state)
    else:
        output = Path(resume).resolve()
        state_path = output / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SEARCH_RUN_SCHEMA_VERSION:
            raise PotentialSearchError(
                "resume directory is not a training-feedback search v1 run"
            )
        settings = state["settings"]
        config = settings["model"]
        validation_sources = settings["validation_sources"]
        baseline_run = settings["baseline_run"]
        baseline_evaluation = settings["baseline_evaluation"]
        runtime_root = settings["runtime_root"]
        saved_rounds = int(settings["max_rounds"])
        saved_repairs = int(settings["max_repairs_per_candidate"])
        requested_rounds = saved_rounds if max_rounds is None else int(max_rounds)
        requested_repairs = (
            saved_repairs
            if max_repairs_per_candidate is None
            else int(max_repairs_per_candidate)
        )
        if requested_rounds < saved_rounds:
            raise PotentialSearchError("resume cannot lower the total round count")
        if requested_repairs < saved_repairs:
            raise PotentialSearchError("resume cannot lower the repair budget")
        settings["max_rounds"] = requested_rounds
        settings["max_repairs_per_candidate"] = requested_repairs
        max_rounds = requested_rounds
        max_repairs_per_candidate = requested_repairs
        beta = float(settings["beta"])
        if beta != DEFAULT_BETA:
            raise PotentialSearchError("saved potential beta is not 1")
        for key, requested in (
            ("train_episodes", train_episodes),
            ("evaluation_episodes", evaluation_episodes),
            ("summary_block_size", summary_block_size),
        ):
            if requested is not None and int(requested) != int(settings[key]):
                raise PotentialSearchError(
                    f"resume {key} differs: saved={settings[key]}, requested={requested}"
                )
        train_episodes = int(settings["train_episodes"])
        evaluation_episodes = int(settings["evaluation_episodes"])
        summary_block_size = int(settings["summary_block_size"])
        worker_timeout = float(settings["worker_timeout"])
        seed = int(settings["training_seed"])
        if state.get("status") == "complete" and requested_rounds > saved_rounds:
            state["search_round"] = len(state["rounds"]) + 1
            state["current_round"] = None
            state["status"] = "running"
            state["stop_reason"] = None
        elif state.get("status") == "complete":
            return _result(state)
        else:
            state["status"] = "running"
            state["stop_reason"] = None

    dataset = bounded_validation_dataset(
        validation_sources,
        episode_limit=int(state["settings"]["validation_episode_limit"]),
    )
    if state.get("validation_dataset") not in (None, dataset.provenance):
        raise PotentialSearchError("resume validation dataset changed")
    state["validation_dataset"] = dataset.provenance
    constants = build_constants(dataset.fixed_metadata)
    source_bundle = build_environment_source_bundle()
    if resume is not None:
        _prepare_resume_compatibility(
            output=output,
            state=state,
            source_bundle=source_bundle,
            constants=constants,
        )
    else:
        state["environment_source_bundle_sha256"] = source_bundle["bundle_sha256"]
        state["environment_source_bundle_content_sha256"] = (
            source_bundle.get("content_sha256")
            or _source_bundle_content_sha256(source_bundle)
        )
    source_dir = output / "prompt_source_context"
    source_dir.mkdir(exist_ok=True)
    _atomic_write_json(source_dir / "environment_source_context.json", source_bundle)
    _write_json(output / "validation_dataset.json", dataset.provenance)
    _write_json(output / "constants.json", constants)

    baseline_validator = baseline_validator or __import__(
        "llm_episode_training", fromlist=["validate_baseline_preflight"]
    ).validate_baseline_preflight
    if dry_run:
        state["baseline_preflight"] = {
            "status": "not_run_dry_run",
            "reason": "dry-run performs no formal baseline or model work",
        }
    else:
        if not baseline_run or not baseline_evaluation:
            raise PotentialSearchError(
                "formal search requires --baseline-run and --baseline-evaluation"
            )
        manifest = Path(state["settings"]["evaluation_manifest"]).resolve()
        try:
            baseline_preflight = baseline_validator(
                baseline_run=baseline_run,
                baseline_evaluation=baseline_evaluation,
                manifest=manifest,
                checkpoint_episode=train_episodes,
                episodes=evaluation_episodes,
                roi_count=DEFAULT_EVALUATION_ROI_COUNT,
                environment_size_m=(1000.0, 1000.0),
                episode_seconds=DEFAULT_EPISODE_SECONDS,
            )
        except Exception as exc:
            failure = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "baseline_evaluation": str(Path(baseline_evaluation).resolve()),
            }
            state["status"] = "paused_baseline_incomplete_or_incompatible"
            state["stop_reason"] = f"{type(exc).__name__}: {exc}"
            state["baseline_preflight"] = failure
            _write_json(output / "baseline_preflight.json", failure)
            _save_state(output, state)
            raise PotentialSearchError(
                f"baseline preflight failed: {type(exc).__name__}: {exc}"
            ) from exc
        saved_preflight = state.get("baseline_preflight")
        if (
            saved_preflight is not None
            and saved_preflight.get("status") == "passed"
            and saved_preflight != baseline_preflight
        ):
            raise PotentialSearchError("baseline inputs changed since the saved preflight")
        state["baseline_preflight"] = baseline_preflight
        state["settings"]["evaluation_manifest"] = baseline_preflight["manifest_path"]
        _write_json(output / "baseline_preflight.json", baseline_preflight)

    if dry_run:
        preview = output / "dry_run"
        preview.mkdir(exist_ok=True)
        candidate_id = _candidate_id(1, 1)
        common = render_common_prompt(
            candidate_id=candidate_id,
            dataset=dataset,
            constants=constants,
            beta=beta,
            source_bundle=source_bundle,
            train_episodes=train_episodes,
            evaluation_episodes=evaluation_episodes,
        )
        fake_current = _new_round(state, 1)
        prompt = _base_generation_prompt(
            state=state,
            current=fake_current,
            candidate_id=candidate_id,
            common=common,
        )
        (preview / "initial_candidate_prompt.txt").write_text(prompt, encoding="utf-8")
        budget = estimate_token_budget(
            prompt,
            context_length=config["context_length"],
            max_output_tokens=config["max_output_tokens"],
        )
        _write_json(preview / "token_budget.json", budget)
        state["status"] = "dry_run_complete"
        _save_state(output, state)
        return _result(state)

    try:
        model_client = client or _make_client(config)
        inventory = (
            None
            if config["provider"] == "openai"
            else model_inventory_summary(model_client.list_models(), config["model"])
        )
    except Exception as exc:
        failure = {
            "recorded_at_utc": _now(),
            "stage": "provider_client_or_model_inventory",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        state["model_setup_failure"] = failure
        state["status"] = "paused_model_transport_or_authentication_failure"
        state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        _save_state(output, state)
        return _result(state)
    effective_context, context_record = _effective_context_budget(
        config["context_length"], inventory
    )
    state["model_inventory"] = inventory
    state["context"] = context_record
    training_runner = training_runner or __import__(
        "llm_episode_training", fromlist=["run_candidate_training"]
    ).run_candidate_training
    evaluation_runner = evaluation_runner or __import__(
        "llm_episode_training", fromlist=["run_candidate_evaluation"]
    ).run_candidate_evaluation
    _save_state(output, state)

    training_seed = int(state["settings"]["training_seed"])

    while int(state["search_round"]) <= int(max_rounds):
        round_number = int(state["search_round"])
        if state.get("current_round") is None:
            state["current_round"] = _new_round(state, round_number)
            _save_state(output, state)
        current = state["current_round"]

        if current["phase"] == "generate":
            index = int(current["generation_index"])
            if index >= 4:
                current["phase"] = "train"
                current["work_index"] = 0
                _save_state(output, state)
                continue
            candidate_id = current["candidate_order"][index]
            record = current["candidates"][candidate_id]
            _progress(
                f"round {round_number}/{max_rounds}; {candidate_id}; "
                f"direction={record['requested_direction']}; corrections={record['corrections_used']}"
            )
            if record["versions"] and int(record["corrections_used"]) >= int(
                max_repairs_per_candidate
            ):
                record["status"] = "repairs_exhausted"
                state["status"] = "paused_candidate_repairs_exhausted"
                state["stop_reason"] = (
                    f"{candidate_id} exhausted {max_repairs_per_candidate} corrections"
                )
                _save_state(output, state)
                return _result(state)
            common = render_common_prompt(
                candidate_id=candidate_id,
                dataset=dataset,
                constants=constants,
                beta=beta,
                source_bundle=source_bundle,
                train_episodes=train_episodes,
                evaluation_episodes=evaluation_episodes,
            )
            generation_request, prompt_sections = _prepare_generation_request(
                output=output,
                state=state,
                current=current,
                candidate_id=candidate_id,
                common=common,
            )
            repair = bool(record["versions"])
            if repair:
                prompt = _repair_prompt(
                    common=common,
                    record=record,
                    duplicate=(record.get("last_failure_stage") == "duplicate"),
                )
                prompt_sections = {
                    **prompt_sections,
                    **_repair_prompt_extra_sections(
                        record,
                        duplicate=(record.get("last_failure_stage") == "duplicate"),
                    ),
                }
                prompt_sections["repair_scaffolding"] = len(prompt) - sum(
                    prompt_sections.values()
                )
            else:
                prompt = generation_request
            budget = estimate_token_budget(
                prompt,
                context_length=effective_context,
                max_output_tokens=config["max_output_tokens"],
            )
            _save_prompt_budget_report(
                output=output,
                candidate_id=candidate_id,
                record=record,
                prompt=prompt,
                budget=budget,
                sections=prompt_sections,
                repair=repair,
            )
            if not budget["fits_client_budget"]:
                state["status"] = "paused_context_budget_exceeded"
                state["stop_reason"] = (
                    f"required prompt for {candidate_id} does not fit the configured context"
                )
                record["last_token_budget"] = budget
                _save_state(output, state)
                return _result(state)
            state["model_calls"] += 1
            model_root = _short_runtime_output_root(
                runtime_root,
                output,
                kind="model",
                round_number=round_number,
                slot_index=index + 1,
            )
            call_dir = model_root / f"c{state['model_calls']:04d}"
            call_record = {
                "call": state["model_calls"],
                "directory": str(call_dir),
                "prompt_sha256": _sha256_text(prompt),
                "token_budget": budget,
                "prompt_budget_report": record["last_prompt_budget_report"],
                "started_at_utc": _now(),
            }
            record.setdefault("model_call_records", []).append(call_record)
            _save_state(output, state)
            try:
                response = _call_model(
                    prompt=prompt,
                    config=config,
                    client=model_client,
                    directory=call_dir,
                    effective_context=effective_context,
                )
            except Exception as exc:
                call_record["status"] = "transport_or_protocol_failure"
                call_record["completed_at_utc"] = _now()
                record["transport_failures"].append(
                    {
                        "recorded_at_utc": _now(),
                        "call": state["model_calls"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                state["status"] = "paused_model_transport_or_protocol_failure"
                state["stop_reason"] = f"{type(exc).__name__}: {exc}"
                _save_state(output, state)
                return _result(state)

            call_record["status"] = "completed"
            call_record["completed_at_utc"] = _now()
            raw = response["content"]
            submission = normalized = None
            response_kind = "raw_unparsed"
            validation: dict[str, Any]
            try:
                submission, parse_metadata = parse_candidate_json_envelope(raw)
                response_kind = "parsed_json"
                submission = _substitute_candidate_id(submission, candidate_id)
                validation = validate_potential_candidate(
                    submission,
                    constants_metadata=constants,
                    obs_arrays=build_obs_arrays(dataset.arrays),
                    timeout=worker_timeout,
                    expected_id=candidate_id,
                )
                normalized = copy.deepcopy(submission)
                extra = True
                validation = {**validation, "parse_metadata": parse_metadata}
                if extra is not None:
                    fingerprint = candidate_fingerprint(normalized)
                    duplicate = _duplicate_diagnostics(
                        state, current, candidate_id, fingerprint
                    )
                    if duplicate is not None:
                        validation = duplicate
                        extra = None
                    else:
                        reference_candidate = (
                            (current.get("reference_candidate") or {}).get("candidate")
                        )
                        direction_report = validate_direction(
                            record["requested_direction"], normalized, reference_candidate
                        )
                        validation["direction"] = direction_report
                        if direction_report["status"] != "passed":
                            validation = {
                                "status": "failed",
                                "stage": "direction",
                                "direction": direction_report,
                            }
                            extra = None
                else:
                    fingerprint = (
                        candidate_fingerprint(normalized)
                        if normalized is not None
                        else None
                    )
            except (CandidateExecutionError,) as exc:
                validation = {
                    "status": "failed",
                    "stage": "execution",
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_details": _exception_details(exc),
                }
                extra = None
                fingerprint = None
            except (CandidateError, ValueError, SyntaxError) as exc:
                validation = {
                    "status": "failed",
                    "stage": "parse_or_schema",
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_details": _exception_details(exc),
                }
                extra = None
                fingerprint = None
            version = len(record["versions"]) + 1
            candidate_root = _short_runtime_output_root(
                runtime_root,
                output,
                kind="candidate",
                round_number=round_number,
                slot_index=index + 1,
            )
            version_dir = candidate_root / f"v{version:02d}"
            _save_candidate_version(
                version_dir,
                submission=submission,
                normalized=normalized,
                validation=validation,
                raw_content=raw,
            )
            record["versions"].append(
                {
                    "version": version,
                    "directory": str(version_dir),
                    "validation": validation,
                    "fingerprint": fingerprint,
                    "response_kind": response_kind,
                }
            )
            record["corrections_used"] = max(0, len(record["versions"]) - 1)
            record["last_submission"] = submission
            record["last_validation"] = validation
            record["last_failure_stage"] = validation.get("stage")
            if extra is None:
                if record["corrections_used"] >= int(max_repairs_per_candidate):
                    record["status"] = "repairs_exhausted"
                    state["status"] = "paused_candidate_repairs_exhausted"
                    state["stop_reason"] = (
                        f"{candidate_id} exhausted {max_repairs_per_candidate} corrections"
                    )
                    _save_state(output, state)
                    return _result(state)
                _progress(
                    f"{candidate_id} failed {validation.get('stage')}; "
                    f"next correction {record['corrections_used'] + 1}/"
                    f"{max_repairs_per_candidate}"
                )
                _save_state(output, state)
                continue

            direction_report = validation["direction"]
            record.update(
                {
                    "status": "validated",
                    "submission": submission,
                    "candidate": normalized,
                    "validation": validation,
                    "fingerprint": fingerprint,
                    "actual_modification_summary": modification_summary(
                        record["requested_direction"],
                        normalized,
                        (current.get("reference_candidate") or {}).get("candidate"),
                        direction_report,
                    ),
                }
            )
            artifact_parent = _short_runtime_output_root(
                runtime_root,
                output,
                kind="artifact",
                round_number=round_number,
                slot_index=index + 1,
            )
            artifact_parent.mkdir(parents=True, exist_ok=False)
            approved = save_potential_artifact(
                artifact_parent,
                candidate=normalized,
                constants_metadata=constants,
                validation_report=validation,
                evaluation_report={
                    "schema_version": "uav-hrl-potential-search-validation-only-v1",
                    "performance_pre_evaluation_performed": False,
                    "lipschitz_evaluation_performed": False,
                    "episode_ordering_evaluation_performed": False,
                },
                provenance={
                    "approval_method": "executable_validation_without_proxy_screening",
                    "search_run": str(output),
                    "search_round": round_number,
                    "candidate_id": candidate_id,
                    "requested_direction": record["requested_direction"],
                    "base_candidate_id": record["base_candidate_id"],
                    "candidate_computation_fingerprint": fingerprint,
                    "validation_dataset": dataset.provenance,
                    "beta": float(beta),
                    "training_contract": {
                        "episodes": train_episodes,
                        "seed": training_seed,
                        "independent_fresh_initialization": True,
                        "exploration_decay_episodes": 1000,
                    },
                    "environment_source_bundle_sha256": source_bundle[
                        "bundle_sha256"
                    ],
                },
                gamma=float(state["settings"]["td3_gamma"]),
                beta=float(beta),
            )
            record["artifact"] = str(approved)
            current["generation_index"] += 1
            _progress(
                f"validated {current['generation_index']}/4 executable, nonduplicate candidates"
            )
            _save_state(output, state)
            continue

        if current["phase"] == "train":
            index = int(current["work_index"])
            if index >= 4:
                current["phase"] = "aggregate"
                _save_state(output, state)
                continue
            candidate_id = current["candidate_order"][index]
            record = current["candidates"][candidate_id]
            if record.get("training", {}).get("status") == "complete":
                current["phase"] = "evaluate"
                _save_state(output, state)
                continue
            training_root = _short_runtime_output_root(
                runtime_root,
                output,
                kind="training",
                round_number=round_number,
                slot_index=index + 1,
            )
            record["training_output_root"] = str(training_root)
            record["training_invocation"] = {
                "method_id": SEARCH_METHOD_ID,
                "artifact": record["artifact"],
                "episodes": train_episodes,
                "seed": training_seed,
                "output_directory": str(training_root),
                "summary_block_size": summary_block_size,
                "fresh_candidate_initialization": True,
            }
            _save_state(output, state)
            _progress(
                f"training start: round {round_number}, {candidate_id}, "
                f"{train_episodes} episodes; output={training_root}"
            )
            try:
                training = training_runner(
                    method_id=SEARCH_METHOD_ID,
                    artifact=record["artifact"],
                    episodes=train_episodes,
                    seed=training_seed,
                    output_directory=training_root,
                    resume_record=record.get("training"),
                    summary_block_size=summary_block_size,
                )
            except Exception as exc:
                state["status"] = "paused_training_failure"
                state["stop_reason"] = f"{candidate_id}: {type(exc).__name__}: {exc}"
                record["failure"] = {
                    "stage": "training",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _save_state(output, state)
                return _result(state)
            record["training"] = training
            if training.get("status") != "complete":
                state["status"] = "paused_training_incomplete"
                state["stop_reason"] = training.get("reason", "training incomplete")
                _save_state(output, state)
                return _result(state)
            fairness = _training_fairness_provenance(
                training=training,
                baseline_run=baseline_run,
                expected_seed=training_seed,
                expected_episodes=train_episodes,
            )
            other_runs = {
                item.get("training", {}).get("run_directory")
                for item in current["candidates"].values()
                if item.get("candidate_id") != candidate_id
            }
            if training["run_directory"] in other_runs:
                raise PotentialSearchError("two candidates reused the same training run")
            record["training_fairness"] = fairness
            current["phase"] = "evaluate"
            _progress(f"training complete: {candidate_id}; run={training['run_directory']}")
            _save_state(output, state)
            continue

        if current["phase"] == "evaluate":
            index = int(current["work_index"])
            candidate_id = current["candidate_order"][index]
            record = current["candidates"][candidate_id]
            if record.get("evaluation", {}).get("status") == "complete":
                current["work_index"] += 1
                current["phase"] = "train"
                _save_state(output, state)
                continue
            evaluation_root = _short_runtime_output_root(
                runtime_root,
                output,
                kind="evaluation",
                round_number=round_number,
                slot_index=index + 1,
            )
            attempt = int(record.get("evaluation_attempts", 0)) + 1
            record["evaluation_attempts"] = attempt
            attempt_dir = evaluation_root / f"a{attempt:02d}"
            record.setdefault("evaluation_invocations", []).append(
                {
                    "attempt": attempt,
                    "method_id": SEARCH_METHOD_ID,
                    "run_directory": record["training"]["run_directory"],
                    "checkpoint_episode": train_episodes,
                    "episodes": evaluation_episodes,
                    "roi_count": DEFAULT_EVALUATION_ROI_COUNT,
                    "environment_size_m": [1000.0, 1000.0],
                    "manifest": state["baseline_preflight"]["manifest_path"],
                    "manifest_content_hash": state["baseline_preflight"][
                        "manifest_content_hash"
                    ],
                    "output_directory": str(attempt_dir),
                }
            )
            _save_state(output, state)
            _progress(
                f"evaluation start: {candidate_id}; checkpoint episode "
                f"{train_episodes}; output={attempt_dir}"
            )
            try:
                evaluation = evaluation_runner(
                    method_id=SEARCH_METHOD_ID,
                    run_directory=record["training"]["run_directory"],
                    checkpoint_episode=train_episodes,
                    episodes=evaluation_episodes,
                    roi_count=DEFAULT_EVALUATION_ROI_COUNT,
                    environment_size_m=(1000.0, 1000.0),
                    manifest=state["baseline_preflight"]["manifest_path"],
                    baseline_run=baseline_run,
                    baseline_evaluation=baseline_evaluation,
                    baseline_preflight=state["baseline_preflight"],
                    output_directory=attempt_dir,
                )
            except Exception as exc:
                record.setdefault("evaluation_failures", []).append(
                    {
                        "attempt": attempt,
                        "output_directory": str(attempt_dir),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                state["status"] = "paused_evaluation_failure"
                state["stop_reason"] = f"{candidate_id}: {type(exc).__name__}: {exc}"
                record["failure"] = record["evaluation_failures"][-1]
                _save_state(output, state)
                return _result(state)
            if evaluation.get("status") != "complete":
                state["status"] = "paused_evaluation_incomplete"
                state["stop_reason"] = evaluation.get("reason", "evaluation incomplete")
                record["evaluation"] = evaluation
                _save_state(output, state)
                return _result(state)
            record["evaluation"] = evaluation
            record["status"] = "completed"
            record.pop("failure", None)
            _progress(
                f"evaluation complete: {candidate_id}; mean episode EE="
                f"{float(evaluation['mean_episode_energy_efficiency_mbit_per_j']):.9g}; "
                f"baseline gap={evaluation.get('baseline_percent_gap')!r}%"
            )
            current["work_index"] += 1
            current["phase"] = "train"
            _save_state(output, state)
            continue

        if current["phase"] == "aggregate":
            scores = []
            for candidate_id in current["candidate_order"]:
                record = current["candidates"][candidate_id]
                evaluation = record.get("evaluation") or {}
                if evaluation.get("status") != "complete":
                    raise PotentialSearchError(
                        "round aggregation requires four completed evaluations"
                    )
                scores.append(
                    (
                        candidate_id,
                        float(
                            evaluation[
                                "mean_episode_energy_efficiency_mbit_per_j"
                            ]
                        ),
                    )
                )
            round_best_id, round_best_score = max(
                scores,
                key=lambda item: (
                    item[1],
                    -current["candidate_order"].index(item[0]),
                ),
            )
            best = state.get("best_candidate")
            if best is None or round_best_score > float(best["mean_episode_energy_efficiency_mbit_per_j"]):
                source = current["candidates"][round_best_id]
                state["best_candidate"] = {
                    "round": round_number,
                    "candidate_id": round_best_id,
                    "candidate": copy.deepcopy(source["candidate"]),
                    "submission": copy.deepcopy(source["submission"]),
                    "artifact": source["artifact"],
                    "training": copy.deepcopy(source["training"]),
                    "evaluation": copy.deepcopy(source["evaluation"]),
                    "mean_episode_energy_efficiency_mbit_per_j": round_best_score,
                }
                current["updated_historical_best"] = True
            else:
                current["updated_historical_best"] = False
            current["score_table"] = [
                {
                    "candidate_id": candidate_id,
                    "mean_episode_energy_efficiency_mbit_per_j": score,
                    "baseline_percent_gap": current["candidates"][candidate_id][
                        "evaluation"
                    ].get("baseline_percent_gap"),
                }
                for candidate_id, score in scores
            ]
            current["round_best_candidate_id"] = round_best_id
            baseline_mean = state["baseline_preflight"]["metrics"][
                "episode_energy_efficiency_mbit_per_j"
            ]["mean"]
            _progress(
                "round scores: "
                + "; ".join(f"{candidate_id}={score:.9g}" for candidate_id, score in scores)
                + f"; baseline={float(baseline_mean):.9g}"
            )
            current["status"] = "complete"
            current["phase"] = "complete"
            state["rounds"].append(copy.deepcopy(current))
            state["current_round"] = None
            _progress(
                f"round {round_number} complete; historical best="
                f"{state['best_candidate']['candidate_id']} "
                f"({state['best_candidate']['mean_episode_energy_efficiency_mbit_per_j']:.9g})"
            )
            if round_number >= int(max_rounds):
                state["status"] = "complete"
                state["stop_reason"] = None
                _write_completed_search_artifacts(output, state)
                _save_state(output, state)
                return _result(state)
            state["search_round"] = round_number + 1
            _save_state(output, state)
            continue

        raise PotentialSearchError(f"unknown round phase: {current['phase']}")

    state["status"] = "complete"
    _save_state(output, state)
    return _result(state)
