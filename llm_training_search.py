"""Train-and-evaluate every LLM reward candidate without proxy screening.

This workflow is intentionally independent from the Lipschitz, reviewer-agent,
and complete-episode ordering searches.  It generates one candidate per model
request, validates only executable/runtime contracts, trains all four accepted
candidates from scratch, evaluates their episode-800 checkpoints on the exact
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
    parse_candidate_json_envelope,
    save_approved_artifact,
)
from llm_design import (
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_BETA,
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
)
from llm_episode_search import (
    EpisodeDataset,
    EpisodeRecord,
    _call_model,
    _make_client,
    _role_config,
    candidate_computation_fingerprint,
    load_complete_episode_dataset,
    validate_search_candidate,
)
from llm_numeric_operations import NUMERIC_OPERATION_RULES_VERSION
from experiment_config import DEFAULT_TRAINING_SEED
from scenario_manifest import ScenarioManifest


SEARCH_RUN_SCHEMA_VERSION = "uav-hrl-llm-training-search-run-v1"
SEARCH_PROMPT_VERSION = "uav-hrl-llm-training-search-prompt-v1"
SEARCH_METHOD_ID = "td3_dinkelbach_llm_train_search"
DEFAULT_OUTPUT_ROOT = Path("results") / "llm_training_searches"
DEFAULT_RUNTIME_ROOT = Path.home() / "uav_hrl_rt"
DEFAULT_ROUNDS = 5
DEFAULT_REPAIRS_PER_CANDIDATE = 5
DEFAULT_TRAIN_EPISODES = 800
DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_EVALUATION_ROI_COUNT = 8
DEFAULT_EPISODE_SECONDS = 60
DEFAULT_VALIDATION_EPISODES = 4
ROUND_DIRECTIONS = ("add", "remove", "reweight", "redesign")

ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = ROOT / "prompts"
COMMON_TEMPLATE = PROMPT_ROOT / "llm_training_search_common.txt"
INITIAL_TEMPLATE = PROMPT_ROOT / "llm_training_search_initial.txt"
FEEDBACK_TEMPLATE = PROMPT_ROOT / "llm_training_search_feedback.txt"
DIRECTION_TEMPLATES = {
    "add": PROMPT_ROOT / "llm_training_search_direction_add.txt",
    "remove": PROMPT_ROOT / "llm_training_search_direction_remove.txt",
    "reweight": PROMPT_ROOT / "llm_training_search_direction_reweight.txt",
    "redesign": PROMPT_ROOT / "llm_training_search_direction_redesign.txt",
}
VALIDATION_REPAIR_TEMPLATE = PROMPT_ROOT / "llm_training_search_validation_repair.txt"
DUPLICATE_REPAIR_TEMPLATE = PROMPT_ROOT / "llm_training_search_duplicate_repair.txt"
_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


class TrainingSearchError(RuntimeError):
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
        raise TrainingSearchError(
            "training-search prompt is missing replacements for: "
            + ", ".join(missing)
        )
    return _PLACEHOLDER.sub(
        lambda match: str(replacements[match.group(1)]), template
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(message: str) -> None:
    print(f"[llm-training-search] {message}", flush=True)


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
    required = {"candidate_id", "design_summary", "features", "code"}
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
        raise TrainingSearchError("validation sources contain no complete episodes")
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
            raise TrainingSearchError(f"source symbol not found: {qualified_name}")
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
        raise TrainingSearchError(f"source marker not found: {relative_path}: {start_marker}")
    end = next(
        (i for i in range(start, len(lines)) if end_marker in lines[i]), None
    )
    if end is None:
        raise TrainingSearchError(f"source marker not found: {relative_path}: {end_marker}")
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
        "HRL_task_aware.train -> _interval_reward, then adds beta times the current-observation LLM reward exactly once.",
    ]
    identity = [
        {
            key: value
            for key, value in item.items()
            if key != "code"
        }
        for item in all_items
    ]
    return {
        "schema_version": "uav-hrl-llm-training-search-source-context-v1",
        "git_sha": _git_sha(),
        "call_relationships": call_relationships,
        "excerpts": categories,
        "rendered": rendered,
        "bundle_sha256": _sha256_json(
            {"git_sha": _git_sha(), "call_relationships": call_relationships, "identity": identity}
        ),
    }


def _minimal_example(candidate_id: str) -> str:
    example = minimal_model_candidate_example()
    return json.dumps(
        {
            "candidate_id": candidate_id,
            "design_summary": "Format-only example; not a design recommendation.",
            **example,
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
            "TRAIN_EPISODES": DEFAULT_TRAIN_EPISODES,
            "EVALUATION_EPISODES": DEFAULT_EVALUATION_EPISODES,
            "EVALUATION_ROI_COUNT": DEFAULT_EVALUATION_ROI_COUNT,
            "EVALUATION_AREA_SPEC": "1000 × 1000 m",
            "EPISODE_SECONDS": DEFAULT_EPISODE_SECONDS,
        },
    )


def _render_stage(template: Path, replacements: dict[str, Any]) -> str:
    return _replace_template(template.read_text(encoding="utf-8"), replacements)


def _short_design(record: dict[str, Any]) -> dict[str, Any]:
    submission = record.get("submission") or {}
    features = submission.get("features") or []
    return {
        "candidate_id": submission.get("candidate_id", record.get("candidate_id")),
        "design_summary": str(submission.get("design_summary", ""))[:600],
        "features": [
            {
                "name": item.get("name"),
                "reward_weight": item.get("reward_weight"),
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
    }


def _evaluation_feedback(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    keys = (
        "status",
        "checkpoint_episode",
        "episode_count",
        "scenario_manifest_hash",
        "mean_episode_energy_efficiency_mbit_per_j",
        "std_episode_energy_efficiency_mbit_per_j",
        "baseline_mean_episode_energy_efficiency_mbit_per_j",
        "baseline_absolute_gap_mbit_per_j",
        "baseline_percent_gap",
        "candidate_metrics",
    )
    return {key: result.get(key) for key in keys}


def _latest_round_feedback(round_record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    training = []
    for candidate_id in round_record["candidate_order"]:
        record = round_record["candidates"][candidate_id]
        results.append(
            {
                **_short_design(record),
                "requested_direction": record.get("requested_direction"),
                "actual_modification_summary": record.get("actual_modification_summary"),
                "evaluation": _evaluation_feedback(record.get("evaluation")),
                "failure": record.get("failure"),
            }
        )
        training.append(
            {
                "candidate_id": candidate_id,
                "status": (record.get("training") or {}).get("status"),
                "blocks": (record.get("training") or {}).get("summaries"),
                "fairness_provenance": record.get("training_fairness"),
                "failure": record.get("failure"),
            }
        )
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
                    "status": record.get("status"),
                    "failure": record.get("failure"),
                }
            )
    return summaries


def _generation_stage_prompt(
    *, state: dict[str, Any], current: dict[str, Any], candidate_id: str
) -> str:
    """Render the non-COMMON part for generation and bounded repair prompts."""

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
        return _render_stage(
            INITIAL_TEMPLATE,
            {
                "CURRENT_ROUND_CANDIDATE_SUMMARIES": json.dumps(
                    summaries, indent=2, ensure_ascii=False
                )
            },
        ).strip()

    latest = state["rounds"][-1]
    latest_results, latest_training = _latest_round_feedback(latest)
    best = _best_candidate_feedback(current["reference_candidate"])
    feedback = _render_stage(
        FEEDBACK_TEMPLATE,
        {
            "BEST_CANDIDATE": json.dumps(best, indent=2, ensure_ascii=False),
            "BASELINE_EVALUATION_RESULTS": json.dumps(
                {
                    "checkpoint_episode": state["baseline_preflight"][
                        "checkpoint_episode"
                    ],
                    "manifest_content_hash": state["baseline_preflight"][
                        "manifest_content_hash"
                    ],
                    "metrics": state["baseline_preflight"]["metrics"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "LATEST_ROUND_CANDIDATE_RESULTS": json.dumps(
                latest_results, indent=2, ensure_ascii=False
            ),
            "LATEST_ROUND_TRAINING_SUMMARIES": json.dumps(
                latest_training, indent=2, ensure_ascii=False
            ),
            "TRAINING_BLOCK_DEFINITION": "1–200, 201–400, 401–600, 601–800",
            "HISTORICAL_CANDIDATE_SUMMARIES": json.dumps(
                _historical_summaries(state, exclude_round=int(latest["round"])),
                indent=2,
                ensure_ascii=False,
            ),
        },
    )
    direction = current["candidates"][candidate_id]["requested_direction"]
    return (
        feedback.strip()
        + "\n\n"
        + DIRECTION_TEMPLATES[direction].read_text(encoding="utf-8").strip()
    )


def _base_generation_prompt(
    *,
    state: dict[str, Any],
    current: dict[str, Any],
    candidate_id: str,
    common: str,
) -> str:
    stage = _generation_stage_prompt(
        state=state, current=current, candidate_id=candidate_id
    )
    return common.rstrip() + "\n\n" + stage + "\n"


def _repair_prompt(
    *, common: str, record: dict[str, Any], duplicate: bool
) -> str:
    if duplicate:
        stage = _render_stage(
            DUPLICATE_REPAIR_TEMPLATE,
            {
                "GENERATION_REQUEST": record["generation_stage_request"],
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
                "GENERATION_REQUEST": record["generation_stage_request"],
                "FAILED_RESPONSE_MATERIAL": json.dumps(
                    failed_response, indent=2, ensure_ascii=False
                ),
                "VALIDATION_FEEDBACK": json.dumps(
                    record.get("last_validation"), indent=2, ensure_ascii=False
                ),
            },
        )
    return common.rstrip() + "\n\n" + stage.strip() + "\n"


def _load_failed_response_material(record: dict[str, Any]) -> dict[str, Any]:
    """Reload the latest failed response from its persisted candidate version."""

    versions = record.get("versions") or []
    if not versions:
        raise TrainingSearchError("repair prompt requires a saved failed version")
    latest = versions[-1]
    directory = Path(latest["directory"])
    response_kind = latest.get("response_kind")
    if response_kind is None:
        raw_path = directory / "raw_response.txt"
        try:
            legacy_raw = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TrainingSearchError(
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
            raise TrainingSearchError(
                f"saved parsed candidate is missing or invalid: {path}"
            ) from exc
        return {
            "response_kind": "parsed_candidate",
            "parsed_candidate": parsed,
        }
    if response_kind != "raw_unparsed":
        raise TrainingSearchError(f"unsupported saved response kind: {response_kind!r}")
    path = directory / "raw_response.txt"
    try:
        raw_response = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainingSearchError(
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
            and node.name == "compute_extra_state"
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
        raise TrainingSearchError("revision direction lacks its fixed reference candidate")
    old_weights = [float(item["reward_weight"]) for item in reference["features"]]
    new_weights = [float(item["reward_weight"]) for item in candidate["features"]]
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
            if candidate_computation_fingerprint(candidate) == candidate_computation_fingerprint(reference):
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
        if candidate_computation_fingerprint(candidate) == candidate_computation_fingerprint(reference):
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
            "feature_count": len(candidate["features"]),
            "weights": [float(item["reward_weight"]) for item in candidate["features"]],
            "basis": None,
        }
    old = [float(item["reward_weight"]) for item in reference["features"]]
    new = [float(item["reward_weight"]) for item in candidate["features"]]
    return {
        "requested_direction": direction,
        "basis_candidate_id": reference.get("candidate_name"),
        "feature_count_before": len(old),
        "feature_count_after": len(new),
        "feature_names_before": [
            str(item["name"]) for item in reference["features"]
        ],
        "feature_names_after": [
            str(item["name"]) for item in candidate["features"]
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
        raise TrainingSearchError("candidate training seed differs from the search contract")
    if int(resolved.get("episodes", -1)) != int(expected_episodes):
        raise TrainingSearchError("candidate training horizon differs from episode 800")
    if candidate_manifest.episode_count != int(expected_episodes):
        raise TrainingSearchError("candidate training manifest has the wrong length")
    if candidate_manifest.episodes != baseline_manifest.episodes[: int(expected_episodes)]:
        raise TrainingSearchError(
            "candidate training scenarios do not equal the baseline run's first 800 scenarios"
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
        raise TrainingSearchError(f"candidate/baseline fairness settings differ: {mismatches}")
    exploration = resolved["exploration_schedule_configuration"]
    if int(exploration["movement_exploration_decay_episodes"]) != 1000 or int(
        exploration["routing_epsilon_decay_episodes"]
    ) != 1000:
        raise TrainingSearchError("episode-800 training compressed the exploration schedule")
    prefix_hash = _sha256_json(candidate_manifest.episodes)
    return {
        "schema_version": "uav-hrl-llm-training-search-fairness-v1",
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
        "episode_800_is_stop_not_decay_horizon": True,
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
        raise TrainingSearchError("later round requires a completed historical best candidate")
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


def _save_state(output: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _now()
    _write_json(output / "state.json", state)


def run_training_search(
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
        if not validation_sources or not provider or not model:
            raise ValueError(
                "new training searches require validation sources, provider, and model"
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
                    str((Path(baseline_evaluation).resolve() / "scenario_manifest.json"))
                    if baseline_evaluation and Path(baseline_evaluation).is_dir()
                    else None
                ),
                "training_seed": int(DEFAULT_TRAINING_SEED),
                "train_episodes": DEFAULT_TRAIN_EPISODES,
                "evaluation_episodes": DEFAULT_EVALUATION_EPISODES,
                "evaluation_roi_count": DEFAULT_EVALUATION_ROI_COUNT,
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
            raise TrainingSearchError(
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
            raise TrainingSearchError("resume cannot lower the total round count")
        if requested_repairs < saved_repairs:
            raise TrainingSearchError("resume cannot lower the repair budget")
        settings["max_rounds"] = requested_rounds
        settings["max_repairs_per_candidate"] = requested_repairs
        max_rounds = requested_rounds
        max_repairs_per_candidate = requested_repairs
        beta = float(settings["beta"])
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
        raise TrainingSearchError("resume validation dataset changed")
    state["validation_dataset"] = dataset.provenance
    constants = build_constants(dataset.fixed_metadata)
    source_bundle = build_environment_source_bundle()
    if state.get("environment_source_bundle_sha256") not in (
        None,
        source_bundle["bundle_sha256"],
    ):
        raise TrainingSearchError("resume source-code prompt bundle changed")
    state["environment_source_bundle_sha256"] = source_bundle["bundle_sha256"]
    source_dir = output / "prompt_source_context"
    source_dir.mkdir(exist_ok=True)
    _write_json(source_dir / "environment_source_context.json", source_bundle)
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
            raise TrainingSearchError(
                "formal search requires --baseline-run and --baseline-evaluation"
            )
        manifest = Path(baseline_evaluation).resolve() / "scenario_manifest.json"
        try:
            baseline_preflight = baseline_validator(
                baseline_run=baseline_run,
                baseline_evaluation=baseline_evaluation,
                manifest=manifest,
                checkpoint_episode=DEFAULT_TRAIN_EPISODES,
                episodes=DEFAULT_EVALUATION_EPISODES,
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
            raise TrainingSearchError(
                f"baseline preflight failed: {type(exc).__name__}: {exc}"
            ) from exc
        saved_preflight = state.get("baseline_preflight")
        if (
            saved_preflight is not None
            and saved_preflight.get("status") == "passed"
            and saved_preflight != baseline_preflight
        ):
            raise TrainingSearchError("baseline inputs changed since the saved preflight")
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
            )
            if not record.get("generation_request"):
                record["generation_stage_request"] = _generation_stage_prompt(
                    state=state,
                    current=current,
                    candidate_id=candidate_id,
                )
                record["generation_request"] = _base_generation_prompt(
                    state=state,
                    current=current,
                    candidate_id=candidate_id,
                    common=common,
                )
            prompt = (
                _repair_prompt(
                    common=common,
                    record=record,
                    duplicate=(record.get("last_failure_stage") == "duplicate"),
                )
                if record["versions"]
                else record["generation_request"]
            )
            budget = estimate_token_budget(
                prompt,
                context_length=effective_context,
                max_output_tokens=config["max_output_tokens"],
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
                normalized, validation, extra = validate_search_candidate(
                    submission,
                    dataset=dataset,
                    constants_metadata=constants,
                    worker_timeout=worker_timeout,
                )
                validation = {**validation, "parse_metadata": parse_metadata}
                if extra is not None:
                    fingerprint = candidate_computation_fingerprint(normalized)
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
                        candidate_computation_fingerprint(normalized)
                        if normalized is not None
                        else None
                    )
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
            approved = save_approved_artifact(
                artifact_parent,
                candidate=normalized,
                constants_metadata=constants,
                validation_report=validation,
                evaluation_report={
                    "schema_version": "uav-hrl-training-search-validation-only-v1",
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
                        "episodes": DEFAULT_TRAIN_EPISODES,
                        "seed": training_seed,
                        "independent_fresh_initialization": True,
                        "exploration_decay_episodes": 1000,
                    },
                    "environment_source_bundle_sha256": source_bundle[
                        "bundle_sha256"
                    ],
                },
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
                "episodes": DEFAULT_TRAIN_EPISODES,
                "seed": training_seed,
                "output_directory": str(training_root),
                "summary_block_size": 200,
                "fresh_candidate_initialization": True,
            }
            _save_state(output, state)
            _progress(f"training start: round {round_number}, {candidate_id}, 800 episodes")
            try:
                training = training_runner(
                    method_id=SEARCH_METHOD_ID,
                    artifact=record["artifact"],
                    episodes=DEFAULT_TRAIN_EPISODES,
                    seed=training_seed,
                    output_directory=training_root,
                    resume_record=record.get("training"),
                    summary_block_size=200,
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
                expected_episodes=DEFAULT_TRAIN_EPISODES,
            )
            other_runs = {
                item.get("training", {}).get("run_directory")
                for item in current["candidates"].values()
                if item.get("candidate_id") != candidate_id
            }
            if training["run_directory"] in other_runs:
                raise TrainingSearchError("two candidates reused the same training run")
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
                    "checkpoint_episode": DEFAULT_TRAIN_EPISODES,
                    "episodes": DEFAULT_EVALUATION_EPISODES,
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
            _progress(f"evaluation start: {candidate_id}; checkpoint episode 800")
            try:
                evaluation = evaluation_runner(
                    method_id=SEARCH_METHOD_ID,
                    run_directory=record["training"]["run_directory"],
                    checkpoint_episode=DEFAULT_TRAIN_EPISODES,
                    episodes=DEFAULT_EVALUATION_EPISODES,
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
                    raise TrainingSearchError(
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
                _save_state(output, state)
                return _result(state)
            state["search_round"] = round_number + 1
            _save_state(output, state)
            continue

        raise TrainingSearchError(f"unknown round phase: {current['phase']}")

    state["status"] = "complete"
    _save_state(output, state)
    return _result(state)
