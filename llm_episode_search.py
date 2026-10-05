"""Single-model, four-candidate offline feature search using complete episodes.

This workflow is deliberately separate from the empirical-Lipschitz design flows.
It validates shared-feature candidates with the same runtime contract, screens them
by episode EE/reward ordering, and delegates long training/evaluation to the normal
experiment entry points (or injected runners in tests).
"""

from __future__ import annotations

import copy
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np

from llm_baseline import FIXED_ARRAY_FIELDS, BaselineDataError, load_sampling_source
from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    feature_reward,
    normalize_candidate_submission,
    parse_candidate_json_envelope,
    save_approved_artifact,
    load_approved_design,
    validate_candidate,
    validate_candidate_staged,
)
from llm_design import (
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_BETA,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    LMStudioClient,
    OpenAIClient,
    _effective_context_budget,
    _git_sha,
    _slug,
    _write_json,
    allocate_design_directory,
    model_inventory_summary,
    planned_chat_request,
    provider_base_url,
    response_model_matches,
)
from llm_design_contract import (
    OBS_INTERFACE_VERSION,
    SUPPORTED_OPERATIONS,
    build_constants,
    build_obs_arrays,
    communication_spec,
    estimate_token_budget,
    movement_and_energy_spec,
    render_environment_interface,
    runtime_diagnostic_contract,
    visual_sensing_spec,
)
from llm_numeric_operations import NUMERIC_OPERATION_RULES_VERSION
from llm_runtime import artifact_identity, load_run_artifact
from training_history import read_committed_training_history, training_history_identity


SEARCH_RUN_SCHEMA_VERSION = "uav-hrl-llm-episode-search-run-v1"
SEARCH_PROMPT_VERSION = "uav-hrl-llm-episode-search-prompt-v1"
SEARCH_METHOD_ID = "td3_dinkelbach_llm_search"
DEFAULT_OUTPUT_ROOT = Path("results") / "llm_episode_searches"
DEFAULT_MAX_SEARCH_ROUNDS = 5
DEFAULT_MAX_REPAIRS_PER_ROUND = 5
DEFAULT_TRAIN_EPISODES = 1500
DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_EVALUATION_ROI_COUNT = 8
DEFAULT_EE_TOLERANCE = 1e-12
DEFAULT_REWARD_TOLERANCE = 1e-12
EXPECTED_CANDIDATE_IDS = tuple(f"candidate_{index}" for index in range(1, 5))
REVALIDATABLE_NUMERIC_RULE_SOURCE_REVISIONS = frozenset(
    {"1460effee1d9ddb1eed31dbc110ce89f36e296d9"}
)
RECOVERABLE_TRAINING_INITIALIZATION_SOURCE_REVISIONS = frozenset(
    {"ef10fa3d5396e236bba3d63b2e152b29339e4226"}
)
RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION = (
    "fa948f14938795b45ded56100af523375481a3db"
)

ROOT = Path(__file__).resolve().parent
COMMON_TEMPLATE = ROOT / "prompts" / "llm_episode_search_common.txt"
INITIAL_TEMPLATE = ROOT / "prompts" / "llm_episode_search_initial.txt"
REPAIR_TEMPLATE = ROOT / "prompts" / "llm_episode_search_repair.txt"
PREEVALUATION_TEMPLATE = ROOT / "prompts" / "llm_episode_search_preevaluation.txt"
TRAINING_TEMPLATE = ROOT / "prompts" / "llm_episode_search_training.txt"
_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


class EpisodeSearchError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_computation_fingerprint(candidate: dict[str, Any]) -> str:
    """Identify exact executable computation and weights, ignoring display names."""

    try:
        code_identity = ast.dump(
            ast.parse(str(candidate["code"]), mode="exec"),
            annotate_fields=True,
            include_attributes=False,
        )
    except SyntaxError:
        code_identity = str(candidate.get("code"))
    return _sha256_json(
        {
            "code_ast": code_identity,
            "reward_weights": [
                float(item["reward_weight"]) for item in candidate.get("features", [])
            ],
            "feature_count": len(candidate.get("features", [])),
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_template(template: str, replacements: dict[str, Any]) -> str:
    required = list(dict.fromkeys(_PLACEHOLDER.findall(template)))
    missing = [name for name in required if name not in replacements]
    if missing:
        raise EpisodeSearchError(
            "episode-search prompt is missing replacements for: " + ", ".join(missing)
        )
    return _PLACEHOLDER.sub(lambda match: str(replacements[match.group(1)]), template)


@dataclass(frozen=True)
class EpisodeRecord:
    index: int
    source_index: int
    source_id: str
    episode_id: int
    scenario_id: str
    rows: np.ndarray


@dataclass(frozen=True)
class EpisodeDataset:
    arrays: dict[str, np.ndarray]
    episodes: tuple[EpisodeRecord, ...]
    fixed_metadata: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def transition_count(self) -> int:
        return int(self.arrays["state"].shape[0])


def load_complete_episode_dataset(source_directories: Iterable[str | Path]) -> EpisodeDataset:
    """Load complete, non-overlapping episodes from validated sampling sources."""

    paths = [Path(value).resolve() for value in source_directories]
    if not paths:
        raise ValueError("at least one episode sampling source is required")
    if len(set(paths)) != len(paths):
        raise BaselineDataError("the same sampling source directory was supplied twice")
    sources = [load_sampling_source(path) for path in paths]
    if len({source["source_id"] for source in sources}) != len(sources):
        raise BaselineDataError("duplicate sampling source content was supplied")
    compatibility = sources[0]["compatibility_contract"]
    for source in sources[1:]:
        if source["compatibility_contract"] != compatibility:
            raise BaselineDataError("episode sources have incompatible environment contracts")

    interval = 1.0
    horizon_seconds = float(compatibility.get("episode_seconds"))
    expected_steps = int(round(horizon_seconds / interval))
    if expected_steps <= 0 or not math.isclose(
        expected_steps * interval, horizon_seconds, rel_tol=0.0, abs_tol=1e-9
    ):
        raise BaselineDataError("episode duration is not an integer movement-step horizon")

    parts = {field: [] for field in FIXED_ARRAY_FIELDS}
    records: list[EpisodeRecord] = []
    sources_metadata: list[dict[str, Any]] = []
    offset = 0
    for source_index, source in enumerate(sources):
        if not bool(np.all(source["usable"])):
            raise BaselineDataError(
                f"episode source contains unusable transitions: {source['directory']}"
            )
        source_episode_ids = np.asarray(source["episode_ids"], dtype=np.int64)
        observed_ids = sorted(set(source_episode_ids.tolist()))
        declared_count = int(source["metadata"].get("episodes", len(observed_ids)))
        if observed_ids != list(range(declared_count)):
            raise BaselineDataError(
                f"episode ids must be unique and consecutive from zero in {source['directory']}"
            )
        for episode_id in observed_ids:
            local_rows = np.flatnonzero(source_episode_ids == episode_id)
            local_steps = np.asarray(source["steps"])[local_rows]
            order = np.argsort(local_steps, kind="stable")
            local_rows = local_rows[order]
            local_steps = local_steps[order]
            if not np.array_equal(local_steps, np.arange(expected_steps, dtype=np.int64)):
                raise BaselineDataError(
                    f"episode {episode_id} in {source['directory']} is incomplete or duplicated"
                )
            scenario_ids = {source["scenario_ids"][int(row)] for row in local_rows}
            if len(scenario_ids) != 1:
                raise BaselineDataError("one episode maps to multiple scenarios")
            destination_rows = np.arange(offset, offset + expected_steps, dtype=np.int64)
            records.append(
                EpisodeRecord(
                    index=len(records),
                    source_index=source_index,
                    source_id=str(source["source_id"]),
                    episode_id=int(episode_id),
                    scenario_id=str(next(iter(scenario_ids))),
                    rows=destination_rows,
                )
            )
            for field in FIXED_ARRAY_FIELDS:
                parts[field].append(np.asarray(source["arrays"][field][local_rows]))
            offset += expected_steps
        sources_metadata.append(
            {
                "source_index": source_index,
                "source_id": source["source_id"],
                "directory": str(source["directory"]),
                "hashes": source["hashes"],
                "episode_count": len(observed_ids),
                "transition_count": int(source["rows"]),
            }
        )
    arrays = {field: np.concatenate(values, axis=0) for field, values in parts.items()}
    fixed_metadata = {
        "schema_version": "uav-hrl-complete-episode-dataset-v1",
        "sample_count": int(arrays["state"].shape[0]),
        "compatibility_contract": compatibility,
    }
    provenance = {
        "schema_version": "uav-hrl-complete-episode-dataset-v1",
        "episode_count": len(records),
        "transition_count": int(arrays["state"].shape[0]),
        "steps_per_episode": expected_steps,
        "sources": sources_metadata,
        "compatibility_hash": _sha256_json(compatibility),
        "dataset_content_sha256": hashlib.sha256(
            "".join(source["source_id"] for source in sources).encode("ascii")
        ).hexdigest(),
    }
    return EpisodeDataset(arrays, tuple(records), fixed_metadata, provenance)


def resolve_evaluation_lambda(
    *, explicit_lambda: float | None = None, training_run: str | Path | None = None
) -> dict[str, Any]:
    """Resolve one fixed screening lambda without guessing or using post-update values."""

    if explicit_lambda is not None:
        value = float(explicit_lambda)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("evaluation lambda must be finite and non-negative")
        return {
            "value": value,
            "method": "explicit_cli_value",
            "source": None,
            "episode_range": None,
        }
    if training_run is None:
        raise ValueError("provide --evaluation-lambda or --lambda-training-run")
    run = Path(training_run).resolve()
    config_path = run / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"training resolved_config.json is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method_id = (config.get("method_spec") or {}).get("method_id", config.get("method_id"))
    if method_id != "td3_dinkelbach":
        raise EpisodeSearchError("lambda history must come from td3_dinkelbach")
    identity_hash = config.get("training_history_identity_manifest_hash")
    if not isinstance(identity_hash, str) or not identity_hash:
        raise EpisodeSearchError("training history identity manifest hash is unavailable")
    identity = training_history_identity(
        "td3_dinkelbach", int(config["seed"]), identity_hash
    )
    rows = read_committed_training_history(run, identity)
    if len(rows) < 100:
        raise EpisodeSearchError("lambda training history has fewer than 100 complete episodes")
    selected = rows[-100:]
    values = np.asarray([row["dinkelbach_lambda_used"] for row in selected], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise EpisodeSearchError("dinkelbach_lambda_used history is incomplete or invalid")
    history_path = run / "training_history.jsonl"
    return {
        "value": float(np.mean(values)),
        "method": "arithmetic_mean_last_100_dinkelbach_lambda_used",
        "source": str(run),
        "source_history_sha256": _file_sha256(history_path),
        "episode_range": [int(selected[0]["episode"]), int(selected[-1]["episode"])],
        "source_method_id": "td3_dinkelbach",
        "sample_count": 100,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _episode_sums(dataset: EpisodeDataset, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != dataset.transition_count:
        raise ValueError("transition values do not align with the episode dataset")
    return np.asarray([np.sum(values[item.rows], axis=0) for item in dataset.episodes])


def episode_reward_components(
    dataset: EpisodeDataset,
    *,
    evaluation_lambda: float,
    beta: float,
    features: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    arrays = dataset.arrays
    delivered = np.asarray(arrays["delivered_mbits"][:, 0], dtype=np.float64)
    energy = np.asarray(arrays["total_mobility_energy"][:, 0], dtype=np.float64)
    penalties = sum(
        np.asarray(arrays[name][:, 0], dtype=np.float64)
        for name in ("c9_penalty", "c10_penalty", "com_range_penalty")
    )
    delivered_episode = _episode_sums(dataset, delivered)
    energy_episode = _episode_sums(dataset, energy)
    if np.any(energy_episode <= 0.0):
        raise EpisodeSearchError("episode movement energy must be finite and positive")
    base_transition = delivered - float(evaluation_lambda) * energy - penalties
    base_episode = _episode_sums(dataset, base_transition)
    if features is None:
        contributions = np.zeros((len(dataset.episodes), 0), dtype=np.float64)
    else:
        features = np.asarray(features, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if features.shape != (dataset.transition_count, weights.size):
            raise ValueError("candidate features and weights do not align")
        contributions = _episode_sums(dataset, float(beta) * features * weights[None, :])
    extra = np.sum(contributions, axis=1)
    return {
        "timely_mbits": delivered_episode,
        "movement_energy_j": energy_episode,
        "energy_efficiency_mbit_per_j": delivered_episode / energy_episode,
        "base_reward": base_episode,
        "feature_contributions": contributions,
        "extra_reward": extra,
        "combined_reward": base_episode + extra,
    }


def ordering_score(
    energy_efficiency: np.ndarray,
    reward: np.ndarray,
    *,
    ee_tolerance: float = DEFAULT_EE_TOLERANCE,
    reward_tolerance: float = DEFAULT_REWARD_TOLERANCE,
    max_examples: int = 5,
) -> dict[str, Any]:
    ee = np.asarray(energy_efficiency, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    if ee.ndim != 1 or reward.shape != ee.shape or ee.size < 2:
        raise ValueError("ordering score requires aligned one-dimensional episode values")
    if not np.all(np.isfinite(ee)) or not np.all(np.isfinite(reward)):
        raise ValueError("ordering score inputs must be finite")
    correct = tied = reversed_count = excluded = 0
    score = 0.0
    examples = []
    # Stream one upper-triangle row at a time.  This vectorizes each bounded
    # chunk without allocating an episode-by-episode matrix.
    for left in range(ee.size - 1):
        right_indices = np.arange(left + 1, ee.size, dtype=np.int64)
        ee_delta = ee[left] - ee[right_indices]
        comparable_mask = np.abs(ee_delta) > float(ee_tolerance)
        excluded += int(np.count_nonzero(~comparable_mask))
        if not np.any(comparable_mask):
            continue
        comparable_right = right_indices[comparable_mask]
        comparable_ee_delta = ee_delta[comparable_mask]
        reward_delta = reward[left] - reward[comparable_right]
        tied_mask = np.abs(reward_delta) <= float(reward_tolerance)
        correct_mask = (~tied_mask) & (
            np.signbit(reward_delta) == np.signbit(comparable_ee_delta)
        )
        reversed_mask = ~(tied_mask | correct_mask)
        tied += int(np.count_nonzero(tied_mask))
        correct += int(np.count_nonzero(correct_mask))
        reversed_count += int(np.count_nonzero(reversed_mask))
        score += float(np.count_nonzero(correct_mask))
        score += 0.5 * float(np.count_nonzero(tied_mask))
        if len(examples) < int(max_examples):
            for local in np.flatnonzero(~correct_mask):
                right = int(comparable_right[local])
                category = "reward_tie" if tied_mask[local] else "reversed"
                examples.append(
                    {
                        "episode_i": left,
                        "episode_j": right,
                        "ee_i": float(ee[left]),
                        "ee_j": float(ee[right]),
                        "reward_i": float(reward[left]),
                        "reward_j": float(reward[right]),
                        "category": category,
                    }
                )
                if len(examples) >= int(max_examples):
                    break
    comparable = correct + tied + reversed_count
    if comparable == 0:
        raise EpisodeSearchError("episode data has no EE-non-tied pairs")
    return {
        "score": float(score / comparable),
        "score_points": float(score),
        "comparable_pair_count": comparable,
        "ee_tie_excluded_pair_count": excluded,
        "correct_count": correct,
        "reward_tie_count": tied,
        "reversed_count": reversed_count,
        "correct_percent": 100.0 * correct / comparable,
        "reward_tie_percent": 100.0 * tied / comparable,
        "reversed_percent": 100.0 * reversed_count / comparable,
        "reward_tie_points": 0.5,
        "ee_tolerance": float(ee_tolerance),
        "reward_tolerance": float(reward_tolerance),
        "examples": examples,
    }


def evaluate_candidates(
    dataset: EpisodeDataset,
    candidates: dict[str, dict[str, Any]],
    features_by_candidate: dict[str, np.ndarray],
    *,
    evaluation_lambda: float,
    beta: float,
    ee_tolerance: float = DEFAULT_EE_TOLERANCE,
    reward_tolerance: float = DEFAULT_REWARD_TOLERANCE,
) -> dict[str, Any]:
    baseline_components = episode_reward_components(
        dataset, evaluation_lambda=evaluation_lambda, beta=beta
    )
    ee = baseline_components["energy_efficiency_mbit_per_j"]
    baseline_score = ordering_score(
        ee,
        baseline_components["base_reward"],
        ee_tolerance=ee_tolerance,
        reward_tolerance=reward_tolerance,
    )
    reports: dict[str, Any] = {}
    for slot in EXPECTED_CANDIDATE_IDS:
        candidate = candidates[slot]
        weights = np.asarray(
            [item["reward_weight"] for item in candidate["features"]], dtype=np.float64
        )
        components = episode_reward_components(
            dataset,
            evaluation_lambda=evaluation_lambda,
            beta=beta,
            features=features_by_candidate[slot],
            weights=weights,
        )
        score = ordering_score(
            ee,
            components["combined_reward"],
            ee_tolerance=ee_tolerance,
            reward_tolerance=reward_tolerance,
        )
        score["strictly_exceeds_baseline"] = bool(score["score"] > baseline_score["score"])
        prioritized_examples: list[list[dict[str, Any]]] = [[], []]
        candidate_reward = components["combined_reward"]
        baseline_reward = baseline_components["base_reward"]
        for left in range(len(dataset.episodes) - 1):
            right_indices = np.arange(left + 1, len(dataset.episodes), dtype=np.int64)
            ee_delta = ee[left] - ee[right_indices]
            comparable = np.abs(ee_delta) > float(ee_tolerance)
            if not np.any(comparable):
                continue
            right_indices = right_indices[comparable]
            ee_delta = ee_delta[comparable]

            def categories(values):
                delta = values[left] - values[right_indices]
                ties = np.abs(delta) <= float(reward_tolerance)
                correct_values = (~ties) & (
                    np.signbit(delta) == np.signbit(ee_delta)
                )
                return ties, correct_values

            candidate_ties, candidate_correct = categories(candidate_reward)
            baseline_ties, baseline_correct = categories(baseline_reward)
            for local in np.flatnonzero(~candidate_correct):
                baseline_category = (
                    "reward_tie" if baseline_ties[local]
                    else "correct" if baseline_correct[local]
                    else "reversed"
                )
                priority = 0 if baseline_category == "correct" else 1
                if len(prioritized_examples[priority]) >= 5:
                    continue
                right = int(right_indices[local])
                candidate_category = (
                    "reward_tie" if candidate_ties[local] else "reversed"
                )
                prioritized_examples[priority].append(
                    {
                        "episode_i": left,
                        "episode_j": right,
                        "ee_i": float(ee[left]),
                        "ee_j": float(ee[right]),
                        "reward_i": float(candidate_reward[left]),
                        "reward_j": float(candidate_reward[right]),
                        "category": candidate_category,
                        "baseline_category": baseline_category,
                        "selection_priority": (
                            "baseline_correct_candidate_incorrect"
                            if priority == 0
                            else "other_candidate_incorrect"
                        ),
                    }
                )
            if all(len(items) >= 5 for items in prioritized_examples):
                break
        score["examples"] = (prioritized_examples[0] + prioritized_examples[1])[:5]
        enriched_examples = []
        for example in score["examples"]:
            left, right = example["episode_i"], example["episode_j"]
            item = copy.deepcopy(example)
            for label, index in (("i", left), ("j", right)):
                episode = dataset.episodes[index]
                item[f"episode_{label}_identity"] = {
                    "source_id": episode.source_id,
                    "episode_id": episode.episode_id,
                    "scenario_id": episode.scenario_id,
                }
                item[f"base_reward_{label}"] = float(components["base_reward"][index])
                item[f"combined_reward_{label}"] = float(components["combined_reward"][index])
                item[f"feature_contributions_{label}"] = [
                    float(value) for value in components["feature_contributions"][index]
                ]
            enriched_examples.append(item)
        score["examples"] = enriched_examples
        score["feature_names"] = [item["name"] for item in candidate["features"]]
        score["feature_weights"] = weights.tolist()
        reports[slot] = {"ordering": score, "episode_components": components}
    eligible = [
        slot for slot in EXPECTED_CANDIDATE_IDS if reports[slot]["ordering"]["strictly_exceeds_baseline"]
    ]
    selected = max(
        eligible,
        key=lambda slot: (reports[slot]["ordering"]["score"], -EXPECTED_CANDIDATE_IDS.index(slot)),
        default=None,
    )
    serializable = {
        "schema_version": "uav-hrl-episode-ordering-evaluation-v1",
        "evaluation_lambda": float(evaluation_lambda),
        "beta": float(beta),
        "episode_count": len(dataset.episodes),
        "pair_contract": {
            "all_unordered_episode_pairs": True,
            "ee_ties_excluded": True,
            "reward_tie_points": 0.5,
            "ee_tolerance": float(ee_tolerance),
            "reward_tolerance": float(reward_tolerance),
        },
        "baseline": baseline_score,
        "candidates": {
            slot: reports[slot]["ordering"] for slot in EXPECTED_CANDIDATE_IDS
        },
        "selected_candidate_id": selected,
        "selection_rule": "highest unrounded score strictly above baseline; stable slot order breaks ties",
        "lipschitz_evaluation_performed": False,
    }
    return {"report": serializable, "components": reports, "selected_candidate_id": selected}


def summarize_training_blocks(
    episode_rows: list[dict[str, Any]],
    *,
    block_size: int = 100,
    expected_episodes: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize non-overlapping episode blocks without changing training metrics."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    episode_numbers = [int(row["episode"]) for row in episode_rows]
    expected_numbers = list(range(1, len(episode_rows) + 1))
    if episode_numbers != expected_numbers:
        raise EpisodeSearchError(
            "LLM training episode metrics are missing, duplicated, or out of order"
        )
    if expected_episodes is not None and len(episode_rows) != int(expected_episodes):
        raise EpisodeSearchError(
            "LLM training episode metrics are incomplete: "
            f"expected 1-{int(expected_episodes)}, received "
            f"{episode_numbers[0] if episode_numbers else 'none'}-"
            f"{episode_numbers[-1] if episode_numbers else 'none'}"
        )
    identities = {
        _canonical_json(
            {
                "candidate_artifact_id": row.get("candidate_artifact_id"),
                "feature_names": row.get("feature_names"),
                "feature_reward_weights": row.get("feature_reward_weights"),
                "beta": row.get("beta"),
            }
        )
        for row in episode_rows
    }
    if len(identities) > 1:
        raise EpisodeSearchError("LLM training episode metrics mix candidate identities")
    if expected_episodes is not None and int(expected_episodes) % block_size:
        raise EpisodeSearchError("expected training horizon is not divisible by block size")
    summaries = []
    for start in range(0, len(episode_rows), block_size):
        block = episode_rows[start : start + block_size]
        if len(block) != block_size:
            raise EpisodeSearchError("LLM training metrics end with an incomplete block")
        feature_names = list(block[0].get("feature_names") or [])
        contributions = np.asarray(
            [row["feature_contribution_sums"] for row in block], dtype=np.float64
        )
        if contributions.shape != (block_size, len(feature_names)):
            raise ValueError("training feature contribution rows are inconsistent")

        def stats(key: str) -> dict[str, float]:
            values = np.asarray([row[key] for row in block], dtype=np.float64)
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}

        summaries.append(
            {
                "episode_range": [int(block[0]["episode"]), int(block[-1]["episode"])],
                "episode_count": block_size,
                "energy_efficiency_mbit_per_j": stats("energy_efficiency_mbit_per_j"),
                "base_reward_sum": stats("base_reward_sum"),
                "extra_reward_sum": stats("extra_reward_sum"),
                "combined_reward_sum": stats("combined_reward_sum"),
                "feature_contributions": {
                    name: {
                        "mean": float(np.mean(contributions[:, index])),
                        "std": float(np.std(contributions[:, index])),
                    }
                    for index, name in enumerate(feature_names)
                },
                "roi_count_distribution": {
                    str(value): sum(int(row.get("roi_count") == value) for row in block)
                    for value in sorted({row.get("roi_count") for row in block})
                },
                "exploration": sorted(
                    {_canonical_json(row.get("exploration")) for row in block}
                ),
                "lambda_semantics": "actual dinkelbach_lambda_used during training",
            }
        )
    return summaries


def parse_candidate_batch(content: str, *, expected_ids: Iterable[str]) -> dict[str, Any]:
    """Parse one envelope, then report each reliably identified slot separately."""

    value, parse_metadata = parse_candidate_json_envelope(content)
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise CandidateError("response must be one object containing only 'candidates'")
    candidates = value["candidates"]
    if not isinstance(candidates, list):
        raise CandidateError("candidates must be an array")
    expected = tuple(expected_ids)
    grouped: dict[str, list[dict[str, Any]]] = {slot: [] for slot in expected}
    batch_errors = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            batch_errors.append(
                {"location": f"$.candidates[{index}]", "error": "candidate must be an object"}
            )
            continue
        slot = item.get("candidate_id")
        if not isinstance(slot, str):
            batch_errors.append(
                {"location": f"$.candidates[{index}].candidate_id", "error": "candidate_id must be a string"}
            )
            continue
        if slot not in expected:
            batch_errors.append(
                {"location": f"$.candidates[{index}].candidate_id", "error": f"unexpected candidate_id {slot!r}"}
            )
            continue
        grouped[slot].append(copy.deepcopy(item))

    slots: dict[str, dict[str, Any]] = {}
    required = {"candidate_id", "design_summary", "features", "code"}
    for slot in expected:
        items = grouped[slot]
        if not items:
            slots[slot] = {
                "submission": {"candidate_id": slot},
                "error": f"required candidate {slot!r} is missing",
            }
            continue
        if len(items) > 1:
            slots[slot] = {
                "submission": {
                    "candidate_id": slot,
                    "duplicate_submissions": items,
                },
                "error": f"candidate_id {slot!r} appears {len(items)} times",
            }
            continue
        item = items[0]
        if set(item) != required:
            missing = sorted(required.difference(item))
            extra = sorted(set(item).difference(required))
            slots[slot] = {
                "submission": item,
                "error": (
                    f"candidate fields must be exactly {sorted(required)}; "
                    f"missing={missing}, extra={extra}"
                ),
            }
            continue
        if not isinstance(item["design_summary"], str) or not item["design_summary"].strip():
            slots[slot] = {
                "submission": item,
                "error": f"{slot}.design_summary must be non-empty",
            }
            continue
        slots[slot] = {"submission": item, "error": None}
    return {
        "slots": slots,
        "batch_errors": batch_errors,
        "parse_metadata": parse_metadata,
    }


def validate_search_candidate(
    submission: dict[str, Any],
    *,
    dataset: EpisodeDataset,
    constants_metadata: dict[str, Any],
    worker_timeout: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], np.ndarray | None]:
    model_form = {"features": submission.get("features"), "code": submission.get("code")}
    try:
        _, candidate, transformation = normalize_candidate_submission(
            model_form, constants_metadata
        )
    except CandidateError as exc:
        return None, {"status": "failed", "stage": "schema", "error": str(exc)}, None
    candidate["candidate_name"] = str(submission["candidate_id"])
    staged = validate_candidate_staged(candidate, constants_metadata)
    if not staged["can_execute"]:
        return candidate, staged, None
    try:
        static = validate_candidate(candidate, constants_metadata)
        obs = build_obs_arrays(dataset.arrays)
        extra, worker_reward, execution = execute_candidate_isolated(
            candidate,
            obs,
            constants_metadata,
            timeout=worker_timeout,
            diagnostic_contract=runtime_diagnostic_contract(
                dataset.fixed_metadata, constants_metadata
            ),
        )
        expected = feature_reward(extra, candidate)
        if not np.allclose(worker_reward, expected, rtol=0.0, atol=1e-12):
            raise CandidateError("worker reward differs from host weighted feature reward")
    except (CandidateError, CandidateExecutionError) as exc:
        return candidate, {"status": "failed", "stage": "execution", "error": str(exc)}, None
    return candidate, {
        "status": "passed",
        "scope": {
            "complete_episode_count": len(dataset.episodes),
            "transition_count": dataset.transition_count,
            "includes_empty_probe": True,
            "lipschitz_evaluation_performed": False,
        },
        "host_transformation": transformation,
        "static": static,
        "execution": execution,
        "candidate_numeric_diagnostics": candidate_numeric_diagnostics(
            dataset.arrays["state"], extra, candidate
        ),
    }, extra


def render_common_prompt(
    *, dataset: EpisodeDataset, constants_metadata: dict[str, Any], beta: float, ranking_spec: str
) -> str:
    minimal = {
        "candidate_id": "candidate_1",
        "design_summary": "Format-only example using one constant state feature.",
        "features": [
            {
                "name": "format_only_zero",
                "description": "A fixed zero in [0,1]; weight zero; format test only.",
                "reward_weight": 0.0,
            }
        ],
        "code": "def compute_extra_state(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)",
    }
    return _replace_template(
        COMMON_TEMPLATE.read_text(encoding="utf-8"),
        {
            "MOVEMENT_AND_ENERGY_SPEC": movement_and_energy_spec(constants_metadata),
            "VISUAL_SENSING_SPEC": visual_sensing_spec(constants_metadata),
            "COMMUNICATION_SPEC": communication_spec(constants_metadata),
            "INPUT_FIELD_TABLE": render_environment_interface(
                dataset.fixed_metadata, constants_metadata, include_system_semantics=False
            ),
            "BASE_REWARD_SPEC": (
                "B_timely_Mbit, movement energy, and the three penalties use the formal "
                "environment definitions above; the host does not expose post-action values in obs."
            ),
            "BETA": format(float(beta), ".17g"),
            "SUPPORTED_OPERATIONS": SUPPORTED_OPERATIONS,
            "MINIMAL_EXECUTABLE_EXAMPLE": json.dumps(minimal, indent=2, ensure_ascii=False),
            "RANKING_EVALUATION_SPEC": ranking_spec,
        },
    )


def render_stage_prompt(common: str, template: Path, replacements: dict[str, Any]) -> str:
    stage = _replace_template(template.read_text(encoding="utf-8"), replacements)
    # ``common`` and replacement values are already rendered data.  Do not
    # recursively reinterpret braces inside candidate code or diagnostics as
    # template placeholders; _replace_template has exhaustively handled the
    # placeholders declared by the stage template itself.
    return common.rstrip() + "\n\n" + stage.strip() + "\n"


def _role_config(**values: Any) -> dict[str, Any]:
    provider = str(values["provider"]).lower()
    if provider not in {"lmstudio", "openai"}:
        raise ValueError(f"unsupported provider: {provider}")
    config = {
        "provider": provider,
        "model": str(values["model"]),
        "base_url": provider_base_url(provider, values.get("base_url")),
        "context_length": int(values["context_length"]),
        "max_output_tokens": int(values["max_output_tokens"]),
        "temperature": float(values["temperature"]),
        "seed": int(values["seed"]),
        "reasoning_effort": values.get("reasoning_effort"),
        "timeout": float(values["timeout"]),
        "connect_timeout": float(values["connect_timeout"]),
        "total_timeout": float(values["total_timeout"]),
        "progress_interval": float(values["progress_interval"]),
    }
    if not config["model"] or config["max_output_tokens"] >= config["context_length"]:
        raise ValueError("model must be set and output tokens must be below context length")
    return config


def _make_client(config: dict[str, Any]):
    cls = OpenAIClient if config["provider"] == "openai" else LMStudioClient
    return cls(
        config["base_url"],
        timeout=config["timeout"],
        connect_timeout=config["connect_timeout"],
        total_timeout=config["total_timeout"],
        progress_interval=config["progress_interval"],
    )


def _call_model(
    *, prompt: str, config: dict[str, Any], client: Any, directory: Path, effective_context: int
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    budget = estimate_token_budget(
        prompt,
        context_length=effective_context,
        max_output_tokens=config["max_output_tokens"],
    )
    _write_json(directory / "token_budget.json", budget)
    _write_json(
        directory / "planned_request.json",
        planned_chat_request(
            config["provider"],
            model=config["model"],
            prompt=prompt,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            seed=config["seed"],
            reasoning_effort=config["reasoning_effort"],
        ),
    )
    if not budget["fits_client_budget"]:
        raise EpisodeSearchError("context_budget_exceeded")
    response = client.chat(
        model=config["model"],
        prompt=prompt,
        temperature=config["temperature"],
        max_output_tokens=config["max_output_tokens"],
        seed=config["seed"],
        attempt_directory=directory,
        reasoning_effort=config["reasoning_effort"],
    )
    _write_json(directory / "response.json", response)
    if not response.get("transport_completed") or response.get("finish_reason") != "stop":
        raise EpisodeSearchError("model response did not complete normally")
    if response.get("tool_calls_seen"):
        raise EpisodeSearchError("episode-search expects plain JSON, not tool calls")
    actual = response.get("actual_model")
    if actual and not response_model_matches(config["provider"], config["model"], str(actual)):
        raise EpisodeSearchError("model response identity does not match the requested model")
    content = response.get("content")
    if not isinstance(content, str) or not content.strip():
        raise EpisodeSearchError("model response has no final JSON content")
    return response


def _public_score_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": report["baseline"],
        "candidates": report["candidates"],
        "selected_candidate_id": report["selected_candidate_id"],
        "selection_rule": report["selection_rule"],
    }


def _evaluation_feedback(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    return {
        key: result.get(key)
        for key in (
            "status",
            "episode_count",
            "roi_count",
            "environment_size_m",
            "scenario_manifest_hash",
            "mean_episode_energy_efficiency_mbit_per_j",
            "std_episode_energy_efficiency_mbit_per_j",
            "baseline_mean_episode_energy_efficiency_mbit_per_j",
            "improves_over_baseline_mean_episode_ee",
            "candidate_metrics",
            "baseline_metrics",
        )
    }


def _best_trained_feedback(best: dict[str, Any] | None) -> dict[str, Any] | None:
    if not best:
        return None
    return {
        "round": best.get("round"),
        "candidate_id": best.get("candidate_id"),
        "candidate_submission": best.get("candidate_submission"),
        "mean_episode_energy_efficiency_mbit_per_j": best.get(
            "mean_episode_energy_efficiency_mbit_per_j"
        ),
        "evaluation": _evaluation_feedback(best.get("evaluation")),
    }


def _save_candidate(
    directory: Path,
    *,
    submission: dict[str, Any],
    candidate: dict[str, Any] | None,
    validation: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "submission.json", submission)
    _write_json(directory / "validation_report.json", validation)
    if candidate is not None:
        _write_json(directory / "candidate.json", candidate)
        (directory / "candidate.py").write_text(candidate["code"], encoding="utf-8")


def _record_batch_parse_failure(
    current: dict[str, Any],
    *,
    expected_slots: Iterable[str],
    raw_content: str,
    error: Exception,
    round_directory: Path,
) -> None:
    """Keep locked slots and make an ambiguous outer response repairable."""

    report = {
        "status": "failed",
        "stage": "json",
        "error": str(error),
        "requirement": "Return one unambiguous JSON object with the requested candidate slots.",
    }
    for slot in expected_slots:
        previous = current["slots"].get(slot) or {}
        submission = {
            "candidate_id": slot,
            "raw_unparsed_response": raw_content,
            "previous_submission": previous.get("submission"),
        }
        version = int(previous.get("version", 0)) + 1
        directory = round_directory / "candidates" / slot / f"version_{version:02d}"
        _save_candidate(
            directory, submission=submission, candidate=None, validation=report
        )
        (directory / "raw_unparsed_response.txt").write_text(
            raw_content, encoding="utf-8"
        )
        current["slots"][slot] = {
            "status": "failed",
            "version": version,
            "submission": submission,
            "candidate": None,
            "validation": copy.deepcopy(report),
            "computation_fingerprint": None,
            "directory": str(directory),
        }


def _apply_candidate_batch(
    current: dict[str, Any],
    *,
    content: str,
    expected_slots: Iterable[str],
    round_directory: Path,
    dataset: EpisodeDataset,
    constants: dict[str, Any],
    worker_timeout: float,
) -> None:
    expected = tuple(expected_slots)
    parsed = parse_candidate_batch(content, expected_ids=expected)
    _write_json(
        round_directory / f"batch_parse_{int(current.get('batch_parse_count', 0)) + 1:02d}.json",
        parsed,
    )
    current["batch_parse_count"] = int(current.get("batch_parse_count", 0)) + 1
    accepted_fingerprints = {
        record["computation_fingerprint"]
        for slot, record in current["slots"].items()
        if slot not in expected and record.get("status") == "validated"
    }
    for slot in expected:
        parsed_slot = parsed["slots"][slot]
        submission = parsed_slot["submission"]
        if parsed_slot["error"] is not None:
            candidate = None
            validation = {
                "status": "failed",
                "stage": "candidate_schema",
                "error": parsed_slot["error"],
                "batch_errors": parsed["batch_errors"],
                "parse_metadata": parsed["parse_metadata"],
            }
            extra = None
        else:
            candidate, validation, extra = validate_search_candidate(
                submission,
                dataset=dataset,
                constants_metadata=constants,
                worker_timeout=worker_timeout,
            )
            validation = {
                **validation,
                "batch_errors": parsed["batch_errors"],
                "parse_metadata": parsed["parse_metadata"],
            }
        fingerprint = candidate_computation_fingerprint(candidate) if candidate else None
        if fingerprint is not None and fingerprint in accepted_fingerprints:
            validation = {
                "status": "failed",
                "stage": "diversity",
                "error": "candidate is semantically identical to another accepted slot",
                "batch_errors": parsed["batch_errors"],
                "parse_metadata": parsed["parse_metadata"],
            }
            extra = None
        if extra is not None:
            accepted_fingerprints.add(fingerprint)
        version = int(current["slots"].get(slot, {}).get("version", 0)) + 1
        candidate_dir = round_directory / "candidates" / slot / f"version_{version:02d}"
        _save_candidate(
            candidate_dir,
            submission=submission,
            candidate=candidate,
            validation=validation,
        )
        if extra is not None:
            np.save(candidate_dir / "features.npy", extra, allow_pickle=False)
        current["slots"][slot] = {
            "status": "validated" if extra is not None else "failed",
            "version": version,
            "submission": submission,
            "candidate": candidate,
            "validation": validation,
            "computation_fingerprint": fingerprint,
            "directory": str(candidate_dir),
        }
        _progress(f"{slot}: {current['slots'][slot]['status']} (version {version})")


def _result(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state["status"],
        "output_directory": state["output_directory"],
        "search_round": state["search_round"],
        "model_calls": state["model_calls"],
        "best_trained_candidate": state.get("best_trained_candidate"),
        "stop_reason": state.get("stop_reason"),
    }


def _progress(message: str) -> None:
    print(f"[llm-episode-search] {message}", flush=True)


def _revision_transition_allows_resume(
    state: dict[str, Any], current_git_sha: str
) -> bool:
    if state.get("git_sha") == current_git_sha:
        return True
    transition = state.get("numeric_operation_rules_transition") or {}
    numeric_transition = bool(
        transition.get("source_git_sha") == state.get("git_sha")
        and transition.get("target_git_sha") == current_git_sha
        and transition.get("target_rules_version")
        == NUMERIC_OPERATION_RULES_VERSION
        and transition.get("status") in {"passed", "completed_with_failures"}
    )
    training_transition = state.get("training_initialization_recovery_transition") or {}
    initialization_transition = bool(
        training_transition.get("source_git_sha") == state.get("git_sha")
        and training_transition.get("target_git_sha") == current_git_sha
        and training_transition.get("status") == "compatible"
        and training_transition.get("reason")
        == "short_training_path_and_initialization_recovery"
    )
    restart_transition = state.get("checkpoint_path_restart_transition") or {}
    return numeric_transition or initialization_transition or bool(
        restart_transition.get("source_git_sha")
        == RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION
        and restart_transition.get("target_git_sha") == current_git_sha
        and restart_transition.get("status") == "compatible"
        and restart_transition.get("reason")
        == "checkpoint_path_shortening_restart_from_episode_1"
    )


def _short_training_output_root(search_output: Path, round_number: int) -> Path:
    """Return a collision-resistant root with headroom for Windows checkpoints."""

    digest = hashlib.sha256(
        (
            f"{Path(search_output).resolve()}|round={int(round_number)}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    return (ROOT / "results" / "t" / digest).resolve()


def _training_recovery_eligibility(
    state: dict[str, Any], *, current_git_sha: str
) -> None:
    if state.get("git_sha") not in RECOVERABLE_TRAINING_INITIALIZATION_SOURCE_REVISIONS:
        raise EpisodeSearchError(
            "saved Git revision is not eligible for the bounded training-initialization recovery"
        )
    if state.get("git_sha") == current_git_sha:
        raise EpisodeSearchError(
            "training-initialization recovery is unnecessary at the current Git revision"
        )
    current = state.get("current_round")
    if (
        not isinstance(current, dict)
        or current.get("phase") != "train"
        or current.get("training") is not None
        or current.get("evaluation_result") is not None
        or state.get("rounds")
    ):
        raise EpisodeSearchError(
            "bounded recovery requires an untrained first-round candidate at the train phase"
        )
    selected = current.get("selected_candidate_id")
    evaluation = current.get("evaluation") or {}
    if (
        selected not in EXPECTED_CANDIDATE_IDS
        or evaluation.get("selected_candidate_id") != selected
        or selected not in (current.get("slots") or {})
        or not current.get("approved_artifact")
    ):
        raise EpisodeSearchError(
            "training recovery selection or approved artifact record is incomplete"
        )


def _record_training_recovery_transition(
    state: dict[str, Any],
    *,
    output: Path,
    dataset: EpisodeDataset,
    constants: dict[str, Any],
    baseline_preflight: dict[str, Any],
    target_git_sha: str,
) -> dict[str, Any]:
    """Bind one ef10fa3 training-stage run to the path/recovery update."""

    current = state["current_round"]
    selected = current["selected_candidate_id"]
    slot = current["slots"][selected]
    evaluation = current["evaluation"]
    evaluation_path = output / f"round_{int(current['round']):02d}" / "pretraining_evaluation.json"
    if not evaluation_path.is_file():
        raise EpisodeSearchError("saved pretraining evaluation report is missing")
    saved_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if saved_evaluation != evaluation:
        raise EpisodeSearchError("saved pretraining evaluation report changed")
    design = load_approved_design(current["approved_artifact"])
    if design.candidate != slot.get("candidate"):
        raise EpisodeSearchError(
            "approved artifact candidate differs from the selected saved candidate"
        )
    if design.constants_metadata != constants:
        raise EpisodeSearchError(
            "approved artifact observation/constants interface changed"
        )
    provenance = design.artifact.get("provenance") or {}
    if (
        Path(provenance.get("search_run", "")).resolve() != output
        or int(provenance.get("search_round", -1)) != int(current["round"])
        or provenance.get("candidate_slot") != selected
        or (provenance.get("dataset") or {}) != dataset.provenance
    ):
        raise EpisodeSearchError("approved artifact search provenance is incompatible")
    if state.get("baseline_preflight") != baseline_preflight:
        raise EpisodeSearchError("baseline/pre-evaluation comparison contract changed")
    source_git_sha = str(state["git_sha"])
    training_root = _short_training_output_root(output, int(current["round"]))
    record = {
        "schema_version": "uav-hrl-llm-episode-search-training-recovery-v1",
        "status": "compatible",
        "reason": "short_training_path_and_initialization_recovery",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_sha": source_git_sha,
        "target_git_sha": target_git_sha,
        "search_run": str(output),
        "round": int(current["round"]),
        "selected_candidate_id": selected,
        "candidate_version": int(slot["version"]),
        "approved_artifact_identity": artifact_identity(design),
        "approved_artifact": str(design.directory),
        "dataset_content_sha256": dataset.provenance.get("dataset_content_sha256"),
        "pretraining_evaluation_sha256": _sha256_json(evaluation),
        "baseline_preflight_sha256": _sha256_json(baseline_preflight),
        "training_output_root": str(training_root),
        "legacy_training_output_root": str(
            output / f"round_{int(current['round']):02d}" / "training"
        ),
        "model_calls": int(state.get("model_calls", 0)),
        "repair_calls": int(current.get("repair_calls", 0)),
    }
    current["training_output_root"] = str(training_root)
    current["legacy_training_output_root"] = record[
        "legacy_training_output_root"
    ]
    state["training_initialization_recovery_transition"] = record
    state.setdefault("training_initialization_recoveries", []).append(record)
    return record


def _checkpoint_path_restart_eligibility(
    state: dict[str, Any], *, current_git_sha: str
) -> None:
    previous = state.get("training_initialization_recovery_transition") or {}
    current = state.get("current_round")
    training = (current or {}).get("training") or {}
    if (
        previous.get("source_git_sha") != state.get("git_sha")
        or previous.get("target_git_sha")
        != RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION
        or previous.get("status") != "compatible"
    ):
        raise EpisodeSearchError(
            "checkpoint-path restart requires the recorded fa948f1 initialization recovery"
        )
    if current_git_sha == RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION:
        raise EpisodeSearchError(
            "checkpoint-path restart requires the path-preflight fix revision"
        )
    if state.get("checkpoint_path_restart_transition") is not None:
        raise EpisodeSearchError("checkpoint-path restart was already recorded")
    if (
        not isinstance(current, dict)
        or current.get("phase") != "train"
        or current.get("evaluation_result") is not None
        or state.get("rounds")
        or current.get("selected_candidate_id") != "candidate_1"
        or training.get("status") != "incomplete"
        or not training.get("run_directory")
    ):
        raise EpisodeSearchError(
            "checkpoint-path restart requires the saved first-round failed training state"
        )


def _completed_episode_evidence(run_directory: Path) -> dict[str, Any]:
    evidence = {}
    for filename in (
        "training_history.jsonl",
        "llm_training_episode_metrics.jsonl",
    ):
        path = run_directory / filename
        rows = []
        if path.is_file():
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, json.JSONDecodeError) as exc:
                raise EpisodeSearchError(
                    f"failed training evidence is unreadable: {path}: {exc}"
                ) from exc
        episodes = [int(row["episode"]) for row in rows if "episode" in row]
        evidence[filename] = {
            "row_count": len(rows),
            "maximum_episode": max(episodes) if episodes else None,
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
    return evidence


def _record_checkpoint_path_restart_transition(
    state: dict[str, Any],
    *,
    output: Path,
    dataset: EpisodeDataset,
    constants: dict[str, Any],
    baseline_preflight: dict[str, Any],
    target_git_sha: str,
) -> dict[str, Any]:
    """Authorize one explicit episode-1 restart for the fa948f1 path failure."""

    from llm_episode_training import _training_run_evidence

    current = state["current_round"]
    selected = current["selected_candidate_id"]
    slot = current["slots"][selected]
    evaluation = current["evaluation"]
    evaluation_path = (
        output
        / f"round_{int(current['round']):02d}"
        / "pretraining_evaluation.json"
    )
    if not evaluation_path.is_file() or json.loads(
        evaluation_path.read_text(encoding="utf-8")
    ) != evaluation:
        raise EpisodeSearchError("saved pretraining evaluation report changed")
    design = load_approved_design(current["approved_artifact"])
    if design.candidate != slot.get("candidate"):
        raise EpisodeSearchError(
            "approved artifact candidate differs from the selected saved candidate"
        )
    if design.constants_metadata != constants:
        raise EpisodeSearchError(
            "approved artifact observation/constants interface changed"
        )
    provenance = design.artifact.get("provenance") or {}
    if (
        Path(provenance.get("search_run", "")).resolve() != output
        or int(provenance.get("search_round", -1)) != int(current["round"])
        or provenance.get("candidate_slot") != selected
        or (provenance.get("dataset") or {}) != dataset.provenance
    ):
        raise EpisodeSearchError("approved artifact search provenance is incompatible")
    if state.get("baseline_preflight") != baseline_preflight:
        raise EpisodeSearchError("baseline/pre-evaluation comparison contract changed")

    failed_run = Path(current["training"]["run_directory"]).resolve()
    failed_evidence = _training_run_evidence(failed_run)
    load_run_artifact(failed_run, artifact_identity(design))
    failure_text = _canonical_json(
        failed_evidence.get("lifecycle_exception") or {}
    )
    if (
        failed_evidence.get("classification") != "progress_without_checkpoint"
        or not failed_evidence.get("has_training_progress")
        or failed_evidence.get("full_checkpoint_directories")
        or "WinError 3" not in failure_text
        or "checkpoints" not in failure_text
    ):
        raise EpisodeSearchError(
            "failed run is not the eligible checkpoint-path progress-without-checkpoint failure"
        )
    previous = state["training_initialization_recovery_transition"]
    previous_root = Path(previous["training_output_root"]).resolve()
    if failed_run.parent.parent != previous_root:
        raise EpisodeSearchError(
            "failed run is outside the recorded fa948f1 training root"
        )

    new_root = _short_training_output_root(output, int(current["round"]))
    prior_roots = []
    for value in (
        current.get("training_output_root"),
        current.get("legacy_training_output_root"),
    ):
        if value and str(Path(value).resolve()) not in prior_roots:
            prior_roots.append(str(Path(value).resolve()))
    record = {
        "schema_version": "uav-hrl-llm-episode-search-checkpoint-path-restart-v1",
        "status": "compatible",
        "reason": "checkpoint_path_shortening_restart_from_episode_1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_search_git_sha": str(state["git_sha"]),
        "source_git_sha": RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION,
        "target_git_sha": target_git_sha,
        "previous_recovery": previous,
        "search_run": str(output),
        "round": int(current["round"]),
        "selected_candidate_id": selected,
        "candidate_version": int(slot["version"]),
        "approved_artifact_identity": artifact_identity(design),
        "dataset_content_sha256": dataset.provenance.get(
            "dataset_content_sha256"
        ),
        "pretraining_evaluation_sha256": _sha256_json(evaluation),
        "baseline_preflight_sha256": _sha256_json(baseline_preflight),
        "failed_run_directory": str(failed_run),
        "failed_run_evidence": failed_evidence,
        "completed_episode_evidence": _completed_episode_evidence(failed_run),
        "prior_training_output_roots": prior_roots,
        "replacement_training_output_root": str(new_root),
        "restart_episode": 1,
        "model_calls": int(state.get("model_calls", 0)),
        "repair_calls": int(current.get("repair_calls", 0)),
    }
    current["prior_training_output_roots"] = prior_roots
    current["training_output_root"] = str(new_root)
    state["checkpoint_path_restart_transition"] = record
    state.setdefault("checkpoint_path_restart_transitions", []).append(record)
    return record


def _next_revalidation_directory(output: Path) -> tuple[int, Path]:
    root = output / "revalidations"
    index = 1
    while (root / f"revalidation_{index:02d}").exists():
        index += 1
    return index, root / f"revalidation_{index:02d}"


def revalidate_episode_search(
    resume: str | Path,
    *,
    reason: str = "numeric_operation_rules_update",
) -> dict[str, Any]:
    """Revalidate saved generation-stage candidates without model/training work."""

    output = Path(resume).resolve()
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SEARCH_RUN_SCHEMA_VERSION:
        raise EpisodeSearchError("revalidation requires a compatible episode-search run")
    current = state.get("current_round")
    if (
        not isinstance(current, dict)
        or current.get("phase") not in {"generate", "repair"}
        or state.get("rounds")
        or current.get("evaluation") is not None
        or current.get("training") is not None
        or current.get("evaluation_result") is not None
    ):
        raise EpisodeSearchError(
            "revalidation is limited to an untrained generation/repair-stage search"
        )
    source_git_sha = str(state.get("git_sha") or "")
    target_git_sha = _git_sha()
    prior_transition = state.get("numeric_operation_rules_transition") or {}
    source_is_supported = (
        source_git_sha == target_git_sha
        or source_git_sha in REVALIDATABLE_NUMERIC_RULE_SOURCE_REVISIONS
        or (
            prior_transition.get("source_git_sha") == source_git_sha
            and prior_transition.get("target_git_sha") == target_git_sha
            and prior_transition.get("target_rules_version")
            == NUMERIC_OPERATION_RULES_VERSION
        )
    )
    if not source_is_supported:
        raise EpisodeSearchError(
            "saved Git revision is not eligible for the bounded numeric-operation-rules revalidation"
        )
    if set(current.get("slots") or {}) != set(EXPECTED_CANDIDATE_IDS):
        raise EpisodeSearchError("revalidation requires all four saved candidate slots")

    settings = state.get("settings") or {}
    dataset = load_complete_episode_dataset(settings.get("episode_sources") or [])
    if state.get("dataset_provenance") != dataset.provenance:
        raise EpisodeSearchError("revalidation episode dataset content is incompatible")
    saved_dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    if saved_dataset != dataset.provenance:
        raise EpisodeSearchError("saved dataset provenance file is incompatible")
    constants = build_constants(dataset.fixed_metadata)
    saved_constants = json.loads((output / "constants.json").read_text(encoding="utf-8"))
    if saved_constants != constants:
        raise EpisodeSearchError("saved observation/constants interface is incompatible")

    from llm_episode_training import validate_baseline_preflight

    baseline_preflight = validate_baseline_preflight(
        baseline_run=settings.get("baseline_run"),
        baseline_evaluation=settings.get("baseline_evaluation"),
        manifest=settings.get("evaluation_manifest"),
        checkpoint_episode=1500,
        episodes=int(settings.get("evaluation_episodes", 0)),
        roi_count=int(settings.get("evaluation_roi_count", 0)),
        environment_size_m=tuple(settings.get("evaluation_area_m") or ()),
        episode_seconds=60,
    )
    saved_preflight = json.loads(
        (output / "baseline_preflight.json").read_text(encoding="utf-8")
    )
    if state.get("baseline_preflight") != baseline_preflight or saved_preflight != baseline_preflight:
        raise EpisodeSearchError("baseline/pre-evaluation comparison contract changed")

    worker_timeout = float(settings["worker_timeout"])
    validated: dict[str, dict[str, Any]] = {}
    accepted_fingerprints: set[str] = set()
    for slot in EXPECTED_CANDIDATE_IDS:
        old = current["slots"][slot]
        submission = copy.deepcopy(old.get("submission"))
        if not isinstance(submission, dict):
            raise EpisodeSearchError(f"{slot} has no saved candidate submission")
        candidate, validation, extra = validate_search_candidate(
            submission,
            dataset=dataset,
            constants_metadata=constants,
            worker_timeout=worker_timeout,
        )
        if not isinstance(old.get("candidate"), dict) or candidate != old["candidate"]:
            raise EpisodeSearchError(
                f"{slot} normalized candidate content changed; this bounded operation-rules migration cannot continue"
            )
        fingerprint = candidate_computation_fingerprint(candidate) if candidate else None
        if extra is not None and fingerprint in accepted_fingerprints:
            validation = {
                "status": "failed",
                "stage": "diversity",
                "error": "candidate is semantically identical to another accepted slot",
            }
            extra = None
        if extra is not None and fingerprint is not None:
            accepted_fingerprints.add(fingerprint)
        validated[slot] = {
            "old": old,
            "submission": submission,
            "candidate": candidate,
            "validation": validation,
            "features": extra,
            "computation_fingerprint": fingerprint,
        }

    revalidation_index, revalidation_dir = _next_revalidation_directory(output)
    slot_records: dict[str, Any] = {}
    for slot, record in validated.items():
        slot_dir = revalidation_dir / "slots" / slot
        _save_candidate(
            slot_dir,
            submission=record["submission"],
            candidate=record["candidate"],
            validation=record["validation"],
        )
        if record["features"] is not None:
            np.save(slot_dir / "features.npy", record["features"], allow_pickle=False)
        old = record["old"]
        slot_records[slot] = {
            "candidate_version": int(old.get("version", 0)),
            "source_directory": old.get("directory"),
            "source_status": old.get("status"),
            "revalidated_status": (
                "validated" if record["features"] is not None else "failed"
            ),
            "directory": str(slot_dir),
        }
        current["slots"][slot] = {
            **old,
            "status": "validated" if record["features"] is not None else "failed",
            "submission": record["submission"],
            "candidate": record["candidate"],
            "validation": record["validation"],
            "computation_fingerprint": record["computation_fingerprint"],
            "directory": str(slot_dir),
            "revalidation": {
                "record": str(revalidation_dir / "revalidation.json"),
                "source_directory": old.get("directory"),
                "candidate_version_unchanged": True,
            },
        }

    all_passed = all(
        current["slots"][slot]["status"] == "validated"
        for slot in EXPECTED_CANDIDATE_IDS
    )
    report = {
        "schema_version": "uav-hrl-llm-episode-search-revalidation-v1",
        "status": "passed" if all_passed else "completed_with_failures",
        "reason": reason,
        "source_git_sha": source_git_sha,
        "target_git_sha": target_git_sha,
        "source_rules_version": state.get("numeric_operation_rules_version", "legacy-v1"),
        "target_rules_version": NUMERIC_OPERATION_RULES_VERSION,
        "dataset_content_sha256": dataset.provenance.get("dataset_content_sha256"),
        "baseline_preflight": baseline_preflight,
        "model_calls_before": int(state.get("model_calls", 0)),
        "model_calls_after": int(state.get("model_calls", 0)),
        "repair_calls_before": int(current.get("repair_calls", 0)),
        "repair_calls_after": int(current.get("repair_calls", 0)),
        "slots": slot_records,
    }
    _write_json(revalidation_dir / "revalidation.json", report)
    transition = {
        "status": report["status"],
        "reason": reason,
        "source_git_sha": source_git_sha,
        "target_git_sha": target_git_sha,
        "source_rules_version": report["source_rules_version"],
        "target_rules_version": NUMERIC_OPERATION_RULES_VERSION,
        "record": str(revalidation_dir / "revalidation.json"),
    }
    state["numeric_operation_rules_version"] = NUMERIC_OPERATION_RULES_VERSION
    state["numeric_operation_rules_transition"] = transition
    state.setdefault("revalidations", []).append(transition)
    if all_passed:
        current["phase"] = "preevaluate"
        state["status"] = "paused_revalidated_ready"
        state["stop_reason"] = "all saved candidates passed; resume to enter pre-evaluation"
    else:
        current["phase"] = "repair"
        exhausted = int(current.get("repair_calls", 0)) >= int(
            settings.get("max_repairs_per_round", 0)
        )
        state["status"] = (
            "paused_repairs_exhausted" if exhausted else "paused_revalidation_failed"
        )
        state["stop_reason"] = "one or more saved candidates still fail validation"
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(state_path, state)
    return _result(state)


def run_episode_search(
    *,
    episode_sources: Iterable[str | Path] | None = None,
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
    evaluation_lambda: float | None = None,
    lambda_training_run: str | Path | None = None,
    max_search_rounds: int | None = None,
    max_repairs_per_round: int | None = None,
    ee_tolerance: float = DEFAULT_EE_TOLERANCE,
    reward_tolerance: float = DEFAULT_REWARD_TOLERANCE,
    baseline_run: str | Path | None = None,
    baseline_evaluation: str | Path | None = None,
    evaluation_manifest: str | Path | None = None,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    revalidate_only: bool = False,
    recover_training_initialization: bool = False,
    restart_failed_training: bool = False,
    dry_run: bool = False,
    client: Any | None = None,
    training_runner: Callable[..., dict[str, Any]] | None = None,
    evaluation_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or resume the complete four-candidate search state machine.

    Test callers inject training/evaluation runners. Formal callers use the
    subprocess wrappers in ``llm_episode_training.py``.
    """

    if sum(
        bool(value)
        for value in (
            revalidate_only,
            recover_training_initialization,
            restart_failed_training,
        )
    ) > 1:
        raise ValueError(
            "revalidation and training recovery modes are mutually exclusive"
        )
    if revalidate_only:
        if resume is None:
            raise ValueError("--revalidate-only requires --resume")
        if max_search_rounds is not None or max_repairs_per_round is not None:
            raise ValueError(
                "revalidate-only does not change search or repair budgets; change them on a later resume"
            )
        return revalidate_episode_search(resume)

    if recover_training_initialization and resume is None:
        raise ValueError("--recover-training-initialization requires --resume")
    if recover_training_initialization and dry_run:
        raise ValueError(
            "training-initialization recovery cannot be combined with dry-run"
        )
    if restart_failed_training and resume is None:
        raise ValueError("--restart-failed-training requires --resume")
    if restart_failed_training and dry_run:
        raise ValueError("failed-training restart cannot be combined with dry-run")

    resume_terminal_no_extension = False
    pending_training_recovery = False
    pending_checkpoint_path_restart = False
    if resume is not None:
        output = Path(resume).resolve()
        state_path = output / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SEARCH_RUN_SCHEMA_VERSION:
            raise EpisodeSearchError(
                "resume contract is incompatible; reviewer-design runs cannot be resumed as episode searches"
            )
        current_git_sha = _git_sha()
        if not _revision_transition_allows_resume(state, current_git_sha):
            if restart_failed_training:
                _checkpoint_path_restart_eligibility(
                    state, current_git_sha=current_git_sha
                )
                pending_checkpoint_path_restart = True
            elif recover_training_initialization:
                _training_recovery_eligibility(
                    state, current_git_sha=current_git_sha
                )
                pending_training_recovery = True
            else:
                raise EpisodeSearchError("resume git revision is incompatible")
        elif recover_training_initialization or restart_failed_training:
            raise EpisodeSearchError(
                "the requested bounded training recovery is not applicable to this run"
            )
        settings = state["settings"]
        episode_sources = settings["episode_sources"]
        config = settings["model"]
        beta = float(settings["beta"])
        worker_timeout = float(settings["worker_timeout"])
        lambda_record = settings["evaluation_lambda"]
        saved_search_rounds = int(settings["max_search_rounds"])
        saved_repairs = int(settings["max_repairs_per_round"])
        requested_search_rounds = (
            saved_search_rounds
            if max_search_rounds is None
            else int(max_search_rounds)
        )
        requested_repairs = (
            saved_repairs
            if max_repairs_per_round is None
            else int(max_repairs_per_round)
        )
        if requested_search_rounds < saved_search_rounds:
            raise EpisodeSearchError("resume cannot lower max_search_rounds")
        if requested_repairs < saved_repairs:
            raise EpisodeSearchError("resume cannot lower max_repairs_per_round")
        max_search_rounds = requested_search_rounds
        max_repairs_per_round = requested_repairs
        settings["max_search_rounds"] = max_search_rounds
        settings["max_repairs_per_round"] = max_repairs_per_round
        ee_tolerance = float(settings["ee_tolerance"])
        reward_tolerance = float(settings["reward_tolerance"])
        baseline_run = settings.get("baseline_run")
        baseline_evaluation = settings.get("baseline_evaluation")
        evaluation_manifest = settings.get("evaluation_manifest")
        train_episodes = int(settings["train_episodes"])
        evaluation_episodes = int(settings["evaluation_episodes"])
        if (
            max_search_rounds > saved_search_rounds
            and state.get("current_round") is None
            and state.get("rounds")
        ):
            previous = state["rounds"][-1]
            next_round = int(previous["round"]) + 1
            state["search_round"] = next_round
            if previous.get("phase") == "completed_trained":
                next_phase = "generate_from_training"
                extra_round_state = {}
            elif previous.get("phase") == "completed_no_training":
                next_phase = "generate_from_preevaluation"
                extra_round_state = {
                    "previous_preevaluation": previous["evaluation"]
                }
            else:
                raise EpisodeSearchError(
                    "completed search cannot be extended from its saved phase"
                )
            state["current_round"] = {
                "round": next_round,
                "phase": next_phase,
                "repair_calls": 0,
                "slots": {},
                "evaluation": None,
                "training": None,
                "evaluation_result": None,
                **extra_round_state,
            }
        elif state.get("current_round") is None and state.get("rounds"):
            resume_terminal_no_extension = True
        if not resume_terminal_no_extension:
            state["status"] = "running"
            state["stop_reason"] = None
    else:
        if not episode_sources or not provider or not model:
            raise ValueError("new episode searches require sources, provider, and model")
        max_search_rounds = (
            DEFAULT_MAX_SEARCH_ROUNDS
            if max_search_rounds is None
            else int(max_search_rounds)
        )
        max_repairs_per_round = (
            DEFAULT_MAX_REPAIRS_PER_ROUND
            if max_repairs_per_round is None
            else int(max_repairs_per_round)
        )
        if max_search_rounds <= 0 or max_repairs_per_round < 0:
            raise ValueError("search rounds must be positive and repairs non-negative")
        if int(train_episodes) != 1500 or int(evaluation_episodes) != 100:
            raise ValueError("formal episode-search contract requires 1500 training and 100 evaluation episodes")
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
        lambda_record = resolve_evaluation_lambda(
            explicit_lambda=evaluation_lambda, training_run=lambda_training_run
        )
        output = allocate_design_directory(
            _slug(model), output_root=output_root, output_dir=output_dir
        )
        now = datetime.now(timezone.utc).isoformat()
        state = {
            "schema_version": SEARCH_RUN_SCHEMA_VERSION,
            "prompt_version": SEARCH_PROMPT_VERSION,
            "status": "running",
            "phase": "generate",
            "created_at_utc": now,
            "updated_at_utc": now,
            "git_sha": _git_sha(),
            "numeric_operation_rules_version": NUMERIC_OPERATION_RULES_VERSION,
            "output_directory": str(output),
            "search_round": 1,
            "model_calls": 0,
            "rounds": [],
            "current_round": None,
            "best_trained_candidate": None,
            "stop_reason": None,
            "settings": {
                "episode_sources": [str(Path(value).resolve()) for value in episode_sources],
                "model": config,
                "beta": float(beta),
                "worker_timeout": float(worker_timeout),
                "evaluation_lambda": lambda_record,
                "max_search_rounds": int(max_search_rounds),
                "max_repairs_per_round": int(max_repairs_per_round),
                "ee_tolerance": float(ee_tolerance),
                "reward_tolerance": float(reward_tolerance),
                "baseline_run": str(Path(baseline_run).resolve()) if baseline_run else None,
                "baseline_evaluation": str(Path(baseline_evaluation).resolve()) if baseline_evaluation else None,
                "evaluation_manifest": str(Path(evaluation_manifest).resolve()) if evaluation_manifest else None,
                "train_episodes": int(train_episodes),
                "evaluation_episodes": int(evaluation_episodes),
                "evaluation_roi_count": 8,
                "evaluation_area_m": [1000.0, 1000.0],
            },
        }
        _write_json(output / "state.json", state)

    if resume_terminal_no_extension:
        return _result(state)

    dataset = load_complete_episode_dataset(episode_sources)
    if state.get("dataset_provenance") not in (None, dataset.provenance):
        raise EpisodeSearchError("resume episode dataset content is incompatible")
    state["dataset_provenance"] = dataset.provenance
    constants = build_constants(dataset.fixed_metadata)
    ranking_spec = (
        "All unordered complete-episode pairs are used. EE ties within the saved tolerance are excluded. "
        "For each remaining pair, matching reward order receives 1 point, a reward tie receives 0.5, "
        "and reversed order receives 0. The score is points divided by EE-non-tied pair count."
    )
    common = render_common_prompt(
        dataset=dataset, constants_metadata=constants, beta=beta, ranking_spec=ranking_spec
    )
    baseline_preflight = None
    if not dry_run:
        if not evaluation_manifest or not baseline_run or not baseline_evaluation:
            raise EpisodeSearchError(
                "formal search requires explicit baseline run, evaluation, and manifest"
            )
        from llm_episode_training import validate_baseline_preflight

        try:
            baseline_preflight = validate_baseline_preflight(
                baseline_run=baseline_run,
                baseline_evaluation=baseline_evaluation,
                manifest=evaluation_manifest,
                checkpoint_episode=1500,
                episodes=evaluation_episodes,
                roi_count=8,
                environment_size_m=(1000.0, 1000.0),
                episode_seconds=60,
            )
        except Exception as exc:
            failure = {
                "schema_version": "uav-hrl-episode-search-baseline-preflight-v1",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            state["status"] = "failed_baseline_preflight"
            state["stop_reason"] = f"{type(exc).__name__}: {exc}"
            state["baseline_preflight_failure"] = failure
            _write_json(output / "baseline_preflight.json", failure)
            _write_json(output / "state.json", state)
            raise EpisodeSearchError(
                f"baseline preflight failed: {type(exc).__name__}: {exc}"
            ) from exc
        if state.get("baseline_preflight") not in (None, baseline_preflight):
            raise EpisodeSearchError("baseline comparison inputs changed since preflight")
        state["baseline_preflight"] = baseline_preflight
        _write_json(output / "baseline_preflight.json", baseline_preflight)
    else:
        state["baseline_preflight"] = {
            "status": "not_run_dry_run",
            "reason": "formal baseline inputs are optional during prompt-only dry-run",
        }
    if pending_training_recovery:
        _record_training_recovery_transition(
            state,
            output=output,
            dataset=dataset,
            constants=constants,
            baseline_preflight=baseline_preflight,
            target_git_sha=_git_sha(),
        )
        _write_json(output / "state.json", state)
    if pending_checkpoint_path_restart:
        _record_checkpoint_path_restart_transition(
            state,
            output=output,
            dataset=dataset,
            constants=constants,
            baseline_preflight=baseline_preflight,
            target_git_sha=_git_sha(),
        )
        _write_json(output / "state.json", state)
    model_client = None if dry_run else (client or _make_client(config))
    inventory = (
        None
        if dry_run or config["provider"] == "openai"
        else model_inventory_summary(model_client.list_models(), config["model"])
    )
    effective_context, context_record = _effective_context_budget(
        config["context_length"], inventory
    )
    state["model_inventory"] = inventory
    state["context"] = context_record
    _write_json(output / "dataset.json", dataset.provenance)
    _write_json(output / "constants.json", constants)
    _write_json(output / "state.json", state)

    if dry_run:
        prompt = render_stage_prompt(common, INITIAL_TEMPLATE, {})
        preview = output / "dry_run"
        preview.mkdir(exist_ok=True)
        (preview / "prompt.txt").write_text(prompt, encoding="utf-8")
        budget = estimate_token_budget(
            prompt,
            context_length=effective_context,
            max_output_tokens=config["max_output_tokens"],
        )
        _write_json(preview / "token_budget.json", budget)
        state["status"] = "dry_run_complete"
        _write_json(output / "state.json", state)
        return _result(state)

    if training_runner is None or evaluation_runner is None:
        from llm_episode_training import run_candidate_evaluation, run_candidate_training

        training_runner = training_runner or run_candidate_training
        evaluation_runner = evaluation_runner or run_candidate_evaluation

    while state["search_round"] <= max_search_rounds:
        round_number = int(state["search_round"])
        round_dir = output / f"round_{round_number:02d}"
        round_dir.mkdir(exist_ok=True)
        current = state.get("current_round")
        if current is None:
            current = {
                "round": round_number,
                "phase": "generate",
                "repair_calls": 0,
                "slots": {},
                "evaluation": None,
                "training": None,
                "evaluation_result": None,
            }
            state["current_round"] = current
            _write_json(output / "state.json", state)

        _progress(
            f"round {round_number}/{max_search_rounds}; phase={current['phase']}; "
            f"repairs={current['repair_calls']}/{max_repairs_per_round}; "
            f"remaining={max(0, max_repairs_per_round - current['repair_calls'])}"
        )

        if current["phase"] in {"generate", "repair"}:
            failed = [
                slot for slot in EXPECTED_CANDIDATE_IDS
                if current["slots"].get(slot, {}).get("status") != "validated"
            ]
            if current["phase"] == "generate":
                prompt = render_stage_prompt(common, INITIAL_TEMPLATE, {})
                expected = EXPECTED_CANDIDATE_IDS
            else:
                if current["repair_calls"] >= max_repairs_per_round:
                    state["status"] = "paused_repairs_exhausted"
                    state["stop_reason"] = "four executable candidates were not obtained"
                    _write_json(output / "state.json", state)
                    return _result(state)
                prompt = render_stage_prompt(
                    common,
                    REPAIR_TEMPLATE,
                    {
                        "FAILED_CANDIDATE_IDS": json.dumps(failed),
                        "FAILED_CANDIDATES": json.dumps(
                            {slot: current["slots"][slot]["submission"] for slot in failed},
                            indent=2,
                        ),
                        "VALIDATION_FEEDBACK": json.dumps(
                            {slot: current["slots"][slot]["validation"] for slot in failed},
                            indent=2,
                        ),
                    },
                )
                expected = tuple(failed)
                current["repair_calls"] += 1
            call_number = state["model_calls"] + 1
            state["model_calls"] = call_number
            _write_json(output / "state.json", state)
            try:
                response = _call_model(
                    prompt=prompt,
                    config=config,
                    client=model_client,
                    directory=round_dir / f"model_call_{call_number:03d}",
                    effective_context=effective_context,
                )
            except Exception as exc:
                state["status"] = "paused_model_call_failed"
                state["stop_reason"] = f"{type(exc).__name__}: {exc}"
                _write_json(output / "state.json", state)
                return _result(state)
            try:
                _apply_candidate_batch(
                    current,
                    content=response["content"],
                    expected_slots=expected,
                    round_directory=round_dir,
                    dataset=dataset,
                    constants=constants,
                    worker_timeout=worker_timeout,
                )
            except CandidateError as exc:
                _record_batch_parse_failure(
                    current,
                    expected_slots=expected,
                    raw_content=response["content"],
                    error=exc,
                    round_directory=round_dir,
                )
                current["phase"] = "repair"
                _write_json(output / "state.json", state)
                continue
            failed = [slot for slot in EXPECTED_CANDIDATE_IDS if current["slots"][slot]["status"] != "validated"]
            current["phase"] = "repair" if failed else "preevaluate"
            _write_json(output / "state.json", state)
            continue

        if current["phase"] == "preevaluate":
            candidates = {slot: current["slots"][slot]["candidate"] for slot in EXPECTED_CANDIDATE_IDS}
            features = {
                slot: np.load(Path(current["slots"][slot]["directory"]) / "features.npy", allow_pickle=False)
                for slot in EXPECTED_CANDIDATE_IDS
            }
            evaluated = evaluate_candidates(
                dataset,
                candidates,
                features,
                evaluation_lambda=float(lambda_record["value"]),
                beta=beta,
                ee_tolerance=ee_tolerance,
                reward_tolerance=reward_tolerance,
            )
            current["evaluation"] = evaluated["report"]
            _write_json(round_dir / "pretraining_evaluation.json", evaluated["report"])
            _progress(
                "pretraining scores: baseline="
                f"{evaluated['report']['baseline']['score']:.6f}; "
                + ", ".join(
                    f"{slot}={evaluated['report']['candidates'][slot]['score']:.6f}"
                    for slot in EXPECTED_CANDIDATE_IDS
                )
            )
            selected = evaluated["selected_candidate_id"]
            if selected is None:
                current["phase"] = "completed_no_training"
                state["rounds"].append(copy.deepcopy(current))
                if round_number >= max_search_rounds:
                    state["status"] = "completed_search_rounds_exhausted"
                    state["stop_reason"] = "no candidate exceeded baseline pretraining score"
                    state["current_round"] = None
                    _write_json(output / "state.json", state)
                    return _result(state)
                state["search_round"] += 1
                state["current_round"] = {
                    "round": round_number + 1,
                    "phase": "generate_from_preevaluation",
                    "repair_calls": 0,
                    "slots": {},
                    "previous_preevaluation": evaluated["report"],
                    "evaluation": None,
                    "training": None,
                    "evaluation_result": None,
                }
                _write_json(output / "state.json", state)
                continue
            current["selected_candidate_id"] = selected
            _progress(f"selected {selected} for a fresh {train_episodes}-episode run")
            selected_dir = Path(current["slots"][selected]["directory"])
            artifact_root = selected_dir / "training_artifact"
            artifact_root.mkdir()
            approved = save_approved_artifact(
                artifact_root,
                candidate=candidates[selected],
                constants_metadata=constants,
                validation_report=current["slots"][selected]["validation"],
                evaluation_report=evaluated["report"],
                provenance={
                    "approval_method": "complete_episode_ordering_pretraining",
                    "search_run": str(output),
                    "search_round": round_number,
                    "candidate_slot": selected,
                    "candidate_design_summary": current["slots"][selected][
                        "submission"
                    ]["design_summary"],
                    "candidate_model_submission_sha256": _sha256_json(
                        current["slots"][selected]["submission"]
                    ),
                    "candidate_computation_fingerprint": current["slots"][selected][
                        "computation_fingerprint"
                    ],
                    "evaluation_lambda": lambda_record,
                    "dataset": dataset.provenance,
                    "beta": float(beta),
                    "lipschitz_evaluation_performed": False,
                },
            )
            current["approved_artifact"] = str(approved)
            current["phase"] = "train"
            _write_json(output / "state.json", state)
            continue

        if current["phase"] == "generate_from_preevaluation":
            prompt = render_stage_prompt(
                common,
                PREEVALUATION_TEMPLATE,
                {
                    "PRETRAINING_SCORE_REPORT": json.dumps(_public_score_report(current["previous_preevaluation"]), indent=2),
                    "EVALUATED_CANDIDATES": json.dumps(
                        {
                            slot: state["rounds"][-1]["slots"][slot]["submission"]
                            for slot in EXPECTED_CANDIDATE_IDS
                        },
                        indent=2,
                    ),
                    "MISORDERED_EPISODE_PAIRS": json.dumps(
                        {slot: current["previous_preevaluation"]["candidates"][slot]["examples"] for slot in EXPECTED_CANDIDATE_IDS}, indent=2
                    ),
                    "BEST_TRAINED_CONTEXT": json.dumps(
                        _best_trained_feedback(state.get("best_trained_candidate")),
                        indent=2,
                    ),
                },
            )
            call_number = state["model_calls"] + 1
            state["model_calls"] = call_number
            _write_json(output / "state.json", state)
            try:
                response = _call_model(
                    prompt=prompt, config=config, client=model_client,
                    directory=round_dir / f"model_call_{call_number:03d}",
                    effective_context=effective_context,
                )
            except Exception as exc:
                state["status"] = "paused_model_call_failed"
                state["stop_reason"] = f"{type(exc).__name__}: {exc}"
                _write_json(output / "state.json", state)
                return _result(state)
            try:
                _apply_candidate_batch(
                    current,
                    content=response["content"],
                    expected_slots=EXPECTED_CANDIDATE_IDS,
                    round_directory=round_dir,
                    dataset=dataset,
                    constants=constants,
                    worker_timeout=worker_timeout,
                )
            except CandidateError as exc:
                _record_batch_parse_failure(
                    current,
                    expected_slots=EXPECTED_CANDIDATE_IDS,
                    raw_content=response["content"],
                    error=exc,
                    round_directory=round_dir,
                )
                current["phase"] = "repair"
                _write_json(output / "state.json", state)
                continue
            current["phase"] = "preevaluate" if all(v["status"] == "validated" for v in current["slots"].values()) else "repair"
            _write_json(output / "state.json", state)
            continue

        if current["phase"] == "train":
            if not current.get("training_output_root"):
                current["training_output_root"] = str(
                    _short_training_output_root(output, round_number)
                )
            if not current.get("legacy_training_output_root"):
                current["legacy_training_output_root"] = str(
                    round_dir / "training"
                )
            # Persist the selected root before entering the subprocess so a
            # failed initialization is never rediscovered by directory order.
            _write_json(output / "state.json", state)
            try:
                training = training_runner(
                    method_id=SEARCH_METHOD_ID,
                    artifact=current["approved_artifact"],
                    episodes=train_episodes,
                    seed=config["seed"],
                    output_directory=current["training_output_root"],
                    legacy_output_directory=current["legacy_training_output_root"],
                    prior_output_directories=current.get(
                        "prior_training_output_roots", []
                    ),
                    resume_record=current.get("training"),
                    restart_from_scratch=pending_checkpoint_path_restart,
                )
            except Exception as exc:
                current["training"] = {
                    "status": "blocked",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "run_directory": (
                        (current.get("training") or {}).get("run_directory")
                    ),
                    "output_root": current["training_output_root"],
                    "legacy_output_root": current["legacy_training_output_root"],
                }
                state["status"] = "paused_training_failure"
                state["stop_reason"] = current["training"]["reason"]
                _write_json(output / "state.json", state)
                return _result(state)
            current["training"] = training
            if training.get("status") != "complete":
                state["status"] = "paused_training"
                state["stop_reason"] = training.get("reason", "training incomplete")
                _write_json(output / "state.json", state)
                return _result(state)
            _progress(f"training complete for round {round_number}")
            current["phase"] = "evaluate"
            _write_json(output / "state.json", state)
            continue

        if current["phase"] == "evaluate":
            if not evaluation_manifest or not baseline_run or not baseline_evaluation:
                raise EpisodeSearchError(
                    "formal evaluation requires explicit baseline run, baseline results, and shared manifest"
                )
            result = evaluation_runner(
                method_id=SEARCH_METHOD_ID,
                run_directory=current["training"]["run_directory"],
                checkpoint_episode=1500,
                episodes=evaluation_episodes,
                roi_count=8,
                environment_size_m=(1000.0, 1000.0),
                manifest=evaluation_manifest,
                baseline_run=baseline_run,
                baseline_evaluation=baseline_evaluation,
                baseline_preflight=baseline_preflight,
                output_directory=round_dir / "evaluation",
            )
            current["evaluation_result"] = result
            if result.get("status") != "complete":
                state["status"] = "paused_evaluation"
                state["stop_reason"] = result.get("reason", "evaluation incomplete")
                _write_json(output / "state.json", state)
                return _result(state)
            _progress(
                "evaluation mean episode EE="
                f"{float(result['mean_episode_energy_efficiency_mbit_per_j']):.9g}"
            )
            mean_ee = float(result["mean_episode_energy_efficiency_mbit_per_j"])
            best = state.get("best_trained_candidate")
            if best is None or mean_ee > float(best["mean_episode_energy_efficiency_mbit_per_j"]):
                state["best_trained_candidate"] = {
                    "round": round_number,
                    "candidate_id": current["selected_candidate_id"],
                    "candidate": current["slots"][current["selected_candidate_id"]]["candidate"],
                    "candidate_submission": current["slots"][
                        current["selected_candidate_id"]
                    ]["submission"],
                    "artifact": current["approved_artifact"],
                    "training": current["training"],
                    "evaluation": result,
                    "mean_episode_energy_efficiency_mbit_per_j": mean_ee,
                }
                current["updated_best"] = True
                _progress("historical best LLM candidate updated")
            else:
                current["updated_best"] = False
            current["phase"] = "completed_trained"
            state["rounds"].append(copy.deepcopy(current))
            if round_number >= max_search_rounds:
                state["status"] = "complete"
                state["current_round"] = None
                _write_json(output / "state.json", state)
                return _result(state)
            state["search_round"] += 1
            state["current_round"] = {
                "round": round_number + 1,
                "phase": "generate_from_training",
                "repair_calls": 0,
                "slots": {},
            }
            _write_json(output / "state.json", state)
            continue

        if current["phase"] == "generate_from_training":
            latest = state["rounds"][-1]
            prompt = render_stage_prompt(
                common,
                TRAINING_TEMPLATE,
                {
                    "LATEST_TRAINED_CANDIDATE": json.dumps(
                        {
                            "candidate_id": latest["selected_candidate_id"],
                            "candidate": latest["slots"][latest["selected_candidate_id"]][
                                "submission"
                            ],
                            "artifact": latest.get("approved_artifact"),
                        },
                        indent=2,
                    ),
                    "TRAINING_SUMMARIES": json.dumps(latest["training"].get("summaries"), indent=2),
                    "EVALUATION_RESULTS": json.dumps(
                        _evaluation_feedback(latest["evaluation_result"]), indent=2
                    ),
                    "BEST_TRAINED_CANDIDATE": json.dumps(
                        _best_trained_feedback(state["best_trained_candidate"]),
                        indent=2,
                        default=str,
                    ),
                    "SEARCH_HISTORY_SUMMARY": json.dumps([
                        {"round": item["round"], "selected": item.get("selected_candidate_id"), "updated_best": item.get("updated_best")}
                        for item in state["rounds"]
                    ], indent=2),
                },
            )
            call_number = state["model_calls"] + 1
            state["model_calls"] = call_number
            _write_json(output / "state.json", state)
            try:
                response = _call_model(
                    prompt=prompt, config=config, client=model_client,
                    directory=round_dir / f"model_call_{call_number:03d}",
                    effective_context=effective_context,
                )
            except Exception as exc:
                state["status"] = "paused_model_call_failed"
                state["stop_reason"] = f"{type(exc).__name__}: {exc}"
                _write_json(output / "state.json", state)
                return _result(state)
            try:
                _apply_candidate_batch(
                    current,
                    content=response["content"],
                    expected_slots=EXPECTED_CANDIDATE_IDS,
                    round_directory=round_dir,
                    dataset=dataset,
                    constants=constants,
                    worker_timeout=worker_timeout,
                )
            except CandidateError as exc:
                _record_batch_parse_failure(
                    current,
                    expected_slots=EXPECTED_CANDIDATE_IDS,
                    raw_content=response["content"],
                    error=exc,
                    round_directory=round_dir,
                )
                current["phase"] = "repair"
                _write_json(output / "state.json", state)
                continue
            current["phase"] = "preevaluate" if all(v["status"] == "validated" for v in current["slots"].values()) else "repair"
            _write_json(output / "state.json", state)
            continue

        raise EpisodeSearchError(f"unknown search phase: {current['phase']}")

    state["status"] = "complete"
    _write_json(output / "state.json", state)
    return _result(state)
