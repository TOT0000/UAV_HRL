import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import llm_episode_search as search
import llm_episode_training as episode_training
import run_experiment
from experiment_config import MethodSpec
from experiment_paths import write_run_status
from llm_candidate import save_approved_artifact
from llm_runtime import (
    LLMRuntimeError,
    artifact_identity,
    copy_approved_artifact,
    llm_checkpoint_path_preflight,
)
from training_checkpoint import CHECKPOINT_PROVENANCE_FIELDS
from training_history import (
    build_training_history_row,
    training_history_identity,
    write_training_history,
)


def _dataset(delivered=(1.0, 2.0, 4.0), energy=(1.0, 1.0, 1.0)):
    episodes = []
    for index in range(len(delivered)):
        episodes.append(
            search.EpisodeRecord(
                index=index,
                source_index=0,
                source_id="source-a",
                episode_id=index,
                scenario_id=f"scenario-{index}",
                rows=np.asarray([index], dtype=np.int64),
            )
        )
    arrays = {
        "state": np.zeros((len(delivered), 2), dtype=np.float32),
        "delivered_mbits": np.asarray(delivered, dtype=np.float32)[:, None],
        "total_mobility_energy": np.asarray(energy, dtype=np.float32)[:, None],
        "c9_penalty": np.zeros((len(delivered), 1), dtype=np.float32),
        "c10_penalty": np.zeros((len(delivered), 1), dtype=np.float32),
        "com_range_penalty": np.zeros((len(delivered), 1), dtype=np.float32),
    }
    return search.EpisodeDataset(
        arrays=arrays,
        episodes=tuple(episodes),
        fixed_metadata={"compatibility_contract": {}},
        provenance={"dataset_content_sha256": "fixture"},
    )


def _candidate(name, weight):
    return {
        "candidate_name": name,
        "features": [{"name": name, "reward_weight": float(weight)}],
    }


def _preflight_fixture():
    return {
        "schema_version": "uav-hrl-episode-search-baseline-preflight-v1",
        "status": "passed",
        "baseline_run": "fixture-baseline-run",
        "baseline_training_run_id": "fixture-baseline-run",
        "checkpoint_episode": 1500,
        "checkpoint_path": "fixture-checkpoint",
        "checkpoint_provenance": {},
        "baseline_evaluation_metadata": "fixture-evaluation.json",
        "baseline_evaluation_metadata_sha256": "fixture-metadata-sha",
        "per_episode_jsonl": "fixture-episodes.jsonl",
        "per_episode_jsonl_sha256": "fixture-rows-sha",
        "manifest_path": "fixture-manifest.json",
        "manifest_file_sha256": "fixture-manifest-file-sha",
        "manifest_content_hash": "fixture-manifest",
        "scenario_ids": ["scenario-0", "scenario-1", "scenario-2"],
        "evaluation_contract": {
            "episodes": 100,
            "roi_count": 8,
            "environment_size_m": [1000.0, 1000.0],
            "episode_seconds": 60,
        },
        "metrics": {},
    }


def _patch_preflight(monkeypatch):
    monkeypatch.setattr(
        episode_training,
        "validate_baseline_preflight",
        lambda **_kwargs: _preflight_fixture(),
    )


def _approved_artifact(root: Path, *, provenance=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = {
        "schema_version": "uav-hrl-llm-shared-feature-candidate-v2",
        "candidate_name": "candidate_1",
        "reward_input_mode": "current_only",
        "features": [
            {
                "index": 0,
                "name": "fixture_zero",
                "dtype": "float32",
                "description": "Fixture-only constant feature.",
                "range": {"minimum": 0.0, "maximum": 1.0},
                "source_fields": [],
                "formula": "0",
                "missing_data_rule": "always zero",
                "reward_weight": 0.0,
            }
        ],
        "code": "def compute_extra_state(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)",
    }
    return save_approved_artifact(
        root,
        candidate=candidate,
        constants_metadata={},
        validation_report={"status": "passed", "fixture": True},
        evaluation_report={"status": "passed", "fixture": True},
        provenance={"beta": 1.0, "fixture": True, **(provenance or {})},
    )


def test_episode_reward_components_and_ordering_tie_rules_are_hand_checkable():
    dataset = _dataset()
    feature = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float64)
    result = search.episode_reward_components(
        dataset,
        evaluation_lambda=0.25,
        beta=2.0,
        features=feature,
        weights=np.asarray([-0.5]),
    )
    assert result["energy_efficiency_mbit_per_j"].tolist() == pytest.approx([1, 2, 4])
    assert result["base_reward"].tolist() == pytest.approx([0.75, 1.75, 3.75])
    assert result["feature_contributions"][:, 0].tolist() == pytest.approx([0, -0.5, -1])
    assert result["combined_reward"].tolist() == pytest.approx([0.75, 1.25, 2.75])

    score = search.ordering_score(
        np.asarray([1.0, 1.0 + 5e-13, 2.0, 3.0]),
        np.asarray([0.0, 9.0, 1.0, 1.0]),
        ee_tolerance=1e-12,
        reward_tolerance=1e-12,
    )
    assert score["ee_tie_excluded_pair_count"] == 1
    assert score["comparable_pair_count"] == 5
    assert score["correct_count"] == 2
    assert score["reward_tie_count"] == 1
    assert score["reversed_count"] == 2
    assert score["score"] == pytest.approx(2.5 / 5.0)


def test_candidate_selection_requires_strict_baseline_improvement_and_slot_tie_break():
    dataset = _dataset(delivered=(3.0, 1.0, 2.0))
    # Baseline reward has the same order as EE and cannot be strictly exceeded.
    candidates = {slot: _candidate(slot, 0.0) for slot in search.EXPECTED_CANDIDATE_IDS}
    features = {slot: np.zeros((3, 1)) for slot in search.EXPECTED_CANDIDATE_IDS}
    result = search.evaluate_candidates(
        dataset, candidates, features, evaluation_lambda=0.0, beta=1.0
    )
    assert result["selected_candidate_id"] is None

    # Penalties make baseline reverse one pair; candidate_1/2 both repair it.
    dataset.arrays["c9_penalty"][:, 0] = [3.0, 0.0, 0.0]
    features["candidate_1"][:, 0] = [1.0, 0.0, 0.0]
    features["candidate_2"][:, 0] = [1.0, 0.0, 0.0]
    candidates["candidate_1"]["features"][0]["reward_weight"] = 4.0
    candidates["candidate_2"]["features"][0]["reward_weight"] = 4.0
    result = search.evaluate_candidates(
        dataset, candidates, features, evaluation_lambda=0.0, beta=1.0
    )
    assert result["selected_candidate_id"] == "candidate_1"
    assert result["report"]["candidates"]["candidate_1"]["score"] == result["report"]["candidates"]["candidate_2"]["score"]


def test_lambda_can_be_explicit_or_derived_from_last_100_used_values(tmp_path):
    explicit = search.resolve_evaluation_lambda(explicit_lambda=0.125)
    assert explicit == {
        "value": 0.125,
        "method": "explicit_cli_value",
        "source": None,
        "episode_range": None,
    }

    identity = training_history_identity("td3_dinkelbach", 9, "manifest")
    rows = []
    for episode in range(1, 121):
        rows.append(
            build_training_history_row(
                identity,
                episode=episode,
                reward=0.0,
                timely_goodput_mbits=0.0,
                mobility_energy_j=1.0,
                dinkelbach_lambda_used=episode / 1000.0,
                dinkelbach_lambda_after_episode=999.0,
                dinkelbach_lambda_updated=False,
                dinkelbach_update_status="fixture",
                dinkelbach_block_index=1,
                dinkelbach_block_episode=episode,
                dinkelbach_block_timely_mbits_so_far=0.0,
                dinkelbach_block_energy_joules_so_far=float(episode),
            )
        )
    write_training_history(tmp_path, rows, identity)
    (tmp_path / "resolved_config.json").write_text(
        json.dumps(
            {
                "method_spec": {"method_id": "td3_dinkelbach"},
                "seed": 9,
                "training_history_identity_manifest_hash": "manifest",
            }
        ),
        encoding="utf-8",
    )
    derived = search.resolve_evaluation_lambda(training_run=tmp_path)
    assert derived["episode_range"] == [21, 120]
    assert derived["value"] == pytest.approx(np.mean(np.arange(21, 121) / 1000.0))
    assert derived["maximum"] != 999.0


def test_training_block_summaries_preserve_component_addition_and_actual_lambda():
    rows = []
    for episode in range(1, 1501):
        rows.append(
            {
                "episode": episode,
                "feature_names": ["a", "b"],
                "feature_contribution_sums": [1.0, -0.25],
                "energy_efficiency_mbit_per_j": float(episode),
                "base_reward_sum": 2.0,
                "extra_reward_sum": 0.75,
                "combined_reward_sum": 2.75,
                "roi_count": 2 + episode % 2,
                "exploration": {"std": 0.1},
                "dinkelbach_lambda_used": episode / 1000,
            }
        )
    summary = search.summarize_training_blocks(rows, expected_episodes=1500)
    assert len(summary) == 15
    assert summary[0]["episode_range"] == [1, 100]
    assert summary[-1]["episode_range"] == [1401, 1500]
    assert summary[0]["feature_contributions"]["a"]["mean"] == 1.0
    assert summary[0]["feature_contributions"]["b"]["mean"] == -0.25
    assert summary[0]["extra_reward_sum"]["mean"] == 0.75
    assert summary[0]["lambda_semantics"].startswith("actual")


def test_batch_parser_and_prompt_contract_have_four_slots_without_reviewer_or_lipschitz(tmp_path):
    values = []
    for slot in search.EXPECTED_CANDIDATE_IDS:
        values.append(
            {
                "candidate_id": slot,
                "design_summary": "distinct fixture",
                "features": [{"name": slot, "description": "fixture", "reward_weight": 0.0}],
                "code": "def compute_extra_state(obs, constants):\n    return [0.0]",
            }
        )
    parsed = search.parse_candidate_batch(json.dumps({"candidates": values}), expected_ids=search.EXPECTED_CANDIDATE_IDS)
    assert tuple(parsed["slots"]) == search.EXPECTED_CANDIDATE_IDS
    assert all(item["error"] is None for item in parsed["slots"].values())
    stage = search.render_stage_prompt("common", search.INITIAL_TEMPLATE, {})
    assert "candidate_1 through candidate_4" in stage
    assert "reviewer" not in stage.lower()
    assert "lipschitz" not in stage.lower()
    assert not search._PLACEHOLDER.findall(stage)
    assert "{{candidate_data}}" in search.render_stage_prompt(
        "common {{candidate_data}}", search.INITIAL_TEMPLATE, {}
    )


def test_computation_fingerprint_ignores_renaming_but_keeps_weight_changes():
    code = "def compute_extra_state(obs, constants):\n    return [0.5]"
    first = {"code": code, "features": [{"name": "a", "description": "one", "reward_weight": 1.0}]}
    renamed = {"code": code, "features": [{"name": "b", "description": "two", "reward_weight": 1.0}]}
    reweighted = {"code": code, "features": [{"name": "a", "description": "one", "reward_weight": -1.0}]}
    assert search.candidate_computation_fingerprint(first) == search.candidate_computation_fingerprint(renamed)
    assert search.candidate_computation_fingerprint(first) != search.candidate_computation_fingerprint(reweighted)


def test_new_method_is_isolated_but_reuses_existing_llm_algorithm_contract():
    baseline = MethodSpec.parse("td3_dinkelbach")
    old_llm = MethodSpec.parse("td3_dinkelbach_llm")
    new_llm = MethodSpec.parse("td3_dinkelbach_llm_search")
    assert baseline.llm_enabled is False
    assert old_llm.llm_enabled and new_llm.llm_enabled
    assert new_llm.method_id != old_llm.method_id
    assert new_llm.agent == old_llm.agent == baseline.agent
    assert new_llm.routing == old_llm.routing == baseline.routing
    assert new_llm.reward_mode == old_llm.reward_mode == baseline.reward_mode


class _MockClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return {
            "content": self.responses.pop(0),
            "reasoning": None,
            "finish_reason": "stop",
            "actual_model": "fixture/model",
            "transport_completed": True,
            "tool_calls_seen": False,
        }


def test_model_call_failure_pauses_with_persisted_call_and_resume_uses_new_call(
    tmp_path, monkeypatch
):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)

    class FailingClient:
        def chat(self, **_kwargs):
            raise RuntimeError("fixture transport failure")

    output = tmp_path / "resumable-run"
    first = search.run_episode_search(
        episode_sources=[tmp_path / "source"],
        provider="openai",
        model="fixture/model",
        context_length=50000,
        max_output_tokens=4096,
        evaluation_lambda=0.0,
        max_search_rounds=1,
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-evaluation",
        evaluation_manifest=tmp_path / "manifest.json",
        output_dir=output,
        client=FailingClient(),
    )
    assert first["status"] == "paused_model_call_failed"
    failed_state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert failed_state["model_calls"] == 1
    assert (output / "round_01" / "model_call_001" / "prompt.txt").is_file()

    submissions = {
        "candidates": [
            {
                "candidate_id": slot,
                "design_summary": f"valid-{slot}",
                "features": [
                    {
                        "name": slot,
                        "description": "fixture",
                        "reward_weight": 0.0,
                    }
                ],
                "code": (
                    "def compute_extra_state(obs, constants):\n"
                    f"    return [{index / 10.0}]"
                ),
            }
            for index, slot in enumerate(search.EXPECTED_CANDIDATE_IDS)
        ]
    }

    monkeypatch.setattr(
        search,
        "validate_search_candidate",
        lambda submission, **_kwargs: (
            {
                "candidate_name": submission["candidate_id"],
                "features": submission["features"],
                "code": submission["code"],
            },
            {"status": "passed"},
            np.zeros((3, 1)),
        ),
    )
    monkeypatch.setattr(
        search,
        "evaluate_candidates",
        lambda *_args, **_kwargs: {
            "report": {
                "baseline": {"score": 1.0},
                "candidates": {
                    slot: {"score": 1.0, "examples": []}
                    for slot in search.EXPECTED_CANDIDATE_IDS
                },
                "selected_candidate_id": None,
                "selection_rule": "fixture",
            },
            "components": {},
            "selected_candidate_id": None,
        },
    )
    resumed = search.run_episode_search(
        resume=output,
        client=_MockClient([json.dumps(submissions)]),
        training_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not train")
        ),
        evaluation_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not evaluate")
        ),
    )
    assert resumed["status"] == "completed_search_rounds_exhausted"
    assert resumed["model_calls"] == 2
    assert (output / "round_01" / "model_call_002" / "response.json").is_file()


def test_partial_repair_locks_passing_slots_and_uses_one_shared_repair_call(tmp_path, monkeypatch):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)

    def validate(submission, **_kwargs):
        failed = submission["design_summary"] == "invalid"
        candidate = {
            "candidate_name": submission["candidate_id"],
            "features": submission["features"],
            "code": submission["code"],
        }
        return candidate, ({"status": "failed", "error": "fixture"} if failed else {"status": "passed"}), (None if failed else np.zeros((3, 1)))

    monkeypatch.setattr(search, "validate_search_candidate", validate)
    monkeypatch.setattr(
        search,
        "evaluate_candidates",
        lambda *_args, **_kwargs: {
            "report": {
                "baseline": {"score": 1.0},
                "candidates": {slot: {"score": 1.0, "examples": []} for slot in search.EXPECTED_CANDIDATE_IDS},
                "selected_candidate_id": None,
                "selection_rule": "fixture",
            },
            "components": {},
            "selected_candidate_id": None,
        },
    )

    def batch(invalid):
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": slot,
                        "design_summary": "invalid" if slot in invalid else f"valid-{slot}",
                        "features": [{"name": f"f-{slot}", "description": "fixture", "reward_weight": 0.0}],
                        "code": f"def compute_extra_state(obs, constants):\n    return [{index / 10.0}]",
                    }
                    for index, slot in enumerate(search.EXPECTED_CANDIDATE_IDS)
                ]
            }
        )

    initial_value = json.loads(batch({"candidate_4"}))
    initial_value["candidates"][1].pop("design_summary")
    initial = json.dumps(initial_value)
    repaired_all = json.loads(batch(set()))["candidates"]
    repaired = json.dumps({"candidates": [repaired_all[1], repaired_all[3]]})
    client = _MockClient([initial, repaired])
    result = search.run_episode_search(
        episode_sources=[tmp_path / "source"],
        provider="openai",
        model="fixture/model",
        context_length=50000,
        max_output_tokens=4096,
        evaluation_lambda=0.0,
        max_search_rounds=1,
        max_repairs_per_round=1,
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-evaluation",
        evaluation_manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "run",
        client=client,
        training_runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not train")),
        evaluation_runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not evaluate")),
    )
    assert result["status"] == "completed_search_rounds_exhausted"
    state = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    finished = state["rounds"][0]
    assert finished["slots"]["candidate_1"]["version"] == 1
    assert finished["slots"]["candidate_3"]["version"] == 1
    assert finished["slots"]["candidate_2"]["version"] == 2
    assert finished["slots"]["candidate_4"]["version"] == 2
    assert finished["repair_calls"] == 1
    assert len(client.prompts) == 2
    assert "candidate_2" in client.prompts[1] and "candidate_4" in client.prompts[1]


def test_mock_training_evaluation_use_formal_counts_and_preserve_historical_best(
    tmp_path, monkeypatch
):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)

    def validate(submission, **_kwargs):
        candidate = {
            "candidate_name": submission["candidate_id"],
            "features": submission["features"],
            "code": submission["code"],
        }
        return candidate, {"status": "passed"}, np.zeros((3, 1))

    monkeypatch.setattr(search, "validate_search_candidate", validate)

    def evaluated(_dataset, _candidates, _features, **_kwargs):
        candidate_rows = {
            slot: {
                "score": 0.75 if slot == "candidate_2" else 0.5,
                "strictly_exceeds_baseline": slot == "candidate_2",
                "examples": [],
            }
            for slot in search.EXPECTED_CANDIDATE_IDS
        }
        return {
            "report": {
                "baseline": {"score": 0.5},
                "candidates": candidate_rows,
                "selected_candidate_id": "candidate_2",
                "selection_rule": "fixture",
            },
            "components": {},
            "selected_candidate_id": "candidate_2",
        }

    monkeypatch.setattr(search, "evaluate_candidates", evaluated)

    def batch(round_index):
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": slot,
                        "design_summary": f"round-{round_index}-{slot}",
                        "features": [{"name": f"r{round_index}-{slot}", "description": "fixture", "reward_weight": 0.0}],
                        "code": f"def compute_extra_state(obs, constants):\n    return [{round_index / 10 + index / 100}]",
                    }
                    for index, slot in enumerate(search.EXPECTED_CANDIDATE_IDS)
                ]
            }
        )

    client = _MockClient([batch(1), batch(2)])
    training_calls = []
    evaluation_calls = []
    restart_record = {
        "restart_status": "authorized_not_started",
        "operation_id": "round-one-restart",
        "failed_run_directory": str(tmp_path / "failed-round-one"),
    }
    monkeypatch.setattr(
        search,
        "_checkpoint_restart_for_current_round",
        lambda _state, current: restart_record
        if int(current["round"]) == 1
        else None,
    )

    def train_runner(**kwargs):
        training_calls.append(kwargs)
        return {
            "status": "complete",
            "run_directory": str(tmp_path / f"trained-{len(training_calls)}"),
            "summaries": [{"episode_range": [1, 100]}],
        }

    def evaluation_runner(**kwargs):
        evaluation_calls.append(kwargs)
        mean_ee = 2.0 if len(evaluation_calls) == 1 else 1.0
        return {
            "status": "complete",
            "episode_count": 100,
            "roi_count": 8,
            "environment_size_m": [1000.0, 1000.0],
            "scenario_manifest_hash": "fixture-manifest",
            "mean_episode_energy_efficiency_mbit_per_j": mean_ee,
            "std_episode_energy_efficiency_mbit_per_j": 0.25,
            "baseline_mean_episode_energy_efficiency_mbit_per_j": 0.75,
            "improves_over_baseline_mean_episode_ee": mean_ee > 0.75,
            "candidate_metrics": {
                "timely_useful_delivery_mbit": {"total": 321.0},
                "movement_energy_j": {"total": 654.0},
                "end_to_end_delay_violation_probability": {
                    "VS": {"value": 0.125},
                    "COM": {"value": 0.25},
                },
            },
            "baseline_metrics": {
                "timely_useful_delivery_mbit": {"total": 111.0},
                "movement_energy_j": {"total": 222.0},
                "end_to_end_delay_violation_probability": {
                    "VS": {"value": 0.5},
                    "COM": {"value": None},
                },
            },
        }

    result = search.run_episode_search(
        episode_sources=[tmp_path / "source"],
        provider="openai",
        model="fixture/model",
        context_length=50000,
        max_output_tokens=4096,
        evaluation_lambda=0.0,
        max_search_rounds=2,
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-evaluation",
        evaluation_manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "run-trained",
        client=client,
        training_runner=train_runner,
        evaluation_runner=evaluation_runner,
    )
    assert result["status"] == "complete"
    assert len(training_calls) == len(evaluation_calls) == 2
    assert training_calls[0]["restart_from_scratch"] is True
    assert training_calls[0]["restart_authorization_id"] == "round-one-restart"
    assert training_calls[1]["restart_from_scratch"] is False
    assert training_calls[1]["restart_authorization_id"] is None
    assert restart_record["restart_status"] == "completed"
    assert restart_record["replacement_run_directory"] == str(
        (tmp_path / "trained-1").resolve()
    )
    assert all(call["episodes"] == 1500 for call in training_calls)
    assert all(call["episodes"] == 100 for call in evaluation_calls)
    assert all(call["roi_count"] == 8 for call in evaluation_calls)
    assert all(call["environment_size_m"] == (1000.0, 1000.0) for call in evaluation_calls)
    state = json.loads((tmp_path / "run-trained" / "state.json").read_text(encoding="utf-8"))
    assert state["best_trained_candidate"]["round"] == 1
    assert state["best_trained_candidate"]["mean_episode_energy_efficiency_mbit_per_j"] == 2.0
    assert "Best trained candidate so far" in client.prompts[1]
    assert '"total": 321.0' in client.prompts[1]
    assert '"value": 0.125' in client.prompts[1]


def test_partial_batch_schema_failure_preserves_other_slots_and_safe_fences():
    entries = [
        {
            "candidate_id": slot,
            "design_summary": f"summary-{slot}",
            "features": [{"name": slot, "description": "fixture", "reward_weight": 0.0}],
            "code": "def compute_extra_state(obs, constants):\n    return [0.0]",
        }
        for slot in search.EXPECTED_CANDIDATE_IDS
    ]
    entries[1].pop("design_summary")
    fenced = "explanation\n```json\n" + json.dumps({"candidates": entries}) + "\n```\nfooter"
    parsed = search.parse_candidate_batch(
        fenced, expected_ids=search.EXPECTED_CANDIDATE_IDS
    )
    assert parsed["parse_metadata"]["parse_method"] == "unique_markdown_json_fence"
    assert parsed["slots"]["candidate_2"]["error"].startswith(
        "candidate fields must be exactly"
    )
    assert all(
        parsed["slots"][slot]["error"] is None
        for slot in ("candidate_1", "candidate_3", "candidate_4")
    )

    duplicated = entries + [dict(entries[0])]
    duplicate_result = search.parse_candidate_batch(
        json.dumps({"candidates": duplicated}),
        expected_ids=search.EXPECTED_CANDIDATE_IDS,
    )
    assert "appears 2 times" in duplicate_result["slots"]["candidate_1"]["error"]
    assert duplicate_result["slots"]["candidate_3"]["error"] is None
    with pytest.raises(Exception, match="multiple|fence|ambiguous"):
        search.parse_candidate_batch(
            "```json\n{}\n```\n```json\n{}\n```",
            expected_ids=search.EXPECTED_CANDIDATE_IDS,
        )


def _raw_training_metric(episode):
    return {
        "episode": int(episode),
        "method_id": "td3_dinkelbach_llm_search",
        "llm_artifact_id": "artifact-fixture",
        "llm_feature_names": ["a", "b"],
        "llm_feature_reward_weights": [0.25, -0.5],
        "llm_feature_contribution_sums": [1.0, -0.25],
        "llm_reward_beta": 1.0,
        "llm_base_reward_sum": 2.0,
        "llm_weighted_extra_reward_sum": 0.75,
        "llm_combined_reward_sum": 2.75,
        "total_timely_useful_mbits": 3.0,
        "total_mobility_energy_j": 4.0,
        "energy_efficiency_mbit_per_j": 0.75,
        "num_GT": 8,
        "dinkelbach_lambda_used": 0.125,
        "movement_exploration": {"std": 0.1},
    }


def test_episode_observer_persists_and_resume_reconciles_checkpoint_tail(tmp_path):
    method = MethodSpec.parse("td3_dinkelbach_llm_search")
    observer = run_experiment._llm_episode_observer(tmp_path, method)
    assert observer is not None
    for episode in range(1, 6):
        observer(_raw_training_metric(episode))
    path = tmp_path / "llm_training_episode_metrics.jsonl"
    assert [row["episode"] for row in run_experiment._read_llm_training_episode_metrics(path)] == [1, 2, 3, 4, 5]

    # An interrupted non-atomic legacy tail is ignored rather than counted.
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"episode": 6')
    assert [row["episode"] for row in run_experiment._read_llm_training_episode_metrics(path)] == [1, 2, 3, 4, 5]

    # Resume from checkpoint 3 removes rows that will be executed again.
    run_experiment._write_llm_training_episode_metrics(
        tmp_path, method, [], completed_episode_limit=3
    )
    for episode in range(4, 7):
        observer(_raw_training_metric(episode))
    rows = run_experiment._read_llm_training_episode_metrics(path)
    assert [row["episode"] for row in rows] == list(range(1, 7))
    summaries = search.summarize_training_blocks(
        rows, block_size=3, expected_episodes=6
    )
    assert [item["episode_range"] for item in summaries] == [[1, 3], [4, 6]]

    with pytest.raises(search.EpisodeSearchError, match="missing, duplicated, or out of order"):
        search.summarize_training_blocks(
            rows[:2] + rows[3:], block_size=1, expected_episodes=5
        )
    changed = [dict(row) for row in rows]
    changed[-1]["candidate_artifact_id"] = "different-artifact"
    with pytest.raises(search.EpisodeSearchError, match="mix candidate identities"):
        search.summarize_training_blocks(changed, block_size=3, expected_episodes=6)


def test_evaluation_metrics_use_pooled_task_denominators_and_report_missing():
    rows = [
        {
            "scenario_id": "a",
            "energy_efficiency_mbit_per_j": 1.0,
            "total_timely_useful_mbits": 10.0,
            "total_mobility_energy_j": 2.0,
            "fov_eligible_packets": 2,
            "fov_violation_packets": 1,
            "com_eligible_packets": 0,
            "com_violation_packets": 0,
        },
        {
            "scenario_id": "b",
            "energy_efficiency_mbit_per_j": 3.0,
            "total_timely_useful_mbits": 20.0,
            "total_mobility_energy_j": 6.0,
            "fov_eligible_packets": 8,
            "fov_violation_packets": 1,
            "com_eligible_packets": 0,
            "com_violation_packets": 0,
        },
    ]
    metrics = episode_training._evaluation_metrics_summary(rows)
    assert metrics["episode_energy_efficiency_mbit_per_j"] == {
        "mean": 2.0,
        "std": 1.0,
        "aggregation": "arithmetic mean and population standard deviation across episodes",
    }
    assert metrics["timely_useful_delivery_mbit"]["total"] == 30.0
    assert metrics["movement_energy_j"]["total"] == 8.0
    vs = metrics["end_to_end_delay_violation_probability"]["VS"]
    assert (vs["violated_packets"], vs["eligible_packets"], vs["value"]) == (2, 10, 0.2)
    com = metrics["end_to_end_delay_violation_probability"]["COM"]
    assert com["value"] is None and com["missing"] is True


def test_resume_can_increase_repair_budget_without_resetting_usage(tmp_path, monkeypatch):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)

    def validate(submission, **_kwargs):
        passed = submission.get("design_summary") == "valid"
        candidate = {
            "candidate_name": submission["candidate_id"],
            "features": submission.get("features", []),
            "code": submission.get("code", ""),
        }
        return candidate, {"status": "passed" if passed else "failed"}, (np.zeros((3, 1)) if passed else None)

    monkeypatch.setattr(search, "validate_search_candidate", validate)
    monkeypatch.setattr(
        search,
        "evaluate_candidates",
        lambda *_args, **_kwargs: {
            "report": {
                "baseline": {"score": 1.0},
                "candidates": {slot: {"score": 1.0, "examples": []} for slot in search.EXPECTED_CANDIDATE_IDS},
                "selected_candidate_id": None,
                "selection_rule": "fixture",
            },
            "components": {},
            "selected_candidate_id": None,
        },
    )

    def batch(summary):
        return json.dumps({"candidates": [
            {
                "candidate_id": slot,
                "design_summary": summary,
                "features": [{"name": slot, "description": "fixture", "reward_weight": 0.0}],
                "code": f"def compute_extra_state(obs, constants):\n    return [{index / 10.0}]",
            }
            for index, slot in enumerate(search.EXPECTED_CANDIDATE_IDS)
        ]})

    output = tmp_path / "repair-budget"
    first = search.run_episode_search(
        episode_sources=[tmp_path / "source"],
        provider="openai",
        model="fixture/model",
        evaluation_lambda=0.0,
        max_search_rounds=1,
        max_repairs_per_round=0,
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-evaluation",
        evaluation_manifest=tmp_path / "manifest.json",
        output_dir=output,
        client=_MockClient([batch("invalid")]),
    )
    assert first["status"] == "paused_repairs_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert state["current_round"]["repair_calls"] == 0

    resumed = search.run_episode_search(
        resume=output,
        max_repairs_per_round=1,
        client=_MockClient([batch("valid")]),
    )
    assert resumed["status"] == "completed_search_rounds_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert state["settings"]["max_repairs_per_round"] == 1
    assert state["rounds"][0]["repair_calls"] == 1
    with pytest.raises(search.EpisodeSearchError, match="cannot lower"):
        search.run_episode_search(
            resume=output,
            max_repairs_per_round=0,
            client=_MockClient([]),
        )


def test_numeric_rule_revalidation_preserves_versions_usage_and_resumes_at_preevaluation(
    tmp_path, monkeypatch
):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)
    monkeypatch.setattr(
        search,
        "_git_sha",
        lambda: next(iter(search.REVALIDATABLE_NUMERIC_RULE_SOURCE_REVISIONS)),
    )
    new_rules = {"enabled": False}

    def validate(submission, **_kwargs):
        candidate = {
            "candidate_name": submission["candidate_id"],
            "features": submission["features"],
            "code": submission["code"],
        }
        failed = submission["candidate_id"] == "candidate_1" and not new_rules["enabled"]
        return (
            candidate,
            {"status": "failed", "error": "astype is not allowed"}
            if failed
            else {"status": "passed"},
            None if failed else np.zeros((3, 1), dtype=np.float32),
        )

    monkeypatch.setattr(search, "validate_search_candidate", validate)

    def batch():
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": slot,
                        "design_summary": "saved fixture",
                        "features": [
                            {
                                "name": slot,
                                "description": "fixture",
                                "reward_weight": 0.0,
                            }
                        ],
                        "code": (
                            "def compute_extra_state(obs, constants):\n"
                            '    mask = obs["movement_mask"]\n'
                            "    return [np.mean(mask.astype(float))]\n"
                            if slot == "candidate_1"
                            else f"def compute_extra_state(obs, constants):\n    return [{int(slot[-1]) / 10.0}]"
                        ),
                    }
                    for slot in search.EXPECTED_CANDIDATE_IDS
                ]
            }
        )

    output = tmp_path / "revalidation-run"
    first = search.run_episode_search(
        episode_sources=[tmp_path / "source"],
        provider="openai",
        model="fixture/model",
        evaluation_lambda=0.0,
        max_search_rounds=1,
        max_repairs_per_round=0,
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-evaluation",
        evaluation_manifest=tmp_path / "manifest.json",
        output_dir=output,
        client=_MockClient([batch()]),
    )
    assert first["status"] == "paused_repairs_exhausted"
    before_state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    before_versions = {
        slot: value["version"]
        for slot, value in before_state["current_round"]["slots"].items()
    }
    before_directories = {
        slot: value["directory"]
        for slot, value in before_state["current_round"]["slots"].items()
    }

    monkeypatch.setattr(search, "_git_sha", lambda: "new-validator-revision")
    new_rules["enabled"] = True
    revalidated = search.run_episode_search(resume=output, revalidate_only=True)
    assert revalidated["status"] == "paused_revalidated_ready"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert state["model_calls"] == before_state["model_calls"] == 1
    assert state["current_round"]["repair_calls"] == 0
    assert state["current_round"]["phase"] == "preevaluate"
    assert {
        slot: value["version"]
        for slot, value in state["current_round"]["slots"].items()
    } == before_versions
    assert all(
        Path(before_directories[slot], "validation_report.json").is_file()
        for slot in search.EXPECTED_CANDIDATE_IDS
    )
    transition = state["numeric_operation_rules_transition"]
    assert transition["source_git_sha"] == next(
        iter(search.REVALIDATABLE_NUMERIC_RULE_SOURCE_REVISIONS)
    )
    assert transition["target_git_sha"] == "new-validator-revision"
    assert transition["target_rules_version"] == search.NUMERIC_OPERATION_RULES_VERSION
    record = json.loads(Path(transition["record"]).read_text(encoding="utf-8"))
    assert record["model_calls_before"] == record["model_calls_after"] == 1
    assert all(
        value["revalidated_status"] == "validated"
        for value in record["slots"].values()
    )

    monkeypatch.setattr(
        search,
        "evaluate_candidates",
        lambda *_args, **_kwargs: {
            "report": {
                "baseline": {"score": 1.0},
                "candidates": {
                    slot: {"score": 1.0, "examples": []}
                    for slot in search.EXPECTED_CANDIDATE_IDS
                },
                "selected_candidate_id": None,
                "selection_rule": "fixture",
            },
            "components": {},
            "selected_candidate_id": None,
        },
    )
    resumed = search.run_episode_search(resume=output, client=_MockClient([]))
    assert resumed["status"] == "completed_search_rounds_exhausted"
    assert resumed["model_calls"] == 1


def test_revalidation_incompatibility_does_not_modify_saved_run(tmp_path, monkeypatch):
    output = tmp_path / "incompatible-revalidation"
    output.mkdir()
    state = {
        "schema_version": search.SEARCH_RUN_SCHEMA_VERSION,
        "status": "paused_repairs_exhausted",
        "git_sha": next(iter(search.REVALIDATABLE_NUMERIC_RULE_SOURCE_REVISIONS)),
        "model_calls": 6,
        "rounds": [],
        "current_round": {
            "round": 1,
            "phase": "repair",
            "repair_calls": 5,
            "slots": {slot: {} for slot in search.EXPECTED_CANDIDATE_IDS},
            "evaluation": None,
            "training": None,
            "evaluation_result": None,
        },
        "settings": {"episode_sources": ["fixture"]},
        "dataset_provenance": {"dataset_content_sha256": "saved"},
    }
    (output / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    before = (output / "state.json").read_bytes()
    monkeypatch.setattr(search, "_git_sha", lambda: "new-validator-revision")
    monkeypatch.setattr(
        search,
        "load_complete_episode_dataset",
        lambda _paths: search.EpisodeDataset(
            arrays={}, episodes=(), fixed_metadata={}, provenance={"dataset_content_sha256": "changed"}
        ),
    )
    with pytest.raises(search.EpisodeSearchError, match="dataset content"):
        search.run_episode_search(resume=output, revalidate_only=True)
    assert (output / "state.json").read_bytes() == before
    assert not (output / "revalidations").exists()


def test_short_training_root_preflights_formal_windows_checkpoint_paths(tmp_path):
    deep = tmp_path
    while len(str(deep.resolve())) < 185:
        deep = deep / "deep-approved-artifact-source"
    source = _approved_artifact(deep)
    search_output = (
        search.ROOT
        / "results"
        / "llm_episode_searches"
        / "gpt-4o"
        / "design-20261005T190924Z-2405a2d6"
    )
    short_root = search._short_training_output_root(search_output, 1)
    run_directory = (
        short_root
        / search.SEARCH_METHOD_ID
        / "run-seed20260817-abcdef0-20261005T194457449086Z-321de4c0"
    )
    report = llm_checkpoint_path_preflight(
        run_directory,
        source,
        maximum_episode=1500,
        enforce_windows_limit=True,
    )
    by_scope = {item["scope"]: item for item in report["paths"]}
    assert report["passed"] is True
    assert report["longest_path"]["scope"] == "model_checkpoint_temporary"
    assert report["longest_path"]["artifact_relative_path"] in {
        "evaluation_report.json",
        "validation_report.json",
    }
    assert by_scope["model_checkpoint_temporary"]["path_length"] < 260
    assert by_scope["full_checkpoint_temporary"]["path_length"] < 260
    assert by_scope["run_artifact"]["path_length"] < 260

    previous_run = (
        search.ROOT
        / "results"
        / "llm_train"
        / "s-ba69f367a822"
        / "r01"
        / search.SEARCH_METHOD_ID
        / "run-seed20260817-fa948f1-20261005T194457449086Z-321de4c0"
    )
    with pytest.raises(LLMRuntimeError, match="Windows legacy path limit"):
        llm_checkpoint_path_preflight(
            previous_run,
            source,
            maximum_episode=1500,
            enforce_windows_limit=True,
        )

    copied_run = tmp_path / "copy-target" / run_directory.name
    copied_run.mkdir(parents=True)
    copied = copy_approved_artifact(source, copied_run)
    assert artifact_identity(copied) == artifact_identity(
        episode_training.load_approved_design(source)
    )
    assert (copied.directory / "artifact.json").is_file()


def test_training_adapter_abandons_initialization_shell_and_starts_fresh(
    tmp_path, monkeypatch
):
    artifact = _approved_artifact(tmp_path / "artifact")
    short_root = tmp_path / "short"
    legacy_root = tmp_path / "legacy"
    shell = legacy_root / search.SEARCH_METHOD_ID / "old-initialization-shell"
    (shell / "llm_artifact").mkdir(parents=True)
    (shell / "llm_artifact" / "candidate.py").write_text("partial", encoding="utf-8")
    write_run_status(shell, "PREPARING")
    design = episode_training.load_approved_design(artifact)
    recovery_state = episode_training._new_training_state(
        method_id=search.SEARCH_METHOD_ID,
        artifact_identity_record=artifact_identity(design),
        episodes=1500,
        seed=20260817,
        output=short_root.resolve(),
    )
    recovery_state["active_run_directory"] = str(shell.resolve())
    episode_training._write_json_atomic(
        short_root / episode_training.SEARCH_TRAINING_STATE_FILENAME,
        recovery_state,
    )
    calls = []

    def fail_new(command, *, cwd):
        calls.append(command)
        assert command[2] == search.SEARCH_METHOD_ID
        assert "resume" not in command
        raise episode_training.EpisodeTrainingError("fixture initialization failure")

    monkeypatch.setattr(episode_training, "_run", fail_new)
    result = episode_training.run_candidate_training(
        method_id=search.SEARCH_METHOD_ID,
        artifact=artifact,
        episodes=1500,
        seed=20260817,
        output_directory=short_root,
        legacy_output_directory=legacy_root,
        resume_record=None,
    )
    assert result["status"] == "incomplete"
    assert result["recovery_action"] == "fresh_initialization_failed"
    assert len(calls) == 1
    state = json.loads(
        (short_root / episode_training.SEARCH_TRAINING_STATE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    legacy_attempt = next(
        item for item in state["attempts"] if item.get("run_directory") == str(shell.resolve())
    )
    assert legacy_attempt["result"] == "abandoned_initialization_failure"
    assert legacy_attempt["evidence"]["partial_artifact_present"] is True


def test_training_adapter_resumes_checkpoint_but_blocks_progress_without_one(
    tmp_path, monkeypatch
):
    artifact = _approved_artifact(tmp_path / "artifact")
    output = tmp_path / "short"
    resumable = output / search.SEARCH_METHOD_ID / "resumable"
    checkpoint = resumable / "checkpoints" / "full" / "ep_0001"
    checkpoint.mkdir(parents=True)
    write_run_status(resumable, "PREPARING")
    copied = copy_approved_artifact(artifact, resumable)
    (resumable / "resolved_config.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "episodes": 1500,
                "llm_artifact_identity": artifact_identity(copied),
            }
        ),
        encoding="utf-8",
    )
    write_run_status(resumable, "RUNNING")
    commands = []

    def complete_resume(command, *, cwd):
        commands.append(command)
        assert command[2] == "resume"
        (resumable / "resolved_config.json").write_text(
            json.dumps({"status": "COMPLETED", "episodes": 1500}),
            encoding="utf-8",
        )
        write_run_status(resumable, "COMPLETED")
        return {"run_directory": str(resumable), "status": "COMPLETED"}

    monkeypatch.setattr(episode_training, "_run", complete_resume)
    monkeypatch.setattr(
        episode_training,
        "_training_summaries",
        lambda *_args, **_kwargs: [{"episode_range": [1, 1500]}],
    )
    result = episode_training.run_candidate_training(
        method_id=search.SEARCH_METHOD_ID,
        artifact=artifact,
        episodes=1500,
        seed=20260817,
        output_directory=output,
        resume_record={"run_directory": str(resumable)},
    )
    assert result["status"] == "complete"
    assert len(commands) == 1

    blocked_root = tmp_path / "blocked"
    blocked = blocked_root / search.SEARCH_METHOD_ID / "started-without-checkpoint"
    blocked.mkdir(parents=True)
    write_run_status(blocked, "PREPARING")
    copy_approved_artifact(artifact, blocked)
    write_run_status(blocked, "RUNNING")
    monkeypatch.setattr(
        episode_training,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not silently restart")
        ),
    )
    with pytest.raises(
        episode_training.EpisodeTrainingError,
        match="progress exists but no resumable checkpoint",
    ):
        episode_training.run_candidate_training(
            method_id=search.SEARCH_METHOD_ID,
            artifact=artifact,
            episodes=1500,
            seed=20260817,
            output_directory=blocked_root,
            resume_record={"run_directory": str(blocked)},
        )


def test_training_adapter_does_not_retrain_when_checkpoint_resume_fails(
    tmp_path, monkeypatch
):
    artifact = _approved_artifact(tmp_path / "artifact")
    output = tmp_path / "short"
    damaged = output / search.SEARCH_METHOD_ID / "damaged-checkpoint"
    (damaged / "checkpoints" / "full" / "ep_0001").mkdir(parents=True)
    write_run_status(damaged, "PREPARING")
    copied = copy_approved_artifact(artifact, damaged)
    (damaged / "resolved_config.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "episodes": 1500,
                "llm_artifact_identity": artifact_identity(copied),
            }
        ),
        encoding="utf-8",
    )
    write_run_status(damaged, "RUNNING")
    commands = []

    def reject_resume(command, *, cwd):
        commands.append(command)
        assert command[2] == "resume"
        raise episode_training.EpisodeTrainingError("checkpoint is damaged")

    monkeypatch.setattr(episode_training, "_run", reject_resume)
    result = episode_training.run_candidate_training(
        method_id=search.SEARCH_METHOD_ID,
        artifact=artifact,
        episodes=1500,
        seed=20260817,
        output_directory=output,
        resume_record={"run_directory": str(damaged)},
    )
    assert result["status"] == "incomplete"
    assert result["recovery_action"] == "resume_failed_no_retraining"
    assert result["run_directory"] == str(damaged.resolve())
    assert len(commands) == 1

    state = json.loads(
        (output / episode_training.SEARCH_TRAINING_STATE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert state["active_run_directory"] == str(damaged.resolve())
    assert state["attempts"][-1]["result"] == "resume_failed_no_retraining"
    assert "checkpoint is damaged" in state["attempts"][-1]["error"]


def test_explicit_restart_preserves_progress_run_and_starts_once(
    tmp_path, monkeypatch
):
    artifact = _approved_artifact(tmp_path / "artifact")
    previous_root = tmp_path / "previous"
    failed = previous_root / search.SEARCH_METHOD_ID / "failed-after-episode-50"
    failed.mkdir(parents=True)
    write_run_status(failed, "PREPARING")
    copied = copy_approved_artifact(artifact, failed)
    (failed / "resolved_config.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "episodes": 1500,
                "llm_artifact_identity": artifact_identity(copied),
            }
        ),
        encoding="utf-8",
    )
    (failed / "training_history.jsonl").write_text(
        json.dumps({"episode": 50}) + "\n", encoding="utf-8"
    )
    write_run_status(failed, "RUNNING")
    write_run_status(
        failed,
        "FAILED",
        exception=RuntimeError("[WinError 3] checkpoints\\models path"),
    )

    monkeypatch.setattr(
        episode_training,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary resume must not restart")
        ),
    )
    with pytest.raises(
        episode_training.EpisodeTrainingError,
        match="progress exists but no resumable checkpoint",
    ):
        episode_training.run_candidate_training(
            method_id=search.SEARCH_METHOD_ID,
            artifact=artifact,
            episodes=1500,
            seed=20260817,
            output_directory=tmp_path / "normal-resume",
            prior_output_directories=[previous_root],
            resume_record={"run_directory": str(failed)},
        )

    replacement_root = tmp_path / "replacement"
    calls = []

    def complete_fresh(command, *, cwd):
        calls.append(command)
        assert command[2] == search.SEARCH_METHOD_ID
        assert "resume" not in command
        replacement = (
            replacement_root
            / search.SEARCH_METHOD_ID
            / "run-seed20260817-new-replacement"
        )
        replacement.mkdir(parents=True)
        write_run_status(replacement, "PREPARING")
        copy_approved_artifact(artifact, replacement)
        (replacement / "resolved_config.json").write_text(
            json.dumps({"status": "COMPLETED", "episodes": 1500}),
            encoding="utf-8",
        )
        write_run_status(replacement, "RUNNING")
        write_run_status(replacement, "COMPLETED")
        return {"run_directory": str(replacement), "status": "COMPLETED"}

    monkeypatch.setattr(episode_training, "_run", complete_fresh)
    monkeypatch.setattr(
        episode_training,
        "_training_summaries",
        lambda *_args, **_kwargs: [{"episode_range": [1, 1500]}],
    )
    result = episode_training.run_candidate_training(
        method_id=search.SEARCH_METHOD_ID,
        artifact=artifact,
        episodes=1500,
        seed=20260817,
        output_directory=replacement_root,
        prior_output_directories=[previous_root],
        resume_record={"run_directory": str(failed)},
        restart_from_scratch=True,
        restart_authorization_id="fixture-restart-operation",
        restart_failed_run_directory=failed,
    )
    assert result["status"] == "complete"
    assert len(calls) == 1
    assert failed.is_dir()
    assert (failed / "training_history.jsonl").is_file()
    state = json.loads(
        (
            replacement_root
            / episode_training.SEARCH_TRAINING_STATE_FILENAME
        ).read_text(encoding="utf-8")
    )
    failed_attempt = next(
        item
        for item in state["attempts"]
        if item.get("run_directory") == str(failed.resolve())
    )
    assert failed_attempt["result"] == "abandoned_for_explicit_restart"
    assert state["explicit_restart_history"][0]["restart_episode"] == 1
    assert (
        state["explicit_restart_history"][0]["operation_id"]
        == "fixture-restart-operation"
    )


def test_stale_search_shell_does_not_override_recorded_checkpoint_run(
    tmp_path, monkeypatch
):
    artifact = _approved_artifact(tmp_path / "artifact")
    output = tmp_path / "output"
    stale = output / search.SEARCH_METHOD_ID / "stale-shell"
    stale.mkdir(parents=True)
    write_run_status(stale, "PREPARING")
    replacement = output / search.SEARCH_METHOD_ID / "replacement-with-checkpoint"
    (replacement / "checkpoints" / "full" / "ep_0050").mkdir(parents=True)
    write_run_status(replacement, "PREPARING")
    copied = copy_approved_artifact(artifact, replacement)
    (replacement / "resolved_config.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "episodes": 1500,
                "llm_artifact_identity": artifact_identity(copied),
            }
        ),
        encoding="utf-8",
    )
    write_run_status(replacement, "RUNNING")
    state = episode_training._new_training_state(
        method_id=search.SEARCH_METHOD_ID,
        artifact_identity_record=artifact_identity(
            episode_training.load_approved_design(artifact)
        ),
        episodes=1500,
        seed=20260817,
        output=output.resolve(),
    )
    state["active_run_directory"] = str(replacement.resolve())
    state["attempts"].append(
        {
            "attempt": 1,
            "run_directory": str(stale.resolve()),
            "result": "abandoned_for_explicit_restart",
        }
    )
    state["checkpoint_restart_authorization"] = {
        "operation_id": "fixture-parent-not-updated",
        "failed_run_directory": str(stale.resolve()),
        "launch_status": "replacement_run_created",
        "replacement_run_directory": str(replacement.resolve()),
    }
    episode_training._write_json_atomic(
        output / episode_training.SEARCH_TRAINING_STATE_FILENAME, state
    )
    commands = []

    def complete_resume(command, *, cwd):
        commands.append(command)
        assert command[2] == "resume"
        assert Path(command[3]).resolve() == replacement.resolve()
        (replacement / "resolved_config.json").write_text(
            json.dumps({"status": "COMPLETED", "episodes": 1500}),
            encoding="utf-8",
        )
        write_run_status(replacement, "COMPLETED")
        return {"run_directory": str(replacement), "status": "COMPLETED"}

    monkeypatch.setattr(episode_training, "_run", complete_resume)
    monkeypatch.setattr(
        episode_training,
        "_training_summaries",
        lambda *_args, **_kwargs: [{"episode_range": [1, 1500]}],
    )
    result = episode_training.run_candidate_training(
        method_id=search.SEARCH_METHOD_ID,
        artifact=artifact,
        episodes=1500,
        seed=20260817,
        output_directory=output,
        resume_record={"run_directory": str(stale)},
        restart_from_scratch=True,
        restart_authorization_id="fixture-parent-not-updated",
        restart_failed_run_directory=stale,
    )
    assert result["status"] == "complete"
    assert len(commands) == 1
    assert result["run_directory"] == str(replacement.resolve())


def test_created_restart_with_progress_but_no_checkpoint_is_not_restarted_again(
    tmp_path, monkeypatch
):
    artifact = _approved_artifact(tmp_path / "artifact")
    output = tmp_path / "output"
    failed = output / search.SEARCH_METHOD_ID / "authorized-failed-run"
    replacement = output / search.SEARCH_METHOD_ID / "replacement-with-progress"
    for directory in (failed, replacement):
        directory.mkdir(parents=True)
        write_run_status(directory, "PREPARING")
        copy_approved_artifact(artifact, directory)
        (directory / "resolved_config.json").write_text(
            json.dumps({"status": "FAILED", "episodes": 1500}),
            encoding="utf-8",
        )
        (directory / "training_history.jsonl").write_text(
            json.dumps({"episode": 1}) + "\n", encoding="utf-8"
        )
        write_run_status(directory, "RUNNING")
        write_run_status(directory, "FAILED", exception=RuntimeError("fixture"))

    state = episode_training._new_training_state(
        method_id=search.SEARCH_METHOD_ID,
        artifact_identity_record=artifact_identity(
            episode_training.load_approved_design(artifact)
        ),
        episodes=1500,
        seed=20260817,
        output=output.resolve(),
    )
    state["active_run_directory"] = str(replacement.resolve())
    state["attempts"].append(
        {
            "attempt": 1,
            "run_directory": str(failed.resolve()),
            "result": "abandoned_for_explicit_restart",
        }
    )
    episode_training._write_json_atomic(
        output / episode_training.SEARCH_TRAINING_STATE_FILENAME, state
    )
    monkeypatch.setattr(
        episode_training,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replacement training must not be launched again")
        ),
    )

    with pytest.raises(
        episode_training.EpisodeTrainingError,
        match="exactly its authorized progress-without-checkpoint run",
    ):
        episode_training.run_candidate_training(
            method_id=search.SEARCH_METHOD_ID,
            artifact=artifact,
            episodes=1500,
            seed=20260817,
            output_directory=output,
            resume_record={"run_directory": str(replacement)},
            restart_from_scratch=True,
            restart_authorization_id="fixture-restart-operation",
            restart_failed_run_directory=failed,
        )
    assert failed.is_dir()
    assert replacement.is_dir()


def test_created_restart_initialization_shell_is_not_duplicated(tmp_path, monkeypatch):
    artifact = _approved_artifact(tmp_path / "artifact")
    output = tmp_path / "output"
    failed = output / search.SEARCH_METHOD_ID / "authorized-failed-run"
    failed.mkdir(parents=True)
    write_run_status(failed, "PREPARING")
    copy_approved_artifact(artifact, failed)
    (failed / "training_history.jsonl").write_text(
        json.dumps({"episode": 50}) + "\n", encoding="utf-8"
    )
    write_run_status(failed, "RUNNING")
    write_run_status(failed, "FAILED", exception=RuntimeError("fixture"))
    replacement = output / search.SEARCH_METHOD_ID / "replacement-shell"
    replacement.mkdir(parents=True)
    write_run_status(replacement, "PREPARING")
    monkeypatch.setattr(
        episode_training,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a second replacement must not be launched")
        ),
    )

    with pytest.raises(
        episode_training.EpisodeTrainingError,
        match="replacement run was already created",
    ):
        episode_training.run_candidate_training(
            method_id=search.SEARCH_METHOD_ID,
            artifact=artifact,
            episodes=1500,
            seed=20260817,
            output_directory=output,
            resume_record={"run_directory": str(failed)},
            restart_from_scratch=True,
            restart_authorization_id="fixture-restart-operation",
            restart_failed_run_directory=failed,
        )
    recovery = json.loads(
        (output / episode_training.SEARCH_TRAINING_STATE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert recovery["status"] == "blocked_replacement_initialization_incomplete"
    assert any(
        item.get("run_directory") == str(replacement.resolve())
        and item.get("result") == "blocked_replacement_initialization_incomplete"
        for item in recovery["attempts"]
    )


def test_checkpoint_restart_state_revision_migration_is_bounded_to_fix_revision():
    state = {
        "git_sha": "original-search-revision",
        "checkpoint_path_restart_transition": {
            "source_git_sha": search.RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION,
            "target_git_sha": search.CHECKPOINT_RESTART_STATE_SOURCE_REVISION,
            "status": "compatible",
            "reason": "checkpoint_path_shortening_restart_from_episode_1",
        },
    }
    assert search._revision_transition_allows_resume(state, "state-management-fix")
    state["checkpoint_restart_state_management_transition"] = {
        "source_git_sha": search.CHECKPOINT_RESTART_STATE_SOURCE_REVISION,
        "target_git_sha": "state-management-fix",
        "status": "compatible",
        "reason": "persist_checkpoint_restart_operation_state",
    }
    assert search._revision_transition_allows_resume(state, "state-management-fix")
    assert not search._revision_transition_allows_resume(state, "unrelated-later-revision")


def test_bounded_training_recovery_preserves_selection_and_skips_model_and_ranking(
    tmp_path, monkeypatch
):
    output = (tmp_path / "saved-search").resolve()
    output.mkdir()
    dataset = _dataset()
    constants = {}
    preflight = _preflight_fixture()
    evaluation = {
        "baseline": {"score": 0.5},
        "candidates": {
            slot: {"score": 0.75 if slot == "candidate_1" else 0.25}
            for slot in search.EXPECTED_CANDIDATE_IDS
        },
        "selected_candidate_id": "candidate_1",
    }
    round_dir = output / "round_01"
    round_dir.mkdir()
    (round_dir / "pretraining_evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    artifact = _approved_artifact(
        round_dir / "artifact-root",
        provenance={
            "search_run": str(output),
            "search_round": 1,
            "candidate_slot": "candidate_1",
            "dataset": dataset.provenance,
        },
    )
    design = episode_training.load_approved_design(artifact)
    slots = {
        slot: {
            "status": "validated",
            "version": index,
            "candidate": design.candidate if slot == "candidate_1" else {"candidate_name": slot},
            "submission": {"candidate_id": slot},
            "directory": str(round_dir / slot),
        }
        for index, slot in enumerate(search.EXPECTED_CANDIDATE_IDS, start=1)
    }
    config = search._role_config(
        provider="openai",
        model="fixture-model",
        base_url=None,
        context_length=50000,
        max_output_tokens=4096,
        temperature=0.3,
        seed=20260817,
        reasoning_effort=None,
        timeout=10,
        connect_timeout=10,
        total_timeout=20,
        progress_interval=1,
    )
    state = {
        "schema_version": search.SEARCH_RUN_SCHEMA_VERSION,
        "prompt_version": search.SEARCH_PROMPT_VERSION,
        "status": "running",
        "git_sha": next(iter(search.RECOVERABLE_TRAINING_INITIALIZATION_SOURCE_REVISIONS)),
        "output_directory": str(output),
        "search_round": 1,
        "model_calls": 3,
        "rounds": [],
        "current_round": {
            "round": 1,
            "phase": "train",
            "repair_calls": 2,
            "slots": slots,
            "evaluation": evaluation,
            "selected_candidate_id": "candidate_1",
            "approved_artifact": str(artifact),
            "training": None,
            "evaluation_result": None,
        },
        "stop_reason": None,
        "settings": {
            "episode_sources": ["fixture-source"],
            "model": config,
            "beta": 1.0,
            "worker_timeout": 10.0,
            "evaluation_lambda": {"value": 0.0},
            "max_search_rounds": 1,
            "max_repairs_per_round": 5,
            "ee_tolerance": 1e-12,
            "reward_tolerance": 1e-12,
            "baseline_run": "fixture-baseline",
            "baseline_evaluation": "fixture-evaluation",
            "evaluation_manifest": "fixture-manifest",
            "train_episodes": 1500,
            "evaluation_episodes": 100,
            "evaluation_roi_count": 8,
            "evaluation_area_m": [1000.0, 1000.0],
        },
        "dataset_provenance": dataset.provenance,
        "baseline_preflight": preflight,
    }
    (output / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (output / "dataset.json").write_text(json.dumps(dataset.provenance), encoding="utf-8")
    (output / "constants.json").write_text(json.dumps(constants), encoding="utf-8")
    (output / "baseline_preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: constants)
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)
    monkeypatch.setattr(
        search,
        "evaluate_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pretraining ranking must not rerun")
        ),
    )
    old_training_root = (tmp_path / "fa948f1-training").resolve()
    replacement_root = (tmp_path / "checkpoint-path-fix-training").resolve()
    failed_run = (
        old_training_root
        / search.SEARCH_METHOD_ID
        / "run-seed20260817-fa948f1-formal-name"
    )
    failed_run.mkdir(parents=True)
    write_run_status(failed_run, "PREPARING")
    copy_approved_artifact(artifact, failed_run)
    (failed_run / "resolved_config.json").write_text(
        json.dumps({"status": "FAILED", "episodes": 1500}), encoding="utf-8"
    )
    (failed_run / "training_history.jsonl").write_text(
        json.dumps({"episode": 50}) + "\n", encoding="utf-8"
    )
    (failed_run / "llm_training_episode_metrics.jsonl").write_text(
        json.dumps({"episode": 50}) + "\n", encoding="utf-8"
    )
    write_run_status(failed_run, "RUNNING")
    write_run_status(
        failed_run,
        "FAILED",
        exception=RuntimeError("[WinError 3] checkpoints\\models path failure"),
    )
    current_revision = {"value": search.RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION}
    monkeypatch.setattr(search, "_git_sha", lambda: current_revision["value"])
    monkeypatch.setattr(
        search,
        "_short_training_output_root",
        lambda _output, _round: (
            old_training_root
            if current_revision["value"]
            == search.RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION
            else replacement_root
        ),
    )
    calls = []

    interrupt_restart = {"enabled": False}

    def training_runner(**kwargs):
        calls.append(kwargs)
        if interrupt_restart["enabled"] and kwargs["restart_from_scratch"]:
            interrupt_restart["enabled"] = False
            raise KeyboardInterrupt("fixture interruption before subprocess launch")
        return {
            "status": "incomplete",
            "reason": "fixture stop before formal training",
            "run_directory": str(failed_run),
            "episodes": 1500,
        }

    result = search.run_episode_search(
        resume=output,
        recover_training_initialization=True,
        client=SimpleNamespace(
            chat=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("model must not be called")
            )
        ),
        training_runner=training_runner,
    )
    assert result["status"] == "paused_training"
    assert result["model_calls"] == 3
    assert len(calls) == 1
    assert calls[0]["artifact"] == str(artifact)
    assert Path(calls[0]["output_directory"]) == old_training_root
    saved = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert saved["current_round"]["selected_candidate_id"] == "candidate_1"
    assert saved["current_round"]["repair_calls"] == 2
    transition = saved["training_initialization_recovery_transition"]
    assert transition["source_git_sha"] == state["git_sha"]
    assert (
        transition["target_git_sha"]
        == search.RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION
    )
    assert transition["approved_artifact_identity"] == artifact_identity(design)

    current_revision["value"] = "checkpoint-path-fix-revision"
    interrupt_restart["enabled"] = True
    with pytest.raises(KeyboardInterrupt, match="before subprocess launch"):
        search.run_episode_search(
            resume=output,
            restart_failed_training=True,
            client=SimpleNamespace(
                chat=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("model must not be called")
                )
            ),
            training_runner=training_runner,
        )
    saved_after_authorization = json.loads(
        (output / "state.json").read_text(encoding="utf-8")
    )
    assert (
        saved_after_authorization["checkpoint_path_restart_transition"][
            "restart_status"
        ]
        == "authorized_not_started"
    )

    restarted = search.run_episode_search(
        resume=output,
        client=SimpleNamespace(
            chat=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("model must not be called")
            )
        ),
        training_runner=training_runner,
    )
    assert restarted["status"] == "paused_training"
    assert restarted["model_calls"] == 3
    assert len(calls) == 3
    assert calls[1]["restart_from_scratch"] is True
    assert calls[2]["restart_from_scratch"] is True
    assert calls[1]["restart_authorization_id"] == calls[2][
        "restart_authorization_id"
    ]
    assert str(old_training_root) in calls[2]["prior_output_directories"]
    assert Path(calls[2]["output_directory"]) == replacement_root
    saved = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert saved["current_round"]["selected_candidate_id"] == "candidate_1"
    assert saved["current_round"]["repair_calls"] == 2
    restart = saved["checkpoint_path_restart_transition"]
    assert restart["source_git_sha"] == search.RESTARTABLE_CHECKPOINT_PATH_RECOVERY_REVISION
    assert restart["target_git_sha"] == "checkpoint-path-fix-revision"
    assert restart["restart_status"] == "authorized_not_started"
    assert restart["failed_run_directory"] == str(failed_run.resolve())
    assert restart["completed_episode_evidence"]["training_history.jsonl"][
        "maximum_episode"
    ] == 50


def test_completed_search_extends_from_saved_preevaluation_without_repeating_round(tmp_path, monkeypatch):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    _patch_preflight(monkeypatch)
    monkeypatch.setattr(
        search,
        "validate_search_candidate",
        lambda submission, **_kwargs: (
            {"candidate_name": submission["candidate_id"], "features": submission["features"], "code": submission["code"]},
            {"status": "passed"},
            np.zeros((3, 1)),
        ),
    )
    monkeypatch.setattr(
        search,
        "evaluate_candidates",
        lambda *_args, **_kwargs: {
            "report": {
                "baseline": {"score": 1.0},
                "candidates": {slot: {"score": 1.0, "examples": []} for slot in search.EXPECTED_CANDIDATE_IDS},
                "selected_candidate_id": None,
                "selection_rule": "fixture",
            },
            "components": {},
            "selected_candidate_id": None,
        },
    )

    def batch(round_number):
        return json.dumps({"candidates": [
            {
                "candidate_id": slot,
                "design_summary": f"round-{round_number}",
                "features": [{"name": f"{slot}-{round_number}", "description": "fixture", "reward_weight": 0.0}],
                "code": f"def compute_extra_state(obs, constants):\n    return [{round_number + index / 10.0}]",
            }
            for index, slot in enumerate(search.EXPECTED_CANDIDATE_IDS)
        ]})

    output = tmp_path / "extend-rounds"
    first_client = _MockClient([batch(1)])
    first = search.run_episode_search(
        episode_sources=[tmp_path / "source"], provider="openai", model="fixture/model",
        evaluation_lambda=0.0, max_search_rounds=1,
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-evaluation",
        evaluation_manifest=tmp_path / "manifest.json",
        output_dir=output, client=first_client,
    )
    assert first["status"] == "completed_search_rounds_exhausted"
    second_client = _MockClient([batch(2)])
    second = search.run_episode_search(
        resume=output, max_search_rounds=2, client=second_client
    )
    assert second["status"] == "completed_search_rounds_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert [item["round"] for item in state["rounds"]] == [1, 2]
    assert state["model_calls"] == 2
    assert "Evaluation rules and scores" in second_client.prompts[0]
    assert "candidate_1" in second_client.prompts[0]


def test_baseline_preflight_binds_run_checkpoint_manifest_and_episode_rows(tmp_path, monkeypatch):
    scenario_ids = [f"scenario-{index:03d}" for index in range(100)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"fixture": true}', encoding="utf-8")
    manifest = SimpleNamespace(
        split="test",
        episode_count=100,
        generation_profile={"fixed_num_gt": 8},
        environment_width_m=1000,
        environment_height_m=1000,
        environment_size_m=None,
        content_hash="manifest-content-hash",
        episodes=tuple({"scenario_id": value} for value in scenario_ids),
    )
    monkeypatch.setattr(episode_training.ScenarioManifest, "load", lambda _path: manifest)

    baseline_run = tmp_path / "baseline-run"
    checkpoint = baseline_run / "checkpoints" / "models" / "ep_1500"
    checkpoint.mkdir(parents=True)
    provenance = {field: f"fixture-{field}" for field in CHECKPOINT_PROVENANCE_FIELDS}
    calls = []

    def resolve(run, checkpoint_episode, *, expected_method):
        calls.append((Path(run), checkpoint_episode, expected_method))
        return {
            "run_dir": baseline_run.resolve(),
            "training_run_id": baseline_run.name,
            "checkpoint": checkpoint.resolve(),
            "checkpoint_artifact_provenance": provenance,
        }

    monkeypatch.setattr(episode_training, "resolve_training_run_checkpoint", resolve)
    rows_path = tmp_path / "baseline-episodes.jsonl"
    rows = [
        {
            "scenario_id": scenario_id,
            "energy_efficiency_mbit_per_j": float(index + 1),
            "total_timely_useful_mbits": 10.0,
            "total_mobility_energy_j": 2.0,
            "fov_eligible_packets": 2,
            "fov_violation_packets": 1,
            "com_eligible_packets": 4,
            "com_violation_packets": 1,
        }
        for index, scenario_id in enumerate(scenario_ids)
    ]
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    point = {
        "training_run_id": baseline_run.name,
        "checkpoint_episode": 1500,
        **provenance,
        "evaluation_episode_count": 100,
        "fixed_num_gt": 8,
        "evaluation_environment_width_m": 1000.0,
        "evaluation_environment_height_m": 1000.0,
        "evaluation_episode_horizon_s": 60,
        "scenario_manifest_hash": manifest.content_hash,
        "scenario_ids": scenario_ids,
        "outputs": {"per_episode_jsonl": str(rows_path)},
    }
    metadata_path = tmp_path / "paper_evaluation_metadata.json"

    def write_metadata(**changes):
        value = {
            "semantic_suite": "fixed_roi",
            "method_id": "td3_dinkelbach",
            "points": [{**point, **changes}],
        }
        metadata_path.write_text(json.dumps(value), encoding="utf-8")

    write_metadata()
    preflight = episode_training.validate_baseline_preflight(
        baseline_run=baseline_run,
        baseline_evaluation=metadata_path,
        manifest=manifest_path,
    )
    assert calls == [(baseline_run, 1500, "td3_dinkelbach")]
    assert preflight["status"] == "passed"
    assert preflight["scenario_ids"] == scenario_ids
    assert preflight["metrics"]["end_to_end_delay_violation_probability"]["COM"]["value"] == 0.25

    write_metadata(training_run_id="other-run")
    with pytest.raises(episode_training.EpisodeTrainingError, match="different training run"):
        episode_training.validate_baseline_preflight(
            baseline_run=baseline_run,
            baseline_evaluation=metadata_path,
            manifest=manifest_path,
        )
    write_metadata(checkpoint_episode=1499)
    with pytest.raises(episode_training.EpisodeTrainingError, match="checkpoint episode 1500"):
        episode_training.validate_baseline_preflight(
            baseline_run=baseline_run,
            baseline_evaluation=metadata_path,
            manifest=manifest_path,
        )
    write_metadata(scenario_manifest_hash="wrong-manifest")
    with pytest.raises(episode_training.EpisodeTrainingError, match="manifest hash"):
        episode_training.validate_baseline_preflight(
            baseline_run=baseline_run,
            baseline_evaluation=metadata_path,
            manifest=manifest_path,
        )


def test_baseline_preflight_failure_happens_before_model_call(tmp_path, monkeypatch):
    dataset = _dataset()
    monkeypatch.setattr(search, "load_complete_episode_dataset", lambda _paths: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "render_common_prompt", lambda **_kwargs: "common")
    monkeypatch.setattr(
        episode_training,
        "validate_baseline_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(
            episode_training.EpisodeTrainingError("preflight fixture failure")
        ),
    )
    client = _MockClient([])
    with pytest.raises(search.EpisodeSearchError, match="preflight fixture failure"):
        search.run_episode_search(
            episode_sources=[tmp_path / "source"],
            provider="openai",
            model="fixture/model",
            evaluation_lambda=0.0,
            baseline_run=tmp_path / "baseline-run",
            baseline_evaluation=tmp_path / "baseline-evaluation",
            evaluation_manifest=tmp_path / "manifest.json",
            output_dir=tmp_path / "preflight-fails",
            client=client,
        )
    assert client.prompts == []
    failure = json.loads(
        (tmp_path / "preflight-fails" / "baseline_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["status"] == "failed"
