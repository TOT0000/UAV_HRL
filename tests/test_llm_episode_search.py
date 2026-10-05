import json
from pathlib import Path

import numpy as np
import pytest

import llm_episode_search as search
from experiment_config import MethodSpec
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
    for episode in range(1, 201):
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
    summary = search.summarize_training_blocks(rows)
    assert len(summary) == 2
    assert summary[0]["episode_range"] == [1, 100]
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
    assert tuple(parsed) == search.EXPECTED_CANDIDATE_IDS
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

    initial = batch({"candidate_2", "candidate_4"})
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

    def train_runner(**kwargs):
        training_calls.append(kwargs)
        return {
            "status": "complete",
            "run_directory": str(tmp_path / f"trained-{len(training_calls)}"),
            "summaries": [{"episode_range": [1, 100]}],
        }

    def evaluation_runner(**kwargs):
        evaluation_calls.append(kwargs)
        return {
            "status": "complete",
            "mean_episode_energy_efficiency_mbit_per_j": 2.0 if len(evaluation_calls) == 1 else 1.0,
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
    assert all(call["episodes"] == 1500 for call in training_calls)
    assert all(call["episodes"] == 100 for call in evaluation_calls)
    assert all(call["roi_count"] == 8 for call in evaluation_calls)
    assert all(call["environment_size_m"] == (1000.0, 1000.0) for call in evaluation_calls)
    state = json.loads((tmp_path / "run-trained" / "state.json").read_text(encoding="utf-8"))
    assert state["best_trained_candidate"]["round"] == 1
    assert state["best_trained_candidate"]["mean_episode_energy_efficiency_mbit_per_j"] == 2.0
    assert "Best trained candidate so far" in client.prompts[1]
