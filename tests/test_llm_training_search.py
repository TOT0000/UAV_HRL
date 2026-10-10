import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import llm_training_search as search
from llm_episode_training import _evaluation_metrics_summary
from llm_episode_search import summarize_training_blocks
from scenario_manifest import generate_manifest


def _submission(candidate_id, values, weights):
    expressions = ", ".join(str(float(value)) for value in values)
    return {
        "candidate_id": candidate_id,
        "design_summary": f"fixture {candidate_id}",
        "features": [
            {
                "name": f"feature_{index}",
                "description": "bounded fixture feature",
                "reward_weight": float(weight),
            }
            for index, weight in enumerate(weights)
        ],
        "code": (
            "def compute_extra_state(obs, constants):\n"
            f"    return [{expressions}]\n"
        ),
    }


def _normalized(submission):
    return {
        "candidate_name": submission["candidate_id"],
        "features": [
            {
                "name": item["name"],
                "description": item["description"],
                "reward_weight": float(item["reward_weight"]),
            }
            for item in submission["features"]
        ],
        "code": submission["code"],
    }


def test_direction_checks_only_enforce_reliably_determined_changes():
    reference = _normalized(_submission("base", [0.4, 0.5], [1.0, 2.0]))
    added = _normalized(_submission("add", [0.4, 0.5, 0.6], [1.0, 2.0, 3.0]))
    removed = _normalized(_submission("remove", [0.4], [1.0]))
    reweighted = _normalized(_submission("weight", [0.4, 0.5], [1.0, 3.0]))
    redesigned = _normalized(_submission("redesign", [0.9], [-1.0]))

    assert search.validate_direction("add", added, reference)["status"] == "passed"
    assert search.validate_direction("remove", removed, reference)["status"] == "passed"
    assert search.validate_direction("reweight", reweighted, reference)["status"] == "passed"
    assert search.validate_direction("redesign", redesigned, reference)["status"] == "passed"

    changed_formula = _normalized(
        _submission("bad-weight", [0.4, 0.6], [1.0, 3.0])
    )
    report = search.validate_direction("reweight", changed_formula, reference)
    assert report["status"] == "failed"
    assert "reweight direction changed feature computation" in report["issues"]

    one = _normalized(_submission("one", [0.4], [1.0]))
    replacement = _normalized(_submission("replacement", [0.7], [1.0]))
    report = search.validate_direction("remove", replacement, one)
    assert report["status"] == "passed"
    assert report["single_feature_remove_exception"] == (
        "single_feature_reference_uses_simpler_replacement"
    )


def test_duplicate_fingerprint_ignores_display_text_and_formatting():
    left = _normalized(_submission("left", [0.4], [2.0]))
    right = _normalized(_submission("right", [0.4], [2.0]))
    right["features"][0]["name"] = "renamed"
    right["features"][0]["description"] = "different prose"
    right["code"] = "# comment\n\ndef compute_extra_state(obs, constants):\n    return [0.4]\n"
    assert search.candidate_computation_fingerprint(left) == (
        search.candidate_computation_fingerprint(right)
    )


def test_source_bundle_and_prompt_templates_include_current_code_without_proxy_gates():
    bundle = search.build_environment_source_bundle()
    assert bundle["git_sha"]
    assert bundle["bundle_sha256"]
    symbols = {
        item["symbol"]
        for values in bundle["excerpts"].values()
        for item in values
    }
    assert "get_global_movement_state" in symbols
    assert "movement_mask_from_state" in symbols
    assert "PacketEngine.inject_packets" in symbols
    assert "_interval_reward" in symbols
    for values in bundle["excerpts"].values():
        for item in values:
            assert item["sha256"]
            assert item["line_start"] <= item["line_end"]

    prompt_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            search.COMMON_TEMPLATE,
            search.INITIAL_TEMPLATE,
            search.FEEDBACK_TEMPLATE,
            *search.DIRECTION_TEMPLATES.values(),
        )
    )
    assert "Return one complete candidate" in prompt_text
    assert "arithmetic mean" in prompt_text
    assert "Lipschitz" not in prompt_text
    assert "ordering consistency" not in prompt_text
    assert "reviewer" not in prompt_text.lower()


def test_800_episode_block_summary_is_hand_checkable():
    rows = []
    for episode in range(1, 801):
        rows.append(
            {
                "episode": episode,
                "candidate_artifact_id": "fixture",
                "feature_names": ["a", "b"],
                "feature_reward_weights": [2.0, -1.0],
                "beta": 1.0,
                "feature_contribution_sums": [2.0, -0.5],
                "base_reward_sum": float(episode),
                "extra_reward_sum": 1.5,
                "combined_reward_sum": float(episode) + 1.5,
                "timely_mbits": 10.0,
                "movement_energy_j": 5.0,
                "energy_efficiency_mbit_per_j": 2.0,
                "roi_count": 8,
                "exploration": {"noise": 0.2},
                "dinkelbach_lambda_used": episode / 1000.0,
            }
        )
    summaries = summarize_training_blocks(
        rows, block_size=200, expected_episodes=800
    )
    assert [item["episode_range"] for item in summaries] == [
        [1, 200],
        [201, 400],
        [401, 600],
        [601, 800],
    ]
    assert summaries[0]["energy_efficiency_mbit_per_j"]["mean"] == 2.0
    assert summaries[0]["timely_useful_delivery_mbit"]["mean"] == 10.0
    assert summaries[0]["movement_energy_j"]["mean"] == 5.0
    assert summaries[0]["feature_contributions"]["a"] == {
        "mean": 2.0,
        "std": 0.0,
    }
    assert summaries[0]["extra_reward_sum"]["mean"] == 1.5


def test_candidate_score_is_arithmetic_mean_of_episode_ee_not_ratio_of_totals():
    rows = [
        {
            "energy_efficiency_mbit_per_j": 1.0,
            "total_timely_useful_mbits": 1.0,
            "total_mobility_energy_j": 1.0,
            "fov_eligible_packets": 1,
            "fov_violation_packets": 0,
            "com_eligible_packets": 1,
            "com_violation_packets": 0,
        },
        {
            "energy_efficiency_mbit_per_j": 3.0,
            "total_timely_useful_mbits": 300.0,
            "total_mobility_energy_j": 100.0,
            "fov_eligible_packets": 1,
            "fov_violation_packets": 0,
            "com_eligible_packets": 1,
            "com_violation_packets": 0,
        },
    ]
    summary = _evaluation_metrics_summary(rows)
    ratio_of_totals = (
        summary["timely_useful_delivery_mbit"]["total"]
        / summary["movement_energy_j"]["total"]
    )
    assert summary["episode_energy_efficiency_mbit_per_j"]["mean"] == 2.0
    assert summary["episode_energy_efficiency_mbit_per_j"]["mean"] != pytest.approx(
        ratio_of_totals
    )


def test_training_fairness_uses_baseline_prefix_and_keeps_1000_episode_decay(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    baseline_manifest = generate_manifest("train", 20260817, 3, num_gt=8)
    candidate_manifest = generate_manifest("train", 20260817, 2, num_gt=8)
    baseline_manifest.save(baseline / "scenario_manifest.json")
    candidate_manifest.save(candidate / "scenario_manifest.json")
    shared = {
        "assignment_strategy": "k_km",
        "assignment_rounds": 2,
        "routing_policy": "safe_ddqn",
        "reward_mode": "dinkelbach",
        "task_observation_mode": "full",
        "task_potential_enabled": True,
        "movement_hyperparameters": {"warmup_joint_transitions": 10000},
        "effective_routing_agent_configuration": {"warmup": 1000},
        "exploration_schedule_configuration": {
            "movement_exploration_decay_episodes": 1000,
            "routing_epsilon_decay_episodes": 1000,
        },
        "training_environment_width_m": 1000,
        "training_environment_height_m": 1000,
        "roi_count": 8,
        "seed": 20260817,
    }
    (baseline / "resolved_config.json").write_text(
        json.dumps({**shared, "episodes": 3}), encoding="utf-8"
    )
    (candidate / "resolved_config.json").write_text(
        json.dumps({**shared, "episodes": 2}), encoding="utf-8"
    )
    report = search._training_fairness_provenance(
        training={"run_directory": str(candidate)},
        baseline_run=baseline,
        expected_seed=20260817,
        expected_episodes=2,
    )
    assert report["status"] == "passed"
    assert report["training_scenario_prefix_matches_baseline"] is True
    assert report["episode_800_is_stop_not_decay_horizon"] is True


def test_search_trains_all_four_then_resumes_with_same_round_reference(
    tmp_path, monkeypatch
):
    dataset = SimpleNamespace(
        provenance={"fixture": "validation-only"},
        fixed_metadata={},
        arrays={"state": np.zeros((1, 1), dtype=np.float32)},
        episodes=(SimpleNamespace(index=0),),
    )
    monkeypatch.setattr(search, "bounded_validation_dataset", lambda *_a, **_k: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(
        search,
        "build_environment_source_bundle",
        lambda: {
            "bundle_sha256": "source-fixture",
            "git_sha": "fixture",
            "call_relationships": [],
            "excerpts": {},
            "rendered": {},
        },
    )
    monkeypatch.setattr(
        search,
        "render_common_prompt",
        lambda *, candidate_id, **_kwargs: f"COMMON candidate={candidate_id}",
    )

    def validate(submission, **_kwargs):
        candidate = _normalized(submission)
        return candidate, {"status": "passed"}, np.zeros((1, len(candidate["features"])))

    monkeypatch.setattr(search, "validate_search_candidate", validate)

    def save_artifact(root, **_kwargs):
        approved = Path(root) / "approved"
        approved.mkdir()
        (approved / "artifact.json").write_text("{}", encoding="utf-8")
        return approved

    monkeypatch.setattr(search, "save_approved_artifact", save_artifact)
    monkeypatch.setattr(
        search,
        "_training_fairness_provenance",
        lambda **kwargs: {
            "status": "passed",
            "candidate_run": kwargs["training"]["run_directory"],
            "training_scenario_prefix_matches_baseline": True,
            "episode_800_is_stop_not_decay_horizon": True,
        },
    )

    responses = [
        _submission("r01_c01", [0.1], [1.0]),
        _submission("r01_c02", [0.2], [1.0]),
        _submission("r01_c03", [0.3], [1.0]),
        _submission("r01_c04", [0.4, 0.5], [1.0, 2.0]),
        _submission("r02_c01", [0.4, 0.5, 0.6], [1.0, 2.0, 3.0]),
        _submission("r02_c02", [0.4], [1.0]),
        _submission("r02_c03", [0.4, 0.5], [1.0, 3.0]),
        _submission("r02_c04", [0.9], [-1.0]),
    ]
    prompts = []

    def call_model(*, prompt, **_kwargs):
        prompts.append(prompt)
        return {
            "content": json.dumps(responses[len(prompts) - 1]),
            "transport_completed": True,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(search, "_call_model", call_model)
    training_calls = []

    def train(**kwargs):
        training_calls.append(kwargs)
        run = Path(kwargs["output_directory"]) / search.SEARCH_METHOD_ID / "run"
        run.mkdir(parents=True, exist_ok=True)
        return {
            "status": "complete",
            "run_directory": str(run),
            "summaries": [{"episode_range": [1, 200]}],
        }

    evaluation_calls = []

    def evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        score = float(min(len(evaluation_calls), 4))
        return {
            "status": "complete",
            "checkpoint_episode": 800,
            "episode_count": 100,
            "scenario_manifest_hash": "manifest-fixture",
            "mean_episode_energy_efficiency_mbit_per_j": score,
            "std_episode_energy_efficiency_mbit_per_j": 0.1,
            "baseline_mean_episode_energy_efficiency_mbit_per_j": 0.5,
            "baseline_absolute_gap_mbit_per_j": score - 0.5,
            "baseline_percent_gap": 100.0 * (score - 0.5) / 0.5,
            "candidate_metrics": {},
            "evaluation_directory": str(kwargs["output_directory"]),
        }

    preflight = {
        "status": "passed",
        "manifest_path": str(tmp_path / "baseline-eval" / "scenario_manifest.json"),
        "manifest_content_hash": "manifest-fixture",
        "checkpoint_episode": 800,
        "metrics": {"episode_energy_efficiency_mbit_per_j": {"mean": 0.5}},
    }
    baseline_validator = lambda **_kwargs: preflight
    output = tmp_path / "search"
    first = search.run_training_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-eval",
        provider="openai",
        model="fixture-model",
        output_dir=output,
        runtime_root=tmp_path / "runtime",
        max_rounds=1,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline_validator,
    )
    assert first["status"] == "complete"
    assert len(training_calls) == len(evaluation_calls) == 4
    assert all(call["episodes"] == 800 for call in training_calls)
    assert all(call["seed"] == 20260817 for call in training_calls)
    assert len({call["output_directory"] for call in training_calls}) == 4
    assert all(call["summary_block_size"] == 200 for call in training_calls)
    assert all(call["checkpoint_episode"] == 800 for call in evaluation_calls)
    assert all(call["episodes"] == 100 for call in evaluation_calls)
    assert all(call["roi_count"] == 8 for call in evaluation_calls)
    assert all(call["manifest"] == preflight["manifest_path"] for call in evaluation_calls)

    second = search.run_training_search(
        resume=output,
        max_rounds=2,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline_validator,
    )
    assert second["status"] == "complete"
    assert len(training_calls) == len(evaluation_calls) == 8
    # The second-round candidates tie the earlier score; the earlier best remains.
    assert second["best_candidate"]["candidate_id"] == "r01_c04"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert len(state["rounds"]) == 2
    assert state["rounds"][1]["reference_candidate_id"] == "r01_c04"
    assert [
        state["rounds"][1]["candidates"][candidate_id]["base_candidate_id"]
        for candidate_id in state["rounds"][1]["candidate_order"]
    ] == ["r01_c04"] * 4
    assert [
        state["rounds"][1]["candidates"][candidate_id]["requested_direction"]
        for candidate_id in state["rounds"][1]["candidate_order"]
    ] == ["add", "remove", "reweight", "redesign"]
    second_round_prompts = prompts[4:]
    assert len(second_round_prompts) == 4
    assert all('"candidate_id": "r01_c04"' in prompt for prompt in second_round_prompts)
    assert all("1–200, 201–400, 401–600, 601–800" in prompt for prompt in second_round_prompts)


def test_runtime_roots_are_shallow_unique_and_stable(tmp_path):
    runtime = tmp_path / "rt"
    first = search._short_runtime_output_root(
        runtime, tmp_path / "search-a", kind="training", round_number=1, slot_index=1
    )
    same = search._short_runtime_output_root(
        runtime, tmp_path / "search-a", kind="training", round_number=1, slot_index=1
    )
    other = search._short_runtime_output_root(
        runtime, tmp_path / "search-a", kind="training", round_number=1, slot_index=2
    )
    evaluation = search._short_runtime_output_root(
        runtime, tmp_path / "search-a", kind="evaluation", round_number=1, slot_index=1
    )
    assert first == same
    assert len(first.name) == 16
    assert len(str(first)) < len(str(tmp_path / "search-a" / "deep" / "checkpoint"))
    assert len({first, other, evaluation}) == 3


def test_duplicate_detection_includes_completed_historical_candidates():
    candidate = _normalized(_submission("old", [0.4], [2.0]))
    fingerprint = search.candidate_computation_fingerprint(candidate)
    historical = {
        "candidate_id": "old",
        "status": "completed",
        "fingerprint": fingerprint,
    }
    state = {
        "rounds": [
            {
                "candidate_order": ["old"],
                "candidates": {"old": historical},
            }
        ]
    }
    current = {
        "candidates": {
            "new": {"candidate_id": "new", "status": "pending_generation"}
        }
    }
    report = search._duplicate_diagnostics(state, current, "new", fingerprint)
    assert report["stage"] == "duplicate"
    assert report["matching_candidate_id"] == "old"
    assert "does not claim general mathematical equivalence" in report["semantics"]


def test_failed_candidate_is_saved_and_increased_repair_budget_resumes(
    tmp_path, monkeypatch
):
    dataset = SimpleNamespace(
        provenance={"fixture": "validation-only"},
        fixed_metadata={},
        arrays={"state": np.zeros((1, 1), dtype=np.float32)},
        episodes=(SimpleNamespace(index=0),),
    )
    monkeypatch.setattr(search, "bounded_validation_dataset", lambda *_a, **_k: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(
        search,
        "build_environment_source_bundle",
        lambda: {
            "bundle_sha256": "source-fixture",
            "git_sha": "fixture",
            "call_relationships": [],
            "excerpts": {},
            "rendered": {},
        },
    )
    monkeypatch.setattr(
        search,
        "render_common_prompt",
        lambda *, candidate_id, **_kwargs: f"COMMON candidate={candidate_id}",
    )
    calls = []

    def invalid_response(**kwargs):
        calls.append(kwargs["prompt"])
        return {
            "content": "not-json",
            "transport_completed": True,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(search, "_call_model", invalid_response)
    preflight = {
        "status": "passed",
        "manifest_path": str(tmp_path / "baseline" / "scenario_manifest.json"),
        "manifest_content_hash": "manifest-fixture",
        "checkpoint_episode": 800,
        "metrics": {"episode_energy_efficiency_mbit_per_j": {"mean": 0.5}},
    }
    output = tmp_path / "failed-search"
    first = search.run_training_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline",
        provider="openai",
        model="fixture",
        output_dir=output,
        runtime_root=tmp_path / "runtime",
        max_rounds=1,
        max_repairs_per_candidate=0,
        client=object(),
        baseline_validator=lambda **_kwargs: preflight,
    )
    assert first["status"] == "paused_candidate_repairs_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = state["current_round"]["candidates"]["r01_c01"]
    assert record["corrections_used"] == 0
    assert len(record["versions"]) == 1
    assert Path(record["versions"][0]["directory"], "validation_report.json").is_file()

    second = search.run_training_search(
        resume=output,
        max_repairs_per_candidate=1,
        client=object(),
        baseline_validator=lambda **_kwargs: preflight,
    )
    assert second["status"] == "paused_candidate_repairs_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = state["current_round"]["candidates"]["r01_c01"]
    assert record["corrections_used"] == 1
    assert len(record["versions"]) == 2
    assert state["model_calls"] == 2
    assert len(calls) == 2
