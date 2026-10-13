import copy
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


def test_training_feedback_compacts_ordered_per_episode_exploration(tmp_path):
    candidates = {}
    expected_first = None
    expected_block_end = None
    for candidate_index in range(1, 5):
        candidate_id = f"r01_c{candidate_index:02d}"
        rows = []
        for episode in range(1, 801):
            noise = 0.2 - 0.00015 * episode
            exploration = {
                "behavior_noise_max": noise,
                "evaluation": False,
                "warmup_random_action_intervals": 0,
            }
            rows.append(
                {
                    "episode": episode,
                    "candidate_artifact_id": candidate_id,
                    "feature_names": ["delivery"],
                    "feature_reward_weights": [0.5],
                    "beta": 1.0,
                    "feature_contribution_sums": [episode / 1000.0],
                    "base_reward_sum": float(episode),
                    "extra_reward_sum": episode / 1000.0,
                    "combined_reward_sum": float(episode) + episode / 1000.0,
                    "timely_mbits": 10.0 + episode,
                    "movement_energy_j": 1000.0 + episode,
                    "energy_efficiency_mbit_per_j": episode / 1_000_000.0,
                    "roi_count": 8,
                    "exploration": exploration,
                    "dinkelbach_lambda_used": episode / 10000.0,
                }
            )
        run = tmp_path / candidate_id
        run.mkdir()
        (run / "llm_training_episode_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        summaries = summarize_training_blocks(
            rows, block_size=200, expected_episodes=800
        )
        submission = _submission(candidate_id, [0.4], [0.5])
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "status": "completed",
            "submission": submission,
            "training": {
                "status": "complete",
                "run_directory": str(run),
                "summaries": summaries,
            },
            "training_fairness": {
                "status": "passed",
                "seed": 20260817,
                "episodes": 800,
                "fresh_initialization": True,
                "training_scenario_prefix_matches_baseline": True,
                "episode_800_is_stop_not_decay_horizon": True,
                "exploration_schedule_configuration": {
                    "movement_exploration_decay_episodes": 1000,
                    "routing_epsilon_decay_episodes": 1000,
                },
            },
            "evaluation": {
                "status": "complete",
                "checkpoint_episode": 800,
                "episode_count": 100,
                "mean_episode_energy_efficiency_mbit_per_j": 0.001,
                "std_episode_energy_efficiency_mbit_per_j": 0.0001,
                "baseline_mean_episode_energy_efficiency_mbit_per_j": 0.002,
                "baseline_absolute_gap_mbit_per_j": -0.001,
                "baseline_percent_gap": -50.0,
                "candidate_metrics": {},
            },
        }
        if candidate_index == 1:
            expected_first = rows[0]["exploration"]["behavior_noise_max"]
            expected_block_end = rows[199]["exploration"]["behavior_noise_max"]

    round_record = {
        "candidate_order": list(candidates),
        "candidates": candidates,
    }
    _results, training = search._latest_round_feedback(
        round_record, best_candidate_id="r01_c01"
    )
    serialized = json.dumps(training, ensure_ascii=False)
    assert serialized.count('"behavior_noise_max"') == 4
    assert len(serialized) < 30_000
    assert training["shared_fairness_and_training_contract"]["training_seed"] == 20260817
    first_block = training["shared_exploration_by_block"][0]
    parameter = first_block["summary"]["parameters"]["behavior_noise_max"]
    assert first_block["source"] == "ordered llm_training_episode_metrics.jsonl"
    assert parameter["start"] == pytest.approx(expected_first)
    assert parameter["end"] == pytest.approx(expected_block_end)
    assert parameter["monotonic"] == "nonincreasing"
    compact_stat = training["candidates"][0]["blocks"][0][
        "feature_contributions"
    ]["delivery"]
    original_stat = candidates["r01_c01"]["training"]["summaries"][0][
        "feature_contributions"
    ]["delivery"]
    assert compact_stat == original_stat


def test_exploration_summary_does_not_infer_trend_without_episode_order():
    unordered = search._exploration_parameter_summary(
        [{"noise": 0.1}, {"noise": 0.2}], ordered=False
    )
    parameter = unordered["parameters"]["noise"]
    assert parameter["trend"] == "unknown_without_episode_order"
    assert "monotonic" not in parameter
    assert parameter["min"] == pytest.approx(0.1)
    assert parameter["max"] == pytest.approx(0.2)
    assert parameter["start"] == {"unavailable": "episode order not saved"}
    assert parameter["end"] == {"unavailable": "episode order not saved"}

    fixed = search._exploration_parameter_summary(
        [{"noise": 0.1}, {"noise": 0.1}], ordered=False
    )
    assert fixed["parameters"]["noise"] == {"value": 0.1}

    ordered = search._exploration_parameter_summary(
        [{"noise": 0.2}, {"noise": 0.1}], ordered=True
    )["parameters"]["noise"]
    assert ordered["start"] == pytest.approx(0.2)
    assert ordered["end"] == pytest.approx(0.1)
    assert ordered["monotonic"] == "nonincreasing"
    assert "trend" not in ordered


def test_cached_prompt_backups_use_short_collision_safe_atomic_paths(
    tmp_path, monkeypatch
):
    actual_output = Path(
        r"C:\Users\user\Desktop\project\YM_Post_Disaster_Training\UAV_HRL"
        r"\results\llm_training_searches\gpt-4o"
        r"\design-20261010T052422Z-03609dfd"
    )
    actual_archive = (
        actual_output
        / "migrations"
        / search.PROMPT_MIGRATION_ID
        / "p"
        / "r02_c01"
    )
    actual_path = actual_archive / f"rq-{'a' * 16}.txt"
    path_report = search._validate_atomic_write_paths(
        actual_path, portable_path_limit=search.WINDOWS_PORTABLE_PATH_LIMIT
    )
    assert path_report["target_path_length"] <= search.WINDOWS_PORTABLE_PATH_LIMIT
    assert (
        path_report["temporary_path_length"]
        <= search.WINDOWS_PORTABLE_PATH_LIMIT
    )
    assert path_report["temporary_path"].endswith(
        f".rq-{'a' * 16}.txt.tmp"
    )

    archive = tmp_path / "migration" / "p" / "r02_c01"
    monkeypatch.setattr(search, "_sha256_text", lambda _text: "a" * 64)
    first, first_hash = search._archive_cached_prompt(
        archive, kind="request", text="first prompt"
    )
    same, same_hash = search._archive_cached_prompt(
        archive, kind="request", text="first prompt"
    )
    second, second_hash = search._archive_cached_prompt(
        archive, kind="request", text="different prompt"
    )
    second_again, _ = search._archive_cached_prompt(
        archive, kind="request", text="different prompt"
    )
    assert first == same
    assert second == second_again
    assert first != second
    assert first.read_text(encoding="utf-8") == "first prompt"
    assert second.read_text(encoding="utf-8") == "different prompt"
    assert first_hash == same_hash == second_hash == "a" * 64
    stage, stage_hash = search._archive_cached_prompt(
        archive, kind="stage", text="generation stage"
    )
    assert stage.name.startswith("st-")
    assert stage.read_text(encoding="utf-8") == "generation stage"
    assert stage_hash == "a" * 64


def test_resume_source_compatibility_uses_excerpt_content_not_git_revision(tmp_path):
    output = tmp_path / "search"
    source_dir = output / "prompt_source_context"
    source_dir.mkdir(parents=True)
    settings = {
        "train_episodes": 800,
        "evaluation_episodes": 100,
        "evaluation_roi_count": 8,
        "evaluation_area_m": [1000.0, 1000.0],
        "episode_seconds": 60,
        "candidate_count_per_round": 4,
        "candidate_generation_calls_per_request": 1,
    }
    old_bundle = {
        "git_sha": "old-git",
        "bundle_sha256": "old-bundle",
        "call_relationships": ["a -> b"],
        "excerpts": {
            "observation": [
                {
                    "source_file": "env.py",
                    "symbol": "observe",
                    "line_start": 1,
                    "line_end": 2,
                    "sha256": "legacy",
                    "code": "def observe():\n    return 1",
                }
            ]
        },
        "rendered": {},
    }
    current_bundle = {
        **old_bundle,
        "git_sha": "new-git",
        "bundle_sha256": "new-bundle",
        "excerpts": {
            "observation": [
                {
                    **old_bundle["excerpts"]["observation"][0],
                    "line_start": 50,
                    "line_end": 51,
                }
            ]
        },
    }
    current_bundle["content_sha256"] = search._source_bundle_content_sha256(
        current_bundle
    )
    state = {
        "prompt_version": search.LEGACY_PROMPT_VERSION,
        "git_sha": "old-git",
        "environment_source_bundle_sha256": "old-bundle",
        "settings": settings,
    }
    (output / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (output / "constants.json").write_text("{}", encoding="utf-8")
    (source_dir / "environment_source_context.json").write_text(
        json.dumps(old_bundle), encoding="utf-8"
    )
    migration = search._prepare_resume_compatibility(
        output=output,
        state=state,
        source_bundle=current_bundle,
        constants={},
    )
    assert migration is not None
    assert state["prompt_version"] == search.SEARCH_PROMPT_VERSION
    assert state["environment_source_bundle_sha256"] == "new-bundle"
    assert (output / "migrations" / search.PROMPT_MIGRATION_ID / "state.before.json").is_file()
    assert search._prepare_resume_compatibility(
        output=output,
        state=state,
        source_bundle=current_bundle,
        constants={},
    ) is None

    changed_bundle = copy.deepcopy(current_bundle)
    changed_bundle["bundle_sha256"] = "changed-bundle"
    changed_bundle["excerpts"]["observation"][0]["code"] = (
        "def observe():\n    return 2"
    )
    changed_bundle["content_sha256"] = search._source_bundle_content_sha256(
        changed_bundle
    )
    (source_dir / "environment_source_context.json").write_text(
        json.dumps(current_bundle), encoding="utf-8"
    )
    with pytest.raises(search.TrainingSearchError, match="source content changed"):
        search._prepare_resume_compatibility(
            output=output,
            state=state,
            source_bundle=changed_bundle,
            constants={},
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
    bundle_holder = {
        "value": {
            "bundle_sha256": "source-fixture",
            "git_sha": "fixture",
            "call_relationships": [],
            "excerpts": {},
            "rendered": {},
        }
    }
    monkeypatch.setattr(
        search, "build_environment_source_bundle", lambda: bundle_holder["value"]
    )
    monkeypatch.setattr(
        search,
        "render_common_prompt",
        lambda *, candidate_id, **_kwargs: (
            f"COMMON candidate={candidate_id}\n" + "E" * 70_000
        ),
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
        blocks = [
            {
                "episode_range": [start, start + 199],
                "episode_count": 200,
                "energy_efficiency_mbit_per_j": {"mean": 0.001, "std": 0.0001},
                "timely_useful_delivery_mbit": {"mean": 20.0, "std": 2.0},
                "movement_energy_j": {"mean": 100000.0, "std": 1000.0},
                "base_reward_sum": {"mean": 1.0, "std": 0.2},
                "extra_reward_sum": {"mean": 0.5, "std": 0.1},
                "combined_reward_sum": {"mean": 1.5, "std": 0.25},
                "feature_contributions": {
                    "feature_0": {"mean": 0.5, "std": 0.1}
                },
                "roi_count_distribution": {"8": 200},
                "exploration": [],
                "lambda_semantics": "actual dinkelbach_lambda_used during training",
            }
            for start in (1, 201, 401, 601)
        ]
        return {
            "status": "complete",
            "run_directory": str(run),
            "summaries": blocks,
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
        context_length=128000,
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

    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["prompt_version"] = search.LEGACY_PROMPT_VERSION
    state["status"] = "paused_context_budget_exceeded"
    state["search_round"] = 2
    state["current_round"] = search._new_round(state, 2)
    pending = state["current_round"]["candidates"]["r02_c01"]
    pending["generation_stage_request"] = "OLD OVERSIZED STAGE"
    old_common = "COMMON candidate=r02_c01\n" + "E" * 70_000
    pending["generation_request"] = (
        old_common.rstrip() + "\n\n" + pending["generation_stage_request"] + "\n"
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    bundle_holder["value"] = {
        "bundle_sha256": "source-fixture-new-git",
        "git_sha": "fixture-new-git",
        "call_relationships": [],
        "excerpts": {},
        "rendered": {},
    }
    preflight.update(
        manifest_schema_version="uav-hrl-scenario-v8",
        manifest_seed=20260927,
    )
    original_estimate = search.estimate_token_budget

    def over_budget(prompt, *, context_length, max_output_tokens):
        return {
            "method": "fixture estimate",
            "exact": False,
            "characters": len(prompt),
            "reserved_output_tokens": max_output_tokens,
            "client_context_budget": context_length,
            "estimated_total_upper": context_length + 1,
            "fits_client_budget": False,
        }

    monkeypatch.setattr(search, "estimate_token_budget", over_budget)
    paused = search.run_training_search(
        resume=output,
        max_rounds=2,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline_validator,
    )
    assert paused["status"] == "paused_context_budget_exceeded"
    assert paused["model_calls"] == 4
    assert len(training_calls) == len(evaluation_calls) == 4
    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    migrated_pending = migrated["current_round"]["candidates"]["r02_c01"]
    assert migrated["prompt_version"] == search.SEARCH_PROMPT_VERSION
    assert migrated["current_round"]["reference_candidate_id"] == "r01_c04"
    assert len(migrated["compatibility_migrations"]) == 1
    assert migrated["baseline_preflight"] == preflight
    assert len(migrated["baseline_preflight_compatibility_updates"]) == 1
    assert len(migrated_pending["generation_request_migrations"]) == 1
    prompt_migration = migrated_pending["generation_request_migrations"][0]
    assert len(prompt_migration["old_generation_request_sha256"]) == 64
    assert len(prompt_migration["old_generation_stage_sha256"]) == 64
    for archive_key in ("old_request_archive", "old_stage_archive"):
        archive_path = Path(prompt_migration[archive_key])
        assert archive_path.is_file()
        atomic_paths = search._validate_atomic_write_paths(
            archive_path,
            portable_path_limit=search.WINDOWS_PORTABLE_PATH_LIMIT,
        )
        assert (
            atomic_paths["temporary_path_length"]
            <= search.WINDOWS_PORTABLE_PATH_LIMIT
        )
    assert "OLD OVERSIZED STAGE" not in migrated_pending["generation_request"]
    assert "def compute_extra_state" in migrated_pending["generation_request"]
    diagnostic_path = Path(
        migrated_pending["last_prompt_budget_report"]["prompt_path"]
    )
    assert diagnostic_path.is_file()
    assert diagnostic_path.read_text(encoding="utf-8") == migrated_pending[
        "generation_request"
    ]
    backup = output / "migrations" / search.PROMPT_MIGRATION_ID / "state.before.json"
    assert backup.is_file()
    backup_text = backup.read_text(encoding="utf-8")

    paused_again = search.run_training_search(
        resume=output,
        max_rounds=2,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline_validator,
    )
    assert paused_again["model_calls"] == 4
    migrated_again = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(migrated_again["compatibility_migrations"]) == 1
    assert len(migrated_again["baseline_preflight_compatibility_updates"]) == 1
    assert len(
        migrated_again["current_round"]["candidates"]["r02_c01"][
            "generation_request_migrations"
        ]
    ) == 1
    assert backup.read_text(encoding="utf-8") == backup_text

    monkeypatch.setattr(search, "estimate_token_budget", original_estimate)

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
    assert "========== DIRECTION_ADD ==========" in second_round_prompts[0]
    assert '"feature_contributions"' in second_round_prompts[0]
    assert '"baseline_percent_gap"' in second_round_prompts[0]
    assert all(candidate_id in second_round_prompts[0] for candidate_id in state["rounds"][0]["candidate_order"])
    completed = json.loads(state_path.read_text(encoding="utf-8"))
    report = completed["rounds"][1]["candidates"]["r02_c01"][
        "last_prompt_budget_report"
    ]
    assert report["full_request_budget"]["fits_client_budget"] is True
    assert report["sections_characters"]["training_block_summaries"] > 0


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
    raw_response = '{"candidate_id": "broken",\nnot-json'

    def invalid_response(**kwargs):
        calls.append(kwargs["prompt"])
        return {
            "content": raw_response,
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
        max_repairs_per_candidate=0,
        client=object(),
        baseline_validator=lambda **_kwargs: preflight,
    )
    assert second["status"] == "paused_candidate_repairs_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = state["current_round"]["candidates"]["r01_c01"]
    assert record["corrections_used"] == 0
    assert len(record["versions"]) == 1
    assert state["model_calls"] == 1
    assert len(calls) == 1

    third = search.run_training_search(
        resume=output,
        max_repairs_per_candidate=1,
        client=object(),
        baseline_validator=lambda **_kwargs: preflight,
    )
    assert third["status"] == "paused_candidate_repairs_exhausted"
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = state["current_round"]["candidates"]["r01_c01"]
    assert record["corrections_used"] == 1
    assert len(record["versions"]) == 2
    assert state["model_calls"] == 2
    assert len(calls) == 2
    assert '"response_kind": "raw_unparsed_model_response"' in calls[1]
    assert '"raw_response": "{\\"candidate_id\\": \\"broken\\",\\nnot-json"' in calls[1]
    assert '"lineno": 2' in calls[1]
    assert '"colno": 1' in calls[1]
    assert '"pos":' in calls[1]
    assert "Complete failed response material:" in calls[1]
    assert "Complete failed candidate:\nnull" not in calls[1]


def test_parseable_validation_failure_repair_prompt_contains_complete_candidate(
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
    submission = _submission("r01_c01", [0.4], [2.0])
    calls = []

    def response(**kwargs):
        calls.append(kwargs["prompt"])
        return {
            "content": json.dumps(submission),
            "transport_completed": True,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(search, "_call_model", response)
    monkeypatch.setattr(
        search,
        "validate_search_candidate",
        lambda candidate, **_kwargs: (
            _normalized(candidate),
            {
                "status": "failed",
                "stage": "execution",
                "error": "fixture runtime validation failure",
            },
            None,
        ),
    )
    preflight = {
        "status": "passed",
        "manifest_path": str(tmp_path / "baseline" / "scenario_manifest.json"),
        "manifest_content_hash": "manifest-fixture",
        "checkpoint_episode": 800,
        "metrics": {"episode_energy_efficiency_mbit_per_j": {"mean": 0.5}},
    }
    result = search.run_training_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline",
        provider="openai",
        model="fixture",
        output_dir=tmp_path / "parsed-failure-search",
        runtime_root=tmp_path / "runtime",
        max_rounds=1,
        max_repairs_per_candidate=1,
        client=object(),
        baseline_validator=lambda **_kwargs: preflight,
    )
    assert result["status"] == "paused_candidate_repairs_exhausted"
    assert len(calls) == 2
    assert '"response_kind": "parsed_candidate"' in calls[1]
    assert '"parsed_candidate": {' in calls[1]
    assert '"candidate_id": "r01_c01"' in calls[1]
    assert submission["code"].splitlines()[0] in calls[1]
    assert '"raw_response"' not in calls[1]
