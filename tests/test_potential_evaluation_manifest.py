import json
from pathlib import Path

import pytest

import llm_episode_training as episode_training
import paper_evaluation
from experiment_config import DEFAULT_TRAINING_SEED
from scenario_manifest import generate_manifest


class _RolloutStarted(Exception):
    pass


def _saved_manifest(tmp_path, *, seed=20260927, episodes=2, roi_count=8):
    manifest = generate_manifest(
        "test",
        manifest_seed=seed,
        episode_count=episodes,
        num_gt=roi_count,
    )
    path = tmp_path / f"manifest-{seed}-{roi_count}.json"
    manifest.save(path)
    return path, manifest


def _baseline_preflight(path, manifest):
    return {
        "manifest_path": str(path.resolve()),
        "manifest_file_sha256": episode_training._file_sha256(path),
        "manifest_content_hash": manifest.content_hash,
        "manifest_schema_version": manifest.schema_version,
        "manifest_seed": int(manifest.manifest_seed),
        "scenario_ids": [str(item["scenario_id"]) for item in manifest.episodes],
        "evaluation_contract": {"episode_seconds": 60},
    }


def test_fixed_roi_explicit_manifest_uses_saved_scenarios_in_order(monkeypatch, tmp_path):
    path, expected = _saved_manifest(tmp_path)
    captured = {}

    def stop_before_rollout(_config, *, scenario_manifest, **_kwargs):
        captured["manifest"] = scenario_manifest
        raise _RolloutStarted

    monkeypatch.setattr(paper_evaluation, "train", stop_before_rollout)
    with pytest.raises(_RolloutStarted):
        paper_evaluation.run_paper_evaluation(
            "kkm_random_action_random_routing",
            suite="fixed_roi",
            manifest_path=path,
            roi_counts=(8,),
            episodes=2,
            episode_seconds=1,
            output_directory=tmp_path / "evaluation",
            flatten_single_point=True,
        )

    actual = captured["manifest"]
    assert actual.manifest_seed == 20260927
    assert actual.content_hash == expected.content_hash
    assert [item["scenario_id"] for item in actual.episodes] == [
        item["scenario_id"] for item in expected.episodes
    ]


def test_fixed_roi_manifest_is_rejected_before_rollout_when_incompatible(
    monkeypatch, tmp_path
):
    path, _manifest = _saved_manifest(tmp_path, roi_count=7)
    rollout_called = False

    def unexpected_rollout(*_args, **_kwargs):
        nonlocal rollout_called
        rollout_called = True

    monkeypatch.setattr(paper_evaluation, "train", unexpected_rollout)
    with pytest.raises(ValueError, match="fixed RoI count is incompatible"):
        paper_evaluation.run_paper_evaluation(
            "kkm_random_action_random_routing",
            suite="fixed_roi",
            manifest_path=path,
            roi_counts=(8,),
            episodes=2,
            output_directory=tmp_path / "incompatible",
            flatten_single_point=True,
        )
    assert rollout_called is False

    with pytest.raises(ValueError, match="conflicts with --manifest-seed"):
        paper_evaluation.run_paper_evaluation(
            "kkm_random_action_random_routing",
            suite="fixed_roi",
            manifest_path=path,
            manifest_seed=DEFAULT_TRAINING_SEED,
            roi_counts=(7,),
            episodes=2,
            output_directory=tmp_path / "conflicting-seed",
            flatten_single_point=True,
        )
    assert rollout_called is False


def test_fixed_roi_without_manifest_keeps_generated_scenario_flow(monkeypatch, tmp_path):
    generated = generate_manifest("test", DEFAULT_TRAINING_SEED, 1, num_gt=8)
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return generated

    monkeypatch.setattr(paper_evaluation, "generate_manifest", generate)
    monkeypatch.setattr(
        paper_evaluation,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_RolloutStarted()),
    )
    with pytest.raises(_RolloutStarted):
        paper_evaluation.run_paper_evaluation(
            "kkm_random_action_random_routing",
            suite="fixed_roi",
            roi_counts=(8,),
            episodes=1,
            episode_seconds=1,
            output_directory=tmp_path / "generated",
            flatten_single_point=True,
        )
    assert calls == [(('test', DEFAULT_TRAINING_SEED, 1), {"num_gt": 8})]


def test_candidate_evaluation_passes_exact_manifest_to_evaluator(monkeypatch, tmp_path):
    path, manifest = _saved_manifest(tmp_path)
    preflight = _baseline_preflight(path, manifest)
    captured = {}

    def stop(command, *, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        raise episode_training.EpisodeTrainingError("stop before evaluator")

    monkeypatch.setattr(episode_training, "_run", stop)
    with pytest.raises(episode_training.EpisodeTrainingError, match="stop before evaluator"):
        episode_training.run_candidate_evaluation(
            method_id="td3_dinkelbach_llm_potential",
            run_directory=tmp_path / "trained-run",
            checkpoint_episode=400,
            episodes=2,
            roi_count=8,
            environment_size_m=(1000.0, 1000.0),
            manifest=path,
            baseline_run=tmp_path / "baseline-run",
            baseline_evaluation=tmp_path / "baseline-evaluation",
            baseline_preflight=preflight,
            output_directory=tmp_path / "candidate-evaluation",
        )

    command = captured["command"]
    manifest_index = command.index("--manifest")
    assert Path(command[manifest_index + 1]) == path.resolve()
    assert "--manifest-seed" not in command


def test_post_rollout_scenario_mismatch_reports_identity_and_first_difference(
    monkeypatch, tmp_path
):
    path, manifest = _saved_manifest(tmp_path)
    preflight = _baseline_preflight(path, manifest)
    rows_path = tmp_path / "candidate-rows.jsonl"
    wrong_ids = [value.replace("20260927", "20260817") for value in preflight["scenario_ids"]]
    rows_path.write_text(
        "".join(
            json.dumps(
                {
                    "scenario_id": scenario_id,
                    "energy_efficiency_mbit_per_j": 1.0,
                }
            )
            + "\n"
            for scenario_id in wrong_ids
        ),
        encoding="utf-8",
    )
    actual_manifest = tmp_path / "actual-scenario-manifest.json"
    result = {
        "points": [
            {
                "checkpoint_episode": 400,
                "evaluation_episode_count": 2,
                "fixed_num_gt": 8,
                "evaluation_episode_horizon_s": 60,
                "evaluation_environment_width_m": 1000.0,
                "evaluation_environment_height_m": 1000.0,
                "manifest_seed": 20260817,
                "scenario_manifest_path": str(actual_manifest),
                "scenario_manifest_hash": "wrong-hash",
                "scenario_ids": wrong_ids,
                "outputs": {"per_episode_jsonl": str(rows_path)},
            }
        ]
    }
    monkeypatch.setattr(episode_training, "_run", lambda *_args, **_kwargs: result)

    with pytest.raises(episode_training.EpisodeTrainingError) as caught:
        episode_training.run_candidate_evaluation(
            method_id="td3_dinkelbach_llm_potential",
            run_directory=tmp_path / "trained-run",
            checkpoint_episode=400,
            episodes=2,
            roi_count=8,
            environment_size_m=(1000.0, 1000.0),
            manifest=path,
            baseline_run=tmp_path / "baseline-run",
            baseline_evaluation=tmp_path / "baseline-evaluation",
            baseline_preflight=preflight,
            output_directory=tmp_path / "candidate-evaluation",
        )

    message = str(caught.value)
    assert "expected_count=2, actual_count=2" in message
    assert "expected_seed=20260927, actual_seed=20260817" in message
    assert "first_mismatch_index=0" in message
    assert preflight["scenario_ids"][0] in message
    assert wrong_ids[0] in message
    assert str(path.resolve()) in message
    assert str(actual_manifest) in message
