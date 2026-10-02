import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from experiment_config import MethodSpec, effective_training_config
from HRL_task_aware import TrainingConfig, _evaluate_llm_observation, train
from llm_candidate import CandidateError, load_approved_design, save_approved_artifact
from llm_design_contract import LEGACY_OBS_INTERFACE_VERSION
from llm_runtime import (
    ApprovedDesignRuntime,
    artifact_identity,
    build_online_obs,
    copy_approved_artifact,
    runtime_constants_metadata,
    LLMRuntimeError,
)
from replay_auxiliary import empty_snapshot
from run_experiment import build_parser, run as run_experiment_training
from paper_evaluation import run_paper_evaluation
from scenario_manifest import generate_manifest
from centralized_movement import MOVEMENT_STATE_DIM, movement_state_feature_schema
from utils_update_v2 import ReplayBufferJoint


def _approved_fixture(
    root: Path,
    *,
    name="integration-fixture-only",
    observation_interface_version=None,
):
    constants = {
        name: {
            "value": value,
            "dtype": "float",
            "unit": "m" if "environment" in name else "s",
            "meaning": "runtime test constant",
            "source": "test fixture",
        }
        for name, value in {
            "environment_width_m": 1000.0,
            "environment_height_m": 1000.0,
            "episode_seconds": 1.0,
            "vs_deadline_seconds": 2.5,
            "com_deadline_seconds": 2.0,
        }.items()
    }
    legacy = observation_interface_version == LEGACY_OBS_INTERFACE_VERSION
    source_field = "obs.state" if legacy else "obs.uav_remaining_energy_fraction"
    formula = (
        "clip(abs(state[1]),0,1)"
        if legacy
        else "mean(uav_remaining_energy_fraction)"
    )
    code = (
        "def compute_extra_state(obs, constants):\n"
        '    x = np.clip(np.abs(obs["state"][1]), 0.0, 1.0)\n'
        "    return np.asarray([x], dtype=np.float32)\n\n"
        if legacy
        else (
            "def compute_extra_state(obs, constants):\n"
            '    x = np.mean(obs["uav_remaining_energy_fraction"])\n'
            "    return np.asarray([x], dtype=np.float32)\n\n"
        )
    )
    candidate = {
        "schema_version": "uav-hrl-llm-shared-feature-candidate-v2",
        "candidate_name": name,
        "reward_input_mode": "current_only",
        "features": [{
            "index": 0,
            "name": "bounded_state_square",
            "dtype": "float32",
            "description": "Test-only derived state feature.",
            "range": {"minimum": 0.0, "maximum": 1.0},
            "source_fields": [source_field],
            "formula": formula,
            "missing_data_rule": "state is required",
            "reward_weight": 2.5,
        }],
        "code": code,
    }
    run = root / f"design-run-{name}"
    run.mkdir(parents=True)
    kwargs = {}
    if observation_interface_version is not None:
        kwargs["observation_interface_version"] = observation_interface_version
    return save_approved_artifact(
        run,
        candidate=candidate,
        constants_metadata=constants,
        validation_report={"status": "passed", "fixture": True},
        evaluation_report={"status": "passed", "passed": True, "fixture": True},
        provenance={
            "beta": 1.0,
            "model_requested": "fixture/model",
            "model_actual": "fixture/model",
            "fixture_only": True,
        },
        **kwargs,
    )


def test_legacy_v1_artifact_keeps_original_observation_field_set(tmp_path):
    approved = _approved_fixture(
        tmp_path,
        name="legacy-v1-fixture",
        observation_interface_version=LEGACY_OBS_INTERFACE_VERSION,
    )
    design = load_approved_design(approved)
    state = np.asarray([0.5, -0.4, 0.0], dtype=np.float32)
    obs = build_online_obs(
        state,
        np.zeros(16, dtype=bool),
        empty_snapshot(),
        interface_version=LEGACY_OBS_INTERFACE_VERSION,
    )
    assert "uav_remaining_energy_fraction" not in obs
    runtime = ApprovedDesignRuntime(design, design.constants_metadata, timeout=10)
    try:
        extra, reward = runtime.evaluate(obs)
    finally:
        runtime.close()
    assert extra.tolist() == pytest.approx([0.4])
    assert reward == pytest.approx(1.0)


def test_method_registry_and_cli_contract_are_isolated():
    baseline = MethodSpec.parse("td3_dinkelbach")
    llm = MethodSpec.parse("td3_dinkelbach_llm")
    assert baseline.llm_enabled is False
    assert llm.llm_enabled is True
    assert llm.agent == baseline.agent
    assert llm.routing == baseline.routing
    assert llm.reward_mode == baseline.reward_mode
    assert llm.task_potential_enabled == baseline.task_potential_enabled
    baseline_args = build_parser().parse_args(["td3_dinkelbach", "--smoke"])
    assert not hasattr(baseline_args, "llm_artifact")
    llm_args = build_parser().parse_args(
        ["td3_dinkelbach_llm", "--llm-artifact", "approved", "--smoke"]
    )
    assert llm_args.llm_artifact == "approved"


def test_old_llm_artifact_contract_is_rejected_with_retraining_message(tmp_path):
    approved = _approved_fixture(tmp_path)
    artifact_path = approved / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schema_version"] = "uav-hrl-approved-design-v1"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(CandidateError, match="redesign and retrain"):
        load_approved_design(approved)


def test_llm_replay_reward_is_separate_and_added_once():
    replay = ReplayBufferJoint(2, 48, max_size=2, record_llm_reward=True)
    replay.add(
        [0.0, 0.0], np.zeros(48, dtype=np.float32), [1.0, 1.0], False,
        delivered_mbits=4.0, total_mobility_energy=2.0,
        c9_penalty=0.1, c10_penalty=0.2, com_range_penalty=0.3,
        llm_extra_reward=0.25,
        current_movement_mask=np.zeros(16, dtype=bool),
        next_movement_mask=np.zeros(16, dtype=bool),
    )
    base = replay._reward_numpy([0], current_lambda=0.5, gamma=1.0)
    combined = replay._reward_numpy(
        [0], current_lambda=0.5, gamma=1.0, llm_reward_beta=1.0
    )
    assert base[0, 0] == pytest.approx(2.4)
    assert combined[0, 0] == pytest.approx(2.65)
    assert replay.delivered_mbits[0, 0] == pytest.approx(4.0)
    assert replay.total_mobility_energy[0, 0] == pytest.approx(2.0)


def test_persistent_runtime_matches_offline_adapter_and_dynamic_constants(tmp_path):
    approved = _approved_fixture(tmp_path)
    design = load_approved_design(approved)
    metadata = runtime_constants_metadata(
        design,
        environment_width_m=1500,
        environment_height_m=1500,
        episode_seconds=3,
        task_deadlines_seconds={"FOV": 2.5, "COM": 2.0},
    )
    state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
    energy_indices = [
        item["index"]
        for item in movement_state_feature_schema()["features"]
        if item["name"].endswith(".energy")
    ]
    state[energy_indices] = 0.4
    obs = build_online_obs(state, np.zeros(16, dtype=bool), empty_snapshot())
    runtime = ApprovedDesignRuntime(design, metadata, timeout=10)
    try:
        extra, reward = runtime.evaluate(obs)
    finally:
        runtime.close()
    offline_extra, offline_reward = design.evaluate_obs_arrays(
        {name: value[None, ...] for name, value in obs.items()}, timeout=10
    )
    assert extra.tolist() == pytest.approx([0.4])
    assert reward == pytest.approx(1.0)
    assert extra == pytest.approx(offline_extra[0])
    assert reward == pytest.approx(offline_reward[0])
    assert metadata["environment_width_m"]["value"] == 1500.0


def test_llm_training_smoke_performs_td3_update_and_embeds_artifact(tmp_path):
    source = _approved_fixture(tmp_path)
    run_dir = tmp_path / "training-run"
    run_dir.mkdir()
    design = copy_approved_artifact(source, run_dir)
    # Prove the training run is self-contained after the original design moves.
    for child in sorted(source.iterdir(), reverse=True):
        child.unlink()
    source.rmdir()
    manifest = generate_manifest("train", 20260927, 1, num_gt=2)
    config = TrainingConfig(
        total_episodes=1,
        mode="smoke",
        episode_seconds=2,
        warmup_joint_transitions=0,
        routing_warmup_transitions=1,
        batch_size=1,
        replay_max_size=8,
        enable_model_checkpoints=True,
        enable_full_resume=True,
        model_checkpoint_every=1,
        full_resume_every=1,
        checkpoint_root=str(run_dir / "checkpoints"),
        run_directory=str(run_dir),
        enable_plots=False,
        enable_csv=False,
        random_seed=20260927,
    )
    result = train(
        config,
        scenario_manifest=manifest,
        method_spec=MethodSpec.parse("td3_dinkelbach_llm"),
        llm_artifact_dir=design.directory,
    )
    assert result["movement_state_dim"] == result["original_movement_state_dim"] + 1
    assert result["critic_updates"] >= 1
    assert result["joint_replay_diagnostics"]["llm_extra_reward_enabled"] is True
    assert result["llm_artifact_identity"] == artifact_identity(design)
    model_checkpoint = run_dir / "checkpoints" / "models" / "ep_0001"
    full_checkpoint = run_dir / "checkpoints" / "full" / "ep_0001"
    assert load_approved_design(model_checkpoint / "llm_artifact")
    assert load_approved_design(full_checkpoint / "llm_artifact")
    with np.load(full_checkpoint / "joint_replay.npz", allow_pickle=False) as replay:
        assert "llm_extra_reward" in replay.files
        assert replay["state"].shape[1] == result["movement_state_dim"]
        assert replay["next_state"][0] == pytest.approx(replay["state"][1])

    resumed = train(
        replace(
            config,
            resume_dir=str(full_checkpoint),
            enable_model_checkpoints=False,
            enable_full_resume=False,
        ),
        scenario_manifest=manifest,
        method_spec=MethodSpec.parse("td3_dinkelbach_llm"),
        llm_artifact_dir=design.directory,
    )
    assert resumed["episodes_run"] == 0
    assert resumed["joint_replay_size"] == result["joint_replay_size"]
    assert resumed["llm_artifact_identity"] == result["llm_artifact_identity"]

    evaluation_manifest = generate_manifest(
        "test", 20260928, 1, num_gt=3, environment_size_m=1500
    )
    evaluation = train(
        TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=1,
            warmup_joint_transitions=0,
            routing_warmup_transitions=1,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260927,
        ),
        scenario_manifest=evaluation_manifest,
        method_spec=MethodSpec.parse("td3_dinkelbach_llm"),
        evaluation=True,
        checkpoint_dir=model_checkpoint,
        expected_checkpoint_episodes=1,
        expected_checkpoint_formal_config=effective_training_config(
            config, MethodSpec.parse("td3_dinkelbach_llm")
        ),
        expected_checkpoint_training_manifest=manifest,
        evaluation_overrides={
            "environment_width_m": 1500,
            "environment_height_m": 1500,
        },
        llm_artifact_dir=design.directory,
    )
    assert evaluation["movement_state_dim"] == result["movement_state_dim"]
    assert evaluation["episode_metrics"][0]["num_GT"] == 3
    assert evaluation["run_metadata"]["evaluation_environment_width_m"] == 1500

    different = _approved_fixture(tmp_path, name="same-dimension-different-design")
    with pytest.raises(RuntimeError, match="checkpoint LLM artifact identity"):
        train(
            TrainingConfig(
                total_episodes=1,
                mode="custom",
                episode_seconds=1,
                enable_model_checkpoints=False,
                enable_full_resume=False,
                enable_plots=False,
                enable_csv=False,
                random_seed=20260927,
            ),
            scenario_manifest=evaluation_manifest,
            method_spec=MethodSpec.parse("td3_dinkelbach_llm"),
            evaluation=True,
            checkpoint_dir=model_checkpoint,
            expected_checkpoint_episodes=1,
            expected_checkpoint_formal_config=effective_training_config(
                config, MethodSpec.parse("td3_dinkelbach_llm")
            ),
            expected_checkpoint_training_manifest=manifest,
            evaluation_overrides={
                "environment_width_m": 1500,
                "environment_height_m": 1500,
            },
            llm_artifact_dir=different,
        )


def test_modified_run_artifact_is_rejected(tmp_path):
    source = _approved_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    design = copy_approved_artifact(source, run_dir)
    candidate = design.directory / "candidate.py"
    candidate.write_text(candidate.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CandidateError, match="changed"):
        load_approved_design(design.directory)


def test_runtime_failure_saves_input_and_stops_worker(tmp_path):
    class BrokenRuntime:
        identity = {"artifact_content_sha256": "fixture"}

        def __init__(self):
            self.closed = False

        def evaluate(self, obs):
            raise ValueError("fixture online failure")

        def close(self):
            self.closed = True

    runtime = BrokenRuntime()
    config = SimpleNamespace(run_directory=str(tmp_path))
    obs = {"state": np.asarray([0.0], dtype=np.float32)}
    with pytest.raises(LLMRuntimeError, match="diagnostic saved"):
        _evaluate_llm_observation(runtime, obs, config, 2, 3, "current")
    assert runtime.closed is True
    root = tmp_path / "llm_runtime_failures"
    assert (root / "episode_0002_step_0003_current.json").is_file()
    with np.load(root / "episode_0002_step_0003_current.npz") as saved:
        assert saved["state"].tolist() == [0.0]


def test_registered_runner_and_paper_evaluation_entry_use_run_artifact(tmp_path):
    approved = _approved_fixture(tmp_path)
    output_root = tmp_path / "results"
    args = build_parser().parse_args(
        [
            "td3_dinkelbach_llm",
            "--llm-artifact", str(approved),
            "--episodes", "1",
            "--episode-seconds", "1",
            "--checkpoint-interval", "1",
            "--seed", "20260927",
            "--roi-count", "2",
            "--output-root", str(output_root),
        ]
    )
    assert run_experiment_training(args) == 0
    run_dir = next((output_root / "td3_dinkelbach_llm").iterdir())
    evaluation = run_paper_evaluation(
        "td3_dinkelbach_llm",
        run_directory=run_dir,
        suite="fixed_roi",
        checkpoint_episode=1,
        roi_counts=(2,),
        episodes=1,
        episode_seconds=1,
        manifest_seed=20260928,
        output_directory=tmp_path / "paper-evaluation",
        flatten_single_point=True,
    )
    identity = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )["llm_artifact_identity"]
    assert evaluation["points"][0]["llm_artifact_id"] == identity[
        "artifact_content_sha256"
    ]
    aggregates = json.loads(
        (tmp_path / "paper-evaluation" / "aggregated_plot_data.json").read_text(
            encoding="utf-8"
        )
    )
    assert {row["llm_artifact_id"] for row in aggregates} == {
        identity["artifact_content_sha256"]
    }
