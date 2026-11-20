import json

import pytest

from HRL_task_aware import TrainingConfig, train
from experiment_config import MethodSpec
from run_experiment import _reward_step_jsonl_sink, build_parser
from scenario_manifest import generate_manifest


def _config(seed):
    return TrainingConfig(
        total_episodes=1,
        mode="smoke",
        episode_seconds=1,
        warmup_joint_transitions=0,
        batch_size=1,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=seed,
    )


def _assert_reconstructable(row):
    assert row["schema_version"] == "uav-hrl-movement-reward-step-v1"
    assert row["episode"] == row["episode_index"]
    assert row["movement_step"] == row["movement_interval_index"]
    assert row["lambda_EE"] == row["dinkelbach_lambda_used"]
    assert row["E_move_J"] == row["movement_energy_j"]
    assert row["delivered_complete_useful_bits"] == pytest.approx(
        row["B_Mbit"] * 1e6
    )
    assert row["ee_objective_term"] == pytest.approx(
        row["B_Mbit"] - row["lambda_EE_times_E_move"]
    )
    assert row["movement_reward_after_constraints"] == pytest.approx(
        row["reward_before_constraint_penalties"]
        - row["applied_constraint_penalty"]
    )
    extra = (
        row["llm_extra_reward"]
        + row["potential_shaping"]
        + row["rnd_weighted_bonus"]
    )
    assert row["final_training_reward"] == pytest.approx(
        row["movement_reward_after_constraints"] + extra
    )
    assert row["final_replay_reward"] == row["final_training_reward"]
    assert sum(item["deduction"] for item in row["penalties"].values()) == pytest.approx(
        row["applied_constraint_penalty"]
    )


def test_train_and_evaluation_reward_step_recording_is_non_intervening():
    method = MethodSpec.parse("kkm_random_action_random_routing")
    training_rows = []
    training = train(
        _config(901),
        scenario_manifest=generate_manifest("train", 901, 1, num_gt=2),
        method_spec=method,
        reward_step_record_sink=training_rows.append,
    )
    assert len(training_rows) == 1
    assert not training_rows[0]["evaluation"]
    _assert_reconstructable(training_rows[0])
    assert training["run_metadata"]["reward_step_records_enabled"]

    manifest = generate_manifest("test", 902, 1, num_gt=2)
    transitions_without = []
    transitions_with = []
    baseline = train(
        _config(902),
        scenario_manifest=manifest,
        method_spec=method,
        evaluation=True,
        transition_observer=transitions_without.append,
    )
    evaluation_rows = []
    recorded = train(
        _config(902),
        scenario_manifest=manifest,
        method_spec=method,
        evaluation=True,
        transition_observer=transitions_with.append,
        reward_step_record_sink=evaluation_rows.append,
    )
    assert len(evaluation_rows) == 1
    assert evaluation_rows[0]["evaluation"]
    _assert_reconstructable(evaluation_rows[0])
    assert baseline["episode_metrics"] == recorded["episode_metrics"]
    assert len(transitions_without) == len(transitions_with) == 1
    for key in (
        "projected_joint_action",
        "actual_displacement_m",
        "state",
        "next_state",
    ):
        assert (transitions_without[0][key] == transitions_with[0][key]).all()


def test_reward_step_cli_flag_and_streaming_jsonl_sink(tmp_path):
    args = build_parser().parse_args(
        ["kkm_random_action_random_routing", "--smoke", "--record-reward-steps"]
    )
    assert args.record_reward_steps
    path = tmp_path / "reward_step_records.jsonl"
    sink = _reward_step_jsonl_sink(path)
    sink({"schema_version": "uav-hrl-movement-reward-step-v1", "value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": "uav-hrl-movement-reward-step-v1",
        "value": 1,
    }
