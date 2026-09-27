import json
from pathlib import Path

import numpy as np
import pytest

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from HRL_task_aware import _interval_reward
from llm_baseline import (
    BaselineDataError,
    balanced_allocation,
    build_fixed_samples,
    estimate_empirical_lipschitz,
    load_fixed_samples,
    load_sampling_source,
    reconstruct_baseline_rewards,
    run_baseline,
)
from replay_auxiliary import empty_snapshot, replay_auxiliary_metadata
from run_llm_baseline import build_parser
from scenario_manifest import generate_manifest
from utils_update_v2 import ReplayBufferJoint


def _write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _sampling_source(tmp_path, name, *, episodes=7, horizon=3, checkpoint=250):
    directory = tmp_path / name
    directory.mkdir()
    manifest = generate_manifest(
        "test", 4000 + checkpoint, episodes, balanced_num_gt=True
    )
    manifest.save(directory / "scenario_manifest.json")
    replay = ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=episodes * horizon,
        record_auxiliary=True,
    )
    transition = 0
    for episode, scenario in enumerate(manifest.episodes):
        for step in range(horizon):
            state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
            state[0] = checkpoint / 2000.0
            state[1] = episode / max(episodes, 1)
            state[2] = step / max(horizon, 1)
            current = empty_snapshot()
            current["snapshot_valid"][0] = True
            current["snapshot_time_s"][0] = step
            # Legal empty/unknown state: no RoI observed, no queue entries, no links.
            next_snapshot = empty_snapshot()
            next_snapshot["snapshot_valid"][0] = True
            next_snapshot["snapshot_time_s"][0] = min(step + 1, horizon)
            replay.add(
                state,
                np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
                state + np.float32(0.001),
                done=step == horizon - 1,
                delivered_mbits=episode + step / 10.0,
                total_mobility_energy=100.0 + step,
                c9_penalty=0.1,
                c10_penalty=0.2,
                com_range_penalty=0.3,
                current_movement_mask=np.zeros(16, dtype=bool),
                next_movement_mask=np.zeros(16, dtype=bool),
                current_auxiliary_snapshot=current,
                next_auxiliary_snapshot=next_snapshot,
                episode_id=episode,
                td3_step=step,
                global_transition_id=transition,
                scenario_index=episode,
                scenario_id=scenario["scenario_id"],
                dinkelbach_lambda=0.123,
            )
            transition += 1
    replay.save_npz(directory / "joint_replay.npz")
    metadata = {
        "schema_version": "uav-hrl-llm-sampling-v1",
        "status": "complete",
        "complete": True,
        "method_id": "td3_dinkelbach",
        "checkpoint_completed_episodes": checkpoint,
        "episode_seconds": horizon,
        "collector_wrapped": False,
        "transition_count": transition,
        "scenario_manifest_hash": manifest.content_hash,
        "scenario_ids_by_index": [
            str(item["scenario_id"]) for item in manifest.episodes
        ],
        "replay_auxiliary": replay_auxiliary_metadata(),
        "joint_replay_field_order": list(replay.all_fields),
        "source_checkpoint_contract": {
            "state_contract": "fixture",
            "movement_state_dim": MOVEMENT_STATE_DIM,
            "task_potential_configuration": {"enabled": True},
        },
        "source_training_environment_contract": {
            "training_environment_width_m": 1000,
            "training_environment_height_m": 1000,
        },
    }
    _write_json(directory / "metadata.json", metadata)
    return directory


def _minimal_arrays(states, delivered, energy=None, penalties=None):
    count = len(states)
    energy = np.zeros(count) if energy is None else np.asarray(energy)
    penalties = np.zeros((count, 3)) if penalties is None else np.asarray(penalties)
    return {
        "state": np.asarray(states, dtype=np.float32),
        "delivered_mbits": np.asarray(delivered, dtype=np.float32).reshape(-1, 1),
        "total_mobility_energy": np.asarray(energy, dtype=np.float32).reshape(-1, 1),
        "c9_penalty": penalties[:, 0].astype(np.float32).reshape(-1, 1),
        "c10_penalty": penalties[:, 1].astype(np.float32).reshape(-1, 1),
        "com_range_penalty": penalties[:, 2].astype(np.float32).reshape(-1, 1),
    }


def test_cli_defaults_and_multiple_sources():
    args = build_parser().parse_args(
        ["--source", "one", "--source", "two"]
    )
    assert args.source == ["one", "two"]
    assert args.samples_per_source == 1000
    assert args.seed == 20260927
    assert args.batch_size == 128
    assert args.lambdas is None


def test_sampling_is_reproducible_balanced_and_preserves_legal_empty_state(tmp_path):
    first_source = _sampling_source(tmp_path, "a", checkpoint=250)
    second_source = _sampling_source(tmp_path, "b", checkpoint=750)
    first, first_meta = build_fixed_samples(
        [second_source, first_source], samples_per_source=14, seed=19
    )
    second, second_meta = build_fixed_samples(
        [first_source, second_source], samples_per_source=14, seed=19
    )
    assert first_meta["sample_content_sha256"] == second_meta["sample_content_sha256"]
    assert np.array_equal(first["source_row"], second["source_row"])
    assert len(set(zip(first["source_index"], first["source_row"]))) == 28
    assert first_meta["overall_distributions"]["sources"] == {"0": 14, "1": 14}
    assert all(not source["exclusion_counts"]["excluded_union"] for source in first_meta["sources"])
    assert not first["current_movement_mask"].any()
    assert not first["current_roi_observable"].any()
    assert not first["current_sr_queue_valid"].any()


def test_balanced_allocation_redistributes_shortage_with_trace():
    allocation, adjustments = balanced_allocation(
        {2: 1, 3: 9}, 6, seed_material="fixed"
    )
    assert allocation == {2: 1, 3: 5}
    assert {item["stratum"] for item in adjustments} == {"2", "3"}
    assert all(item["reason"] == "local_capacity_shortage_redistribution" for item in adjustments)


def test_source_shortage_missing_field_and_incompatible_sources_fail(tmp_path):
    source = _sampling_source(tmp_path, "source", episodes=2, horizon=2)
    with pytest.raises(BaselineDataError, match="fewer than requested"):
        build_fixed_samples([source], samples_per_source=5)

    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("metadata.json", "scenario_manifest.json"):
        (broken / name).write_bytes((source / name).read_bytes())
    with np.load(source / "joint_replay.npz", allow_pickle=False) as archive:
        np.savez_compressed(
            broken / "joint_replay.npz",
            **{key: archive[key] for key in archive.files if key != "current_snapshot_valid"},
        )
    with pytest.raises(BaselineDataError, match="field set disagrees"):
        load_sampling_source(broken)

    incompatible = _sampling_source(tmp_path, "incompatible", episodes=2, horizon=2, checkpoint=750)
    metadata_path = incompatible / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_training_environment_contract"]["training_environment_width_m"] = 2000
    _write_json(metadata_path, metadata)
    with pytest.raises(BaselineDataError, match="incompatible"):
        build_fixed_samples([source, incompatible], samples_per_source=2)


def test_reward_reconstruction_matches_complete_training_reward():
    arrays = _minimal_arrays(
        [[0.0], [1.0]],
        delivered=[3.0, 4.0],
        energy=[100.0, 200.0],
        penalties=[[0.1, 0.2, 0.3], [0.0, 0.4, 0.2]],
    )
    value = reconstruct_baseline_rewards(arrays, [0.001])["0.001"]
    for index in range(2):
        expected = _interval_reward(
            arrays["delivered_mbits"][index, 0],
            arrays["total_mobility_energy"][index, 0],
            0.001,
            {
                "c9_penalty_mean": arrays["c9_penalty"][index, 0],
                "c10_penalty_mean": arrays["c10_penalty"][index, 0],
                "com_range_penalty_mean": arrays["com_range_penalty"][index, 0],
            },
        )
        assert value[index] == pytest.approx(expected)


def test_lipschitz_hand_case_and_batch_size_invariance():
    arrays = _minimal_arrays([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], [0.0, 10.0, 15.0])
    rewards = reconstruct_baseline_rewards(arrays, [0.0])
    first = estimate_empirical_lipschitz(
        arrays["state"], rewards, arrays=arrays, batch_size=1
    )
    second = estimate_empirical_lipschitz(
        arrays["state"], rewards, arrays=arrays, batch_size=3
    )
    estimate = first["estimates_by_lambda_mbit_per_joule"]["0"]
    assert estimate["l_hat"] == pytest.approx(2.0)
    assert (estimate["maximum_pair"]["i"], estimate["maximum_pair"]["j"]) == (0, 1)
    assert first["primary_pair_set_sha256"] == second["primary_pair_set_sha256"]
    assert first["distance_distribution_all_pairs"] == second["distance_distribution_all_pairs"]
    assert first["estimates_by_lambda_mbit_per_joule"] == second["estimates_by_lambda_mbit_per_joule"]


def test_zero_near_zero_and_no_primary_pair_are_explicit():
    arrays = _minimal_arrays([[0.0], [0.0], [5e-9]], [0.0, 1.0, 2.0])
    rewards = reconstruct_baseline_rewards(arrays, [0.0])
    report = estimate_empirical_lipschitz(
        arrays["state"],
        rewards,
        arrays=arrays,
        batch_size=2,
        distance_epsilon=1e-8,
        reward_epsilon=1e-12,
    )
    estimate = report["estimates_by_lambda_mbit_per_joule"]["0"]
    assert report["zero_distance_pair_count"] == 1
    assert report["near_zero_distance_pair_count"] == 2
    assert report["primary_pair_count"] == 0
    assert estimate["status"] == "unavailable_no_primary_pairs"
    assert estimate["l_hat"] is None
    assert estimate["zero_distance"]["reward_conflict_count"] == 1
    assert estimate["near_zero_distance"]["maximum_calculable_ratio_pair"]["ratio"] == pytest.approx(4e8)


def test_fixed_sample_save_reload_reuses_identical_samples_and_pairs(tmp_path):
    source = _sampling_source(tmp_path, "source", episodes=7, horizon=3)
    first = run_baseline(
        source_directories=[source],
        samples_per_source=14,
        seed=71,
        lambdas=[0.0, 0.001],
        batch_size=3,
        output_dir=tmp_path / "baseline-first",
    )
    fixed_arrays, fixed_metadata = load_fixed_samples(first["output_directory"])
    assert fixed_arrays["state"].shape == (14, MOVEMENT_STATE_DIM)
    assert fixed_metadata["pair_contract"]["primary_pair_set_sha256"] == first["report"]["lipschitz"]["primary_pair_set_sha256"]

    second = run_baseline(
        source_directories=[source],
        fixed_sample_directory=first["output_directory"],
        lambdas=None,
        batch_size=5,
        output_dir=tmp_path / "baseline-second",
    )
    assert second["report"]["reloaded_fixed_samples"] is True
    assert second["report"]["lipschitz"]["primary_pair_set_sha256"] == first["report"]["lipschitz"]["primary_pair_set_sha256"]
    assert second["report"]["lipschitz"]["estimates_by_lambda_mbit_per_joule"] == first["report"]["lipschitz"]["estimates_by_lambda_mbit_per_joule"]

    with pytest.raises(BaselineDataError, match="lambda list"):
        run_baseline(
            fixed_sample_directory=first["output_directory"],
            lambdas=[0.0],
            output_dir=tmp_path / "baseline-third",
        )

    bundle = Path(first["output_directory"]) / "fixed_samples.npz"
    payload = bytearray(bundle.read_bytes())
    payload[-1] ^= 1
    bundle.write_bytes(payload)
    with pytest.raises(BaselineDataError, match="SHA-256 changed"):
        load_fixed_samples(first["output_directory"])
