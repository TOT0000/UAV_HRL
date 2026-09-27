import copy

import numpy as np

from Packet_scheduler_v1 import PacketEngine, TASK_DEADLINE_SECONDS
from Simulator import Simulator
from replay_auxiliary import (
    capture_replay_snapshot,
    empty_snapshot,
    replay_auxiliary_metadata,
)
from training_checkpoint import JOINT_REPLAY_FIELDS, _load_replay, _save_replay
from utils_update_v2 import ReplayBufferJoint


def _scene():
    env = Simulator(num_UAV=16)
    env.num_GT = 2
    env.reset_environment()
    target = env.gts[0]
    target.mark_found(1)
    sr = env.SR_team_gogo(target)
    env.multi_tasks = {uid: [{"task_type": "Search"}] for uid in range(16)}
    env.multi_tasks[1] = [
        {
            "task_type": "FOV",
            "target_id": 0,
            "target_obj_id": 0,
            "target_pos": target.get_position(),
        },
        {
            "task_type": "COM",
            "target_id": int(sr.id),
            "target_obj_id": int(sr.id),
            "target_pos": sr.get_position(),
        },
    ]
    engine = PacketEngine(num_uav=16)
    fov = engine.create_packet(1, "FOV", 100.0, 0.0)
    fov["rem_bits"] = 40.0
    com = engine.create_packet(1, "COM", 200.0, 0.25)
    com["rem_bits"] = 125.0
    engine.backlog_bits[1] = 165.0
    engine.create_sr_packet(int(sr.id), 300.0, 0.5)
    return env, engine, sr


def test_snapshot_preserves_fifo_raw_values_deadlines_tasks_and_visibility():
    env, engine, sr = _scene()
    channel_before = copy.deepcopy(env.channel_state_dict())
    rngs = (
        env.environment_rng,
        env.assignment_rng,
        env.channel_large_scale_rng,
        env.channel_small_scale_rng,
    )
    rng_before = [copy.deepcopy(rng.bit_generator.state) for rng in rngs]
    snapshot = capture_replay_snapshot(env, engine, 1.0)

    assert snapshot["snapshot_valid"][0]
    assert snapshot["uav_backlog_bits"][1] == 165.0
    assert snapshot["uav_vs_backlog_bits"][1] == 40.0
    assert snapshot["uav_com_backlog_bits"][1] == 125.0
    assert snapshot["uav_hol_type"][1] == 1
    assert snapshot["uav_hol_remaining_deadline_s"][1] == (
        TASK_DEADLINE_SECONDS["FOV"] - 1.0
    )
    assert snapshot["uav_vs_min_remaining_deadline_s"][1] == (
        TASK_DEADLINE_SECONDS["FOV"] - 1.0
    )
    assert snapshot["uav_com_min_remaining_deadline_s"][1] == (
        TASK_DEADLINE_SECONDS["COM"] - 0.75
    )
    assert snapshot["sr_observable"][sr.id]
    assert snapshot["sr_packet_count"][sr.id] == 1
    assert snapshot["sr_hol_remaining_deadline_s"][sr.id] == (
        TASK_DEADLINE_SECONDS["COM"] - 0.5
    )
    assert snapshot["roi_observable"].sum() == 1
    assert snapshot["roi_id"][0] == 0
    assert snapshot["task_pair_valid"][1].sum() == 2
    assert snapshot["vs_pair_valid"][1]
    assert snapshot["vs_image_quantity"][1] >= 0.0
    assert env.channel_state_dict().keys() == channel_before.keys()
    for key, before in channel_before.items():
        after = env.channel_state_dict()[key]
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(after, before)
        else:
            assert after == before
    assert [rng.bit_generator.state for rng in rngs] == rng_before


def test_current_next_assignment_alignment_and_snapshot_copy():
    env, engine, sr = _scene()
    current = capture_replay_snapshot(env, engine, 0.0)
    env.multi_tasks[1] = [{"task_type": "Hovering"}]
    next_snapshot = capture_replay_snapshot(env, engine, 1.0)
    replay = ReplayBufferJoint(2, 3, max_size=2, record_auxiliary=True)
    replay.add(
        [1, 2],
        [0, 0, 0],
        [3, 4],
        False,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        current_auxiliary_snapshot=current,
        next_auxiliary_snapshot=next_snapshot,
        episode_id=7,
        td3_step=3,
        global_transition_id=99,
        scenario_index=7,
        scenario_id="scenario-a",
        dinkelbach_lambda=0.25,
    )
    current["task_type"].fill(0)
    assert replay.current_task_pair_valid[0, 1].sum() == 2
    assert replay.next_task_pair_valid[0, 1].sum() == 1
    assert replay.current_vs_pair_valid[0, 1]
    assert not replay.next_vs_pair_valid[0, 1]
    assert replay.episode_id[0, 0] == 7
    assert replay.dinkelbach_lambda[0, 0] == 0.25


def test_missing_objects_empty_queues_and_invalid_vs_geometry_stay_masked():
    env, engine, _sr = _scene()
    env.multi_tasks[2] = []
    target_position = np.asarray(env.gts[0].get_position(), dtype=float)
    env.uav_dict[1].x_u = float(target_position[0] + 1_000.0)
    env.uav_dict[1].y_u = float(target_position[1])
    env.uav_dict[1].z_u = float(target_position[2] + 100.0)
    snapshot = capture_replay_snapshot(env, engine, 0.0)

    assert snapshot["uav_queue_valid"][2]
    assert snapshot["uav_queue_empty"][2]
    assert not snapshot["uav_hol_valid"][2]
    assert not snapshot["task_pair_valid"][2].any()
    assert 1 not in snapshot["roi_id"][snapshot["roi_observable"]]
    assert not snapshot["roi_object_exists"][1]
    np.testing.assert_array_equal(snapshot["roi_position_m"][1], np.zeros(3))
    assert snapshot["vs_pair_valid"][1]
    assert not snapshot["vs_geometry_valid"][1]
    assert not snapshot["vs_c10_geometry_valid"][1]
    assert not snapshot["vs_capture_valid"][1]


def test_ring_alignment_npz_no_pickle_and_legacy_missing_auxiliary(tmp_path):
    replay = ReplayBufferJoint(2, 3, max_size=2, record_auxiliary=True)
    for transition_id in range(3):
        current = empty_snapshot()
        next_snapshot = empty_snapshot()
        current["snapshot_valid"][0] = True
        next_snapshot["snapshot_valid"][0] = True
        current["snapshot_time_s"][0] = transition_id
        next_snapshot["snapshot_time_s"][0] = transition_id + 1
        replay.add(
            [transition_id, 0],
            [0, 0, 0],
            [transition_id + 1, 0],
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            current_auxiliary_snapshot=current,
            next_auxiliary_snapshot=next_snapshot,
            global_transition_id=transition_id,
        )
    assert replay.ptr == 1 and replay.size == 2
    assert replay.global_transition_id[:, 0].tolist() == [2, 1]
    assert replay.current_snapshot_time_s[:, 0].tolist() == [2.0, 1.0]
    path = tmp_path / "joint_replay.npz"
    replay.save_npz(path)
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["global_transition_id"][:, 0].tolist() == [2, 1]
        assert not any(arrays[name].dtype.hasobject for name in arrays.files)

    legacy = ReplayBufferJoint(2, 3, max_size=2)
    legacy.add([1, 2], [0, 0, 0], [3, 4], False, 0, 0, 0, 0, 0)
    legacy_path = tmp_path / "legacy.npz"
    metadata = _save_replay(legacy_path, legacy, JOINT_REPLAY_FIELDS)
    restored = ReplayBufferJoint(2, 3, max_size=2, record_auxiliary=True)
    _load_replay(legacy_path, restored, JOINT_REPLAY_FIELDS, metadata)
    assert restored.size == 1
    assert not restored.auxiliary_valid[0, 0]


def test_auxiliary_cost_estimate_is_explicit_and_fixed_shape():
    metadata = replay_auxiliary_metadata()
    assert metadata["auxiliary_bytes_per_transition"] > 0
    assert metadata["current_and_next_fields"]["u2u_distance_m"][
        "shape_per_transition"
    ] == [16, 16]
    assert metadata["unknown_roi_policy"].startswith("undiscovered")
