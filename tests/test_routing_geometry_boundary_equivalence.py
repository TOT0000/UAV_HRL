from unittest import mock

import pytest

from Packet_scheduler_v1 import PacketEngine
from Simulator import COMMUNICATION_RANGE_M, Simulator
from rng_contract import NamedRNGStreams


OUTSIDE_WHEN_VECTORIZED = (
    320.87354054226296,
    235.8662355217209,
    137.51386304667665,
)
INSIDE_WHEN_VECTORIZED = (
    282.05215222731897,
    283.58188529097686,
    105.28183289706679,
)
REFERENCE_ORIGIN = (0.0, 0.0, 100.0)


def _environment(seed=20260817, *, geometry_cache_enabled=True):
    env = Simulator(16, rng_streams=NamedRNGStreams(seed))
    env.num_GT = 2
    env.reset_environment()
    env._routing_geometry_cache_enabled = bool(geometry_cache_enabled)
    return env


def _set_uav_position(env, uav_id, position):
    uav = env.uav_dict[int(uav_id)]
    uav.x_u, uav.y_u, uav.z_u = map(float, position)


@pytest.mark.parametrize(
    ("second_position", "reported_scalar_distance"),
    (
        (OUTSIDE_WHEN_VECTORIZED, 400.00000000000006),
        (INSIDE_WHEN_VECTORIZED, 400.0),
    ),
)
def test_u2u_cache_preserves_scalar_distance_boundary_semantics(
    second_position, reported_scalar_distance
):
    env = _environment()
    _set_uav_position(env, 0, REFERENCE_ORIGIN)
    _set_uav_position(env, 1, second_position)

    # NumPy's scalar and batched norm kernels can round these two vectors on
    # opposite sides of 400 m depending on the CPU/BLAS build.  The cache
    # contract is therefore the old per-link distance_3d call, not a batched
    # value that merely happens to agree on this host.
    native_distance = env.distance_3d

    def reported_distance(first, second):
        if (tuple(first), tuple(second)) == (
            REFERENCE_ORIGIN,
            second_position,
        ):
            return reported_scalar_distance
        return native_distance(first, second)

    with mock.patch.object(
        env, "distance_3d", side_effect=reported_distance
    ) as scalar_distance:
        geometry = env._physical_routing_geometry(validate_uav_ids=(0, 1))
    assert (
        mock.call(REFERENCE_ORIGIN, second_position)
        in scalar_distance.call_args_list
    )
    assert geometry["u2u_distances"][0, 1] == reported_scalar_distance
    assert env.is_u2u_in_range(0, 1) is (
        reported_scalar_distance <= COMMUNICATION_RANGE_M
    )


@pytest.mark.parametrize(
    "endpoint",
    (
        (200.0, 300.0, 200.0),
        OUTSIDE_WHEN_VECTORIZED,
        INSIDE_WHEN_VECTORIZED,
        (400.0, 0.0, 100.0),
        (400.00000000000006, 0.0, 100.0),
        (400.001, 0.0, 100.0),
    ),
)
def test_cached_u2u_u2g_and_s2u_distances_exactly_match_scalar_reference(endpoint):
    env = _environment()
    _set_uav_position(env, 0, REFERENCE_ORIGIN)
    _set_uav_position(env, 1, endpoint)
    env.GS_pos = tuple(endpoint)
    sr = env.SR_teams[0]
    sr.x, sr.y, sr.z = REFERENCE_ORIGIN

    u2u_reference = env.distance_3d(REFERENCE_ORIGIN, endpoint)
    u2g_reference = env.distance_3d(REFERENCE_ORIGIN, endpoint)
    s2u_reference = env.distance_3d(REFERENCE_ORIGIN, endpoint)
    geometry = env._physical_routing_geometry(
        validate_uav_ids=(0, 1), validate_sr_ids=(0,), validate_gs=True
    )

    assert geometry["u2u_distances"][0, 1] == u2u_reference
    assert geometry["u2g_distances"][0] == u2g_reference
    assert geometry["s2u_distances"][0, 1] == s2u_reference
    assert env.is_u2u_in_range(0, 1) is (
        u2u_reference <= COMMUNICATION_RANGE_M
    )
    assert env.is_u2g_in_range(0) is (u2g_reference <= COMMUNICATION_RANGE_M)
    assert env.is_s2u_in_range(0, 1) is (
        s2u_reference <= COMMUNICATION_RANGE_M
    )


def _run_boundary_packet_service(*, geometry_cache_enabled):
    env = _environment(geometry_cache_enabled=geometry_cache_enabled)
    for uav_id in range(env.num_UAV):
        _set_uav_position(env, uav_id, (900.0, 900.0, 100.0))
    _set_uav_position(env, 0, REFERENCE_ORIGIN)
    _set_uav_position(env, 1, OUTSIDE_WHEN_VECTORIZED)
    env.GS_pos = (
        OUTSIDE_WHEN_VECTORIZED[0],
        OUTSIDE_WHEN_VECTORIZED[1],
        0.0,
    )
    env.update_u2u_channels()
    env.update_u2g_channels()
    env.prepare_channel_routing_slot(0)

    # Reproduce the reported host's scalar result independently of the
    # vectorized norm kernel installed on the machine running this test.
    original_distance = env.distance_3d

    def reported_scalar_distance(first, second):
        pair = {tuple(first), tuple(second)}
        if pair == {REFERENCE_ORIGIN, OUTSIDE_WHEN_VECTORIZED}:
            return 400.00000000000006
        return original_distance(first, second)

    env.distance_3d = reported_scalar_distance

    engine = PacketEngine(env.num_UAV)
    packet = engine.create_packet(0, "COM", 256.0, generation_time=0.0)
    actions = {sender: sender for sender in range(env.num_UAV)}
    actions[0] = 1
    actions[1] = env.GS_ID
    result = engine.serve_event_driven_slot(env, actions, 0.0)
    return packet, result


def test_boundary_packet_service_matches_uncached_scalar_reference():
    reference_packet, reference_result = _run_boundary_packet_service(
        geometry_cache_enabled=False
    )
    cached_packet, cached_result = _run_boundary_packet_service(
        geometry_cache_enabled=True
    )

    assert reference_packet["terminal_outcome"] == "expired_dropped"
    assert reference_result["timely_goodput_bits"] == 0.0
    assert cached_packet["terminal_outcome"] == reference_packet["terminal_outcome"]
    assert cached_result["timely_goodput_bits"] == reference_result[
        "timely_goodput_bits"
    ]


def test_geometry_cache_invalidates_after_position_reset_and_channel_load():
    env = _environment()
    _set_uav_position(env, 0, REFERENCE_ORIGIN)
    _set_uav_position(env, 1, OUTSIDE_WHEN_VECTORIZED)
    initial_expected = (
        env.distance_3d(REFERENCE_ORIGIN, OUTSIDE_WHEN_VECTORIZED)
        <= COMMUNICATION_RANGE_M
    )
    assert env.is_u2u_in_range(0, 1) is initial_expected
    first_builds = env.routing_geometry_cache_statistics()["builds"]

    _set_uav_position(env, 1, (401.0, 0.0, 100.0))
    assert env.is_u2u_in_range(0, 1) is False
    assert env.routing_geometry_cache_statistics()["builds"] == first_builds + 1

    state = env.channel_state_dict()
    env.load_channel_state_dict(state)
    assert env.routing_geometry_cache_statistics()["populated"] is False
    assert env.is_u2u_in_range(0, 1) is False

    env.reset_environment()
    assert env.routing_geometry_cache_statistics() == {
        "enabled": True,
        "builds": 0,
        "hits": 0,
        "populated": False,
    }
