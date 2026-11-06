import numpy as np
import pytest

from HRL_task_aware import _interval_reward, _movement_complete_useful_bits
from experiment_config import METHOD_REGISTRY, MethodSpec, comparison_method_configuration
from llm_explorer import (
    LLM_EXPLORER_CALL_SCHEDULE_VERSION,
    FirstGenLLMExplorerConfig,
    LLMExplorerController,
    environment_constants_and_definitions_first_gen,
    explorer_metadata,
    llm_explorer_call_scheduled,
)


class _NoCallsClient:
    def __init__(self):
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("a schedule-skipped episode must not call the LLM")


@pytest.mark.parametrize(
    ("episode", "scheduled"),
    [
        (1, True),
        (500, True),
        (501, True),
        (502, False),
        (999, True),
        (1000, False),
        (1001, True),
        (1004, False),
        (1005, True),
        (1497, True),
        (1500, False),
        (1501, True),
        (1505, True),
    ],
)
def test_first_gen_schedule_boundaries_use_global_one_based_episode(
    episode, scheduled
):
    assert llm_explorer_call_scheduled(episode) is scheduled


def _begin(controller, *, episode_index, transitions):
    return controller.begin_episode(
        episode_index=episode_index,
        total_joint_transitions=transitions,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={"known_rois": [], "uavs": []},
        environment_constants={"episode_seconds": 60},
    )


def test_warmup_and_schedule_skips_are_distinct_and_clear_guidance():
    client = _NoCallsClient()
    controller = LLMExplorerController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer"},
        config=FirstGenLLMExplorerConfig(),
        client=client,
    )
    controller.previous_episode_feedback = {"episode": 499}

    warmup = _begin(controller, episode_index=499, transitions=9_999)
    assert warmup["td3_reason"] == "warmup"

    controller.active_guidance = {
        **controller._unguided("test"),
        "enabled": True,
        "td3_bias": np.ones((16, 3), dtype=np.float32),
        "routing_weights": np.ones((16, 17), dtype=np.float64),
    }
    skipped = _begin(controller, episode_index=501, transitions=10_000)
    assert skipped["td3_reason"] == "schedule_skip"
    assert skipped["routing_reason"] == "schedule_skip"
    assert np.count_nonzero(skipped["td3_bias"]) == 0
    assert skipped["routing_weights"] is None
    assert skipped["routing_prior"] is None
    assert client.calls == 0


def test_schedule_identity_and_exact_resume_keep_global_episode_number():
    config = FirstGenLLMExplorerConfig()
    source = LLMExplorerController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer"},
        config=config,
        client=_NoCallsClient(),
    )
    source.previous_episode_feedback = {"episode": 501}
    state = source.state_dict()

    client = _NoCallsClient()
    restored = LLMExplorerController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer"},
        config=config,
        client=client,
    )
    restored.load_state_dict(state)
    result = _begin(restored, episode_index=501, transitions=10_000)
    assert result["td3_reason"] == "schedule_skip"
    assert client.calls == 0
    metadata = explorer_metadata(config)
    assert metadata["call_schedule_version"] == LLM_EXPLORER_CALL_SCHEDULE_VERSION


def test_late_complete_bits_feed_movement_but_not_routing_reward():
    slot = {
        "total_timely_useful_bits": 0.0,
        "total_late_useful_bits": 250_000.0,
        "total_all_useful_bits": 250_000.0,
        "global_routing_reward": 0.0,
    }
    delivered_mbits = _movement_complete_useful_bits(slot) / 1e6
    movement_reward = _interval_reward(
        delivered_mbits,
        0.0,
        0.0,
        {
            "c9_penalty_mean": 0.0,
            "c10_penalty_mean": 0.9,
            "com_range_penalty_mean": 0.0,
        },
    )
    assert delivered_mbits == pytest.approx(0.25)
    assert movement_reward == pytest.approx(0.25)
    assert slot["global_routing_reward"] == 0.0


def test_partial_or_unfinished_bits_are_not_movement_goodput():
    slot = {
        "total_timely_useful_bits": 0.0,
        "total_late_useful_bits": 0.0,
        "total_all_useful_bits": 0.0,
        "raw_final_hop_bits": 123_456.0,
    }
    assert _movement_complete_useful_bits(slot) == 0.0


def test_every_learning_method_disables_c10_but_keeps_diagnostics():
    for method_id in METHOD_REGISTRY:
        method = MethodSpec.parse(method_id)
        if not method.learns_movement:
            continue
        assert not method.c10_penalty_enabled, method_id
        resolved = comparison_method_configuration(method)
        assert resolved["effective_movement_constraint_penalty_weights"]["c10"] == 0.0
        assert resolved["task_potential_configuration"]["c10"][
            "reward_penalty_enabled"
        ] is False

    environment = environment_constants_and_definitions_first_gen(
        episode_seconds=60,
        routing_slot_seconds=0.25,
        environment_width_m=1000,
        environment_height_m=1000,
    )
    reward = environment["movement_reward"]
    assert reward["c10_penalty_enabled"] is False
    assert "P_C10" not in reward["formula"]
    assert reward["penalty_contract"]["C10"]["role"] == (
        "diagnostic_only_not_subtracted_from_movement_reward"
    )
