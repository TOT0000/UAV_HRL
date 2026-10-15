import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from DDQN import DDQN
from experiment_config import METHOD_REGISTRY, MethodSpec
from llm_explorer import (
    LLMExplorerController,
    build_observable_initial_state,
    masked_routing_exploration_distribution,
    parse_guidance_response,
    render_prompt,
    stratified_routing_samples,
)
from td3 import TD3


def _guidance(td3_value=0.0, routing=None):
    if routing is None:
        routing = [[1.0 / 17.0] * 17 for _ in range(16)]
    return json.dumps(
        {
            "td3_bias": [[td3_value, -td3_value, 0.0] for _ in range(16)],
            "routing_prior": routing,
        }
    )


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return {
            "content": response,
            "actual_model": kwargs["model"],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "finish_reason": "stop",
        }


def _previous_feedback():
    return {
        "metrics": {
            "energy_efficiency_bit_per_j": 1.0,
            "td3_original_cumulative_reward": 2.0,
            "safe_ddqn_original_cumulative_reward": 3.0,
        },
        "movement_decisions": [],
        "routing_samples": [],
        "full_episode_statistics": {"routing": {"total_real_decisions": 0}},
        "applied_guidance": {
            "enabled": False,
            "td3_reason": "warmup",
            "routing_reason": "warmup",
        },
    }


def _begin(controller, transitions=10_000, episode_index=167):
    return controller.begin_episode(
        episode_index=episode_index,
        total_joint_transitions=transitions,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={"known_rois": [], "uavs": []},
        environment_constants={"episode_seconds": 60},
    )


def test_registry_appends_explorer_without_using_legacy_llm_flag():
    assert tuple(METHOD_REGISTRY)[-1] == "td3_dinkelbach_llm_explorer"
    method = MethodSpec.parse("td3_dinkelbach_llm_explorer")
    baseline = MethodSpec.parse("td3_dinkelbach")
    assert method.llm_explorer_enabled
    assert not method.llm_enabled
    assert not baseline.llm_explorer_enabled
    assert method.agent == baseline.agent == "td3"
    assert method.routing == baseline.routing == "safe_ddqn"
    assert method.reward_mode == baseline.reward_mode == "dinkelbach"


def test_prompt_renderer_does_not_reinterpret_inserted_braces():
    inserted = 'analysis containing {{literal_model_text}} and {"x": 1}'
    assert render_prompt("prefix {{VALUE}} suffix", {"VALUE": inserted}) == (
        f"prefix {inserted} suffix"
    )


def test_mask_then_normalize_then_80_20_mix_and_edge_cases():
    prior = np.zeros(17)
    prior[3] = 3.0
    prior[5] = 1.0
    mask = np.zeros(17, dtype=bool)
    mask[[3, 5]] = True
    distribution, zero_mass = masked_routing_exploration_distribution(prior, mask)
    assert not zero_mass
    np.testing.assert_allclose(distribution[[3, 5]], [0.7, 0.3])
    assert distribution.sum() == pytest.approx(1.0)
    assert np.all(distribution[~mask] == 0.0)

    distribution, zero_mass = masked_routing_exploration_distribution(
        np.eye(1, 17, 2).ravel(), mask
    )
    assert zero_mass
    np.testing.assert_allclose(distribution[[3, 5]], [0.5, 0.5])

    single = np.zeros(17, dtype=bool)
    single[16] = True
    distribution, _ = masked_routing_exploration_distribution(prior, single)
    assert distribution[16] == pytest.approx(1.0)


def test_safe_ddqn_uses_prior_only_in_epsilon_branch():
    agent = DDQN(state_dim=4, action_dim=17, master_seed=4)

    class CaptureRng:
        def __init__(self):
            self.probabilities = None

        def random(self):
            return 0.0

        def choice(self, values, p=None):
            self.probabilities = p
            return int(np.asarray(values)[np.argmax(p)])

        def normal(self, *args, **kwargs):
            return np.zeros(kwargs.get("size", args[-1] if args else 17))

    rng = CaptureRng()
    agent.exploration_rng = rng
    mask = np.zeros(17, dtype=bool)
    mask[[1, 4]] = True
    prior = np.zeros(17)
    prior[4] = 1.0
    action, detail = agent.select_action(
        np.zeros(4),
        0,
        mask=mask,
        epsilon=1.0,
        logits_noise_std=0.0,
        exploration_prior=prior,
        return_diagnostics=True,
    )
    assert action == 4
    np.testing.assert_allclose(rng.probabilities[[1, 4]], [0.1, 0.9])
    assert detail["decision_branch"] == "epsilon_exploration"

    rng.probabilities = None
    agent.select_action(
        np.zeros(4),
        0,
        mask=mask,
        epsilon=0.0,
        logits_noise_std=0.0,
        exploration_prior=prior,
    )
    assert rng.probabilities is None


def test_td3_bias_is_actual_offset_before_field_projection():
    model = TD3(state_dim=2, action_dim=3, max_action=1.0, master_seed=2)

    class ConstantActor(torch.nn.Module):
        def forward(self, state):
            return state.new_tensor([[0.9, 0.9, -0.9]])

    model.actor = ConstantActor()
    action, detail = model.select_action(
        np.zeros(2, dtype=np.float32),
        add_noise=False,
        noise_std=0.3,
        mean_bias=np.array([0.2, 0.3, -0.2]),
        return_diagnostics=True,
    )
    np.testing.assert_allclose(detail["pre_projection_action"], [1.1, 1.2, -1.1])
    np.testing.assert_allclose(action, [1.0, -0.8, -1.0], atol=1e-6)
    np.testing.assert_allclose(detail["mean_bias"], [0.2, 0.3, -0.2])


@pytest.mark.parametrize(
    "content,error_layer",
    [
        ("truncated {", "td3"),
        (_guidance(float("nan")), "td3"),
        (json.dumps({"td3_bias": [[0.0] * 3], "routing_prior": [[1 / 17] * 17] * 16}), "td3"),
        (_guidance(0.0, routing=[[-0.1] + [1.1 / 16] * 16] * 16), "routing"),
    ],
)
def test_guidance_json_validation_rejects_bad_values(content, error_layer):
    _, _, errors = parse_guidance_response(content, sigma_episode_start=0.2)
    assert error_layer in errors


def test_warmup_has_no_calls_then_two_stage_guidance_at_transition_boundary(tmp_path):
    client = SequenceClient(["analysis", _guidance(0.1)])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "test"},
        client=client,
        rng_seed=7,
    )
    controller.previous_episode_feedback = _previous_feedback()
    warmup = _begin(controller, transitions=9_999, episode_index=10)
    assert not warmup["enabled"]
    assert warmup["td3_reason"] == "warmup"
    assert client.calls == []

    guidance = _begin(controller, transitions=10_000, episode_index=11)
    assert guidance["enabled"]
    assert not guidance["td3_fallback"]
    assert not guidance["routing_fallback"]
    assert len(client.calls) == 2
    assert all(call["model"] == "gpt-4o-mini" for call in client.calls)
    assert all("system_prompt" in call for call in client.calls)


def test_partial_layer_retry_keeps_valid_first_layer_and_counts_retry(tmp_path):
    bad_routing = [[1.0 / 17.0] * 17 for _ in range(16)]
    bad_routing[0][0] = -1.0
    client = SequenceClient(
        [
            "analysis",
            _guidance(0.1, bad_routing),
            json.dumps(
                {
                    "td3_bias": [[9.0, 0.0, 0.0]] * 16,
                    "routing_prior": [[1.0 / 17.0] * 17 for _ in range(16)],
                }
            ),
        ]
    )
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "partial"},
        client=client,
    )
    controller.previous_episode_feedback = _previous_feedback()
    guidance = _begin(controller)
    assert not guidance["td3_fallback"]
    assert not guidance["routing_fallback"]
    assert guidance["td3_guidance_attempt"] == 1
    assert guidance["routing_guidance_attempt"] == 2
    assert controller.statistics["failed_attempts_total"] == 1
    assert controller.statistics["retry_success_total"] == 1
    assert controller.statistics["fallback_episodes_total"] == 0


def test_analysis_failure_skips_generation_and_falls_back(tmp_path):
    client = SequenceClient([TimeoutError("first"), TimeoutError("second")])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "timeout"},
        client=client,
    )
    controller.previous_episode_feedback = _previous_feedback()
    guidance = _begin(controller)
    assert guidance["td3_fallback"] and guidance["routing_fallback"]
    assert len(client.calls) == 2
    assert controller.statistics["failed_attempts_total"] == 2
    assert controller.statistics["fallback_episodes_total"] == 1
    events = (tmp_path / "llm_explorer" / "events.jsonl").read_text()
    assert "TimeoutError" in events


def test_matching_saved_responses_replay_without_new_requests(tmp_path):
    first_client = SequenceClient(["analysis", _guidance(0.05)])
    first = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "cache"},
        client=first_client,
    )
    first.previous_episode_feedback = _previous_feedback()
    original = _begin(first)
    assert len(first_client.calls) == 2

    no_call_client = SequenceClient([])
    replay = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "cache"},
        client=no_call_client,
    )
    replay.previous_episode_feedback = _previous_feedback()
    restored = _begin(replay)
    assert no_call_client.calls == []
    assert restored["reused"]
    np.testing.assert_allclose(restored["td3_bias"], original["td3_bias"])
    assert replay.statistics["requests_total"] == 2


def test_stratified_routing_sampling_does_not_invent_decisions():
    records = [
        {"sender_uav_id": 0, "time_seconds": float(index), "selected_action": 0}
        for index in range(150)
    ] + [
        {"sender_uav_id": 3, "time_seconds": float(index), "selected_action": 16}
        for index in range(3)
    ]
    sampled = stratified_routing_samples(records, maximum=100)
    assert len(sampled) == 100
    assert {item["sender_uav_id"] for item in sampled} == {0, 3}
    assert sum(item["sender_uav_id"] == 3 for item in sampled) == 3
    assert stratified_routing_samples([], maximum=100) == []


def test_hidden_unobserved_roi_changes_do_not_change_payload():
    hidden = SimpleNamespace(
        id=0, x=100.0, y=200.0, z=0.0, radius=30.0, is_found=False
    )
    known = SimpleNamespace(
        id=1, x=300.0, y=400.0, z=0.0, radius=40.0, is_found=True
    )
    uav = SimpleNamespace(x_u=1.0, y_u=2.0, z_u=100.0, energy=9000.0, battery=90.0)
    sr_hidden = SimpleNamespace(
        id=0, x=9.0, y=8.0, z=0.0, assigned_gt_id=0, arrived=False
    )
    sr_known = SimpleNamespace(
        id=1, x=7.0, y=6.0, z=0.0, assigned_gt_id=1, arrived=True
    )
    env = SimpleNamespace(
        gts=[hidden, known],
        SR_teams=[sr_hidden, sr_known],
        multi_tasks={0: []},
        uav_dict={0: uav},
        GS_pos=np.array([0.0, 0.0, 0.0]),
    )
    packet_engine = SimpleNamespace(backlog_bits={0: 0.0})
    before = build_observable_initial_state(env, packet_engine)
    hidden.x = 99999.0
    hidden.radius = 999.0
    sr_hidden.x = 88888.0
    after = build_observable_initial_state(env, packet_engine)
    assert before == after
    serialized = json.dumps(after)
    assert "99999" not in serialized
    assert "88888" not in serialized
    assert "num_GT" not in serialized


def test_evaluation_controller_never_constructs_or_calls_client(tmp_path):
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "evaluation"},
        evaluation=True,
        client=SequenceClient([AssertionError("must not call")]),
    )
    result = _begin(controller)
    assert not result["enabled"]
    assert result["td3_reason"] == "evaluation_disabled"
    assert controller.statistics["requests_total"] == 0
