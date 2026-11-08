import json

import numpy as np
import pytest
import torch
from torch import nn

import DDQN as ddqn_module
from DDQN import DDQN, forwarding_or_forced_wait_mask
from HRL_task_aware import _select_routing_actions
from llm_explorer import (
    FirstGenLLMExplorerConfig,
    LLMExplorerController,
    LLM_EXPLORER_NO_C10_METHOD_ID,
    LLM_EXPLORER_NO_C10_NUMERICAL_OUTPUT_PROTOCOL,
    LLM_EXPLORER_NO_C10_PROMPT_VERSION,
    DeterministicMockExplorerClient,
    ExplorerValidationError,
    explorer_metadata,
    first_gen_numerical_response_format,
    named_routing_weights_to_internal,
    parse_named_guidance_response,
    render_prompt,
    masked_routing_exploration_distribution,
    routing_weights_to_named,
)
from routing_agents import ControlledDDQN, RandomRoutingController


class ConstantNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values.unsqueeze(0).expand(inputs.shape[0], -1)


def _routing_state(sender, effective_mask):
    state = torch.zeros((1, 42), dtype=torch.float32)
    state[0, sender] = 1.0
    state[0, 10:13] = torch.tensor(effective_mask, dtype=torch.float32)
    return state


def _neutral_named(weight=1.0):
    internal = np.full((16, 17), float(weight), dtype=np.float64)
    internal[np.arange(16), np.arange(16)] = 0.0
    return routing_weights_to_named(internal)


def test_forwarding_mask_excludes_wait_and_forces_wait_only_when_needed():
    np.testing.assert_array_equal(
        forwarding_or_forced_wait_mask([True, True, False], 0),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        forwarding_or_forced_wait_mask([False, False, False], 1),
        [False, True, False],
    )


def test_safe_ddqn_greedy_epsilon_and_llm_epsilon_never_voluntarily_wait():
    agent = DDQN(state_dim=42, action_dim=3, hidden_dim=4, master_seed=7)
    agent.q_network = ConstantNetwork([100.0, 5.0, 4.0])
    agent.cost_network = ConstantNetwork([0.0, 0.0, 0.0])
    state = _routing_state(0, [True, True, True])[0].numpy()
    assert agent.select_action(
        state, 0, mask=[True, True, True], epsilon=0.0, logits_noise_std=0.0
    ) == 1
    for _ in range(40):
        assert agent.select_action(
            state, 0, mask=[True, True, True], epsilon=1.0, logits_noise_std=0.0
        ) in {1, 2}
    distribution, used_fallback = masked_routing_exploration_distribution(
        [1_000.0, 0.0, 0.0], [False, True, True], prior_weight=1.0
    )
    assert used_fallback is True
    np.testing.assert_allclose(distribution, [0.0, 0.5, 0.5])
    for _ in range(40):
        assert agent.select_action(
            state,
            0,
            mask=[True, True, True],
            epsilon=1.0,
            logits_noise_std=0.0,
            exploration_prior=[1_000.0, 0.0, 0.0],
            exploration_prior_weight=1.0,
        ) in {1, 2}


@pytest.mark.parametrize("epsilon", [1.0, 0.0])
def test_forced_wait_skips_epsilon_prior_sampling_and_fallback(
    monkeypatch, epsilon
):
    agent = DDQN(state_dim=42, action_dim=3, hidden_dim=4, master_seed=17)
    agent.q_network = ConstantNetwork([100.0, 5.0, 4.0])
    agent.cost_network = ConstantNetwork([0.0, 0.0, 0.0])
    state = _routing_state(0, [True, False, False])[0].numpy()
    sampling_calls = []
    callback_calls = []

    def forbidden_sampling(*args, **kwargs):
        sampling_calls.append((args, kwargs))
        raise AssertionError("forced WAIT must not normalize or sample LLM weights")

    monkeypatch.setattr(
        ddqn_module,
        "masked_routing_exploration_distribution",
        forbidden_sampling,
    )
    action, detail = agent.select_action(
        state,
        0,
        mask=[True, False, False],
        epsilon=epsilon,
        logits_noise_std=0.0,
        exploration_prior=[0.0, 1.0, 1.0],
        exploration_prior_weight=1.0,
        zero_prior_mass_callback=lambda: callback_calls.append(True),
        return_diagnostics=True,
    )

    assert action == 0
    assert detail["decision_branch"] == "forced_wait_no_legal_forwarding_action"
    assert detail["zero_legal_prior_mass_fallback"] is False
    assert detail["exploration_distribution"] is None
    assert detail["legal_mask"] == [True, False, False]
    assert len(detail["q_r"]) == len(detail["q_c"]) == len(detail["q_safe"]) == 3
    assert sampling_calls == []
    assert callback_calls == []


def test_zero_legal_forwarding_weight_falls_back_only_over_forwarding_actions(
    monkeypatch,
):
    agent = DDQN(state_dim=42, action_dim=3, hidden_dim=4, master_seed=23)
    agent.q_network = ConstantNetwork([100.0, 5.0, 4.0])
    agent.cost_network = ConstantNetwork([0.0, 0.0, 0.0])
    state = _routing_state(0, [True, True, True])[0].numpy()
    original = ddqn_module.masked_routing_exploration_distribution
    captured = []
    callback_calls = []

    def capture_distribution(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(
        ddqn_module,
        "masked_routing_exploration_distribution",
        capture_distribution,
    )
    action, detail = agent.select_action(
        state,
        0,
        mask=[True, True, True],
        epsilon=1.0,
        logits_noise_std=0.0,
        exploration_prior=[1.0, 0.0, 0.0],
        exploration_prior_weight=1.0,
        zero_prior_mass_callback=lambda: callback_calls.append(True),
        return_diagnostics=True,
    )

    assert action in {1, 2}
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0][0], [0.0, 0.5, 0.5])
    assert captured[0][1] is True
    assert detail["zero_legal_prior_mass_fallback"] is True
    assert callback_calls == [True]


def test_valid_legal_forwarding_weights_preserve_weighted_sampling(monkeypatch):
    agent = DDQN(state_dim=42, action_dim=3, hidden_dim=4, master_seed=29)
    agent.q_network = ConstantNetwork([100.0, 5.0, 4.0])
    agent.cost_network = ConstantNetwork([0.0, 0.0, 0.0])
    state = _routing_state(0, [True, True, True])[0].numpy()
    original = ddqn_module.masked_routing_exploration_distribution
    sampling_calls = []

    def capture_distribution(*args, **kwargs):
        sampling_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ddqn_module,
        "masked_routing_exploration_distribution",
        capture_distribution,
    )
    action, detail = agent.select_action(
        state,
        0,
        mask=[True, True, True],
        epsilon=1.0,
        logits_noise_std=0.0,
        exploration_prior=[0.0, 0.0, 1.0],
        exploration_prior_weight=1.0,
        return_diagnostics=True,
    )

    assert action == 2
    assert len(sampling_calls) == 1
    assert detail["decision_branch"] == "epsilon_exploration"
    assert detail["zero_legal_prior_mass_fallback"] is False
    np.testing.assert_allclose(detail["exploration_distribution"], [0.0, 0.0, 1.0])


def test_outer_routing_selection_preserves_forced_wait_diagnostics():
    agent = DDQN(state_dim=42, action_dim=3, hidden_dim=4, master_seed=31)
    agent.q_network = ConstantNetwork([100.0, 5.0, 4.0])
    agent.cost_network = ConstantNetwork([0.0, 0.0, 0.0])
    callback_calls = []
    actions, details = _select_routing_actions(
        agent,
        {0: _routing_state(0, [True, False, False])[0].numpy()},
        {0: np.asarray([True, False, False])},
        epsilon=1.0,
        exploration_prior=np.asarray([[0.0, 1.0, 1.0]]),
        exploration_guidance_kind="routing_weights",
        exploration_prior_weight=1.0,
        zero_prior_mass_callback=lambda: callback_calls.append(True),
        return_diagnostics=True,
    )

    assert actions == {0: 0}
    assert set(details) == {0}
    detail = details[0]
    assert detail["decision_branch"] == "forced_wait_no_legal_forwarding_action"
    assert detail["zero_legal_prior_mass_fallback"] is False
    assert detail["routing_weights_applied"] is False
    assert detail["zero_legal_routing_weight_after_mask"] is False
    assert detail["routing_weight_uniform_fallback_used"] is False
    assert detail["routing_weight_uniform_fallback_reason"] is None
    assert callback_calls == []


def test_controlled_ddqn_random_and_bootstrap_share_forwarding_policy():
    dqn = ControlledDDQN(state_dim=42, action_dim=3, hidden_dim=4, master_seed=9)
    dqn.q_network = ConstantNetwork([100.0, 5.0, 4.0])
    dqn.target_q_network = ConstantNetwork([900.0, 20.0, 30.0])
    state = _routing_state(0, [True, True, False])
    assert dqn.select_action(
        state[0].numpy(), 0, mask=[True, True, False], epsilon=0.0
    ) == 1
    target = dqn._standard_targets(
        state, torch.tensor([[2.0]]), torch.tensor([[1.0]])
    )
    assert torch.isfinite(target).all()
    assert target.item() == pytest.approx(2.0 + dqn.gamma * 20.0)

    random_agent = RandomRoutingController(rng=np.random.default_rng(4))
    for _ in range(20):
        assert random_agent.select_action(
            state[0].numpy(), 0, mask=[True, True, True]
        ) in {1, 2}
    assert random_agent.select_action(
        state[0].numpy(), 0, mask=[True, False, False]
    ) == 0


def test_safe_targets_select_same_legal_action_for_reward_and_cost():
    agent = DDQN.__new__(DDQN)
    agent.action_dim = 3
    agent.gamma = 0.9
    agent.lambda_cost = 0.0
    agent.q_network = ConstantNetwork([100.0, 7.0, 6.0])
    agent.cost_network = ConstantNetwork([0.0, 0.0, 0.0])
    agent.target_q_network = ConstantNetwork([1_000.0, 20.0, 30.0])
    agent.target_cost_network = ConstantNetwork([2_000.0, 40.0, 60.0])
    q, c, actions = agent._safe_targets(
        _routing_state(0, [True, True, True]),
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
    )
    assert actions.item() == 1
    assert q.item() == pytest.approx(19.0)
    assert c.item() == pytest.approx(38.0)
    assert torch.isfinite(q).all() and torch.isfinite(c).all()


def test_named_routing_contract_maps_by_name_and_rejects_bad_objects():
    named = _neutral_named()
    named["UAV_3"]["GS"] = 7.5
    named["UAV_3"]["UAV_0"] = 2.5
    internal = named_routing_weights_to_internal(named)
    assert internal.shape == (16, 17)
    assert internal[3, 3] == 0.0
    assert internal[3, 16] == 7.5
    assert internal[3, 0] == 2.5
    assert set(named["UAV_3"]) == {
        *(f"UAV_{index}" for index in range(16) if index != 3), "GS"
    }

    missing = json.loads(json.dumps(named))
    del missing["UAV_3"]["GS"]
    with pytest.raises(ExplorerValidationError, match="missing=.*GS"):
        named_routing_weights_to_internal(missing)
    boolean = json.loads(json.dumps(named))
    boolean["UAV_3"]["GS"] = True
    with pytest.raises(ExplorerValidationError, match="JSON number"):
        named_routing_weights_to_internal(boolean)

    duplicate = json.dumps(
        {"td3_bias": [[0.0, 0.0, 0.0]] * 16, "routing_weights": named}
    ).replace('"GS": 1.0', '"GS": 1.0, "GS": 2.0', 1)
    _, _, errors = parse_named_guidance_response(duplicate, 0.2)
    assert "duplicate JSON object key" in errors["routing"]


def test_no_c10_schema_mock_templates_metadata_and_checkpoint_are_isolated():
    response_format = first_gen_numerical_response_format(
        LLM_EXPLORER_NO_C10_METHOD_ID
    )
    mock = DeterministicMockExplorerClient()
    response = mock.chat(
        model="offline",
        prompt="numerical",
        system_prompt="system",
        response_format=response_format,
    )
    td3, routing, errors = parse_named_guidance_response(
        response["content"], 0.0
    )
    assert not errors
    assert td3.shape == (16, 3)
    assert routing.shape == (16, 17)
    assert np.all(routing[np.arange(16), np.arange(16)] == 0.0)

    config = FirstGenLLMExplorerConfig(
        prompt_version=LLM_EXPLORER_NO_C10_PROMPT_VERSION
    )
    controller = LLMExplorerController(
        run_directory=None,
        run_identity={"method_id": LLM_EXPLORER_NO_C10_METHOD_ID},
        config=config,
        evaluation=True,
    )
    original = LLMExplorerController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer"},
        evaluation=True,
    )
    assert controller._templates() != original._templates()
    analysis = render_prompt(
        controller._templates()[1],
        {
            "PHASE_AND_EXPLORATION_SETTINGS": {},
            "EPISODE_OUTCOMES": {},
            "PREVIOUS_COMPLETE_GUIDANCE": {},
            "PREVIOUS_APPLICATION_STATUS": {},
            "CURRENT_OBSERVABLE_INITIAL_STATE": {},
            "RESOLVED_SETTINGS_AND_DATA_DICTIONARY": {},
            "MOVEMENT_POWER_REFERENCE_POINTS": {},
            "MOVEMENT_AND_SERVICE_TABLES": {},
            "ROUTING_TABLES": {},
            "REPRESENTATIVE_CASES": [],
        },
    )
    guidance = render_prompt(
        controller._templates()[3],
        {
            "ANALYSIS_STAGE_OUTPUT": "analysis",
            "CURRENT_NUMERICAL_SETTINGS_AND_DEFINITIONS": {},
            "CURRENT_OBSERVABLE_INITIAL_STATE": {},
            "PREVIOUS_COMPLETE_TD3_BIAS": [],
            "PREVIOUS_COMPLETE_ROUTING_WEIGHTS": {},
            "PREVIOUS_COMPLETE_ROUTING_PRIOR": {},
            "PREVIOUS_APPLICATION_STATUS": {},
            "VALIDATION_FEEDBACK_OR_EMPTY": "",
        },
    )
    assert "{{" not in analysis + guidance + controller._templates()[0] + controller._templates()[2]
    metadata = explorer_metadata(config, LLM_EXPLORER_NO_C10_METHOD_ID)
    assert metadata["numerical_output_protocol"] == LLM_EXPLORER_NO_C10_NUMERICAL_OUTPUT_PROTOCOL
    assert metadata["routing_output_format"] == "named_other_15_uavs_plus_gs"

    state = controller.state_dict()
    restored = LLMExplorerController(
        run_directory=None,
        run_identity={"method_id": LLM_EXPLORER_NO_C10_METHOD_ID},
        config=config,
        evaluation=True,
    )
    restored.load_state_dict(state)
    incompatible = json.loads(json.dumps(state))
    incompatible["config"]["prompt_version"] = (
        "uav-hrl-llm-explorer-prompts-v11-complete-ee-no-c10-variant-v1"
    )
    with pytest.raises(ExplorerValidationError, match="incompatible"):
        restored.load_state_dict(incompatible)
