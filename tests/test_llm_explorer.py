import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from DDQN import DDQN
from experiment_config import METHOD_REGISTRY, MethodSpec
from llm_explorer import (
    LLMExplorerConfig,
    LLMExplorerController,
    build_observable_initial_state,
    compact_feedback_for_llm,
    compact_routing_record,
    llm_explorer_path_preflight,
    masked_routing_exploration_distribution,
    parse_guidance_response,
    prompt_token_budget,
    render_prompt,
    stratified_routing_samples,
)
from observation_strategy import routing_state_feature_names
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
        if isinstance(response, dict):
            return response
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


def _full_size_feedback():
    movement = []
    for interval in range(60):
        movement.append(
            {
                "episode_index": 166,
                "interval_index": interval,
                "time_seconds": float(interval),
                "action_source": "actor_plus_llm_bias_plus_gaussian_exploration",
                "behavior_noise_std": 0.5 - interval * 1e-5,
                "known_roi_ids_at_sample_time": [1, 3],
                "uavs": [
                    {
                        "uav_id": uid,
                        "position_m": [uid * 10.12345, interval * 2.5, 100.0],
                        "remaining_energy_j": 10_000.0 - interval * 3.14159 - uid,
                        "task_types": ["FOV" if uid % 3 == 0 else "COM"],
                        "observable_targets": [
                            {
                                "kind": "roi" if uid % 3 == 0 else "sr",
                                "id": uid % 4,
                                "relative_position_m": [12.3456, -2.3456, -100.0],
                            }
                        ],
                        "movement_mask": uid % 4 != 0,
                        "actor_normalized_action": [0.1234567, -0.7654321, 0.0],
                        "applied_llm_bias": [0.05, -0.05, 0.0],
                        "executed_normalized_action": [0.1734567, -0.8154321, 0.01],
                    }
                    for uid in range(16)
                ],
            }
        )
    names = routing_state_feature_names()
    routing = []
    for index in range(100):
        sender = index % 16
        state = np.zeros(len(names), dtype=float)
        state[names.index(f"uav_id_one_hot[{sender}]")] = 1.0
        state[names.index("energy")] = 0.8
        state[names.index("backlog")] = 0.2
        state[names.index("assigned_task_com")] = 1.0
        state[names.index("assigned_source_uav")] = 1.0
        state[names.index("position_x")] = 100.0 + sender
        state[names.index("position_y")] = 200.0 + index
        state[names.index("position_z")] = 0.4
        state[names.index("distance_to_ground_station")] = 0.6
        state[names.index("eta_to_ground_station")] = 0.5
        state[names.index("dinkelbach_lambda")] = 0.3
        state[names.index("hol_assigned_task_com")] = 1.0
        state[names.index("hol_deadline_remaining")] = 0.75
        state[names.index("hol_packet_fraction_remaining")] = 0.9
        legal = [action in {sender, (sender + 1) % 16, 16} for action in range(17)]
        for action in range(17):
            state[names.index(f"effective_action_mask[{action}]")] = float(legal[action])
            state[names.index(f"link_delay[{action}]")] = action / 20.0
            state[names.index(f"link_capacity[{action}]")] = (17 - action) / 17.0
            state[names.index(f"next_hop_backlog[{action}]")] = action / 17.0
            state[names.index(f"gs_progress_norm[{action}]")] = (action - sender) / 16.0
            state[names.index(f"next_hop_assigned_fov[{action}]")] = float(action % 5 == 0)
        routing.append(
            {
                "decision_index": index,
                "time_seconds": index * 0.25,
                "sender_uav_id": sender,
                "selected_action": 16 if index % 2 else sender,
                "compact_routing_state": state.tolist(),
                "legal_action_mask": [int(value) for value in legal],
                "reference_capacity_bps_by_legal_action": {
                    str(action): 1_000_000.0 + action * 1000
                    for action, is_legal in enumerate(legal)
                    if is_legal
                },
                "sender_total_backlog_bits": 1_000_000.0 + index,
                "hol_packet_id": 10_000 + index,
                "hol_task_type": "COM",
                "hol_remaining_bits": 20_000.0,
                "hol_remaining_deadline_seconds": 1.25,
                "epsilon": 0.5,
                "decision_branch": "epsilon_exploration",
                "zero_legal_prior_mass_fallback": False,
            }
        )
    return {
        "metrics": {
            "energy_efficiency_bit_per_j": 1234.56789,
            "td3_original_cumulative_reward": 2.5,
            "safe_ddqn_original_cumulative_reward": 3.5,
        },
        "movement_decisions": movement,
        "routing_samples": routing,
        "full_episode_statistics": {
            "movement_decision_count": 60,
            "routing_sample_count": 100,
        },
        "applied_guidance": {
            "enabled": True,
            "td3_bias": [[0.05, -0.05, 0.0] for _ in range(16)],
            "routing_prior": [[1 / 17] * 17 for _ in range(16)],
            "td3_fallback": False,
            "routing_fallback": False,
            "reused": False,
        },
    }


def _begin(
    controller,
    transitions=10_000,
    episode_index=167,
    current_state=None,
    environment_constants=None,
):
    return controller.begin_episode(
        episode_index=episode_index,
        total_joint_transitions=transitions,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state=(
            {"known_rois": [], "uavs": []}
            if current_state is None
            else current_state
        ),
        environment_constants=(
            {"episode_seconds": 60}
            if environment_constants is None
            else environment_constants
        ),
    )


def test_registry_appends_explorer_without_using_legacy_llm_flag():
    assert "td3_dinkelbach_llm_explorer" in METHOD_REGISTRY
    assert tuple(METHOD_REGISTRY)[-1] == "td3_dinkelbach_llm_explorer_v3"
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

    replay_client = SequenceClient([])
    replay = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "timeout"},
        client=replay_client,
    )
    replay.previous_episode_feedback = _previous_feedback()
    restored = _begin(replay)
    assert restored["reused"]
    assert restored["td3_fallback"] and restored["routing_fallback"]
    assert replay_client.calls == []
    assert replay.statistics["fallback_episodes_total"] == 1


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


def test_real_size_compact_prompts_preserve_samples_and_fit_all_attempts(tmp_path):
    previous = _full_size_feedback()
    invalid_routing = [[1 / 17] * 17 for _ in range(16)]
    invalid_routing[0][0] = -0.1
    analysis = "\n".join(
        ["Movement and routing evidence remains observable and bounded."] * 300
    )
    client = SequenceClient(
        [analysis, _guidance(0.1, invalid_routing), _guidance(0.1)]
    )
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "full-size"},
        client=client,
    )
    controller.previous_episode_feedback = previous
    current_state = {
        "observation_time": "episode_start_before_first_action",
        "known_rois": [
            {"roi_id": 1, "position_m": [300.0, 400.0, 0.0], "radius_m": 40.0}
        ],
        "observable_sr_teams": [
            {"sr_id": 1, "position_m": [310.0, 390.0, 0.0], "assigned_roi_id": 1}
        ],
        "uavs": [
            {
                "uav_id": uid,
                "position_m": [uid * 25.0, 100.0, 100.0],
                "remaining_energy_j": 10_000.0,
                "battery_percent": 100.0,
                "tasks": [
                    {
                        "task_type": "COM" if uid % 2 else "FOV",
                        "target_obj_id": 1,
                        "target_status": "assigned_observable",
                    }
                ],
                "queue_backlog_bits": 0.0,
            }
            for uid in range(16)
        ],
        "ground_station": {"action_index": 16, "position_m": [0.0, 0.0, 0.0]},
    }
    result = _begin(
        controller,
        current_state=current_state,
        environment_constants={
            "episode_seconds": 60,
            "routing_slot_seconds": 0.25,
            "operating_area_m": {"width": 1000.0, "height": 1000.0},
            "validity": {
                "unknown": "not observed",
                "zero": "observed numeric zero",
                "empty_queue": "observed zero backlog",
                "unassigned": "no target",
            },
        },
    )
    assert not result["td3_fallback"] and not result["routing_fallback"]
    assert len(client.calls) == 3

    compact = compact_feedback_for_llm(previous)
    assert len(compact["movement"]["decision_rows"]) == 60
    assert len(compact["movement"]["uav_rows"]) == 60 * 16
    assert len(compact["routing"]["rows"]) == 100
    assert [row[1] for row in compact["movement"]["uav_rows"][:16]] == list(range(16))
    assert compact["movement"]["uav_rows"][0][9] is not None
    assert compact["movement"]["uav_rows"][0][10] is not None
    assert compact["movement"]["uav_rows"][0][11] is not None

    for call in client.calls:
        budget = prompt_token_budget(
            call["system_prompt"],
            call["prompt"],
            reserved_output_tokens=call["max_output_tokens"],
        )
        assert budget["fits"], budget
    budget_files = sorted(
        (tmp_path / "llm_explorer" / "episode_000168").glob(
            "*/attempt_*/token_budget.json"
        )
    )
    assert len(budget_files) == 3
    assert all(json.loads(path.read_text())["fits"] for path in budget_files)


def test_local_context_overflow_never_calls_api_or_retries_same_prompt(tmp_path):
    client = SequenceClient([])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "overflow"},
        client=client,
    )
    previous = _previous_feedback()
    previous["metrics"]["oversized_observable_note"] = "超限0123456789" * 40_000
    controller.previous_episode_feedback = previous
    result = _begin(controller)
    assert result["td3_fallback"] and result["routing_fallback"]
    assert client.calls == []
    assert controller.statistics["requests_total"] == 0
    assert controller.statistics["failure_counts"]["local_preflight"] == 1


def test_td3_episode_start_bound_survives_step_sigma_decay():
    model = TD3(state_dim=2, action_dim=3, max_action=1.0, master_seed=2)
    bias = np.array([0.5, -0.5, np.float32(0.5)], dtype=np.float32)
    model.select_action(
        np.zeros(2, dtype=np.float32),
        noise_std=0.49999,
        mean_bias=bias,
        mean_bias_bound=0.5,
    )
    with pytest.raises(ValueError, match="episode-start bound"):
        model.select_action(
            np.zeros(2, dtype=np.float32),
            noise_std=0.49,
            mean_bias=np.array([0.5002, 0.0, 0.0]),
            mean_bias_bound=0.5,
        )


def test_truncated_analysis_is_failed_with_usage_and_never_cached(tmp_path):
    truncated = {
        "content": "partial analysis",
        "actual_model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        "finish_reason": "length",
        "transport_completed": True,
        "done_received": True,
        "terminal_chunk_received": True,
    }
    client = SequenceClient([truncated, truncated])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "truncated"},
        client=client,
    )
    controller.previous_episode_feedback = _previous_feedback()
    result = _begin(controller)
    assert result["td3_fallback"] and result["routing_fallback"]
    assert controller.statistics["requests_total"] == 2
    assert controller.statistics["failed_attempts_total"] == 2
    assert controller.statistics["failure_counts"]["response_incomplete"] == 2
    assert controller.statistics["token_usage"]["total_tokens"] == 48
    assert not (
        tmp_path
        / "llm_explorer"
        / "episode_000168"
        / "analysis"
        / "completed.json"
    ).exists()


def test_response_artifact_failure_does_not_double_count_requests(tmp_path, monkeypatch):
    import llm_explorer as explorer_module

    original = explorer_module._atomic_json

    def fail_metadata(path, value):
        if path.name == "response_metadata.json":
            raise OSError("injected metadata write failure")
        return original(path, value)

    monkeypatch.setattr(explorer_module, "_atomic_json", fail_metadata)
    client = SequenceClient(["analysis", _guidance(0.1)])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "write-failure"},
        client=client,
    )
    controller.previous_episode_feedback = _previous_feedback()
    result = _begin(controller)
    assert not result["td3_fallback"] and not result["routing_fallback"]
    assert len(client.calls) == 2
    assert controller.statistics["requests_total"] == 2
    assert controller.statistics["failed_attempts_total"] == 0
    assert controller.statistics["failure_counts"]["artifact_write"] == 2


def test_partial_and_fallback_episode_result_replays_without_calls_or_double_count(tmp_path):
    bad_routing = [[1 / 17] * 17 for _ in range(16)]
    bad_routing[0][0] = -0.1
    client = SequenceClient(
        [
            "analysis",
            _guidance(0.1, bad_routing),
            _guidance(9.0, bad_routing),
        ]
    )
    first = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "partial-fallback"},
        client=client,
    )
    first.previous_episode_feedback = _previous_feedback()
    original = _begin(first)
    assert not original["td3_fallback"] and original["routing_fallback"]
    assert first.statistics["fallback_episodes_total"] == 1

    replay = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "partial-fallback"},
        client=SequenceClient([]),
    )
    replay.previous_episode_feedback = _previous_feedback()
    restored = _begin(replay)
    assert restored["reused"]
    assert not restored["td3_fallback"] and restored["routing_fallback"]
    assert replay.statistics["fallback_episodes_total"] == 1
    replay._record_fallback_episode(167, td3=False, routing=True)
    assert replay.statistics["fallback_episodes_total"] == 1


def test_two_episode_cache_replay_keeps_next_new_request_seeds(tmp_path):
    initial = _previous_feedback()
    original_client = SequenceClient(
        ["analysis-1", _guidance(0.05), "analysis-2", _guidance(0.04)]
    )
    original = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "seed-chain"},
        client=original_client,
    )
    original.previous_episode_feedback = initial
    state_before_first = original.state_dict()
    _begin(original, episode_index=167)
    original.complete_episode(episode_index=167, metrics={"episode": 167})
    _begin(original, episode_index=168)
    second_seeds = [call["seed"] for call in original_client.calls[2:]]

    shutil.rmtree(tmp_path / "llm_explorer" / "episode_000169")
    replay_client = SequenceClient(["analysis-2", _guidance(0.04)])
    replay = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "seed-chain"},
        client=replay_client,
    )
    replay.load_state_dict(state_before_first)
    replay.previous_episode_feedback = initial
    replayed_first = _begin(replay, episode_index=167)
    assert replayed_first["reused"]
    replay.complete_episode(episode_index=167, metrics={"episode": 167})
    _begin(replay, episode_index=168)
    assert [call["seed"] for call in replay_client.calls] == second_seeds


def test_routing_compact_schema_uses_authoritative_layout():
    names = routing_state_feature_names()
    state = np.arange(len(names), dtype=float) / 1000.0
    record = {
        "time_seconds": 1.25,
        "sender_uav_id": 2,
        "selected_action": 16,
        "compact_routing_state": state.tolist(),
        "legal_action_mask": [1 if i in (2, 4, 16) else 0 for i in range(17)],
        "reference_capacity_bps_by_legal_action": {"2": 0.0, "4": 2e6, "16": 3e6},
        "sender_total_backlog_bits": 0.0,
        "hol_packet_id": 3,
        "hol_task_type": "COM",
        "hol_remaining_bits": 0.0,
        "hol_remaining_deadline_seconds": 0.0,
        "epsilon": 0.5,
        "decision_branch": "safe_q_greedy",
    }
    compact = compact_routing_record(record)
    features = compact["legal_action_features"]
    action_four_index = features[0].index(4)
    assert features[1][action_four_index] == round(
        state[names.index("link_capacity[4]")], 3
    )
    assert compact["decision"][5] == 0.0
    assert compact["decision"][8] == 0.0
    assert compact["decision"][9] == 0.0
    assert compact["legal_action_indices"] == [2, 4, 16]


def test_windows_path_preflight_includes_stream_and_atomic_temp_names(tmp_path):
    report = llm_explorer_path_preflight(tmp_path, maximum_episode=4000)
    paths = {entry["relative_path"] for entry in report["paths"]}
    assert any("response_content.partial.txt" in path for path in paths)
    assert any(".response_metadata.json.tmp" in path for path in paths)
    assert any("token_budget.json" in path for path in paths)
    unicode_report = llm_explorer_path_preflight(
        tmp_path / "中文 output root", maximum_episode=4000
    )
    assert "中文 output root" in unicode_report["longest"]["absolute_path"]
    with pytest.raises(ValueError, match="shorter --output-root"):
        llm_explorer_path_preflight(
            tmp_path / (("深 層 directory ") * 12),
            maximum_episode=4000,
            windows_limit=180,
        )
