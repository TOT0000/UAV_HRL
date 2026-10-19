import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from DDQN import DDQN
from experiment_config import METHOD_REGISTRY, MethodSpec
from HRL_task_aware import TrainingConfig, _select_routing_actions, train
from llm_explorer import (
    FIRST_GEN_ROUTING_PRIOR_WEIGHT,
    LLM_EXPLORER_CACHE_VERSION,
    LLM_EXPLORER_PROMPT_VERSION,
    LLM_EXPLORER_SUMMARY_VERSION,
    LLM_EXPLORER_VERSION,
    LLMExplorerConfig,
    LLMExplorerController,
    DeterministicMockExplorerClient,
    assemble_first_gen_feedback,
    build_observable_initial_state,
    compact_feedback_for_llm,
    compact_routing_record,
    environment_constants_and_definitions,
    environment_constants_and_definitions_first_gen,
    explorer_metadata,
    first_gen_movement_summary,
    first_gen_present_for_llm,
    first_gen_resolved_settings_and_dictionary,
    first_gen_routing_summary,
    llm_explorer_path_preflight,
    masked_routing_exploration_distribution,
    parse_guidance_response,
    prompt_token_budget,
    render_prompt,
    stratified_routing_samples,
)
from observation_strategy import routing_state_feature_names
from llm_explorer_td3_only import (
    LLMExplorerTD3OnlyController,
    environment_definitions_td3_only,
)
from llm_explorer_v3 import (
    LLMExplorerV3Controller,
    environment_definitions_v3,
)
from llm_state_rule_explorer import (
    LLMStateRuleController,
    environment_definitions_state_rule,
)
from run_experiment import main as run_experiment_main
from scenario_manifest import generate_manifest
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
        "schema_version": LLM_EXPLORER_VERSION,
        "summary_version": LLM_EXPLORER_SUMMARY_VERSION,
        "episode_outcomes": {
            "energy_efficiency_bit_per_j": 1.0,
            "td3_original_cumulative_reward": 2.0,
            "safe_ddqn_original_cumulative_reward": 3.0,
        },
        "previous_complete_guidance": {
            "td3_bias": {
                "status": "unknown_no_applied_guidance",
                "reason": "warmup",
                "value": None,
            },
            "routing_prior": {
                "status": "unknown_no_applied_guidance",
                "reason": "warmup",
                "value": None,
            },
        },
        "previous_application_status": {
            "guidance_enabled": False,
            "td3_fallback": True,
            "routing_fallback": True,
        },
        "uav_core_summary": {"by_uav": {}},
        "routing_core_summary": {"by_sender": {}},
        "actual_slot_service_summary": {"slot_count": 0},
        "representative_cases": [],
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
                        "pre_projection_normalized_action": [0.1734567, -0.8154321, 0.01],
                        "policy_projected_normalized_action": [0.1734567, -0.8154321, 0.01],
                        "td3_guidance_active": True,
                        "actor_pre_projection_llm_bias": [0.05, -0.05, 0.0],
                        "td3_controlled_applied_llm_bias": (
                            [0.05, -0.05, 0.0] if uid % 4 != 0 else None
                        ),
                        "applied_llm_bias": [0.05, -0.05, 0.0],
                        "executed_normalized_action": [0.1734567, -0.8154321, 0.01],
                        "movement_energy_j": 100.12345 + uid * 0.25,
                        "service_observation": {
                            "snapshot_available": True,
                            "uav_queue": {
                                "valid": True,
                                "empty": False,
                                "backlog_bits": 10_000.123 + uid * 100,
                                "hol_present": True,
                                "hol_remaining_deadline_s": 0.009876,
                            },
                            "fov": {
                                "pair_valid": uid % 3 == 0,
                                "capture_valid": uid % 3 == 0,
                                "coverage_ratio": 0.812345,
                                "image_quantity": 1.234567,
                                "c9_margin_m": 12.34567,
                                "c10_margin_m": -1.23456,
                            },
                            "com": {
                                "target_valid": uid % 3 != 0,
                                "sr_queue_valid": uid % 3 != 0,
                                "sr_queue_empty": False,
                                "sr_backlog_bits": 20_000.456 + uid * 100,
                                "sr_hol_remaining_deadline_s": 0.008765,
                                "s2u_link_valid": uid % 3 != 0,
                                "s2u_in_range": uid % 2 == 0,
                                "s2u_reference_capacity_mbps": 2.345678,
                            },
                        },
                    }
                    for uid in range(16)
                ],
            }
        )
    names = routing_state_feature_names()
    routing = []
    for index in range(240 * 16):
        sender = index % 16
        slot = index // 16
        state = np.zeros(len(names), dtype=float)
        state[names.index(f"uav_id_one_hot[{sender}]")] = 1.0
        state[names.index("energy")] = 0.8
        state[names.index("backlog")] = 0.2
        state[names.index("assigned_task_com")] = 1.0
        state[names.index("assigned_source_uav")] = 1.0
        state[names.index("position_x")] = 100.0 + sender
        state[names.index("position_y")] = 200.0 + slot
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
                "time_seconds": slot * 0.25,
                "sender_uav_id": sender,
                "selected_action": 16 if index % 2 else sender,
                "compact_routing_state": state.tolist(),
                "legal_action_mask": [int(value) for value in legal],
                "current_expected_capacity_bps_by_legal_action": {
                    str(action): 1_000_000.0 + action * 1000
                    for action, is_legal in enumerate(legal)
                    if is_legal
                },
                "maximum_reference_normalizer_bps_by_legal_action": {
                    str(action): 9_000_000.0 + action * 1000
                    for action, is_legal in enumerate(legal)
                    if is_legal
                },
                "capacity_context_by_legal_action": {
                    str(action): {
                        "link_type": (
                            "WAIT" if action == sender else "U2G" if action == 16 else "U2U"
                        ),
                        "reference_bandwidth_hz": (
                            0.0 if action == sender else 10_000_000.0
                        ),
                        "distance_3d_m": 0.0 if action == sender else 100.0 + action,
                    }
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
    return assemble_first_gen_feedback(
        metrics={
            "energy_efficiency_bit_per_j": 1234.56789,
            "td3_original_cumulative_reward": 2.5,
            "safe_ddqn_original_cumulative_reward": 3.5,
        },
        movement_records=movement,
        routing_records=routing,
        slot_service_records=[
            {
                "time_seconds": slot * 0.25,
                "slot_seconds": 0.25,
                "actual_hol_wait_seconds_available": True,
                "actual_hol_wait_seconds_by_sender": {
                    str(sender): (0.01 if sender == slot % 16 else 0.0)
                    for sender in range(16)
                },
                "packet_decision_uses": [
                    {
                        "packet_id": 50_000 + slot,
                        "sender_uav_id": slot % 16,
                        "selected_action": (slot + 1) % 16,
                    }
                ],
                "transmitted_bits_by_link": {
                    f"U2U:{slot % 16}->{(slot + 1) % 16}": 128.5 + slot
                },
                "active_links": [
                    {
                        "link_type": "U2U",
                        "sender_id": slot % 16,
                        "receiver_id": (slot + 1) % 16,
                        "transmitted_bits": 128.5 + slot,
                        "active_seconds": 0.001234,
                        "capacity_bits_per_second": 2_345_678.9,
                    }
                ],
                "completed_hops": (
                    [
                        {
                            "link_type": "U2U",
                            "sender_id": slot % 16,
                            "receiver_id": (slot + 1) % 16,
                            "packet_id": 50_000 + slot,
                        }
                    ]
                    if slot % 2 == 0
                    else []
                ),
                "global_routing_reward": 0.0,
                "shared_reward_row_sum": 0.0,
            }
            for slot in range(240)
        ],
        applied_guidance={
            "enabled": True,
            "td3_bias": [
                [0.001 * (uid + 1), -0.0007 * (uid + 1), 0.0003 * uid]
                for uid in range(16)
            ],
            "routing_prior": [
                (
                    lambda weights: [value / sum(weights) for value in weights]
                )([float(1 + ((action + uid) % 17)) for action in range(17)])
                for uid in range(16)
            ],
            "td3_fallback": False,
            "routing_fallback": False,
            "reused": False,
        },
    )


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
    assert tuple(METHOD_REGISTRY)[-4:] == (
        "td3_dinkelbach_llm_explorer",
        "td3_dinkelbach_llm_explorer_v3",
        "td3_dinkelbach_llm_explorer_td3_only",
        "td3_dinkelbach_llm_explorer_state_rule",
    )
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


def test_first_generation_three_decimal_presentation_is_detached_and_typed():
    array = np.array([1.23456, -0.000012345, 0.00049], dtype=np.float64)
    source = {
        "array": array,
        "count": 7,
        "flag": True,
        "unknown": None,
        "lambda_EE_Mbit_per_J_used": 0.000012345,
    }
    presented = first_gen_present_for_llm(source)

    assert presented["array"] == [1.235, 0.0, 0.0]
    assert presented["count"] == 7 and isinstance(presented["count"], int)
    assert presented["flag"] is True
    assert presented["unknown"] is None
    assert presented["lambda_EE_Mbit_per_J_used"] == 1.23e-05
    np.testing.assert_array_equal(
        array, np.array([1.23456, -0.000012345, 0.00049])
    )


def test_representative_case_durations_use_explicit_ms_fields_without_mutation():
    previous = _previous_feedback()
    previous["representative_cases"] = [
        {
            "kind": "short_service",
            "time_s": 12.34567,
            "active_seconds": 2.2838932352442498e-05,
            "vector": np.array([1.23456, 2.34567]),
        },
        {
            "kind": "longer_service",
            "time_s": 13.0,
            "active_seconds": 0.001298719,
        },
        {
            "kind": "nested_hol",
            "time_s": 14.0,
            "queue_and_sensing": {
                "uav_queue": {"hol_remaining_deadline_s": 2.2838932352442498e-05},
                "com": {"sr_hol_remaining_deadline_s": 0.001298719},
            },
        },
        {"kind": "true_zero", "time_s": 15.0, "active_seconds": 0.0},
        {"kind": "tiny_nonzero", "time_s": 16.0, "active_seconds": 1e-10},
        {"kind": "none", "time_s": 17.0, "active_seconds": None},
        {
            "kind": "unknown",
            "time_s": 18.0,
            "queue_and_sensing": {
                "com": {"sr_hol_remaining_deadline_s": "unknown"}
            },
        },
    ]
    source = copy.deepcopy(previous["representative_cases"])

    cases = compact_feedback_for_llm(previous)["representative_cases"]

    assert cases[0]["time_s"] == 12.346
    assert "time_ms" not in cases[0]
    assert cases[0]["active_service_ms"] == 0.023
    assert cases[1]["active_service_ms"] == 1.299
    nested = cases[2]["queue_and_sensing"]
    assert nested["uav_queue"]["hol_remaining_deadline_ms"] == 0.023
    assert nested["com"]["sr_hol_remaining_deadline_ms"] == 1.299
    assert cases[3]["active_service_ms"] == 0.0
    assert cases[4]["active_service_ms"] == 1e-7
    assert cases[5]["active_service_ms"] is None
    assert (
        cases[6]["queue_and_sensing"]["com"]["sr_hol_remaining_deadline_ms"]
        == "unknown"
    )
    assert all("active_seconds" not in case for case in cases)
    np.testing.assert_array_equal(
        previous["representative_cases"][0]["vector"], source[0]["vector"]
    )
    assert previous["representative_cases"][0]["active_seconds"] == source[0]["active_seconds"]


def test_first_gen_penalty_contract_and_capacity_normalizers_are_runtime_derived():
    from Channel_model import (
        reference_u2g_max_capacity_mbps,
        reference_u2u_max_capacity_mbps,
    )
    from experiment_config import TOTAL_COMMUNICATION_BANDWIDTH_HZ

    environment = environment_constants_and_definitions_first_gen(
        episode_seconds=60,
        routing_slot_seconds=0.25,
    )
    resolved = first_gen_resolved_settings_and_dictionary(environment)
    penalties = resolved["camera_and_penalties"]["penalty_contract"]

    assert penalties["C9"]["formula"] == (
        "P_C9 = w_C9 * mean_over_S_C9(1 - clip(b1*h/(d+epsilon),0,1))"
    )
    assert penalties["C10"]["formula"] == (
        "P_C10 = w_C10 * mean_over_S_C10(1 - clip(min(d_L,d_R)/(r+epsilon),0,1))"
    )
    assert penalties["COM"]["formula"] == (
        "P_COM = w_COM * mean_over_S_COM(1 - clip(R_com/(D_3D+epsilon),0,1))"
    )
    assert penalties["weights"] == {"w_C9": 1.0, "w_C10": 1.0, "w_COM": 1.0}
    assert penalties["normalization_epsilon"] == 1e-12
    assert penalties["C9"]["population"] == "currently assigned FOV pairs"
    assert "C9-valid" in penalties["C10"]["population"]
    assert penalties["COM"]["population"] == "all currently assigned COM UAV-SR pairs"
    assert "empty population contributes zero" in penalties["empty_population"]

    normalizers = resolved["data_dictionary"][
        "routing_state_normalization_constants"
    ]
    assert normalizers["semantics"] == (
        "Routing-state normalization constants; not current link capacities "
        "or realized service rates."
    )
    values = normalizers["configuration"]["by_link_type_bps"]
    assert values["U2U"] == pytest.approx(
        reference_u2u_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ) * 1e6
    )
    assert values["U2G"] == pytest.approx(
        reference_u2g_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ) * 1e6
    )


def test_rounded_history_is_reference_only_and_new_output_remains_strict():
    previous = _previous_feedback()
    source_prior = [[1.0 / 17.0] * 17 for _ in range(16)]
    source_bias = [[0.123456, -0.012345, 0.0001234] for _ in range(16)]
    previous["previous_complete_guidance"] = {
        "td3_bias": {
            "status": "known_applied_episode_guidance",
            "shape": [16, 3],
            "value": source_bias,
        },
        "routing_prior": {
            "status": "known_applied_episode_guidance",
            "shape": [16, 17],
            "value": source_prior,
        },
    }
    compact = compact_feedback_for_llm(previous)
    displayed_prior = compact["previous_complete_guidance"]["routing_prior"][
        "value"
    ]

    assert sum(displayed_prior[0]) == pytest.approx(1.003)
    assert source_prior[0][0] == pytest.approx(1.0 / 17.0)
    assert source_bias[0] == [0.123456, -0.012345, 0.0001234]
    td3, routing, errors = parse_guidance_response(_guidance(0.123456), 0.2)
    assert errors == {}
    assert td3 is not None and routing is not None
    assert routing[0].sum() == pytest.approx(1.0, abs=1e-12)


def test_other_explorer_environment_and_system_prompts_match_parent_baseline():
    kwargs = {
        "episode_seconds": 60,
        "routing_slot_seconds": 0.25,
        "environment_width_m": 1000.0,
        "environment_height_m": 1000.0,
        "task_deadlines_seconds": {"FOV": 0.015, "COM": 0.01},
    }
    expected = {
        "v3": (
            "9f362f3a2c3d7743f4cf954774e0ede67274721efd94d8dee79a0e3f92480a5c",
            "578cc4bc85ae3089ab97382e6dbfb2227c763559c72ba89985502d0f924904bb",
        ),
        "td3_only": (
            "271a70cfb9c17bb7edbc00535775c31b81af8ddef8e707e636e4f0beacbb735e",
            "5b711f05970beec01e4cea6340523a6dc2b8ff2758ddef8fbc40bcd9fcfca3dd",
        ),
        "state_rule": (
            "b16ddb898871056e94a092052e6a4f87bd4a96eaa5a45314161224a84fd11c1e",
            "f8514aee68e88ecf6e7575c8a5d36577dab74728dd4fa482a8fbe4791ae0e78b",
        ),
    }
    variants = (
        ("v3", environment_definitions_v3, LLMExplorerV3Controller),
        (
            "td3_only",
            environment_definitions_td3_only,
            LLMExplorerTD3OnlyController,
        ),
        (
            "state_rule",
            environment_definitions_state_rule,
            LLMStateRuleController,
        ),
    )
    for name, environment_builder, controller_type in variants:
        environment = environment_builder(**kwargs)
        controller = controller_type(
            run_directory=None,
            run_identity={"isolation": name},
            evaluation=True,
        )
        system_template = controller._templates()[0]
        rendered = controller._render_system_prompt(system_template, environment)
        environment_hash = hashlib.sha256(
            json.dumps(
                environment,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert (environment_hash, prompt_hash) == expected[name]

    shared = environment_constants_and_definitions(**kwargs)
    first_gen = environment_constants_and_definitions_first_gen(**kwargs)
    assert "propulsion" not in shared
    assert "feedback_dictionary" not in shared
    assert "traffic_defaults_packets_per_second" not in shared
    assert "propulsion" in first_gen
    assert "feedback_dictionary" in first_gen
    assert "traffic_defaults_packets_per_second" in first_gen


def test_mask_then_normalize_uses_100_percent_prior_and_edge_cases():
    prior = np.zeros(17)
    prior[3] = 3.0
    prior[5] = 1.0
    mask = np.zeros(17, dtype=bool)
    mask[[3, 5]] = True
    distribution, zero_mass = masked_routing_exploration_distribution(
        prior, mask, prior_weight=FIRST_GEN_ROUTING_PRIOR_WEIGHT
    )
    assert not zero_mass
    np.testing.assert_allclose(distribution[[3, 5]], [0.75, 0.25])
    assert distribution.sum() == pytest.approx(1.0)
    assert np.all(distribution[~mask] == 0.0)

    distribution, zero_mass = masked_routing_exploration_distribution(
        np.eye(1, 17, 2).ravel(),
        mask,
        prior_weight=FIRST_GEN_ROUTING_PRIOR_WEIGHT,
    )
    assert zero_mass
    np.testing.assert_allclose(distribution[[3, 5]], [0.5, 0.5])

    single = np.zeros(17, dtype=bool)
    single[16] = True
    distribution, _ = masked_routing_exploration_distribution(
        prior, single, prior_weight=FIRST_GEN_ROUTING_PRIOR_WEIGHT
    )
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
    assert "[Complete first-stage analysis]" in client.calls[1]["prompt"]
    assert "analysis" in client.calls[1]["prompt"]
    assert "unknown_no_applied_guidance" in client.calls[1]["prompt"]
    assert "[16, 17]" not in client.calls[1]["prompt"]


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


def test_old_first_generation_cache_identity_is_not_reused(tmp_path):
    first = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "old-cache"},
        client=SequenceClient(["old analysis", _guidance(0.05)]),
    )
    first.previous_episode_feedback = _previous_feedback()
    _begin(first)

    episode_root = tmp_path / "llm_explorer" / "episode_000168"
    identity_files = [
        episode_root / "episode_result.json",
        episode_root / "analysis" / "completed.json",
        episode_root / "numerical" / "completed.json",
    ]
    for path in identity_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["identity"]["cache_version"] = (
            "episode-fixed-dual-guidance-cache-v2"
        )
        path.write_text(json.dumps(value), encoding="utf-8")

    client = SequenceClient(["new analysis", _guidance(0.04)])
    replay = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "old-cache"},
        client=client,
    )
    replay.previous_episode_feedback = _previous_feedback()
    restored = _begin(replay)
    assert restored["reused"] is False
    assert len(client.calls) == 2


def test_numerical_template_change_invalidates_episode_result_only_as_needed(
    tmp_path, monkeypatch
):
    first = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "numerical-template-cache"},
        client=SequenceClient(["cached analysis", _guidance(0.04)]),
    )
    first.previous_episode_feedback = _previous_feedback()
    _begin(first)

    replay_client = SequenceClient([_guidance(0.03)])
    replay = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "numerical-template-cache"},
        client=replay_client,
    )
    replay.previous_episode_feedback = _previous_feedback()
    templates = replay._templates()
    revised_numerical_system = templates[2] + "\nNumerical cache identity revision."
    monkeypatch.setattr(
        replay,
        "_templates",
        lambda: (
            templates[0], templates[1], revised_numerical_system, templates[3]
        ),
    )

    result = _begin(replay)
    assert len(replay_client.calls) == 1
    assert replay_client.calls[0]["system_prompt"].endswith(
        "Numerical cache identity revision."
    )
    assert result["td3_bias"][0, 0] == pytest.approx(0.03)
    assert result["reused"] is False


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
        environment_constants=environment_constants_and_definitions_first_gen(
            episode_seconds=60,
            routing_slot_seconds=0.25,
        ),
    )
    assert not result["td3_fallback"] and not result["routing_fallback"]
    assert len(client.calls) == 3
    assert client.calls[0]["system_prompt"] != client.calls[1]["system_prompt"]
    assert "[Evidence and observability]" in client.calls[0]["system_prompt"]
    assert "[Response format]" in client.calls[1]["system_prompt"]
    assert "actual_hol_wait_duration_ms" in client.calls[0]["prompt"]
    assert '"b1":' in client.calls[0]["prompt"]
    assert '"b2":' in client.calls[0]["prompt"]
    assert "horizontal_10_mps" in client.calls[0]["prompt"]
    assert '"n_r":' not in client.calls[0]["prompt"]
    assert "P_C9 = w_C9 * mean_over_S_C9" in client.calls[0]["prompt"]
    assert "P_C10 = w_C10 * mean_over_S_C10" in client.calls[0]["prompt"]
    assert "P_COM = w_COM * mean_over_S_COM" in client.calls[0]["prompt"]
    assert "Routing-state normalization constants" in client.calls[0]["prompt"]
    assert "fixed_maximum_capacity_normalizer_bps" not in client.calls[0]["prompt"]
    assert "movement_and_service" not in client.calls[1]["prompt"].lower()
    assert "representative_cases" not in client.calls[1]["prompt"]
    assert "P_C9 = w_C9" not in client.calls[1]["prompt"]
    assert "Routing-state normalization constants" not in client.calls[1]["prompt"]

    compact = compact_feedback_for_llm(previous)
    assert len(compact["movement_and_service_tables"]["rows"]) == 16
    assert len(compact["routing_tables"]["rows"]) == 16
    movement_columns = compact["movement_and_service_tables"]["columns"]
    routing_columns = compact["routing_tables"]["columns"]
    movement_decision_index = movement_columns.index("recorded_decision_count")
    routing_decision_index = routing_columns.index("decision_count")
    epsilon_counts_index = routing_columns.index(
        "epsilon_exploration_action_counts_17"
    )
    greedy_counts_index = routing_columns.index("safe_q_greedy_action_counts_17")
    legal_counts_index = routing_columns.index("legal_action_opportunity_counts_17")
    representative_links_index = routing_columns.index(
        "representative_links_maximum_3"
    )
    assert all(
        row[movement_decision_index] == 60
        for row in compact["movement_and_service_tables"]["rows"]
    )
    assert sum(
        row[routing_decision_index]
        for row in compact["routing_tables"]["rows"]
    ) == 240 * 16
    assert all(
        len(row[epsilon_counts_index])
        == len(row[greedy_counts_index])
        == len(row[legal_counts_index])
        == 17
        for row in compact["routing_tables"]["rows"]
    )
    assert all(
        len(row[representative_links_index]) <= 3
        for row in compact["routing_tables"]["rows"]
    )
    assert "fixed_maximum_capacity_normalizer_bps" not in json.dumps(
        compact["routing_tables"]
    )
    link_layout = compact["routing_tables"]["representative_link_layout"]
    assert "current_geometry_expected_capacity_bps_summary" in link_layout
    assert "actual_allocated_service_capacity_bps_summary" in link_layout
    assert "actual_active_service_ms" in link_layout
    for row in compact["routing_tables"]["rows"]:
        for link_row in row[representative_links_index]:
            assert len(link_row) == len(link_layout)
    assert len(compact["representative_cases"]) == 8
    assert compact["previous_complete_guidance"]["td3_bias"]["shape"] == [16, 3]
    assert compact["previous_complete_guidance"]["routing_prior"]["shape"] == [16, 17]

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
    episode_root = tmp_path / "llm_explorer" / "episode_000168"
    assert (episode_root / "analysis" / "system_prompt.txt").read_text() == client.calls[0]["system_prompt"]
    assert (episode_root / "analysis" / "user_prompt.txt").read_text() == client.calls[0]["prompt"]
    assert (episode_root / "numerical" / "system_prompt.txt").read_text() == client.calls[1]["system_prompt"]
    assert (episode_root / "numerical" / "user_prompt.txt").read_text() == client.calls[1]["prompt"]
    assert (episode_root / "numerical" / "attempt_02" / "prompt.txt").read_text() == client.calls[2]["prompt"]


def test_local_context_overflow_never_calls_api_or_retries_same_prompt(tmp_path):
    client = SequenceClient([])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "overflow"},
        client=client,
    )
    previous = _previous_feedback()
    previous["representative_cases"] = [
        {"kind": "oversized_observable_case", "note": "超限0123456789" * 40_000}
    ]
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
    assert any("numerical" in path for path in paths)
    standard_root = llm_explorer_path_preflight(
        Path(r"C:\uav_llm_runs"), maximum_episode=1500
    )
    assert standard_root["longest"]["characters"] <= 240
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


def test_first_generation_training_path_passes_full_prior_weight():
    class CaptureAgent:
        def __init__(self):
            self.weights = []

        def select_action(self, *args, **kwargs):
            self.weights.append(kwargs.get("exploration_prior_weight", 0.8))
            legal = np.asarray(args[2], dtype=bool)
            action = int(np.flatnonzero(legal)[0])
            return action, {
                "decision_branch": "epsilon_exploration",
                "zero_legal_prior_mass_fallback": False,
                "exploration_distribution": None,
            }

    agent = CaptureAgent()
    states = {0: np.zeros(4)}
    masks = {0: np.asarray([True] + [False] * 16)}
    _select_routing_actions(
        agent,
        states,
        masks,
        epsilon=1.0,
        exploration_prior=np.ones((16, 17)),
        exploration_prior_weight=FIRST_GEN_ROUTING_PRIOR_WEIGHT,
    )
    assert agent.weights == [1.0]


def test_guidance_remains_eligible_after_episode_600(tmp_path):
    client = SequenceClient(["analysis after 600", _guidance(0.0)])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "after-600"},
        client=client,
    )
    controller.previous_episode_feedback = _previous_feedback()
    result = _begin(controller, episode_index=600, transitions=50_000)
    assert result["enabled"]
    assert len(client.calls) == 2


def test_numerical_prompt_contains_complete_known_matrices(tmp_path):
    prior = np.zeros((16, 17), dtype=float)
    for sender in range(16):
        prior[sender, (sender + 1) % 16] = 1.0
    previous = _previous_feedback()
    previous["previous_complete_guidance"] = {
        "td3_bias": {
            "status": "known_applied_episode_guidance",
            "shape": [16, 3],
            "value": [[0.01 * uid, 0.0, 0.0] for uid in range(16)],
        },
        "routing_prior": {
            "status": "known_applied_episode_guidance",
            "shape": [16, 17],
            "value": prior.tolist(),
        },
    }
    client = SequenceClient(["distinct complete analysis", _guidance(0.0)])
    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "complete-history"},
        client=client,
    )
    controller.previous_episode_feedback = previous
    _begin(controller)
    numerical = client.calls[1]["prompt"]
    assert "distinct complete analysis" in numerical
    presented = compact_feedback_for_llm(previous)["previous_complete_guidance"]
    assert json.dumps(presented["td3_bias"], separators=(",", ":")) in numerical
    assert json.dumps(presented["routing_prior"], separators=(",", ":")) in numerical


def test_movement_summary_uses_circular_heading_and_target_bearing():
    def row(heading, relative):
        uavs = []
        for uid in range(16):
            uavs.append(
                {
                    "uav_id": uid,
                    "task_types": ["FOV"] if uid == 0 else [],
                    "movement_mask": uid == 0,
                    "actor_normalized_action": [0.0, heading if uid == 0 else 0.0, 0.0],
                    "policy_projected_normalized_action": [0.0, heading if uid == 0 else 0.0, 0.0],
                    "executed_normalized_action": [0.0, heading if uid == 0 else 0.0, 0.0],
                    "applied_llm_bias": [0.0, 0.0, 0.0] if uid == 0 else None,
                    "observable_targets": (
                        [{"kind": "roi", "id": 1, "relative_position_m": relative}]
                        if uid == 0
                        else []
                    ),
                    "service_observation": None,
                    "movement_energy_j": 1.0,
                }
            )
        return {"time_seconds": len(relative), "uavs": uavs}

    summary, _ = first_gen_movement_summary(
        [row(0.99, [-1.0, 0.01, 0.0]), row(-0.99, [-1.0, -0.01, 0.0])],
        {"enabled": False, "td3_fallback": True},
    )
    uav = summary["by_uav"]["UAV_0"]
    assert abs(
        abs(
            uav["actor_heading_circular_rad_td3_controlled_only"][
                "circular_mean_rad"
            ]
        )
        - np.pi
    ) < 0.05
    bearing = uav["service_target_geometry"]["by_service"]["FOV"][
        "bearing_xy_rad"
    ]
    assert abs(abs(bearing["circular_mean_rad"]) - np.pi) < 0.05
    assert bearing["resultant_length"] > 0.99


def test_movement_summary_aligns_signed_heading_error_height_and_target_switches():
    def record(time_s, *, task, kind, target_id, heading, relative):
        rows = []
        for uid in range(16):
            active = uid == 0
            rows.append(
                {
                    "uav_id": uid,
                    "task_types": [task] if active else [],
                    "movement_mask": active,
                    "actor_normalized_action": [0.0, heading if active else 0.0, 0.0],
                    "policy_projected_normalized_action": [0.0, heading if active else 0.0, 0.0],
                    "executed_normalized_action": [0.0, heading if active else 0.0, 0.0],
                    "observable_targets": (
                        [{"kind": kind, "id": target_id, "relative_position_m": relative}]
                        if active
                        else []
                    ),
                    "service_observation": None,
                    "movement_energy_j": 2.0,
                }
            )
        return {"time_seconds": time_s, "uavs": rows}

    summary, _ = first_gen_movement_summary(
        [
            record(0.0, task="FOV", kind="roi", target_id=1, heading=0.99, relative=[-1.0, 0.01, -20.0]),
            record(1.0, task="COM", kind="sr", target_id=8, heading=-0.99, relative=[-1.0, -0.01, -5.0]),
            record(2.0, task="FOV", kind="roi", target_id=2, heading=0.99, relative=[-1.0, 0.01, -30.0]),
        ],
        {"enabled": False, "td3_fallback": True},
    )
    uav = summary["by_uav"]["UAV_0"]
    fov_all = uav["service_target_geometry"]["by_service"]["FOV"]
    fov_td3 = uav["td3_controlled_service_target_geometry"]["by_service"]["FOV"]
    com_td3 = uav["td3_controlled_service_target_geometry"]["by_service"]["COM"]

    assert uav["role_decision_counts"] == {"COM": 1, "FOV": 2}
    assert fov_all["relative_height_uav_minus_target_m"]["mean"] == 25.0
    assert fov_all["observed_target_switch_count"] == 1
    assert set(fov_td3["signed_heading_error_by_target"]) == {"1", "2"}
    assert abs(fov_td3["signed_target_bearing_minus_actor_heading_rad"]["circular_mean_rad"]) < 0.05
    assert abs(com_td3["signed_target_bearing_minus_actor_heading_rad"]["circular_mean_rad"]) < 0.05
    assert uav["movement_energy_j"]["all_controllers_sum_j"] == 6.0
    assert uav["movement_energy_j"]["td3_controlled_intervals_sum_j"] == 6.0


def _first_gen_movement_record(
    interval, *, controlled_uav_ids=(), bias_value=0.02,
    guidance_active=True,
):
    controlled = {int(value) for value in controlled_uav_ids}
    bias = [float(bias_value), 0.0, 0.0]
    return {
        "time_seconds": float(interval),
        "uavs": [
            {
                "uav_id": uav_id,
                "task_types": ["FOV"] if uav_id == 0 else [],
                "movement_mask": uav_id in controlled,
                "actor_normalized_action": [0.1, 0.0, 0.0],
                "pre_projection_normalized_action": [
                    0.1 + float(bias_value),
                    0.0,
                    0.0,
                ],
                "policy_projected_normalized_action": [
                    0.1 + float(bias_value),
                    0.0,
                    0.0,
                ],
                "executed_normalized_action": (
                    [0.1 + float(bias_value), 0.0, 0.0]
                    if uav_id in controlled
                    else [-1.0, 0.0, 0.0]
                ),
                "td3_guidance_active": bool(guidance_active),
                "actor_pre_projection_llm_bias": (
                    bias if guidance_active else None
                ),
                "td3_controlled_applied_llm_bias": (
                    bias
                    if guidance_active and uav_id in controlled
                    else None
                ),
                # A TD3 diagnostic zero/value can exist independently of
                # whether the host movement mask admits TD3 control.
                "applied_llm_bias": bias,
                "observable_targets": (
                    [
                        {
                            "kind": "roi",
                            "id": 1,
                            "relative_position_m": [10.0, 0.0, -100.0],
                        }
                    ]
                    if uav_id == 0
                    else []
                ),
                "service_observation": None,
                "movement_energy_j": 1.0,
            }
            for uav_id in range(16)
        ],
    }


def _active_first_gen_guidance(value=0.02):
    return {
        "enabled": True,
        "td3_fallback": False,
        "routing_fallback": False,
        "td3_bias": [[float(value), 0.0, 0.0] for _ in range(16)],
        "routing_prior": [[1.0 / 17.0] * 17 for _ in range(16)],
    }


def test_movement_bias_application_requires_td3_control_but_keeps_requested_bias():
    records = [
        _first_gen_movement_record(index, controlled_uav_ids=())
        for index in range(3)
    ]
    summary, cases = first_gen_movement_summary(
        records, _active_first_gen_guidance()
    )
    uav = summary["by_uav"]["UAV_0"]
    assert uav["td3_controlled_decision_count"] == 0
    assert uav["requested_episode_fixed_raw_bias_normalized"] == [
        0.02,
        0.0,
        0.0,
    ]
    assert uav["actor_pre_projection_guidance_bias_normalized"]["speed"][
        "valid_sample_count"
    ] == 3
    assert uav["actually_added_bias_normalized"]["speed"][
        "valid_sample_count"
    ] == 0
    assert uav["bias_application_count"] == 0
    assert uav["td3_controlled_service_target_geometry"]["by_service"][
        "FOV"
    ]["distance_3d_m"]["valid_sample_count"] == 0
    assert all(
        case["td3_controlled_applied_llm_bias"] is None for case in cases
    )


def test_movement_bias_application_tracks_mask_switches_and_global_count():
    records = [
        _first_gen_movement_record(0, controlled_uav_ids=()),
        _first_gen_movement_record(1, controlled_uav_ids={0}),
        _first_gen_movement_record(2, controlled_uav_ids=()),
    ]
    guidance = _active_first_gen_guidance()
    movement, cases = first_gen_movement_summary(records, guidance)
    uav = movement["by_uav"]["UAV_0"]
    assert uav["td3_controlled_decision_count"] == 1
    assert uav["bias_application_count"] == 1
    assert uav["actually_added_bias_normalized"]["speed"][
        "valid_sample_count"
    ] == 1
    assert any(
        case["uav_id"] == 0
        and case["movement_mask"] is True
        and case["td3_controlled_applied_llm_bias"] == [0.02, 0.0, 0.0]
        for case in cases
    )

    feedback = assemble_first_gen_feedback(
        metrics={},
        movement_records=records,
        routing_records=[],
        slot_service_records=[],
        applied_guidance=guidance,
    )
    by_uav_count = sum(
        item["bias_application_count"]
        for item in feedback["uav_core_summary"]["by_uav"].values()
    )
    assert by_uav_count == 1
    assert feedback["previous_application_status"][
        "td3_bias_application_count"
    ] == by_uav_count


def test_active_zero_bias_is_application_but_fallback_zero_is_not():
    active_record = _first_gen_movement_record(
        0, controlled_uav_ids={0}, bias_value=0.0, guidance_active=True
    )
    active, active_cases = first_gen_movement_summary(
        [active_record], _active_first_gen_guidance(0.0)
    )
    active_uav = active["by_uav"]["UAV_0"]
    assert active_uav["bias_application_count"] == 1
    assert active_uav["actually_added_bias_normalized"]["speed"]["mean"] == 0.0
    assert any(
        case["uav_id"] == 0
        and case["td3_controlled_applied_llm_bias"] == [0.0, 0.0, 0.0]
        for case in active_cases
    )

    inactive_record = _first_gen_movement_record(
        0, controlled_uav_ids={0}, bias_value=0.0, guidance_active=False
    )
    fallback, fallback_cases = first_gen_movement_summary(
        [inactive_record], {"enabled": False, "td3_fallback": True}
    )
    fallback_uav = fallback["by_uav"]["UAV_0"]
    assert fallback_uav["requested_episode_fixed_raw_bias_normalized"] is None
    assert fallback_uav["bias_application_count"] == 0
    assert fallback_uav["actually_added_bias_normalized"]["speed"][
        "valid_sample_count"
    ] == 0
    assert all(
        case["td3_controlled_applied_llm_bias"] is None
        for case in fallback_cases
    )


def test_routing_summary_separates_preselection_service_and_shared_reward():
    names = routing_state_feature_names()
    state = np.zeros(len(names), dtype=float)
    records = []
    for branch, action in (("epsilon_exploration", 1), ("safe_q_greedy", 0)):
        records.append(
            {
                "time_seconds": len(records) * 0.25,
                "sender_uav_id": 0,
                "selected_action": action,
                "decision_branch": branch,
                "legal_action_mask": [1, 1] + [0] * 14 + [1],
                "compact_routing_state": state.tolist(),
                "current_expected_capacity_bps_by_legal_action": {"0": 0.0, "1": 2e6, "16": 3e6},
                "maximum_reference_normalizer_bps_by_legal_action": {"0": 0.0, "1": 9e6, "16": 8e6},
                "capacity_context_by_legal_action": {
                    "0": {"link_type": "WAIT", "reference_bandwidth_hz": 0.0},
                    "1": {"link_type": "U2U", "reference_bandwidth_hz": 10e6},
                    "16": {"link_type": "U2G", "reference_bandwidth_hz": 10e6},
                },
                "sender_total_backlog_bits": 100.0,
                "hol_packet_id": 7,
                "hol_remaining_deadline_seconds": 0.005,
                "zero_legal_prior_mass_fallback": False,
            }
        )
    slots = [
        {
            "time_seconds": 0.0,
            "slot_seconds": 0.25,
            "active_links": [
                {
                    "sender_id": 0,
                    "receiver_id": 1,
                    "link_type": "U2U",
                    "transmitted_bits": 50.0,
                    "active_seconds": 0.01,
                    "capacity_bits_per_second": 1.5e6,
                }
            ],
            "completed_hops": [],
            "packet_decision_uses": [
                {
                    "packet_id": 7,
                    "sender_uav_id": 0,
                    "selected_action": 1,
                }
            ],
            "global_routing_reward": 0.005,
            "shared_reward_row_sum": 0.08,
        }
    ]
    summary, _ = first_gen_routing_summary(records, slots)
    sender = summary["by_sender"]["UAV_0"]
    assert sender["epsilon_exploration_action_counts_17"][1] == 1
    assert sender["safe_q_greedy_action_counts_17"][0] == 1
    assert sender["preselected_action_packet_use_count"] == 1
    assert sender["preselected_action_packet_use_counts_17"][1] == 1
    assert sender["actual_uav_service"]["transmitted_bits"] == 50.0
    assert sender["actual_uav_service"]["completed_hops"] == 0
    actual_link = next(
        link
        for link in sender["representative_links_maximum_3"]
        if link["action_index"] == 1
    )
    assert actual_link["current_geometry_expected_capacity_bps"]["mean"] == 2e6
    assert actual_link["fixed_maximum_capacity_normalizer_bps"]["mean"] == 9e6
    assert actual_link["actual_allocated_service_capacity_bps"]["mean"] == 1.5e6
    assert summary["global_shared_routing_reward"]["episode_sum_counting_each_slot_once"] == 0.005
    assert summary["global_shared_routing_reward"]["duplicated_reward_row_sum_diagnostic"] == 0.08


def test_first_generation_versions_metadata_and_old_checkpoint_rejection(tmp_path):
    metadata = explorer_metadata(LLMExplorerConfig())
    assert metadata["schema_version"] == LLM_EXPLORER_VERSION
    assert metadata["prompt_version"] == LLM_EXPLORER_PROMPT_VERSION
    assert metadata["summary_version"] == LLM_EXPLORER_SUMMARY_VERSION
    assert metadata["cache_version"] == LLM_EXPLORER_CACHE_VERSION
    assert "independent_numerical_system" in metadata["request_prompt_contract"]
    assert "100_percent" in metadata["routing_intervention"]

    controller = LLMExplorerController(
        run_directory=tmp_path,
        run_identity={"run": "version"},
        client=SequenceClient([]),
    )
    state = controller.state_dict()
    restored = LLMExplorerController(
        run_directory=tmp_path / "restored",
        run_identity={"run": "version"},
        client=SequenceClient([]),
    )
    restored.load_state_dict(state)
    assert restored.state_dict()["schema_version"] == LLM_EXPLORER_VERSION

    state["schema_version"] = "uav-hrl-llm-explorer-firstgen-v5"
    with pytest.raises(Exception, match="checkpoint version is incompatible"):
        controller.load_state_dict(state)


def test_first_generation_observers_do_not_change_warmup_training_result():
    def config():
        return TrainingConfig(
            total_episodes=1,
            mode="smoke",
            episode_seconds=1,
            warmup_joint_transitions=0,
            routing_warmup_transitions=1,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            run_directory=None,
            random_seed=20261019,
        )

    baseline = train(
        config(), method_spec=MethodSpec.parse("td3_dinkelbach")
    )["episode_metrics"][0]
    first_generation = train(
        config(),
        method_spec=MethodSpec.parse("td3_dinkelbach_llm_explorer"),
        llm_explorer_mock=True,
    )["episode_metrics"][0]
    for key in (
        "routing_reward_total",
        "existing_movement_reward",
        "combined_movement_reward",
        "total_timely_useful_bits",
        "total_mobility_energy_j",
        "routing_cost_sum",
        "system_qos_eligible_packets",
        "system_qos_violation_count",
        "movement_base_reward_sum",
        "movement_final_reward_sum",
        "routing_wait_count",
        "partial_transmission_count",
        "coverage",
    ):
        assert first_generation[key] == pytest.approx(baseline[key])


def test_two_episode_mock_training_applies_first_generation_guidance(
    tmp_path, monkeypatch
):
    client = SequenceClient(["guided analysis", _guidance(0.02)])
    original_begin_episode = LLMExplorerController.begin_episode

    def begin_after_simulated_warmup(self, *args, **kwargs):
        kwargs["total_joint_transitions"] = max(
            int(kwargs.get("total_joint_transitions", 0)), 10_000
        )
        return original_begin_episode(self, *args, **kwargs)

    monkeypatch.setattr(
        LLMExplorerController, "begin_episode", begin_after_simulated_warmup
    )
    result = train(
        TrainingConfig(
            total_episodes=2,
            mode="smoke",
            episode_seconds=8,
            warmup_joint_transitions=0,
            routing_warmup_transitions=1,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            run_directory=str(tmp_path),
            random_seed=20261019,
        ),
        scenario_manifest=generate_manifest(
            "train", 20261019, 2, num_gt=2
        ),
        method_spec=MethodSpec.parse("td3_dinkelbach_llm_explorer"),
        llm_explorer_client=client,
    )
    assert result["episodes_run"] == 2
    assert result["llm_explorer"]["statistics"]["requests_total"] == 2
    assert len(client.calls) == 2
    feedback = json.loads(
        (
            tmp_path
            / "llm_explorer"
            / "episode_000002"
            / "feedback.json"
        ).read_text(encoding="utf-8")
    )
    application = feedback["previous_application_status"]
    assert application["guidance_enabled"] is True
    assert application["td3_fallback"] is False
    assert application["routing_fallback"] is False
    assert application["td3_bias_application_count"] > 0
    assert application["td3_bias_application_count"] == sum(
        item["bias_application_count"]
        for item in feedback["uav_core_summary"]["by_uav"].values()
    )
    assert "actual_hol_wait_duration_ms" in client.calls[0]["prompt"]


def test_first_generation_cli_train_evaluate_and_resume_without_llm_call(
    monkeypatch
):
    calls = []

    def unexpected_chat(self, **kwargs):
        calls.append(kwargs)
        raise AssertionError("short warmup/evaluation/resume must not call the LLM")

    monkeypatch.setattr(
        DeterministicMockExplorerClient, "chat", unexpected_chat
    )
    with tempfile.TemporaryDirectory(prefix="fg-") as short_temp:
        output_root = Path(short_temp)
        method = "td3_dinkelbach_llm_explorer"
        assert run_experiment_main(
            [
                method,
                "--episodes",
                "1",
                "--episode-seconds",
                "1",
                "--checkpoint-interval",
                "1",
                "--roi-count",
                "2",
                "--seed",
                "20261019",
                "--output-root",
                str(output_root),
                "--llm-explorer-mock",
            ]
        ) == 0
        run_dir = next((output_root / method).iterdir())
        assert (run_dir / "checkpoints" / "models" / "ep_0001").is_dir()

        assert run_experiment_main(
            [
                "evaluate",
                str(run_dir),
                "--checkpoint-episode",
                "1",
                "--smoke",
            ]
        ) == 0
        evaluation_dir = next((run_dir / "evaluation" / "ep_1").iterdir())
        metadata = json.loads(
            (evaluation_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["smoke_evaluation"] is True
        assert metadata["evaluation_invariants"]["exploration_disabled"] is True
        assert all(metadata["evaluation_invariants"].values())
        assert calls == []

        assert run_experiment_main(
            ["resume", str(run_dir), "--target-episodes", "2"]
        ) == 0
        assert (run_dir / "checkpoints" / "models" / "ep_0002").is_dir()
        assert calls == []
