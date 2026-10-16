import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tiktoken

from experiment_config import MethodSpec
from llm_explorer import (
    ExplorerValidationError,
    LLMExplorerController,
    llm_explorer_path_preflight,
    masked_routing_exploration_distribution,
    prompt_token_budget,
)
from llm_explorer_v3 import (
    LLMExplorerV3Controller,
    LLM_EXPLORER_V3_CASE_VERSION,
    LLM_EXPLORER_V3_SUMMARY_VERSION,
    environment_definitions_v3,
    first_eligible_episode,
    llm_guidance_weight,
    movement_episode_summary,
    named_routing_prior_to_matrix,
    parse_guidance_response_v3,
    routing_episode_summary,
    routing_matrix_to_named,
    select_representative_cases,
)
from observation_strategy import routing_state_feature_names


def _named_uniform():
    matrix = np.full((16, 17), 1.0 / 17.0)
    return routing_matrix_to_named(matrix)


def _guidance(bias=0.1):
    return json.dumps(
        {
            "td3_bias": [[bias, 0.0, 0.0] for _ in range(16)],
            "routing_prior": _named_uniform(),
        }
    )


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return {
            "content": value,
            "actual_model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            "finish_reason": "stop",
        }


def _feedback():
    return {
        "metrics": {
            "episode": 167,
            "system_qos_violation_probability": 0.1,
            "routing_stage_violation_count": 2,
        },
        "movement_summaries": {f"UAV_{i}": {"td3_controlled_actor_decision_count": 0} for i in range(16)},
        "routing_summaries": {f"UAV_{i}": {"decision_count": 0} for i in range(16)},
        "representative_cases": [],
        "applied_guidance": {
            "enabled": False,
            "td3_bias": [[0.0] * 3 for _ in range(16)],
            "routing_prior": None,
        },
    }


def _begin(controller, episode, transitions):
    return controller.begin_episode(
        episode_index=episode - 1,
        total_joint_transitions=transitions,
        sigma_episode_start=0.4,
        routing_epsilon=0.5,
        current_observable_initial_state={"uav_rows": []},
        environment_constants={"episode_seconds": 60, "units": "SI"},
    )


def _rich_initial_state():
    return {
        "time": "episode_start_before_first_action",
        "hidden_roi_count_disclosed": False,
        "uav_columns": [
            "id",
            "x_m",
            "y_m",
            "z_m",
            "energy_j",
            "backlog_bits",
            "observable_tasks_[type,target_or_null]",
        ],
        "uav_rows": [
            [
                uav_id,
                float(50 + 60 * uav_id),
                float(75 + 35 * uav_id),
                float(80 + uav_id),
                float(90_000 - 100 * uav_id),
                float(10_000 * uav_id),
                [["Search", None], ["FOV", uav_id % 4]],
            ]
            for uav_id in range(16)
        ],
        "known_rois_current_episode": [
            [roi_id, 100.0 + roi_id, 200.0 + roi_id, 0.0, 35.0]
            for roi_id in range(4)
        ],
        "observable_sr_current_episode": [
            [sr_id, 300.0 + sr_id, 400.0 + sr_id, 0.0, sr_id % 4, False]
            for sr_id in range(8)
        ],
        "ground_station_position_m": [500.0, 500.0, 0.0],
    }


def _full_scale_records():
    feature_names = routing_state_feature_names()
    feature_index = {name: index for index, name in enumerate(feature_names)}
    movement = []
    routing = []
    for step in range(60):
        movement.append(
            {
                "episode_index": 199,
                "time_seconds": float(step),
                "action_source": (
                    "actor_plus_llm_bias_plus_gaussian_exploration"
                    if step % 2 == 0
                    else "actor_plus_gaussian_exploration"
                ),
                "uavs": [
                    {
                        "uav_id": uav_id,
                        "task_types": [
                            ["Search", "FOV", "COM", "Hovering"][(uav_id + step) % 4]
                        ],
                        "movement_mask": True,
                        "actor_normalized_action": [
                            0.92 if (step + uav_id) % 5 == 0 else -0.2 + 0.02 * uav_id,
                            -0.95 + ((step + 3 * uav_id) % 20) / 10.0,
                            -0.5 + ((step + uav_id) % 10) / 10.0,
                        ],
                        "observable_targets": [
                            {
                                "relative_position_m": [
                                    float(10 + uav_id),
                                    float(step),
                                    float(2 + uav_id % 3),
                                ]
                            }
                        ],
                        "movement_energy_j": float(100 + uav_id + step / 10),
                    }
                    for uav_id in range(16)
                ],
            }
        )
        for sender in range(16):
            state = np.zeros(len(feature_names), dtype=float)
            legal = np.ones(17, dtype=int)
            # Keep many legal neighbors while leaving a true unknown action.
            unavailable = (sender + 8) % 16
            if unavailable == sender:
                unavailable = (unavailable + 1) % 16
            legal[unavailable] = 0
            for action in range(17):
                state[feature_index[f"link_capacity[{action}]"]] = (
                    0.0 if action == 0 else (action + step % 7) / 25.0
                )
                state[feature_index[f"next_hop_backlog[{action}]"]] = (
                    sender + action + step
                ) / 100.0
                state[feature_index[f"gs_progress_norm[{action}]"]] = (
                    action - sender
                ) / 16.0
            wait_anomaly = step % 15 == 0
            zero_mass = step % 17 == 0
            selected = sender if wait_anomaly else (sender + 1) % 16
            distribution = np.full(17, 0.4 / 16.0, dtype=float)
            distribution[sender] = 0.6
            routing.append(
                {
                    "time_seconds": step + sender / 100.0,
                    "sender_uav_id": sender,
                    "selected_action": selected,
                    "compact_routing_state": state.tolist(),
                    "legal_action_mask": legal.tolist(),
                    "sender_total_backlog_bits": float(50_000 + 100 * step + sender),
                    "hol_remaining_deadline_seconds": (
                        None if step == 0 else float(5.0 - step / 20.0)
                    ),
                    "decision_branch": (
                        "epsilon_exploration" if step % 2 == 0 else "safe_q_greedy"
                    ),
                    "zero_legal_prior_mass_fallback": zero_mass,
                    "exploration_distribution": distribution.tolist(),
                }
            )
    return movement, routing


def _active_guidance():
    raw = np.asarray(
        [[0.1 + 0.001 * uav_id, -0.02, 0.01] for uav_id in range(16)],
        dtype=np.float32,
    )
    prior = np.full((16, 17), 1.0 / 17.0, dtype=np.float32)
    return {
        "enabled": True,
        "td3_fallback": False,
        "routing_fallback": False,
        "td3_reason": None,
        "routing_reason": None,
        "td3_bias": raw,
        "td3_bias_raw": raw,
        "td3_bias_effective": 0.75 * raw,
        "routing_prior": prior,
        "routing_prior_named": routing_matrix_to_named(prior),
        "annealing_weight": 0.75,
        "routing_alpha": 0.6,
        "sigma_episode_start": 0.4,
    }


def _full_metrics():
    return {
        "energy_efficiency_bit_per_j": 1234.5,
        "timely_useful_delivery_bits": 987654.0,
        "movement_energy_j": 800.0,
        "td3_original_cumulative_reward": -10.0,
        "safe_ddqn_original_cumulative_reward": 20.0,
        "lambda_EE_Mbit_per_J_used": 0.01,
        "lambda_cost_used": 0.2,
        "lambda_cost_after_episode": 0.21,
        "system_qos_eligible_packets": 32,
        "system_qos_violation_count": 4,
        "system_qos_violation_probability": 0.125,
        "routing_stage_eligible_packets": 40,
        "routing_stage_violation_count": 6,
        "routing_immediate_cost_sum": 2.5,
    }


def test_v3_method_is_additive_and_old_controller_identity_is_unchanged(tmp_path):
    old = MethodSpec.parse("td3_dinkelbach_llm_explorer")
    new = MethodSpec.parse("td3_dinkelbach_llm_explorer_v3")
    assert old.llm_explorer_enabled and not old.llm_explorer_v3_enabled
    assert new.llm_explorer_enabled and new.llm_explorer_v3_enabled
    assert LLMExplorerController(
        run_directory=tmp_path, run_identity={"method": old.method_id}, evaluation=True
    ).root.name == "llm_explorer"
    assert LLMExplorerV3Controller(
        run_directory=tmp_path, run_identity={"method": new.method_id}, evaluation=True
    ).root.name == "llm_explorer_v3"


def test_fixed_1_based_annealing_boundaries_and_nonstandard_episode_length():
    start = first_eligible_episode(10_000, 60)
    assert start == 168
    assert llm_guidance_weight(167, start_episode=start) == 0.0
    assert llm_guidance_weight(168, start_episode=start) == 1.0
    assert llm_guidance_weight(384, start_episode=start) == 0.5
    assert llm_guidance_weight(600, start_episode=start) == 0.0
    assert llm_guidance_weight(800, start_episode=start) == 0.0
    assert first_eligible_episode(10_000, 37) == 272
    assert llm_guidance_weight(600, start_episode=600) == 0.0


def test_warmup_cutoff_and_evaluation_do_not_construct_or_call_client(tmp_path):
    client = SequenceClient([AssertionError("must not call")])
    controller = LLMExplorerV3Controller(
        run_directory=tmp_path / "warmup",
        run_identity={"run": "warmup"},
        client=client,
    )
    controller.previous_episode_feedback = _feedback()
    warmup = _begin(controller, 167, 9_960)
    assert not warmup["enabled"] and warmup["td3_reason"] == "warmup"
    assert controller.client is None and client.calls == []

    cutoff = LLMExplorerV3Controller(
        run_directory=tmp_path / "cutoff",
        run_identity={"run": "cutoff"},
        client=client,
    )
    cutoff.previous_episode_feedback = _feedback()
    result = _begin(cutoff, 600, 35_940)
    assert result["td3_reason"] == "annealing_finished"
    assert not result["td3_fallback"] and not result["routing_fallback"]
    assert cutoff.client is None and cutoff.statistics["requests_total"] == 0

    evaluation = LLMExplorerV3Controller(
        run_directory=tmp_path / "eval",
        run_identity={"run": "eval"},
        evaluation=True,
        client=client,
    )
    evaluation.previous_episode_feedback = _feedback()
    assert _begin(evaluation, 384, 20_000)["td3_reason"] == "evaluation_disabled"
    assert evaluation.client is None and client.calls == []


def test_v3_calls_twice_and_exposes_raw_and_once_scaled_bias(tmp_path):
    client = SequenceClient(["Concise supported analysis.", _guidance(0.2)])
    controller = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "active"},
        client=client,
    )
    controller.previous_episode_feedback = _feedback()
    guidance = _begin(controller, 384, 20_000)
    assert len(client.calls) == 2
    assert guidance["annealing_weight"] == 0.5
    assert guidance["routing_alpha"] == 0.4
    np.testing.assert_allclose(guidance["td3_bias_raw"][:, 0], 0.2)
    np.testing.assert_allclose(guidance["td3_bias_effective"][:, 0], 0.1)
    assert guidance["routing_prior_named"]["UAV_0"]["WAIT"] == pytest.approx(1 / 17)
    assert all("{{" not in call["prompt"] for call in client.calls)
    assert all("{{" not in call["system_prompt"] for call in client.calls)
    budget_paths = list(
        (tmp_path / "llm_explorer_v3" / "episode_000384").glob(
            "*/attempt_*/token_budget.json"
        )
    )
    assert len(budget_paths) == 2
    assert all(
        json.loads(path.read_text())["input_tokens"] <= 20_000
        for path in budget_paths
    )
    controller.complete_episode(
        episode_index=383,
        metrics={
            "system_qos_violation_probability": 0.0,
            "routing_stage_violation_count": 0,
        },
    )
    saved = json.loads(
        (tmp_path / "llm_explorer_v3" / "state.json").read_text()
    )
    assert saved["schema_version"] == "uav-hrl-llm-explorer-v3"

    replay_client = SequenceClient([])
    replay = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "active"},
        client=replay_client,
    )
    replay.previous_episode_feedback = _feedback()
    restored = _begin(replay, 384, 20_000)
    assert restored["reused"] and replay_client.calls == []
    np.testing.assert_allclose(restored["td3_bias_effective"][:, 0], 0.1)


def test_routing_alpha_is_point_eight_times_weight():
    prior = np.zeros(17)
    prior[16] = 1.0
    legal = np.zeros(17, dtype=bool)
    legal[[0, 16]] = True
    distribution, zero_mass = masked_routing_exploration_distribution(
        prior, legal, prior_weight=0.4
    )
    assert not zero_mass
    assert distribution[0] == pytest.approx(0.3)
    assert distribution[16] == pytest.approx(0.7)


def test_case_reduction_keeps_verified_core_under_input_limit(tmp_path):
    previous = _feedback()
    previous["representative_cases"] = [
        {"kind": "oversized_case", "detail": "observable " * 2_500}
        for _ in range(8)
    ]
    client = SequenceClient(["Concise analysis.", _guidance(0.0)])
    controller = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "reduction"},
        client=client,
    )
    controller.previous_episode_feedback = previous
    result = _begin(controller, 168, 10_020)
    assert result["enabled"] and len(client.calls) == 2
    analysis_call = client.calls[0]
    budget = prompt_token_budget(
        analysis_call["system_prompt"],
        analysis_call["prompt"],
        reserved_output_tokens=analysis_call["max_output_tokens"],
    )
    assert budget["input_tokens"] <= 20_000
    assert "Per-UAV movement summaries" in analysis_call["prompt"]
    assert "Per-sender routing summaries" in analysis_call["prompt"]
    assert analysis_call["prompt"].count("oversized_case") < 8


def test_irreducible_oversize_stops_locally_with_explicit_fallback(tmp_path):
    previous = _feedback()
    previous["metrics"]["uncompressible_observation"] = "observable " * 25_000
    client = SequenceClient([])
    controller = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "irreducible-budget"},
        client=client,
    )
    controller.previous_episode_feedback = previous
    result = _begin(controller, 168, 10_020)
    assert not result["enabled"]
    assert result["td3_reason"] == "analysis_stage_failed"
    assert result["routing_reason"] == "analysis_stage_failed"
    assert client.calls == []
    assert controller.statistics["failure_counts"]["local_preflight"] == 1
    assert controller.statistics["fallback_episodes_total"] == 1


def test_named_routing_mapping_and_strict_format_errors():
    named = _named_uniform()
    matrix = named_routing_prior_to_matrix(named)
    assert matrix.shape == (16, 17)
    assert matrix[0, 0] == pytest.approx(named["UAV_0"]["WAIT"])
    assert matrix[0, 16] == pytest.approx(named["UAV_0"]["GS"])
    assert matrix[0, 1] == pytest.approx(named["UAV_0"]["UAV_1"])

    missing = dict(named)
    missing.pop("UAV_15")
    with pytest.raises(ExplorerValidationError):
        named_routing_prior_to_matrix(missing)
    own_name = json.loads(json.dumps(named))
    own_name["UAV_2"]["UAV_2"] = own_name["UAV_2"].pop("WAIT")
    with pytest.raises(ValueError, match="WAIT"):
        named_routing_prior_to_matrix(own_name)
    boolean = json.loads(json.dumps(named))
    boolean["UAV_0"]["WAIT"] = True
    with pytest.raises(ValueError, match="booleans"):
        named_routing_prior_to_matrix(boolean)
    duplicate = '{"td3_bias":[],"td3_bias":[],"routing_prior":{}}'
    _, _, errors = parse_guidance_response_v3(duplicate, 0.4)
    assert "duplicate JSON key" in errors["td3"]
    matrix_response = json.dumps(
        {"td3_bias": [[0.0] * 3] * 16, "routing_prior": [[1 / 17] * 17] * 16}
    )
    _, routing, errors = parse_guidance_response_v3(matrix_response, 0.4)
    assert routing is None and "routing" in errors


def _movement_record(mask, actor_speed, heading, action_source="actor_plus_gaussian_exploration"):
    return {
        "episode_index": 167,
        "time_seconds": 1.0,
        "action_source": action_source,
        "uavs": [
            {
                "uav_id": uid,
                "task_types": ["Search"] if uid == 0 else ["FOV"],
                "movement_mask": bool(mask if uid == 0 else False),
                "actor_normalized_action": (
                    [actor_speed, heading, 0.0] if uid == 0 else None
                ),
                "observable_targets": [],
                "movement_energy_j": 1.0,
            }
            for uid in range(16)
        ],
    }


def test_movement_summary_filters_mask_and_uses_circular_heading():
    records = [
        _movement_record(False, 1.0, 0.99, "warmup_random"),
        _movement_record(True, 0.0, 0.99),
        _movement_record(True, 0.2, -0.99),
    ]
    summary, _ = movement_episode_summary(
        records,
        {
            "td3_bias_raw": np.zeros((16, 3)),
            "td3_bias_effective": np.zeros((16, 3)),
            "sigma_episode_start": 0.4,
        },
    )
    uav = summary["UAV_0"]
    assert uav["movement_mask_true_count"] == 2
    assert uav["td3_controlled_actor_decision_count"] == 2
    assert uav["actor_horizontal_speed_mps"]["mean"] == pytest.approx(5.5)
    assert abs(abs(uav["actor_heading_circular_mean_rad"]) - np.pi) < 0.05


def test_routing_full_statistics_are_not_derived_from_sample_limit():
    names = routing_state_feature_names()
    state = np.zeros(len(names), dtype=float)
    records = []
    for index in range(150):
        records.append(
            {
                "time_seconds": index * 0.25,
                "sender_uav_id": 0,
                "selected_action": 0 if index < 100 else 16,
                "compact_routing_state": state.tolist(),
                "legal_action_mask": [1 if action in (0, 16) else 0 for action in range(17)],
                "sender_total_backlog_bits": 100.0,
                "hol_remaining_deadline_seconds": 1.0,
                "decision_branch": "epsilon_exploration" if index < 120 else "safe_q_greedy",
                "zero_legal_prior_mass_fallback": index < 10,
                "exploration_distribution": [0.8] + [0.0] * 15 + [0.2],
            }
        )
    summary, cases = routing_episode_summary(records)
    sender = summary["UAV_0"]
    assert sender["decision_count"] == 150
    assert sender["branches"]["epsilon_exploration"]["decision_count"] == 120
    assert sender["branches"]["safe_q_greedy"]["decision_count"] == 30
    assert sender["branches"]["epsilon_exploration"]["zero_legal_prior_mass_count"] == 10
    assert sender["legal_action_occurrence_counts"] == {"GS": 150, "WAIT": 150}
    assert cases


def test_full_scale_prompt_budget_statistics_and_raw_diagnostics(tmp_path):
    encoder = tiktoken.get_encoding("o200k_base")
    analysis_seed = (
        "Evidence remains observational. Movement and routing recommendations "
        "must preserve uncertainty and cite measured denominators. "
    ) * 500
    near_limit_analysis = encoder.decode(encoder.encode(analysis_seed)[:1350])
    client = SequenceClient([near_limit_analysis, "{}", _guidance(0.0)])
    controller = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "full-scale-budget"},
        client=client,
    )
    movement, routing = _full_scale_records()
    controller.active_guidance = _active_guidance()
    controller.movement_records = copy.deepcopy(movement)
    controller.routing_records = copy.deepcopy(routing)
    controller.complete_episode(episode_index=199, metrics=_full_metrics())

    feedback = copy.deepcopy(controller.previous_episode_feedback)
    assert feedback["summary_version"] == LLM_EXPLORER_V3_SUMMARY_VERSION
    assert feedback["case_selection_version"] == LLM_EXPLORER_V3_CASE_VERSION
    assert feedback["routing_summaries"]["UAV_0"]["decision_count"] == 60
    assert feedback["routing_summaries"]["UAV_15"]["decision_count"] == 60
    assert feedback["metrics"]["energy_efficiency_bit_per_j"] == 1234.5
    assert feedback["metrics"]["timely_useful_delivery_bits"] == 987654.0
    assert feedback["metrics"]["movement_energy_j"] == 800.0
    assert feedback["metrics"]["system_qos_eligible_packets"] == 32
    assert feedback["metrics"]["routing_stage_eligible_packets"] == 40

    raw = json.loads(
        (
            tmp_path
            / "llm_explorer_v3"
            / "episode_000200"
            / "feedback.json"
        ).read_text(encoding="utf-8")
    )
    assert len(raw["movement_decisions"]) == 60
    assert len(raw["routing_decisions"]) == 60 * 16
    assert raw["routing_summaries"] == feedback["routing_summaries"]
    assert "min" in raw["routing_summaries"]["UAV_0"][
        "legal_action_observable_summaries"
    ]["WAIT"]["reference_capacity_norm"]

    compact = controller._compact_feedback(feedback)
    route_columns = compact["routing_summaries"]["columns"]
    route_row = compact["routing_summaries"]["rows"][0]
    legal = route_row[route_columns.index("legal_occurrence_count_by_action_index")]
    capacities = route_row[
        route_columns.index("reference_capacity_mean_by_action_index")
    ]
    assert legal[8] == 0 and capacities[8] is None
    assert legal[0] == 60 and capacities[0] == 0.0
    assert "null means missing/unknown; numeric 0 is observed zero" in compact[
        "routing_summaries"
    ]["unknown_and_zero"]
    assert "routing_prior_named" not in compact["applied_guidance"]
    assert "td3_bias_rows" not in compact["applied_guidance"]
    assert compact["applied_guidance"]["td3_bias_location"].startswith(
        "movement summary rows"
    )

    result = controller.begin_episode(
        episode_index=200,
        total_joint_transitions=12_000,
        sigma_episode_start=0.4,
        routing_epsilon=0.5,
        current_observable_initial_state=_rich_initial_state(),
        environment_constants=environment_definitions_v3(
            episode_seconds=60,
            routing_slot_seconds=0.25,
        ),
    )
    assert result["enabled"]
    assert len(client.calls) == 3
    assert len(encoder.encode(client.calls[0]["prompt"])) > 6_000
    assert '"selection_profile":"movement=1,routing=2"' in client.calls[0][
        "prompt"
    ]
    assert "ordinary_recorded_routing_decision" in client.calls[0]["prompt"]
    assert "response must contain exactly td3_bias and routing_prior" in client.calls[2][
        "prompt"
    ]
    budgets = [
        prompt_token_budget(
            call["system_prompt"],
            call["prompt"],
            reserved_output_tokens=call["max_output_tokens"],
        )
        for call in client.calls
    ]
    assert all(budget["input_tokens"] <= 20_000 for budget in budgets)
    assert budgets[0]["input_tokens"] <= 12_000
    assert budgets[1]["input_tokens"] <= 12_000
    assert budgets[2]["input_tokens"] <= 12_000


def test_case_selection_is_diverse_deterministic_and_preserves_ordinary():
    movement, routing = _full_scale_records()
    movement_summary, movement_candidates = movement_episode_summary(
        movement, _active_guidance()
    )
    routing_summary, routing_candidates = routing_episode_summary(
        routing, maximum_cases=5, episode=200
    )
    selected = select_representative_cases(
        movement_candidates, routing_candidates
    )
    repeated = select_representative_cases(
        movement_candidates, routing_candidates
    )
    assert selected == repeated
    assert movement_summary["UAV_0"]["td3_controlled_actor_decision_count"] == 60
    assert routing_summary["UAV_0"]["decision_count"] == 60

    selected_movement = [case for case in selected if case["domain"] == "movement"]
    selected_routing = [case for case in selected if case["domain"] == "routing"]
    assert len(selected_movement) == 3
    assert len({case["uav_id"] for case in selected_movement}) == 3
    assert len({int(float(case["time_s"]) // 20) for case in selected_movement}) == 3
    assert len(selected_routing) == 5
    assert len({case["kind"] for case in selected_routing}) >= 3
    assert len({case["sender"] for case in selected_routing}) >= 3
    assert any(
        case["kind"] == "ordinary_recorded_routing_decision"
        for case in selected_routing
    )

    reduced = select_representative_cases(
        movement_candidates,
        routing_candidates,
        movement_maximum=1,
        routing_maximum=2,
    )
    reduced_routing = [case for case in reduced if case["domain"] == "routing"]
    assert len(reduced_routing) == 2
    assert any(
        case["kind"] == "ordinary_recorded_routing_decision"
        for case in reduced_routing
    )
    assert all(case in movement_candidates + routing_candidates for case in reduced)

    ordinary_only = [
        {
            "domain": "routing",
            "kind": "ordinary_recorded_routing_decision",
            "time_s": float(sender * 5),
            "sender": f"UAV_{sender}",
            "selected_action": "GS",
        }
        for sender in range(8)
    ]
    ordinary_selected = select_representative_cases(
        [], ordinary_only, movement_maximum=0, routing_maximum=5
    )
    assert len(ordinary_selected) == 5
    assert len({case["sender"] for case in ordinary_selected}) == 5


def test_v1_summary_checkpoint_loads_without_reset_and_uses_v2_cache_identity(
    tmp_path,
):
    source = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "resume-compatible"},
        evaluation=True,
        rng_seed=123,
    )
    legacy_feedback = _feedback()
    legacy_feedback.update(
        summary_version="verified-episode-summary-v1",
        case_selection_version="deterministic-representative-cases-v1",
    )
    source.previous_episode_feedback = legacy_feedback
    source.rng.random(7)
    state = source.state_dict()
    expected_rng_state = copy.deepcopy(state["rng_state"])

    restored = LLMExplorerV3Controller(
        run_directory=tmp_path,
        run_identity={"run": "resume-compatible"},
        evaluation=True,
        rng_seed=999,
    )
    restored.load_state_dict(state)
    assert restored.previous_episode_feedback == legacy_feedback
    assert restored.rng.bit_generator.state == expected_rng_state
    compact_once = restored._compact_feedback(legacy_feedback)
    compact_twice = restored._compact_feedback(legacy_feedback)
    assert compact_once == compact_twice

    system_template, analysis_template, _ = restored._templates()
    environment = environment_definitions_v3(
        episode_seconds=60,
        routing_slot_seconds=0.25,
    )
    system_prompt = restored._render_system_prompt(system_template, environment)
    settings = restored._exploration_settings(
        sigma_episode_start=0.4,
        routing_epsilon=0.5,
    )
    prompt_once = restored._render_analysis_prompt(
        system_prompt=system_prompt,
        analysis_template=analysis_template,
        previous=compact_once,
        environment_constants=environment,
        current_observable_initial_state=_rich_initial_state(),
        settings=settings,
    )
    prompt_twice = restored._render_analysis_prompt(
        system_prompt=system_prompt,
        analysis_template=analysis_template,
        previous=compact_twice,
        environment_constants=environment,
        current_observable_initial_state=_rich_initial_state(),
        settings=settings,
    )
    assert prompt_once == prompt_twice

    identity = restored._cache_identity(
        episode_index=200,
        stage="analysis",
        system_prompt="system",
        user_prompt="prompt",
    )
    assert identity["summary_version"] == LLM_EXPLORER_V3_SUMMARY_VERSION
    assert identity["case_selection_version"] == LLM_EXPLORER_V3_CASE_VERSION
    assert identity["summary_version"] != legacy_feedback["summary_version"]


def test_v3_windows_preflight_checks_named_artifact_paths(tmp_path):
    report = llm_explorer_path_preflight(
        tmp_path,
        maximum_episode=800,
        artifact_directory="llm_explorer_v3",
        named_routing=True,
    )
    assert report["named_routing"]
    assert "llm_explorer_v3" in report["longest"]["relative_path"]
