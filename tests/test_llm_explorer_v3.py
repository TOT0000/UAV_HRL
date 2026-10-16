import json
from types import SimpleNamespace

import numpy as np
import pytest

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
    first_eligible_episode,
    llm_guidance_weight,
    movement_episode_summary,
    named_routing_prior_to_matrix,
    parse_guidance_response_v3,
    routing_episode_summary,
    routing_matrix_to_named,
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


def test_v3_windows_preflight_checks_named_artifact_paths(tmp_path):
    report = llm_explorer_path_preflight(
        tmp_path,
        maximum_episode=800,
        artifact_directory="llm_explorer_v3",
        named_routing=True,
    )
    assert report["named_routing"]
    assert "llm_explorer_v3" in report["longest"]["relative_path"]
