import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tiktoken

from experiment_config import MethodSpec
from HRL_task_aware import (
    TrainingConfig,
    _llm_routing_exploration_prior,
    _select_routing_actions,
    train,
)
from llm_explorer import (
    ExplorerValidationError,
    LLMExplorerController,
    llm_explorer_path_preflight,
    prompt_token_budget,
)
from llm_explorer_v3 import LLMExplorerV3Controller
from llm_explorer_td3_only import (
    LLM_EXPLORER_TD3_ONLY_VERSION,
    LLMExplorerTD3OnlyController,
    communication_bottleneck_summary,
    environment_definitions_td3_only,
    explorer_metadata_td3_only,
    movement_episode_summary_td3_only,
    output_schema_td3_only,
    parse_guidance_response_td3_only,
    recorded_raw_td3_bias,
)
from observation_strategy import routing_state_feature_names
from run_experiment import _explorer_preflight_options, build_parser


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
            "actual_model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
            },
            "finish_reason": "stop",
        }


def _raw_matrix():
    return np.arange(1, 49, dtype=np.float32).reshape(16, 3) / 1000.0


def _guidance(raw=None):
    matrix = _raw_matrix() if raw is None else np.asarray(raw, dtype=float)
    return json.dumps({"td3_bias": matrix.tolist()}, separators=(",", ":"))


def _active_guidance(raw=None, weight=0.75):
    matrix = _raw_matrix() if raw is None else np.asarray(raw, dtype=np.float32)
    return {
        "enabled": True,
        "td3_bias": matrix,
        "td3_bias_raw": matrix,
        "td3_bias_effective": weight * matrix,
        "td3_fallback": False,
        "td3_reason": None,
        "routing_prior": None,
        "routing_fallback": False,
        "routing_reason": "routing_guidance_disabled",
        "annealing_weight": weight,
        "routing_alpha": 0.0,
        "sigma_episode_start": 0.4,
        "analysis_attempt": 1,
        "td3_guidance_attempt": 1,
    }


def _metrics():
    return {
        "energy_efficiency_bit_per_j": 1234.5,
        "timely_useful_delivery_bits": 987654.0,
        "movement_energy_j": 800.0,
        "td3_original_cumulative_reward": -10.0,
        "safe_ddqn_original_cumulative_reward": 20.0,
        "system_qos_eligible_packets": 32,
        "system_qos_violation_count": 4,
        "system_qos_violation_probability": 0.125,
        "routing_stage_eligible_packets": 40,
        "routing_stage_violation_count": 6,
    }


def _initial_state():
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
            [uav_id, 20.0 * uav_id, 15.0 * uav_id, 80.0, 90_000.0, uav_id * 100.0, [["FOV", None]]]
            for uav_id in range(16)
        ],
        "known_rois_current_episode": [],
        "observable_sr_current_episode": [],
        "ground_station_position_m": [500.0, 500.0, 0.0],
    }


def _movement_record(step):
    return {
        "episode_index": 199,
        "time_seconds": float(step),
        "action_source": "actor_plus_llm_bias_plus_gaussian_exploration",
        "uavs": [
            {
                "uav_id": uav_id,
                "task_types": [["FOV", "COM", "Search", "Hovering"][(step + uav_id) % 4]],
                "movement_mask": (step + uav_id) % 4 < 2,
                "actor_normalized_action": [
                    0.96 if (step + uav_id) % 7 == 0 else -0.2 + 0.01 * uav_id,
                    -0.8 + ((step + uav_id) % 16) / 10.0,
                    -0.5 + ((step + uav_id) % 10) / 10.0,
                ],
                "executed_normalized_action": [
                    0.99 if (step + uav_id) % 9 == 0 else -0.1 + 0.01 * uav_id,
                    -0.9 + ((step + uav_id) % 18) / 10.0,
                    -0.4 + ((step + uav_id) % 8) / 10.0,
                ],
                "applied_llm_bias": (0.75 * _raw_matrix()[uav_id]).tolist(),
                "observable_targets": [
                    {
                        "kind": "roi",
                        "id": uav_id % 4,
                        "relative_position_m": [10.0 + uav_id, float(step), 2.0],
                    }
                ],
                "movement_energy_j": 100.0 + uav_id + step / 10.0,
            }
            for uav_id in range(16)
        ],
    }


def _routing_record(step, sender):
    state = np.zeros(len(routing_state_feature_names()), dtype=float)
    legal = np.ones(17, dtype=int)
    legal[(sender + 8) % 16] = 0
    references = {
        str(action): float((action + 1) * 100_000)
        for action in np.flatnonzero(legal)
        if action != sender
    }
    if step % 11 == 0 and references:
        references[next(iter(references))] = 0.0
    return {
        "time_seconds": step + sender / 100.0,
        "sender_uav_id": sender,
        "selected_action": (sender + 1) % 16,
        "compact_routing_state": state.tolist(),
        "legal_action_mask": legal.tolist(),
        "reference_capacity_bps_by_legal_action": references,
        "sender_total_backlog_bits": float(50_000 + step * 100 + sender),
        "hol_remaining_deadline_seconds": None if step == 0 else max(0.0, 2.5 - step / 20.0),
        "decision_branch": "epsilon_exploration" if step % 2 == 0 else "safe_q_greedy",
        "zero_legal_prior_mass_fallback": False,
        "exploration_distribution": None,
    }


def _full_feedback(controller, steps=60):
    controller.active_guidance = _active_guidance()
    controller.movement_records = [_movement_record(step) for step in range(steps)]
    controller.routing_records = [
        _routing_record(step, sender)
        for step in range(steps)
        for sender in range(16)
    ]
    controller.complete_episode(episode_index=199, metrics=_metrics())
    return copy.deepcopy(controller.previous_episode_feedback)


def _begin(controller, *, episode=384, transitions=20_000):
    return controller.begin_episode(
        episode_index=episode - 1,
        total_joint_transitions=transitions,
        sigma_episode_start=0.4,
        routing_epsilon=0.5,
        current_observable_initial_state=_initial_state(),
        environment_constants=environment_definitions_td3_only(
            episode_seconds=60, routing_slot_seconds=0.1
        ),
    )


def test_method_registration_cli_metadata_and_old_namespaces_are_unchanged(tmp_path):
    method = MethodSpec.parse("td3_dinkelbach_llm_explorer_td3_only")
    assert method.llm_explorer_enabled
    assert method.llm_explorer_td3_only_enabled
    assert method.llm_explorer_annealed_enabled
    assert not method.llm_explorer_v3_enabled
    parsed = build_parser().parse_args(
        ["td3_dinkelbach_llm_explorer_td3_only", "--smoke"]
    )
    assert parsed.method == method

    controller = LLMExplorerTD3OnlyController(
        run_directory=tmp_path,
        run_identity={"method_id": method.method_id},
        evaluation=True,
    )
    metadata = explorer_metadata_td3_only(controller.config)
    assert controller.root.name == "llm_explorer_td3_only"
    assert metadata["routing_guidance"] == "disabled"
    assert metadata["routing_intervention"].startswith("none_original")
    assert output_schema_td3_only()["required"] == ["td3_bias"]
    assert _explorer_preflight_options(method) == {
        "artifact_directory": "llm_explorer_td3_only",
        "named_routing": False,
        "td3_only": True,
    }

    old = LLMExplorerController(
        run_directory=tmp_path, run_identity={"old": 1}, evaluation=True
    )
    v3 = LLMExplorerV3Controller(
        run_directory=tmp_path, run_identity={"v3": 1}, evaluation=True
    )
    assert old.root.name == "llm_explorer"
    assert v3.root.name == "llm_explorer_v3"
    assert v3.schema_version == "uav-hrl-llm-explorer-v3"
    assert "routing_prior" in v3._templates()[2]
    assert "routing_prior" not in controller._templates()[2].split(
        "Do not output routing_prior", 1
    )[0]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"td3_bias": [[0.0, 0.0, 0.0]] * 15}, "shape"),
        ({"td3_bias": [[0.5, 0.0, 0.0]] * 16}, "bound"),
        ({"td3_bias": [[True, 0.0, 0.0]] * 16}, "booleans"),
        ({"td3_bias": [["0.1", 0.0, 0.0]] * 16}, "JSON numbers"),
        ({"td3_bias": [[0.0, 0.0, 0.0]] * 16, "routing_prior": {}}, "exactly"),
    ],
)
def test_strict_td3_only_validation(payload, message):
    bias, errors = parse_guidance_response_td3_only(json.dumps(payload), 0.4)
    assert bias is None and message in errors["td3"]

    duplicate = '{"td3_bias":[],"td3_bias":[]}'
    bias, errors = parse_guidance_response_td3_only(duplicate, 0.4)
    assert bias is None and "duplicate JSON key" in errors["td3"]
    nonfinite = '{"td3_bias":[' + ",".join(["[NaN,0,0]"] * 16) + "]}"
    bias, errors = parse_guidance_response_td3_only(nonfinite, 0.4)
    assert bias is None and "NaN or Infinity" in errors["td3"]


def test_two_calls_accept_td3_only_scale_w_once_and_cache_replays(tmp_path):
    client = SequenceClient(["Adjust UAV 1; retain the others.", _guidance()])
    controller = LLMExplorerTD3OnlyController(
        run_directory=tmp_path,
        run_identity={"run": "active"},
        client=client,
    )
    controller.previous_episode_feedback = _full_feedback(
        LLMExplorerTD3OnlyController(
            run_directory=tmp_path / "source",
            run_identity={"run": "source"},
            evaluation=True,
        ),
        steps=2,
    )
    result = _begin(controller)
    assert len(client.calls) == 2
    assert result["enabled"] and not result["td3_fallback"]
    assert result["routing_prior"] is None
    assert not result["routing_fallback"]
    assert result["routing_reason"] == "routing_guidance_disabled"
    assert controller.statistics["routing_fallback_episodes_total"] == 0
    assert result["annealing_weight"] == pytest.approx(0.5)
    np.testing.assert_allclose(result["td3_bias_raw"], _raw_matrix())
    np.testing.assert_allclose(result["td3_bias_effective"], 0.5 * _raw_matrix())
    assert all("{{" not in call["prompt"] for call in client.calls)
    assert all("{{" not in call["system_prompt"] for call in client.calls)
    completed = json.loads(
        (
            tmp_path
            / "llm_explorer_td3_only"
            / "episode_000384"
            / "guidance"
            / "completed.json"
        ).read_text()
    )
    assert set(json.loads(completed["content"])) == {"td3_bias"}
    assert completed["routing_source_attempt"] is None
    validation = json.loads(
        (
            tmp_path
            / "llm_explorer_td3_only"
            / "episode_000384"
            / "guidance"
            / "attempt_01"
            / "validation.json"
        ).read_text()
    )
    assert validation["routing_guidance_required"] is False
    assert validation["accepted_routing_this_attempt"] is None

    replay_client = SequenceClient([])
    replay = LLMExplorerTD3OnlyController(
        run_directory=tmp_path,
        run_identity={"run": "active"},
        client=replay_client,
    )
    replay.previous_episode_feedback = copy.deepcopy(controller.previous_episode_feedback)
    restored = _begin(replay)
    assert restored["reused"] and replay_client.calls == []
    assert restored["routing_prior"] is None
    np.testing.assert_allclose(restored["td3_bias_effective"], 0.5 * _raw_matrix())


def test_previous_raw_bias_is_complete_and_unknown_is_not_zero(tmp_path):
    previous = _full_feedback(
        LLMExplorerTD3OnlyController(
            run_directory=tmp_path / "source",
            run_identity={"run": "source"},
            evaluation=True,
        ),
        steps=2,
    )
    client = SequenceClient(
        ["Adjust UAV 1 only; retain every other direction.", _guidance(np.zeros((16, 3)))]
    )
    controller = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "target",
        run_identity={"run": "target"},
        client=client,
    )
    controller.previous_episode_feedback = previous
    _begin(controller, episode=201, transitions=12_000)
    guidance_prompt = client.calls[1]["prompt"]
    marker = "Previous episode's complete recorded raw TD3 bias:\n"
    raw_payload = json.loads(guidance_prompt.split(marker, 1)[1].splitlines()[0])
    assert raw_payload["uav_id_order"] == list(range(16))
    assert raw_payload["columns"] == ["speed", "heading", "vertical"]
    np.testing.assert_allclose(raw_payload["rows"], _raw_matrix())
    assert "before multiplication by its annealing weight" in guidance_prompt
    assert "null rows value means the raw bias is unknown, not zero" in guidance_prompt

    unknown = {
        "schema_version": "unrelated-old-record",
        "movement_summaries": {},
        "applied_guidance": {"td3_bias": _raw_matrix().tolist()},
    }
    payload = recorded_raw_td3_bias(unknown)
    assert payload["source"] == "unknown_not_recorded"
    assert payload["rows"] is None

    disabled = {
        "schema_version": LLM_EXPLORER_TD3_ONLY_VERSION,
        "movement_summaries": {
            f"UAV_{uav_id}": {"raw_bias": [0.0, 0.0, 0.0]}
            for uav_id in range(16)
        },
        "applied_guidance": {
            "enabled": False,
            "td3_fallback": False,
            "td3_bias_raw": np.zeros((16, 3)).tolist(),
        },
    }
    payload = recorded_raw_td3_bias(disabled)
    assert payload["source"] == "unknown_no_valid_previous_intervention"
    assert payload["rows"] is None


def test_warmup_evaluation_and_episode_600_make_no_calls(tmp_path):
    never = SequenceClient([AssertionError("must not call")])
    warmup = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "warmup",
        run_identity={"run": "warmup"},
        client=never,
    )
    warmup.previous_episode_feedback = {"metrics": {}}
    result = _begin(warmup, episode=167, transitions=9_960)
    assert result["td3_reason"] == "warmup" and not result["td3_fallback"]

    cutoff = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "cutoff",
        run_identity={"run": "cutoff"},
        client=never,
    )
    cutoff.previous_episode_feedback = {"metrics": {}}
    result = _begin(cutoff, episode=600, transitions=35_940)
    assert result["td3_reason"] == "annealing_finished"

    evaluation = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "evaluation",
        run_identity={"run": "evaluation"},
        evaluation=True,
        client=never,
    )
    evaluation.previous_episode_feedback = {"metrics": {}}
    result = _begin(evaluation, episode=384, transitions=20_000)
    assert result["td3_reason"] == "evaluation_disabled"
    assert never.calls == []
    assert warmup.statistics["fallback_episodes_total"] == 0
    assert cutoff.statistics["fallback_episodes_total"] == 0


def test_retry_and_failures_count_td3_only_not_routing(tmp_path):
    source = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "source",
        run_identity={"run": "source"},
        evaluation=True,
    )
    previous = _full_feedback(source, steps=1)
    retry_client = SequenceClient(
        ["Supported analysis.", '{"td3_bias":[],"routing_prior":{}}', _guidance()]
    )
    retry = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "retry",
        run_identity={"run": "retry"},
        client=retry_client,
    )
    retry.previous_episode_feedback = previous
    result = _begin(retry)
    assert result["enabled"] and len(retry_client.calls) == 3
    assert retry.statistics["failed_attempts_total"] == 1
    assert retry.statistics["retry_success_total"] == 1
    assert retry.statistics["routing_fallback_episodes_total"] == 0
    assert "response must contain exactly td3_bias" in retry_client.calls[2]["prompt"]

    analysis_fail_client = SequenceClient([RuntimeError("offline"), RuntimeError("offline")])
    failed = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "failed",
        run_identity={"run": "failed"},
        client=analysis_fail_client,
    )
    failed.previous_episode_feedback = previous
    result = _begin(failed)
    assert not result["enabled"] and result["td3_fallback"]
    assert not result["routing_fallback"]
    assert failed.statistics["td3_fallback_episodes_total"] == 1
    assert failed.statistics["routing_fallback_episodes_total"] == 0

    guidance_fail_client = SequenceClient(["Supported analysis.", "{}", "{}"])
    guidance_failed = LLMExplorerTD3OnlyController(
        run_directory=tmp_path / "guidance-failed",
        run_identity={"run": "guidance-failed"},
        client=guidance_fail_client,
    )
    guidance_failed.previous_episode_feedback = previous
    result = _begin(guidance_failed)
    assert result["enabled"] and result["td3_fallback"]
    assert result["routing_prior"] is None and not result["routing_fallback"]
    assert guidance_failed.statistics["td3_fallback_episodes_total"] == 1
    assert guidance_failed.statistics["routing_fallback_episodes_total"] == 0


def test_contaminated_routing_prior_is_never_forwarded_to_safe_ddqn():
    method = MethodSpec.parse("td3_dinkelbach_llm_explorer_td3_only")
    contaminated = {
        "enabled": True,
        "routing_fallback": False,
        "routing_prior": np.full((16, 17), 1.0 / 17.0),
    }
    assert _llm_routing_exploration_prior(method, object(), contaminated) is None

    v3 = MethodSpec.parse("td3_dinkelbach_llm_explorer_v3")
    assert _llm_routing_exploration_prior(v3, object(), contaminated) is contaminated[
        "routing_prior"
    ]

    class RecordingSafeDDQN:
        def __init__(self):
            self.calls = []

        def select_action(self, **kwargs):
            self.calls.append(kwargs)
            action = int(kwargs["uav_id"])
            if kwargs.get("return_diagnostics"):
                return action, {
                    "decision_branch": "epsilon_exploration",
                    "zero_legal_prior_mass_fallback": False,
                    "exploration_distribution": None,
                }
            return action

    agent = RecordingSafeDDQN()
    states = {0: np.zeros(len(routing_state_feature_names()), dtype=np.float32)}
    masks = {0: np.ones(17, dtype=bool)}
    actions, diagnostics = _select_routing_actions(
        agent,
        states,
        masks,
        epsilon=0.5,
        exploration_prior=_llm_routing_exploration_prior(
            method, object(), contaminated
        ),
        exploration_prior_weight=0.0,
        return_diagnostics=True,
        capture_diagnostics=True,
    )
    assert actions == {0: 0}
    assert diagnostics[0]["exploration_distribution"] is None
    assert "exploration_prior" not in agent.calls[0]
    assert "exploration_prior_weight" not in agent.calls[0]


def test_communication_summary_uses_full_real_records_and_clear_unknowns():
    records = [_routing_record(step, sender) for step in range(3) for sender in range(16)]
    summary = communication_bottleneck_summary(records, _metrics())
    system = summary["system"]
    assert system["recorded_routing_decisions"] == 48
    assert system["observed_sender_count_out_of_16"] == [16, 16]
    assert system["known_backlog_decisions"] == 48
    assert system["positive_backlog_ratio_of_known"] == 1.0
    assert system["missing_hol_deadline_decisions"] == 16
    assert system["known_hol_deadline_decisions"] == 32
    assert system["valid_legal_mask_decisions"] == 48
    assert system["system_qos_violation_count"] == 4
    assert summary["per_sender_rows"][0][1] == 3
    assert "not realized throughput" in summary["units_and_definitions"][
        "reference_capacity"
    ]
    assert "per-UAV timely GS-delivery attribution" in summary["unavailable"]
    assert "selected_action" not in json.dumps(summary)


def test_movement_summary_is_full_record_deterministic_and_distinguishes_biases():
    records = [_movement_record(step) for step in range(12)]
    guidance = _active_guidance(weight=0.75)
    first, first_cases = movement_episode_summary_td3_only(records, guidance)
    second, second_cases = movement_episode_summary_td3_only(records, guidance)
    assert first == second and first_cases == second_cases
    assert len(first_cases) <= 3
    uav0 = first["UAV_0"]
    assert uav0["movement_mask_true_count"] == 6
    assert uav0["td3_controlled_actor_decision_count"] == 6
    assert uav0["raw_bias"] == pytest.approx(_raw_matrix()[0].tolist())
    assert uav0["effective_bias"] == pytest.approx(
        (0.75 * _raw_matrix()[0]).tolist()
    )
    assert uav0["actually_applied_effective_bias_count"] == 6
    assert uav0["executed_normalized_action_by_component"][0]["count"] == 6
    assert uav0["actor_abs_ge_0.95_ratio_by_component"] is not None


def test_full_scale_prompts_obey_budget_and_keep_complete_context(tmp_path):
    encoder = tiktoken.get_encoding("o200k_base")
    analysis_seed = (
        "Evidence is observational; cite denominators and separate movement from routing effects. "
    ) * 500
    near_limit_analysis = encoder.decode(encoder.encode(analysis_seed)[:1350])
    client = SequenceClient([near_limit_analysis, "{}", _guidance(np.zeros((16, 3)))])
    controller = LLMExplorerTD3OnlyController(
        run_directory=tmp_path,
        run_identity={"run": "full-scale"},
        client=client,
    )
    _full_feedback(controller, steps=60)
    raw_feedback = json.loads(
        (tmp_path / "llm_explorer_td3_only" / "episode_000200" / "feedback.json").read_text()
    )
    assert len(raw_feedback["movement_decisions"]) == 60
    assert len(raw_feedback["routing_decisions"]) == 960
    assert raw_feedback["communication_bottleneck_summary"]["system"][
        "recorded_routing_decisions"
    ] == 960

    result = _begin(controller, episode=201, transitions=12_000)
    assert result["enabled"] and len(client.calls) == 3
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
    analysis_prompt = client.calls[0]["prompt"]
    guidance_prompt = client.calls[1]["prompt"]
    assert "Compact communication bottleneck summary relevant to movement" in analysis_prompt
    assert "per-UAV timely GS-delivery attribution" in analysis_prompt
    assert "selected_action" not in analysis_prompt
    assert "hidden_roi_count_disclosed\":false" in analysis_prompt
    assert "known_rois_current_episode\":[]" in analysis_prompt
    assert '"rows":[[0.001,0.002,0.003]' in guidance_prompt
    assert "routing_prior\"" not in guidance_prompt


def test_cache_state_schema_isolated_and_windows_preflight_covers_800(tmp_path):
    controller = LLMExplorerTD3OnlyController(
        run_directory=tmp_path,
        run_identity={"run": "state"},
        evaluation=True,
        rng_seed=123,
    )
    controller.previous_episode_feedback = {"metrics": {"episode": 2}}
    controller.rng.random(5)
    state = controller.state_dict()
    restored = LLMExplorerTD3OnlyController(
        run_directory=tmp_path,
        run_identity={"run": "state"},
        evaluation=True,
        rng_seed=999,
    )
    restored.load_state_dict(state)
    assert restored.previous_episode_feedback == controller.previous_episode_feedback
    assert restored.rng.bit_generator.state == controller.rng.bit_generator.state

    v3 = LLMExplorerV3Controller(
        run_directory=tmp_path / "v3",
        run_identity={"run": "state"},
        evaluation=True,
    )
    with pytest.raises(ExplorerValidationError, match="version is incompatible"):
        v3.load_state_dict(state)

    identity = controller._cache_identity(
        episode_index=200,
        stage="guidance",
        system_prompt="system",
        user_prompt="prompt",
    )
    assert identity["schema_version"] == LLM_EXPLORER_TD3_ONLY_VERSION
    assert identity["routing_guidance"] == "disabled"
    assert identity["guidance_context_version"].startswith("self-contained")

    report = llm_explorer_path_preflight(
        Path(r"C:\uav_llm_runs")
        / "td3_dinkelbach_llm_explorer_td3_only"
        / "run-seed20260817-abcdef0-20261017T000000000000Z-12345678",
        maximum_episode=800,
        artifact_directory="llm_explorer_td3_only",
        named_routing=False,
        td3_only=True,
    )
    assert report["td3_only"]
    assert report["longest"]["characters"] <= report["windows_limit_characters"]
    assert any("attempt_02" in path["relative_path"] for path in report["paths"])
    assert any(".response_metadata.json.tmp" in path["relative_path"] for path in report["paths"])


def test_mock_short_training_integration_completes_without_paid_api(tmp_path):
    result = train(
        TrainingConfig(
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
            random_seed=20260817,
        ),
        method_spec=MethodSpec.parse("td3_dinkelbach_llm_explorer_td3_only"),
        llm_explorer_mock=True,
    )
    assert result["episodes_run"] == 1
    assert result["llm_explorer"]["configuration"]["routing_guidance"] == "disabled"
    assert result["llm_explorer"]["statistics"]["requests_total"] == 0
    assert result["llm_explorer"]["statistics"]["routing_fallback_episodes_total"] == 0
