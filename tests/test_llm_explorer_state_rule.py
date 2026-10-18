import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, _llm_routing_exploration_prior, train
from llm_explorer import ExplorerValidationError, PromptBudgetError, llm_explorer_path_preflight
from llm_explorer_state_rule import (
    STATE_RULE_SCHEMA_VERSION,
    build_runtime_feature_rows,
    compile_state_rule,
    evaluate_rule_for_joint_action,
    parse_and_compile_state_rule,
    runtime_feature_catalog,
)
from llm_state_rule_explorer import (
    LLMStateRuleConfig,
    LLMStateRuleController,
    environment_definitions_state_rule,
    explorer_metadata_state_rule,
    slot_service_summary,
)
from replay_auxiliary import TASK_TYPE_ENCODING, empty_snapshot
from rng_contract import NamedRNGStreams
from run_experiment import _explorer_preflight_options, build_parser
from td3 import TD3


def rule(speed=0.0, heading=0.0, vertical=0.0):
    return {
        "schema_version": STATE_RULE_SCHEMA_VERSION,
        "speed": speed,
        "heading": heading,
        "vertical": vertical,
    }


def field(name):
    return {"field": name}


def op(name, *args):
    return {"op": name, "args": list(args)}


def balanced_add(leaves):
    values = [float(index) for index in range(leaves)]
    while len(values) > 1:
        next_values = []
        for index in range(0, len(values), 2):
            next_values.append(
                values[index]
                if index + 1 == len(values)
                else op("add", values[index], values[index + 1])
            )
        values = next_values
    return values[0]


class Uav:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_position(self):
        return self.position.copy()


def runtime_fixture():
    env = SimpleNamespace(
        uav_dict={
            uav_id: Uav((10.0 * uav_id, 5.0 * uav_id, 80.0 + uav_id % 3))
            for uav_id in range(16)
        }
    )
    snapshot = empty_snapshot()
    snapshot["snapshot_valid"][0] = True
    snapshot["snapshot_time_s"][0] = 12.0
    snapshot["uav_queue_valid"][:] = True
    snapshot["uav_queue_empty"][:] = True
    snapshot["u2g_link_valid"][:] = True
    snapshot["u2g_distance_m"][:] = np.arange(16) + 100.0
    snapshot["u2g_in_range"][:] = True
    snapshot["u2g_reference_capacity_mbps"][:] = np.arange(16) + 20.0
    snapshot["u2u_link_valid"][:] = ~np.eye(16, dtype=bool)
    snapshot["u2u_in_range"][:] = ~np.eye(16, dtype=bool)
    snapshot["u2u_distance_m"][:] = 100.0
    snapshot["u2u_reference_capacity_mbps"][:] = 1.0
    np.fill_diagonal(snapshot["u2u_reference_capacity_mbps"], 0.0)

    snapshot["roi_id"][0] = 3
    snapshot["roi_observable"][0] = True
    snapshot["roi_position_m"][0] = [100.0, 0.0, 0.0]
    snapshot["task_pair_valid"][0, 0] = True
    snapshot["task_target_valid"][0, 0] = True
    snapshot["task_type"][0, 0] = TASK_TYPE_ENCODING["FOV"]
    snapshot["task_target_id"][0, 0] = 3
    snapshot["vs_pair_valid"][0] = True
    snapshot["vs_roi_id"][0] = 3
    snapshot["vs_horizontal_distance_m"][0] = 100.0
    snapshot["vs_c9_margin_m"][0] = -5.0
    snapshot["vs_c10_geometry_valid"][0] = True
    snapshot["vs_c10_margin_m"][0] = 2.0
    snapshot["vs_capture_valid"][0] = True
    snapshot["vs_coverage_ratio"][0] = 0.6
    snapshot["vs_image_quantity"][0] = 0.4

    snapshot["sr_observable"][1] = True
    snapshot["sr_position_m"][1] = [160.0, 20.0, 0.0]
    snapshot["sr_queue_valid"][1] = True
    snapshot["sr_queue_empty"][1] = False
    snapshot["sr_backlog_bits"][1] = 512.0
    snapshot["sr_packet_count"][1] = 2
    snapshot["sr_hol_deadline_valid"][1] = True
    snapshot["sr_hol_remaining_deadline_s"][1] = 0.004
    snapshot["task_pair_valid"][1, 0] = True
    snapshot["task_target_valid"][1, 0] = True
    snapshot["task_type"][1, 0] = TASK_TYPE_ENCODING["COM"]
    snapshot["task_target_id"][1, 0] = 1
    snapshot["s2u_link_valid"][1, 1] = True
    snapshot["s2u_distance_m"][1, 1] = 170.0
    snapshot["s2u_in_range"][1, 1] = True
    snapshot["s2u_reference_capacity_mbps"][1, 1] = 37.0
    return env, snapshot


def test_registry_cli_metadata_namespace_and_path_contract():
    method = MethodSpec.parse("td3_dinkelbach_llm_explorer_state_rule")
    assert method.llm_explorer_enabled
    assert method.llm_explorer_state_rule_enabled
    assert method.llm_explorer_movement_only_enabled
    assert not method.llm_explorer_annealed_enabled
    assert not method.llm_explorer_td3_only_enabled
    parsed = build_parser().parse_args(
        ["td3_dinkelbach_llm_explorer_state_rule", "--smoke"]
    )
    assert parsed.method == method
    assert _explorer_preflight_options(method) == {
        "artifact_directory": "llm_rule",
        "named_routing": False,
        "td3_only": False,
        "state_rule": True,
    }
    metadata = explorer_metadata_state_rule(LLMStateRuleConfig())
    assert metadata["guidance_weight_w"] == 1.0
    assert metadata["annealing_end_episode"] is None
    assert metadata["legacy_input_hard_limit_tokens"] is None
    report = llm_explorer_path_preflight(
        r"C:\uav_llm_runs\td3_dinkelbach_llm_explorer_state_rule\seed_20260817_abcdef",
        maximum_episode=1500,
        **_explorer_preflight_options(method),
    )
    assert report["state_rule"] is True
    assert report["representative_state_rule_ast_nodes"] == 99
    assert report["guidance_reserved_output_tokens"] == 4096
    assert report["representative_full_guidance_output_tokens"] < 4096
    assert report["longest"]["characters"] <= report["windows_limit_characters"]
    largest_rule = compile_state_rule(
        rule(
            speed=balanced_add(17),
            heading=balanced_add(17),
            vertical=balanced_add(17),
        )
    )
    assert sum(largest_rule.component_node_counts) == 99


@pytest.mark.parametrize(
    "expression,features,expected",
    [
        (op("add", 2, 3), {}, 5.0),
        (op("sub", 2, 3), {}, -1.0),
        (op("mul", 2, 3), {}, 6.0),
        (op("safe_div", 2, 0), {}, 0.0),
        (op("neg", 2), {}, -2.0),
        (op("abs", -2), {}, 2.0),
        (op("tanh", 0), {}, 0.0),
        (op("min", 2, 3), {}, 2.0),
        (op("max", 2, 3), {}, 3.0),
        (op("clip", 3, -1, 1), {}, 1.0),
        (op("sin", math.pi / 2), {}, 1.0),
        (op("cos", math.pi), {}, -1.0),
        (op("atan2", 0, 0), {}, 0.0),
        (op("wrap_angle", math.pi), {}, -math.pi),
        (op("if", op("gt", field("uav_x_m"), 2), 4, 5), {"uav_x_m": 3}, 4.0),
    ],
)
def test_whitelisted_operations(expression, features, expected):
    compiled = compile_state_rule(rule(speed=expression))
    base = {
        name: (False if spec["type"] == "boolean" else 0.0)
        for name, spec in {
            item["name"]: item
            for item in runtime_feature_catalog()["fields"]
        }.items()
    }
    base.update(features)
    value, _ = compiled.evaluate(base)
    assert value[0] == pytest.approx(expected)


def test_schema_duplicate_keys_arity_types_fields_nonfinite_complexity_and_depth():
    with pytest.raises(ExplorerValidationError, match="duplicate JSON key"):
        parse_and_compile_state_rule(
            '{"schema_version":"td3-state-bias-rule-v1","speed":0,"speed":1,"heading":0,"vertical":0}'
        )
    bad_values = [
        ({"schema_version": STATE_RULE_SCHEMA_VERSION, "speed": 0}, "root keys"),
        (rule(speed={"field": "made_up"}), "unknown formal field"),
        (rule(speed={"op": "add", "args": [1]}), "requires exactly"),
        (rule(speed={"op": "eval", "args": []}), "unsupported operation"),
        (rule(speed=True), "Boolean constants"),
        (rule(speed=float("nan")), "finite"),
        (rule(speed=op("add", field("movement_mask"), 1)), "numeric arguments"),
        (rule(speed=op("if", field("actor_speed"), 1, 0)), "Boolean condition"),
        (rule(speed=op("clip", 0, 1, -1)), "lower must be"),
        (rule(speed=balanced_add(32)), "node count"),
        (rule(speed=balanced_add(18), heading=balanced_add(18), vertical=balanced_add(18)), "total node count"),
    ]
    deep = 0.0
    for _ in range(8):
        deep = op("neg", deep)
    bad_values.append((rule(speed=deep), "depth"))
    for value, message in bad_values:
        with pytest.raises(ExplorerValidationError, match=message):
            compile_state_rule(value)


def test_lazy_if_and_short_circuit_skip_runtime_invalid_clip():
    invalid = op("clip", 0.0, field("actor_speed"), -1.0)
    compiled = compile_state_rule(
        rule(
            speed=op("if", field("movement_mask"), invalid, 0.25),
            heading=op(
                "if",
                op("and", field("movement_mask"), op("gt", invalid, 0.0)),
                1.0,
                0.0,
            ),
            vertical=op(
                "if",
                op("or", op("not", field("movement_mask")), op("gt", invalid, 0.0)),
                0.5,
                0.0,
            ),
        )
    )
    features = {name: (False if kind == "boolean" else 0.0) for name, kind in __import__("llm_explorer_state_rule").FIELD_TYPES.items()}
    features.update(movement_mask=False, actor_speed=1.0)
    value, trace = compiled.evaluate(features)
    np.testing.assert_allclose(value, [0.25, 0.0, 0.5])
    assert trace["short_circuits"] == 2


def test_heading_periodicity_uses_radians_and_normalized_output():
    expression = op(
        "safe_div",
        op(
            "wrap_angle",
            op(
                "sub",
                field("fov_target_bearing_rad"),
                op("mul", field("actor_heading"), math.pi),
            ),
        ),
        math.pi,
    )
    compiled = compile_state_rule(rule(heading=expression))
    features = {name: (False if kind == "boolean" else 0.0) for name, kind in __import__("llm_explorer_state_rule").FIELD_TYPES.items()}
    features.update(actor_heading=0.99, fov_target_bearing_rad=-0.99 * math.pi)
    value, _ = compiled.evaluate(features)
    assert value[1] == pytest.approx(0.02)


def test_latest_features_role_change_mask_transition_and_current_capacity():
    env, snapshot = runtime_fixture()
    actors = np.zeros((16, 3), dtype=float)
    first = build_runtime_feature_rows(env, snapshot, np.zeros(16, dtype=bool), actors)
    assert first[0]["has_fov_task"] and not first[0]["movement_mask"]
    assert first[0]["fov_relative_x_m"] == pytest.approx(100.0)
    assert first[1]["has_com_task"]
    assert first[1]["com_reference_capacity_mbps"] == pytest.approx(37.0)
    assert first[1]["com_sr_backlog_bits"] == pytest.approx(512.0)

    snapshot["task_type"][0, 0] = TASK_TYPE_ENCODING["COM"]
    snapshot["task_target_id"][0, 0] = 1
    snapshot["s2u_link_valid"][1, 0] = True
    snapshot["s2u_reference_capacity_mbps"][1, 0] = 11.0
    snapshot["s2u_distance_m"][1, 0] = 160.0
    snapshot["s2u_in_range"][1, 0] = True
    mask = np.zeros(16, dtype=bool)
    mask[0] = True
    second = build_runtime_feature_rows(env, snapshot, mask, actors)
    assert second[0]["movement_mask"] and second[0]["has_com_task"]
    assert not second[0]["has_fov_task"]
    assert second[0]["com_reference_capacity_mbps"] == pytest.approx(11.0)
    snapshot["s2u_reference_capacity_mbps"][1, 0] = 22.0
    third = build_runtime_feature_rows(env, snapshot, mask, actors)
    assert third[0]["com_reference_capacity_mbps"] == pytest.approx(22.0)
    snapshot["s2u_link_valid"][1, 0] = False
    fourth = build_runtime_feature_rows(env, snapshot, mask, actors)
    assert fourth[0]["com_target_valid"]
    assert fourth[0]["com_relative_x_m"] == pytest.approx(160.0)
    assert not fourth[0]["com_reference_capacity_valid"]
    assert fourth[0]["com_reference_capacity_mbps"] == 0.0


def test_same_rule_changes_with_latest_state_and_clips_once_without_sigma_scaling():
    compiled = compile_state_rule(
        rule(
            speed=op("mul", 2.0, field("actor_speed")),
            heading=op("if", field("has_fov_task"), 0.3, -0.3),
            vertical=0.05,
        )
    )
    rows = [
        {name: (False if kind == "boolean" else 0.0) for name, kind in __import__("llm_explorer_state_rule").FIELD_TYPES.items()}
        for _ in range(16)
    ]
    rows[0].update(actor_speed=0.4, has_fov_task=True, movement_mask=True)
    rows[1].update(actor_speed=-0.2, has_fov_task=False, movement_mask=True)
    mask = np.zeros(16, dtype=bool)
    mask[:2] = True
    bias, diagnostics = evaluate_rule_for_joint_action(
        compiled, rows, mask, sigma_episode_start=0.1, weight=1.0
    )
    matrix = bias.reshape(16, 3)
    np.testing.assert_allclose(matrix[0], [0.1, 0.1, 0.05])
    np.testing.assert_allclose(matrix[1], [-0.1, -0.1, 0.05])
    np.testing.assert_allclose(matrix[2:], 0.0)
    assert diagnostics[0]["raw_bias"] == pytest.approx([0.8, 0.3, 0.05])
    assert diagnostics[0]["clipped_components"] == 2


def test_runtime_failure_is_per_uav_and_zero_fallback():
    compiled = compile_state_rule(
        rule(speed=op("clip", 0.0, field("actor_speed"), field("actor_vertical")))
    )
    rows = [
        {name: (False if kind == "boolean" else 0.0) for name, kind in __import__("llm_explorer_state_rule").FIELD_TYPES.items()}
        for _ in range(16)
    ]
    rows[0].update(actor_speed=1.0, actor_vertical=-1.0)
    rows[1].update(actor_speed=-1.0, actor_vertical=1.0)
    mask = np.zeros(16, dtype=bool)
    mask[:2] = True
    bias, diagnostics = evaluate_rule_for_joint_action(
        compiled, rows, mask, sigma_episode_start=0.2
    )
    assert diagnostics[0]["runtime_failure"]
    np.testing.assert_allclose(bias.reshape(16, 3)[0], 0.0)
    assert diagnostics[1]["runtime_failure"] is None


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
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "finish_reason": "stop",
        }


def previous_feedback():
    return {
        "metrics": {"energy_efficiency_bit_per_j": 10.0},
        "observed_task_and_traffic_summary": {"FOV": "observed"},
        "movement_and_rule_application_summaries": {"totals": {"controlled_decisions": 3}},
        "fov_com_and_slot_service_summaries": {"actual_within_slot_service": {"slot_count": 4}},
        "representative_cases": [],
        "applied_guidance": {
            "enabled": True,
            "td3_fallback": False,
            "source_episode": 167,
            "rule": rule(speed=0.01),
        },
    }


def test_two_stage_prompt_is_self_contained_cache_isolated_and_resume_restores_rule(tmp_path):
    generated = rule(speed=op("if", field("has_fov_task"), 0.2, 0.0))
    client = SequenceClient(["complete analysis with {braces}", json.dumps(generated)])
    identity = {"method_id": "td3_dinkelbach_llm_explorer_state_rule", "seed": 1}
    controller = LLMStateRuleController(
        run_directory=tmp_path,
        run_identity=identity,
        client=client,
    )
    controller.previous_episode_feedback = previous_feedback()
    result = controller.begin_episode(
        episode_index=700,
        total_joint_transitions=10_000,
        sigma_episode_start=0.12,
        routing_epsilon=0.5,
        current_observable_initial_state={"known": "opening state"},
        environment_constants=environment_definitions_state_rule(
            episode_seconds=60,
            routing_slot_seconds=0.25,
            task_deadlines_seconds={"FOV": 0.015, "COM": 0.010},
        ),
    )
    assert len(client.calls) == 2
    assert result["rule"] == generated
    assert result["guidance_weight_w"] == 1.0
    assert result["source_episode"] == 701
    rule_prompt = client.calls[1]["prompt"]
    assert "complete analysis with {braces}" in rule_prompt
    assert "actor_speed" in rule_prompt
    assert '"speed":0.01' in rule_prompt
    assert "opening state" in rule_prompt
    assert "{{" not in (tmp_path / "llm_rule" / "episode_000701" / "guidance" / "user_prompt.txt").read_text(encoding="utf-8")

    state = controller.state_dict()
    restored = LLMStateRuleController(
        run_directory=None,
        run_identity=identity,
        config=controller.config,
        evaluation=True,
    )
    restored.evaluation = False
    restored.load_state_dict(state)
    assert restored.active_guidance["rule"] == generated
    assert restored._compiled_rule is not None

    wrong = copy.deepcopy(state)
    wrong["schema_version"] = "uav-hrl-llm-explorer-td3-only-v1"
    with pytest.raises(ExplorerValidationError, match="incompatible"):
        restored.load_state_dict(wrong)


def test_api_validation_fallback_zero_rule_and_retry_budget_recheck(tmp_path):
    client = SequenceClient(
        [
            "analysis",
            "not json",
            json.dumps(rule(speed=0.0)),
        ]
    )
    controller = LLMStateRuleController(
        run_directory=tmp_path,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=client,
    )
    controller.previous_episode_feedback = previous_feedback()
    result = controller.begin_episode(
        episode_index=200,
        total_joint_transitions=10_000,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={},
        environment_constants=environment_definitions_state_rule(
            episode_seconds=60, routing_slot_seconds=0.25
        ),
    )
    assert len(client.calls) == 3
    assert result["td3_fallback"] is False
    assert result["rule"] == rule(speed=0.0)
    retry_budget = tmp_path / "llm_rule" / "episode_000201" / "guidance" / "attempt_02" / "token_budget.json"
    assert retry_budget.is_file()
    assert controller.statistics["failure_counts"]["content_validation"] == 1

    failing = SequenceClient(["analysis", "bad", "also bad"])
    fallback = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=failing,
    )
    fallback.previous_episode_feedback = previous_feedback()
    value = fallback.begin_episode(
        episode_index=201,
        total_joint_transitions=10_000,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={},
        environment_constants=environment_definitions_state_rule(
            episode_seconds=60, routing_slot_seconds=0.25
        ),
    )
    assert value["td3_fallback"] is True
    assert value["rule"] is None


def test_warmup_and_evaluation_make_no_calls_and_episode_600_has_no_cutoff():
    client = SequenceClient([])
    controller = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=client,
        lazy_client=True,
    )
    warmup = controller.begin_episode(
        episode_index=500,
        total_joint_transitions=9_999,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={},
        environment_constants={},
    )
    assert warmup["td3_reason"] == "warmup"
    assert not client.calls

    evaluation = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=client,
        evaluation=True,
    )
    disabled = evaluation.begin_episode(
        episode_index=700,
        total_joint_transitions=99_999,
        sigma_episode_start=0.1,
    )
    assert disabled["td3_reason"] == "evaluation_disabled"
    assert not client.calls

    active_client = SequenceClient(["analysis", json.dumps(rule())])
    active = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=active_client,
    )
    active.previous_episode_feedback = previous_feedback()
    result = active.begin_episode(
        episode_index=700,
        total_joint_transitions=50_000,
        sigma_episode_start=0.1,
        routing_epsilon=0.2,
        current_observable_initial_state={},
        environment_constants=environment_definitions_state_rule(
            episode_seconds=60, routing_slot_seconds=0.25
        ),
    )
    assert result["enabled"] and result["guidance_weight_w"] == 1.0


def test_token_budget_has_no_20000_limit_and_rejects_context_overflow_before_send():
    with pytest.raises(ValueError, match="4,096-token safety margin"):
        LLMStateRuleConfig(safety_margin_tokens=2048)
    controller = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        evaluation=True,
    )
    allowed = controller._prompt_budget(
        episode_index=0,
        stage="analysis",
        attempt=1,
        system_prompt="system",
        user_prompt="evidence " * 25_000,
        max_output_tokens=4096,
        attempt_root=None,
    )
    assert allowed["input_tokens"] > 20_000
    assert allowed["fits"]
    with pytest.raises(PromptBudgetError):
        controller._prompt_budget(
            episode_index=0,
            stage="guidance",
            attempt=2,
            system_prompt="system",
            user_prompt="evidence " * 140_000,
            max_output_tokens=4096,
            attempt_root=None,
        )
    assert controller.statistics["requests_total"] == 0
    assert controller.statistics["failure_counts"]["local_preflight"] == 1


def test_td3_provider_uses_one_actor_call_and_preserves_gaussian_rng_order():
    streams_a = NamedRNGStreams(77)
    streams_b = NamedRNGStreams(77)
    guided = TD3(2, 48, 1.0, rng_streams=streams_a)
    baseline = TD3(2, 48, 1.0, rng_streams=streams_b)
    baseline.actor.load_state_dict(copy.deepcopy(guided.actor.state_dict()))
    calls = []
    hooks = []
    hooks.append(guided.actor.register_forward_hook(lambda *args: calls.append("actor")))
    provided = []

    def provider(actor_action):
        provided.append(actor_action.copy())
        return np.zeros(48, dtype=np.float32)

    state = np.asarray([0.2, -0.1], dtype=np.float32)
    guided_action, diagnostics = guided.select_action(
        state,
        add_noise=True,
        noise_std=0.2,
        mean_bias_provider=provider,
        mean_bias_bound=0.2,
        return_diagnostics=True,
    )
    baseline_action, baseline_diagnostics = baseline.select_action(
        state,
        add_noise=True,
        noise_std=0.2,
        return_diagnostics=True,
    )
    hooks[0].remove()
    assert calls == ["actor"]
    assert len(provided) == 1
    np.testing.assert_allclose(provided[0], diagnostics["actor_action"])
    np.testing.assert_allclose(
        diagnostics["gaussian_noise"], baseline_diagnostics["gaussian_noise"]
    )
    np.testing.assert_allclose(guided_action, baseline_action)


def test_safe_ddqn_ignores_even_contaminated_state_rule_guidance():
    method = MethodSpec.parse("td3_dinkelbach_llm_explorer_state_rule")
    contaminated = {
        "enabled": True,
        "routing_fallback": False,
        "routing_prior": np.ones((16, 17)),
    }
    assert _llm_routing_exploration_prior(method, object(), contaminated) is None


def test_slot_service_summary_does_not_equate_empty_slot_start_with_no_service():
    summary = slot_service_summary(
        [
            {
                "start_of_slot_nonempty_sender_ids": [],
                "serviced_packet_ids": [10],
                "s2u_completed_packets": 1,
                "uav_relay_completions": 2,
                "gs_delivery_outcomes": 1,
                "same_slot_multi_hop_packets": 1,
                "transmitted_bits_by_link": {"S2U:0:1": 256.0, "U2U:1:2": 256.0},
            }
        ]
    )
    assert summary["slots_with_no_starting_uav_hol_but_actual_service"] == 1
    assert summary["s2u_completed_packets"] == 1
    assert summary["uav_relay_completions"] == 2
    assert summary["same_slot_multi_hop_packets"] == 1


def test_mock_short_training_integration_completes_without_paid_api():
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
        method_spec=MethodSpec.parse("td3_dinkelbach_llm_explorer_state_rule"),
        llm_explorer_mock=True,
    )
    assert result["episodes_run"] == 1
    assert result["llm_explorer"]["configuration"]["method_id"] == (
        "td3_dinkelbach_llm_explorer_state_rule"
    )
    assert result["llm_explorer"]["statistics"]["requests_total"] == 0
