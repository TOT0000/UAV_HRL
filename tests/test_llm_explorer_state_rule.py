import copy
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, _llm_routing_exploration_prior, train
from llm_explorer import (
    ExplorerValidationError,
    PromptBudgetError,
    first_gen_slot_service_record,
    llm_explorer_path_preflight,
)
from llm_explorer_state_rule import (
    STATE_RULE_SCHEMA_VERSION,
    build_runtime_feature_rows,
    compile_state_rule,
    evaluate_rule_for_joint_action,
    parse_and_compile_state_rule,
    runtime_feature_catalog,
)
from llm_state_rule_explorer import (
    LLM_STATE_RULE_CACHE_VERSION,
    LLM_STATE_RULE_PROMPT_VERSION,
    LLM_STATE_RULE_SUMMARY_VERSION,
    DeterministicMockStateRuleClient,
    LLMStateRuleConfig,
    LLMStateRuleController,
    communication_bottleneck_summary_state_rule,
    environment_definitions_state_rule,
    explorer_metadata_state_rule,
    routing_expected_capacity_observation,
    state_rule_movement_summary,
    state_rule_slot_service_record,
    slot_service_summary,
)
from replay_auxiliary import TASK_TYPE_ENCODING, empty_snapshot
from rng_contract import NamedRNGStreams
from run_experiment import _explorer_preflight_options, build_parser, main as run_experiment_main
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


class OneUavEventEnvironment:
    """Minimal deterministic environment that exercises PacketEngine events."""

    def __init__(self, *, capacity_mbps=1.0):
        self.num_UAV = 1
        self.GS_ID = 1
        self.GS_pos = (100.0, 0.0, 0.0)
        self.uav_dict = {0: Uav((0.0, 0.0, 100.0))}
        self.B_tot = 10e6
        self.current_time = 0.0
        self.source_uavs = {0}
        self.multi_tasks = {0: []}
        self.SR_teams = []
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []
        self.capacity_mbps = float(capacity_mbps)

    def is_routing_link_in_range(self, sender, receiver):
        return int(sender) != int(receiver)

    def is_s2u_in_range(self, sr_id, receiver):
        del sr_id, receiver
        return True

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        links = {
            (int(sender), int(receiver))
            for sender, receiver in dict(proposed_links).items()
            if int(sender) != int(receiver)
        }
        source_links = {
            (int(sr), int(receiver))
            for sr, receiver in dict(s2u_links or {}).items()
        }
        count = len(links) + len(source_links)
        capacity_mbps = self.capacity_mbps if count else 0.0
        bandwidth = self.B_tot / count if count else 0.0
        profile = np.full(50, capacity_mbps, dtype=float)
        self.active_link_capacity_profiles_mbps = {
            link: profile.copy() for link in links
        }
        self.active_s2u_capacity_profiles_mbps = {
            link: profile.copy() for link in source_links
        }
        self.active_s2u_capacities = {
            link: capacity_mbps for link in source_links
        }
        bandwidths = {link: bandwidth for link in links}
        bandwidths.update({("S2U", *link): bandwidth for link in source_links})
        return {link: capacity_mbps for link in links}, bandwidths


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
    wrong["schema_version"] = "uav-hrl-llm-state-rule-explorer-v1"
    with pytest.raises(ExplorerValidationError, match="incompatible"):
        restored.load_state_dict(wrong)


def test_v3_cache_identity_and_v2_checkpoint_feedback_migrates_from_raw(tmp_path):
    identity = {"method_id": "td3_dinkelbach_llm_explorer_state_rule"}
    controller = LLMStateRuleController(
        run_directory=tmp_path,
        run_identity=identity,
        evaluation=True,
    )
    cache_identity = controller._cache_identity(
        episode_index=1,
        stage="episode_result",
        system_prompt="system",
        user_prompt="prompt",
    )
    assert cache_identity["cache_version"] == LLM_STATE_RULE_CACHE_VERSION
    assert cache_identity["cache_version"].endswith("v3")
    old_cache_identity = dict(cache_identity)
    old_cache_identity["cache_version"] = "state-rule-complete-input-cache-v2"
    old_cache_path = controller._episode_result_path(1)
    old_cache_path.parent.mkdir(parents=True, exist_ok=True)
    old_cache_path.write_text(
        json.dumps(
            {
                "schema_version": controller.schema_version,
                "identity": old_cache_identity,
                "guidance": controller._serializable_guidance(
                    controller.active_guidance
                ),
            }
        ),
        encoding="utf-8",
    )
    assert controller._load_episode_result(1, cache_identity, 0.1) is None
    metadata = explorer_metadata_state_rule(controller.config)
    assert metadata["prompt_version"] == LLM_STATE_RULE_PROMPT_VERSION
    assert metadata["summary_version"] == LLM_STATE_RULE_SUMMARY_VERSION

    metrics = {"migration_marker": 7}
    controller.previous_episode_feedback = {
        "schema_version": controller.schema_version,
        "summary_version": "state-rule-observed-service-summary-v2",
        "metrics": metrics,
        "observed_task_and_traffic_summary": {},
        "movement_and_rule_application_summaries": {},
        "fov_com_and_slot_service_summaries": {
            "routing_decision_observations": {},
            "actual_within_slot_service": {
                "slots_with_no_starting_uav_hol_but_actual_service": 99
            },
        },
        "representative_cases": [],
        "applied_guidance": controller._serializable_guidance(
            controller.active_guidance
        ),
    }
    state = controller.state_dict()
    state["config"]["prompt_version"] = (
        "uav-hrl-llm-state-rule-prompts-v2"
    )
    episode_root = tmp_path / "llm_rule" / "episode_000001"
    episode_root.mkdir(parents=True)
    (episode_root / "feedback.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "movement_decisions": [],
                "routing_decisions": [
                    {
                        **_saved_routing_record(0),
                        "sender_total_backlog_bits": 321.0,
                        "hol_packet_id": 4,
                        "hol_remaining_deadline_seconds": 0.002,
                    }
                ],
                "slot_service_records": [
                    {
                        "start_of_slot_nonempty_sender_ids": [],
                        "transmitted_bits_by_link": {},
                    }
                ],
                "applied_guidance": controller._serializable_guidance(
                    controller.active_guidance
                ),
            }
        ),
        encoding="utf-8",
    )

    restored = LLMStateRuleController(
        run_directory=tmp_path,
        run_identity=identity,
        evaluation=True,
    )
    restored.load_state_dict(state)
    migrated = restored.previous_episode_feedback
    assert migrated["summary_version"] == LLM_STATE_RULE_SUMMARY_VERSION
    assert migrated["diagnostic_summary_migration"][
        "rebuilt_from_saved_raw_records"
    ] is True
    summaries = migrated["fov_com_and_slot_service_summaries"]
    assert summaries["routing_decision_observations"][
        "routing_queue_and_hol_observations"
    ]["system"]["sender_total_backlog_bits"]["max"] == 321.0
    assert summaries["actual_within_slot_service"][
        "slots_with_no_starting_uav_hol_but_actual_service"
    ] == 0

    no_raw = LLMStateRuleController(
        run_directory=None,
        run_identity=identity,
        evaluation=True,
    )
    no_raw.load_state_dict(state)
    unavailable = no_raw.previous_episode_feedback[
        "fov_com_and_slot_service_summaries"
    ]["routing_decision_observations"]["routing_queue_and_hol_observations"]
    assert unavailable["available"] is False
    assert "not replaced by zeros" in unavailable["scope"]
    assert no_raw.previous_episode_feedback[
        "fov_com_and_slot_service_summaries"
    ]["actual_within_slot_service"][
        "slots_with_no_starting_uav_hol_but_actual_service"
    ] is None


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


def test_training_feature_summary_capture_survives_warmup_active_and_fallback():
    env, snapshot = runtime_fixture()
    mask = np.ones(16, dtype=bool)
    actors = np.zeros((16, 3), dtype=np.float32)
    controller = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=SequenceClient([]),
        lazy_client=True,
    )

    controller.begin_episode(
        episode_index=0,
        total_joint_transitions=0,
        sigma_episode_start=0.2,
    )
    controller.observe_movement_boundary(
        env, snapshot, mask, None, episode_seconds=60
    )
    warmup_rows = controller.consume_last_runtime_rows()
    assert warmup_rows[0]["features"]["fov_target_valid"] is True
    assert warmup_rows[0]["features"]["actor_action_valid"] is False

    controller._compiled_rule = compile_state_rule(rule(speed=0.01))
    controller.evaluate_movement_bias(
        env,
        snapshot,
        mask,
        actors,
        sigma_episode_start=0.2,
        episode_seconds=60,
    )
    controller.observe_movement_boundary(
        env, snapshot, mask, actors, episode_seconds=60
    )
    active_rows = controller.consume_last_runtime_rows()
    assert active_rows[0]["evaluation_status"] == "rule_evaluated"
    assert active_rows[0]["features"]["actor_action_valid"] is True

    controller._compiled_rule = None
    controller._last_runtime_rows = []
    controller.active_guidance = controller._unguided("guidance_invalid_or_failed")
    controller.observe_movement_boundary(
        env, snapshot, mask, actors, episode_seconds=60
    )
    fallback_rows = controller.consume_last_runtime_rows()
    assert fallback_rows[0]["evaluation_status"] == (
        "not_evaluated_no_active_rule"
    )
    assert fallback_rows[0]["features"]["fov_target_valid"] is True


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


def _serve_sr_only_slot(*, capacity_mbps, size_bits, deadline_seconds):
    env = OneUavEventEnvironment(capacity_mbps=capacity_mbps)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 1.0, "COM": deadline_seconds},
    )
    packet = engine.create_sr_packet(0, size_bits, generation_time=0.0)
    packet["s2u_receiver"] = 0
    result = engine.serve_event_driven_slot(
        env,
        {0: 0},
        0.0,
        observed_hol_by_sender={0: None},
        observed_backlog_bits_by_sender={0: 0.0},
        effective_masks_by_sender={0: np.asarray([True, False])},
    )
    return engine, packet, result


def _serve_sr_then_forward_to_gs_in_same_slot():
    env = OneUavEventEnvironment(capacity_mbps=1.0)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 1.0, "COM": 1.0},
    )
    packet = engine.create_sr_packet(0, 100_000.0, generation_time=0.0)
    packet["s2u_receiver"] = 0
    result = engine.serve_event_driven_slot(
        env,
        {0: env.GS_ID},
        0.0,
        observed_hol_by_sender={0: None},
        observed_backlog_bits_by_sender={0: 0.0},
        effective_masks_by_sender={0: np.asarray([True, True])},
    )
    return packet, result


def test_real_packet_engine_zero_capacity_expiry_is_not_actual_service():
    _, packet, result = _serve_sr_only_slot(
        capacity_mbps=0.0,
        size_bits=100_000.0,
        deadline_seconds=0.01,
    )
    assert result["transmitted_bits_by_link"] == {}
    assert result["outcomes"][0]["packet"]["terminal_outcome"] == (
        "expired_dropped"
    )
    record = state_rule_slot_service_record(result, current_time=0.0)
    assert record["terminal_packet_ids"] == [packet["id"]]
    assert record["serviced_packet_ids"] == []
    assert record["had_actual_transmission"] is False
    summary = slot_service_summary([record])
    assert summary["slots_with_actual_transmission_observed"] == 0
    assert summary["slots_with_no_actual_transmission_observed"] == 1
    assert summary["slots_with_no_starting_uav_hol_but_actual_service"] == 0


def test_real_packet_engine_s2u_completion_from_empty_uav_start_is_service():
    _, packet, result = _serve_sr_only_slot(
        capacity_mbps=1.0,
        size_bits=100_000.0,
        deadline_seconds=1.0,
    )
    assert result["start_of_slot_routing_sender_ids"] == ()
    assert result["s2u_arrivals"][0]["id"] == packet["id"]
    record = state_rule_slot_service_record(result, current_time=0.0)
    assert record["had_actual_transmission"] is True
    assert record["positive_transmission_bits"] == pytest.approx(100_000.0)
    assert record["serviced_packet_ids"] == [packet["id"]]
    summary = slot_service_summary([record])
    assert summary["slots_with_no_starting_uav_hol_but_actual_service"] == 1


def test_real_packet_engine_partial_s2u_bits_count_without_hop_completion():
    _, packet, result = _serve_sr_only_slot(
        capacity_mbps=1.0,
        size_bits=1_000_000.0,
        deadline_seconds=1.0,
    )
    assert result["s2u_arrivals"] == []
    assert result["relay_arrivals"] == []
    assert result["outcomes"] == []
    record = state_rule_slot_service_record(result, current_time=0.0)
    assert record["had_actual_transmission"] is True
    assert record["positive_transmission_bits"] == pytest.approx(250_000.0)
    assert record["serviced_packet_ids"] == []
    assert packet["rem_bits"] == pytest.approx(750_000.0)
    summary = slot_service_summary([record])
    assert summary["slots_with_actual_transmission_observed"] == 1
    assert summary["slots_with_no_starting_uav_hol_but_actual_service"] == 1


def test_first_generation_record_preserves_real_same_slot_and_partial_service():
    _, packet, completed = _serve_sr_only_slot(
        capacity_mbps=1.0,
        size_bits=100_000.0,
        deadline_seconds=1.0,
    )
    record = first_gen_slot_service_record(
        completed, current_time=0.0, slot_seconds=0.25
    )
    assert record["start_of_slot_nonempty_sender_ids"] == []
    assert record["had_actual_transmission"] is True
    assert record["positive_transmission_bits"] == pytest.approx(100_000.0)
    assert record["completed_hops"][0]["packet_id"] == packet["id"]
    assert record["completed_hops"][0]["link_type"] == "S2U"

    _, _, partial = _serve_sr_only_slot(
        capacity_mbps=1.0,
        size_bits=1_000_000.0,
        deadline_seconds=1.0,
    )
    partial_record = first_gen_slot_service_record(
        partial, current_time=0.0, slot_seconds=0.25
    )
    assert partial_record["positive_transmission_bits"] == pytest.approx(250_000.0)
    assert partial_record["completed_hops"] == []
    summary = slot_service_summary([record])
    assert summary["slots_with_actual_transmission_observed"] == 1
    assert summary["slots_with_no_starting_uav_hol_but_actual_service"] == 1


def test_first_generation_records_empty_start_then_same_slot_forwarding():
    packet, result = _serve_sr_then_forward_to_gs_in_same_slot()
    record = first_gen_slot_service_record(
        result,
        current_time=0.0,
        slot_seconds=0.25,
        preselected_actions_by_sender={0: 1},
    )
    assert record["start_of_slot_nonempty_sender_ids"] == []
    assert record["had_actual_transmission"] is True
    assert record["positive_transmission_bits"] == pytest.approx(200_000.0)
    assert [hop["link_type"] for hop in record["completed_hops"]] == [
        "S2U",
        "U2G",
    ]
    assert record["completed_hops"][1]["packet_id"] == packet["id"]
    summary = slot_service_summary([record])
    assert summary["slots_with_no_starting_uav_hol_but_actual_service"] == 1
    assert summary["s2u_completed_packets"] == 1
    assert summary["gs_delivery_outcomes"] == 1


def test_real_packet_engine_slot_delivery_uses_nested_outcome_and_fov_useful_bits():
    env = OneUavEventEnvironment()
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 1.0, "COM": 1.0}
    )
    packet = engine.create_packet(
        0,
        "FOV",
        1000.0,
        0.0,
        capture_coverage_ratio=0.25,
        capture_image_quantity=0.5,
    )
    result = engine.serve_event_driven_slot(
        env,
        {0: env.GS_ID},
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 1000.0},
        effective_masks_by_sender={0: np.ones(2, dtype=bool)},
    )
    assert result["outcomes"][0]["packet"]["terminal_outcome"] == (
        "on_time_delivered"
    )
    assert "delivered_to_gs" not in result["outcomes"][0]
    assert "reason" not in result["outcomes"][0]

    record = state_rule_slot_service_record(result, current_time=0.0)
    assert record["on_time_gs_delivery_packets"] == 1
    assert record["late_gs_delivery_packets"] == 0
    assert record["gs_delivery_outcomes"] == 1
    assert record["fov_timely_delivered_raw_bits"] == pytest.approx(1000.0)
    assert record["fov_timely_useful_bits"] == pytest.approx(250.0)
    assert record["total_timely_useful_bits"] == pytest.approx(250.0)

    packet_metrics = engine.packet_metric_summary()
    metrics = {
        "fov_on_time_delivered_packets": packet_metrics["FOV"][
            "on_time_delivered_packets"
        ],
        "fov_late_delivered_packets": packet_metrics["FOV"][
            "late_delivered_packets"
        ],
        "com_on_time_delivered_packets": 0,
        "com_late_delivered_packets": 0,
        "timely_useful_delivery_bits": engine.total_timely_useful_bits,
        "fov_timely_delivered_raw_bits": engine.fov_timely_delivered_raw_bits,
        "fov_timely_useful_bits": engine.fov_timely_useful_bits,
        "com_timely_delivered_bits": engine.com_timely_delivered_bits,
    }
    summary = slot_service_summary([record], metrics)
    assert all(
        value is True
        for value in summary[
            "recorded_slot_sum_matches_authoritative_episode_total"
        ].values()
    )


def test_terminal_outcome_classification_and_packet_id_deduplication():
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 1.0, "COM": 1.0}
    )
    late_packet = engine.create_packet(0, "COM", 100.0, 0.0)
    late_event = engine._mark_deadline_violation(
        late_packet, 1.2, sender=0, reason="late_delivered"
    )
    dropped_packet = engine.create_packet(0, "FOV", 100.0, 0.0)
    dropped_event = engine._mark_deadline_violation(
        dropped_packet, 1.2, sender=0, reason="deadline"
    )
    late_outcome = engine._outcome_from_violation(late_event)
    dropped_outcome = engine._outcome_from_violation(dropped_event)
    slot_result = {
        "packet_decision_uses": [
            {"packet_id": late_packet["id"]},
            {"packet_id": late_packet["id"]},
        ],
        "s2u_arrivals": [],
        "relay_arrivals": [],
        "outcomes": [late_outcome, late_outcome, dropped_outcome],
        "start_of_slot_routing_sender_ids": (0,),
        "active_links": [],
        "transmitted_bits_by_link": {},
    }
    record = state_rule_slot_service_record(
        slot_result,
        pre_slot_violations=[dropped_event],
        current_time=1.0,
    )
    assert record["late_gs_delivery_packets"] == 1
    assert record["on_time_gs_delivery_packets"] == 0
    assert record["gs_delivery_outcomes"] == 1
    assert record["expired_dropped_packets"] == 1
    assert record["terminal_outcomes_by_task"]["COM"][
        "late_delivered_packets"
    ] == 1
    assert record["terminal_outcomes_by_task"]["FOV"][
        "expired_dropped_packets"
    ] == 1
    assert record["terminal_packet_ids"] == sorted(
        [late_packet["id"], dropped_packet["id"]]
    )


def test_current_expected_routing_capacity_varies_with_geometry_without_rng_use():
    env = Simulator(16, rng_streams=NamedRNGStreams(908))
    env.num_GT = 4
    env.reset_environment()
    env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
        0.0,
        0.0,
        100.0,
    )
    env.uav_dict[1].x_u, env.uav_dict[1].y_u, env.uav_dict[1].z_u = (
        50.0,
        0.0,
        100.0,
    )
    env.update_u2u_channels()
    env.update_u2g_channels()
    rng_before = copy.deepcopy(env.channel_small_scale_rng.bit_generator.state)
    close = routing_expected_capacity_observation(env, 0, [0, 1, env.GS_ID])
    assert env.channel_small_scale_rng.bit_generator.state == rng_before

    env.uav_dict[1].x_u = 350.0
    env.update_u2u_channels()
    env.update_u2g_channels()
    rng_before = copy.deepcopy(env.channel_small_scale_rng.bit_generator.state)
    far = routing_expected_capacity_observation(env, 0, [0, 1, env.GS_ID])
    assert env.channel_small_scale_rng.bit_generator.state == rng_before

    close_capacity = close["current_expected_capacity_bps_by_legal_action"]["1"]
    far_capacity = far["current_expected_capacity_bps_by_legal_action"]["1"]
    assert close_capacity > far_capacity > 0.0
    assert close["maximum_reference_normalizer_bps_by_legal_action"]["1"] == (
        far["maximum_reference_normalizer_bps_by_legal_action"]["1"]
    )
    assert close_capacity != close[
        "maximum_reference_normalizer_bps_by_legal_action"
    ]["1"]
    assert close["capacity_context_by_legal_action"]["1"] == {
        "link_type": "U2U",
        "sender_uav_id": 0,
        "receiver_id": 1,
        "distance_3d_m": 50.0,
        "reference_bandwidth_hz": 10e6,
        "fading_or_los_treatment": "expected Rician fading; no new channel sample",
    }


def _saved_routing_record(sender, *, backlog_marker=True, hol_marker=True):
    record = {
        "sender_uav_id": int(sender),
        "selected_action": 16,
        "current_expected_capacity_bps_by_legal_action": {
            str(sender): 0.0,
            "16": 2_000_000.0,
        },
        "maximum_reference_normalizer_bps_by_legal_action": {
            str(sender): 0.0,
            "16": 5_000_000.0,
        },
        "capacity_context_by_legal_action": {
            str(sender): {
                "link_type": "WAIT",
                "sender_uav_id": int(sender),
                "receiver_id": int(sender),
                "distance_3d_m": 0.0,
                "reference_bandwidth_hz": 0.0,
                "fading_or_los_treatment": "not applicable",
            },
            "16": {
                "link_type": "U2G",
                "sender_uav_id": int(sender),
                "receiver_id": 16,
                "distance_3d_m": 100.0,
                "reference_bandwidth_hz": 10e6,
                "fading_or_los_treatment": "installed expected fading",
                "installed_los_state": True,
            },
        },
    }
    if backlog_marker:
        record["sender_total_backlog_bits"] = 0.0
    if hol_marker:
        record["hol_packet_id"] = None
        record["hol_remaining_deadline_seconds"] = None
    return record


def test_capacity_equal_queue_summary_distinguishes_backlog_hol_empty_and_unknown():
    empty = _saved_routing_record(0)
    unknown = _saved_routing_record(
        1, backlog_marker=False, hol_marker=False
    )
    loaded = copy.deepcopy(empty)
    loaded.update(
        sender_total_backlog_bits=1_500_000.0,
        hol_packet_id=91,
        hol_remaining_deadline_seconds=0.001,
    )
    empty_summary = communication_bottleneck_summary_state_rule(
        [empty, unknown], {}
    )
    loaded_summary = communication_bottleneck_summary_state_rule(
        [loaded, unknown], {}
    )
    assert empty_summary[
        "current_expected_capacity_bps_by_link_type"
    ] == loaded_summary["current_expected_capacity_bps_by_link_type"]
    empty_queue = empty_summary["routing_queue_and_hol_observations"]
    loaded_queue = loaded_summary["routing_queue_and_hol_observations"]
    assert empty_queue != loaded_queue
    assert empty_queue["system"]["backlog_zero_count"] == 1
    assert empty_queue["system"]["no_hol_count"] == 1
    assert empty_queue["system"]["hol_presence_unknown_count"] == 1
    assert empty_queue["system"]["backlog_unknown_count"] == 1
    assert loaded_queue["system"]["backlog_positive_count"] == 1
    assert loaded_queue["system"]["sender_total_backlog_bits"]["max"] == (
        1_500_000.0
    )
    assert loaded_queue["system"]["hol_present_count"] == 1
    assert loaded_queue["system"]["hol_remaining_deadline_seconds"][
        "min"
    ] == pytest.approx(0.001)


def _movement_record(time_s, rows):
    return {
        "episode_index": 0,
        "time_seconds": float(time_s),
        "uavs": rows,
    }


def _movement_item(uav_id, tasks, features, *, mask=True, state_rule=None):
    diagnostic = {
        "raw_bias": [0.0, 0.0, 0.0],
        "bounded_bias": [0.0, 0.0, 0.0],
        "effective_bias": [0.0, 0.0, 0.0],
        "clipped_components": 0,
        "runtime_failure": None,
        "evaluation_status": "not_evaluated_no_active_rule",
        "trace": {
            "conditions_evaluated": 0,
            "conditions_true": 0,
            "short_circuits": 0,
        },
        "features": features,
    }
    if state_rule:
        diagnostic.update(state_rule)
    return {
        "uav_id": int(uav_id),
        "task_types": list(tasks),
        "movement_mask": bool(mask),
        "actor_normalized_action": (
            [0.0, 0.0, 0.0]
            if features and features.get("actor_action_valid")
            else None
        ),
        "executed_normalized_action": [0.0, 0.0, 0.0],
        "movement_energy_j": 1.0,
        "state_rule": diagnostic,
    }


def test_complete_service_geometry_denominators_dual_role_and_warmup_background():
    first = {
        "actor_action_valid": False,
        "has_fov_task": True,
        "has_com_task": True,
        "fov_target_valid": True,
        "fov_c9_satisfied": True,
        "fov_c10_geometry_valid": True,
        "fov_c10_satisfied": True,
        "fov_capture_valid": True,
        "fov_coverage_ratio": 0.5,
        "fov_image_quantity": 0.25,
        "com_target_valid": True,
        "com_in_range": True,
        "com_distance_3d_m": 100.0,
        "com_reference_capacity_valid": True,
        "com_reference_capacity_mbps": 20.0,
        "com_sr_queue_valid": True,
        "com_sr_queue_empty": True,
        "com_sr_backlog_bits": 0.0,
        "com_sr_packet_count": 0.0,
        "com_sr_hol_deadline_valid": False,
    }
    second = {
        **first,
        "actor_action_valid": True,
        "fov_c9_satisfied": False,
        "fov_c10_geometry_valid": False,
        "fov_c10_satisfied": False,
        "fov_capture_valid": False,
        "fov_coverage_ratio": 0.0,
        "com_in_range": False,
        "com_distance_3d_m": 450.0,
        "com_reference_capacity_valid": False,
        "com_reference_capacity_mbps": 0.0,
        "com_sr_queue_empty": False,
        "com_sr_backlog_bits": 100.0,
        "com_sr_packet_count": 1.0,
        "com_sr_hol_deadline_valid": True,
        "com_sr_hol_remaining_deadline_s": 0.1,
    }
    third = {
        "actor_action_valid": True,
        "has_fov_task": True,
        "has_com_task": False,
        "fov_target_valid": False,
    }
    records = []
    for time_s, features, tasks, mask in (
        (0, first, ["FOV", "COM"], True),
        (1, second, ["FOV", "COM"], True),
        (2, third, ["FOV"], False),
    ):
        rows = [
            _movement_item(0, tasks, features, mask=mask),
            *[
                _movement_item(uav_id, [], None, mask=False)
                for uav_id in range(1, 16)
            ],
        ]
        records.append(_movement_record(time_s, rows))
    summary, _ = state_rule_movement_summary(records)
    scopes = summary["by_uav"]["UAV_0"]["service_geometry_by_control_scope"]
    all_decisions = scopes["all_decisions"]
    assert all_decisions["actor_action_availability"] == {
        "count": 2,
        "denominator": 3,
        "ratio": pytest.approx(2 / 3),
    }
    assert all_decisions["dual_fov_com_assigned_decisions"] == 2
    assert all_decisions["fov"]["assigned_target_known"]["denominator"] == 3
    assert all_decisions["fov"]["c9_satisfied"] == {
        "count": 1,
        "denominator": 2,
        "ratio": 0.5,
    }
    assert all_decisions["fov"]["c10_satisfied_given_valid_geometry"] == {
        "count": 1,
        "denominator": 1,
        "ratio": 1.0,
    }
    assert all_decisions["com"]["in_range"]["denominator"] == 2
    assert all_decisions["com"]["current_expected_capacity_valid"] == {
        "count": 1,
        "denominator": 2,
        "ratio": 0.5,
    }
    assert all_decisions["com"]["sr_queue_empty_given_valid"] == {
        "count": 1,
        "denominator": 2,
        "ratio": 0.5,
    }
    assert scopes["td3_controlled_decisions"]["decision_count"] == 2
    assert scopes["other_controller_decisions"]["decision_count"] == 1
    assert summary["by_uav"]["UAV_1"][
        "service_geometry_by_control_scope"
    ]["all_decisions"]["available"] is False


def test_representative_cases_use_full_multi_uav_episode_and_time_regions():
    records = []
    for time_s in range(60):
        rows = []
        for uav_id in range(16):
            role = ("FOV", "COM", "FOV_COM")[uav_id % 3]
            tasks = ["FOV", "COM"] if role == "FOV_COM" else [role]
            features = {
                "actor_action_valid": True,
                "has_fov_task": "FOV" in tasks,
                "has_com_task": "COM" in tasks,
            }
            diagnostic = None
            if time_s == 10:
                diagnostic = {"effective_bias": [0.1, 0.0, 0.0]}
            elif time_s == 30:
                diagnostic = {
                    "raw_bias": [1.0, 0.0, 0.0],
                    "bounded_bias": [0.2, 0.0, 0.0],
                    "effective_bias": [0.2, 0.0, 0.0],
                    "clipped_components": 1,
                }
            elif time_s == 50:
                diagnostic = {"runtime_failure": "ArithmeticError: test"}
            rows.append(
                _movement_item(
                    uav_id, tasks, features, mask=True, state_rule=diagnostic
                )
            )
        records.append(_movement_record(time_s, rows))
    _, cases = state_rule_movement_summary(records)
    assert len(cases) == 3
    assert len({case["uav_id"] for case in cases}) >= 2
    assert {case["service_role"] for case in cases} == {"fov", "com", "dual"}
    assert len({case["rule_behavior"] for case in cases}) == 3
    assert {int(float(case["time_s"]) // 20) for case in cases} == {0, 1, 2}
    assert len({json.dumps(case, sort_keys=True) for case in cases}) == 3


def test_huge_json_integer_retries_then_succeeds_and_invalid_cache_is_ignored(tmp_path):
    huge = json.dumps(rule(speed=10**1000), separators=(",", ":"))
    with pytest.raises(ExplorerValidationError, match="finite float"):
        parse_and_compile_state_rule(huge)
    deeply_nested = (
        '{"schema_version":"td3-state-bias-rule-v1","speed":'
        + '{"op":"neg","args":[' * 1200
        + "0"
        + "]}" * 1200
        + ',"heading":0,"vertical":0}'
    )
    with pytest.raises(ExplorerValidationError, match="JSON|nesting|depth"):
        parse_and_compile_state_rule(deeply_nested)

    client = SequenceClient(["analysis", huge, json.dumps(rule(speed=0.02))])
    controller = LLMStateRuleController(
        run_directory=tmp_path,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=client,
    )
    controller.previous_episode_feedback = previous_feedback()
    result = controller.begin_episode(
        episode_index=210,
        total_joint_transitions=10_000,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={},
        environment_constants=environment_definitions_state_rule(
            episode_seconds=60, routing_slot_seconds=0.25
        ),
    )
    assert result["rule"] == rule(speed=0.02)
    assert controller.statistics["failure_counts"]["content_validation"] == 1
    assert len(client.calls) == 3
    validation = json.loads(
        (
            tmp_path
            / "llm_rule"
            / "episode_000211"
            / "guidance"
            / "attempt_01"
            / "validation.json"
        ).read_text(encoding="utf-8")
    )
    assert "finite float" in validation["errors"]["td3"]

    identity = {"cache": "invalid-huge-rule"}
    path = controller._episode_result_path(999)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": controller.schema_version,
                "identity": identity,
                "guidance": {
                    "enabled": True,
                    "td3_fallback": False,
                    "rule": rule(speed=10**1000),
                },
            }
        ),
        encoding="utf-8",
    )
    assert controller._load_episode_result(999, identity, 0.2) is None


def test_all_huge_rules_fallback_and_keyboard_interrupt_propagates():
    huge = json.dumps(rule(speed=10**1000), separators=(",", ":"))
    failing = SequenceClient(["analysis", huge, huge])
    controller = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=failing,
    )
    controller.previous_episode_feedback = previous_feedback()
    result = controller.begin_episode(
        episode_index=211,
        total_joint_transitions=10_000,
        sigma_episode_start=0.2,
        routing_epsilon=0.5,
        current_observable_initial_state={},
        environment_constants=environment_definitions_state_rule(
            episode_seconds=60, routing_slot_seconds=0.25
        ),
    )
    assert result["td3_fallback"] is True
    assert result["rule"] is None
    assert controller.statistics["failure_counts"]["content_validation"] == 2

    interrupted = LLMStateRuleController(
        run_directory=None,
        run_identity={"method_id": "td3_dinkelbach_llm_explorer_state_rule"},
        client=SequenceClient([KeyboardInterrupt()]),
    )
    interrupted.previous_episode_feedback = previous_feedback()
    with pytest.raises(KeyboardInterrupt):
        interrupted.begin_episode(
            episode_index=212,
            total_joint_transitions=10_000,
            sigma_episode_start=0.2,
            routing_epsilon=0.5,
            current_observable_initial_state={},
            environment_constants=environment_definitions_state_rule(
                episode_seconds=60, routing_slot_seconds=0.25
            ),
        )


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


def test_zero_rule_diagnostics_preserve_baseline_core_training_result():
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
            random_seed=20261018,
        )

    baseline = train(
        config(), method_spec=MethodSpec.parse("td3_dinkelbach")
    )["episode_metrics"][0]
    state_rule = train(
        config(),
        method_spec=MethodSpec.parse(
            "td3_dinkelbach_llm_explorer_state_rule"
        ),
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
        assert state_rule[key] == pytest.approx(baseline[key])


def test_real_cli_train_evaluate_and_resume_without_llm_call(monkeypatch):
    calls = []

    def unexpected_chat(self, **kwargs):
        calls.append(kwargs)
        raise AssertionError("short warmup/evaluation/resume must not call the LLM")

    monkeypatch.setattr(
        DeterministicMockStateRuleClient, "chat", unexpected_chat
    )
    with tempfile.TemporaryDirectory(prefix="sr-") as short_temp:
        output_root = Path(short_temp)
        method = "td3_dinkelbach_llm_explorer_state_rule"
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


def test_two_episode_mock_training_enters_guidance_and_sends_v3_summaries(
    tmp_path, monkeypatch
):
    client = SequenceClient(["analysis", json.dumps(rule())])
    original_begin_episode = LLMStateRuleController.begin_episode

    def begin_after_simulated_warmup(self, *args, **kwargs):
        kwargs["total_joint_transitions"] = max(
            int(kwargs.get("total_joint_transitions", 0)), 10_000
        )
        return original_begin_episode(self, *args, **kwargs)

    monkeypatch.setattr(
        LLMStateRuleController, "begin_episode", begin_after_simulated_warmup
    )
    result = train(
        TrainingConfig(
            total_episodes=2,
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
            random_seed=20260818,
        ),
        method_spec=MethodSpec.parse(
            "td3_dinkelbach_llm_explorer_state_rule"
        ),
        llm_explorer_client=client,
    )
    assert result["episodes_run"] == 2
    assert result["llm_explorer"]["statistics"]["requests_total"] == 2
    assert len(client.calls) == 2
    for call in client.calls:
        prompt = call["prompt"]
        assert "service_geometry_by_control_scope" in prompt
        assert "current_expected_capacity_bps_by_link_type" in prompt
        assert "actual_shared_fdma_service_by_link_type" in prompt
        assert "on_time_gs_delivery_packets" in prompt
        assert "routing_queue_and_hol_observations" in prompt
        assert "sender_total_backlog_bits" in prompt
        assert "hol_remaining_deadline_seconds" in prompt
