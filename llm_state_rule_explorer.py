"""State-dependent LLM rule guidance for TD3 movement exploration only."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiment_config import (
    COM_PACKET_SIZE_BITS,
    REFERENCE_COM_BANDWIDTH_HZ,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
)
from visual_sensing import VS_PACKET_MAX_BITS
from llm_explorer import (
    ExplorerValidationError,
    LLMExplorerConfig,
    LLMExplorerController,
    NUM_UAV,
    _atomic_json,
    _json_clone,
    environment_constants_and_definitions,
    prompt_token_budget,
    render_prompt,
)
from llm_explorer_td3_only import communication_bottleneck_summary
from llm_explorer_v3 import _diverse_case_subset, _numeric_summary, first_eligible_episode
from llm_explorer_state_rule import (
    STATE_RULE_FEATURE_CATALOG_VERSION,
    STATE_RULE_LANGUAGE_VERSION,
    STATE_RULE_SCHEMA_VERSION,
    CompiledStateRule,
    build_runtime_feature_rows,
    compile_state_rule,
    evaluate_rule_for_joint_action,
    parse_and_compile_state_rule,
    runtime_feature_catalog,
    state_rule_output_schema,
)


LLM_STATE_RULE_METHOD_ID = "td3_dinkelbach_llm_explorer_state_rule"
LLM_STATE_RULE_VERSION = "uav-hrl-llm-state-rule-explorer-v1"
LLM_STATE_RULE_PROMPT_VERSION = "uav-hrl-llm-state-rule-prompts-v1"
LLM_STATE_RULE_SUMMARY_VERSION = "state-rule-observed-service-summary-v1"
LLM_STATE_RULE_CACHE_VERSION = "state-rule-complete-input-cache-v1"
LLM_STATE_RULE_ANALYSIS_OUTPUT_TOKENS = 4096
LLM_STATE_RULE_RULE_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class LLMStateRuleConfig(LLMExplorerConfig):
    prompt_version: str = LLM_STATE_RULE_PROMPT_VERSION
    analysis_max_output_tokens: int = LLM_STATE_RULE_ANALYSIS_OUTPUT_TOKENS
    guidance_max_output_tokens: int = LLM_STATE_RULE_RULE_OUTPUT_TOKENS
    episode_joint_transitions: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_version != LLM_STATE_RULE_PROMPT_VERSION:
            raise ValueError("state-rule prompt version is fixed by the method contract")
        if int(self.episode_joint_transitions) <= 0:
            raise ValueError("episode_joint_transitions must be positive")
        if int(self.safety_margin_tokens) != 4_096:
            raise ValueError("state-rule requests require a fixed 4,096-token safety margin")
        for name, value in (
            ("analysis_max_output_tokens", self.analysis_max_output_tokens),
            ("guidance_max_output_tokens", self.guidance_max_output_tokens),
        ):
            if not 1 <= int(value) <= 16_384:
                raise ValueError(f"{name} must fit the gpt-4o-mini 16,384-token output limit")


class DeterministicMockStateRuleClient:
    """Offline two-stage client; never performs a paid request."""

    def chat(self, *, model: str, prompt: str, **_: Any) -> dict[str, Any]:
        if "Use the following evidence to design" in prompt:
            content = (
                "1. Main problems and numerical evidence\n"
                "The offline mock has no evidence supporting directional intervention.\n"
                "2. Shared rule design\nReturn zero on all three components.\n"
                "3. Expected outcomes and possible costs\nPreserve baseline exploration.\n"
                "4. Uncertainty and metrics to inspect next episode\nMock inference only."
            )
        else:
            content = json.dumps(
                {
                    "schema_version": STATE_RULE_SCHEMA_VERSION,
                    "speed": 0.0,
                    "heading": 0.0,
                    "vertical": 0.0,
                },
                separators=(",", ":"),
            )
        return {
            "content": content,
            "actual_model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
        }


def environment_definitions_state_rule(**kwargs: Any) -> dict[str, Any]:
    base = environment_constants_and_definitions(**kwargs)
    visual = base["visual_sensing"]
    return {
        "schema_version": LLM_STATE_RULE_VERSION,
        "resolved_system_settings": {
            **base,
            "state_rule": {
                "weight_w": 1.0,
                "rule_update_interval_episodes": 1,
                "warmup_joint_transitions": 10_000,
                "evaluation_bias_enabled": False,
                "routing_guidance": "disabled",
            },
        },
        "resolved_traffic_settings": {
            "com_packet_bits": float(COM_PACKET_SIZE_BITS),
            "fov_max_packet_bits": float(VS_PACKET_MAX_BITS),
            "task_deadlines_seconds": base["task_deadlines_seconds"],
            "generation_model": base["packet_generation_model"],
            "reference_com_bandwidth_hz": float(REFERENCE_COM_BANDWIDTH_HZ),
            "total_shared_bandwidth_hz": float(TOTAL_COMMUNICATION_BANDWIDTH_HZ),
        },
        "resolved_reward_and_metric_definitions": {
            "movement_reward": base["movement_reward"],
            "routing_reward": base["routing_reward"],
            "safe_ddqn": base["safe_ddqn"],
            "energy_efficiency": "timely useful GS-delivered bits divided by all-UAV movement energy in joules",
            "visual_sensing": visual,
        },
        "runtime_feature_catalog": runtime_feature_catalog(),
        "rule_language": {
            "schema_version": STATE_RULE_SCHEMA_VERSION,
            "semantics_version": STATE_RULE_LANGUAGE_VERSION,
            "output_schema": state_rule_output_schema(),
        },
    }


def _summaries(values: list[list[float]]) -> list[dict[str, Any]]:
    if not values:
        return [_numeric_summary([]) for _ in range(3)]
    array = np.asarray(values, dtype=float).reshape(-1, 3)
    return [_numeric_summary(array[:, index].tolist(), 6) for index in range(3)]


def state_rule_movement_summary(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate every decision; cases are illustrative and separately labeled."""

    by_uav: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    totals = Counter()
    for uav_id in range(NUM_UAV):
        actors: list[list[float]] = []
        executed: list[list[float]] = []
        raw: list[list[float]] = []
        bounded: list[list[float]] = []
        effective: list[list[float]] = []
        energies: list[float] = []
        role_counts: Counter[str] = Counter()
        max_role_run: Counter[str] = Counter()
        current_role_run: Counter[str] = Counter()
        clipped = zero = failures = controlled = 0
        conditions = conditions_true = short_circuits = 0
        for record in records:
            uavs = record.get("uavs", [])
            if uav_id >= len(uavs) or not isinstance(uavs[uav_id], dict):
                continue
            item = uavs[uav_id]
            tasks = [str(task) for task in item.get("task_types", [])]
            active_roles = set(tasks)
            for role in list(current_role_run):
                if role not in active_roles:
                    current_role_run[role] = 0
            for role in active_roles:
                role_counts[role] += 1
                current_role_run[role] += 1
                max_role_run[role] = max(max_role_run[role], current_role_run[role])
            if not bool(item.get("movement_mask", False)):
                continue
            controlled += 1
            actor = item.get("actor_normalized_action")
            actual = item.get("executed_normalized_action")
            rule = item.get("state_rule") if isinstance(item.get("state_rule"), dict) else {}
            if actor is not None:
                actors.append(list(map(float, actor)))
            if actual is not None:
                executed.append(list(map(float, actual)))
            for target, destination in (
                (rule.get("raw_bias"), raw),
                (rule.get("bounded_bias"), bounded),
                (rule.get("effective_bias"), effective),
            ):
                if isinstance(target, list) and len(target) == 3:
                    destination.append(list(map(float, target)))
            clipped += int(rule.get("clipped_components", 0))
            failures += int(rule.get("runtime_failure") is not None)
            trace = rule.get("trace") if isinstance(rule.get("trace"), dict) else {}
            conditions += int(trace.get("conditions_evaluated", 0))
            conditions_true += int(trace.get("conditions_true", 0))
            short_circuits += int(trace.get("short_circuits", 0))
            applied = np.asarray(rule.get("effective_bias", [0.0, 0.0, 0.0]), dtype=float)
            zero += int(not np.any(np.abs(applied) > 1e-12))
            if item.get("movement_energy_j") is not None:
                energies.append(float(item["movement_energy_j"]))
            if len(cases) < 24 and (clipped or failures or np.any(np.abs(applied) > 1e-12)):
                cases.append({
                    "domain": "movement_state_rule",
                    "episode": int(record.get("episode_index", -1)) + 1,
                    "time_s": record.get("time_seconds"),
                    "uav_id": uav_id,
                    "tasks": tasks,
                    "features": rule.get("features"),
                    "actor_action": actor,
                    "raw_bias": rule.get("raw_bias"),
                    "bounded_bias": rule.get("bounded_bias"),
                    "effective_bias": rule.get("effective_bias"),
                    "executed_action": actual,
                    "runtime_failure": rule.get("runtime_failure"),
                })
        totals.update(
            controlled_decisions=controlled,
            clipped_components=clipped,
            zero_bias_decisions=zero,
            runtime_failures=failures,
            conditions_evaluated=conditions,
            conditions_true=conditions_true,
            short_circuits=short_circuits,
        )
        by_uav[f"UAV_{uav_id}"] = {
            "role_decision_counts": dict(sorted(role_counts.items())),
            "longest_consecutive_role_duration_s": {
                key: int(value) for key, value in sorted(max_role_run.items())
            },
            "td3_controlled_decisions": controlled,
            "actor_action_by_component": _summaries(actors),
            "raw_bias_by_component": _summaries(raw),
            "bounded_bias_by_component": _summaries(bounded),
            "effective_bias_by_component": _summaries(effective),
            "executed_action_by_component": _summaries(executed),
            "clipped_component_count": clipped,
            "zero_effective_bias_decision_count": zero,
            "runtime_failure_count": failures,
            "condition_evaluation_count": conditions,
            "condition_true_count": conditions_true,
            "short_circuit_count": short_circuits,
            "movement_energy_j": _numeric_summary(energies, 4),
        }
    return {
        "scope": "complete episode; one row per UAV; durations use one-second movement decisions",
        "totals": dict(totals),
        "by_uav": by_uav,
    }, _diverse_case_subset(cases, 3)


def slot_service_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    transmitted_bits: list[float] = []
    slots_with_empty_start_and_service = 0
    for record in records:
        start_senders = set(map(int, record.get("start_of_slot_nonempty_sender_ids", [])))
        service_packet_ids = set(map(int, record.get("serviced_packet_ids", [])))
        totals["slot_count"] += 1
        totals["s2u_completed_packets"] += int(record.get("s2u_completed_packets", 0))
        totals["uav_relay_completions"] += int(record.get("uav_relay_completions", 0))
        totals["gs_delivery_outcomes"] += int(record.get("gs_delivery_outcomes", 0))
        totals["same_slot_multi_hop_packets"] += int(record.get("same_slot_multi_hop_packets", 0))
        if service_packet_ids and not start_senders:
            # This is deliberately only a slot-level fact. Packet IDs and
            # sender IDs are separate namespaces, so no per-UAV attribution is
            # inferred from an empty start followed by actual service.
            slots_with_empty_start_and_service += 1
        transmitted_bits.extend(float(value) for value in record.get("transmitted_bits_by_link", {}).values())
    return {
        "scope": "actual event-time service over complete recorded routing slots; system/link evidence, not per-UAV causal attribution",
        **dict(totals),
        "slots_with_no_starting_uav_hol_but_actual_service": slots_with_empty_start_and_service,
        "transmitted_bits_per_active_link": _numeric_summary(transmitted_bits, 2),
        "note": "slot-start empty queues do not imply no service; generation, S2U access, and same-slot forwarding can occur later in the slot",
    }


class LLMStateRuleController(LLMExplorerController):
    schema_version = LLM_STATE_RULE_VERSION
    artifact_directory_name = "llm_rule"

    def __init__(
        self,
        *,
        config: LLMStateRuleConfig | None = None,
        mock: bool = False,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        resolved = config or LLMStateRuleConfig()
        if mock and client is None:
            client = DeterministicMockStateRuleClient()
            mock = False
        super().__init__(config=resolved, mock=mock, client=client, **kwargs)
        self.config: LLMStateRuleConfig
        self._compiled_rule: CompiledStateRule | None = None
        self._last_runtime_rows: list[dict[str, Any]] = []
        self.slot_service_records: list[dict[str, Any]] = []
        self.runtime_failure_total = 0

    @staticmethod
    def _unguided(reason: str) -> dict[str, Any]:
        failed = str(reason) in {"analysis_stage_failed", "guidance_invalid_or_failed"}
        return {
            "enabled": False,
            "rule": None,
            "td3_fallback": failed,
            "td3_reason": str(reason),
            "routing_prior": None,
            "routing_fallback": False,
            "routing_reason": "routing_guidance_disabled",
            "analysis_attempt": None,
            "td3_guidance_attempt": None,
            "routing_guidance_attempt": None,
            "reused": False,
            "guidance_weight_w": 0.0,
            "sigma_episode_start": None,
        }

    @staticmethod
    def _serializable_guidance(guidance: dict[str, Any]) -> dict[str, Any]:
        result = dict(guidance)
        candidate = result.get("td3_bias")
        if isinstance(candidate, dict):
            result["rule"] = candidate
            result.pop("td3_bias", None)
        for key, value in tuple(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return _json_clone(result)

    @staticmethod
    def _restore_guidance(value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        rule = result.get("rule")
        if rule is not None:
            result["rule"] = compile_state_rule(rule).rule
        result.pop("td3_bias", None)
        result["routing_prior"] = None
        result["routing_fallback"] = False
        result["routing_reason"] = "routing_guidance_disabled"
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        rule = self.active_guidance.get("rule")
        self._compiled_rule = None if rule is None else compile_state_rule(rule)
        self.runtime_failure_total = int(state.get("runtime_failure_total", 0))

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result.update(
            runtime_failure_total=int(self.runtime_failure_total),
            state_rule_source_episode=self.active_guidance.get("source_episode"),
            state_rule_intervention_active=bool(
                self.active_guidance.get("enabled")
                and not self.active_guidance.get("td3_fallback")
            ),
        )
        return result

    def _templates(self) -> tuple[str, str, str]:
        root = Path(__file__).resolve().parent / "prompts"
        return tuple(
            (root / name).read_text(encoding="utf-8")
            for name in (
                "llm_state_rule_system.txt",
                "llm_state_rule_analysis.txt",
                "llm_state_rule_generation.txt",
            )
        )

    def _render_system_prompt(self, template: str, environment_constants: dict[str, Any]) -> str:
        return render_prompt(
            template,
            {
                "RESOLVED_SYSTEM_SETTINGS": environment_constants["resolved_system_settings"],
                "RESOLVED_TRAFFIC_SETTINGS": environment_constants["resolved_traffic_settings"],
                "RESOLVED_REWARD_AND_METRIC_DEFINITIONS": environment_constants["resolved_reward_and_metric_definitions"],
                "RUNTIME_FEATURE_CATALOG": environment_constants["runtime_feature_catalog"],
            },
        )

    def _cache_identity(self, **kwargs: Any) -> dict[str, Any]:
        identity = super()._cache_identity(**kwargs)
        identity.update(
            schema_version=LLM_STATE_RULE_VERSION,
            cache_version=LLM_STATE_RULE_CACHE_VERSION,
            rule_schema_version=STATE_RULE_SCHEMA_VERSION,
            rule_language_version=STATE_RULE_LANGUAGE_VERSION,
            feature_catalog_version=STATE_RULE_FEATURE_CATALOG_VERSION,
            routing_guidance="disabled",
        )
        return identity

    def _routing_guidance_required(self) -> bool:
        return False

    def _stage_display_name(self, stage: str) -> str:
        return "rule" if stage == "guidance" else str(stage)

    def _before_request(self, *, episode_index: int, stage: str, attempt: int) -> None:
        print(
            f"[LLM state rule] episode={episode_index + 1} "
            f"stage={self._stage_display_name(stage)} "
            f"attempt={attempt} request_start retry={str(attempt > 1).lower()} "
            f"failed_attempts={self.statistics['failed_attempts_total']}",
            flush=True,
        )

    def _parse_guidance_response(
        self, content: str, sigma_episode_start: float
    ) -> tuple[dict[str, Any] | None, None, dict[str, str]]:
        del sigma_episode_start
        try:
            return parse_and_compile_state_rule(content).rule, None, {}
        except (TypeError, ValueError) as exc:
            return None, None, {"td3": str(exc)}

    def _guidance_cache_content(self, td3_bias: Any, routing_prior: Any) -> str:
        del routing_prior
        return json.dumps(
            compile_state_rule(td3_bias).rule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _load_episode_result(
        self, episode_index: int, identity: dict[str, Any], sigma: float
    ) -> dict[str, Any] | None:
        del sigma
        path = self._episode_result_path(episode_index)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != self.schema_version or value.get("identity") != identity:
                return None
            guidance = self._restore_guidance(value["guidance"])
            if guidance.get("enabled") and not guidance.get("td3_fallback"):
                rule = compile_state_rule(guidance["rule"]).rule
                guidance["rule"] = rule
                guidance["td3_bias"] = rule
            guidance["reused"] = True
            guidance["cache_source"] = "episode_result"
            return guidance
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_episode_result(
        self, *, episode_index: int, identity: dict[str, Any], guidance: dict[str, Any]
    ) -> None:
        path = self._episode_result_path(episode_index)
        if path is None:
            return
        try:
            _atomic_json(
                path,
                {
                    "schema_version": self.schema_version,
                    "identity": identity,
                    "guidance": self._serializable_guidance(guidance),
                },
            )
        except OSError as exc:
            self._record_local_failure(
                episode_index=episode_index, stage="episode_result", attempt=0,
                category="artifact_write", reason=f"episode result cache: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _previous_rule(previous: Mapping[str, Any]) -> dict[str, Any]:
        guidance = previous.get("applied_guidance")
        guidance = guidance if isinstance(guidance, dict) else {}
        rule = guidance.get("rule")
        if rule is None or guidance.get("td3_fallback") or not guidance.get("enabled"):
            return {
                "available": False,
                "reason": guidance.get("td3_reason") or "no reliable prior complete rule",
                "rule": None,
            }
        return {"available": True, "source_episode": guidance.get("source_episode"), "rule": _json_clone(rule)}

    @staticmethod
    def _application_status(previous: Mapping[str, Any]) -> dict[str, Any]:
        guidance = previous.get("applied_guidance")
        guidance = guidance if isinstance(guidance, dict) else {}
        movement = previous.get("movement_and_rule_application_summaries")
        totals = movement.get("totals", {}) if isinstance(movement, dict) else {}
        return {
            "enabled": bool(guidance.get("enabled", False)),
            "fallback": bool(guidance.get("td3_fallback", False)),
            "disabled_reason": guidance.get("td3_reason"),
            "reused_from_cache": bool(guidance.get("reused", False)),
            "source_episode": guidance.get("source_episode"),
            "actual_application_totals": _json_clone(totals),
            "routing_guidance": "disabled_not_requested",
        }

    def _compact_feedback(self, previous: dict[str, Any]) -> dict[str, Any]:
        return {
            "metrics": _json_clone(previous.get("metrics", {})),
            "previous_complete_rule": self._previous_rule(previous),
            "previous_rule_application_status": self._application_status(previous),
            "observed_task_and_traffic_summary": _json_clone(previous.get("observed_task_and_traffic_summary", {})),
            "movement_and_rule_application_summaries": _json_clone(previous.get("movement_and_rule_application_summaries", {})),
            "fov_com_and_slot_service_summaries": _json_clone(previous.get("fov_com_and_slot_service_summaries", {})),
            "representative_cases": _json_clone(previous.get("representative_cases", [])),
        }

    def _exploration_settings(self, *, sigma_episode_start: float, routing_epsilon: float) -> dict[str, Any]:
        del routing_epsilon
        return {
            "upcoming_episode": getattr(self, "_upcoming_episode", None),
            "total_joint_transitions": getattr(self, "_total_joint_transitions", None),
            "warmup_joint_transitions": self.config.warmup_joint_transitions,
            "first_eligible_episode_if_every_episode_has_configured_length": first_eligible_episode(
                self.config.warmup_joint_transitions, self.config.episode_joint_transitions
            ),
            "activation_uses_actual_joint_transition_count": True,
            "sigma_episode_start": float(sigma_episode_start),
            "raw_bias_host_clip": [-float(sigma_episode_start), float(sigma_episode_start)],
            "guidance_weight_w": 1.0,
            "weight_schedule": "fixed at 1 through the configured training horizon; no episode-600 cutoff",
            "rule_structure_fixed_for_episode": True,
            "rule_output_varies_with_latest_decision_state": True,
            "routing_guidance": "disabled; original safe-DDQN flow",
        }

    def _analysis_replacements(self, *, previous: dict[str, Any], current_observable_initial_state: dict[str, Any], settings: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "PREVIOUS_COMPLETE_RULE": previous["previous_complete_rule"],
            "PREVIOUS_RULE_APPLICATION_STATUS": previous["previous_rule_application_status"],
            "OBSERVED_TASK_AND_TRAFFIC_SUMMARY": previous["observed_task_and_traffic_summary"],
            "VERIFIED_EPISODE_METRICS": previous["metrics"],
            "MOVEMENT_AND_RULE_APPLICATION_SUMMARIES": previous["movement_and_rule_application_summaries"],
            "FOV_COM_AND_SLOT_SERVICE_SUMMARIES": previous["fov_com_and_slot_service_summaries"],
            "REPRESENTATIVE_CASES": previous["representative_cases"],
        }

    def _render_analysis_prompt(self, *, system_prompt: str, analysis_template: str, previous: dict[str, Any], current_observable_initial_state: dict[str, Any], settings: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        cases = previous.get("representative_cases", [])
        last_prompt = ""
        for maximum in (3, 2, 1, 0):
            candidate = dict(previous)
            candidate["representative_cases"] = _diverse_case_subset(
                [item for item in cases if isinstance(item, dict)], maximum
            )
            last_prompt = render_prompt(
                analysis_template,
                self._analysis_replacements(
                    previous=candidate,
                    current_observable_initial_state=current_observable_initial_state,
                    settings=settings,
                ),
            )
            budget = prompt_token_budget(
                system_prompt, last_prompt,
                reserved_output_tokens=self.config.analysis_max_output_tokens,
                context_window_tokens=self.config.context_window_tokens,
                safety_margin_tokens=self.config.safety_margin_tokens,
                encoding_name=self.config.tokenizer_encoding,
            )
            if budget["fits"]:
                return last_prompt
        return last_prompt

    def _guidance_replacements(self, *, analysis: str, previous: dict[str, Any], environment_constants: dict[str, Any], current_observable_initial_state: dict[str, Any], settings: dict[str, Any], validation_feedback: str) -> dict[str, Any]:
        return {
            "ANALYSIS_STAGE_OUTPUT": analysis,
            "VERIFIED_KEY_EVIDENCE_SUMMARY": {
                "episode_metrics": previous["metrics"],
                "movement_and_rule_application": previous["movement_and_rule_application_summaries"],
                "service": previous["fov_com_and_slot_service_summaries"],
            },
            "RUNTIME_FEATURE_CATALOG": environment_constants["runtime_feature_catalog"],
            "RULE_LANGUAGE_AND_OUTPUT_SCHEMA": environment_constants["rule_language"],
            "PREVIOUS_COMPLETE_RULE": previous["previous_complete_rule"],
            "PREVIOUS_RULE_APPLICATION_STATUS": previous["previous_rule_application_status"],
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "VALIDATION_FEEDBACK_OR_EMPTY": validation_feedback or "No local validation feedback is present.",
        }

    def begin_episode(self, *, episode_index: int, total_joint_transitions: int, sigma_episode_start: float, **kwargs: Any) -> dict[str, Any]:
        self.movement_records = []
        self.routing_records = []
        self.slot_service_records = []
        self._last_runtime_rows = []
        self._compiled_rule = None
        self._upcoming_episode = int(episode_index) + 1
        self._total_joint_transitions = int(total_joint_transitions)
        if self.evaluation:
            self.active_guidance = self._unguided("evaluation_disabled")
            return self.active_guidance
        if int(total_joint_transitions) < int(self.config.warmup_joint_transitions):
            self.active_guidance = self._unguided("warmup")
            return self.active_guidance
        result = LLMExplorerController.begin_episode(
            self,
            episode_index=episode_index,
            total_joint_transitions=total_joint_transitions,
            sigma_episode_start=sigma_episode_start,
            **kwargs,
        )
        candidate = result.get("td3_bias")
        if result.get("enabled") and not result.get("td3_fallback") and isinstance(candidate, dict):
            self._compiled_rule = compile_state_rule(candidate)
            result["rule"] = self._compiled_rule.rule
            result["source_episode"] = int(episode_index) + 1
        else:
            result["rule"] = None
        result.pop("td3_bias", None)
        result.update(
            guidance_weight_w=(1.0 if self._compiled_rule is not None else 0.0),
            sigma_episode_start=float(sigma_episode_start),
            routing_prior=None,
            routing_fallback=False,
            routing_reason="routing_guidance_disabled",
        )
        self.active_guidance = result
        print(
            f"[LLM state rule] episode={episode_index + 1} stage=complete "
            f"enabled={str(bool(result.get('enabled'))).lower()} "
            f"fallback={str(bool(result.get('td3_fallback'))).lower()} w={result['guidance_weight_w']:.1f} "
            f"failed_attempts={self.statistics['failed_attempts_total']}",
            flush=True,
        )
        return result

    def evaluate_movement_bias(
        self,
        env: Any,
        snapshot: Mapping[str, Any],
        movement_mask: Any,
        actor_action: Any,
        *,
        sigma_episode_start: float,
        episode_seconds: float,
    ) -> np.ndarray:
        if self._compiled_rule is None:
            self._last_runtime_rows = []
            return np.zeros(NUM_UAV * 3, dtype=np.float32)
        try:
            features = build_runtime_feature_rows(
                env, snapshot, movement_mask, actor_action,
                episode_seconds=episode_seconds,
            )
            bias, rows = evaluate_rule_for_joint_action(
                self._compiled_rule,
                features,
                movement_mask,
                sigma_episode_start=sigma_episode_start,
                weight=1.0,
            )
        except (ArithmeticError, ExplorerValidationError, TypeError, ValueError) as exc:
            mask = np.asarray(movement_mask, dtype=bool).reshape(NUM_UAV)
            bias = np.zeros(NUM_UAV * 3, dtype=np.float32)
            rows = [
                {
                    "uav_id": uav_id,
                    "movement_mask": bool(mask[uav_id]),
                    "raw_bias": [0.0, 0.0, 0.0],
                    "bounded_bias": [0.0, 0.0, 0.0],
                    "effective_bias": [0.0, 0.0, 0.0],
                    "clipped_components": 0,
                    "runtime_failure": (
                        f"{type(exc).__name__}: {exc}" if mask[uav_id] else None
                    ),
                    "trace": {
                        "conditions_evaluated": 0,
                        "conditions_true": 0,
                        "short_circuits": 0,
                    },
                    "features": None,
                }
                for uav_id in range(NUM_UAV)
            ]
        self._last_runtime_rows = rows
        failures = sum(item.get("runtime_failure") is not None for item in rows)
        self.runtime_failure_total += failures
        return bias

    def consume_last_runtime_rows(self) -> list[dict[str, Any]]:
        return _json_clone(self._last_runtime_rows)

    def record_slot_service(self, record: dict[str, Any]) -> None:
        self.slot_service_records.append(_json_clone(record))

    def complete_episode(self, *, episode_index: int, metrics: dict[str, Any]) -> None:
        movement, cases = state_rule_movement_summary(self.movement_records)
        communication = communication_bottleneck_summary(self.routing_records, metrics)
        service = slot_service_summary(self.slot_service_records)
        roles = {
            key: value["role_decision_counts"]
            for key, value in movement["by_uav"].items()
        }
        feedback = {
            "schema_version": LLM_STATE_RULE_VERSION,
            "summary_version": LLM_STATE_RULE_SUMMARY_VERSION,
            "metrics": _json_clone(metrics),
            "observed_task_and_traffic_summary": {
                "role_decision_counts_by_uav": roles,
                "source_and_task_metrics": {
                    key: value for key, value in metrics.items()
                    if key.startswith(("fov_", "com_"))
                },
            },
            "movement_and_rule_application_summaries": movement,
            "fov_com_and_slot_service_summaries": {
                "routing_decision_observations": communication,
                "actual_within_slot_service": service,
            },
            "representative_cases": cases,
            "applied_guidance": self._serializable_guidance(self.active_guidance),
        }
        self.previous_episode_feedback = _json_clone(feedback)
        if self.root is not None:
            episode_root = self.root / f"episode_{episode_index + 1:06d}"
            _atomic_json(
                episode_root / "feedback.json",
                {
                    **feedback,
                    "movement_decisions": self.movement_records,
                    "routing_decisions": self.routing_records,
                    "slot_service_records": self.slot_service_records,
                },
            )
            _atomic_json(self.root / "state.json", self.state_dict())
        print(
            f"[LLM state rule] episode={episode_index + 1} summary "
            f"controlled={movement['totals'].get('controlled_decisions', 0)} "
            f"runtime_failures={movement['totals'].get('runtime_failures', 0)} "
            f"requests={self.statistics['requests_total']} "
            f"failed_attempts={self.statistics['failed_attempts_total']} "
            f"fallback_episodes={self.statistics['td3_fallback_episodes_total']}",
            flush=True,
        )


def explorer_metadata_state_rule(config: LLMStateRuleConfig) -> dict[str, Any]:
    return {
        "schema_version": LLM_STATE_RULE_VERSION,
        "method_id": LLM_STATE_RULE_METHOD_ID,
        "prompt_version": config.prompt_version,
        "summary_version": LLM_STATE_RULE_SUMMARY_VERSION,
        "cache_version": LLM_STATE_RULE_CACHE_VERSION,
        "rule_schema_version": STATE_RULE_SCHEMA_VERSION,
        "rule_language_version": STATE_RULE_LANGUAGE_VERSION,
        "feature_catalog_version": STATE_RULE_FEATURE_CATALOG_VERSION,
        "model": config.model,
        "warmup_joint_transitions": config.warmup_joint_transitions,
        "episode_joint_transitions": config.episode_joint_transitions,
        "update_interval_episodes": 1,
        "guidance_weight_w": 1.0,
        "annealing_end_episode": None,
        "analysis_reserved_output_tokens": config.analysis_max_output_tokens,
        "rule_reserved_output_tokens": config.guidance_max_output_tokens,
        "context_window_tokens": config.context_window_tokens,
        "safety_margin_tokens": config.safety_margin_tokens,
        "legacy_input_hard_limit_tokens": None,
        "td3_intervention": "latest-state shared rule; raw then host clip to episode-start sigma; fixed w=1",
        "routing_intervention": "none_original_safe_ddqn_epsilon_greedy",
        "evaluation_calls_llm": False,
    }
