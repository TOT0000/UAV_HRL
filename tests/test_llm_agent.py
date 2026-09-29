import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from smolagents import ChatMessage, ChatMessageToolCall, MessageRole, Model, TokenUsage
from smolagents.models import ChatMessageToolCallFunction

import llm_agent
from llm_streaming import capture_chat_stream
from llm_agent import AgentWorkspace, ModelCallBudget, run_agent
from llm_candidate import execute_candidate_isolated
from llm_design import APIError, evaluate_candidate
from test_llm_design import ChunkedResponse, _candidate, _fixed_artifact


class ScriptedToolModel(Model):
    def __init__(self, actions, model_id="mock/tool-model"):
        super().__init__(model_id=model_id)
        self.actions = list(actions)
        self.calls = []

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": [tool.name for tool in tools_to_call_from or []],
            }
        )
        if not self.actions:
            raise AssertionError("unexpected model call")
        name, arguments = self.actions.pop(0)
        call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            id=f"mock-call-{len(self.calls)}",
            type="function",
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[call],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class MultiToolModel(ScriptedToolModel):
    """Return one or more ordered tool calls for each scripted generation."""

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": [tool.name for tool in tools_to_call_from or []],
            }
        )
        if not self.actions:
            raise AssertionError("unexpected model call")
        actions = self.actions.pop(0)
        calls = [
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
                id=f"mock-call-{len(self.calls)}-{index}",
                type="function",
            )
            for index, (name, arguments) in enumerate(actions)
        ]
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=calls,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class TransportFailureModel(Model):
    def __init__(self):
        super().__init__(model_id="mock/transport-failure")
        self.calls = 0

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        self.calls += 1
        raise APIError("fixture transport failed", category="stream_connection_error")


class PlainTextModel(Model):
    def __init__(self, responses, model_id="mock/plain-text"):
        super().__init__(model_id=model_id)
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": [tool.name for tool in tools_to_call_from or []],
            }
        )
        if not self.responses:
            raise AssertionError("unexpected model call")
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=self.responses.pop(0),
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class RequestAwareToolModel(Model):
    """Script tool calls only after assertions over the actual request."""

    def __init__(self, steps, model_id="mock/request-aware"):
        super().__init__(model_id=model_id)
        self.steps = list(steps)
        self.calls = []

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        request_text = json.dumps(
            [message.dict() for message in messages],
            ensure_ascii=False,
            default=str,
        )
        self.calls.append(
            {
                "messages": messages,
                "request_text": request_text,
                "tools": [tool.name for tool in tools_to_call_from or []],
            }
        )
        if not self.steps:
            raise AssertionError("unexpected model call")
        action = self.steps.pop(0)(request_text)
        calls = [
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
                id=f"aware-call-{len(self.calls)}-{index}",
                type="function",
            )
            for index, (name, arguments) in enumerate(action)
        ]
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=calls,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class QueryThenTransportFailureModel(Model):
    def __init__(self):
        super().__init__(model_id="mock/query-then-transport-failure")
        self.calls = []

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[
                    ChatMessageToolCall(
                        function=ChatMessageToolCallFunction(
                            name="query_samples",
                            arguments={
                                "fields": ["obs.movement_mask"],
                                "start": 0,
                                "limit": 20,
                                "condition_field": None,
                                "condition": None,
                            },
                        ),
                        id="query-before-failure",
                        type="function",
                    )
                ],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        raise APIError("fixture transport failed", category="stream_connection_error")


def _candidate_id(candidate):
    return f"candidate-{llm_agent._content_hash(candidate)[:12]}"


def _workspace(tmp_path, fixed, *, name="workspace", max_model_calls=20):
    return AgentWorkspace(
        directory=tmp_path / name,
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock/tool-model",
        beta=1.0,
        batch_size=2,
        worker_timeout=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        max_model_calls=max_model_calls,
    )


def _request_settings(*, context_length=100_000, max_output_tokens=1_024):
    return {
        "base_url": "http://127.0.0.1:1234/v1",
        "context_length": context_length,
        "max_output_tokens": max_output_tokens,
        "temperature": llm_agent.DEFAULT_TEMPERATURE,
        "seed": llm_agent.DEFAULT_SEED,
        "reasoning_effort": None,
        "timeout": llm_agent.DEFAULT_API_TIMEOUT_SECONDS,
        "connect_timeout": llm_agent.DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
        "total_timeout": llm_agent.DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
        "progress_interval": llm_agent.DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    }


def _tree_hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tool_sse(*, model, finish_reason, arguments, refusal=None, done=True):
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "stream-call-1",
                "type": "function",
                "function": {
                    "name": "inspect_interface",
                    "arguments": arguments,
                },
            }
        ]
    }
    if refusal is not None:
        delta["refusal"] = refusal
    event = {
        "model": model,
        "choices": [{"delta": delta, "finish_reason": finish_reason}],
    }
    chunks = [f"data: {json.dumps(event)}\n\n".encode("utf-8")]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks


def _fragmented_tool_sse(*, model, tool_name, arguments, finish_reason):
    serialized = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    cut_points = sorted(
        {
            1,
            max(2, len(serialized) // 5),
            max(3, len(serialized) // 2),
            max(4, len(serialized) - 3),
            len(serialized),
        }
    )
    fragments = []
    start = 0
    for end in cut_points:
        if end > start:
            fragments.append(serialized[start:end])
            start = end
    chunks = []
    for index, fragment in enumerate(fragments):
        function = {"arguments": fragment}
        call = {"index": 0, "function": function}
        if index == 0:
            call.update({"id": "stream-candidate-1", "type": "function"})
            function["name"] = tool_name
        event = {
            "model": model,
            "choices": [
                {
                    "delta": {"tool_calls": [call]},
                    "finish_reason": (
                        finish_reason if index == len(fragments) - 1 else None
                    ),
                }
            ],
        }
        chunks.append(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def _plain_text_sse(*, model, content):
    event = {
        "model": model,
        "choices": [
            {
                "delta": {"content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return [
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode(),
        b"data: [DONE]\n\n",
    ]


def _direct_streaming_model(
    tmp_path, fixed, *, chunks, provider="openai", model_id=None
):
    workspace = _workspace(tmp_path, fixed, name=f"stream-{provider}")
    model_id = model_id or (
        "gpt-4o" if provider == "openai" else "openai/gpt-oss-20b"
    )
    class FixtureClient:
        def __init__(self):
            self.base_url = "http://127.0.0.1:1234/v1"
            self.token = None
            self.timeout = 1
            self.connect_timeout = 1
            self.total_timeout = 2
            self.progress_interval = 60
            self.response = ChunkedResponse(chunks)

        def _open_stream(self, request, **kwargs):
            return self.response

        @staticmethod
        def _safe(value):
            return str(value)

    budget = ModelCallBudget(2)
    model = llm_agent.StreamingProviderModel(
        workspace=workspace,
        client=FixtureClient(),
        provider=provider,
        model_id=model_id,
        budget=budget,
        context_length=100_000,
        max_output_tokens=1_024,
        temperature=0.3,
        seed=20260927,
        reasoning_effort=None,
    )
    return workspace, model, budget


def test_mock_agent_autonomously_queries_revises_tests_and_approves(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    bad = _candidate(name="bad")
    bad["code"] = "import os\ndef compute_extra_state(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)\n"
    good = _candidate(name="good")
    bad_id = _candidate_id(bad)
    good_id = _candidate_id(good)
    model = ScriptedToolModel(
        [
            ("query_samples", {"fields": ["obs.movement_mask"], "start": 0, "limit": 1, "condition_field": None, "condition": None}),
            ("submit_candidate", {"candidate": bad, "parent_candidate_id": None}),
            ("test_candidate", {"candidate_id": bad_id}),
            ("submit_candidate", {"candidate": good, "parent_candidate_id": bad_id}),
            ("test_candidate", {"candidate_id": good_id}),
            ("formal_evaluate", {"candidate_id": good_id}),
        ]
    )
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=10,
        context_length=100_000,
        output_dir=tmp_path / "agent",
    )
    assert result["status"] == "approved"
    assert len(model.calls) == 6
    assert Path(result["approved_artifact"], "artifact.json").is_file()
    state = json.loads((tmp_path / "agent" / "agent_state.json").read_text())
    assert state["approved_candidate_id"] == good_id
    assert state["candidates"][bad_id]["validation_status"] == "static_failed"
    assert state["candidates"][good_id]["parent_candidate_id"] == bad_id
    assert [item["tool"] for item in state["tool_operations"]] == [
        "query_samples",
        "submit_candidate",
        "test_candidate",
        "submit_candidate",
        "test_candidate",
        "formal_evaluate",
    ]
    assert all(item["status"] == "completed" for item in state["tool_operations"])
    assert len(state["framework_tool_calls"]) == 6
    assert all(item["status"] == "completed" for item in state["framework_tool_calls"])


def test_unapproved_final_answer_is_rejected_until_budget_pause(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    final_model = ScriptedToolModel(
        [
            ("final_answer", {"answer": "I am done and approve my own design."}),
            ("final_answer", {"answer": "I still want to stop."}),
            ("final_answer", {"answer": "Stop now."}),
        ]
    )
    final = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=final_model.model_id,
        model_backend=final_model,
        max_model_calls=3,
        context_length=100_000,
        output_dir=tmp_path / "final-only",
    )
    assert final["status"] == "paused_budget_exhausted"
    assert final["approved_artifact"] is None
    assert final["model_calls_used"] == 3
    assert len(final_model.calls) == 3
    second_request = json.dumps(
        [message.dict() for message in final_model.calls[1]["messages"]],
        ensure_ascii=False,
        default=str,
    )
    assert "completion_rejected" in second_request
    assert "no_host_approved_artifact" in second_request
    assert "I am done and approve my own design." in second_request
    state = json.loads((tmp_path / "final-only" / "agent_state.json").read_text())
    assert len(state["completion_rejections"]) == 3
    assert {item["status"] for item in state["framework_tool_calls"]} == {
        "completion_rejected"
    }

    inspect_model = ScriptedToolModel(
        [("inspect_interface", {"section": "evaluation"})]
    )
    exhausted = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=inspect_model.model_id,
        model_backend=inspect_model,
        max_model_calls=1,
        context_length=100_000,
        output_dir=tmp_path / "budget",
    )
    assert exhausted["status"] == "paused_budget_exhausted"
    assert exhausted["model_calls_used"] == 1
    assert len(inspect_model.calls) == 1
    assert exhausted["approved_artifact"] is None


def test_completion_rejection_feedback_allows_submit_and_full_test(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="after-rejected-completion")
    candidate_id = _candidate_id(candidate)
    model = ScriptedToolModel(
        [
            (
                "submit_candidate",
                {"candidate": candidate, "parent_candidate_id": None},
            ),
            ("final_answer", {"answer": "Next I would design a candidate."}),
            ("test_candidate", {"candidate_id": candidate_id}),
        ]
    )
    directory = tmp_path / "rejected-then-test"
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=3,
        context_length=100_000,
        output_dir=directory,
    )
    assert result["status"] == "paused_budget_exhausted"
    assert len(model.calls) == 3
    next_request = json.dumps(
        [message.dict() for message in model.calls[2]["messages"]],
        ensure_ascii=False,
        default=str,
    )
    assert "completion_rejected" in next_request
    assert "Next I would design a candidate." in next_request
    assert candidate_id in next_request
    assert "static_passed" in next_request
    state = json.loads((directory / "agent_state.json").read_text())
    rejection = state["completion_rejections"][0]["result"]
    assert rejection["current_candidate_id"] == candidate_id
    assert rejection["validation_status"] == "static_passed"
    assert rejection["model_calls_remaining"] == 1
    test_result = state["tool_operations"][-1]["result_summary"]
    assert test_result["status"] == "passed"
    assert state["candidates"][candidate_id]["validation_status"] == "full_passed"
    assert state["approved_candidate_id"] is None


def test_unapproved_final_answer_does_not_skip_later_tools_in_same_response(
    tmp_path,
):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="same-response-after-final")
    candidate_id = _candidate_id(candidate)
    model = MultiToolModel(
        [
            [
                ("final_answer", {"answer": "Premature plan text."}),
                (
                    "submit_candidate",
                    {"candidate": candidate, "parent_candidate_id": None},
                ),
                ("test_candidate", {"candidate_id": candidate_id}),
            ]
        ]
    )
    directory = tmp_path / "multi-final-continues"
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=1,
        context_length=100_000,
        output_dir=directory,
    )
    assert result["status"] == "paused_budget_exhausted"
    state = json.loads((directory / "agent_state.json").read_text())
    assert [item["status"] for item in state["framework_tool_calls"]] == [
        "completion_rejected",
        "completed",
        "completed",
    ]
    assert [item["tool"] for item in state["tool_operations"]] == [
        "submit_candidate",
        "test_candidate",
    ]
    assert state["candidates"][candidate_id]["validation_status"] == "full_passed"


def test_unapproved_final_answer_can_be_followed_by_approval_in_same_response(
    tmp_path,
):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="same-response-host-approval")
    candidate_id = _candidate_id(candidate)
    model = MultiToolModel(
        [
            [
                (
                    "submit_candidate",
                    {"candidate": candidate, "parent_candidate_id": None},
                )
            ],
            [
                ("final_answer", {"answer": "I think this is already complete."}),
                ("formal_evaluate", {"candidate_id": candidate_id}),
                ("inspect_interface", {"section": "overview"}),
            ],
        ]
    )
    directory = tmp_path / "rejected-final-then-approved"
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=directory,
    )
    assert result["status"] == "approved"
    assert len(model.calls) == 2
    state = json.loads((directory / "agent_state.json").read_text())
    assert state["approved_candidate_id"] == candidate_id
    assert [item["status"] for item in state["framework_tool_calls"]] == [
        "completed",
        "completion_rejected",
        "completed",
        "skipped_after_approval",
    ]
    assert [item["tool"] for item in state["tool_operations"]] == [
        "submit_candidate",
        "formal_evaluate",
    ]


def test_plain_text_completion_is_rejected_and_budgeted(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    model = PlainTextModel(
        [
            "I am finished without using a tool.",
            "I still decline to use a tool.",
        ]
    )
    directory = tmp_path / "plain-text-rejected"
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=directory,
    )
    assert result["status"] == "paused_budget_exhausted"
    assert result["model_calls_used"] == 2
    assert len(model.calls) == 2
    state = json.loads((directory / "agent_state.json").read_text())
    assert [item["source"] for item in state["completion_rejections"]] == [
        "plain_text_response",
        "plain_text_response",
    ]
    assert all(
        item["status"] == "completion_rejected"
        for item in state["framework_tool_calls"]
    )


def test_completion_rejection_history_survives_resume_and_budget_extension(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "completion-resume"
    first = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock/tool-model",
        model_backend=ScriptedToolModel(
            [("final_answer", {"answer": "first premature stop"})]
        ),
        max_model_calls=1,
        context_length=100_000,
        output_dir=directory,
    )
    assert first["status"] == "paused_budget_exhausted"
    resumed_model = ScriptedToolModel(
        [("final_answer", {"answer": "second premature stop"})]
    )
    resumed = run_agent(
        resume=directory,
        additional_model_calls=1,
        model_backend=resumed_model,
    )
    assert resumed["status"] == "paused_budget_exhausted"
    assert resumed["model_calls_used"] == 2
    request = json.dumps(
        [message.dict() for message in resumed_model.calls[0]["messages"]],
        ensure_ascii=False,
        default=str,
    )
    assert "first premature stop" in request
    assert "completion_rejected" in request
    state = json.loads((directory / "agent_state.json").read_text())
    assert len(state["completion_rejections"]) == 2
    assert state["budget_extension_events"][-1]["additional_model_calls"] == 1


def test_candidate_reports_cache_and_approval_are_bound_to_content(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = AgentWorkspace(
        directory=tmp_path / "workspace",
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock",
        beta=1.0,
        batch_size=2,
        worker_timeout=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        max_model_calls=20,
    )
    failing = _candidate(passing=False, name="failing")
    passing = _candidate(passing=True, name="passing")
    first = workspace.submit_candidate(failing, None)
    second = workspace.submit_candidate(passing, first["candidate_id"])
    duplicate = workspace.submit_candidate(json.loads(json.dumps(passing)), None)
    assert duplicate["status"] == "duplicate_candidate"
    assert duplicate["candidate_id"] == second["candidate_id"]
    assert second["parent_candidate_id"] == first["candidate_id"]
    first_eval = workspace.formal_evaluate(first["candidate_id"])
    first_cached = workspace.formal_evaluate(first["candidate_id"])
    second_eval = workspace.formal_evaluate(second["candidate_id"])
    assert first_eval["passed"] is False
    assert first_cached["cache_hit"] is True
    assert second_eval["passed"] is True
    assert workspace.state["approved_candidate_id"] == second["candidate_id"]
    cache_candidates = {item["candidate_id"] for item in workspace.state["evaluation_cache"].values()}
    assert cache_candidates == {first["candidate_id"], second["candidate_id"]}
    artifact = json.loads(Path(workspace.state["approved_artifact"], "artifact.json").read_text())
    assert artifact["provenance"]["candidate_id"] == second["candidate_id"]

    stored_report_path = workspace.state["candidates"][second["candidate_id"]][
        "evaluations"
    ][workspace._evaluation_cache_key(workspace.state["candidates"][second["candidate_id"]])]
    stored_report = json.loads((workspace.directory / stored_report_path).read_text())
    extra, _, _ = execute_candidate_isolated(
        passing,
        workspace.obs_arrays,
        workspace.constants,
        timeout=workspace.worker_timeout,
        diagnostic_contract=workspace.diagnostic_contract,
    )
    legacy_report = evaluate_candidate(
        workspace.context,
        passing,
        extra,
        beta=workspace.beta,
        batch_size=workspace.batch_size,
        absolute_tolerance=workspace.absolute_tolerance,
        relative_tolerance=workspace.relative_tolerance,
        obs_arrays=workspace.obs_arrays,
        constants_metadata=workspace.constants,
    )
    assert stored_report["passed"] == legacy_report["passed"]
    assert stored_report["by_lambda"] == legacy_report["by_lambda"]


def test_full_candidate_test_uses_all_samples_and_detects_later_variation(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="full-scope-diagnostics")
    candidate = _candidate(
        name="time-varies-after-first-sample",
        code=(
            "def compute_extra_state(obs, constants):\n"
            "    value = np.clip(obs[\"snapshot_time_s\"][0] / constants[\"episode_seconds\"], 0.0, 1.0)\n"
            "    return np.asarray([value], dtype=np.float32)\n"
        ),
    )
    candidate["features"][0].update(
        {
            "name": "normalized_current_time",
            "description": "Fixture feature that is zero initially and varies later.",
            "source_fields": [
                "obs.snapshot_time_s",
                "constants.episode_seconds",
            ],
            "formula": "clip(snapshot_time_s / episode_seconds, 0, 1)",
            "missing_data_rule": "The current snapshot time is always present.",
        }
    )
    submitted = workspace.submit_candidate(candidate, None)
    result = workspace.test_candidate(submitted["candidate_id"])
    total = int(workspace.fixed_metadata["sample_count"])
    feature = result["numeric_diagnostics"]["features"][0]
    assert result["status"] == "passed"
    assert result["test_scope"] == "full_fixed_samples"
    assert result["fixed_sample_count"] == total
    assert result["tested_sample_count"] == total
    assert result["successful_output_sample_count"] == total
    assert result["complete_fixed_sample"] is True
    assert feature["constant_on_fixed_samples"] is False
    assert feature["standard_deviation"] > 0.0
    assert result["checks_not_run"] == ["formal Lipschitz evaluation"]
    assert workspace.approved is False


@pytest.mark.parametrize("sample_count", [0, 1])
def test_numeric_diagnostics_do_not_call_insufficient_samples_constant(
    tmp_path, sample_count
):
    candidate = _candidate(name="one-sample-diagnostic")
    diagnostics = llm_agent.candidate_numeric_diagnostics(
        original_state=np.zeros((sample_count, 2), dtype=np.float32),
        extra_state=np.zeros((sample_count, 1), dtype=np.float32),
        candidate=candidate,
    )
    feature = diagnostics["features"][0]
    assert diagnostics["sample_count"] == sample_count
    assert feature["constant_on_fixed_samples"] is None
    assert "fewer than two" in feature["constant_diagnostic_reason"]
    assert any("variation cannot be determined" in item for item in diagnostics["warnings"])


def test_failed_full_test_has_no_complete_numeric_diagnostics(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="failed-full-scope")
    candidate = _candidate(
        name="runtime-failure",
        code=(
            "def compute_extra_state(obs, constants):\n"
            "    value = float(obs[\"state\"][0]) / 0.0\n"
            "    return np.asarray([value], dtype=np.float32)\n"
        ),
    )
    candidate["features"][0].update(
        {
            "name": "runtime_failure",
            "description": "Fixture that fails at runtime.",
            "source_fields": ["obs.state"],
            "formula": "state[0] / 0 fixture",
        }
    )
    submitted = workspace.submit_candidate(candidate, None)
    result = workspace.test_candidate(submitted["candidate_id"])
    assert result["status"] == "failed"
    assert result["complete_fixed_sample"] is False
    assert result["tested_sample_count"] == workspace.fixed_metadata["sample_count"]
    assert result["successful_output_sample_count"] == 0
    assert "numeric_diagnostics" not in result
    assert "full fixed-sample numeric diagnostics" in result["checks_not_run"]


def test_legacy_small_cache_is_not_reused_as_full_test(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="legacy-small-cache")
    submitted = workspace.submit_candidate(_candidate(name="legacy-cache"), None)
    record = workspace.state["candidates"][submitted["candidate_id"]]
    record["tests"]["legacy-small"] = {
        "status": "passed",
        "mode": "small",
        "report": "old-small-report.json",
        "tool_result": {
            "status": "passed",
            "candidate_id": submitted["candidate_id"],
            "sample_count": 1,
        },
    }
    result = workspace.test_candidate(submitted["candidate_id"])
    assert result["status"] == "passed"
    assert result["cache_hit"] is False
    assert result["tested_sample_count"] == workspace.fixed_metadata["sample_count"]
    assert workspace.approved is False


def test_resume_dry_run_preserves_candidate_and_does_not_interrupt_source(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "resume"
    workspace = AgentWorkspace(
        directory=directory,
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock",
        beta=1.0,
        batch_size=2,
        worker_timeout=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        max_model_calls=4,
    )
    submitted = workspace.submit_candidate(_candidate(), None)
    workspace.begin_operation("test_candidate", {"candidate_id": submitted["candidate_id"]})
    workspace.state["model_calls_used"] = 2
    workspace.state["request_settings"] = _request_settings()
    workspace._save()
    before = _tree_hashes(directory)
    resumed = run_agent(
        resume=directory,
        dry_run=True,
        context_length=100_000,
        max_output_tokens=1024,
        output_dir=tmp_path / "resume-preview",
    )
    assert resumed["status"] == "dry_run_complete"
    assert Path(resumed["output_directory"]) == (tmp_path / "resume-preview")
    assert resumed["source_run_directory"] == str(directory.resolve())
    assert _tree_hashes(directory) == before
    state = json.loads((directory / "agent_state.json").read_text())
    assert state["current_candidate_id"] == submitted["candidate_id"]
    assert state["model_calls_used"] == 2
    assert state["tool_operations"][-1]["status"] == "running"
    assert (tmp_path / "resume-preview" / "resume_preview_metadata.json").is_file()

    actual = run_agent(
        resume=directory,
        model_backend=ScriptedToolModel(
            [
                ("final_answer", {"answer": "fixture real resume"}),
                ("final_answer", {"answer": "fixture real resume again"}),
            ]
        ),
    )
    assert actual["status"] == "paused_budget_exhausted"
    state_after_resume = json.loads((directory / "agent_state.json").read_text())
    assert state_after_resume["tool_operations"][-1]["status"] == "interrupted"
    assert (
        state_after_resume["tool_operations"][-1]["result_summary"][
            "reusable_as_pass"
        ]
        is False
    )


def test_new_candidate_keeps_parent_issues_unverified_until_full_test(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = AgentWorkspace(
        directory=tmp_path / "issue-state",
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock",
        beta=1.0,
        batch_size=2,
        worker_timeout=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        max_model_calls=5,
    )
    bad = _candidate(name="bad-parent")
    bad["code"] = (
        "import os\n"
        "def compute_extra_state(obs, constants):\n"
        "    return np.asarray([0.0], dtype=np.float32)\n"
    )
    parent = workspace.submit_candidate(bad, None)
    child = workspace.submit_candidate(
        _candidate(name="corrected-child"), parent["candidate_id"]
    )
    record = workspace.state["candidates"][child["candidate_id"]]
    assert record["inherited_issue_context"]
    assert {item["status"] for item in record["inherited_issue_context"]} == {
        "resolved"
    }
    assert workspace.test_candidate(child["candidate_id"])["status"] == "passed"
    assert {item["status"] for item in record["inherited_issue_context"]} == {
        "resolved"
    }


def test_tools_reject_paths_and_sample_queries_only_expose_current_adapter(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = AgentWorkspace(
        directory=tmp_path / "controlled",
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock",
        beta=1.0,
        batch_size=2,
        worker_timeout=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        max_model_calls=2,
    )
    invalid = workspace.query_samples(
        fields=["next_state"], start=0, limit=1, condition_field=None, condition=None
    )
    assert invalid["status"] == "invalid_arguments"
    current = workspace.query_samples(
        fields=["obs.movement_mask", "obs.sr_id", "obs.sr_observable"],
        start=0,
        limit=1,
        condition_field="obs.movement_mask",
        condition="all_false",
    )
    assert current["status"] == "ok"
    assert current["feature_input_timing"] == "action-pre current-only"
    assert set(current["samples"][0]["feature_input_data"]) == {
        "obs.movement_mask",
        "obs.sr_id",
        "obs.sr_observable",
    }
    with pytest.raises(ValueError, match="unknown candidate id"):
        workspace.get_history("C:/arbitrary/path", "candidate")

    invalid_parent = workspace.submit_candidate(
        _candidate(), "candidate-not-present"
    )
    assert invalid_parent == {
        "status": "invalid_arguments",
        "error": "parent candidate id is unknown",
    }


def test_model_call_budget_is_exact():
    budget = ModelCallBudget(2, used=1)
    assert budget.consume() == 2
    with pytest.raises(llm_agent.AgentBudgetError):
        budget.consume()


def test_resume_with_exhausted_call_budget_does_not_make_an_extra_call(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "exhausted-resume"
    initial_model = ScriptedToolModel(
        [
            ("inspect_interface", {"section": "evaluation"}),
            ("inspect_interface", {"section": "candidate"}),
        ]
    )
    initial = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=initial_model.model_id,
        model_backend=initial_model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=directory,
    )
    assert initial["model_calls_used"] == 2
    resumed_model = ScriptedToolModel([])
    resumed = run_agent(
        resume=directory,
        model_backend=resumed_model,
    )
    assert resumed["status"] == "paused_budget_exhausted"
    assert resumed["model_calls_used"] == 2
    assert resumed_model.calls == []
    assert "budget is exhausted" in resumed["stop_reason"]


def test_streamed_tool_arguments_execute_only_after_lossless_aggregation(tmp_path):
    result = capture_chat_stream(
        ChunkedResponse(
            [
                b'data: {"model":"openai/gpt-oss-20b","choices":[{"delta":{"reasoning":"checking"}}]}\n\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-7","type":"function","function":{"name":"query_","arguments":"{\\"fields\\":["}}]}}]}\n\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"samples","arguments":"\\"obs.movement_mask\\"],\\"start\\":0,\\"limit\\":1,\\"condition_field\\":null,\\"condition\\":null}"}}]},"finish_reason":"tool_calls"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        ),
        directory=tmp_path,
        idle_timeout=1,
        total_timeout=2,
        progress_interval=60,
    )
    assert result["reasoning"] == "checking"
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [
        {
            "index": 0,
            "id": "call-7",
            "type": "function",
            "function": {
                "name": "query_samples",
                "arguments": '{"fields":["obs.movement_mask"],"start":0,"limit":1,"condition_field":null,"condition":null}',
            },
        }
    ]
    assert json.loads(result["tool_calls"][0]["function"]["arguments"])["limit"] == 1


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("lmstudio", "openai/gpt-oss-20b"),
        ("lmstudio", "qwen/qwen3.5-9b"),
        ("lmstudio", "google/gemma-4-e4b"),
        ("openai", "gpt-4o"),
    ],
)
def test_complete_plain_text_stream_becomes_host_completion_request(
    tmp_path, provider, model_id
):
    fixed = _fixed_artifact(tmp_path)
    content = "I plan to continue later, but I am stopping now."
    workspace, model, budget = _direct_streaming_model(
        tmp_path,
        fixed,
        chunks=_plain_text_sse(model=model_id, content=content),
        provider=provider,
        model_id=model_id,
    )
    response = model.generate(
        [ChatMessage(role=MessageRole.USER, content="fixture")],
        tools_to_call_from=llm_agent.build_agent_tools(workspace),
    )
    assert response.content == content
    assert response.tool_calls[0].function.name == "final_answer"
    assert response.tool_calls[0].function.arguments == {"answer": content}
    saved = json.loads(
        (workspace.directory / "model_call_001" / "response.json").read_text()
    )
    assert saved["content"] == content
    assert saved["tool_calls"] == []
    assert budget.used == 1


def test_submit_candidate_api_schema_embeds_canonical_object_schema(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="candidate-tool-schema")
    tool = next(
        item
        for item in llm_agent.build_agent_tools(workspace)
        if item.name == "submit_candidate"
    )
    schema = llm_agent.get_tool_json_schema(tool)
    parameters = schema["function"]["parameters"]
    candidate = parameters["properties"]["candidate"]
    canonical = llm_agent.candidate_schema()
    assert candidate["type"] == "object"
    assert candidate["additionalProperties"] is False
    assert candidate["required"] == canonical["required"]
    assert candidate["properties"]["code"]["type"] == "string"
    assert candidate["properties"]["features"]["items"]["type"] == "object"
    assert "$ref" not in json.dumps(candidate)
    assert "candidate" in parameters["required"]
    assert "candidate_json" not in parameters["properties"]

    test_tool = next(
        item
        for item in llm_agent.build_agent_tools(workspace)
        if item.name == "test_candidate"
    )
    test_parameters = llm_agent.get_tool_json_schema(test_tool)["function"][
        "parameters"
    ]
    assert test_parameters["required"] == ["candidate_id"]
    assert set(test_parameters["properties"]) == {"candidate_id"}


@pytest.mark.parametrize(
    ("provider", "model_id", "finish_reason"),
    [
        ("lmstudio", "openai/gpt-oss-20b", "stop"),
        ("lmstudio", "qwen/qwen3.5-9b", "stop"),
        ("lmstudio", "google/gemma-4-e4b", "stop"),
        ("openai", "gpt-4o", "tool_calls"),
    ],
)
def test_nested_candidate_object_survives_fragmented_stream_exactly(
    tmp_path, provider, model_id, finish_reason
):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name=f"streamed-{model_id}")
    candidate["code"] = (
        "def compute_extra_state(obs, constants):\n"
        "    # double \" quote, single ' quote, path C:\\\\tmp\\\\feature\n"
        "    value = np.clip(obs['state'][0] * obs[\"state\"][0], 0.0, 1.0)\n"
        "    return np.asarray([value], dtype=np.float32)\n"
    )
    arguments = {"candidate": candidate, "parent_candidate_id": None}
    chunks = _fragmented_tool_sse(
        model=model_id,
        tool_name="submit_candidate",
        arguments=arguments,
        finish_reason=finish_reason,
    )
    workspace, model, budget = _direct_streaming_model(
        tmp_path,
        fixed,
        chunks=chunks,
        provider=provider,
        model_id=model_id,
    )
    response = model.generate(
        [ChatMessage(role=MessageRole.USER, content="fixture")],
        tools_to_call_from=llm_agent.build_agent_tools(workspace),
    )
    decoded = response.tool_calls[0].function.arguments
    assert decoded == arguments
    assert decoded["candidate"]["code"] == candidate["code"]
    tool = next(
        item
        for item in llm_agent.build_agent_tools(workspace)
        if item.name == "submit_candidate"
    )
    submitted = json.loads(tool.forward(**decoded))
    assert submitted["status"] == "static_passed"
    assert workspace.state["candidates"][submitted["candidate_id"]]["candidate"][
        "code"
    ] == candidate["code"]
    tested = workspace.test_candidate(submitted["candidate_id"])
    assert tested["status"] == "passed"
    assert budget.used == 1


@pytest.mark.parametrize("value", [None, [], '{"schema_version":"nested"}'])
def test_submit_candidate_rejects_non_object_values_clearly(tmp_path, value):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name=f"invalid-{type(value).__name__}")
    tool = next(
        item
        for item in llm_agent.build_agent_tools(workspace)
        if item.name == "submit_candidate"
    )
    result = json.loads(tool.forward(value, None))
    assert result["status"] == "invalid_arguments"
    assert result["candidate_created"] is False
    assert "candidate must be a JSON object" in result["error"]
    if isinstance(value, str):
        assert "JSON-encoded string" in result["error"]
    assert workspace.state["candidates"] == {}


def test_missing_fields_and_invalid_python_never_become_executable(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="invalid-candidates")
    missing = _candidate(name="missing-code")
    missing.pop("code")
    missing_result = workspace.submit_candidate(missing, None)
    assert missing_result["status"] == "static_failed"
    assert missing_result["can_execute"] is False
    assert any("code" in issue["location"] for issue in missing_result["errors"])
    assert workspace.test_candidate(missing_result["candidate_id"])[
        "status"
    ] == "prerequisite_failed"

    invalid = _candidate(name="invalid-python")
    invalid["code"] = (
        "def compute_extra_state(obs, constants):\n"
        "    ctrl = obs[\"movement_mask\"]\n"
        "    return np.asarray([ctrl[0]], dtype=np.float32\n"
    )
    invalid_result = workspace.submit_candidate(invalid, None)
    assert invalid_result["status"] == "static_failed"
    assert invalid_result["can_execute"] is False
    saved = workspace.state["candidates"][invalid_result["candidate_id"]]["candidate"]
    assert saved["code"] == invalid["code"]
    assert any(
        issue["code"] == "STATIC_PYTHON_SYNTAX"
        for issue in invalid_result["errors"]
    )


def test_transport_failure_stops_without_becoming_a_candidate_error(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    model = TransportFailureModel()
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=5,
        context_length=100_000,
        output_dir=tmp_path / "transport-failure",
    )
    state = json.loads(
        (tmp_path / "transport-failure" / "agent_state.json").read_text()
    )
    assert result["status"] == "failed"
    assert model.calls == 1
    assert state["candidates"] == {}
    assert state["tool_operations"] == []
    assert "transport" in state["stop_reason"].lower()


def test_multi_tool_approval_stops_sequentially_and_skips_remaining_call(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="multi-tool-approved")
    candidate_id = _candidate_id(candidate)
    model = MultiToolModel(
        [
            [
                (
                    "submit_candidate",
                    {"candidate": candidate, "parent_candidate_id": None},
                )
            ],
            [
                ("formal_evaluate", {"candidate_id": candidate_id}),
                ("inspect_interface", {"section": "overview"}),
            ],
        ]
    )
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=5,
        context_length=100_000,
        output_dir=tmp_path / "multi-approved",
    )
    assert result["status"] == "approved"
    assert len(model.calls) == 2
    state = json.loads((tmp_path / "multi-approved" / "agent_state.json").read_text())
    assert [item["tool"] for item in state["tool_operations"]] == [
        "submit_candidate",
        "formal_evaluate",
    ]
    assert [item["status"] for item in state["framework_tool_calls"]] == [
        "completed",
        "completed",
        "skipped_after_approval",
    ]
    assert "skipped_after_approval" in state["framework_tool_calls"][-1]["output"]


def test_multi_tool_nonapproval_continues_in_model_order(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(passing=False, name="multi-tool-not-approved")
    candidate_id = _candidate_id(candidate)
    model = MultiToolModel(
        [
            [
                (
                    "submit_candidate",
                    {"candidate": candidate, "parent_candidate_id": None},
                )
            ],
            [
                ("formal_evaluate", {"candidate_id": candidate_id}),
                ("inspect_interface", {"section": "evaluation"}),
            ],
        ]
    )
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=tmp_path / "multi-failed",
    )
    assert result["status"] == "paused_budget_exhausted"
    state = json.loads((tmp_path / "multi-failed" / "agent_state.json").read_text())
    assert [item["tool"] for item in state["tool_operations"]] == [
        "submit_candidate",
        "formal_evaluate",
        "inspect_interface",
    ]
    assert all(item["status"] == "completed" for item in state["framework_tool_calls"])


@pytest.mark.parametrize(
    ("failing_name", "failing_arguments", "expected_text", "expected_operations"),
    [
        ("unknown_fixture_tool", {}, "unknown_fixture_tool", ["submit_candidate"]),
        (
            "inspect_interface",
            {"section": "not-a-section"},
            "section must be",
            ["submit_candidate", "inspect_interface"],
        ),
    ],
)
def test_multi_tool_failure_preserves_prior_result_in_next_model_messages(
    tmp_path, failing_name, failing_arguments, expected_text, expected_operations
):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(passing=False, name=f"before-{failing_name}")
    candidate_id = _candidate_id(candidate)
    model = MultiToolModel(
        [
            [
                (
                    "submit_candidate",
                    {"candidate": candidate, "parent_candidate_id": None},
                ),
                (failing_name, failing_arguments),
                ("inspect_interface", {"section": "evaluation"}),
            ],
            [("final_answer", {"answer": "stop after correcting the tool call"})],
        ]
    )
    directory = tmp_path / f"partial-{failing_name}"
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=directory,
    )
    assert result["status"] == "paused_budget_exhausted"
    assert len(model.calls) == 2
    next_request = json.dumps(
        [message.dict() for message in model.calls[1]["messages"]],
        ensure_ascii=False,
        default=str,
    )
    assert candidate_id in next_request
    assert "mock-call-1-0" in next_request
    assert "mock-call-1-1" in next_request
    assert "mock-call-1-2" in next_request
    assert expected_text in next_request
    assert "completed" in next_request
    assert "failed" in next_request
    assert "skipped_after_tool_error" in next_request
    state = json.loads((directory / "agent_state.json").read_text())
    assert [item["tool"] for item in state["tool_operations"]] == expected_operations
    assert state["tool_operations"][-1]["status"] == (
        "completed" if failing_name == "unknown_fixture_tool" else "failed"
    )
    assert [item["status"] for item in state["framework_tool_calls"]] == [
        "completed",
        "failed",
        "skipped_after_tool_error",
        "completion_rejected",
    ]
    assert state["candidate_order"] == [candidate_id]


@pytest.mark.parametrize(
    ("finish_reason", "arguments", "refusal", "category"),
    [
        ("length", '{"section":"overview"}', None, "output_truncated"),
        ("length", '{"section":', None, "output_truncated"),
        ("stop", '{"section":"overview"}', "I cannot comply", "model_refusal"),
        ("content_filter", '{"section":"overview"}', None, "content_filter"),
        ("unknown_reason", '{"section":"overview"}', None, "invalid_finish_reason"),
    ],
)
def test_stream_completion_failures_never_expose_tool_calls_for_execution(
    tmp_path, finish_reason, arguments, refusal, category
):
    fixed = _fixed_artifact(tmp_path)
    chunks = _tool_sse(
        model="gpt-4o",
        finish_reason=finish_reason,
        arguments=arguments,
        refusal=refusal,
    )
    workspace, model, budget = _direct_streaming_model(
        tmp_path, fixed, chunks=chunks, provider="openai"
    )
    with pytest.raises(APIError) as caught:
        model.generate(
            [ChatMessage(role=MessageRole.USER, content="fixture")],
            tools_to_call_from=llm_agent.build_agent_tools(workspace),
        )
    assert caught.value.category == category
    assert budget.used == 1
    assert workspace.state["tool_operations"] == []
    saved = json.loads(
        (workspace.directory / "model_call_001" / "response.json").read_text()
    )
    assert saved["finish_reason"] == finish_reason
    assert saved["tool_calls"]
    assert saved["refusal"] == refusal


def test_missing_stream_completion_signal_is_saved_and_not_executed(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    chunks = _tool_sse(
        model="gpt-4o",
        finish_reason=None,
        arguments='{"section":"overview"}',
    )
    workspace, model, budget = _direct_streaming_model(
        tmp_path, fixed, chunks=chunks, provider="openai"
    )
    with pytest.raises(APIError) as caught:
        model.generate(
            [ChatMessage(role=MessageRole.USER, content="fixture")],
            tools_to_call_from=llm_agent.build_agent_tools(workspace),
        )
    assert caught.value.category == "stream_incomplete"
    assert budget.used == 1
    assert workspace.state["tool_operations"] == []
    stream_status = json.loads(
        (workspace.directory / "model_call_001" / "stream_status.json").read_text()
    )
    assert stream_status["status"] == "stream_incomplete"


@pytest.mark.parametrize(
    ("provider", "finish_reason", "actual_model"),
    [
        ("openai", "tool_calls", "gpt-4o"),
        ("lmstudio", "stop", "openai/gpt-oss-20b"),
    ],
)
def test_recognized_complete_stream_returns_tool_call(
    tmp_path, provider, finish_reason, actual_model
):
    fixed = _fixed_artifact(tmp_path)
    chunks = _tool_sse(
        model=actual_model,
        finish_reason=finish_reason,
        arguments='{"section":"overview"}',
    )
    workspace, model, budget = _direct_streaming_model(
        tmp_path, fixed, chunks=chunks, provider=provider
    )
    response = model.generate(
        [ChatMessage(role=MessageRole.USER, content="fixture")],
        tools_to_call_from=llm_agent.build_agent_tools(workspace),
    )
    assert budget.used == 1
    assert response.tool_calls[0].function.name == "inspect_interface"
    assert response.tool_calls[0].function.arguments == {"section": "overview"}


def test_formal_issue_survives_child_runtime_pass_and_resume_messages(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "resume-memory"
    workspace = AgentWorkspace(
        directory=directory,
        fixed_sample=fixed,
        provider="lmstudio",
        model="mock/tool-model",
        beta=1.0,
        batch_size=2,
        worker_timeout=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        max_model_calls=4,
    )
    parent = workspace.submit_candidate(
        _candidate(passing=False, name="failed-parent"), None
    )
    failed = workspace.formal_evaluate(parent["candidate_id"])
    assert failed["passed"] is False
    child_candidate = _candidate(name="revised-child")
    child = workspace.submit_candidate(
        child_candidate, parent["candidate_id"]
    )
    assert workspace.test_candidate(child["candidate_id"])["status"] == "passed"
    inherited_formal = [
        issue
        for issue in workspace.state["candidates"][child["candidate_id"]][
            "inherited_issue_context"
        ]
        if issue["check_stage"] == "formal_evaluation"
    ]
    assert inherited_formal
    assert {issue["status"] for issue in inherited_formal} == {"not_revalidated"}
    workspace.state["model_calls_used"] = 1
    workspace.state["request_settings"] = _request_settings()
    workspace.state["status"] = "paused_budget_exhausted"
    workspace._save()

    model = ScriptedToolModel(
        [
            ("final_answer", {"answer": "fixture stop after inspecting resumed state"}),
            ("final_answer", {"answer": "still not approved"}),
            ("final_answer", {"answer": "budget-ending request"}),
        ]
    )
    result = run_agent(resume=directory, model_backend=model)
    assert result["status"] == "paused_budget_exhausted"
    actual_request = json.dumps(
        [message.dict() for message in model.calls[0]["messages"]],
        ensure_ascii=False,
        default=str,
    )
    assert child["candidate_id"] in actual_request
    assert parent["candidate_id"] in actual_request
    assert "LIPSCHITZ_NOT_IMPROVED" in actual_request
    assert "selected_formal_evaluation" in actual_request
    assert "nearest evaluated ancestor" in actual_request
    assert "by_lambda" in actual_request
    assert "record_type" in actual_request
    assert "evaluation" in actual_request


def test_work_summary_bounds_twenty_candidate_evaluation_history(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="bounded-history")
    parent_id = None
    candidate_ids = []
    report_paths = []
    for index in range(20):
        submitted = workspace.submit_candidate(
            _candidate(passing=False, name=f"revision-{index:02d}"),
            parent_id,
        )
        candidate_id = submitted["candidate_id"]
        candidate_ids.append(candidate_id)
        record = workspace.state["candidates"][candidate_id]
        by_lambda = {}
        diagnostic_by_lambda = {}
        formal_issues = []
        for lambda_value in ("0", "0.1"):
            by_lambda[lambda_value] = {
                "lambda_mbit_per_joule": float(lambda_value),
                "baseline_l_hat": 10.0,
                "candidate_l_hat": 10.5,
                "improvement": -0.5,
                "required_margin": 1e-5,
                "passed": False,
                "maximum_pair": {"i": 0, "j": 1},
            }
            diagnostic_by_lambda[lambda_value] = {
                "maximum_pair_ref": f"pair-{lambda_value}"
            }
            formal_issues.append(
                {
                    "code": "LIPSCHITZ_NOT_IMPROVED",
                    "problem": "fixture repeated formal failure",
                    "check_stage": "formal_evaluation",
                    "status": "open",
                    "source_candidate_id": candidate_id,
                    "lambda": lambda_value,
                }
            )
        relative_report = (
            f"candidates/{candidate_id}/evaluation_fixture_{index:02d}.json"
        )
        report_paths.append(relative_report)
        llm_agent._write_json(
            workspace.directory / relative_report,
            {
                "status": "failed",
                "passed": False,
                "by_lambda": by_lambda,
                "evaluation_diagnostics": {"by_lambda": diagnostic_by_lambda},
            },
        )
        record["evaluations"][f"fixture-{index:02d}"] = relative_report
        record["evaluation_status"] = "failed"
        record["formal_evaluation_issues"] = formal_issues
        workspace._save()
        parent_id = candidate_id

    summary = workspace.work_state(include_candidate=True)
    current = summary["current_candidate"]
    unresolved = current["unresolved_issue_summary"]
    assert current["candidate"] == workspace.state["candidates"][parent_id]["candidate"]
    assert unresolved["total_distinct_unresolved"] == 2
    assert unresolved["omitted_distinct_unresolved"] == 0
    assert {issue["lambda"] for issue in unresolved["issues"]} == {"0", "0.1"}
    assert all(issue["source_candidate_count"] == 20 for issue in unresolved["issues"])
    assert all(
        issue["omitted_source_candidate_count"] == 12
        for issue in unresolved["issues"]
    )
    assert current["selected_formal_evaluation"]["candidate_id"] == parent_id
    assert current["selected_formal_evaluation"]["report_id"] == report_paths[-1]
    assert "related_formal_evaluations" not in current
    assert json.dumps(summary).count(report_paths[-1]) == 1
    history_index = current["evaluation_history_index"]
    assert history_index["total_evaluated_candidates"] == 20
    assert len(history_index["recent"]) == llm_agent.WORK_SUMMARY_MAX_EVALUATION_INDEXES
    assert history_index["omitted_count"] == 8
    assert len(json.dumps(summary, ensure_ascii=False)) < 60_000

    first_history = workspace.get_history(candidate_ids[0], "evaluation")
    assert first_history["evaluations"][0]["path_id"] == report_paths[0]
    page = workspace.get_history(None, "run", start=0, limit=7)
    assert len(page["candidate_history_page"]) == 7
    assert page["next_start"] == 7
    assert page["total_candidate_count"] == 20

    workspace.state["request_settings"] = _request_settings(context_length=100_000)
    workspace.state["status"] = "paused_budget_exhausted"
    workspace._save()
    before_preview = _tree_hashes(workspace.directory)
    preview = run_agent(
        resume=workspace.directory,
        dry_run=True,
        model_backend=ScriptedToolModel([]),
        output_dir=tmp_path / "bounded-history-preview",
    )
    assert preview["metadata"]["initial_context_budget"]["fits_client_budget"]
    assert _tree_hashes(workspace.directory) == before_preview
    prompt = Path(preview["output_directory"], "agent_task_prompt.txt").read_text()
    assert parent_id in prompt
    assert workspace.state["candidates"][parent_id]["candidate"]["candidate_name"] in prompt
    assert "compute_extra_state" in prompt
    assert prompt.count("critical_pair_diagnostics") == 1
    assert report_paths[0] not in prompt


def test_work_summary_omitted_issue_details_remain_indexed(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="bounded-issues")
    submitted = workspace.submit_candidate(_candidate(name="many-issues"), None)
    candidate_id = submitted["candidate_id"]
    record = workspace.state["candidates"][candidate_id]
    record["issues"] = [
        {
            "code": f"FIXTURE_{index:02d}",
            "problem": f"distinct fixture issue {index}",
            "check_stage": "runtime_validation",
            "status": "open",
            "source_candidate_id": candidate_id,
            "candidate_function": "compute_extra_state",
            "candidate_line": index + 1,
        }
        for index in range(20)
    ]
    workspace._save()
    unresolved = workspace.work_state()["current_candidate"][
        "unresolved_issue_summary"
    ]
    assert unresolved["total_distinct_unresolved"] == 20
    assert unresolved["included_distinct_unresolved"] == 16
    assert unresolved["omitted_distinct_unresolved"] == 4
    assert unresolved["omitted_details_history_index"] == {
        "candidate_id": candidate_id,
        "record_type": "issues",
    }
    history = workspace.get_history(candidate_id, "issues")
    assert len(history["issues"]) == 20


def test_resume_budget_extension_is_cumulative_and_dry_run_is_nonmutating(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "budget-extension"
    initial_model = ScriptedToolModel(
        [("inspect_interface", {"section": "overview"})]
    )
    initial = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=initial_model.model_id,
        model_backend=initial_model,
        max_model_calls=1,
        context_length=100_000,
        output_dir=directory,
    )
    assert initial["status"] == "paused_budget_exhausted"

    no_extension_model = ScriptedToolModel([])
    no_extension = run_agent(resume=directory, model_backend=no_extension_model)
    assert no_extension["status"] == "paused_budget_exhausted"
    assert no_extension_model.calls == []

    before_dry = _tree_hashes(directory)
    dry_preview = tmp_path / "budget-extension-preview"
    dry = run_agent(
        resume=directory,
        model_backend=ScriptedToolModel([]),
        additional_model_calls=2,
        dry_run=True,
        output_dir=dry_preview,
    )
    assert dry["status"] == "dry_run_complete"
    assert Path(dry["output_directory"]) == dry_preview
    assert _tree_hashes(directory) == before_dry
    assert dry["metadata"]["generation"]["effective_max_model_calls"] == 3
    state_after_dry = json.loads((directory / "agent_state.json").read_text())
    assert state_after_dry["settings"]["max_model_calls"] == 1
    assert state_after_dry["model_calls_used"] == 1
    assert state_after_dry["budget_extension_events"] == []

    before_invalid = (directory / "agent_state.json").read_bytes()
    with pytest.raises(ValueError, match="must be positive"):
        run_agent(
            resume=directory,
            model_backend=ScriptedToolModel([]),
            additional_model_calls=0,
        )
    assert (directory / "agent_state.json").read_bytes() == before_invalid

    resumed_model = ScriptedToolModel(
        [
            ("final_answer", {"answer": "fixture finished without approval"}),
            ("final_answer", {"answer": "fixture still wants to stop"}),
        ]
    )
    resumed = run_agent(
        resume=directory,
        model_backend=resumed_model,
        additional_model_calls=2,
    )
    assert resumed["status"] == "paused_budget_exhausted"
    assert resumed["model_calls_used"] == 3
    state = json.loads((directory / "agent_state.json").read_text())
    assert state["settings"]["max_model_calls"] == 3
    assert state["budget_extension_events"][-1]["previous_max_model_calls"] == 1
    assert state["budget_extension_events"][-1]["new_max_model_calls"] == 3
    assert state["budget_extension_events"][-1]["model_calls_used"] == 1
    assert state["resume_events"]


def test_resume_dry_run_context_failure_does_not_modify_source(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "context-preview-source"
    workspace = _workspace(
        tmp_path,
        fixed,
        name="context-preview-source",
        max_model_calls=3,
    )
    workspace.state["request_settings"] = _request_settings(
        context_length=2_000, max_output_tokens=1_024
    )
    workspace._save()
    before = _tree_hashes(directory)
    preview = tmp_path / "context-preview-output"
    with pytest.raises(llm_agent.AgentContextBudgetError):
        run_agent(
            resume=directory,
            dry_run=True,
            model_backend=ScriptedToolModel([]),
            output_dir=preview,
        )
    assert _tree_hashes(directory) == before
    failure = json.loads((preview / "failure.json").read_text())
    assert failure["category"] == "context_budget_exceeded"
    assert failure["source_run_mutated"] is False


def test_resume_dry_run_compatibility_failure_does_not_modify_source(
    tmp_path, monkeypatch
):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "incompatible-preview-source"
    workspace = _workspace(
        tmp_path,
        fixed,
        name="incompatible-preview-source",
        max_model_calls=3,
    )
    workspace.state["request_settings"] = _request_settings()
    workspace._save()
    before = _tree_hashes(directory)
    saved_sha = workspace.state["git_sha"]
    monkeypatch.setattr(llm_agent, "_git_sha", lambda: f"different-{saved_sha}")
    with pytest.raises(ValueError, match="git revision"):
        run_agent(
            resume=directory,
            dry_run=True,
            model_backend=ScriptedToolModel([]),
            output_dir=tmp_path / "incompatible-preview-output",
        )
    assert _tree_hashes(directory) == before
    assert not (tmp_path / "incompatible-preview-output").exists()


def test_old_string_submit_tool_contract_is_rejected_on_resume(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="old-submit-contract")
    old_state = json.loads(json.dumps(workspace.state))
    old_state["contracts"]["tools"] = "uav-hrl-llm-feature-agent-tools-v2"
    with pytest.raises(ValueError, match="resume contracts are incompatible"):
        AgentWorkspace(
            directory=workspace.directory,
            fixed_sample=fixed,
            provider="lmstudio",
            model="mock/tool-model",
            beta=1.0,
            batch_size=2,
            worker_timeout=20,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-6,
            max_model_calls=20,
            resume_state=old_state,
            read_only=True,
        )


def test_approved_run_cannot_be_resumed(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "approved-resume"
    candidate = _candidate(name="approved-no-resume")
    candidate_id = _candidate_id(candidate)
    model = ScriptedToolModel(
        [
            (
                "submit_candidate",
                {"candidate": candidate, "parent_candidate_id": None},
            ),
            ("formal_evaluate", {"candidate_id": candidate_id}),
        ]
    )
    assert run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=directory,
    )["status"] == "approved"
    with pytest.raises(ValueError, match="cannot be resumed"):
        run_agent(
            resume=directory,
            model_backend=ScriptedToolModel([]),
            additional_model_calls=2,
        )


def test_request_budget_includes_tool_schemas(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="tool-budget")
    model = ScriptedToolModel([])
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="A short request whose conversation alone fits.",
        )
    ]
    without_tools, _ = llm_agent._request_token_budget(
        model,
        messages,
        [],
        context_length=100_000,
        max_output_tokens=256,
    )
    with_tools, completion = llm_agent._request_token_budget(
        model,
        messages,
        llm_agent.build_agent_tools(workspace),
        context_length=100_000,
        max_output_tokens=256,
    )
    assert completion["tools"]
    assert (
        with_tools["estimated_total_upper"]
        > without_tools["estimated_total_upper"]
    )
    boundary = without_tools["estimated_total_upper"]
    without_tools_at_boundary, _ = llm_agent._request_token_budget(
        model,
        messages,
        [],
        context_length=boundary,
        max_output_tokens=256,
    )
    with_tools_at_boundary, _ = llm_agent._request_token_budget(
        model,
        messages,
        llm_agent.build_agent_tools(workspace),
        context_length=boundary,
        max_output_tokens=256,
    )
    assert without_tools_at_boundary["fits_client_budget"] is True
    assert with_tools_at_boundary["fits_client_budget"] is False


def test_minimum_request_over_budget_sends_nothing_and_counts_nothing(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    model = ScriptedToolModel([])
    directory = tmp_path / "context-too-small"
    with pytest.raises(llm_agent.AgentContextBudgetError):
        run_agent(
            fixed_sample=fixed,
            provider="lmstudio",
            model=model.model_id,
            model_backend=model,
            max_model_calls=2,
            context_length=2_000,
            max_output_tokens=1_024,
            output_dir=directory,
        )
    assert model.calls == []
    state = json.loads((directory / "agent_state.json").read_text())
    assert state["model_calls_used"] == 0
    assert state["status"] == "failed_context_budget"
    request_shape = json.loads((directory / "initial_request_shape.json").read_text())
    assert request_shape["tools"]


def test_legacy_entry_imports_without_optional_smolagents_dependency():
    script = r'''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "smolagents" or name.startswith("smolagents."):
        raise ImportError("blocked optional dependency")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import run_llm_design
print("legacy-import-ok")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "legacy-import-ok" in completed.stdout


def test_context_compaction_keeps_complete_candidate_and_tool_pairs(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="compaction-candidate")
    candidate_id = _candidate_id(candidate)
    model = ScriptedToolModel(
        [
            (
                "submit_candidate",
                {"candidate": candidate, "parent_candidate_id": None},
            ),
            ("inspect_interface", {"section": "overview"}),
            (
                "get_history",
                {"candidate_id": candidate_id, "record_type": "candidate"},
            ),
            ("final_answer", {"answer": "Stopping without formal approval."}),
        ]
    )
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=4,
        context_length=20_500,
        max_output_tokens=1_024,
        output_dir=tmp_path / "compaction",
    )
    state = json.loads((tmp_path / "compaction" / "agent_state.json").read_text())
    assert result["status"] == "paused_budget_exhausted"
    assert state["context_compactions"]
    assert all(
        item["candidate_code_truncated"] is False
        and item["tool_call_result_pairs_split"] is False
        for item in state["context_compactions"]
    )
    compact_tasks = []
    for message in model.calls[-1]["messages"]:
        serialized_message = json.dumps(
            message.dict(), ensure_ascii=False, default=str
        )
        if "The current work record is:" in serialized_message:
            compact_tasks.append(serialized_message)
    assert len(compact_tasks) == 1
    assert compact_tasks[0].count("The current work record is:") == 1
    assert candidate_id in compact_tasks[0]
    final_request = json.dumps(
        [message.dict() for message in model.calls[-1]["messages"]],
        ensure_ascii=False,
        default=str,
    )
    assert candidate["candidate_name"] in final_request
    assert "Host-generated compact work-state summary." not in compact_tasks[0]
    assert len(state["framework_tool_calls"]) == 4
    assert [item["status"] for item in state["framework_tool_calls"]] == [
        "completed",
        "completed",
        "completed",
        "completion_rejected",
    ]


def test_pending_query_page_survives_compaction_into_actual_request(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="uses-observed-sample")

    def inspect_first(_request):
        return [("inspect_interface", {"section": "overview"})]

    def query_after_interface(request):
        assert "inspect_interface" in request
        return [
            (
                "query_samples",
                {
                    "fields": ["obs.movement_mask", "obs.uav_backlog_bits"],
                    "start": 0,
                    "limit": 20,
                    "condition_field": None,
                    "condition": None,
                },
            )
        ]

    def submit_only_after_page(request):
        assert "query_samples" in request
        assert "aware-call-2-0" in request
        assert "returned_sample_count" in request
        assert "samples" in request
        assert "sample_0" in request
        assert "next_query_arguments" in request
        return [
            (
                "submit_candidate",
                {"candidate": candidate, "parent_candidate_id": None},
            )
        ]

    model = RequestAwareToolModel(
        [inspect_first, query_after_interface, submit_only_after_page]
    )
    result = run_agent(
        fixed_sample=fixed,
        provider="openai",
        model="gpt-4o",
        model_backend=model,
        max_model_calls=3,
        context_length=21_000,
        max_output_tokens=1_024,
        output_dir=tmp_path / "pending-query-compaction",
    )
    state = json.loads(
        (tmp_path / "pending-query-compaction" / "agent_state.json").read_text()
    )
    assert result["status"] == "paused_budget_exhausted"
    assert state["context_compactions"]
    query_record = next(
        item
        for item in state["framework_tool_calls"]
        if item["name"] == "query_samples"
    )
    assert query_record["delivery_status"] == "delivered"
    assert query_record["delivered_model_call_number"] == 3
    raw = json.loads(
        (
            tmp_path
            / "pending-query-compaction"
            / query_record["raw_result_path"]
        ).read_text()
    )
    assert raw["result"]["requested_limit"] == 20
    assert raw["result"]["returned_sample_count"] > 0


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("lmstudio", "openai/gpt-oss-20b"),
        ("lmstudio", "qwen/qwen3.5-9b"),
        ("lmstudio", "google/gemma-4-e4b"),
        ("openai", "gpt-4o"),
    ],
)
def test_multi_tool_results_are_all_paired_in_next_actual_request(
    tmp_path, provider, model_id
):
    fixed = _fixed_artifact(tmp_path)

    def issue_two_calls(_request):
        return [
            (
                "query_samples",
                {
                    "fields": ["obs.movement_mask"],
                    "start": 0,
                    "limit": 20,
                    "condition_field": None,
                    "condition": None,
                },
            ),
            ("inspect_interface", {"section": "evaluation"}),
        ]

    def verify_both(request):
        for call_id, tool in (
            ("aware-call-1-0", "query_samples"),
            ("aware-call-1-1", "inspect_interface"),
        ):
            assert call_id in request
            assert tool in request
        assert "returned_sample_count" in request
        assert "data_lines" in request
        return [("final_answer", {"answer": "fixture stop"})]

    model = RequestAwareToolModel([issue_two_calls, verify_both], model_id=model_id)
    output = tmp_path / f"multi-result-{provider}-{model_id.replace('/', '-')}"
    result = run_agent(
        fixed_sample=fixed,
        provider=provider,
        model=model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=output,
    )
    assert result["status"] == "paused_budget_exhausted"
    state = json.loads((output / "agent_state.json").read_text())
    delivered = [
        item
        for item in state["framework_tool_calls"][:2]
        if item["delivery_status"] == "delivered"
    ]
    assert len(delivered) == 2
    assert {item["delivered_model_call_number"] for item in delivered} == {2}


def test_transport_failure_keeps_completed_query_pending(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    model = QueryThenTransportFailureModel()
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=100_000,
        output_dir=tmp_path / "pending-after-transport",
    )
    assert result["status"] == "failed"
    state = json.loads(
        (tmp_path / "pending-after-transport" / "agent_state.json").read_text()
    )
    query_record = state["framework_tool_calls"][0]
    assert query_record["status"] == "completed"
    assert query_record["delivery_status"] == "pending"
    assert len(state["tool_operations"]) == 1


def test_resume_delivers_pending_result_without_reexecuting_tool(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    directory = tmp_path / "pending-resume"
    first = ScriptedToolModel(
        [
            (
                "query_samples",
                {
                    "fields": ["obs.movement_mask"],
                    "start": 0,
                    "limit": 20,
                    "condition_field": None,
                    "condition": None,
                },
            )
        ]
    )
    initial = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=first.model_id,
        model_backend=first,
        max_model_calls=1,
        context_length=100_000,
        output_dir=directory,
    )
    assert initial["status"] == "paused_budget_exhausted"

    def verify_resume(request):
        assert "pending_tool_result_delivery" in request
        assert "query_samples" in request
        assert "sample_0" in request
        return [("final_answer", {"answer": "fixture stop after reading"})]

    resumed_model = RequestAwareToolModel([verify_resume], model_id=first.model_id)
    resumed = run_agent(
        resume=directory,
        model_backend=resumed_model,
        additional_model_calls=1,
    )
    assert resumed["status"] == "paused_budget_exhausted"
    state = json.loads((directory / "agent_state.json").read_text())
    assert [item["tool"] for item in state["tool_operations"]].count(
        "query_samples"
    ) == 1
    query_record = state["framework_tool_calls"][0]
    assert query_record["delivery_status"] == "delivered"
    assert query_record["delivered_model_call_number"] == 2


def test_query_sample_pages_are_stable_complete_and_make_progress(
    tmp_path, monkeypatch
):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="sample-pages")
    monkeypatch.setattr(llm_agent, "PAGED_TOOL_PAYLOAD_MAX_CHARS", 2_800)
    arguments = {
        "fields": [
            "obs.movement_mask",
            "obs.uav_backlog_bits",
            "obs.uav_queue_valid",
        ],
        "start": 0,
        "limit": 20,
        "condition_field": None,
        "condition": None,
    }
    references = []
    while True:
        page = workspace.query_samples(**arguments)
        assert page["status"] == "ok"
        assert page["returned_sample_count"] > 0
        assert page["fields"] == arguments["fields"]
        references.extend(page["returned_sample_refs"])
        if not page["has_more"]:
            break
        next_arguments = page["next_query_arguments"]
        assert next_arguments["start"] > arguments["start"]
        assert next_arguments["fields"] == arguments["fields"]
        assert next_arguments["condition_field"] == arguments["condition_field"]
        assert next_arguments["condition"] == arguments["condition"]
        arguments = next_arguments
    assert len(references) == workspace.fixed_metadata["sample_count"]
    assert len(references) == len(set(references))
    assert references == [f"sample_{index}" for index in range(len(references))]

    final = workspace.query_samples(
        fields=["obs.movement_mask"],
        start=len(references),
        limit=20,
        condition_field=None,
        condition=None,
    )
    assert final["status"] == "ok"
    assert final["returned_sample_count"] == 0
    assert final["has_more"] is False


def test_single_sample_too_large_never_offers_zero_progress_page(
    tmp_path, monkeypatch
):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="oversized-sample")
    monkeypatch.setattr(llm_agent, "PAGED_TOOL_PAYLOAD_MAX_CHARS", 250)
    result = workspace.query_samples(
        fields=["obs.state"],
        start=0,
        limit=20,
        condition_field=None,
        condition=None,
    )
    assert result["status"] == "page_too_large"
    assert result["returned_sample_count"] == 0
    assert result["next_query_arguments"] is None
    assert result["has_more"] is False


def test_pending_result_that_cannot_fit_stops_before_incomplete_request(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    model = ScriptedToolModel(
        [
            (
                "query_samples",
                {
                    "fields": [
                        "obs.state",
                        "obs.movement_mask",
                        "obs.uav_backlog_bits",
                        "obs.uav_queue_valid",
                    ],
                    "start": 0,
                    "limit": 20,
                    "condition_field": None,
                    "condition": None,
                },
            )
        ]
    )
    result = run_agent(
        fixed_sample=fixed,
        provider="lmstudio",
        model=model.model_id,
        model_backend=model,
        max_model_calls=2,
        context_length=18_500,
        max_output_tokens=1_024,
        output_dir=tmp_path / "pending-does-not-fit",
    )
    assert result["status"] == "failed_context_budget"
    assert len(model.calls) == 1
    state = json.loads(
        (tmp_path / "pending-does-not-fit" / "agent_state.json").read_text()
    )
    record = state["framework_tool_calls"][0]
    assert record["status"] == "completed"
    assert record["delivery_status"] == "pending"
    failure = state["context_compactions"][-1]
    assert failure["status"] == "context_budget_exceeded"
    assert record["record_id"] in failure["pending_result_record_ids"]


def test_interface_pages_preserve_order_and_offer_exact_next_arguments(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="interface-pages")
    first = workspace.inspect_interface("overview", start=0, limit=3)
    assert first["status"] == "ok"
    assert first["returned_line_count"] == 3
    assert first["has_more"] is True
    second = workspace.inspect_interface(**first["next_query_arguments"])
    assert second["page_start"] == 3
    assert second["data_lines"]
    assert first["data_lines"] != second["data_lines"]
    assert first["total_line_count"] == second["total_line_count"]
    assert second["next_query_arguments"]["section"] == "overview"


def test_history_report_pages_are_model_retrievable_without_arbitrary_paths(
    tmp_path
):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="history-report-pages")
    submitted = workspace.submit_candidate(_candidate(name="history-report"), None)
    candidate_id = submitted["candidate_id"]
    tested = workspace.test_candidate(candidate_id)
    report_id = tested["report_id"]
    lines = []
    arguments = {
        "candidate_id": candidate_id,
        "record_type": "report",
        "start": 0,
        "limit": 7,
        "report_id": report_id,
    }
    while True:
        page = workspace.get_history(**arguments)
        assert page["status"] == "ok"
        assert page["report_id"] == report_id
        assert workspace._serialized_size(page) <= llm_agent.MODEL_TOOL_RESULT_MAX_CHARS
        lines.extend(page["data_lines"])
        if not page["has_more"]:
            break
        arguments = page["next_query_arguments"]
    reconstructed = json.loads("\n".join(lines))
    assert reconstructed["candidate_id"] == candidate_id
    assert reconstructed["test_scope"] == "full_fixed_samples"

    rejected = workspace.get_history(
        candidate_id,
        "report",
        report_id="C:/arbitrary/path.json",
    )
    assert rejected["status"] == "invalid_arguments"


def test_bounded_query_index_links_to_paged_tool_call_history(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    workspace = _workspace(tmp_path, fixed, name="tool-call-history")
    for index in range(10):
        call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(
                name="query_samples",
                arguments={
                    "fields": ["obs.movement_mask"],
                    "start": index,
                    "limit": 1,
                    "condition_field": None,
                    "condition": None,
                },
            ),
            id=f"history-query-{index}",
            type="function",
        )
        workspace.register_framework_calls([call])
        workspace.finish_framework_call(
            call.id,
            status="completed",
            output={
                "status": "ok",
                "requested_start": index,
                "returned_sample_count": 1,
                "returned_sample_refs": [f"sample_{index}"],
            },
        )
    summary = workspace.work_state()
    recent = summary["recent_completed_queries"]
    assert recent["total_count"] == 10
    assert recent["included_count"] == llm_agent.WORK_SUMMARY_MAX_QUERY_INDEXES
    history = workspace.get_history(
        None, "tool_calls", start=0, limit=4
    )
    assert history["returned_count"] == 4
    assert history["has_more"] is True
    second = workspace.get_history(**history["next_query_arguments"])
    assert second["tool_call_records"][0]["tool_call_id"] == "history-query-4"
    detail = workspace.get_history(
        **history["tool_call_records"][0]["result_history_query"]
    )
    assert detail["record_type"] == "tool_result"
    assert detail["record_id"] == "framework-call-000001"
    assert detail["result"]["returned_sample_refs"] == ["sample_0"]


@pytest.mark.parametrize(
    ("provider", "model", "reasoning_effort"),
    [
        ("lmstudio", "openai/gpt-oss-20b", "low"),
        ("lmstudio", "qwen/qwen3.5-9b", None),
        ("lmstudio", "google/gemma-4-e4b", None),
        ("openai", "gpt-4o", None),
    ],
)
def test_provider_models_share_dry_run_contract_without_requests(
    tmp_path, provider, model, reasoning_effort
):
    fixed = _fixed_artifact(tmp_path)
    result = run_agent(
        fixed_sample=fixed,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        context_length=100_000,
        output_dir=tmp_path / f"dry-{provider}-{model.replace('/', '-')}",
        dry_run=True,
    )
    assert result["status"] == "dry_run_complete"
    assert result["model_calls_used"] == 0
    assert result["metadata"]["provider"] == provider
    assert result["metadata"]["model"] == model
    assert result["metadata"]["generation"]["reasoning_effort"] == reasoning_effort
    system_prompt = Path(
        result["output_directory"], "framework_system_prompt.txt"
    ).read_text(encoding="utf-8")
    assert "Only the host can complete this task" in system_prompt
    assert "It is the only way to complete the task" not in system_prompt
