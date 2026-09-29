import json
from pathlib import Path
import subprocess
import sys

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


class TransportFailureModel(Model):
    def __init__(self):
        super().__init__(model_id="mock/transport-failure")
        self.calls = 0

    def generate(self, messages, tools_to_call_from=None, **kwargs):
        self.calls += 1
        raise APIError("fixture transport failed", category="stream_connection_error")


def _candidate_id(candidate):
    return f"candidate-{llm_agent._content_hash(candidate)[:12]}"


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
            ("submit_candidate", {"candidate_json": json.dumps(bad), "parent_candidate_id": None}),
            ("test_candidate", {"candidate_id": bad_id, "mode": "small", "sample_indices": [0]}),
            ("submit_candidate", {"candidate_json": json.dumps(good), "parent_candidate_id": bad_id}),
            ("test_candidate", {"candidate_id": good_id, "mode": "small", "sample_indices": [0, 2, 4]}),
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


def test_final_answer_small_test_and_budget_never_approve(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    final_model = ScriptedToolModel(
        [("final_answer", {"answer": "I am done and approve my own design."})]
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
    assert final["status"] == "stopped_without_approval"
    assert final["approved_artifact"] is None

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
    assert exhausted["status"] == "stopped_without_approval"
    assert exhausted["model_calls_used"] == 1
    assert len(inspect_model.calls) == 1
    assert exhausted["approved_artifact"] is None


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
    first = workspace.submit_candidate(json.dumps(failing), None)
    second = workspace.submit_candidate(json.dumps(passing), first["candidate_id"])
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


def test_resume_preserves_candidate_and_marks_inflight_operation_interrupted(tmp_path):
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
    submitted = workspace.submit_candidate(json.dumps(_candidate()), None)
    workspace.begin_operation("test_candidate", {"candidate_id": submitted["candidate_id"]})
    workspace.state["model_calls_used"] = 2
    workspace.state["request_settings"] = {
        "base_url": "http://127.0.0.1:1234/v1",
        "context_length": 100_000,
        "max_output_tokens": 1_024,
        "temperature": llm_agent.DEFAULT_TEMPERATURE,
        "seed": llm_agent.DEFAULT_SEED,
        "reasoning_effort": None,
        "timeout": llm_agent.DEFAULT_API_TIMEOUT_SECONDS,
        "connect_timeout": llm_agent.DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
        "total_timeout": llm_agent.DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
        "progress_interval": llm_agent.DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    }
    workspace._save()
    resumed = run_agent(
        resume=directory,
        dry_run=True,
        context_length=100_000,
        max_output_tokens=1024,
    )
    assert resumed["status"] == "dry_run_complete"
    state = json.loads((directory / "agent_state.json").read_text())
    assert state["current_candidate_id"] == submitted["candidate_id"]
    assert state["model_calls_used"] == 2
    assert state["tool_operations"][-1]["status"] == "interrupted"
    assert state["tool_operations"][-1]["result_summary"]["reusable_as_pass"] is False


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
    parent = workspace.submit_candidate(json.dumps(bad), None)
    child = workspace.submit_candidate(
        json.dumps(_candidate(name="corrected-child")), parent["candidate_id"]
    )
    record = workspace.state["candidates"][child["candidate_id"]]
    assert record["inherited_issue_context"]
    assert {item["status"] for item in record["inherited_issue_context"]} == {
        "not_revalidated"
    }
    assert workspace.test_candidate(
        child["candidate_id"], mode="small", sample_indices=[0]
    )["status"] == "passed"
    assert {item["status"] for item in record["inherited_issue_context"]} == {
        "not_revalidated"
    }
    assert workspace.test_candidate(child["candidate_id"], mode="full")[
        "status"
    ] == "passed"
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
        json.dumps(_candidate()), "candidate-not-present"
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
    assert resumed["status"] == "stopped_without_approval"
    assert resumed["model_calls_used"] == 2
    assert resumed_model.calls == []
    assert "already exhausted" in resumed["stop_reason"]


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
                {"candidate_json": json.dumps(candidate), "parent_candidate_id": None},
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
        context_length=19_000,
        max_output_tokens=1_024,
        output_dir=tmp_path / "compaction",
    )
    state = json.loads((tmp_path / "compaction" / "agent_state.json").read_text())
    assert result["status"] == "stopped_without_approval"
    assert state["context_compactions"]
    assert all(
        item["candidate_code_truncated"] is False
        and item["tool_call_result_pairs_split"] is False
        for item in state["context_compactions"]
    )
    compact_summaries = []
    for message in model.calls[-1]["messages"]:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.startswith(
            "Host-generated compact work-state summary."
        ):
            compact_summaries.append(json.loads(content.split("\n", 1)[1]))
    assert compact_summaries
    assert compact_summaries[-1]["current_candidate"]["candidate"] == candidate
    assert len(state["framework_tool_calls"]) == 4
    assert all(item["status"] == "completed" for item in state["framework_tool_calls"])


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
