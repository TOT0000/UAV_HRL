import json
from pathlib import Path

import numpy as np
import pytest

import llm_review_design
import llm_design
import run_llm_review_design
from llm_candidate import (
    SAFE_NUMPY_CALLS,
    load_approved_design,
    normalize_candidate_submission,
    validate_candidate,
)
from llm_design_contract import (
    SUPPORTED_OPERATIONS,
    build_obs_arrays,
    load_design_inputs,
)
from llm_runtime import ApprovedDesignRuntime
from test_llm_design import _fixed_artifact


class ReviewMockClient:
    def __init__(self, responses, *, model, loaded_context=50000):
        self.responses = list(responses)
        self.model = model
        self.loaded_context = int(loaded_context)
        self.calls = []

    def list_models(self):
        return {
            "openai": {"data": [{"id": self.model}]},
            "native": {
                "models": [
                    {
                        "key": self.model,
                        "max_context_length": self.loaded_context,
                        "loaded_instances": [
                            {
                                "id": self.model,
                                "config": {"context_length": self.loaded_context},
                            }
                        ],
                    }
                ]
            },
            "native_error": None,
        }

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _provider_response(content, *, model, finish_reason="stop", complete=True):
    return {
        "content": content,
        "reasoning": None,
        "finish_reason": finish_reason,
        "actual_model": model,
        "usage": {},
        "transport_completed": complete,
        "done_received": complete,
        "terminal_chunk_received": complete,
        "tool_calls_seen": False,
    }


def _submission(*, name="bounded-square", code=None, weight=0.25):
    if code is None:
        code = (
            "def compute_extra_state(obs, constants):\n"
            "    value = obs['state'][0] * obs['state'][0]\n"
            "    return np.asarray([np.clip(value, 0.0, 1.0)], dtype=np.float32)\n"
        )
    return {
        "features": [
            {
                "name": name,
                "description": "Bounded square of one current normalized state value.",
                "reward_weight": weight,
            }
        ],
        "code": code,
    }


def _review(*, needs_revision, summary="reviewed", suggestions=None):
    return json.dumps(
        {
            "needs_revision": needs_revision,
            "summary": summary,
            "suggestions": suggestions or [],
        }
    )


def _run_kwargs(fixed, output, proposer, reviewer, **overrides):
    values = {
        "fixed_sample": fixed,
        "proposer_provider": "lmstudio",
        "proposer_model": proposer.model,
        "proposer_context_length": 50000,
        "proposer_max_output_tokens": 2048,
        "reviewer_provider": "openai",
        "reviewer_model": reviewer.model,
        "reviewer_context_length": 50000,
        "reviewer_max_output_tokens": 2048,
        "output_dir": output,
        "proposer_client": proposer,
        "reviewer_client": reviewer,
    }
    values.update(overrides)
    return values


def test_repair_then_review_uses_complete_context_and_approves(tmp_path, monkeypatch):
    monkeypatch.setattr(
        llm_design,
        "evaluate_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("review-design flow must not call Lipschitz evaluation")
        ),
    )
    fixed = _fixed_artifact(tmp_path)
    invalid = _submission(
        name="missing-field",
        code=(
            "def compute_extra_state(obs, constants):\n"
            "    return np.asarray([obs['not_a_field'][0]], dtype=np.float32)\n"
        ),
    )
    valid = _submission()
    proposer = ReviewMockClient(
        [
            _provider_response(json.dumps(invalid), model="qwen/proposer"),
            _provider_response(json.dumps(valid), model="qwen/proposer"),
        ],
        model="qwen/proposer",
    )
    reviewer = ReviewMockClient(
        [_provider_response(_review(needs_revision=False), model="gpt-4o")],
        model="gpt-4o",
    )
    result = llm_review_design.run_review_design(
        **_run_kwargs(fixed, tmp_path / "run", proposer, reviewer)
    )
    assert result["status"] == "approved"
    assert result["proposer_calls"] == 2
    assert result["reviewer_calls"] == 1
    assert result["review_rounds_completed"] == 1
    assert "not_a_field" in proposer.calls[1]["prompt"]
    assert json.dumps(invalid, indent=2) in proposer.calls[1]["prompt"]
    review_prompt = reviewer.calls[0]["prompt"]
    assert "System and task" in review_prompt
    assert "obs['state'][0] * obs['state'][0]" in review_prompt
    assert '"status": "passed"' in review_prompt
    assert "lipschitz" not in review_prompt.lower()
    assert "baseline_l_hat" not in review_prompt
    assert "pairwise" not in review_prompt.lower()

    loaded = load_approved_design(result["approved_artifact"])
    assert loaded.artifact["approval_method"] == "model_review"
    assert loaded.artifact["provenance"]["approval_method"] == "model_review"
    evaluation = json.loads(
        (Path(result["approved_artifact"]) / "evaluation_report.json").read_text()
    )
    assert evaluation["lipschitz_evaluation_performed"] is False
    no_proposer = ReviewMockClient([], model="qwen/proposer")
    no_reviewer = ReviewMockClient([], model="gpt-4o")
    already_approved = llm_review_design.run_review_design(
        resume=result["output_directory"],
        proposer_client=no_proposer,
        reviewer_client=no_reviewer,
    )
    assert already_approved["status"] == "approved"
    assert not no_proposer.calls and not no_reviewer.calls


def test_review_revision_resets_repairs_without_consuming_extra_review(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    valid_one = _submission(name="first")
    broken_two = _submission(
        name="second",
        code=(
            "def compute_extra_state(obs, constants):\n"
            "    return np.asarray([missing], dtype=np.float32)\n"
        ),
    )
    valid_two = _submission(name="second")
    suggestion = {
        "target": "first",
        "reason": "The description and implementation should be revised.",
        "suggested_change": "Return a complete revised candidate.",
    }
    proposer = ReviewMockClient(
        [
            _provider_response(json.dumps(valid_one), model="qwen/proposer"),
            _provider_response(json.dumps(broken_two), model="qwen/proposer"),
            _provider_response(json.dumps(valid_two), model="qwen/proposer"),
        ],
        model="qwen/proposer",
    )
    reviewer = ReviewMockClient(
        [
            _provider_response(
                _review(needs_revision=True, suggestions=[suggestion]), model="gpt-4o"
            ),
            _provider_response(_review(needs_revision=False), model="gpt-4o"),
        ],
        model="gpt-4o",
    )
    result = llm_review_design.run_review_design(
        **_run_kwargs(fixed, tmp_path / "run", proposer, reviewer)
    )
    assert result["status"] == "approved"
    assert result["proposer_calls"] == 3
    assert result["reviewer_calls"] == 2
    assert result["review_rounds_completed"] == 2
    assert result["code_repairs_used_in_cycle"] == 1
    assert "Return a complete revised candidate" in proposer.calls[1]["prompt"]
    assert "Return a complete revised candidate" in proposer.calls[2]["prompt"]


def test_limits_and_resume_preserve_counts_and_completed_phases(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    invalid = "not JSON"
    first_client = ReviewMockClient(
        [_provider_response(invalid, model="qwen/proposer") for _ in range(6)],
        model="qwen/proposer",
    )
    unused_reviewer = ReviewMockClient([], model="gpt-4o")
    first = llm_review_design.run_review_design(
        **_run_kwargs(
            fixed,
            tmp_path / "repair-run",
            first_client,
            unused_reviewer,
            max_code_repairs=5,
        )
    )
    assert first["status"] == "paused_code_repairs_exhausted"
    assert first["proposer_calls"] == 6
    assert first["reviewer_calls"] == 0

    valid_client = ReviewMockClient(
        [_provider_response(json.dumps(_submission()), model="qwen/proposer")],
        model="qwen/proposer",
    )
    approving = ReviewMockClient(
        [_provider_response(_review(needs_revision=False), model="gpt-4o")],
        model="gpt-4o",
    )
    resumed = llm_review_design.run_review_design(
        resume=first["output_directory"],
        max_code_repairs=6,
        proposer_client=valid_client,
        reviewer_client=approving,
    )
    assert resumed["status"] == "approved"
    assert resumed["proposer_calls"] == 7
    assert resumed["code_repairs_used_in_cycle"] == 6

    requesting = _review(
        needs_revision=True,
        suggestions=[
            {
                "target": "design",
                "reason": "revise",
                "suggested_change": "produce a new complete candidate",
            }
        ],
    )
    p1 = ReviewMockClient(
        [
            _provider_response(
                json.dumps(_submission(name=f"round-{index}")), model="qwen/p"
            )
            for index in range(1, 6)
        ],
        model="qwen/p",
    )
    r1 = ReviewMockClient(
        [_provider_response(requesting, model="gpt/r") for _ in range(5)],
        model="gpt/r",
    )
    paused = llm_review_design.run_review_design(
        **_run_kwargs(
            fixed,
            tmp_path / "review-run",
            p1,
            r1,
            max_review_rounds=5,
        )
    )
    assert paused["status"] == "paused_review_rounds_exhausted"
    assert paused["proposer_calls"] == 5
    assert paused["reviewer_calls"] == 5
    assert paused["review_rounds_completed"] == 5
    same_limit = llm_review_design.run_review_design(
        resume=paused["output_directory"],
        proposer_client=ReviewMockClient([], model="qwen/p"),
        reviewer_client=ReviewMockClient([], model="gpt/r"),
    )
    assert same_limit["status"] == "paused_review_rounds_exhausted"

    p2 = ReviewMockClient(
        [_provider_response(json.dumps(_submission(name="round-2")), model="qwen/p")],
        model="qwen/p",
    )
    r2 = ReviewMockClient(
        [_provider_response(_review(needs_revision=False), model="gpt/r")],
        model="gpt/r",
    )
    resumed_review = llm_review_design.run_review_design(
        resume=paused["output_directory"],
        max_review_rounds=6,
        proposer_client=p2,
        reviewer_client=r2,
    )
    assert resumed_review["status"] == "approved"
    assert resumed_review["proposer_calls"] == 6
    assert resumed_review["reviewer_calls"] == 6
    assert len(p2.calls) == 1


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_provider_response("not json", model="gpt-4o"), "paused_invalid_review"),
        (
            _provider_response(
                _review(needs_revision=False),
                model="gpt-4o",
                finish_reason="length",
            ),
            "paused_truncated_response",
        ),
        (RuntimeError("transport failed"), "paused_transport_failure"),
    ],
)
def test_invalid_incomplete_or_failed_review_never_approves(tmp_path, response, expected):
    fixed = _fixed_artifact(tmp_path)
    proposer = ReviewMockClient(
        [_provider_response(json.dumps(_submission()), model="qwen/p")], model="qwen/p"
    )
    reviewer = ReviewMockClient([response], model="gpt-4o")
    result = llm_review_design.run_review_design(
        **_run_kwargs(fixed, tmp_path / "run", proposer, reviewer)
    )
    assert result["status"] == expected
    assert result["approved_artifact"] is None
    assert not (Path(result["output_directory"]) / "approved").exists()


def test_python_and_numpy_min_max_work_through_validator_worker_and_runtime(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    submission = {
        "features": [
            {"name": "span", "description": "Energy span.", "reward_weight": 0.1},
            {"name": "bounded_mean", "description": "Bounded mean.", "reward_weight": -0.1},
        ],
        "code": (
            "def compute_extra_state(obs, constants):\n"
            "    energy = obs['uav_remaining_energy_fraction']\n"
            "    low = np.min(energy)\n"
            "    high = np.max(energy)\n"
            "    span = max(0.0, min(1.0, high - low))\n"
            "    bounded = np.maximum(0.0, np.minimum(1.0, energy))\n"
            "    return np.asarray([span, np.mean(bounded)], dtype=np.float32)\n"
        ),
    }
    proposer = ReviewMockClient(
        [_provider_response(json.dumps(submission), model="qwen/p")], model="qwen/p"
    )
    reviewer = ReviewMockClient(
        [_provider_response(_review(needs_revision=False), model="gpt-4o")],
        model="gpt-4o",
    )
    result = llm_review_design.run_review_design(
        **_run_kwargs(fixed, tmp_path / "run", proposer, reviewer)
    )
    assert result["status"] == "approved"
    assert {"np.min", "np.max", "np.minimum", "np.maximum"}.issubset(
        SAFE_NUMPY_CALLS
    )
    assert "max, maximum" in SUPPORTED_OPERATIONS
    assert "min, minimum" in SUPPORTED_OPERATIONS
    loaded = load_approved_design(result["approved_artifact"])
    arrays, _, _, _ = load_design_inputs(fixed)
    extra, reward = loaded.evaluate_fixed_samples(arrays)
    assert extra.shape == (arrays["state"].shape[0], 2)
    assert np.all(np.isfinite(extra))
    assert np.allclose(reward, extra @ loaded.weights)
    batched_obs = build_obs_arrays(arrays)
    one_obs = {
        name: np.asarray(values[0]).copy() for name, values in batched_obs.items()
    }
    runtime = ApprovedDesignRuntime(loaded, loaded.constants_metadata, timeout=10.0)
    try:
        online_extra, online_reward = runtime.evaluate(one_obs)
    finally:
        runtime.close()
    assert np.allclose(online_extra, extra[0])
    assert online_reward == pytest.approx(reward[0])


def test_dry_run_keeps_role_settings_separate_and_context_failure_sends_nothing(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    proposer = ReviewMockClient([], model="qwen/p")
    reviewer = ReviewMockClient([], model="gpt-4o")
    dry = llm_review_design.run_review_design(
        **_run_kwargs(
            fixed,
            tmp_path / "dry",
            proposer,
            reviewer,
            dry_run=True,
            proposer_base_url="http://127.0.0.1:1234/v1",
            reviewer_base_url="https://api.openai.com/v1",
            proposer_temperature=0.2,
            reviewer_temperature=0.0,
            proposer_reasoning_effort="low",
            reviewer_reasoning_effort="high",
        )
    )
    assert dry["status"] == "dry_run_complete"
    assert not proposer.calls and not reviewer.calls
    preview = dry["dry_run"]
    assert preview["proposer"]["settings"]["base_url"].startswith("http://")
    assert preview["reviewer"]["settings"]["base_url"].startswith("https://")
    assert preview["proposer"]["planned_request"]["reasoning_effort"] == "low"
    assert preview["reviewer"]["planned_request"]["reasoning_effort"] == "high"
    assert preview["generation_requests_sent"] == 0

    too_small = ReviewMockClient([], model="qwen/small", loaded_context=2000)
    no_review = ReviewMockClient([], model="gpt-4o")
    stopped = llm_review_design.run_review_design(
        **_run_kwargs(
            fixed,
            tmp_path / "small",
            too_small,
            no_review,
            proposer_context_length=2000,
            proposer_max_output_tokens=1024,
        )
    )
    assert stopped["status"] == "paused_context_budget"
    assert not too_small.calls and not no_review.calls


def test_cli_exposes_independent_limits_and_role_settings():
    parser = run_llm_review_design.build_parser()
    args = parser.parse_args(
        [
            "--fixed-sample",
            "fixed",
            "--proposer-provider",
            "lmstudio",
            "--proposer-model",
            "qwen/p",
            "--reviewer-provider",
            "openai",
            "--reviewer-model",
            "gpt-4o",
            "--max-review-rounds",
            "7",
            "--max-code-repairs",
            "3",
        ]
    )
    assert args.max_review_rounds == 7
    assert args.max_code_repairs == 3
    assert args.proposer_provider == "lmstudio"
    assert args.reviewer_provider == "openai"
