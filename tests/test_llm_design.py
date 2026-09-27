import json
from pathlib import Path

import numpy as np
import pytest
import llm_design
import run_llm_design

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM, movement_state_feature_schema
from llm_baseline import run_baseline
from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    load_approved_design,
    parse_candidate_json,
    validate_candidate,
)
from llm_design import (
    APIError,
    EvaluationContext,
    LMStudioClient,
    evaluate_candidate,
    model_inventory_summary,
    run_design,
)
from llm_design_contract import (
    build_constants,
    build_obs_arrays,
    candidate_schema,
    format_schema_and_example,
    render_prompt,
)
from replay_auxiliary import empty_snapshot, replay_auxiliary_metadata
from scenario_manifest import generate_manifest
from utils_update_v2 import ReplayBufferJoint


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _fixed_artifact(tmp_path, *, duplicate_first_state=False):
    source = tmp_path / "sample-source"
    source.mkdir()
    episodes = 2
    horizon = 3
    manifest = generate_manifest("test", 991, episodes, balanced_num_gt=True)
    manifest.save(source / "scenario_manifest.json")
    replay = ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=episodes * horizon,
        record_auxiliary=True,
    )
    transition = 0
    for episode, scenario in enumerate(manifest.episodes):
        for step in range(horizon):
            state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
            state[0] = transition / 10.0
            if duplicate_first_state and transition == 1:
                state[0] = 0.0
            snapshot = empty_snapshot()
            snapshot["snapshot_valid"][0] = True
            snapshot["snapshot_time_s"][0] = step
            following = empty_snapshot()
            following["snapshot_valid"][0] = True
            following["snapshot_time_s"][0] = min(step + 1, horizon)
            replay.add(
                state,
                np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
                state,
                done=step == horizon - 1,
                delivered_mbits=float(transition),
                total_mobility_energy=100.0,
                c9_penalty=0.1,
                c10_penalty=0.2,
                com_range_penalty=0.3,
                current_movement_mask=np.zeros(16, dtype=bool),
                next_movement_mask=np.zeros(16, dtype=bool),
                current_auxiliary_snapshot=snapshot,
                next_auxiliary_snapshot=following,
                episode_id=episode,
                td3_step=step,
                global_transition_id=transition,
                scenario_index=episode,
                scenario_id=scenario["scenario_id"],
                dinkelbach_lambda=0.0,
            )
            transition += 1
    replay.save_npz(source / "joint_replay.npz")
    visual = {
        "vs_camera": {
            "f_m": 0.035,
            "image_width_m": 0.0156,
            "image_length_m": 0.0235,
        },
        "b1": 2 * 0.035 / 0.0156,
        "b2": 2 * 0.035 / 0.0235,
        "default_roi_radius_m": 80.0,
    }
    metadata = {
        "schema_version": "uav-hrl-llm-sampling-v1",
        "status": "complete",
        "complete": True,
        "method_id": "td3_dinkelbach",
        "checkpoint_completed_episodes": 250,
        "episode_seconds": horizon,
        "collector_wrapped": False,
        "transition_count": transition,
        "scenario_manifest_hash": manifest.content_hash,
        "scenario_ids_by_index": [str(item["scenario_id"]) for item in manifest.episodes],
        "replay_auxiliary": replay_auxiliary_metadata(),
        "joint_replay_field_order": list(replay.all_fields),
        "source_checkpoint_contract": {
            "state_contract": "fixture",
            "movement_state_dim": MOVEMENT_STATE_DIM,
            "num_uav": 16,
            "movement_state_feature_schema": movement_state_feature_schema(),
            "visual_sensing_configuration": visual,
            "channel_configuration": {"routing_slot_seconds": 0.25},
            "production_task_deadline_seconds": {"FOV": 2.5, "COM": 2.0},
            "maximum_3d_communication_distance_m": 400.0,
            "ground_station_position_m": [0.0, 0.0, 0.0],
            "task_potential_configuration": {
                "constraint_penalty_weights": {"c9": 1.0, "c10": 1.0, "com_range": 1.0}
            },
        },
        "source_training_environment_contract": {
            "training_environment_width_m": 1000,
            "training_environment_height_m": 1000,
        },
    }
    _write_json(source / "metadata.json", metadata)
    result = run_baseline(
        source_directories=[source],
        samples_per_source=transition,
        lambdas=[0.0, 0.001],
        batch_size=2,
        output_dir=tmp_path / "fixed-baseline",
    )
    return Path(result["output_directory"])


def _candidate(*, passing=True, code=None, name="candidate"):
    if code is None:
        feature = (
            "np.clip(obs[\"state\"][0] * obs[\"state\"][0], 0.0, 1.0)"
            if passing
            else "0.0"
        )
        code = (
            "def compute_extra_state(obs, constants):\n"
            f"    return np.asarray([{feature}], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    return {
        "schema_version": "uav-hrl-llm-candidate-v1",
        "candidate_name": name,
        "reward_input_mode": "current_only",
        "features": [
            {
                "index": 0,
                "name": "state_distance_helper",
                "dtype": "float32",
                "description": "A bounded current-state feature.",
                "range": {"minimum": -1.0, "maximum": 1.0},
                "source_fields": ["obs.state"],
                "formula": "bounded square of original state index 0",
                "missing_data_rule": "The original state is always present.",
            }
        ],
        "reward_terms": [
            {
                "index": 0,
                "name": "zero_term",
                "dtype": "float32",
                "description": "No supplementary reward in this fixture.",
                "range": {"minimum": 0.0, "maximum": 1.0},
                "source_fields": ["obs.state"],
                "formula": "0",
                "missing_data_rule": "Always zero.",
                "weight": 0.0,
            }
        ],
        "code": code,
    }


def _response(candidate, *, finish_reason="stop", model="qwen/qwen3.5-9b"):
    return {
        "request": {
            "model": model,
            "temperature": 0.3,
            "max_tokens": 4096,
            "seed": 20260927,
        },
        "raw": {"model": model, "choices": []},
        "content": json.dumps(candidate),
        "reasoning": None,
        "finish_reason": finish_reason,
        "actual_model": model,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "adapter": "qwen" if "qwen" in model else "gemma",
        "fallbacks": [],
        "seed_sent": True,
        "structured_output_sent": True,
    }


class MockClient:
    def __init__(self, responses, model="qwen/qwen3.5-9b"):
        self.responses = list(responses)
        self.model = model
        self.calls = []

    def list_models(self):
        return {
            "openai": {"data": [{"id": self.model}]},
            "native": {
                "models": [
                    {
                        "key": self.model,
                        "display_name": "fixture",
                        "quantization": {"name": "Q4_K_M"},
                        "max_context_length": 32768,
                        "loaded_instances": [
                            {"id": self.model, "config": {"context_length": 20000}}
                        ],
                    }
                ]
            },
            "native_error": None,
        }

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FallbackHTTPClient(LMStudioClient):
    def __init__(self, first_error):
        super().__init__(timeout=1, retries=0)
        self.first_error = first_error
        self.payloads = []

    def _request(self, method, url, payload=None):
        self.payloads.append(json.loads(json.dumps(payload)))
        if len(self.payloads) == 1:
            raise APIError(self.first_error)
        return {
            "model": "qwen/qwen3.5-9b",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"ok":true}'},
                }
            ],
            "usage": {"total_tokens": 2},
        }


@pytest.fixture
def design_fixture(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    from llm_design_contract import load_design_inputs

    arrays, metadata, baseline, constants = load_design_inputs(fixed)
    return fixed, arrays, metadata, baseline, constants


def test_prompt_is_complete_current_only_and_example_parses(design_fixture):
    _, _, metadata, baseline, constants = design_fixture
    prompt = render_prompt(
        fixed_metadata=metadata,
        baseline_report=baseline,
        constants_metadata=constants,
        beta=1.0,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        round_request="Generate the first candidate.",
    )
    assert "{{" not in prompt
    assert "current-only" in prompt
    assert "d_R=(h^2+d^2)/sqrt" in prompt
    assert "never use an object ID as a compact row" in prompt
    block = format_schema_and_example(candidate_schema())
    example = json.loads(block.split("Parseable format-only example", 1)[1].split(":\n", 1)[1])
    validate_candidate(example, constants)


def test_reward_term_schema_declares_weight_without_conflicting_inheritance():
    schema = candidate_schema()
    reward_schema = schema["$defs"]["reward_term"]
    assert reward_schema["additionalProperties"] is False
    assert "weight" in reward_schema["required"]
    assert reward_schema["properties"]["weight"] == {"type": "number"}
    assert "allOf" not in reward_schema


def test_dry_run_cli_prints_model_parameters_and_token_budget(monkeypatch, capsys):
    received = {}

    def fake_run_design(**kwargs):
        received.update(kwargs)
        return {
            "status": "dry_run_complete",
            "output_directory": "out",
            "metadata": {
                "model": {"requested_api_identifier": kwargs["model"]},
                "generation": {
                    "temperature": kwargs["temperature"],
                    "max_output_tokens": kwargs["max_output_tokens"],
                    "seed_requested": kwargs["seed"],
                    "max_attempts": kwargs["max_attempts"],
                    "api_timeout_seconds_per_request": kwargs["timeout"],
                    "worker_timeout_seconds": kwargs["worker_timeout"],
                },
                "context": {"effective_budget": kwargs["context_length"]},
                "first_prompt_token_budget": {
                    "estimated_prompt_tokens_lower": 100,
                    "estimated_prompt_tokens_upper": 120,
                    "reserved_output_tokens": kwargs["max_output_tokens"],
                    "estimated_total_upper": 248,
                    "fits_client_budget": True,
                },
            },
        }

    monkeypatch.setattr(
        run_llm_design,
        "run_design",
        fake_run_design,
    )
    assert (
        run_llm_design.main(
            [
                "--fixed-sample",
                "fixture",
                "--model",
                "qwen/qwen3.5-9b",
                "--timeout",
                "601",
                "--worker-timeout",
                "7",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Model: qwen/qwen3.5-9b" in output
    assert "temperature=0.3" in output
    assert "prompt_estimate=100..120" in output
    assert "fits=True" in output
    assert "api_per_request=601.0, worker=7.0" in output
    assert received["timeout"] == 601.0
    assert received["worker_timeout"] == 7.0


def test_lm_studio_structured_and_seed_fallbacks_are_explicit():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    structured = FallbackHTTPClient("response_format json_schema unsupported")
    result = structured.chat(
        model="qwen/qwen3.5-9b",
        prompt="return json",
        temperature=0.3,
        max_output_tokens=10,
        seed=7,
        schema=schema,
    )
    assert "response_format" in structured.payloads[0]
    assert "response_format" not in structured.payloads[1]
    assert result["structured_output_sent"] is False
    assert result["fallbacks"]

    seed = FallbackHTTPClient("seed is unsupported")
    result = seed.chat(
        model="qwen/qwen3.5-9b",
        prompt="return json",
        temperature=0.3,
        max_output_tokens=10,
        seed=7,
        schema=schema,
    )
    assert "seed" in seed.payloads[0]
    assert "seed" not in seed.payloads[1]
    assert result["seed_sent"] is False


def test_missing_model_lists_visible_api_identifiers():
    with pytest.raises(APIError, match="available model IDs.*visible/model"):
        model_inventory_summary(
            {
                "openai": {"data": [{"id": "visible/model"}]},
                "native": None,
                "native_error": None,
            },
            "missing/model",
        )


def test_forbidden_json_and_code_are_rejected(design_fixture):
    _, _, _, _, constants = design_fixture
    with pytest.raises(CandidateError, match="duplicate JSON key"):
        parse_candidate_json('{"a":1,"a":2}')
    with pytest.raises(CandidateError, match="non-finite"):
        parse_candidate_json('{"a":NaN}')
    extra = _candidate()
    extra["unexpected"] = True
    with pytest.raises(CandidateError, match="top-level"):
        validate_candidate(extra, constants)
    forbidden = _candidate(
        code=(
            "import os\n"
            "def compute_extra_state(obs, constants):\n    return np.asarray([0], dtype=np.float32)\n"
            "def compute_reward_terms(obs, constants):\n    return np.asarray([0], dtype=np.float32)\n"
        )
    )
    with pytest.raises(CandidateError, match="two top-level functions|disallowed"):
        validate_candidate(forbidden, constants)


@pytest.mark.parametrize(
    "feature_lines",
    [
        '    return np.asarray([obs["state"][0]], dtype=np.float32)',
        (
            '    copied = obs["state"][0]\n'
            "    forwarded = copied\n"
            "    return np.asarray([forwarded], dtype=np.float32)"
        ),
    ],
)
def test_direct_original_state_copy_is_statically_rejected(design_fixture, feature_lines):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            f"{feature_lines}\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    with pytest.raises(CandidateError, match=r"feature\[0\].*direct copy.*state.*\[0\]"):
        validate_candidate(candidate, constants)


def test_duplicate_statically_normalized_feature_expressions_are_rejected(design_fixture):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    derived = np.clip(obs["state"][0] * obs["state"][0], 0.0, 1.0)\n'
            "    return np.asarray([derived, derived], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    second = dict(candidate["features"][0])
    second.update(index=1, name="second", formula="intentionally different metadata")
    candidate["features"].append(second)
    with pytest.raises(CandidateError, match=r"features\[0\] and \[1\].*same"):
        validate_candidate(candidate, constants)


def test_augassign_updates_binding_instead_of_reusing_stale_alias(design_fixture):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    x = np.clip(obs["state"][0], -1.0, 1.0)\n'
            "    y = x\n"
            "    x *= x\n"
            "    return np.asarray([x, y], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    second = dict(candidate["features"][0])
    second.update(index=1, name="pre_square_value", formula="bounded pre-square value")
    candidate["features"].append(second)
    report = validate_candidate(candidate, constants)
    assert report["explicit_feature_redundancy_check"]["status"].endswith("passed")


def test_control_flow_is_reported_unresolved_not_as_confirmed_duplicate(design_fixture):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    x = np.clip(obs["state"][0], -1.0, 1.0)\n'
            "    y = x\n"
            "    if x > 0.0:\n"
            "        x = x * x\n"
            "    return np.asarray([x, y], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    second = dict(candidate["features"][0])
    second.update(index=1, name="branch_independent", formula="pre-branch value")
    candidate["features"].append(second)
    report = validate_candidate(candidate, constants)
    assert report["explicit_feature_redundancy_check"]["status"] == "not_statically_resolved"


def test_derived_feature_is_allowed_and_numeric_coincidence_remains_warning(design_fixture):
    _, arrays, _, _, constants = design_fixture
    derived = _candidate()
    validation = validate_candidate(derived, constants)
    assert validation["explicit_feature_redundancy_check"]["status"].endswith("passed")

    coincident = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    return np.asarray([abs(obs["state"][0])], dtype=np.float32)\n\n'
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    validate_candidate(coincident, constants)
    extra, terms, _ = execute_candidate_isolated(
        coincident, build_obs_arrays(arrays), constants, timeout=10
    )
    diagnostics = candidate_numeric_diagnostics(arrays["state"], extra, terms, coincident)
    assert any("numerically duplicates original state" in item for item in diagnostics["warnings"])


def test_direct_state_copy_cannot_create_approved_artifact(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    direct = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    return np.asarray([obs["state"][0]], dtype=np.float32)\n\n'
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    result = run_design(
        fixed_sample=fixed,
        model="qwen/qwen3.5-9b",
        client=MockClient([_response(direct)]),
        max_attempts=1,
        output_dir=tmp_path / "direct-copy",
        worker_timeout=10,
    )
    assert result["status"] == "failed_no_approved_candidate"
    assert not (tmp_path / "direct-copy" / "approved").exists()


@pytest.mark.parametrize(
    "code, message",
    [
        (
            "def compute_extra_state(obs, constants):\n    return np.asarray([[0.0]], dtype=np.float32)\n\ndef compute_reward_terms(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)\n",
            "one-dimensional",
        ),
        (
            "def compute_extra_state(obs, constants):\n    return np.asarray([0.0], dtype=np.float64)\n\ndef compute_reward_terms(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)\n",
            "dtype float32",
        ),
        (
            "def compute_extra_state(obs, constants):\n    return np.asarray([2.0], dtype=np.float32)\n\ndef compute_reward_terms(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)\n",
            "outside",
        ),
        (
            "def compute_extra_state(obs, constants):\n    return np.asarray([np.log(-1.0)], dtype=np.float32)\n\ndef compute_reward_terms(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)\n",
            "NaN",
        ),
        (
            "def compute_extra_state(obs, constants):\n    obs[\"state\"][0] = 0.0\n    return np.asarray([0.0], dtype=np.float32)\n\ndef compute_reward_terms(obs, constants):\n    return np.asarray([0.0], dtype=np.float32)\n",
            "assignment",
        ),
    ],
)
def test_worker_rejects_shape_dtype_range_nonfinite_and_input_rebinding(
    design_fixture, code, message
):
    _, arrays, _, _, constants = design_fixture
    candidate = _candidate(code=code)
    if 'obs["state"][0] =' in code:
        with pytest.raises(CandidateError, match=message):
            validate_candidate(candidate, constants)
    else:
        with pytest.raises(CandidateExecutionError, match=message):
            execute_candidate_isolated(
                candidate, build_obs_arrays(arrays), constants, timeout=10
            )


def test_worker_timeout_is_terminable(design_fixture):
    _, arrays, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            "    while True:\n        pass\n"
            "    return np.asarray([0.0], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        )
    )
    with pytest.raises(CandidateExecutionError, match="exceeded"):
        execute_candidate_isolated(
            candidate, build_obs_arrays(arrays), constants, timeout=1.0
        )


@pytest.mark.parametrize("invalid_output", ["state", "reward"])
def test_empty_probe_uses_full_output_range_validation(design_fixture, invalid_output):
    _, arrays, _, _, constants = design_fixture
    obs_arrays = {
        name: np.asarray(value).copy()
        for name, value in build_obs_arrays(arrays).items()
    }
    obs_arrays["state"][:, 0] = 0.5
    feature_value = "bad" if invalid_output == "state" else "0.0"
    reward_value = "bad" if invalid_output == "reward" else "0.0"
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    bad = 2.0 if np.count_nonzero(obs["state"]) == 0 else 0.0\n'
            f"    return np.asarray([{feature_value}], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            '    bad = 2.0 if np.count_nonzero(obs["state"]) == 0 else 0.0\n'
            f"    return np.asarray([{reward_value}], dtype=np.float32)\n"
        )
    )
    expected = (
        r"compute_extra_state\(empty probe\).*feature\[0\]"
        if invalid_output == "state"
        else r"compute_reward_terms\(empty probe\).*reward_terms\[0\]"
    )
    with pytest.raises(CandidateExecutionError, match=expected):
        execute_candidate_isolated(candidate, obs_arrays, constants, timeout=10)


def test_legal_empty_probe_still_passes(design_fixture):
    _, arrays, _, _, constants = design_fixture
    extra, terms, report = execute_candidate_isolated(
        _candidate(), build_obs_arrays(arrays), constants, timeout=10
    )
    assert extra.shape == (6, 1)
    assert terms.shape == (6, 1)
    assert report["empty_probe_check"] == "passed"


@pytest.mark.parametrize("model", ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])
def test_mock_api_first_candidate_passes_for_qwen_and_gemma(tmp_path, model):
    fixed = _fixed_artifact(tmp_path)
    client = MockClient([_response(_candidate(), model=model)], model=model)
    result = run_design(
        fixed_sample=fixed,
        model=model,
        client=client,
        output_dir=tmp_path / "design",
        timeout=10,
    )
    assert result["status"] == "approved"
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == model
    assert client.calls[0]["temperature"] == 0.3
    assert client.calls[0]["max_output_tokens"] == 4096
    assert client.calls[0]["seed"] == 20260927
    assert (Path(result["approved_artifact"]) / "artifact.json").is_file()


def test_api_and_worker_timeouts_are_independent(tmp_path, monkeypatch):
    fixed = _fixed_artifact(tmp_path)
    observed = []
    original = llm_design.execute_candidate_isolated

    def capture_worker_timeout(*args, **kwargs):
        observed.append(kwargs["timeout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(llm_design, "execute_candidate_isolated", capture_worker_timeout)
    client = MockClient([_response(_candidate())])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        output_dir=tmp_path / "separate-timeouts",
        timeout=601,
        worker_timeout=9,
    )
    assert result["status"] == "approved"
    assert observed == [9]
    generation = result["metadata"]["generation"]
    assert generation["api_timeout_seconds_per_request"] == 601
    assert generation["worker_timeout_seconds"] == 9


def test_revision_then_pass_and_max_attempt_exhaustion(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    failing = _response(_candidate(passing=False, name="first"))
    passing = _response(_candidate(passing=True, name="second"))
    client = MockClient([failing, passing])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=2,
        output_dir=tmp_path / "revision",
        timeout=10,
    )
    assert result["status"] == "approved"
    second_prompt = (tmp_path / "revision" / "attempt_02" / "prompt.txt").read_text()
    assert '"candidate_name": "first"' in second_prompt
    assert "Latest validation/evaluation feedback for that same output" in second_prompt

    exhausted_client = MockClient([failing, failing])
    exhausted = run_design(
        fixed_sample=fixed,
        model=exhausted_client.model,
        client=exhausted_client,
        max_attempts=2,
        output_dir=tmp_path / "exhausted",
        timeout=10,
    )
    assert exhausted["status"] == "failed_no_approved_candidate"
    assert not (tmp_path / "exhausted" / "approved").exists()


def test_parse_failure_revision_includes_latest_raw_content_and_error(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    invalid_text = '{"candidate_name":"broken", invalid-json-here}'
    invalid = {**_response(_candidate()), "content": invalid_text}
    client = MockClient([invalid, _response(_candidate(name="fixed"))])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=2,
        output_dir=tmp_path / "parse-revision",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    prompt = (tmp_path / "parse-revision" / "attempt_02" / "prompt.txt").read_text()
    assert invalid_text in prompt
    assert "invalid JSON" in prompt
    assert "Previous failed raw final content" in prompt


def test_latest_invalid_json_does_not_reuse_older_parsed_candidate(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    old = _response(_candidate(passing=False, name="older-parsed-candidate"))
    newest_text = "LATEST-BROKEN-JSON"
    newest = {**_response(_candidate()), "content": newest_text}
    client = MockClient([old, newest, _response(_candidate(name="fixed"))])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=3,
        output_dir=tmp_path / "latest-output",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    prompt = (tmp_path / "latest-output" / "attempt_03" / "prompt.txt").read_text()
    assert newest_text in prompt
    assert '"candidate_name": "older-parsed-candidate"' not in prompt


def test_long_failed_output_is_marked_and_reasoning_only_is_not_replayed(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    long_invalid = '{"candidate_name":"' + ("x" * 60_000) + '", invalid-json-here}'
    long_client = MockClient(
        [{**_response(_candidate()), "content": long_invalid}, _response(_candidate())]
    )
    result = run_design(
        fixed_sample=fixed,
        model=long_client.model,
        client=long_client,
        max_attempts=2,
        context_length=16_000,
        output_dir=tmp_path / "long-output",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    prompt = (tmp_path / "long-output" / "attempt_02" / "prompt.txt").read_text()
    assert "[TRUNCATED:" in prompt
    assert long_invalid not in prompt
    assert "invalid-json-here" in prompt
    budget = json.loads(
        (tmp_path / "long-output" / "attempt_02" / "token_budget.json").read_text()
    )
    assert budget["fits_client_budget"] is True

    reasoning_client = MockClient(
        [
            {
                **_response(_candidate()),
                "content": None,
                "reasoning": "PRIVATE-REASONING-MUST-NOT-BE-REPLAYED",
            },
            _response(_candidate()),
        ]
    )
    result = run_design(
        fixed_sample=fixed,
        model=reasoning_client.model,
        client=reasoning_client,
        max_attempts=2,
        output_dir=tmp_path / "reasoning-output",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    prompt = (tmp_path / "reasoning-output" / "attempt_02" / "prompt.txt").read_text()
    assert "reasoning, but no final JSON" in prompt
    assert "PRIVATE-REASONING-MUST-NOT-BE-REPLAYED" not in prompt


@pytest.mark.parametrize(
    "response, expected",
    [
        (APIError("timeout"), "api_failure"),
        (
            {
                **_response(_candidate()),
                "finish_reason": "length",
            },
            "candidate_format_or_validation_failure",
        ),
        ({**_response(_candidate()), "content": "not json"}, "candidate_format_or_validation_failure"),
        (
            {**_response(_candidate()), "content": None, "reasoning": "analysis only"},
            "candidate_format_or_validation_failure",
        ),
    ],
)
def test_api_failure_truncation_and_illegal_json_are_distinct(tmp_path, response, expected):
    fixed = _fixed_artifact(tmp_path)
    client = MockClient([response])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=1,
        output_dir=tmp_path / "design",
        timeout=10,
    )
    assert result["status"] == "failed_no_approved_candidate"
    assert result["metadata"]["attempt_history"][0]["status"] == expected


def test_context_overflow_dry_run_never_calls_api(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    client = MockClient([])
    with pytest.raises(Exception, match="context budget"):
        run_design(
            fixed_sample=fixed,
            model=client.model,
            client=client,
            context_length=100,
            max_output_tokens=50,
            output_dir=tmp_path / "dry",
            dry_run=True,
        )
    assert client.calls == []
    metadata = json.loads((tmp_path / "dry" / "run_metadata.json").read_text())
    assert metadata["status"] == "failed_context_budget"


def test_candidate_uses_fixed_pair_set_and_zero_pair_remains_excluded():
    state = np.asarray([[0.0], [0.0], [1.0]], dtype=np.float64)
    distances = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    arrays = {
        "delivered_mbits": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        "total_mobility_energy": np.zeros((3, 1), dtype=np.float32),
        "c9_penalty": np.zeros((3, 1), dtype=np.float32),
        "c10_penalty": np.zeros((3, 1), dtype=np.float32),
        "com_range_penalty": np.zeros((3, 1), dtype=np.float32),
    }
    context = EvaluationContext(
        original_state=state,
        original_distances=distances,
        primary_mask=distances > 1e-8,
        zero_mask=distances == 0,
        near_mask=np.zeros(3, dtype=bool),
        base_rewards={"0": np.asarray([0.0, 1.0, 2.0])},
        lambdas=(0.0,),
        baseline_values={"0": 2.0},
        distance_epsilon=1e-8,
        reward_epsilon=1e-12,
        fixed_metadata={
            "selection": [{"fixed_index": i} for i in range(3)]
        },
        fixed_arrays=arrays,
        pair_hash="fixed",
    )
    extra = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32)
    terms = np.zeros((3, 1), dtype=np.float32)
    report = evaluate_candidate(
        context,
        _candidate(),
        extra,
        terms,
        beta=1.0,
        batch_size=1,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
    )
    assert report["fixed_primary_pair_count"] == 2
    assert report["baseline_excluded_pairs"]["zero_pairs_with_nonzero_augmented_distance"] == 1
    assert report["baseline_excluded_pairs"]["included_in_primary_candidate_estimate"] is False
    assert report["direct_concat_distance_check"]["passed"] is True


def test_approved_artifact_reload_recomputes_identically(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    client = MockClient([_response(_candidate())])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        output_dir=tmp_path / "design",
        timeout=10,
    )
    design = load_approved_design(result["approved_artifact"])
    from llm_design_contract import load_design_inputs

    arrays, _, _, _ = load_design_inputs(fixed)
    extra, terms, reward = design.evaluate_fixed_samples(arrays, timeout=10)
    assert extra.shape == (6, 1)
    assert terms.shape == (6, 1)
    assert np.array_equal(reward, np.zeros(6))
    assert not list((tmp_path / "design").glob("*training*"))
