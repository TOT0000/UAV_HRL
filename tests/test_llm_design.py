import json
from pathlib import Path

import numpy as np
import pytest

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM, movement_state_feature_schema
from llm_baseline import run_baseline
from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
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
        feature = "obs[\"state\"][0]" if passing else "0.0"
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
                "formula": feature if code is None else "candidate formula",
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
    assert "Latest validation/evaluation feedback" in second_prompt

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
