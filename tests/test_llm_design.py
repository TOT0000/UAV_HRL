import copy
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import urllib.request

import numpy as np
import pytest
import llm_candidate
import llm_candidate_worker
import llm_design
import llm_streaming
import run_llm_design

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM, movement_state_feature_schema
from llm_baseline import run_baseline
from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    candidate_semantic_fingerprint,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    feature_reward,
    load_approved_design,
    parse_candidate_json,
    parse_candidate_json_envelope,
    validate_candidate,
)
from llm_design import (
    APIError,
    EvaluationContext,
    LMStudioClient,
    OpenAIClient,
    evaluate_candidate,
    model_inventory_summary,
    response_model_matches,
    run_design,
)
from llm_streaming import (
    SSEDecoder,
    StreamTransportError,
    capture_chat_stream,
    open_http_stream,
)
from llm_design_contract import (
    PROMPT_TEMPLATE_PATH,
    PROMPT_VERSION,
    SUPPORTED_OPERATIONS,
    build_constants,
    build_obs_arrays,
    candidate_schema,
    format_schema_and_example,
    render_environment_interface,
    render_prompt,
    runtime_diagnostic_contract,
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
            f"    return np.asarray([{feature}], dtype=np.float32)\n"
        )
    elif "\ndef compute_reward_terms" in code:
        # Legacy snippets below focus on compute_extra_state behavior. Adapt
        # them to the v2 single-function contract before building the fixture.
        code = code.split("\ndef compute_reward_terms", 1)[0].rstrip() + "\n"
    return {
        "schema_version": "uav-hrl-llm-shared-feature-candidate-v2",
        "candidate_name": name,
        "reward_input_mode": "current_only",
        "features": [
            {
                "index": 0,
                "name": "state_distance_helper",
                "dtype": "float32",
                "description": "A bounded current-state feature.",
                "range": {"minimum": 0.0, "maximum": 1.0},
                "source_fields": ["obs.state"],
                "formula": "bounded square of original state index 0",
                "missing_data_rule": "The original state is always present.",
                "reward_weight": 0.0,
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
            "stream": True,
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
        "structured_output_sent": False,
        "transport_completed": True,
        "done_received": True,
        "terminal_chunk_received": True,
        "tool_calls_seen": False,
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


class ChunkedResponse:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def read1(self, _size):
        if not self.chunks:
            return b""
        value = self.chunks.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        self.closed = True


class StreamingHTTPClient(LMStudioClient):
    def __init__(self, chunks, **kwargs):
        super().__init__(retries=0, progress_interval=60, **kwargs)
        self.response = ChunkedResponse(chunks)
        self.sent_request = None

    def _open_stream(
        self, request, *, connect_timeout, header_timeout, total_timeout
    ):
        self.sent_request = request
        self.connect_timeout_used = connect_timeout
        self.header_timeout_used = header_timeout
        self.total_timeout_used = total_timeout
        return self.response


class OpenAIStreamingClient(OpenAIClient):
    def __init__(self, chunks, **kwargs):
        super().__init__(retries=0, progress_interval=60, **kwargs)
        self.response = ChunkedResponse(chunks)
        self.sent_request = None

    def _open_stream(
        self, request, *, connect_timeout, header_timeout, total_timeout
    ):
        self.sent_request = request
        return self.response


@contextmanager
def _local_stream_server(behavior):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            try:
                behavior(self)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def _send_stream_headers(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.flush()


def _send_sse(handler, data):
    handler.wfile.write((f"data: {data}\n\n").encode("utf-8"))
    handler.wfile.flush()


def _complete_sse(handler, *, content="ok", model="qwen/qwen3.5-9b"):
    _send_sse(
        handler,
        json.dumps(
            {
                "model": model,
                "choices": [
                    {"delta": {"content": content}, "finish_reason": "stop"}
                ],
            },
        ),
    )
    _send_sse(handler, "[DONE]")


@pytest.fixture
def design_fixture(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    from llm_design_contract import load_design_inputs

    arrays, metadata, baseline, constants = load_design_inputs(fixed)
    return fixed, arrays, metadata, baseline, constants


def test_prompt_is_complete_current_only_and_example_parses(design_fixture):
    _, arrays, metadata, baseline, constants = design_fixture
    prompt = render_prompt(
        fixed_metadata=metadata,
        baseline_report=baseline,
        constants_metadata=constants,
        beta=1.0,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-6,
        round_request="Generate the first candidate.",
    )
    assert not any(
        token in prompt
        for token in (
            "{{ENVIRONMENT_INTERFACE}}",
            "{{SUPPORTED_OPERATIONS}}",
            "{{OUTPUT_JSON_SPEC_AND_EXAMPLE}}",
            "{{BASELINE_RESULTS_BY_LAMBDA}}",
            "{{IMPROVEMENT_TOLERANCE}}",
            "{{BETA}}",
            "{{ROUND_REQUEST}}",
        )
    )
    assert "observable information available before the TD3 movement action" in prompt
    assert "Features are computed before the action" in prompt
    assert "total useful data delivered to the GS within its deadlines" in prompt
    assert "Select data using the documented indices and field-specific validity flags" in prompt
    assert "Return only the complete candidate JSON object" in prompt
    assert "d_R=(h^2+d^2)/sqrt" in prompt
    assert "never use an object ID as a compact row" in prompt
    block = format_schema_and_example(candidate_schema())
    example = json.loads(block.split("Parseable interface example", 1)[1].split(":\n", 1)[1])
    validate_candidate(example, constants)
    features, reward, _ = execute_candidate_isolated(
        example, build_obs_arrays(arrays), constants, timeout=10
    )
    assert features.shape == (6, 1)
    assert reward.tolist() == pytest.approx([0.0] * 6)

    populated = {
        name: np.asarray(value).copy()
        for name, value in build_obs_arrays(arrays).items()
    }
    populated["movement_mask"][:, :2] = True
    populated["uav_queue_valid"][:, :2] = True
    populated["uav_queue_empty"][:, 0] = True
    populated["uav_queue_empty"][:, 1] = False
    populated_features, populated_reward, _ = execute_candidate_isolated(
        example, populated, constants, timeout=10
    )
    assert populated_features[:, 0].tolist() == pytest.approx([0.5] * 6)
    assert populated_reward.tolist() == pytest.approx([0.0] * 6)


def test_master_prompt_has_exact_placeholder_contract_and_supported_operations_match():
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert PROMPT_VERSION == "uav-hrl-llm-design-prompt-v3"
    assert {
        token
        for token in (
            "{{ENVIRONMENT_INTERFACE}}",
            "{{SUPPORTED_OPERATIONS}}",
            "{{OUTPUT_JSON_SPEC_AND_EXAMPLE}}",
            "{{BASELINE_RESULTS_BY_LAMBDA}}",
            "{{IMPROVEMENT_TOLERANCE}}",
            "{{BETA}}",
            "{{ROUND_REQUEST}}",
        )
        if token in template
    } == {
        "{{ENVIRONMENT_INTERFACE}}",
        "{{SUPPORTED_OPERATIONS}}",
        "{{OUTPUT_JSON_SPEC_AND_EXAMPLE}}",
        "{{BASELINE_RESULTS_BY_LAMBDA}}",
        "{{IMPROVEMENT_TOLERANCE}}",
        "{{BETA}}",
        "{{ROUND_REQUEST}}",
    }
    assert template.count("{{") == 7
    for name in llm_candidate.SAFE_BUILTIN_CALLS:
        assert name in SUPPORTED_OPERATIONS
    for name in llm_candidate.SAFE_NUMPY_CALLS:
        assert name.removeprefix("np.") in SUPPORTED_OPERATIONS
    assert "No import statements" in SUPPORTED_OPERATIONS


def test_state_interface_is_derived_from_saved_authoritative_schema(design_fixture):
    _, _, metadata, _, constants = design_fixture
    modified = copy.deepcopy(metadata)
    schema = modified["compatibility_contract"]["source_checkpoint_contract"][
        "movement_state_feature_schema"
    ]
    schema["features"].append(
        {
            "index": schema["dimension"],
            "name": "fixture_additional_global",
            "kind": "continuous",
            "minimum": 0.0,
            "maximum": 1.0,
            "normalization": "fixture authoritative normalization",
        }
    )
    schema["dimension"] += 1
    interface = render_environment_interface(modified, constants)
    assert f"shape ({schema['dimension']},)" in interface
    assert (
        f"Index {schema['dimension'] - 1}: fixture_additional_global; "
        "fixture authoritative normalization."
    ) in interface


def test_all_providers_save_the_same_fully_rendered_first_prompt(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    cases = (
        ("lmstudio", "qwen/qwen3.5-9b"),
        ("lmstudio", "google/gemma-4-e4b"),
        ("openai", "gpt-4o"),
    )
    prompts = []
    for index, (provider, model) in enumerate(cases):
        output = tmp_path / f"shared-prompt-{index}"
        result = run_design(
            fixed_sample=fixed,
            provider=provider,
            model=model,
            output_dir=output,
            dry_run=True,
        )
        assert result["status"] == "dry_run_complete"
        prompt = (output / "prompt_attempt_01.txt").read_text(encoding="utf-8")
        assert "{{ROUND_REQUEST}}" not in prompt
        assert "1. Task and control scope" in prompt
        assert "7. Request for this round" in prompt
        prompts.append(prompt)
    assert prompts[0] == prompts[1] == prompts[2]


def test_feature_schema_declares_reward_weight_without_conflicting_inheritance():
    schema = candidate_schema()
    feature_schema = schema["$defs"]["feature"]
    assert feature_schema["additionalProperties"] is False
    assert "reward_weight" in feature_schema["required"]
    assert feature_schema["properties"]["reward_weight"] == {"type": "number"}
    assert "allOf" not in feature_schema


def test_shared_feature_weights_and_host_reward_are_strict():
    candidate = _candidate()
    candidate["features"] = [
        {
            **candidate["features"][0],
            "index": index,
            "name": f"feature_{index}",
            "formula": f"derived formula {index}",
            "reward_weight": weight,
        }
        for index, weight in enumerate((0.5, -0.25, 0.0))
    ]
    candidate["code"] = (
        "def compute_extra_state(obs, constants):\n"
        '    x = np.clip(np.abs(obs["state"][0]), 0.0, 1.0)\n'
        "    return np.asarray([x * x, x, x * x * x], dtype=np.float32)\n"
    )
    values = np.asarray([0.8, 0.4, 0.2], dtype=np.float32)
    r_extra = float(feature_reward(values, candidate))
    assert r_extra == pytest.approx(0.3)
    original = np.asarray([9.0, 8.0], dtype=np.float32)
    assert np.concatenate((original, values)).tolist() == pytest.approx(
        [9.0, 8.0, 0.8, 0.4, 0.2]
    )
    assert 2.0 + 2.0 * r_extra == pytest.approx(2.6)

    for invalid in (True, float("inf")):
        rejected = _candidate()
        rejected["features"][0]["reward_weight"] = invalid
        with pytest.raises(CandidateError, match="reward_weight|number|finite"):
            validate_candidate(rejected, {"unused": {"value": 0}})
    rejected = _candidate()
    rejected["features"][0]["reward_weight"] = 1.01
    with pytest.raises(CandidateError, match=r"sum\(abs"):
        validate_candidate(rejected, {"unused": {"value": 0}})


def test_v1_and_mixed_candidates_are_not_silently_converted(design_fixture):
    _, _, _, _, constants = design_fixture
    old = _candidate()
    old["schema_version"] = "uav-hrl-llm-candidate-v1"
    old["reward_terms"] = []
    with pytest.raises(CandidateError, match="top-level|incompatible"):
        validate_candidate(old, constants)
    mixed = _candidate()
    mixed["reward_terms"] = [{"legacy": True}]
    with pytest.raises(CandidateError, match="top-level"):
        validate_candidate(mixed, constants)


def test_candidate_fingerprint_ignores_only_name_comments_and_formatting():
    original = _candidate(name="first")
    renamed = _candidate(
        name="second",
        code=(
            "# formatting-only comment\n"
            "def compute_extra_state(obs, constants):\n"
            '    return np.asarray([ np.clip(obs["state"][0] * obs["state"][0], 0.0, 1.0) ], dtype=np.float32)\n'
        ),
    )
    assert candidate_semantic_fingerprint(original) == candidate_semantic_fingerprint(renamed)
    changed_weight = json.loads(json.dumps(renamed))
    changed_weight["features"][0]["reward_weight"] = 0.1
    assert candidate_semantic_fingerprint(original) != candidate_semantic_fingerprint(changed_weight)
    changed_sources = json.loads(json.dumps(renamed))
    changed_sources["features"][0]["source_fields"].append("obs.movement_mask")
    assert candidate_semantic_fingerprint(original) != candidate_semantic_fingerprint(changed_sources)


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
                "--connect-timeout",
                "31",
                "--total-timeout",
                "1801",
                "--progress-interval",
                "11",
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
    assert "connect=31.0, stream_idle=601.0, request_total=1801.0, worker=7.0" in output
    assert received["timeout"] == 601.0
    assert received["connect_timeout"] == 31.0
    assert received["total_timeout"] == 1801.0
    assert received["progress_interval"] == 11.0
    assert received["worker_timeout"] == 7.0
    assert received["provider"] == "lmstudio"
    assert received["base_url"] is None


def test_cli_redacts_provider_credentials_from_errors(monkeypatch, capsys):
    secret = "fixture-cli-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def fail(**_kwargs):
        raise RuntimeError(f"unexpected Authorization: Bearer {secret}")

    monkeypatch.setattr(run_llm_design, "run_design", fail)
    assert run_llm_design.main(
        [
            "--provider",
            "openai",
            "--fixed-sample",
            "fixture",
            "--model",
            "gpt-4o",
        ]
    ) == 2
    error = capsys.readouterr().err
    assert secret not in error
    assert "[REDACTED]" in error


@pytest.mark.parametrize("model", ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])
def test_dry_run_saves_streaming_unstructured_request(tmp_path, model):
    fixed = _fixed_artifact(tmp_path)
    result = run_design(
        fixed_sample=fixed,
        model=model,
        output_dir=tmp_path / "dry-run",
        dry_run=True,
    )
    assert result["status"] == "dry_run_complete"
    request = json.loads(
        (tmp_path / "dry-run" / "request_attempt_01.json").read_text()
    )
    assert request["model"] == model
    assert request["stream"] is True
    assert "response_format" not in request
    assert result["metadata"]["generation_requests_sent"] == 0


@pytest.mark.parametrize("model", ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])
def test_lm_studio_request_streams_without_response_format(tmp_path, model):
    chunks = [
        (
            'data: {"model":"%s","choices":[{"delta":{"content":"{\\"ok\\":true}"},'
            '"finish_reason":"stop"}]}\n\n' % model
        ).encode(),
        b"data: [DONE]\n\n",
    ]
    client = StreamingHTTPClient(chunks, timeout=1, total_timeout=2)
    result = client.chat(
        model=model,
        prompt="return json",
        temperature=0.3,
        max_output_tokens=10,
        seed=7,
        attempt_directory=tmp_path,
    )
    saved = json.loads((tmp_path / "request.json").read_text())
    sent = json.loads(client.sent_request.data.decode("utf-8"))
    assert saved == sent == result["request"]
    assert saved["stream"] is True
    assert "response_format" not in saved
    assert result["structured_output_sent"] is False
    assert result["transport_completed"] is True


def test_openai_request_uses_official_chat_parameters_and_redacts_key(
    tmp_path, monkeypatch
):
    secret = "fixture-openai-secret-must-not-be-saved"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    chunks = [
        b'data: {"model":"gpt-4o-2024-11-20","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
        b'data: {"model":"gpt-4o-2024-11-20","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n',
        b"data: [DONE]\n\n",
    ]
    client = OpenAIStreamingClient(chunks, timeout=1, total_timeout=2)
    assert client.retries == 0
    result = client.chat(
        model="gpt-4o",
        prompt="return json",
        temperature=0.3,
        max_output_tokens=4096,
        seed=20260927,
        attempt_directory=tmp_path,
    )
    payload = json.loads(client.sent_request.data.decode("utf-8"))
    assert client.sent_request.full_url == "https://api.openai.com/v1/chat/completions"
    assert client.sent_request.get_header("Authorization") == f"Bearer {secret}"
    assert payload["max_completion_tokens"] == 4096
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.3
    assert payload["seed"] == 20260927
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "response_format" not in payload
    assert result["actual_model"] == "gpt-4o-2024-11-20"
    assert result["usage"]["total_tokens"] == 3
    assert result["raw"]["reconstructed_from_stream"] is True
    assert (tmp_path / "stream_events.jsonl").is_file()
    saved_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.iterdir()
        if path.is_file()
    )
    assert secret not in saved_text


def test_openai_key_is_not_sent_to_lmstudio(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai-only-secret")
    monkeypatch.delenv("LM_STUDIO_API_TOKEN", raising=False)
    client = StreamingHTTPClient(
        [
            b'data: {"model":"qwen/qwen3.5-9b","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
        timeout=1,
        total_timeout=2,
    )
    client.chat(
        model="qwen/qwen3.5-9b",
        prompt="return json",
        temperature=0.3,
        max_output_tokens=10,
        seed=7,
        attempt_directory=tmp_path,
    )
    assert client.sent_request.get_header("Authorization") is None


def test_openai_missing_key_fails_but_dry_run_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIClient(timeout=1, total_timeout=2)
    with pytest.raises(APIError, match="OPENAI_API_KEY") as captured:
        client.chat(
            model="gpt-4o",
            prompt="return json",
            temperature=0.3,
            max_output_tokens=10,
            seed=7,
            attempt_directory=tmp_path / "missing-key",
        )
    assert captured.value.category == "authentication_error"

    fixed = _fixed_artifact(tmp_path)
    result = run_design(
        fixed_sample=fixed,
        provider="openai",
        model="gpt-4o",
        output_dir=tmp_path / "openai-dry-run",
        dry_run=True,
    )
    assert result["status"] == "dry_run_complete"
    assert result["metadata"]["generation_requests_sent"] == 0
    assert result["metadata"]["model"]["inventory"] is None
    request = json.loads(
        (tmp_path / "openai-dry-run" / "request_attempt_01.json").read_text()
    )
    assert request["max_completion_tokens"] == 4096
    assert "max_tokens" not in request


def test_openai_model_alias_accepts_only_documented_gpt4o_snapshots():
    assert response_model_matches("openai", "gpt-4o", "gpt-4o")
    assert response_model_matches("openai", "gpt-4o", "gpt-4o-2024-08-06")
    assert not response_model_matches("openai", "gpt-4o", "gpt-4o-mini")
    assert not response_model_matches("openai", "gpt-4o", "gpt-4o-made-up")
    assert response_model_matches(
        "openai", "gpt-4o-2024-11-20", "gpt-4o-2024-11-20"
    )
    assert not response_model_matches(
        "openai", "gpt-4o-2024-11-20", "gpt-4o"
    )
    with pytest.raises(ValueError, match="official base URL"):
        OpenAIClient("https://example.invalid/v1")


def test_openai_provider_skips_lmstudio_inventory_and_accepts_snapshot(
    tmp_path,
):
    fixed = _fixed_artifact(tmp_path)

    class NoInventoryClient(MockClient):
        def list_models(self):
            raise AssertionError("OpenAI provider must not call LM Studio model APIs")

    response = _response(_candidate(), model="gpt-4o")
    response["actual_model"] = "gpt-4o-2024-11-20"
    response["raw"]["model"] = "gpt-4o-2024-11-20"
    client = NoInventoryClient([response], model="gpt-4o")
    result = run_design(
        fixed_sample=fixed,
        provider="openai",
        model="gpt-4o",
        client=client,
        max_attempts=1,
        output_dir=tmp_path / "openai-mock-run",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    assert result["metadata"]["model"]["provider"] == "openai"
    assert result["metadata"]["model"]["actual_api_identifiers_reported"] == [
        "gpt-4o-2024-11-20"
    ]


def test_sse_decoder_handles_utf8_and_event_boundaries():
    decoder = SSEDecoder()
    raw = (
        ': keepalive\r\n\r\n'
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"候選"}}]}\n\n'
        'data: [DONE]\n\n'
    ).encode("utf-8")
    marker = raw.index("候".encode("utf-8")) + 1
    chunks = [raw[:7], raw[7:marker], raw[marker:marker + 1], raw[marker + 1:]]
    events = []
    for chunk in chunks:
        events.extend(decoder.feed(chunk))
    events.extend(decoder.feed(b"", final=True))
    assert len(events) == 3
    assert json.loads(events[0])["choices"][0]["delta"]["role"] == "assistant"
    assert json.loads(events[1])["choices"][0]["delta"]["content"] == "候選"
    assert events[2] == "[DONE]"


def test_stream_accumulates_content_reasoning_and_optional_usage(tmp_path):
    response = ChunkedResponse(
        [
            b'data: {"model":"fixture","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
    )
    result = capture_chat_stream(
        response,
        directory=tmp_path,
        idle_timeout=1,
        total_timeout=2,
        progress_interval=60,
    )
    assert result["content"] == "answer"
    assert result["reasoning"] == "think"
    assert result["usage"] is None
    assert result["done_received"] is True
    assert (tmp_path / "response_content.partial.txt").read_text() == "answer"
    assert (tmp_path / "reasoning.partial.txt").read_text() == "think"

    with_usage = capture_chat_stream(
        ChunkedResponse(
            [
                b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n',
                b"data: [DONE]\n\n",
            ]
        ),
        directory=tmp_path / "with-usage",
        idle_timeout=1,
        total_timeout=2,
        progress_interval=60,
    )
    assert with_usage["usage"] == {"prompt_tokens": 2, "completion_tokens": 1}

    refused = capture_chat_stream(
        ChunkedResponse(
            [
                b'data: {"choices":[{"delta":{"refusal":"not allowed"},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        ),
        directory=tmp_path / "refusal",
        idle_timeout=1,
        total_timeout=2,
        progress_interval=60,
    )
    assert refused["refusal"] == "not allowed"
    assert refused["content"] == ""


def test_partial_stream_is_flushed_before_connection_failure(tmp_path):
    response = ChunkedResponse(
        [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            OSError("connection reset"),
        ]
    )
    with pytest.raises(StreamTransportError) as captured:
        capture_chat_stream(
            response,
            directory=tmp_path,
            idle_timeout=1,
            total_timeout=2,
            progress_interval=60,
        )
    assert captured.value.category == "stream_connection_error"
    assert (tmp_path / "response_content.partial.txt").read_text() == "partial"
    assert (tmp_path / "stream_events.jsonl").read_text().strip()
    status = json.loads((tmp_path / "stream_status.json").read_text())
    assert status["status"] == "stream_connection_error"


@pytest.mark.parametrize(
    "chunks, category",
    [
        (
            [b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'],
            "stream_incomplete",
        ),
        ([b'data: {"error":{"message":"bad stream"}}\n\n'], "stream_server_error"),
        ([b"data: not-json\n\n"], "stream_protocol_error"),
    ],
)
def test_stream_incomplete_server_and_protocol_errors_are_classified(
    tmp_path, chunks, category
):
    with pytest.raises(StreamTransportError) as captured:
        capture_chat_stream(
            ChunkedResponse(chunks),
            directory=tmp_path,
            idle_timeout=1,
            total_timeout=2,
            progress_interval=60,
        )
    assert captured.value.category == category
    assert json.loads((tmp_path / "stream_status.json").read_text())["status"] == category


def test_stream_total_timeout_and_cancel_are_persisted(tmp_path, monkeypatch):
    release = threading.Event()

    class BlockingResponse:
        def read1(self, _size):
            release.wait(5)
            return b""

        def close(self):
            release.set()

    with pytest.raises(StreamTransportError) as timeout_error:
        capture_chat_stream(
            BlockingResponse(),
            directory=tmp_path / "timeout",
            idle_timeout=1,
            total_timeout=0.05,
            progress_interval=60,
        )
    assert timeout_error.value.category == "stream_total_timeout"

    original_get = llm_streaming.queue.Queue.get

    def cancel_get(self, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(llm_streaming.queue.Queue, "get", cancel_get)
    with pytest.raises(StreamTransportError) as cancelled:
        capture_chat_stream(
            ChunkedResponse([]),
            directory=tmp_path / "cancel",
            idle_timeout=1,
            total_timeout=2,
            progress_interval=60,
        )
    assert cancelled.value.category == "stream_cancelled"
    assert json.loads(
        (tmp_path / "cancel" / "stream_status.json").read_text()
    )["status"] == "cancelled"
    monkeypatch.setattr(llm_streaming.queue.Queue, "get", original_get)


@pytest.mark.parametrize("model", ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])
def test_real_http_delayed_first_sse_outlives_connect_timeout(tmp_path, model):
    def behavior(handler):
        _send_stream_headers(handler)
        time.sleep(0.35)
        _complete_sse(handler, model=model)

    with _local_stream_server(behavior) as base_url:
        directory = tmp_path / model.split("/")[0]
        directory.mkdir()
        client = LMStudioClient(
            base_url,
            connect_timeout=0.05,
            timeout=1.0,
            total_timeout=2.0,
            retries=0,
            progress_interval=60,
        )
        started = time.monotonic()
        result = client.chat(
            model=model,
            prompt="return json",
            temperature=0.3,
            max_output_tokens=10,
            seed=7,
            attempt_directory=directory,
        )
        elapsed = time.monotonic() - started
    assert result["content"] == "ok"
    assert elapsed >= 0.30
    assert elapsed < 1.5


def test_real_http_header_wait_uses_idle_not_connect_timeout(tmp_path):
    def behavior(handler):
        time.sleep(0.30)
        _send_stream_headers(handler)
        _complete_sse(handler)

    with _local_stream_server(behavior) as base_url:
        client = LMStudioClient(
            base_url,
            connect_timeout=0.05,
            timeout=0.8,
            total_timeout=2.0,
            retries=0,
            progress_interval=60,
        )
        result = client.chat(
            model="qwen/qwen3.5-9b",
            prompt="return json",
            temperature=0.3,
            max_output_tokens=10,
            seed=7,
            attempt_directory=tmp_path,
        )
    assert result["content"] == "ok"


def test_real_http_header_wait_respects_post_connect_idle_limit(tmp_path):
    def behavior(handler):
        time.sleep(0.8)
        _send_stream_headers(handler)

    with _local_stream_server(behavior) as base_url:
        client = LMStudioClient(
            base_url,
            connect_timeout=0.05,
            timeout=0.20,
            total_timeout=2.0,
            retries=0,
            progress_interval=60,
        )
        started = time.monotonic()
        with pytest.raises(APIError) as captured:
            client.chat(
                model="qwen/qwen3.5-9b",
                prompt="return json",
                temperature=0.3,
                max_output_tokens=10,
                seed=7,
                attempt_directory=tmp_path,
            )
        elapsed = time.monotonic() - started
    assert captured.value.category == "response_header_timeout"
    assert elapsed < 0.70
    status = json.loads((tmp_path / "stream_status.json").read_text())
    assert status["status"] == "response_header_timeout"


def test_real_http_idle_timeout_aborts_blocked_reader_with_bounded_cleanup(tmp_path):
    def behavior(handler):
        _send_stream_headers(handler)
        time.sleep(1.0)

    with _local_stream_server(behavior) as base_url:
        client = LMStudioClient(
            base_url,
            connect_timeout=0.1,
            timeout=0.20,
            total_timeout=2.0,
            retries=0,
            progress_interval=60,
        )
        started = time.monotonic()
        with pytest.raises(APIError) as captured:
            client.chat(
                model="qwen/qwen3.5-9b",
                prompt="return json",
                temperature=0.3,
                max_output_tokens=10,
                seed=7,
                attempt_directory=tmp_path,
            )
        elapsed = time.monotonic() - started
    assert captured.value.category == "stream_idle_timeout"
    assert elapsed < 0.75
    status = json.loads((tmp_path / "stream_status.json").read_text())
    assert status["status"] == "stream_idle_timeout"
    assert status["reader_terminated"] is True


def test_real_http_total_timeout_preserves_partial_stream_and_stops_reader(tmp_path):
    def behavior(handler):
        _send_stream_headers(handler)
        for _ in range(30):
            _send_sse(
                handler,
                '{"choices":[{"delta":{"content":"x"}}]}',
            )
            time.sleep(0.04)

    with _local_stream_server(behavior) as base_url:
        client = LMStudioClient(
            base_url,
            connect_timeout=0.1,
            timeout=0.5,
            total_timeout=0.24,
            retries=0,
            progress_interval=60,
        )
        started = time.monotonic()
        with pytest.raises(APIError) as captured:
            client.chat(
                model="qwen/qwen3.5-9b",
                prompt="return json",
                temperature=0.3,
                max_output_tokens=10,
                seed=7,
                attempt_directory=tmp_path,
            )
        elapsed = time.monotonic() - started
    assert captured.value.category == "stream_total_timeout"
    assert elapsed < 0.80
    assert (tmp_path / "response_content.partial.txt").read_text()
    status = json.loads((tmp_path / "stream_status.json").read_text())
    assert status["content_characters"] > 0
    assert status["reader_terminated"] is True


def test_real_http_cancellation_aborts_blocked_reader(tmp_path):
    def behavior(handler):
        _send_stream_headers(handler)
        _send_sse(
            handler,
            '{"choices":[{"delta":{"reasoning_content":"partial-thought"}}]}',
        )
        time.sleep(1.0)

    with _local_stream_server(behavior) as base_url:
        request = urllib.request.Request(
            base_url + "/chat/completions",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        response = open_http_stream(
            request,
            connect_timeout=0.1,
            header_timeout=0.5,
            total_timeout=2.0,
        )
        cancelled = threading.Event()
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        started = time.monotonic()
        with pytest.raises(StreamTransportError) as captured:
            capture_chat_stream(
                response,
                directory=tmp_path,
                idle_timeout=0.8,
                total_timeout=2.0,
                progress_interval=60,
                cancel_event=cancelled,
            )
        elapsed = time.monotonic() - started
        timer.cancel()
    assert captured.value.category == "stream_cancelled"
    assert elapsed < 0.70
    assert (tmp_path / "reasoning.partial.txt").read_text() == "partial-thought"
    status = json.loads((tmp_path / "stream_status.json").read_text())
    assert status["status"] == "stream_cancelled"
    assert status["reader_terminated"] is True


def test_real_http_interrupted_stream_cannot_approve_candidate(tmp_path):
    fixed = _fixed_artifact(tmp_path)

    def behavior(handler):
        _send_stream_headers(handler)
        _send_sse(
            handler,
            '{"choices":[{"delta":{"content":"{\\"partial\\":true}"}}]}',
        )

    class LocalClient(LMStudioClient):
        def list_models(self):
            return MockClient([], model="qwen/qwen3.5-9b").list_models()

    with _local_stream_server(behavior) as base_url:
        client = LocalClient(
            base_url,
            connect_timeout=0.1,
            timeout=0.5,
            total_timeout=2.0,
            retries=0,
            progress_interval=60,
        )
        result = run_design(
            fixed_sample=fixed,
            model="qwen/qwen3.5-9b",
            client=client,
            max_attempts=1,
            output_dir=tmp_path / "interrupted-design",
            worker_timeout=10,
        )
    assert result["status"] == "failed_no_approved_candidate"
    assert result["metadata"]["attempt_history"][0]["status"] == "stream_incomplete"
    assert not (tmp_path / "interrupted-design" / "approved").exists()


def test_real_http_error_body_read_obeys_idle_timeout(tmp_path):
    def behavior(handler):
        handler.send_response(400)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.flush()
        time.sleep(1.0)

    with _local_stream_server(behavior) as base_url:
        client = LMStudioClient(
            base_url,
            connect_timeout=0.1,
            timeout=0.20,
            total_timeout=2.0,
            retries=0,
            progress_interval=60,
        )
        started = time.monotonic()
        with pytest.raises(APIError) as captured:
            client.chat(
                model="qwen/qwen3.5-9b",
                prompt="return json",
                temperature=0.3,
                max_output_tokens=10,
                seed=7,
                attempt_directory=tmp_path,
            )
        elapsed = time.monotonic() - started
    assert captured.value.category == "stream_idle_timeout"
    assert elapsed < 0.70
    status = json.loads((tmp_path / "stream_status.json").read_text())
    assert status["http_status"] == 400
    assert status["reader_terminated"] is True


def test_http_error_is_classified_and_request_is_saved(tmp_path):
    class ErrorResponse(ChunkedResponse):
        status = 400
        reason = "bad request"

        def abort(self):
            self.closed = True

    class HTTPErrorClient(LMStudioClient):
        def _open_stream(
            self, request, *, connect_timeout, header_timeout, total_timeout
        ):
            return ErrorResponse([b'{"error":"invalid"}'])

    client = HTTPErrorClient(timeout=1, total_timeout=2, retries=0)
    with pytest.raises(APIError) as captured:
        client.chat(
            model="qwen/qwen3.5-9b",
            prompt="return json",
            temperature=0.3,
            max_output_tokens=10,
            seed=7,
            attempt_directory=tmp_path,
        )
    assert captured.value.category == "http_error"
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["stream"] is True
    assert "response_format" not in request


def test_openai_http_error_is_classified_and_secret_is_redacted(
    tmp_path, monkeypatch
):
    secret = "fixture-sensitive-error-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    class ErrorResponse(ChunkedResponse):
        status = 401
        reason = "unauthorized"

        def abort(self):
            self.closed = True

    client = OpenAIStreamingClient([], timeout=1, total_timeout=2)
    client.response = ErrorResponse(
        [json.dumps({"error": {"message": f"bad Authorization: Bearer {secret}"}}).encode()]
    )
    with pytest.raises(APIError) as captured:
        client.chat(
            model="gpt-4o",
            prompt="return json",
            temperature=0.3,
            max_output_tokens=10,
            seed=7,
            attempt_directory=tmp_path,
        )
    assert captured.value.category == "authentication_error"
    assert secret not in str(captured.value)
    status = (tmp_path / "stream_status.json").read_text()
    assert secret not in status
    assert "[REDACTED]" in status


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
    with pytest.raises(CandidateError, match="one top-level function|disallowed"):
        validate_candidate(forbidden, constants)


def test_candidate_parser_accepts_only_plain_or_single_json_fence():
    candidate = _candidate()
    raw = json.dumps(candidate)
    plain, plain_meta = parse_candidate_json_envelope(f"  {raw}\n")
    fenced, fenced_meta = parse_candidate_json_envelope(f"```json\n{raw}\n```")
    unlabelled, unlabelled_meta = parse_candidate_json_envelope(f"```\n{raw}\n```")
    assert plain == fenced == unlabelled == candidate
    assert plain_meta["markdown_envelope_removed"] is False
    assert fenced_meta["markdown_envelope_removed"] is True
    assert unlabelled_meta["markdown_envelope_removed"] is True
    assert fenced["code"] == candidate["code"]
    assert fenced["features"][0]["reward_weight"] == candidate["features"][0][
        "reward_weight"
    ]


@pytest.mark.parametrize(
    "text, message",
    [
        ("before\n```json\n{}\n```", "outside"),
        ("```json\n{}\n```\n```json\n{}\n```", "multiple"),
        ("```json\n{}", "incomplete"),
        ("```python\n{}\n```", "unsupported"),
        ("```json\n{bad}\n```", "invalid JSON inside"),
    ],
)
def test_candidate_parser_rejects_ambiguous_or_invalid_fences(text, message):
    with pytest.raises(CandidateError, match=message):
        parse_candidate_json(text)


def test_candidate_worker_does_not_inherit_provider_credentials(
    design_fixture, monkeypatch
):
    _, arrays, _, _, constants = design_fixture
    monkeypatch.setenv("OPENAI_API_KEY", "worker-must-not-see-openai")
    monkeypatch.setenv("LM_STUDIO_API_TOKEN", "worker-must-not-see-lmstudio")
    original_run = llm_candidate.subprocess.run
    observed = {}

    def checked_run(*args, **kwargs):
        observed.update(kwargs.get("env") or {})
        return original_run(*args, **kwargs)

    monkeypatch.setattr(llm_candidate.subprocess, "run", checked_run)
    extra, reward, _ = execute_candidate_isolated(
        _candidate(), build_obs_arrays(arrays), constants, timeout=10
    )
    assert extra.shape[0] == arrays["state"].shape[0]
    assert reward.shape[0] == arrays["state"].shape[0]
    assert "OPENAI_API_KEY" not in observed
    assert "LM_STUDIO_API_TOKEN" not in observed


def test_staged_validation_collects_independent_static_errors(design_fixture):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "import os\n"
            "def compute_extra_state(bad, signature):\n"
            '    value = np.sin(obs["not_a_field"])\n'
            "    return np.asarray([value + open('x')], dtype=np.float32)\n"
        )
    )
    report = llm_design.validate_candidate_staged(candidate, constants)
    codes = {item["code"] for item in report["errors"]}
    assert "STATIC_TOP_LEVEL" in codes
    assert "STATIC_FUNCTION_SIGNATURE" in codes
    assert "STATIC_DISALLOWED_SYNTAX" in codes
    assert "STATIC_NUMPY_CALL" in codes
    assert "STATIC_FUNCTION_CALL" in codes
    assert "STATIC_FORBIDDEN_FIELD" in codes
    assert report["checks"]["execution"]["status"] == "not_run"


def test_staged_validation_collects_independent_schema_errors(design_fixture):
    _, _, _, _, constants = design_fixture
    candidate = _candidate()
    candidate["unexpected"] = True
    candidate["features"][0]["dtype"] = "float64"
    candidate["features"][0]["range"] = {"minimum": -2.0, "maximum": 2.0}
    candidate["features"][0]["reward_weight"] = float("inf")
    report = llm_design.validate_candidate_staged(candidate, constants)
    codes = {item["code"] for item in report["errors"]}
    assert {
        "SCHEMA_EXTRA_FIELD",
        "SCHEMA_DTYPE",
        "SCHEMA_FEATURE_RANGE",
        "SCHEMA_WEIGHT",
    }.issubset(codes)
    assert report["can_execute"] is False


def test_static_failures_are_not_executed_and_all_reach_revision_prompt(
    tmp_path, monkeypatch
):
    fixed = _fixed_artifact(tmp_path)
    invalid = _candidate(
        name="multi-static-error",
        code=(
            "import os\n"
            "def compute_extra_state(obs, constants):\n"
            '    value = np.sin(obs["missing"])\n'
            "    return np.asarray([value + open('x')], dtype=np.float32)\n"
        ),
    )
    calls = []
    original = llm_design.execute_candidate_isolated

    def guarded(candidate, *args, **kwargs):
        calls.append(candidate["candidate_name"])
        assert candidate["candidate_name"] != "multi-static-error"
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(llm_design, "execute_candidate_isolated", guarded)
    result = run_design(
        fixed_sample=fixed,
        model="qwen/qwen3.5-9b",
        client=MockClient([_response(invalid), _response(_candidate(name="fixed"))]),
        max_attempts=2,
        output_dir=tmp_path / "multi-static",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    assert calls == ["fixed"]
    report = json.loads(
        (tmp_path / "multi-static" / "attempt_01" / "validation_report.json").read_text()
    )
    assert len(report["errors"]) >= 3
    prompt = (tmp_path / "multi-static" / "attempt_02" / "prompt.txt").read_text()
    assert "multi-static-error" in prompt
    assert "STATIC_DISALLOWED_SYNTAX" in prompt
    assert "STATIC_FUNCTION_CALL" in prompt


def test_source_field_feedback_maps_reliable_features_and_not_unresolved_indices(
    design_fixture,
):
    _, _, _, _, constants = design_fixture
    mapped = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    mask = obs["movement_mask"]\n'
            '    values = obs["uav_backlog_bits"]\n'
            "    value = np.mean(np.where(mask, values, 0.0))\n"
            "    return np.asarray([value], dtype=np.float32)\n"
        )
    )
    report = llm_design.validate_candidate_staged(mapped, constants)
    mapped_errors = [
        item
        for item in report["errors"]
        if item["code"] == "STATIC_UNDECLARED_FEATURE_SOURCE"
    ]
    assert mapped_errors
    assert all(item["location"] == "$.features[0].source_fields" for item in mapped_errors)
    assert any("obs.movement_mask" in item["problem"] and "code:" in item["problem"] for item in mapped_errors)

    unresolved = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = np.clip(obs["state"][0] * obs["state"][0], 0.0, 1.0)\n'
            '    if np.any(obs["movement_mask"]):\n'
            "        value = value * 0.5\n"
            "    return np.asarray([value], dtype=np.float32)\n"
        )
    )
    report = llm_design.validate_candidate_staged(unresolved, constants)
    errors = [
        item
        for item in report["errors"]
        if item["code"] == "STATIC_UNDECLARED_FIELD_UNRESOLVED_FEATURE"
    ]
    assert errors
    assert "prevents reliable mapping" in errors[0]["problem"]
    assert "features[0]" not in errors[0]["location"]


def test_repeated_failed_candidate_is_diagnosed_and_cannot_be_approved(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    first = _candidate(
        name="failed-first",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    return np.asarray([obs["state"][0]], dtype=np.float32)\n'
        ),
    )
    repeated = json.loads(json.dumps(first))
    repeated["candidate_name"] = "renamed-only"
    repeated["code"] = "# same program\n" + repeated["code"]
    repeated_again = json.loads(json.dumps(first))
    repeated_again["candidate_name"] = "renamed-twice"
    repeated_again["code"] = "# another formatting-only change\n" + repeated_again["code"]
    client = MockClient(
        [
            _response(first),
            _response(repeated),
            _response(repeated_again),
            _response(_candidate(name="actually-fixed")),
        ]
    )
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=4,
        output_dir=tmp_path / "duplicate-revision",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    report = json.loads(
        (tmp_path / "duplicate-revision" / "attempt_02" / "validation_report.json").read_text()
    )
    assert [item["code"] for item in report["errors"]] == [
        "DUPLICATE_FAILED_CANDIDATE",
        "STATIC_EXPLICIT_FEATURE_REDUNDANCY",
    ]
    assert report["checks"]["duplicate_failed_candidate"]["matched_attempt"] == 1
    assert "previous_unresolved_feedback" not in json.dumps(report)
    next_report = json.loads(
        (tmp_path / "duplicate-revision" / "attempt_03" / "validation_report.json").read_text()
    )
    assert [item["code"] for item in next_report["errors"]] == [
        "DUPLICATE_FAILED_CANDIDATE",
        "STATIC_EXPLICIT_FEATURE_REDUNDANCY",
    ]
    assert next_report["checks"]["duplicate_failed_candidate"]["matched_attempt"] == 1
    assert "previous_unresolved_feedback" not in json.dumps(next_report)
    feedback = llm_design._feedback_from_validation(
        report, "duplicate_failed_candidate"
    )
    variants = dict(llm_design._feedback_prompt_variants(feedback))
    for name in ("full", "compact_all_roots"):
        codes = [item["code"] for item in variants[name]["confirmed_errors"]]
        assert "DUPLICATE_FAILED_CANDIDATE" in codes
        assert "STATIC_EXPLICIT_FEATURE_REDUNDANCY" in codes
    duplicate_summary = variants["compact_all_roots"]["prompt_feedback_summary"]
    assert duplicate_summary["original_error_count"] == 2
    assert duplicate_summary["included_error_count"] == 2
    assert duplicate_summary["omitted_error_count"] == 0
    selected, omitted = llm_design._select_feedback_representatives(
        feedback["confirmed_errors"], 1
    )
    assert omitted == 0
    assert {item["code"] for item in selected} == {
        "DUPLICATE_FAILED_CANDIDATE",
        "STATIC_EXPLICIT_FEATURE_REDUNDANCY",
    }
    prompt = (tmp_path / "duplicate-revision" / "attempt_04" / "prompt.txt").read_text()
    assert "semantically unchanged from failed attempt 1" in prompt
    assert "STATIC_EXPLICIT_FEATURE_REDUNDANCY" in prompt


def test_structured_missing_source_roots_remain_independently_actionable(
    design_fixture,
):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = np.mean(obs["uav_hol_type"]) + np.mean(obs["uav_hol_valid"])\n'
            "    return np.asarray([np.clip(value, 0.0, 1.0)], dtype=np.float32)\n"
        )
    )
    report = llm_design.validate_candidate_staged(candidate, constants)
    errors = [
        item
        for item in report["errors"]
        if item["code"] == "STATIC_UNDECLARED_FEATURE_SOURCE"
    ]
    assert {(item["feature_index"], item["source_field"]) for item in errors} == {
        (0, "obs.uav_hol_type"),
        (0, "obs.uav_hol_valid"),
    }
    feedback = llm_design._feedback_from_validation(report, "static_failure")
    compact = dict(llm_design._feedback_prompt_variants(feedback))["compact_all_roots"]
    compact_errors = [
        item
        for item in compact["confirmed_errors"]
        if item["code"] == "STATIC_UNDECLARED_FEATURE_SOURCE"
    ]
    assert {(item["feature_index"], item["source_field"]) for item in compact_errors} == {
        (0, "obs.uav_hol_type"),
        (0, "obs.uav_hol_valid"),
    }
    assert all("Add 'obs.uav_hol_" in item["requirement"] for item in compact_errors)


def test_same_missing_source_field_on_two_features_is_not_merged(design_fixture):
    _, _, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    flags = obs["uav_hol_valid"]\n'
            "    return np.asarray([np.mean(flags), np.max(flags)], dtype=np.float32)\n"
        )
    )
    second = dict(candidate["features"][0])
    second.update(
        {
            "index": 1,
            "name": "second_hol_summary",
            "description": "A second distinct HOL validity summary.",
            "formula": "maximum of current HOL validity flags",
        }
    )
    candidate["features"] = [candidate["features"][0], second]
    report = llm_design.validate_candidate_staged(candidate, constants)
    errors = [
        item
        for item in report["errors"]
        if item["code"] == "STATIC_UNDECLARED_FEATURE_SOURCE"
        and item.get("source_field") == "obs.uav_hol_valid"
    ]
    assert {item["feature_index"] for item in errors} == {0, 1}
    compact = dict(
        llm_design._feedback_prompt_variants(
            llm_design._feedback_from_validation(report, "static_failure")
        )
    )["compact_all_roots"]
    matching = [
        item
        for item in compact["confirmed_errors"]
        if item.get("source_field") == "obs.uav_hol_valid"
    ]
    assert {item["feature_index"] for item in matching} == {0, 1}


def test_repeated_structured_root_merges_occurrences_and_locations():
    issues = [
        {
            "code": "STATIC_UNDECLARED_FEATURE_SOURCE",
            "stage": "static",
            "location": "$.features[0].source_fields",
            "feature_index": 0,
            "source_field": "obs.uav_hol_valid",
            "feature_mapping": "resolved",
            "source_locations": ["code:2:10"],
            "problem": "missing declaration",
            "requirement": "declare obs.uav_hol_valid",
        },
        {
            "code": "STATIC_UNDECLARED_FEATURE_SOURCE",
            "stage": "static",
            "location": "$.features[0].source_fields",
            "feature_index": 0,
            "source_field": "obs.uav_hol_valid",
            "feature_mapping": "resolved",
            "source_locations": ["code:8:12"],
            "problem": "missing declaration again",
            "requirement": "declare obs.uav_hol_valid",
        },
    ]
    selected, omitted = llm_design._select_feedback_representatives(issues, None)
    assert omitted == 0
    assert len(selected) == 1
    assert selected[0]["occurrence_count"] == 2
    assert selected[0]["representative_locations"] == ["code:2:10", "code:8:12"]
    feedback = {
        "category": "static_failure",
        "confirmed_errors": issues,
        "confirmed_error_count": 2,
    }
    summary = dict(llm_design._feedback_prompt_variants(feedback))[
        "compact_all_roots"
    ]["prompt_feedback_summary"]
    assert summary["original_error_count"] == 1
    assert summary["included_error_count"] == 1
    assert summary["omitted_error_count"] == 0
    assert summary["original_error_occurrence_count"] == 2


@pytest.mark.parametrize("model", ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])
def test_over_budget_validation_feedback_is_summarized_before_next_request(
    tmp_path, monkeypatch, model
):
    fixed = _fixed_artifact(tmp_path)
    invalid = _candidate(name="many-independent-errors")
    errors = []
    for index in range(120):
        code = ("STATIC_FIELD", "STATIC_CALL", "SCHEMA_RANGE")[index % 3]
        errors.append(
            {
                "code": code,
                "stage": "static" if code.startswith("STATIC") else "schema",
                "location": f"candidate line {index + 1}",
                "problem": f"root problem {index}: " + ("detail " * 180),
                "requirement": f"requirement for {code}: " + ("rule " * 120),
            }
        )
    original_validate = llm_design.validate_candidate_staged

    def staged(candidate, constants):
        if candidate.get("candidate_name") == "many-independent-errors":
            return {
                "status": "failed",
                "can_execute": False,
                "errors": errors,
                "checks": {"static": {"status": "failed", "completed": True}},
            }
        return original_validate(candidate, constants)

    monkeypatch.setattr(llm_design, "validate_candidate_staged", staged)
    client = MockClient(
        [
            _response(invalid, model=model),
            _response(
                _candidate(
                    name="corrected",
                    code=(
                        "def compute_extra_state(obs, constants):\n"
                        '    value = np.clip(obs["state"][0] ** 2, 0.0, 1.0)\n'
                        "    return np.asarray([value], dtype=np.float32)\n"
                    ),
                ),
                model=model,
            ),
        ],
        model=model,
    )
    result = run_design(
        fixed_sample=fixed,
        model=model,
        client=client,
        max_attempts=2,
        context_length=40_000,
        output_dir=tmp_path / "feedback-summary",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    assert len(client.calls) == 2
    full_report = json.loads(
        (tmp_path / "feedback-summary" / "attempt_01" / "validation_report.json").read_text()
    )
    assert len(full_report["errors"]) == 120
    prompt_info = json.loads(
        (tmp_path / "feedback-summary" / "attempt_02" / "prompt_feedback.json").read_text()
    )
    summary = prompt_info["feedback"]["prompt_feedback_summary"]
    assert prompt_info["strategy"] != "full"
    assert summary["original_error_count"] == 120
    assert summary["included_error_count"] < 120
    assert summary["omitted_error_count"] > 0
    prompt = (tmp_path / "feedback-summary" / "attempt_02" / "prompt.txt").read_text()
    assert '"candidate_name": "many-independent-errors"' in prompt
    assert all(code in prompt for code in ("STATIC_FIELD", "STATIC_CALL", "SCHEMA_RANGE"))
    assert "omitted_error_count" in prompt
    assert result["metadata"]["context"]["effective_budget"] == 20_000
    assert result["metadata"]["generation"]["max_output_tokens"] == 4096


@pytest.mark.parametrize("model", ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])
def test_budgeted_revision_prompt_keeps_each_structured_source_fix(
    tmp_path, monkeypatch, model
):
    fixed = _fixed_artifact(tmp_path)
    invalid = _candidate(
        name="missing-two-source-fields",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = np.clip(obs["state"][0] ** 3, 0.0, 1.0)\n'
            "    return np.asarray([value], dtype=np.float32)\n"
        ),
    )
    long_detail = " repeated diagnostic detail" * 3000
    errors = [
        {
            "code": "STATIC_UNDECLARED_FEATURE_SOURCE",
            "stage": "static",
            "location": "$.features[0].source_fields",
            "feature_index": 0,
            "source_field": field,
            "feature_mapping": "resolved",
            "source_locations": [f"code:{line}:12"],
            "problem": f"feature[0] uses {field} but does not declare it.{long_detail}",
            "requirement": f"Add {field} to features[0].source_fields.{long_detail}",
        }
        for field, line in (("obs.uav_hol_type", 2), ("obs.uav_hol_valid", 3))
    ]
    original_validate = llm_design.validate_candidate_staged

    def staged(candidate, constants):
        if candidate.get("candidate_name") == "missing-two-source-fields":
            return {
                "status": "failed",
                "can_execute": False,
                "errors": errors,
                "checks": {"static": {"status": "failed", "completed": True}},
            }
        return original_validate(candidate, constants)

    monkeypatch.setattr(llm_design, "validate_candidate_staged", staged)
    client = MockClient(
        [_response(invalid, model=model), _response(_candidate(name="fixed"), model=model)],
        model=model,
    )
    output = tmp_path / f"structured-feedback-{model.split('/')[0]}"
    result = run_design(
        fixed_sample=fixed,
        model=model,
        client=client,
        max_attempts=2,
        context_length=40_000,
        output_dir=output,
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    assert len(client.calls) == 2
    prompt = client.calls[1]["prompt"]
    assert '"candidate_name": "missing-two-source-fields"' in prompt
    assert "obs.uav_hol_type" in prompt
    assert "obs.uav_hol_valid" in prompt
    assert "features[0].source_fields" in prompt
    prompt_info = json.loads((output / "attempt_02" / "prompt_feedback.json").read_text())
    summary = prompt_info["feedback"]["prompt_feedback_summary"]
    assert prompt_info["strategy"] == "compact_all_roots"
    assert summary["original_error_count"] == 2
    assert summary["included_error_count"] == 2
    assert summary["omitted_error_count"] == 0
    assert summary["original_error_occurrence_count"] == 2
    saved_report = json.loads((output / "attempt_01" / "validation_report.json").read_text())
    assert saved_report["errors"] == errors


def test_complete_candidate_is_not_truncated_when_minimum_feedback_cannot_fit(
    tmp_path, monkeypatch
):
    fixed = _fixed_artifact(tmp_path)
    marker = "COMPLETE-CANDIDATE-END-MARKER"
    huge = _candidate(
        name="candidate-too-large-for-revision",
        code=(
            "# " + ("x" * 60_000) + f" {marker}\n"
            "def compute_extra_state(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        ),
    )
    original_validate = llm_design.validate_candidate_staged

    def staged(candidate, constants):
        if candidate.get("candidate_name") == "candidate-too-large-for-revision":
            return {
                "status": "failed",
                "can_execute": False,
                "errors": [
                    {
                        "code": "STATIC_TEST_FAILURE",
                        "stage": "static",
                        "location": "candidate line 1",
                        "problem": "fixture failure",
                        "requirement": "fix the candidate",
                    }
                ],
                "checks": {"static": {"status": "failed", "completed": True}},
            }
        return original_validate(candidate, constants)

    monkeypatch.setattr(llm_design, "validate_candidate_staged", staged)
    client = MockClient([_response(huge)])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=2,
        context_length=20_000,
        output_dir=tmp_path / "candidate-overflow",
        worker_timeout=10,
    )
    assert result["status"] == "failed_no_approved_candidate"
    assert len(client.calls) == 1
    assert result["metadata"]["attempt_history"][-1]["status"] == "context_budget_exceeded"
    prompt = (tmp_path / "candidate-overflow" / "attempt_02" / "prompt.txt").read_text()
    assert marker in prompt
    assert "[TRUNCATED:" not in prompt
    assert result["metadata"]["final_feedback"]["category"] == "context_budget_exceeded"


def test_runtime_errors_are_deduplicated_with_representative_samples(design_fixture):
    _, arrays, _, _, constants = design_fixture
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = float(obs["state"][0])\n'
            "    if value < 0.15:\n"
            "        return np.asarray([[0.0]], dtype=np.float32)\n"
            "    return np.asarray([np.log(-1.0)], dtype=np.float32)\n"
        )
    )
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            candidate, build_obs_arrays(arrays), constants, timeout=10
        )
    report = captured.value.report
    codes = {item["code"] for item in report["errors"]}
    assert "RUNTIME_SHAPE" in codes
    assert "RUNTIME_NONFINITE" in codes
    assert all(item["occurrence_count"] >= 1 for item in report["errors"])
    assert all(
        len(item["representative_samples"]) <= 3 for item in report["errors"]
    )


def test_runtime_grouping_keeps_distinct_exception_types_and_candidate_lines(
    design_fixture,
):
    _, arrays, _, _, constants = design_fixture
    candidate = _candidate(
        name="distinct-runtime-roots",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = float(obs["state"][0])\n'
            "    if value < 0.15:\n"
            '        output = obs["state"][9999]\n'
            "    else:\n"
            "        output = 1.0 / (value - value)\n"
            "    return np.asarray([output], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        ),
    )
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            candidate, build_obs_arrays(arrays), constants, timeout=10
        )
    runtime = [
        item
        for item in captured.value.report["errors"]
        if item["code"] == "RUNTIME_FUNCTION_ERROR"
    ]
    assert {item["exception_type"] for item in runtime} == {
        "IndexError",
        "ZeroDivisionError",
    }
    assert {item["candidate_function"] for item in runtime} == {
        "compute_extra_state"
    }
    assert len({item["candidate_line"] for item in runtime}) == 2
    assert all(item["occurrence_count"] >= 2 for item in runtime)
    assert all(len(item["representative_samples"]) <= 3 for item in runtime)


def test_same_exception_type_at_different_candidate_lines_is_not_merged(
    design_fixture,
):
    _, arrays, _, _, constants = design_fixture
    candidate = _candidate(
        name="same-type-different-lines",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = float(obs["state"][0])\n'
            "    if value < 0.15:\n"
            "        output = 1.0 / (value - value)\n"
            "    else:\n"
            "        output = 2.0 / (value - value)\n"
            "    return np.asarray([output], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        ),
    )
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            candidate, build_obs_arrays(arrays), constants, timeout=10
        )
    runtime = [
        item
        for item in captured.value.report["errors"]
        if item.get("exception_type") == "ZeroDivisionError"
    ]
    assert len(runtime) == 2
    assert len({item["candidate_line"] for item in runtime}) == 2


def test_distinct_runtime_roots_are_visible_in_revision_prompt(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    invalid = _candidate(
        name="runtime-root-feedback",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = float(obs["state"][0])\n'
            "    if value < 0.15:\n"
            '        output = obs["state"][9999]\n'
            "    else:\n"
            "        output = 1.0 / (value - value)\n"
            "    return np.asarray([output], dtype=np.float32)\n\n"
            "def compute_reward_terms(obs, constants):\n"
            "    return np.asarray([0.0], dtype=np.float32)\n"
        ),
    )
    client = MockClient([_response(invalid), _response(_candidate(name="fixed"))])
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=2,
        output_dir=tmp_path / "runtime-feedback",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    prompt = (tmp_path / "runtime-feedback" / "attempt_02" / "prompt.txt").read_text()
    assert "IndexError" in prompt
    assert "ZeroDivisionError" in prompt
    assert "candidate_line" in prompt


def _uav_slice_runtime_candidate(name="uav-slice-runtime"):
    candidate = _candidate(
        name=name,
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    state = obs["state"]\n'
            '    movement_mask = obs["movement_mask"]\n'
            "    uav_energy = state[7::17]\n"
            "    uav_queue_load = state[8::17]\n"
            "    average_uav_energy = np.clip(\n"
            "        np.mean(uav_energy[movement_mask]), 0.0, 1.0\n"
            "    )\n"
            "    return np.asarray([average_uav_energy], dtype=np.float32)\n"
        ),
    )
    candidate["features"][0]["source_fields"] = ["obs.state", "obs.movement_mask"]
    candidate["features"][0]["formula"] = (
        "mean clipped UAV energy over the current movement mask"
    )
    return candidate


def test_runtime_feedback_includes_source_shapes_and_authoritative_uav_layout(
    design_fixture,
):
    _, arrays, metadata, _, constants = design_fixture
    contract = runtime_diagnostic_contract(metadata, constants)
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            _uav_slice_runtime_candidate(),
            build_obs_arrays(arrays),
            constants,
            timeout=10,
            diagnostic_contract=contract,
        )
    issue = next(
        item
        for item in captured.value.report["errors"]
        if item.get("exception_type") == "IndexError"
    )
    diagnostic = issue["runtime_diagnostics"]
    assert issue["candidate_line"] == 7
    assert diagnostic["local_array_summaries"]["uav_energy"] == {
        "type": "ndarray",
        "shape": [31],
        "dtype": "float32",
    }
    assert diagnostic["local_array_summaries"]["movement_mask"]["shape"] == [16]
    assert diagnostic["related_interface"]["obs.state"]["uav_block"] == {
        "start": 0,
        "stop_exclusive": 272,
        "num_uav": 16,
        "features_per_uav": 17,
        "layout_source": "saved source_checkpoint_contract.movement_state_feature_schema",
    }
    excerpt = diagnostic["candidate_source_excerpt"]
    assert {item["line"] for item in excerpt} >= {4, 7}
    assert any("state[7::17]" in item["code"] for item in excerpt)
    assert "half-open UAV block [0:272)" in issue["requirement"]
    assert "length-16 movement_mask" in issue["requirement"]
    assert diagnostic["index_source_analysis"]["status"] == "confirmed_state_stride"
    assert issue["operation_fingerprint"] == diagnostic["traceback_operation"][
        "fingerprint"
    ]


def _local_mask_mismatch_candidate(*, comment="", name="local-mask-mismatch"):
    candidate = _candidate(
        name=name,
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    state = obs["state"]\n'
            "    unused = state[7::17]\n"
            '    movement_mask = obs["movement_mask"]\n'
            "    short = np.zeros(3)\n"
            f"{comment}"
            "    value = short[movement_mask]\n"
            "    return np.asarray([np.sum(value)], dtype=np.float32)\n"
        ),
    )
    candidate["features"][0]["source_fields"] = ["obs.state", "obs.movement_mask"]
    candidate["features"][0]["formula"] = "sum of a local masked test vector"
    return candidate


def test_unrelated_open_state_slice_does_not_receive_uav_slice_fix(design_fixture):
    _, arrays, metadata, _, constants = design_fixture
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            _local_mask_mismatch_candidate(),
            build_obs_arrays(arrays),
            constants,
            timeout=10,
            diagnostic_contract=runtime_diagnostic_contract(metadata, constants),
        )
    issue = captured.value.report["errors"][0]
    diagnostic = issue["runtime_diagnostics"]
    assert issue["exception_type"] == "IndexError"
    assert "dimension is 3" in issue["problem"]
    assert diagnostic["index_source_analysis"]["status"] == "unresolved"
    assert "not reliably linked" in diagnostic["index_source_analysis"]["reason"]
    assert set(diagnostic["related_interface"]) == {"obs.movement_mask"}
    assert "half-open UAV block" not in issue["requirement"]
    assert "targeted_requirement" not in diagnostic


def test_reassigned_state_alias_is_not_treated_as_obs_state_slice(design_fixture):
    _, arrays, metadata, _, constants = design_fixture
    candidate = _candidate(
        name="reassigned-state-alias",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    state = obs["state"]\n'
            "    unused = state[7::17]\n"
            "    state = np.zeros(3)\n"
            '    movement_mask = obs["movement_mask"]\n'
            "    value = state[movement_mask]\n"
            "    return np.asarray([np.sum(value)], dtype=np.float32)\n"
        ),
    )
    candidate["features"][0]["source_fields"] = ["obs.state", "obs.movement_mask"]
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            candidate,
            build_obs_arrays(arrays),
            constants,
            timeout=10,
            diagnostic_contract=runtime_diagnostic_contract(metadata, constants),
        )
    issue = captured.value.report["errors"][0]
    assert issue["runtime_diagnostics"]["index_source_analysis"]["status"] == "unresolved"
    assert "half-open UAV block" not in issue["requirement"]


def test_runtime_operation_fingerprint_ignores_comment_line_shift(design_fixture):
    _, arrays, metadata, _, constants = design_fixture
    reports = []
    for comment in ("", "    # inserted comment and blank line\n\n"):
        with pytest.raises(CandidateExecutionError) as captured:
            execute_candidate_isolated(
                _local_mask_mismatch_candidate(comment=comment),
                build_obs_arrays(arrays),
                constants,
                timeout=10,
                diagnostic_contract=runtime_diagnostic_contract(metadata, constants),
            )
        reports.append(captured.value.report)
    first, second = (report["errors"][0] for report in reports)
    assert first["candidate_line"] != second["candidate_line"]
    assert first["operation_fingerprint"] == second["operation_fingerprint"]
    tracker = llm_design._IssueTracker()
    tracker.apply(
        reports[0], attempt=1, candidate_identity={"semantic_fingerprint": "first"}
    )
    tracker.apply(
        reports[1], attempt=2, candidate_identity={"semantic_fingerprint": "second"}
    )
    records = tracker.snapshot()["records"]
    assert len(records) == 1
    assert records[0]["latest_location"].endswith(
        f"candidate line {second['candidate_line']}"
    )
    assert [item["location"] for item in records[0]["history"][:2]] == [
        first["location"],
        second["location"],
    ]


def test_tracker_and_feedback_keep_same_signature_at_distinct_operations():
    base = {
        "code": "RUNTIME_FUNCTION_ERROR",
        "stage": "execution",
        "exception_type": "IndexError",
        "candidate_function": "compute_extra_state",
        "problem_signature": "same normalized message",
        "problem": "IndexError: same normalized message",
        "requirement": "fix this operation",
        "occurrence_count": 4,
    }
    issues = [
        {
            **base,
            "location": "compute_extra_state at candidate line 10",
            "candidate_line": 10,
            "operation_fingerprint": "operation-a",
        },
        {
            **base,
            "location": "compute_extra_state at candidate line 20",
            "candidate_line": 20,
            "operation_fingerprint": "operation-b",
        },
    ]
    tracker = llm_design._IssueTracker()
    tracker.apply(
        {
            "errors": issues,
            "checks": {"execution": {"status": "failed", "completed": True}},
        },
        attempt=1,
        candidate_identity={"semantic_fingerprint": "candidate-a"},
    )
    carried = tracker.apply(
        llm_design._json_failure_report(CandidateError("invalid JSON")),
        attempt=2,
        candidate_identity={"kind": "unparsed_response", "raw": "candidate-b"},
    )
    assert len(tracker.snapshot()["records"]) == 3  # two runtime + current JSON
    historical = [item for item in carried if item["stage"] == "execution"]
    assert len(historical) == 2
    assert {item["source_location"] for item in historical} == {
        "compute_extra_state at candidate line 10",
        "compute_extra_state at candidate line 20",
    }
    feedback = llm_design._feedback_from_validation(
        llm_design._json_failure_report(CandidateError("invalid JSON")),
        "candidate_json_failure",
        historical_unverified=historical,
    )
    compact = dict(llm_design._feedback_prompt_variants(feedback))[
        "compact_all_roots"
    ]
    assert len(compact["historical_unverified_errors"]) == 2


def test_two_runtime_locations_survive_json_failure_prompt_and_duplicate_check(
    tmp_path,
):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(
        name="two-runtime-locations",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = float(obs["state"][0])\n'
            '    movement_mask = obs["movement_mask"]\n'
            "    first = np.zeros(3)\n"
            "    second = np.zeros(3)\n"
            "    if value < 0.15:\n"
            "        selected = first[movement_mask]\n"
            "    else:\n"
            "        selected = second[movement_mask]\n"
            "    return np.asarray([np.sum(selected)], dtype=np.float32)\n"
        ),
    )
    candidate["features"][0]["source_fields"] = ["obs.state", "obs.movement_mask"]
    raw_claim = "The runtime error is fixed.\n" + json.dumps(candidate)
    client = MockClient(
        [
            _response(candidate),
            {**_response(candidate), "content": raw_claim},
            _response(candidate),
        ]
    )
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=3,
        output_dir=tmp_path / "two-runtime-history",
        worker_timeout=10,
    )
    assert result["status"] == "failed_no_approved_candidate"
    first_report = json.loads(
        (tmp_path / "two-runtime-history" / "attempt_01" / "validation_report.json").read_text()
    )
    runtime = [item for item in first_report["errors"] if item["stage"] == "execution"]
    assert len(runtime) == 2
    assert len({item["operation_fingerprint"] for item in runtime}) == 2
    assert all(item["occurrence_count"] >= 2 for item in runtime)

    third_prompt = client.calls[2]["prompt"]
    assert "Previous failed raw final content" in third_prompt
    assert raw_claim in third_prompt or "[TRUNCATED:" in third_prompt
    assert "JSON_PARSE_ERROR" in third_prompt
    assert third_prompt.count("previously_found_not_revalidated") >= 2
    assert "candidate line 7" in third_prompt
    assert "candidate line 9" in third_prompt
    duplicate = json.loads(
        (tmp_path / "two-runtime-history" / "attempt_03" / "validation_report.json").read_text()
    )
    assert [item["code"] for item in duplicate["errors"]] == [
        "DUPLICATE_FAILED_CANDIDATE",
        "RUNTIME_FUNCTION_ERROR",
        "RUNTIME_FUNCTION_ERROR",
    ]
    compact = dict(
        llm_design._feedback_prompt_variants(
            llm_design._feedback_from_validation(
                duplicate, "duplicate_failed_candidate"
            )
        )
    )["compact_all_roots"]
    assert len(
        [
            item
            for item in compact["confirmed_errors"]
            if item["code"] == "RUNTIME_FUNCTION_ERROR"
        ]
    ) == 2


def test_diagnostic_failure_never_replaces_original_runtime_error(monkeypatch):
    def broken(_obs, _constants):
        raise IndexError("original candidate failure")

    def fail_diagnostics(*_args, **_kwargs):
        raise RuntimeError("diagnostic implementation failed")

    monkeypatch.setattr(llm_candidate_worker, "_runtime_diagnostics", fail_diagnostics)
    issues = llm_candidate_worker._IssueAccumulator()
    result = llm_candidate_worker._check_function(
        broken,
        {"state": np.zeros(1, dtype=np.float32)},
        {},
        1,
        "compute_extra_state",
        {"fixed_sample_index": 0},
        issues,
        candidate_source="def compute_extra_state(obs, constants):\n    pass\n",
        diagnostic_contract={},
    )
    assert result is None
    assert len(issues.errors) == 1
    assert issues.errors[0]["problem"] == "IndexError: original candidate failure"
    diagnostic = issues.errors[0]["runtime_diagnostics"]
    assert diagnostic["diagnostic_status"] == "failed_without_affecting_runtime_error"
    assert diagnostic["diagnostic_error_type"] == "RuntimeError"


def test_unrelated_index_error_does_not_get_uav_slice_advice(design_fixture):
    _, arrays, metadata, _, constants = design_fixture
    unrelated = _candidate(
        name="unrelated-index",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    value = obs["state"][9999]\n'
            "    return np.asarray([value * value], dtype=np.float32)\n"
        ),
    )
    with pytest.raises(CandidateExecutionError) as captured:
        execute_candidate_isolated(
            unrelated,
            build_obs_arrays(arrays),
            constants,
            timeout=10,
            diagnostic_contract=runtime_diagnostic_contract(metadata, constants),
        )
    issue = captured.value.report["errors"][0]
    assert issue["exception_type"] == "IndexError"
    assert "half-open UAV block" not in issue["requirement"]
    assert "targeted_requirement" not in issue["runtime_diagnostics"]


@pytest.mark.parametrize(
    "provider,model",
    [
        ("lmstudio", "qwen/qwen3.5-9b"),
        ("lmstudio", "google/gemma-4-e4b"),
        ("openai", "gpt-4o"),
    ],
)
def test_parse_failure_carries_prior_runtime_issue_into_actual_next_request(
    tmp_path, provider, model
):
    fixed = _fixed_artifact(tmp_path)
    faulty = _uav_slice_runtime_candidate()
    raw_claim = "I fixed the indexing issue.\n" + json.dumps(faulty)
    responses = [
        _response(faulty, model=model),
        {**_response(faulty, model=model), "content": raw_claim},
        _response(faulty, model=model),
    ]
    client = MockClient(responses, model=model)
    result = run_design(
        fixed_sample=fixed,
        provider=provider,
        model=model,
        client=client,
        max_attempts=3,
        output_dir=tmp_path / f"history-{provider}-{model.split('/')[-1]}",
        worker_timeout=10,
    )
    assert result["status"] == "failed_no_approved_candidate"
    assert len(client.calls) == 3
    third_prompt = client.calls[2]["prompt"]
    assert raw_claim in third_prompt
    assert "JSON_PARSE_ERROR" in third_prompt
    assert "previously_found_not_revalidated" in third_prompt
    assert "IndexError" in third_prompt
    assert "state[7::17]" in third_prompt
    assert "half-open UAV block [0:272)" in third_prompt
    incoming = json.loads(
        (
            tmp_path
            / f"history-{provider}-{model.split('/')[-1]}"
            / "attempt_03"
            / "prompt_feedback.json"
        ).read_text()
    )
    assert incoming["feedback"]["confirmed_errors"][0]["code"] == "JSON_PARSE_ERROR"
    assert incoming["feedback"]["historical_unverified_errors"][0][
        "source_attempt"
    ] == 1
    duplicate_report = json.loads(
        (
            tmp_path
            / f"history-{provider}-{model.split('/')[-1]}"
            / "attempt_03"
            / "validation_report.json"
        ).read_text()
    )
    assert [item["code"] for item in duplicate_report["errors"]][:2] == [
        "DUPLICATE_FAILED_CANDIDATE",
        "RUNTIME_FUNCTION_ERROR",
    ]
    assert "half-open UAV block" in duplicate_report["errors"][1]["requirement"]


def test_runtime_issue_resolves_only_after_execution_passes_and_history_remains(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    faulty = _uav_slice_runtime_candidate()
    raw_claim = "Fixed.\n" + json.dumps(faulty)
    client = MockClient(
        [
            _response(faulty),
            {**_response(faulty), "content": raw_claim},
            _response(_candidate(name="runtime-fixed")),
        ]
    )
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=3,
        output_dir=tmp_path / "runtime-resolved",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    tracker = json.loads((tmp_path / "runtime-resolved" / "issue_tracker.json").read_text())
    runtime = next(
        item for item in tracker["records"] if item["stage"] == "execution"
    )
    assert runtime["status"] == "resolved"
    assert runtime["resolved_attempt"] == 3
    assert runtime["resolution_evidence"]["passed_checks"] == [
        "execution",
        "json",
        "schema",
        "static",
    ]
    assert any(
        item["status"] == "previously_found_not_revalidated"
        for item in runtime["history"]
    )


def test_compact_feedback_keeps_current_and_historical_issue_provenance():
    feedback = {
        "category": "candidate_json_failure",
        "confirmed_errors": [
            {
                "code": "JSON_PARSE_ERROR",
                "stage": "json",
                "location": "line 1, column 1",
                "problem": "explanatory text outside JSON",
                "requirement": "Return only the complete JSON object.",
            }
        ],
        "confirmed_error_count": 1,
        "historical_unverified_errors": [
            {
                "code": "RUNTIME_FUNCTION_ERROR",
                "stage": "execution",
                "location": "compute_extra_state at candidate line 7",
                "exception_type": "IndexError",
                "candidate_function": "compute_extra_state",
                "candidate_line": 7,
                "problem_signature": "boolean index dimension mismatch",
                "problem": "boolean index length mismatch",
                "requirement": "Limit the state slice to the UAV block.",
                "issue_status": "previously_found_not_revalidated",
                "source_attempt": 1,
                "source_candidate": {"semantic_fingerprint": "abc"},
                "source_location": "compute_extra_state at candidate line 7",
                "status_note": "The execution check did not run in attempt 2.",
            }
        ],
        "historical_unverified_error_count": 1,
    }
    variants = dict(llm_design._feedback_prompt_variants(feedback))
    smallest = variants["compact_all_roots"]
    assert [item["code"] for item in smallest["confirmed_errors"]] == [
        "JSON_PARSE_ERROR"
    ]
    historical = smallest["historical_unverified_errors"]
    assert len(historical) == 1
    assert historical[0]["issue_status"] == "previously_found_not_revalidated"
    assert historical[0]["source_attempt"] == 1
    summary = smallest["prompt_feedback_summary"]
    assert summary["historical_unverified_original_count"] == 1
    assert summary["historical_unverified_included_count"] == 1


def test_earlier_new_runtime_failure_does_not_resolve_old_runtime_issue(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    earlier = _candidate(
        name="earlier-runtime-error",
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    output = obs["state"][9999]\n'
            "    return np.asarray([output * output], dtype=np.float32)\n"
        ),
    )
    client = MockClient(
        [
            _response(_uav_slice_runtime_candidate()),
            _response(earlier),
            _response(_candidate(name="fixed-after-two-runtime-errors")),
        ]
    )
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=3,
        output_dir=tmp_path / "earlier-runtime",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    third_prompt = client.calls[2]["prompt"]
    assert "index 9999 is out of bounds" in third_prompt
    assert "previously_found_not_revalidated" in third_prompt
    assert "state[7::17]" in third_prompt


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
    extra, reward, _ = execute_candidate_isolated(
        coincident, build_obs_arrays(arrays), constants, timeout=10
    )
    diagnostics = candidate_numeric_diagnostics(arrays["state"], extra, coincident)
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


def test_empty_probe_uses_full_output_range_validation(design_fixture):
    _, arrays, _, _, constants = design_fixture
    obs_arrays = {
        name: np.asarray(value).copy()
        for name, value in build_obs_arrays(arrays).items()
    }
    obs_arrays["state"][:, 0] = 0.5
    candidate = _candidate(
        code=(
            "def compute_extra_state(obs, constants):\n"
            '    bad = 2.0 if np.count_nonzero(obs["state"]) == 0 else 0.0\n'
            "    return np.asarray([bad], dtype=np.float32)\n"
        )
    )
    with pytest.raises(
        CandidateExecutionError,
        match=r"compute_extra_state\(empty probe\).*feature\[0\]",
    ):
        execute_candidate_isolated(candidate, obs_arrays, constants, timeout=10)


def test_legal_empty_probe_still_passes(design_fixture):
    _, arrays, _, _, constants = design_fixture
    extra, reward, report = execute_candidate_isolated(
        _candidate(), build_obs_arrays(arrays), constants, timeout=10
    )
    assert extra.shape == (6, 1)
    assert reward.shape == (6,)
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
    assert "{{ROUND_REQUEST}}" not in second_prompt
    assert "1. Task and control scope" in second_prompt
    assert "7. Request for this round" in second_prompt

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
    first_incoming = json.loads(
        (tmp_path / "parse-revision" / "attempt_01" / "prompt_feedback.json").read_text()
    )
    assert first_incoming["record_role"] == "incoming_feedback_used_to_build_request"
    assert first_incoming["target_attempt"] == 1
    assert first_incoming["source_attempt"] is None
    assert first_incoming["feedback"] is None
    generated = json.loads(
        (tmp_path / "parse-revision" / "attempt_01" / "feedback.json").read_text()
    )
    assert generated["feedback_provenance"] == {
        "record_role": "validation_feedback_generated_from_response",
        "source_attempt": 1,
        "intended_next_attempt": 2,
    }
    second_incoming = json.loads(
        (tmp_path / "parse-revision" / "attempt_02" / "prompt_feedback.json").read_text()
    )
    assert second_incoming["source_attempt"] == 1
    assert second_incoming["feedback"]["confirmed_errors"][0]["code"] == "JSON_PARSE_ERROR"


def test_single_json_fence_is_recorded_then_fully_validated(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    candidate = _candidate(name="fenced-candidate")
    response = _response(candidate)
    response["content"] = "```json\n" + json.dumps(candidate) + "\n```"
    result = run_design(
        fixed_sample=fixed,
        model="qwen/qwen3.5-9b",
        client=MockClient([response]),
        max_attempts=1,
        output_dir=tmp_path / "fenced-run",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    metadata = json.loads(
        (tmp_path / "fenced-run" / "attempt_01" / "candidate_parse_metadata.json").read_text()
    )
    assert metadata["markdown_envelope_removed"] is True
    assert metadata["accepted_envelope"] == "single_json_fence"
    assert (tmp_path / "fenced-run" / "attempt_01" / "response_content.txt").read_text().startswith(
        "```json"
    )

    invalid = _candidate(name="fenced-but-invalid")
    invalid["unexpected"] = True
    invalid_response = _response(invalid)
    invalid_response["content"] = "```\n" + json.dumps(invalid) + "\n```"
    failed = run_design(
        fixed_sample=fixed,
        model="qwen/qwen3.5-9b",
        client=MockClient([invalid_response]),
        max_attempts=1,
        output_dir=tmp_path / "fenced-invalid-run",
        worker_timeout=10,
    )
    assert failed["status"] == "failed_no_approved_candidate"
    assert not (tmp_path / "fenced-invalid-run" / "approved").exists()


def test_ambiguous_fence_feedback_is_specific_and_reaches_next_prompt(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    invalid_text = "Explanation\n```json\n{}\n```"
    client = MockClient(
        [
            {**_response(_candidate()), "content": invalid_text},
            _response(_candidate(name="fixed")),
        ]
    )
    result = run_design(
        fixed_sample=fixed,
        model=client.model,
        client=client,
        max_attempts=2,
        output_dir=tmp_path / "outer-text-revision",
        worker_timeout=10,
    )
    assert result["status"] == "approved"
    prompt = (tmp_path / "outer-text-revision" / "attempt_02" / "prompt.txt").read_text()
    assert invalid_text in prompt
    assert "explanatory text outside" in prompt
    assert "one `json`/unlabelled Markdown fence" in prompt


def test_openai_refusal_stops_without_revision_or_approved_artifact(tmp_path):
    fixed = _fixed_artifact(tmp_path)
    response = _response(_candidate(), model="gpt-4o")
    response["refusal"] = "I cannot provide this output."
    client = MockClient([response], model="gpt-4o")
    result = run_design(
        fixed_sample=fixed,
        provider="openai",
        model="gpt-4o",
        client=client,
        max_attempts=2,
        output_dir=tmp_path / "openai-refusal",
        worker_timeout=10,
    )
    assert result["status"] == "failed_no_approved_candidate"
    assert len(client.calls) == 1
    assert result["metadata"]["attempt_history"] == [
        {"attempt": 1, "status": "model_refusal"}
    ]
    assert not (tmp_path / "openai-refusal" / "approved").exists()


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
        context_length=40_000,
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
            {**_response(_candidate()), "transport_completed": False},
            "stream_incomplete",
        ),
        (
            {
                **_response(_candidate()),
                "finish_reason": "length",
            },
            "candidate_format_or_validation_failure",
        ),
        (
            {**_response(_candidate()), "tool_calls_seen": True},
            "candidate_format_or_validation_failure",
        ),
        ({**_response(_candidate()), "content": "not json"}, "candidate_json_failure"),
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
    report = evaluate_candidate(
        context,
        _candidate(),
        extra,
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
    extra, reward = design.evaluate_fixed_samples(arrays, timeout=10)
    assert extra.shape == (6, 1)
    assert np.array_equal(reward, np.zeros(6))
    assert not list((tmp_path / "design").glob("*training*"))
