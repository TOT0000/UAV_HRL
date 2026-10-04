import json

import numpy as np
import pytest

from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    ModelCandidateSchemaError,
    enrich_model_candidate,
    execute_candidate_isolated,
    feature_reward,
    model_candidate_schema_issues,
    validate_candidate,
)
from llm_design import run_design
from llm_candidate_worker import CandidateOutputValidationError, _run_function
from llm_design_contract import (
    build_obs_arrays,
    load_design_inputs,
    model_candidate_schema,
    render_prompt,
)
from run_llm_design import build_parser
from test_llm_design import MockClient, _fixed_artifact, _response



def _constants():
    return {
        "num_uav": {
            "value": 16,
            "dtype": "int",
            "unit": None,
            "meaning": "fixture",
            "source": "fixture",
        }
    }


def _submission(code="def compute_extra_state(obs, constants):\n    return [0.25]\n", weight=2.5):
    return {
        "features": [
            {
                "name": "fixture_feature",
                "description": "Fixture feature normalized to [0,1].",
                "reward_weight": weight,
            }
        ],
        "code": code,
    }


def test_simplified_candidate_is_enriched_without_model_metadata():
    submitted = _submission()
    enriched, transformation = enrich_model_candidate(submitted, _constants())
    validate_candidate(enriched, _constants())
    assert set(submitted) == {"features", "code"}
    assert set(submitted["features"][0]) == {"name", "description", "reward_weight"}
    assert enriched["reward_input_mode"] == "current_only"
    assert enriched["features"][0]["index"] == 0
    assert enriched["features"][0]["dtype"] == "float32"
    assert enriched["features"][0]["range"] == {"minimum": 0.0, "maximum": 1.0}
    assert transformation["dependency_mapping_precision"].startswith("aggregate")


def test_simplified_candidate_runs_through_full_fixed_sample_worker(tmp_path):
    baseline = _fixed_artifact(tmp_path)
    arrays, _, _, constants = load_design_inputs(baseline)
    candidate, _ = enrich_model_candidate(_submission(), constants)
    features, reward, report = execute_candidate_isolated(
        candidate,
        build_obs_arrays(arrays),
        constants,
        timeout=30,
    )
    assert features.shape == (arrays["state"].shape[0], 1)
    assert features.dtype == np.float32
    assert features[:, 0].tolist() == pytest.approx([0.25] * len(features))
    assert reward.tolist() == pytest.approx([0.625] * len(features))
    assert report["status"] == "passed"


@pytest.mark.parametrize(
    "value",
    ([0.25], np.asarray([0.25], dtype=np.float64), np.asarray([0.25], dtype=np.float32)),
)
def test_worker_accepts_numeric_list_and_arrays_then_converts_float32(value):
    obs = {"state": np.asarray([0.5], dtype=np.float32)}
    result = _run_function(lambda obs, constants: value, obs, {}, 1, "fixture")
    assert result.dtype == np.float32
    assert result.tolist() == pytest.approx([0.25])


@pytest.mark.parametrize(
    "value,match",
    [
        ("0.25", "numeric list"),
        ([[0.25]], "shape"),
        ([0.25, 0.5], "shape"),
        ([float("nan")], "NaN or Infinity"),
        ([float("inf")], "NaN or Infinity"),
        ([1.1], r"outside \[0,1\]"),
    ],
)
def test_worker_rejects_invalid_raw_outputs(value, match):
    obs = {"state": np.asarray([0.5], dtype=np.float32)}
    with pytest.raises(ValueError, match=match):
        _run_function(lambda obs, constants: value, obs, {}, 1, "fixture")


def test_large_signed_weights_and_extra_reward_are_allowed_but_nonfinite_is_not():
    candidate, _ = enrich_model_candidate(_submission(weight=4.0), _constants())
    validate_candidate(candidate, _constants())
    assert float(feature_reward(np.asarray([0.75]), candidate)) == pytest.approx(3.0)
    candidate["features"][0]["reward_weight"] = float("inf")
    with pytest.raises(CandidateError, match="finite"):
        validate_candidate(candidate, _constants())


def test_model_schema_collects_multiple_independent_paths():
    submitted = {
        "features": [
            {"name": "same", "description": " ", "reward_weight": "bad"},
            {
                "name": "same",
                "description": 7,
                "reward_weight": "also bad",
                "unexpected": True,
            },
        ],
        "extra": 1,
    }
    issues = model_candidate_schema_issues(submitted)
    paths = {item["json_path"] for item in issues}
    assert "$.code" in paths
    assert "$.extra" in paths
    assert "$.features[0].description" in paths
    assert "$.features[0].reward_weight" in paths
    assert "$.features[1].description" in paths
    assert "$.features[1].reward_weight" in paths
    assert "$.features[1].unexpected" in paths
    assert any(item["code"] == "MODEL_SCHEMA_DUPLICATE_NAME" for item in issues)
    with pytest.raises(ModelCandidateSchemaError) as caught:
        enrich_model_candidate(submitted, _constants())
    assert len(caught.value.issues) == len(issues)


def test_worker_range_issues_keep_each_feature_and_do_not_clip():
    obs = {"state": np.asarray([0.5], dtype=np.float32)}
    with pytest.raises(CandidateOutputValidationError) as caught:
        _run_function(lambda obs, constants: [1.2, -0.2], obs, {}, 2, "fixture")
    bounds = [item for item in caught.value.issues if item["code"] == "RUNTIME_FEATURE_BOUNDS"]
    assert {item["feature_index"] for item in bounds} == {0, 1}
    assert sorted({item["observed_value"] for item in bounds}) == pytest.approx([-0.2, 1.2])


def test_worker_report_aggregates_indexed_range_issues_across_samples(tmp_path):
    baseline = _fixed_artifact(tmp_path)
    arrays, _, _, constants = load_design_inputs(baseline)
    submitted = {
        "features": [
            {"name": "too_high", "description": "fixture", "reward_weight": 0.0},
            {"name": "too_low", "description": "fixture", "reward_weight": 0.0},
        ],
        "code": "def compute_extra_state(obs, constants):\n    return [1.2, -0.2]\n",
    }
    candidate, _ = enrich_model_candidate(submitted, constants)
    with pytest.raises(CandidateExecutionError) as caught:
        execute_candidate_isolated(
            candidate,
            build_obs_arrays(arrays),
            constants,
            timeout=30,
        )
    report = caught.value.report
    issues = [
        item for item in report["errors"] if item["code"] == "RUNTIME_FEATURE_BOUNDS"
    ]
    assert {item["feature_index"] for item in issues} == {0, 1}
    assert {item["feature_name"] for item in issues} == {"too_high", "too_low"}
    assert all(item["occurrence_count"] > 1 for item in issues)
    assert all(len(item["representative_samples"]) <= 3 for item in issues)
    assert not (tmp_path / "unexpected-clipped-output.npz").exists()


def test_design_revision_distinguishes_json_and_aggregated_model_schema(tmp_path):
    baseline = _fixed_artifact(tmp_path)
    invalid = {
        "features": [
            {"name": "first", "description": "fixture", "reward_weight": "bad"},
            {"name": "second", "description": "fixture", "reward_weight": "also bad"},
        ],
        "code": "def compute_extra_state(obs, constants):\n    return [0.0, 0.0]\n",
    }
    client = MockClient([_response(invalid), _response(_submission())])
    output = tmp_path / "schema-revision"
    run_design(
        fixed_sample=baseline,
        model=client.model,
        client=client,
        max_attempts=2,
        output_dir=output,
        worker_timeout=20,
    )
    report = json.loads((output / "attempt_01" / "validation_report.json").read_text())
    assert report["checks"]["json"]["status"] == "passed"
    assert report["checks"]["schema"]["status"] == "failed"
    assert report["checks"]["static"]["status"] == "not_run"
    paths = {item["json_path"] for item in report["errors"]}
    assert paths == {
        "$.features[0].reward_weight",
        "$.features[1].reward_weight",
    }
    assert all(item["code"] != "JSON_PARSE_ERROR" for item in report["errors"])
    assert json.loads(
        (output / "attempt_01" / "parsed_model_candidate.json").read_text()
    ) == invalid
    second_prompt = (output / "attempt_02" / "prompt.txt").read_text()
    assert '"reward_weight": "bad"' in second_prompt
    assert "$.features[0].reward_weight" in second_prompt
    assert "$.features[1].reward_weight" in second_prompt
    assert "JSON_PARSE_ERROR" not in second_prompt


def test_prompt_uses_simplified_contract_and_no_weight_sum_rule(tmp_path):
    baseline_dir = _fixed_artifact(tmp_path)
    arrays, metadata, baseline, constants = load_design_inputs(baseline_dir)
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
    assert "There is no restriction on the sum or absolute sum of weights" in prompt
    assert "source_fields" not in prompt
    assert '"features"' in prompt and '"code"' in prompt
    assert model_candidate_schema()["required"] == ["features", "code"]
    assert "image_quantity=RoI area/camera footprint area" in prompt
    assert "raw ratio may exceed 1" in prompt
    assert "physical_bits=packet_max_bits*clip(image_quantity,0,1)" in prompt
    assert prompt.count("d_L=(h^2+d^2)/(b1*h+d)") == 1
    assert prompt.count("physical_bits=packet_max_bits*clip(image_quantity,0,1)") == 1


@pytest.mark.parametrize("model", ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b"])
def test_arbitrary_qwen_inventory_id_uses_common_cli_path(tmp_path, model):
    baseline = _fixed_artifact(tmp_path)
    args = build_parser().parse_args(
        ["--fixed-sample", str(baseline), "--model", model, "--dry-run"]
    )
    assert args.provider == "lmstudio"
    assert args.model == model
