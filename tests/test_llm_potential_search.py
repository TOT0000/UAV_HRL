import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import llm_potential
import llm_potential_search as search


def _submission(candidate_id):
    value = int(candidate_id[-2:]) / 10.0 + int(candidate_id[1:3]) / 100.0
    return {
        "candidate_id": candidate_id,
        "design_summary": f"fixture {candidate_id}",
        "components": [
            {
                "name": f"component_{candidate_id}",
                "description": "Bounded fixture component.",
                "weight": 1.0,
            }
        ],
        "code": (
            "def compute_potential_components(obs, constants):\n"
            f"    return [{value}]\n"
        ),
    }


def _block():
    return {
        "episode_range": [1, 50],
        "episode_count": 50,
        "step_count": 3000,
        "energy_efficiency_mbit_per_j": {"mean": 1.0, "standard_deviation": 0.1},
        "timely_useful_delivery_mbit": {"mean": 10.0, "standard_deviation": 1.0},
        "movement_energy_j": {"mean": 20.0, "standard_deviation": 2.0},
        "raw_reward_sum": {"mean": 1.0, "standard_deviation": 0.2},
        "shaping_sum": {"mean": -0.1, "standard_deviation": 0.01},
        "training_reward_sum": {"mean": 0.9, "standard_deviation": 0.2},
        "potential_components": [{"name": "fixture"}],
        "exploration": [],
    }


def _install_search_mocks(monkeypatch, tmp_path):
    baseline_evaluation = tmp_path / "baseline-eval"
    baseline_point = baseline_evaluation / "roi_8"
    baseline_point.mkdir(parents=True)
    baseline_manifest = baseline_point / "scenario_manifest.json"
    baseline_manifest.write_text("{}", encoding="utf-8")
    (baseline_evaluation / "paper_evaluation_metadata.json").write_text(
        json.dumps(
            {"points": [{"scenario_manifest_path": str(baseline_manifest)}]}
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        provenance={"fixture": "validation-only"},
        fixed_metadata={},
        arrays={},
        episodes=(SimpleNamespace(index=0),),
    )
    monkeypatch.setattr(search, "bounded_validation_dataset", lambda *_a, **_k: dataset)
    monkeypatch.setattr(search, "build_constants", lambda _metadata: {})
    monkeypatch.setattr(search, "build_obs_arrays", lambda _arrays: {})
    monkeypatch.setattr(
        search,
        "build_environment_source_bundle",
        lambda: {
            "bundle_sha256": "fixture-source",
            "content_sha256": "fixture-content",
            "git_sha": "fixture",
            "call_relationships": [],
            "excerpts": {},
            "rendered": {},
        },
    )
    monkeypatch.setattr(
        search,
        "render_common_prompt",
        lambda *, candidate_id, **_kwargs: f"COMMON {candidate_id}",
    )
    monkeypatch.setattr(
        search,
        "validate_potential_candidate",
        lambda submission, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        search,
        "validate_direction",
        lambda direction, candidate, reference: {
            "status": "passed",
            "requested_direction": direction,
            "formula_comparison_available": False,
            "single_feature_remove_exception": None,
        },
    )

    def save_artifact(root, **_kwargs):
        approved = Path(root) / "approved"
        approved.mkdir()
        (approved / "artifact.json").write_text("{}", encoding="utf-8")
        return approved

    monkeypatch.setattr(search, "save_potential_artifact", save_artifact)
    monkeypatch.setattr(
        search,
        "_training_fairness_provenance",
        lambda **kwargs: {
            "status": "passed",
            "candidate_run": kwargs["training"]["run_directory"],
            "training_scenario_prefix_matches_baseline": True,
            "episode_400_is_stop_not_decay_horizon": True,
        },
    )

    model_calls = []

    def call_model(*, prompt, **_kwargs):
        marker = prompt.split("COMMON ", 1)[1].split()[0]
        model_calls.append(marker)
        return {
            "content": json.dumps(_submission(marker)),
            "transport_completed": True,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(search, "_call_model", call_model)
    training_calls = []

    def train(**kwargs):
        training_calls.append(kwargs)
        run = Path(kwargs["output_directory"]) / search.SEARCH_METHOD_ID / "run"
        run.mkdir(parents=True, exist_ok=True)
        (run / "checkpoints" / "full" / "ep_0400").mkdir(parents=True)
        return {"status": "complete", "run_directory": str(run), "summaries": [_block()]}

    evaluation_calls = []

    def evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        score = float(len(evaluation_calls))
        return {
            "status": "complete",
            "checkpoint_episode": 400,
            "episode_count": 100,
            "scenario_manifest_hash": "fixture-manifest",
            "mean_episode_energy_efficiency_mbit_per_j": score,
            "std_episode_energy_efficiency_mbit_per_j": 0.1,
            "baseline_mean_episode_energy_efficiency_mbit_per_j": 0.5,
            "baseline_absolute_gap_mbit_per_j": score - 0.5,
            "baseline_percent_gap": 100.0 * (score - 0.5) / 0.5,
            "candidate_metrics": {},
            "evaluation_directory": str(kwargs["output_directory"]),
        }

    preflight = {
        "status": "passed",
        "manifest_path": str(baseline_manifest),
        "manifest_content_hash": "fixture-manifest",
        "checkpoint_episode": 400,
        "metrics": {"episode_energy_efficiency_mbit_per_j": {"mean": 0.5}},
    }
    return model_calls, training_calls, evaluation_calls, train, evaluate, lambda **_k: preflight


def test_mock_search_trains_all_candidates_and_extends_three_rounds_to_five(
    tmp_path, monkeypatch
):
    model_calls, training_calls, evaluation_calls, train, evaluate, baseline = (
        _install_search_mocks(monkeypatch, tmp_path)
    )
    output = tmp_path / "search"
    first = search.run_potential_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-eval",
        provider="openai",
        model="fixture-model",
        context_length=128000,
        output_dir=output,
        runtime_root=tmp_path / "runtime",
        max_rounds=3,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline,
    )
    assert first["status"] == "complete"
    assert len(model_calls) == len(training_calls) == len(evaluation_calls) == 12
    assert all(call["episodes"] == 400 for call in training_calls)
    assert all(call["summary_block_size"] == 50 for call in training_calls)
    assert all(call["checkpoint_episode"] == 400 for call in evaluation_calls)
    assert all(call["episodes"] == 100 for call in evaluation_calls)
    assert (output / "best_candidate.json").is_file()
    assert (output / "all_results.json").is_file()
    continuation = json.loads(
        (output / "manual_continuation.json").read_text(encoding="utf-8")
    )
    assert continuation["complete_resume_checkpoint_exists"] is True
    assert continuation["automatic_continuation_started"] is False

    resumed = search.run_potential_search(
        resume=output,
        max_rounds=5,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline,
    )
    assert resumed["status"] == "complete"
    assert len(model_calls) == len(training_calls) == len(evaluation_calls) == 20
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert [item["round"] for item in state["rounds"]] == [1, 2, 3, 4, 5]
    assert state["best_candidate"]["candidate_id"] == "r05_c04"
    assert state["rounds"][3]["reference_candidate_id"] == "r03_c04"


def test_evaluation_failure_resume_retries_new_attempt_without_generation_or_training(
    tmp_path, monkeypatch
):
    model_calls, training_calls, evaluation_calls, train, evaluate, baseline = (
        _install_search_mocks(monkeypatch, tmp_path)
    )
    legacy_preflight = baseline()
    current_preflight = {
        **legacy_preflight,
        "manifest_schema_version": "uav-hrl-scenario-v8",
        "manifest_seed": 20260927,
    }
    attempts = []

    def fail_first_evaluation(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("fixture scenario mismatch")
        return evaluate(**kwargs)

    output = tmp_path / "evaluation-resume"
    first = search.run_potential_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-eval",
        provider="openai",
        model="fixture-model",
        context_length=128000,
        output_dir=output,
        runtime_root=tmp_path / "runtime",
        max_rounds=1,
        client=object(),
        training_runner=train,
        evaluation_runner=fail_first_evaluation,
        baseline_validator=lambda **_kwargs: legacy_preflight,
    )
    assert first["status"] == "paused_evaluation_failure"
    assert len(model_calls) == 4
    assert len(training_calls) == len(attempts) == 1
    paused = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = paused["current_round"]["candidates"]["r01_c01"]
    assert record.get("evaluation") is None
    assert record["evaluation_failures"][0]["attempt"] == 1
    assert paused["rounds"] == []
    assert paused["best_candidate"] is None
    assert "manifest_schema_version" not in paused["baseline_preflight"]
    saved_legacy = json.loads(
        (output / "baseline_preflight.json").read_text(encoding="utf-8")
    )
    assert "manifest_seed" not in saved_legacy

    with pytest.raises(search.PotentialSearchError, match="baseline preflight failed"):
        search.run_potential_search(
            resume=output,
            max_rounds=1,
            client=object(),
            training_runner=train,
            evaluation_runner=fail_first_evaluation,
            baseline_validator=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("fixture baseline unavailable")
            ),
        )
    failed_validation = json.loads(
        (output / "state.json").read_text(encoding="utf-8")
    )
    assert failed_validation["baseline_preflight"] == legacy_preflight
    assert failed_validation["baseline_preflight_failure"]["status"] == "failed"
    assert json.loads(
        (output / "baseline_preflight.json").read_text(encoding="utf-8")
    ) == saved_legacy
    assert len(model_calls) == 4
    assert len(training_calls) == len(attempts) == 1

    resumed = search.run_potential_search(
        resume=output,
        max_rounds=1,
        client=object(),
        training_runner=train,
        evaluation_runner=fail_first_evaluation,
        baseline_validator=lambda **_kwargs: current_preflight,
    )
    assert resumed["status"] == "complete"
    assert len(model_calls) == 4
    assert len(training_calls) == 4
    assert len(attempts) == 5
    assert len(evaluation_calls) == 4
    saved = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = saved["rounds"][0]["candidates"]["r01_c01"]
    assert record["evaluation_attempts"] == 2
    assert len(record["evaluation_failures"]) == 1
    assert record["evaluation_failures"][0]["output_directory"].endswith("a01")
    assert record["evaluation_invocations"][1]["output_directory"].endswith("a02")
    assert record["evaluation"]["status"] == "complete"
    assert saved["best_candidate"]["candidate_id"] == "r01_c04"
    assert saved["baseline_preflight"] == current_preflight
    assert json.loads(
        (output / "baseline_preflight.json").read_text(encoding="utf-8")
    ) == current_preflight
    assert len(saved["baseline_preflight_compatibility_updates"]) == 1
    assert (output / "baseline_preflight_compatibility.json").is_file()

    repeated = search.run_potential_search(
        resume=output,
        max_rounds=1,
        client=object(),
        training_runner=train,
        evaluation_runner=fail_first_evaluation,
        baseline_validator=lambda **_kwargs: current_preflight,
    )
    assert repeated["status"] == "complete"
    repeated_state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert len(repeated_state["baseline_preflight_compatibility_updates"]) == 1
    assert len(model_calls) == len(training_calls) == 4
    assert len(attempts) == 5


def test_actual_prompt_assembly_has_full_best_latest_round_and_no_placeholders(
    monkeypatch
):
    monkeypatch.setattr(search, "movement_and_energy_spec", lambda _c: "movement spec")
    monkeypatch.setattr(search, "visual_sensing_spec", lambda _c: "vs spec")
    monkeypatch.setattr(search, "communication_spec", lambda _c: "com spec")
    monkeypatch.setattr(search, "render_environment_interface", lambda *_a, **_k: "fields")
    dataset = SimpleNamespace(fixed_metadata={})
    bundle = {
        "call_relationships": ["source relationship"],
        "rendered": {
            "movement_and_energy": "movement code",
            "visual_sensing": "vs code",
            "communication_and_service": "com code",
            "observation": "observation code",
            "delivery_and_base_reward": "reward code",
        },
    }
    best_submission = _submission("r01_c04")
    candidates = {}
    for index in range(1, 5):
        candidate_id = f"r01_c{index:02d}"
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "status": "completed",
            "submission": _submission(candidate_id),
            "candidate": _submission(candidate_id),
            "requested_direction": "initial",
            "training": {"status": "complete", "summaries": [_block()]},
            "evaluation": {
                "status": "complete",
                "checkpoint_episode": 400,
                "episode_count": 100,
                "mean_episode_energy_efficiency_mbit_per_j": float(index),
                "baseline_percent_gap": float(index),
            },
        }
    completed = {
        "round": 1,
        "candidate_order": list(candidates),
        "candidates": candidates,
    }
    state = {
        "rounds": [completed],
        "best_candidate": {
            "round": 1,
            "candidate_id": "r01_c04",
            "candidate": best_submission,
            "submission": best_submission,
            "evaluation": candidates["r01_c04"]["evaluation"],
            "mean_episode_energy_efficiency_mbit_per_j": 4.0,
        },
        "baseline_preflight": {
            "checkpoint_episode": 400,
            "metrics": {"episode_energy_efficiency_mbit_per_j": {"mean": 0.5}},
        },
    }
    current = search._new_round(state, 2)
    # Preserve a validated, not-yet-trained current-round design in later prompts.
    current["candidates"]["r02_c01"].update(
        status="validated", submission=_submission("r02_c01")
    )
    common = search.render_common_prompt(
        candidate_id="r02_c02",
        dataset=dataset,
        constants={},
        beta=1.0,
        source_bundle=bundle,
    )
    prompt = search._base_generation_prompt(
        state=state,
        current=current,
        candidate_id="r02_c02",
        common=common,
    )
    assert "{{" not in prompt
    assert json.dumps(best_submission["code"], ensure_ascii=False)[1:-1] in prompt
    assert all(candidate_id in prompt for candidate_id in candidates)
    assert "r02_c01" in prompt
    assert "potential_components" in prompt
    lowered = prompt.lower()
    assert "lipschitz" not in lowered
    assert "ordering consistency" not in lowered
    assert "reviewer" not in lowered


def test_baseline_resolution_uses_metadata_manifest_and_rejects_ambiguity(tmp_path):
    evaluation = tmp_path / "one"
    point = evaluation / "roi_8"
    point.mkdir(parents=True)
    manifest = point / "scenario_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    (evaluation / "paper_evaluation_metadata.json").write_text(
        json.dumps({"points": [{"scenario_manifest_path": str(manifest)}]}),
        encoding="utf-8",
    )
    resolved, resolved_manifest = search._resolve_baseline_evaluation(tmp_path)
    assert resolved == evaluation.resolve()
    assert resolved_manifest == manifest.resolve()
    assert search._resolve_baseline_evaluation(point) == (
        evaluation.resolve(),
        manifest.resolve(),
    )

    other = tmp_path / "two"
    other.mkdir()
    (other / "paper_evaluation_metadata.json").write_text(
        json.dumps({"points": [{"scenario_manifest_path": str(manifest)}]}),
        encoding="utf-8",
    )
    with pytest.raises(search.PotentialSearchError, match="multiple baseline"):
        search._resolve_baseline_evaluation(tmp_path)


def test_repair_prompt_has_complete_error_context_and_no_placeholders(tmp_path):
    version = tmp_path / "v01"
    version.mkdir()
    (version / "raw_response.txt").write_text("not-json", encoding="utf-8")
    record = {
        "candidate_id": "r01_c01",
        "generation_stage_request": "Generate the complete candidate.",
        "versions": [{"directory": str(version), "response_kind": "raw_unparsed"}],
        "last_validation": {
            "status": "failed",
            "stage": "parse_or_schema",
            "error": "invalid JSON",
            "unresolved_checks": ["execution was not run"],
        },
    }
    prompt = search._repair_prompt(common="COMMON", record=record, duplicate=False)
    assert "{{" not in prompt
    assert "invalid JSON" in prompt
    assert "execution was not run" in prompt
    assert "not-json" in prompt


def _direction_candidate(candidate_id, code, weights):
    return {
        "candidate_id": candidate_id,
        "design_summary": "direction fixture",
        "components": [
            {
                "name": f"component_{index}",
                "description": "fixture",
                "weight": weight,
            }
            for index, weight in enumerate(weights)
        ],
        "code": code,
    }


def test_execution_report_is_saved_complete_and_repair_feedback_keeps_all_roots(
    tmp_path,
):
    code = (
        "def compute_potential_components(obs, constants):\n"
        "    value = obs['distance'][99]\n"
        "    return [value]\n"
    )
    report = {
        "status": "failed",
        "error_count": 2,
        "errors": [
            {
                "code": "RUNTIME_FUNCTION_ERROR",
                "stage": "execution",
                "location": "compute_extra_state at candidate line 2",
                "problem": "IndexError: index 99 is out of bounds",
                "requirement": "Respect the valid distance vector length.",
                "occurrence_count": 3,
                "candidate_function": "compute_extra_state",
                "candidate_line": 2,
                "representative_samples": [
                    {"case": "index-a"},
                    {"case": "index-b"},
                    {"case": "index-c"},
                ],
                "runtime_diagnostics": {"candidate_source_excerpt": []},
            },
            {
                "code": "RUNTIME_NONFINITE",
                "stage": "execution",
                "location": "compute_extra_state at candidate line 3",
                "problem": "component 0 is not finite",
                "requirement": "Return finite component values.",
                "occurrence_count": 1,
                "candidate_function": "compute_extra_state",
                "candidate_line": 3,
                "representative_samples": [{"case": "nonfinite"}],
            },
        ],
        "checks": {"execution": {"status": "failed"}},
    }
    error = search.CandidateExecutionError("worker failed", report=report)
    details = search._exception_details(error, submitted_code=code)
    validation = {
        "status": "failed",
        "stage": "execution",
        "error": "CandidateExecutionError: worker failed",
        "error_details": details,
    }
    submission = _direction_candidate("r01_c01", code, [1.0])
    version = tmp_path / "v01"
    search._save_candidate_version(
        version,
        submission=submission,
        normalized=submission,
        validation=validation,
        raw_content=json.dumps(submission),
    )
    saved = json.loads((version / "validation_report.json").read_text(encoding="utf-8"))
    saved_report = saved["error_details"]["execution_report"]
    assert [item["code"] for item in saved_report["errors"]] == [
        "RUNTIME_FUNCTION_ERROR",
        "RUNTIME_NONFINITE",
    ]
    assert len(saved_report["errors"][0]["representative_samples"]) == 3
    assert saved_report == report
    mapped_report = saved["error_details"]["candidate_facing_execution_report"]
    assert mapped_report["errors"][0]["candidate_function"] == (
        "compute_potential_components"
    )
    assert mapped_report["errors"][0]["runtime_diagnostics"][
        "candidate_source_excerpt"
    ][1]["code"] == "    value = obs['distance'][99]"

    record = {
        "candidate_id": "r01_c01",
        "generation_stage_request": "Generate the complete candidate.",
        "versions": [{"directory": str(version), "response_kind": "parsed_json"}],
        "last_validation": validation,
    }
    prompt = search._repair_prompt(common="COMMON", record=record, duplicate=False)
    assert "RUNTIME_FUNCTION_ERROR" in prompt
    assert "RUNTIME_NONFINITE" in prompt
    assert "IndexError: index 99 is out of bounds" in prompt
    assert "Return finite component values." in prompt
    assert "index-a" in prompt and "index-b" in prompt
    assert "index-c" not in prompt
    assert "representative_samples_omitted" in prompt
    assert "still_unresolved_not_assumed_fixed" in prompt
    assert "compute_extra_state" not in prompt


def test_potential_adapter_renames_function_without_changing_source_lines():
    source = (
        "def compute_potential_components(obs, constants):\n"
        "    first = obs['distance']\n"
        "\n"
        "    return [first]\n"
    )
    adapted = llm_potential._rename_candidate_function(
        source, "compute_potential_components", "compute_extra_state"
    )
    assert adapted.splitlines()[0] == "def compute_extra_state(obs, constants):"
    assert adapted.splitlines()[1:] == source.splitlines()[1:]


def test_search_execution_failure_uses_complete_report_in_next_model_request(
    tmp_path, monkeypatch
):
    _calls, _training_calls, _evaluation_calls, train, evaluate, baseline = (
        _install_search_mocks(monkeypatch, tmp_path)
    )
    prompts = []

    def call_model(*, prompt, **_kwargs):
        prompts.append(prompt)
        marker = prompt.split("COMMON ", 1)[1].split()[0]
        return {
            "content": json.dumps(_submission(marker)),
            "transport_completed": True,
            "finish_reason": "stop",
        }

    first = True
    full_report = {
        "status": "failed",
        "errors": [
            {
                "code": "ROOT_A",
                "location": "fixed sample",
                "problem": "first independent root",
                "requirement": "repair root A",
                "occurrence_count": 2,
                "representative_samples": [{"sample": 1}],
            },
            {
                "code": "ROOT_B",
                "location": "empty probe",
                "problem": "second independent root",
                "requirement": "repair root B",
                "occurrence_count": 1,
                "representative_samples": [{"sample": "empty"}],
            },
        ],
        "error_count": 2,
    }

    def validate(_submission_value, **_kwargs):
        nonlocal first
        if first:
            first = False
            raise search.CandidateExecutionError("two roots", report=full_report)
        return {"status": "passed"}

    monkeypatch.setattr(search, "_call_model", call_model)
    monkeypatch.setattr(search, "validate_potential_candidate", validate)
    output = tmp_path / "search-execution-report"
    result = search.run_potential_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-eval",
        provider="openai",
        model="fixture-model",
        context_length=128000,
        output_dir=output,
        runtime_root=tmp_path / "runtime-execution-report",
        max_rounds=1,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline,
    )
    assert result["status"] == "complete"
    assert "ROOT_A" in prompts[1] and "ROOT_B" in prompts[1]
    assert "repair root A" in prompts[1] and "repair root B" in prompts[1]
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    first_version = Path(
        state["rounds"][0]["candidates"]["r01_c01"]["versions"][0]["directory"]
    )
    saved = json.loads(
        (first_version / "validation_report.json").read_text(encoding="utf-8")
    )
    assert saved["error_details"]["execution_report"] == full_report


def test_direction_resolution_detects_changed_formula_despite_same_local_name():
    reference = _direction_candidate(
        "reference",
        "def compute_potential_components(obs, constants):\n"
        "    x = obs['distance'] * 0.5\n"
        "    return [x]\n",
        [1.0],
    )
    candidate = _direction_candidate(
        "candidate",
        "def compute_potential_components(obs, constants):\n"
        "    x = obs['distance'] * 0.25\n"
        "    y = obs['queue'] * 0.1\n"
        "    return [x, y]\n",
        [1.0, 2.0],
    )
    result = search.validate_direction("add", candidate, reference)
    assert result["status"] == "failed"
    assert result["formula_comparison_status"] == "confirmed"
    assert any("formulas" in item for item in result["confirmed_violations"])


@pytest.mark.parametrize(
    ("constructor", "through_local"),
    [("asarray", False), ("asarray", True), ("array", False), ("array", True)],
)
def test_reweight_detects_return_dtype_change(constructor, through_local):
    def code(dtype):
        if not through_local:
            body = f"    return np.{constructor}([0.8], dtype={dtype})\n"
        else:
            body = (
                "    values = [0.8]\n"
                f"    converted = np.{constructor}(values, dtype={dtype})\n"
                "    return converted\n"
            )
        return "def compute_potential_components(obs, constants):\n" + body

    reference = _direction_candidate("reference", code("float"), [1.0])
    candidate = _direction_candidate("candidate", code("int"), [2.0])
    result = search.validate_direction("reweight", candidate, reference)
    assert result["status"] == "failed"
    assert result["formula_comparison_status"] == "confirmed"
    difference = result["formula_differences"]["return_conversion_difference"]
    assert "dtype=float" in difference["reference"]
    assert "dtype=int" in difference["candidate"]
    assert any(
        "return conversion changed" in issue
        and "dtype=float" in issue
        and "dtype=int" in issue
        for issue in result["confirmed_violations"]
    )


def test_reweight_accepts_same_return_conversion_after_local_renames():
    reference = _direction_candidate(
        "reference",
        "def compute_potential_components(obs, constants):\n"
        "    values = [0.8]\n"
        "    selected_dtype = float\n"
        "    converted = np.asarray(values, dtype=selected_dtype)\n"
        "    return converted\n",
        [1.0],
    )
    candidate = _direction_candidate(
        "candidate",
        "def compute_potential_components(obs, constants):\n"
        "    renamed_values = [0.8]\n"
        "    renamed_dtype = float\n"
        "    renamed_result = np.asarray(renamed_values, dtype=renamed_dtype)\n"
        "    return renamed_result\n",
        [2.0],
    )
    result = search.validate_direction("reweight", candidate, reference)
    assert result["status"] == "passed"
    assert result["formula_comparison_status"] == "confirmed"
    assert result["formula_differences"]["return_conversion_difference"] is None
    assert result["confirmed_violations"] == []


@pytest.mark.parametrize(
    ("direction", "reference_code", "candidate_code", "old_weights", "new_weights"),
    [
        (
            "add",
            "    original = obs['distance'] * 0.5\n    return [original]\n",
            "    renamed = obs['distance'] * 0.5\n"
            "    added = obs['queue'] * 0.1\n    return [renamed, added]\n",
            [1.0],
            [1.0, 2.0],
        ),
        (
            "remove",
            "    first = obs['distance'] * 0.5\n"
            "    second = obs['queue'] * 0.1\n    return [first, second]\n",
            "    renamed = obs['distance'] * 0.5\n    return [renamed]\n",
            [1.0, 2.0],
            [1.0],
        ),
        (
            "reweight",
            "    first = obs['distance'] * 0.5\n"
            "    second = obs['queue'] * 0.1\n    return [first, second]\n",
            "    renamed_a = obs['distance'] * 0.5\n"
            "    renamed_b = obs['queue'] * 0.1\n"
            "    return [renamed_a, renamed_b]\n",
            [1.0, 2.0],
            [3.0, 4.0],
        ),
    ],
)
def test_direction_resolution_accepts_local_rename_for_same_formula(
    direction, reference_code, candidate_code, old_weights, new_weights
):
    reference = _direction_candidate(
        "reference",
        "def compute_potential_components(obs, constants):\n" + reference_code,
        old_weights,
    )
    candidate = _direction_candidate(
        "candidate",
        "def compute_potential_components(obs, constants):\n" + candidate_code,
        new_weights,
    )
    result = search.validate_direction(direction, candidate, reference)
    assert result["status"] == "passed"
    assert result["formula_comparison_status"] == "confirmed"
    assert result["confirmed_violations"] == []


def test_direction_resolution_reports_unconfirmed_control_flow_without_rejection():
    code = (
        "def compute_potential_components(obs, constants):\n"
        "    if obs['valid']:\n"
        "        value = obs['distance']\n"
        "    else:\n"
        "        value = 0.0\n"
        "    return [value]\n"
    )
    reference = _direction_candidate("reference", code, [1.0])
    candidate = _direction_candidate("candidate", code, [2.0])
    result = search.validate_direction("reweight", candidate, reference)
    assert result["status"] == "unconfirmed"
    assert result["formula_comparison_status"] == "unconfirmed"
    assert result["confirmed_violations"] == []
    assert result["unconfirmed_checks"][0]["status"] == "unconfirmed_not_rejected"


def test_model_adapter_receives_each_complete_long_candidate_definition_once(
    tmp_path, monkeypatch
):
    _model_calls, _training_calls, _evaluation_calls, train, evaluate, baseline = (
        _install_search_mocks(monkeypatch, tmp_path)
    )
    prompts = []
    submissions = {}

    def long_submission(candidate_id):
        result = _submission(candidate_id)
        result["components"][0]["weight"] = float(int(candidate_id[-2:]))
        padding = f"FULL_FORMULA_{candidate_id}_" + (candidate_id * 520)
        result["code"] = (
            "def compute_potential_components(obs, constants):\n"
            f"    # {padding}\n"
            f"    return [{int(candidate_id[-2:]) / 10.0 + int(candidate_id[1:3]) / 100.0}]\n"
        )
        submissions[candidate_id] = result
        return result

    def call_model(*, prompt, **_kwargs):
        prompts.append(prompt)
        marker = prompt.split("COMMON ", 1)[1].split()[0]
        return {
            "content": json.dumps(long_submission(marker)),
            "transport_completed": True,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(search, "_call_model", call_model)
    result = search.run_potential_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-eval",
        provider="openai",
        model="fixture-model",
        context_length=128000,
        output_dir=tmp_path / "search-long",
        runtime_root=tmp_path / "runtime-long",
        max_rounds=2,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline,
    )
    assert result["status"] == "complete"
    first_revision_prompt = prompts[4]
    for index in range(1, 5):
        candidate_id = f"r01_c{index:02d}"
        code_literal = json.dumps(
            submissions[candidate_id]["code"], ensure_ascii=False
        )[1:-1]
        assert len(submissions[candidate_id]["code"]) > 2400
        assert first_revision_prompt.count(code_literal) == 1
        assert f'"weight": {float(index)}' in first_revision_prompt
        assert f'"candidate_definition_reference": "{candidate_id}"' in (
            first_revision_prompt
        )
    assert '"mean_episode_energy_efficiency_mbit_per_j": 1.0' in (
        first_revision_prompt
    )


def test_context_budget_pause_saves_full_prompt_and_does_not_call_model(
    tmp_path, monkeypatch
):
    model_calls, _training_calls, _evaluation_calls, train, evaluate, baseline = (
        _install_search_mocks(monkeypatch, tmp_path)
    )
    budget_calls = 0

    def budget(_prompt, **_kwargs):
        nonlocal budget_calls
        budget_calls += 1
        return {
            "fits_client_budget": budget_calls <= 4,
            "method": "fixture_exact",
            "exact": True,
            "estimated_input_tokens": 1,
        }

    monkeypatch.setattr(search, "estimate_token_budget", budget)
    output = tmp_path / "search-budget"
    result = search.run_potential_search(
        validation_sources=[tmp_path / "validation"],
        baseline_run=tmp_path / "baseline-run",
        baseline_evaluation=tmp_path / "baseline-eval",
        provider="openai",
        model="fixture-model",
        context_length=128000,
        output_dir=output,
        runtime_root=tmp_path / "runtime-budget",
        max_rounds=2,
        client=object(),
        training_runner=train,
        evaluation_runner=evaluate,
        baseline_validator=baseline,
    )
    assert result["status"] == "paused_context_budget_exceeded"
    assert model_calls == [f"r01_c{index:02d}" for index in range(1, 5)]
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    record = state["current_round"]["candidates"]["r02_c01"]
    stop = record["context_budget_stop"]
    assert stop["request_was_sent"] is False
    assert stop["candidate_definitions_were_truncated_or_removed"] is False
    assert any(
        item["section"] == "complete_candidate_definitions"
        for item in stop["largest_sections"]
    )
    prompt_path = Path(stop["prompt_budget_report"]["prompt_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    assert all(
        json.dumps(
            _submission(f"r01_c{index:02d}")["code"], ensure_ascii=False
        )[1:-1]
        in prompt
        for index in range(1, 5)
    )


def test_previous_prompt_version_is_migrated_for_resume(tmp_path):
    constants = {"fixture": {"value": 1}}
    source_bundle = {
        "bundle_sha256": "same-source",
        "content_sha256": "same-content",
        "git_sha": "new-git",
        "excerpts": {},
        "rendered": {},
    }
    state = {
        "prompt_version": search.LEGACY_PROMPT_VERSION,
        "environment_source_bundle_sha256": "same-source",
        "git_sha": "old-git",
        "settings": {
            "evaluation_area_m": [1000.0, 1000.0],
            "episode_seconds": 60,
            "candidate_count_per_round": 4,
            "candidate_generation_calls_per_request": 1,
            "train_episodes": 400,
            "evaluation_episodes": 100,
            "evaluation_roi_count": 8,
            "summary_block_size": 50,
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "constants.json").write_text(json.dumps(constants), encoding="utf-8")
    migration = search._prepare_resume_compatibility(
        output=tmp_path,
        state=state,
        source_bundle=source_bundle,
        constants=constants,
    )
    assert migration is not None
    assert migration["from_prompt_version"] == search.LEGACY_PROMPT_VERSION
    assert migration["to_prompt_version"] == search.SEARCH_PROMPT_VERSION
    assert state["prompt_version"] == search.SEARCH_PROMPT_VERSION
    assert Path(migration["state_backup"]).is_file()
