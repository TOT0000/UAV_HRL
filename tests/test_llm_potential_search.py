import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
