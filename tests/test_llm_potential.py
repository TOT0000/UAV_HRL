import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from experiment_config import MethodSpec
from llm_design_contract import build_obs_arrays
from llm_potential import (
    MergeableMoments,
    PotentialError,
    artifact_identity,
    candidate_fingerprint,
    compute_shaping,
    load_potential_artifact,
    save_potential_artifact,
    summarize_component_records,
    validate_potential_candidate,
)
from llm_runtime import build_online_obs
from replay_auxiliary import empty_snapshot
from run_experiment import build_parser, run as run_training, run_evaluate, run_resume
from utils_update_v2 import ReplayBufferJoint


def _constants():
    return {
        name: {
            "value": value,
            "dtype": "float",
            "unit": "m" if "environment" in name else "s",
            "meaning": "test fixture",
            "source": "test fixture",
        }
        for name, value in {
            "environment_width_m": 1000.0,
            "environment_height_m": 1000.0,
            "episode_seconds": 60.0,
            "vs_deadline_seconds": 2.5,
            "com_deadline_seconds": 2.0,
        }.items()
    }


def _candidate(candidate_id="fixture", weight=2.0):
    return {
        "candidate_id": candidate_id,
        "design_summary": "Mean remaining energy as a bounded fixture potential.",
        "components": [
            {
                "name": "mean_energy",
                "description": "Mean normalized UAV energy; already bounded in [0,1].",
                "weight": float(weight),
            }
        ],
        "code": (
            "def compute_potential_components(obs, constants):\n"
            "    value = np.mean(obs['uav_remaining_energy_fraction'])\n"
            "    return np.asarray([value], dtype=np.float32)\n"
        ),
    }


def _obs_arrays():
    state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
    # Energy is local feature index 4 in the authoritative movement schema.
    state[np.arange(16) * 17 + 4] = 0.5
    obs = build_online_obs(state, np.zeros(16, dtype=bool), empty_snapshot())
    return {
        key: np.stack((np.asarray(value), np.asarray(value)), axis=0)
        for key, value in obs.items()
    }


def test_potential_shaping_hand_calculations_and_terminal_conventions():
    current = np.asarray([0.2, 0.8])
    following = np.asarray([0.5, 0.1])
    weights = np.asarray([2.0, -3.0])

    ordinary = compute_shaping(
        current, following, weights, gamma=1.0, terminated=False,
        bootstrap_allowed=True,
    )
    assert ordinary["current_potential"] == pytest.approx(-2.0)
    assert ordinary["effective_next_potential"] == pytest.approx(0.7)
    assert ordinary["component_shaping"] == pytest.approx([0.6, 2.1])
    assert ordinary["shaping"] == pytest.approx(2.7)

    discounted = compute_shaping(
        current, following, weights, gamma=0.5, terminated=False,
        bootstrap_allowed=True,
    )
    assert discounted["shaping"] == pytest.approx(2.35)

    terminal = compute_shaping(
        current, following, weights, gamma=1.0, terminated=True,
        bootstrap_allowed=False,
    )
    assert terminal["effective_next_potential"] == 0.0
    assert terminal["shaping"] == pytest.approx(2.0)

    with pytest.raises(PotentialError, match="non-terminal"):
        compute_shaping(
            current, following, weights, gamma=1.0, terminated=False,
            bootstrap_allowed=False,
        )


def test_gamma_one_terminal_sequence_telescopes():
    weights = np.asarray([1.5, -0.5])
    states = [
        np.asarray([0.2, 0.4]),
        np.asarray([0.7, 0.1]),
        np.asarray([0.3, 0.8]),
    ]
    total = 0.0
    for index in range(len(states)):
        following = states[index + 1] if index + 1 < len(states) else states[index]
        item = compute_shaping(
            states[index], following, weights, gamma=1.0,
            terminated=index == len(states) - 1,
            bootstrap_allowed=index < len(states) - 1,
        )
        total += item["shaping"]
    assert total == pytest.approx(-float(weights @ states[0]))


def test_replay_keeps_base_lambda_reconstruction_separate_from_shaping():
    replay = ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=2,
        record_potential_shaping=True,
    )
    state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
    replay.add(
        state,
        np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
        state,
        False,
        delivered_mbits=10.0,
        total_mobility_energy=2.0,
        c9_penalty=1.0,
        c10_penalty=0.5,
        com_range_penalty=0.25,
        potential_shaping=3.0,
    )
    first = replay._reward_numpy(
        np.asarray([0]), current_lambda=1.0, gamma=1.0,
        potential_shaping_enabled=True,
    )
    second = replay._reward_numpy(
        np.asarray([0]), current_lambda=2.0, gamma=1.0,
        potential_shaping_enabled=True,
    )
    assert first.item() == pytest.approx(10.0 - 2.0 - 1.75 + 3.0)
    assert second.item() == pytest.approx(10.0 - 4.0 - 1.75 + 3.0)
    assert replay.state.shape[1] == MOVEMENT_STATE_DIM
    assert replay.potential_shaping[0, 0] == pytest.approx(3.0)


def test_candidate_validation_artifact_type_and_duplicate_fingerprint(tmp_path):
    candidate = _candidate()
    report = validate_potential_candidate(
        candidate,
        constants_metadata=_constants(),
        obs_arrays=_obs_arrays(),
        timeout=20.0,
        expected_id="fixture",
    )
    assert report["status"] == "passed"
    artifact = save_potential_artifact(
        tmp_path / "run",
        candidate=candidate,
        constants_metadata=_constants(),
        validation_report=report,
        evaluation_report={"status": "validation_only"},
        provenance={"fixture": True},
        gamma=1.0,
    )
    loaded = load_potential_artifact(artifact)
    assert artifact_identity(loaded)["gamma"] == 1.0
    assert artifact_identity(loaded)["beta"] == 1.0

    renamed = _candidate("renamed")
    renamed["design_summary"] = "Different prose"
    renamed["components"][0]["name"] = "renamed_component"
    renamed["components"][0]["description"] = "Different prose."
    renamed["code"] = "# formatting only\n" + renamed["code"]
    assert candidate_fingerprint(candidate) == candidate_fingerprint(renamed)

    old = artifact / "artifact.json"
    payload = json.loads(old.read_text(encoding="utf-8"))
    payload["schema_version"] = "uav-hrl-approved-shared-feature-design-v2"
    old.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PotentialError, match="not an approved potential"):
        load_potential_artifact(artifact)


def test_mergeable_component_statistics_match_direct_calculation_and_missing():
    rows = []
    direct = []
    for episode in range(2):
        potential = MergeableMoments()
        ordinary = MergeableMoments()
        terminal = MergeableMoments()
        for value in (episode + 1.0, episode + 2.0):
            potential.add(value)
        ordinary.add(0.25 + episode)
        terminal.add(-1.0 - episode)
        direct.extend((episode + 1.0, episode + 2.0))
        rows.append(
            {
                "components": [{
                    "weighted_potential": potential.to_dict(),
                    "ordinary_shaping": ordinary.to_dict(),
                    "terminal_shaping": terminal.to_dict(),
                }]
            }
        )
    merged = summarize_component_records(rows, component_names=["x"])[0]
    assert merged["weighted_potential"]["count"] == 4
    assert merged["weighted_potential"]["mean"] == pytest.approx(np.mean(direct))
    assert merged["weighted_potential"]["standard_deviation"] == pytest.approx(
        np.std(direct)
    )
    assert MergeableMoments().to_dict()["mean"] is None
    assert MergeableMoments().to_dict()["missing"] is True


def test_method_and_cli_are_isolated_from_existing_methods():
    baseline = MethodSpec.parse("td3_dinkelbach")
    potential = MethodSpec.parse("td3_dinkelbach_llm_potential")
    assert baseline.potential_enabled is False
    assert potential.potential_enabled is True
    assert potential.llm_enabled is False
    assert potential.agent == baseline.agent
    assert potential.reward_mode == baseline.reward_mode
    assert potential.to_dict()["method_id"] == "td3_dinkelbach_llm_potential"
    baseline_args = build_parser().parse_args(["td3_dinkelbach", "--smoke"])
    assert not hasattr(baseline_args, "potential_artifact")
    potential_args = build_parser().parse_args(
        [
            "td3_dinkelbach_llm_potential",
            "--potential-artifact",
            "approved",
            "--smoke",
        ]
    )
    assert potential_args.potential_artifact == "approved"


def test_small_training_full_checkpoint_resume_and_evaluation(request):
    tmp_path = Path(tempfile.mkdtemp(prefix="uav-pot-e2e-"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    candidate = {
        "candidate_id": "e2e-fixture",
        "design_summary": "Squared remaining-time fixture potential.",
        "components": [{
            "name": "time_square",
            "description": "Square of remaining episode fraction, normalized to [0,1].",
            "weight": 0.5,
        }],
        "code": (
            "def compute_potential_components(obs, constants):\n"
            "    value = float(obs['remaining_episode_fraction'])\n"
            "    return np.asarray([value * value], dtype=np.float32)\n"
        ),
    }
    artifact = save_potential_artifact(
        tmp_path / "artifact-source",
        candidate=candidate,
        constants_metadata=_constants(),
        validation_report={"status": "fixture"},
        evaluation_report={"status": "fixture"},
        provenance={"fixture": True},
        gamma=1.0,
    )
    output = tmp_path / "runs"
    args = build_parser().parse_args(
        [
            "td3_dinkelbach_llm_potential",
            "--potential-artifact", str(artifact),
            "--episodes", "1",
            "--episode-seconds", "1",
            "--roi-count", "2",
            "--checkpoint-interval", "1",
            "--output-root", str(output),
        ]
    )
    assert run_training(args) == 0
    runs = list((output / "td3_dinkelbach_llm_potential").iterdir())
    assert len(runs) == 1
    run = runs[0]
    first_identity = artifact_identity(load_potential_artifact(run / "potential_artifact"))
    assert first_identity["gamma"] == 1.0
    assert (run / "checkpoints" / "full" / "ep_0001" / "potential_artifact").is_dir()
    assert (run / "checkpoints" / "models" / "ep_0001" / "potential_artifact").is_dir()

    resume_args = build_parser().parse_args(
        ["resume", str(run), "--target-episodes", "2"]
    )
    assert run_resume(resume_args) == 0
    checkpoint = run / "checkpoints" / "full" / "ep_0002"
    assert checkpoint.is_dir()
    assert artifact_identity(
        load_potential_artifact(checkpoint / "potential_artifact")
    ) == first_identity
    rows = [
        json.loads(line)
        for line in (run / "potential_training_episode_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["episode"] for row in rows] == [1, 2]
    assert all(
        row["raw_reward_sum"] + row["shaping_sum"]
        == pytest.approx(row["training_reward_sum"])
        for row in rows
    )
    with np.load(checkpoint / "joint_replay.npz", allow_pickle=False) as replay:
        assert replay["state"].shape == (2, MOVEMENT_STATE_DIM)
        assert replay["potential_shaping"].shape == (2, 1)
        assert float(np.sum(replay["potential_shaping"])) == pytest.approx(
            sum(row["shaping_sum"] for row in rows)
        )

    evaluate_args = build_parser().parse_args(
        [
            "evaluate", str(run),
            "--checkpoint-episode", "2",
            "--episodes", "1",
            "--episode-seconds", "1",
            "--smoke",
        ]
    )
    assert run_evaluate(evaluate_args) == 0
    evaluations = list((run / "evaluation" / "ep_2").iterdir())
    assert len(evaluations) == 1
    metadata = json.loads(
        (evaluations[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["movement_state_dimensions"]["total"] == MOVEMENT_STATE_DIM
    assert metadata["potential_artifact_identity"] == first_identity
