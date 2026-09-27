import json

import numpy as np
import pytest
import torch

from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from run_llm_sampling import (
    _allocate_output_directory,
    _read_checkpoint,
    build_parser,
    collect_samples,
)
from scenario_manifest import ScenarioManifest, generate_manifest


def test_cli_accepts_arbitrary_positive_episode_and_noise_values():
    parser = build_parser()
    defaults = parser.parse_args(["--checkpoint", "model"])
    assert defaults.episodes == 100
    assert defaults.noise_std == 0.1
    custom = parser.parse_args(
        [
            "--checkpoint",
            "model",
            "--episodes",
            "7",
            "--noise-std",
            "0.025",
            "--sampling-seed",
            "17",
        ]
    )
    assert custom.episodes == 7
    assert custom.noise_std == 0.025
    assert custom.sampling_seed == 17


def test_balanced_sampling_manifest_and_round_trip(tmp_path):
    manifest = generate_manifest(
        "test", 1234, 15, balanced_num_gt=True
    )
    counts = [entry["num_GT"] for entry in manifest.episodes]
    frequencies = {value: counts.count(value) for value in range(2, 9)}
    assert max(frequencies.values()) - min(frequencies.values()) <= 1
    path = manifest.save(tmp_path / "manifest.json")
    assert ScenarioManifest.load(path).content_hash == manifest.content_hash


def test_checkpoint_method_is_read_from_metadata_and_unsupported_is_rejected(tmp_path):
    checkpoint = tmp_path / "ep_0250"
    checkpoint.mkdir()
    torch.save({}, checkpoint / "models.pt")
    payload = {
        "checkpoint_type": "model-only",
        "episode": 249,
        "experiment": {
            "method_spec": {"method_id": "td3_dinkelbach"},
            "training_seed": 9,
            "formal_config": {"episode_seconds": 60},
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    source = _read_checkpoint(checkpoint)
    assert source["completed_episodes"] == 250
    assert source["method"].routing == "safe_ddqn"

    payload["experiment"]["method_spec"]["method_id"] = "td3_dinkelbach_dqn"
    (checkpoint / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe-DDQN"):
        _read_checkpoint(checkpoint)


def test_output_directories_are_unique_and_do_not_overwrite(tmp_path):
    first = _allocate_output_directory(tmp_path, "td3_dinkelbach", 250)
    second = _allocate_output_directory(tmp_path, "td3_dinkelbach", 250)
    assert first != second
    assert first.is_dir() and second.is_dir()
def test_model_checkpoint_sampling_smoke_writes_reloadable_enriched_replay(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    method = MethodSpec.parse("td3_dinkelbach")
    training_seed = 2718
    train(
        TrainingConfig(
            total_episodes=1,
            mode="train",
            episode_seconds=1,
            routing_slot_seconds=0.25,
            warmup_joint_transitions=100,
            routing_warmup_transitions=100,
            batch_size=1,
            replay_max_size=4,
            model_checkpoint_every=1,
            checkpoint_root=str(checkpoint_root),
            enable_model_checkpoints=True,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=training_seed,
        ),
        scenario_manifest=generate_manifest("train", 3141, 1, num_gt=2),
        method_spec=method,
    )
    summary = collect_samples(
        checkpoint_root / "models" / "ep_0001",
        episodes=1,
        noise_std=0.03125,
        sampling_seed=1618,
        output_root=tmp_path / "samples",
    )
    assert summary["complete"] is True
    assert summary["transition_count"] == 1
    assert summary["sampling_actor_updates"] == 0
    assert summary["sampling_critic_updates"] == 0
    assert summary["sampling_routing_updates"] == 0
    assert all(summary["evaluation_invariants"].values())
    with np.load(summary["joint_replay"], allow_pickle=False) as replay:
        assert replay["state"].shape[0] == 1
        assert replay["current_sr_backlog_bits"].shape[0] == 1
        assert replay["next_uav_backlog_bits"].shape[0] == 1
        assert replay["auxiliary_valid"].tolist() == [[True]]
        assert replay["dinkelbach_lambda"].shape == (1, 1)
