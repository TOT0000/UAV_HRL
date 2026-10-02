"""Approved LLM design loading and persistent current-only execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from llm_candidate import ApprovedDesign, feature_reward, load_approved_design
from llm_candidate_worker import (
    SAFE_BUILTINS,
    _run_function,
    _validate_output_ranges,
)
from llm_design_contract import (
    OBS_INTERFACE_VERSION,
    derive_named_state_fields,
    obs_keys_for_interface,
    runtime_constants,
)
from replay_auxiliary import SNAPSHOT_FIELD_SPECS


LLM_RUNTIME_CONTRACT_VERSION = "uav-hrl-llm-shared-feature-runtime-v2"
RUN_ARTIFACT_DIRECTORY_NAME = "llm_artifact"


class LLMRuntimeError(RuntimeError):
    pass


def artifact_identity(design: ApprovedDesign) -> dict[str, Any]:
    provenance = design.artifact.get("provenance") or {}
    try:
        beta = float(provenance["beta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMRuntimeError("approved artifact has no finite beta") from exc
    if not math.isfinite(beta):
        raise LLMRuntimeError("approved artifact beta must be finite")
    return {
        "runtime_contract_version": LLM_RUNTIME_CONTRACT_VERSION,
        "artifact_content_sha256": str(design.artifact["content_sha256"]),
        "candidate_name": str(design.artifact["candidate_name"]),
        "feature_count": int(design.artifact["feature_count"]),
        "feature_reward_weights": [
            float(value) for value in design.artifact["feature_reward_weights"]
        ],
        "beta": beta,
        "observation_interface_version": str(
            design.artifact["observation_interface_version"]
        ),
        "source_model_requested": provenance.get("model_requested"),
        "source_model_actual": provenance.get("model_actual"),
    }


def copy_approved_artifact(source: str | Path, run_directory: str | Path):
    """Validate, copy, and revalidate an immutable design inside a run."""

    source_design = load_approved_design(source)
    destination = Path(run_directory).resolve() / RUN_ARTIFACT_DIRECTORY_NAME
    if destination.exists():
        raise FileExistsError(f"run LLM artifact already exists: {destination}")
    shutil.copytree(source_design.directory, destination)
    copied_design = load_approved_design(destination)
    if artifact_identity(copied_design) != artifact_identity(source_design):
        raise LLMRuntimeError("copied approved artifact identity changed")
    return copied_design


def load_run_artifact(
    run_directory: str | Path, expected_identity: dict[str, Any] | None = None
):
    design = load_approved_design(
        Path(run_directory).resolve() / RUN_ARTIFACT_DIRECTORY_NAME
    )
    identity = artifact_identity(design)
    if expected_identity is not None and identity != expected_identity:
        raise LLMRuntimeError(
            "run LLM artifact differs from the recorded approved design"
        )
    return design


def runtime_constants_metadata(
    design: ApprovedDesign,
    *,
    environment_width_m: float,
    environment_height_m: float,
    episode_seconds: float,
    task_deadlines_seconds: dict[str, float],
) -> dict[str, Any]:
    """Overlay only environment-varying values on the approved interface."""

    metadata = copy.deepcopy(design.constants_metadata)
    updates = {
        "environment_width_m": float(environment_width_m),
        "environment_height_m": float(environment_height_m),
        "episode_seconds": float(episode_seconds),
        "vs_deadline_seconds": float(task_deadlines_seconds["FOV"]),
        "com_deadline_seconds": float(task_deadlines_seconds["COM"]),
    }
    missing = sorted(set(updates).difference(metadata))
    if missing:
        raise LLMRuntimeError(
            f"approved artifact lacks dynamic environment constants: {missing}"
        )
    for name, value in updates.items():
        if not math.isfinite(value) or value <= 0.0:
            raise LLMRuntimeError(f"runtime constant {name} must be finite positive")
        metadata[name]["value"] = value
        metadata[name]["runtime_source"] = "current training/evaluation environment"
    return metadata


def build_online_obs(
    state,
    movement_mask,
    snapshot,
    *,
    interface_version: str = OBS_INTERFACE_VERSION,
) -> dict[str, np.ndarray]:
    if set(snapshot) != set(SNAPSHOT_FIELD_SPECS):
        raise LLMRuntimeError("online snapshot field set is incompatible")
    obs = {
        "state": np.asarray(state, dtype=np.float32).copy(),
        "movement_mask": np.asarray(movement_mask, dtype=bool).copy(),
    }
    if obs["state"].ndim != 1 or obs["movement_mask"].ndim != 1:
        raise LLMRuntimeError("online state and movement mask must be one-dimensional")
    for name, spec in SNAPSHOT_FIELD_SPECS.items():
        value = np.asarray(snapshot[name])
        if value.shape != tuple(spec["shape"]) or value.dtype != spec["dtype"]:
            raise LLMRuntimeError(
                f"online snapshot {name} has incompatible shape or dtype"
            )
        obs[name] = value.copy()
    if interface_version == OBS_INTERFACE_VERSION:
        obs.update(derive_named_state_fields(obs["state"]))
    if set(obs) != set(obs_keys_for_interface(interface_version)):
        raise LLMRuntimeError("online observation field set is incompatible")
    return obs


def _worker_main(connection, candidate, constants, observation_interface_version):
    try:
        namespace = {"__builtins__": SAFE_BUILTINS, "np": np}
        exec(
            compile(candidate["code"], "<approved-candidate-runtime>", "exec"),
            namespace,
            namespace,
        )
        extra_function = namespace["compute_extra_state"]
        feature_count = len(candidate["features"])
        connection.send({"status": "ready"})
        while True:
            request = connection.recv()
            if request is None:
                return
            obs = request["obs"]
            if set(obs) != set(obs_keys_for_interface(observation_interface_version)):
                raise ValueError("runtime exposes an unexpected observation field set")
            for value in obs.values():
                value.setflags(write=False)
            extra = _run_function(
                extra_function, obs, constants, feature_count, "compute_extra_state"
            )
            _validate_output_ranges(extra, candidate, "online observation")
            if request.get("mode") == "state":
                connection.send({"status": "ok", "extra": extra})
                continue
            reward = float(feature_reward(extra, candidate))
            connection.send(
                {"status": "ok", "extra": extra, "reward": reward}
            )
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(
                {"status": "error", "type": type(exc).__name__, "message": str(exc)}
            )
        except BaseException:
            pass
    finally:
        connection.close()


@dataclass
class ApprovedDesignRuntime:
    design: ApprovedDesign
    constants_metadata: dict[str, Any]
    timeout: float = 60.0

    def __post_init__(self):
        if not math.isfinite(float(self.timeout)) or float(self.timeout) <= 0.0:
            raise ValueError("LLM worker timeout must be finite positive")
        self.identity = artifact_identity(self.design)
        self.observation_interface_version = self.identity[
            "observation_interface_version"
        ]
        self.beta = float(self.identity["beta"])
        self.constants = runtime_constants(self.constants_metadata)
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_worker_main,
            args=(
                child,
                self.design.candidate,
                self.constants,
                self.observation_interface_version,
            ),
            daemon=True,
        )
        self._process.start()
        child.close()
        response = self._receive("initialization")
        if response.get("status") != "ready":
            self.close()
            raise LLMRuntimeError(f"LLM runtime initialization failed: {response}")

    @property
    def feature_count(self):
        return int(self.identity["feature_count"])

    def _receive(self, phase):
        if not self._connection.poll(float(self.timeout)):
            self.close()
            raise LLMRuntimeError(f"LLM candidate worker timed out during {phase}")
        try:
            response = self._connection.recv()
        except EOFError as exc:
            self.close()
            raise LLMRuntimeError(
                f"LLM candidate worker exited during {phase}"
            ) from exc
        return response

    def evaluate(self, obs):
        if not self._process.is_alive():
            raise LLMRuntimeError("LLM candidate worker is not running")
        self._connection.send({"obs": obs, "mode": "all"})
        response = self._receive("online evaluation")
        if response.get("status") != "ok":
            raise LLMRuntimeError(
                "LLM candidate evaluation failed: "
                f"{response.get('type')}: {response.get('message')}"
            )
        return (
            np.asarray(response["extra"], dtype=np.float32),
            float(response["reward"]),
        )

    def evaluate_state(self, obs):
        if not self._process.is_alive():
            raise LLMRuntimeError("LLM candidate worker is not running")
        self._connection.send({"obs": obs, "mode": "state"})
        response = self._receive("online next-state evaluation")
        if response.get("status") != "ok":
            raise LLMRuntimeError(
                "LLM candidate state evaluation failed: "
                f"{response.get('type')}: {response.get('message')}"
            )
        return np.asarray(response["extra"], dtype=np.float32)

    def close(self):
        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if connection is not None:
            connection.close()
