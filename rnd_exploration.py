"""Random Network Distillation support for the isolated TD3 collector method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import torch
from torch import nn


RND_METHOD_ID = "td3_dinkelbach_rnd_collect"
RND_CHECKPOINT_FORMAT_VERSION = "uav-hrl-rnd-v1"


@dataclass(frozen=True)
class RNDConfig:
    eta: float = 0.5
    gamma_rnd: float = 0.99
    learning_rate: float = 1e-4
    epsilon: float = 1e-8
    predictor_batch_size: int = 64
    initialization_seed: int = 2026081701
    observation_clip: float = 5.0
    minimum_warmup_observations: int = 2

    def __post_init__(self):
        numeric_positive = {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "observation_clip": self.observation_clip,
        }
        if not np.isfinite(self.eta) or self.eta < 0.0:
            raise ValueError("RND eta must be finite and non-negative")
        if not np.isfinite(self.gamma_rnd) or not 0.0 <= self.gamma_rnd <= 1.0:
            raise ValueError("RND gamma must be finite and in [0, 1]")
        for name, value in numeric_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"RND {name} must be finite and positive")
        if int(self.predictor_batch_size) <= 0:
            raise ValueError("RND predictor batch size must be positive")
        if int(self.minimum_warmup_observations) < 2:
            raise ValueError("RND requires at least two warmup observations")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("RND configuration must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unknown RND configuration fields: {sorted(unknown)}")
        return cls(**value)


class RunningMeanVariance:
    """Float64 population moments updated without mutating input arrays."""

    def __init__(self, shape=()):
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        if shape == ():
            self.shape = ()
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)

    @property
    def variance(self):
        if self.count < 2:
            return np.ones(self.shape, dtype=np.float64)
        variance = self.m2 / float(self.count)
        return np.maximum(variance, 0.0)

    def update(self, values):
        array = np.asarray(values, dtype=np.float64)
        if self.shape:
            array = np.reshape(array, (-1, *self.shape))
        else:
            array = np.reshape(array, (-1,))
        if array.shape[0] == 0:
            return
        if not np.isfinite(array).all():
            raise ValueError("running-statistics input must be finite")
        batch_count = int(array.shape[0])
        batch_mean = array.mean(axis=0, dtype=np.float64)
        centered = array - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = np.array(batch_mean, dtype=np.float64, copy=True)
            self.m2 = np.array(batch_m2, dtype=np.float64, copy=True)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = np.asarray(
            self.mean + delta * (batch_count / float(total)), dtype=np.float64
        ).copy()
        self.m2 = np.asarray(
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / float(total)),
            dtype=np.float64,
        ).copy()
        self.count = total

    def state_dict(self):
        return {
            "shape": self.shape,
            "count": int(self.count),
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
        }

    def load_state_dict(self, state):
        shape = tuple(state["shape"])
        if shape != self.shape:
            raise RuntimeError(
                f"running-statistics shape mismatch: checkpoint={shape}, current={self.shape}"
            )
        count = int(state["count"])
        mean = np.asarray(state["mean"], dtype=np.float64)
        m2 = np.asarray(state["m2"], dtype=np.float64)
        if count < 0 or mean.shape != self.shape or m2.shape != self.shape:
            raise RuntimeError("running-statistics checkpoint is invalid")
        if not np.isfinite(mean).all() or not np.isfinite(m2).all() or (m2 < 0).any():
            raise RuntimeError("running-statistics checkpoint contains invalid values")
        self.count = count
        self.mean = mean.copy()
        self.m2 = m2.copy()


class _TargetMLP(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, value):
        return self.network(value)


class _PredictorMLP(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, value):
        return self.network(value)


class RNDExplorer:
    """Causal RND statistics plus a once-only predictor observation queue."""

    def __init__(self, state_dim, config=None, device=None):
        self.state_dim = int(state_dim)
        if self.state_dim <= 0:
            raise ValueError("RND state dimension must be positive")
        self.config = config if isinstance(config, RNDConfig) else RNDConfig.from_dict(config)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index or torch.cuda.current_device()]
        # fork_rng restores global CPU/CUDA RNG state, so constructing RND cannot
        # perturb any baseline actor, critic, routing, or exploration stream.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config.initialization_seed))
            if devices:
                torch.cuda.manual_seed_all(int(self.config.initialization_seed))
            self.target = _TargetMLP(self.state_dim).to(self.device)
            self.predictor = _PredictorMLP(self.state_dim).to(self.device)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=float(self.config.learning_rate)
        )
        self.observation_stats = RunningMeanVariance((self.state_dim,))
        self.q_stats = RunningMeanVariance(())
        self.q_accumulator = 0.0
        self.pending_observations = []
        self.observation_count = 0
        self.predictor_update_count = 0
        self.last_predictor_loss = None

    def _normalization_snapshot(self):
        return (
            self.observation_stats.mean.copy(),
            self.observation_stats.variance.copy(),
        )

    def normalize(self, states, snapshot=None):
        tensor = states if torch.is_tensor(states) else torch.as_tensor(states)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if tensor.shape[-1] != self.state_dim:
            raise ValueError(
                f"RND state shape mismatch: expected trailing {self.state_dim}, got {tuple(tensor.shape)}"
            )
        mean, variance = snapshot or self._normalization_snapshot()
        mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype, device=self.device)
        variance_tensor = torch.as_tensor(variance, dtype=tensor.dtype, device=self.device)
        normalized = (tensor - mean_tensor) / torch.sqrt(
            variance_tensor + float(self.config.epsilon)
        )
        return normalized.clamp(
            -float(self.config.observation_clip), float(self.config.observation_clip)
        )

    def raw_error(self, states, snapshot=None):
        normalized = self.normalize(states, snapshot=snapshot)
        with torch.no_grad():
            prediction = self.predictor(normalized)
            target = self.target(normalized)
            return torch.mean((prediction - target) ** 2, dim=-1, keepdim=True)

    def _historical_q_variance(self):
        # Finite initialization is used only until two chronological q samples
        # exist. No clipping is applied once an empirical variance is available.
        if self.q_stats.count < 2:
            return 1.0
        return float(np.asarray(self.q_stats.variance))

    def observe(self, next_state, *, queue_for_predictor):
        state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError("RND observation has an invalid shape or value")
        self.observation_stats.update(state.reshape(1, -1))
        snapshot = self._normalization_snapshot()
        error = float(self.raw_error(state.reshape(1, -1), snapshot=snapshot).item())
        previous_variance = self._historical_q_variance()
        bonus = error / np.sqrt(previous_variance + float(self.config.epsilon))
        q_value = float(self.config.gamma_rnd) * self.q_accumulator + error
        self.q_accumulator = q_value
        self.q_stats.update(np.asarray([q_value], dtype=np.float64))
        self.observation_count += 1
        if queue_for_predictor:
            self.pending_observations.append(state.copy())
        return {
            "raw_error": error,
            "normalized_bonus": float(bonus),
            "weighted_bonus": float(self.config.eta * bonus),
            "training_reward_source": "online_pre_predictor_update",
            "q_value": q_value,
            "q_variance_previous": previous_variance,
        }

    def intrinsic_reward(self, next_states):
        """Re-score shuffled replay states without changing any RND statistics."""
        errors = self.raw_error(next_states)
        denominator = np.sqrt(
            self._historical_q_variance() + float(self.config.epsilon)
        )
        return errors / float(denominator)

    def diagnostic(self, next_state):
        error = float(self.raw_error(np.asarray(next_state).reshape(1, -1)).item())
        bonus = error / np.sqrt(
            self._historical_q_variance() + float(self.config.epsilon)
        )
        return {
            "raw_error": error,
            "normalized_bonus": float(bonus),
            "weighted_bonus": float(self.config.eta * bonus),
            "source": "checkpoint_frozen_diagnostic",
        }

    def train_pending(self, *, force=False):
        batch_size = int(self.config.predictor_batch_size)
        losses = []
        while len(self.pending_observations) >= batch_size or (
            force and self.pending_observations
        ):
            size = batch_size if len(self.pending_observations) >= batch_size else len(self.pending_observations)
            values = np.stack(self.pending_observations[:size], axis=0)
            del self.pending_observations[:size]
            snapshot = self._normalization_snapshot()
            normalized = self.normalize(values, snapshot=snapshot)
            with torch.no_grad():
                target = self.target(normalized)
            prediction = self.predictor(normalized)
            loss = torch.mean((prediction - target) ** 2)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.predictor_update_count += 1
            self.last_predictor_loss = float(loss.detach().cpu())
            losses.append(self.last_predictor_loss)
        return losses

    def metadata(self):
        return {
            "format_version": RND_CHECKPOINT_FORMAT_VERSION,
            "state_dim": self.state_dim,
            "configuration": self.config.to_dict(),
            "error_definition": "mean squared error over 64 output dimensions",
            "observation_normalization": "per-dimension running mean/variance, float64, clip [-5,5]",
            "return_normalization": "e_t / sqrt(previous chronological q variance + epsilon)",
            "q_recurrence": "q_t = gamma_rnd * q_(t-1) + e_t; continues across ordinary episode boundaries",
            "observation_count": int(self.observation_count),
            "predictor_update_count": int(self.predictor_update_count),
            "pending_predictor_observations": len(self.pending_observations),
        }

    def state_dict(self, *, full):
        value = {
            "format_version": RND_CHECKPOINT_FORMAT_VERSION,
            "state_dim": self.state_dim,
            "configuration": self.config.to_dict(),
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "observation_stats": self.observation_stats.state_dict(),
            "q_stats": self.q_stats.state_dict(),
            "q_accumulator": float(self.q_accumulator),
            "observation_count": int(self.observation_count),
            "predictor_update_count": int(self.predictor_update_count),
            "last_predictor_loss": self.last_predictor_loss,
            "checkpoint_purpose": "full_resume" if full else "inference_and_frozen_diagnostics",
        }
        if full:
            value.update(
                optimizer=self.optimizer.state_dict(),
                pending_observations=(
                    np.stack(self.pending_observations, axis=0)
                    if self.pending_observations
                    else np.empty((0, self.state_dim), dtype=np.float32)
                ),
            )
        return value

    def load_state_dict(self, state, *, require_full):
        if not isinstance(state, dict) or state.get("format_version") != RND_CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError("RND checkpoint is missing or has an incompatible format")
        if int(state.get("state_dim", -1)) != self.state_dim:
            raise RuntimeError("RND checkpoint state dimension is incompatible")
        if RNDConfig.from_dict(state.get("configuration")) != self.config:
            raise RuntimeError("RND checkpoint configuration is incompatible")
        required = {"target", "predictor", "observation_stats", "q_stats", "q_accumulator"}
        if not required.issubset(state):
            raise RuntimeError("RND checkpoint is incomplete")
        if require_full and not {"optimizer", "pending_observations"}.issubset(state):
            raise RuntimeError("full-resume RND checkpoint is incomplete")
        self.target.load_state_dict(state["target"], strict=True)
        self.predictor.load_state_dict(state["predictor"], strict=True)
        self.observation_stats.load_state_dict(state["observation_stats"])
        self.q_stats.load_state_dict(state["q_stats"])
        self.q_accumulator = float(state["q_accumulator"])
        if not np.isfinite(self.q_accumulator):
            raise RuntimeError("RND q accumulator is invalid")
        self.observation_count = int(state.get("observation_count", 0))
        self.predictor_update_count = int(state.get("predictor_update_count", 0))
        self.last_predictor_loss = state.get("last_predictor_loss")
        self.pending_observations = []
        if require_full:
            self.optimizer.load_state_dict(state["optimizer"])
            pending = np.asarray(state["pending_observations"], dtype=np.float32)
            if pending.ndim != 2 or pending.shape[1] != self.state_dim:
                raise RuntimeError("RND pending predictor batch has an incompatible shape")
            self.pending_observations = [row.copy() for row in pending]
        self.target.requires_grad_(False)
        self.target.eval()

    def fingerprint(self):
        digest = hashlib.sha256()

        def update(value):
            if torch.is_tensor(value):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(b"tensor")
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes(order="C"))
            elif isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest.update(b"ndarray")
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes(order="C"))
            elif isinstance(value, dict):
                digest.update(b"dict")
                for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                    update(key)
                    update(value[key])
            elif isinstance(value, (list, tuple)):
                digest.update(type(value).__name__.encode("ascii"))
                for item in value:
                    update(item)
            elif value is None:
                digest.update(b"none")
            elif isinstance(value, (str, int, float, bool, np.generic)):
                digest.update(type(value).__name__.encode("ascii"))
                digest.update(repr(value.item() if isinstance(value, np.generic) else value).encode("utf-8"))
            else:
                raise TypeError(
                    f"unsupported RND fingerprint value: {type(value).__name__}"
                )

        # Evaluation invariants concern the frozen networks and normalization
        # state. Optimizer serialization may canonicalize parameter-group
        # internals on load even when its numerical state is identical.
        update(self.state_dict(full=False))
        return digest.hexdigest()


def is_rnd_method(method_spec):
    return getattr(method_spec, "method_id", None) == RND_METHOD_ID
