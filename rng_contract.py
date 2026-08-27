"""Authoritative named-RNG contract for executable UAV-HRL operations."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import MappingProxyType

import numpy as np
import torch


RNG_CONTRACT_VERSION = "named-subsystem-streams-v1"

# These identifiers are persistent compatibility data.  Never renumber an
# existing stream: append a new identifier when a new stochastic subsystem is
# introduced.
RNG_STREAM_IDS = MappingProxyType(
    {
        "scenario_generation": 1,
        "environment_dynamics": 2,
        "evaluation_environment": 3,
        "movement_actor_init": 10,
        "movement_critic1_init": 11,
        "movement_critic2_init": 12,
        "movement_exploration": 13,
        "movement_replay_sampling": 14,
        "td3_target_policy_noise": 15,
        "safe_ddqn_network_init": 20,
        "safe_ddqn_cost_network_init": 21,
        "safe_ddqn_exploration": 22,
        "safe_ddqn_replay_sampling": 23,
        "standard_dqn_network_init": 30,
        "standard_dqn_exploration": 31,
        "standard_dqn_replay_sampling": 32,
        "random_assignment": 40,
        "random_movement": 41,
        "random_routing": 42,
        "evaluation_random_assignment": 50,
        "evaluation_random_movement": 51,
        "evaluation_random_routing": 52,
    }
)


def named_seed(master_seed, stream_name):
    """Derive one stable 63-bit seed without Python's randomized hash()."""

    try:
        stream_id = RNG_STREAM_IDS[str(stream_name)]
    except KeyError as exc:
        raise KeyError(f"unknown RNG stream: {stream_name}") from exc
    words = np.random.SeedSequence(
        [int(master_seed) & 0xFFFFFFFF, int(stream_id)]
    ).generate_state(2, dtype=np.uint32)
    return int((int(words[0]) << 32) | int(words[1])) & ((1 << 63) - 1)


@contextmanager
def isolated_torch_initialization(seed):
    """Seed module construction while restoring all process-global torch RNGs."""

    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def build_torch_module(factory, master_seed, stream_name, device):
    """Construct a module from a named init stream with no global RNG side effect."""

    with isolated_torch_initialization(named_seed(master_seed, stream_name)):
        return factory().to(device)


class NamedRNGStreams:
    """Lazily materialized, checkpointable NumPy and torch generators."""

    def __init__(self, master_seed):
        self.master_seed = int(master_seed)
        self._numpy = {}
        self._torch = {}

    def numpy(self, stream_name):
        stream_name = str(stream_name)
        if stream_name not in RNG_STREAM_IDS:
            raise KeyError(f"unknown RNG stream: {stream_name}")
        if stream_name not in self._numpy:
            self._numpy[stream_name] = np.random.default_rng(
                named_seed(self.master_seed, stream_name)
            )
        return self._numpy[stream_name]

    def torch(self, stream_name, device="cpu"):
        stream_name = str(stream_name)
        device = torch.device(device)
        key = (stream_name, str(device))
        if stream_name not in RNG_STREAM_IDS:
            raise KeyError(f"unknown RNG stream: {stream_name}")
        if key not in self._torch:
            generator = torch.Generator(device=device)
            generator.manual_seed(named_seed(self.master_seed, stream_name))
            self._torch[key] = generator
        return self._torch[key]

    def state_dict(self):
        return {
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "master_seed": self.master_seed,
            "numpy": {
                name: copy.deepcopy(generator.bit_generator.state)
                for name, generator in sorted(self._numpy.items())
            },
            "torch": {
                f"{name}|{device}": generator.get_state().clone().cpu()
                for (name, device), generator in sorted(self._torch.items())
            },
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("named RNG state must be an object")
        if state.get("rng_contract_version") != RNG_CONTRACT_VERSION:
            raise RuntimeError("named RNG state contract is incompatible")
        if int(state.get("master_seed")) != self.master_seed:
            raise RuntimeError("named RNG state master seed is incompatible")
        for name, generator_state in state.get("numpy", {}).items():
            self.numpy(name).bit_generator.state = copy.deepcopy(generator_state)
        for encoded, generator_state in state.get("torch", {}).items():
            name, device = encoded.rsplit("|", 1)
            self.torch(name, device=device).set_state(generator_state.cpu())

    def metadata(self):
        return {
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "master_seed": self.master_seed,
            "stream_derivation": "numpy.SeedSequence([master_seed, fixed_stream_id])",
            "subsystem_stream_ids": dict(RNG_STREAM_IDS),
            "training_evaluation_separation": {
                "environment": ["environment_dynamics", "evaluation_environment"],
                "random_assignment": [
                    "random_assignment",
                    "evaluation_random_assignment",
                ],
                "random_movement": [
                    "random_movement",
                    "evaluation_random_movement",
                ],
                "random_routing": [
                    "random_routing",
                    "evaluation_random_routing",
                ],
            },
            "global_rng_policy": "not used by formal executable operations",
        }
