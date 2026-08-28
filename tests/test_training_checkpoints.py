import contextlib
from dataclasses import asdict
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from DDQN import DDQN
from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from experiment_config import (
    EXPLORATION_SCHEDULE_VERSION,
    GROUND_STATION_POSITION_M,
    INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION,
    MethodSpec,
    MOVEMENT_EXPLORATION_DECAY_EPISODES,
    ROUTING_EPSILON_DECAY_EPISODES,
    SR_ROUTE_LIFECYCLE_VERSION,
    TASK_POTENTIAL_CONTRACT_VERSION,
    UAV_INITIAL_LAYOUT_VERSION,
    effective_training_config,
    task_potential_contract_metadata,
)
from HRL_task_aware import (
    _seed_training_rng,
    _uses_warmup_random_action,
    formal_training_config,
    parse_training_config,
)
from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from td3 import TD3
from routing_lifecycle import RoutingLearnerLifecycle
from scenario_manifest import extend_training_manifest, generate_manifest
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    FULL_RESUME_LOGGING_SCHEMA_VERSION,
    PRE_CONTINUOUS_GATEWAY_PROJECTION_CHECKPOINT_SCHEMA_VERSION,
    PRE_UNIFIED_400M_COMMUNICATION_CHECKPOINT_SCHEMA_VERSION,
    PRE_ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION,
    ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION,
    load_full_resume_checkpoint,
    load_model_checkpoint,
    inspect_full_resume_checkpoint,
    save_full_resume_checkpoint,
    save_model_checkpoint,
)
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint
from fov_ema_fixtures import initialized_fov_ema_state
from channel_fixtures import initialized_channel_lifecycle_state
from routing_transition_fixtures import routing_transition_checkpoint_fixture


ROUTING_STATE_DIM = 90
ROUTING_ACTION_DIM = 11


def _model_provenance_fixture(episode_count):
    method = MethodSpec()
    config = formal_training_config(int(episode_count), random_seed=1)
    formal_config = effective_training_config(config, method)
    return {
        "method_id": method.method_id,
        "method_spec": method.to_dict(),
        "method_spec_fingerprint": method.fingerprint,
        "training_seed": 1,
        "git_sha": "fixture-training-sha",
        "formal_config": formal_config,
    }


def _empty_routing_lifecycle():
    return RoutingLearnerLifecycle().state_dict()


def _lifecycle_training_state(
    episode_count,
    lambda_after=0.0,
    *,
    routing_slots=0,
    routing_updates=0,
    routing_warmup=1000,
    movement_post_warmup=0,
):
    after_log = [0.0] * int(episode_count)
    if after_log:
        after_log[-1] = float(lambda_after)
    lifecycle = RoutingLearnerLifecycle(
        warmup_transitions=int(routing_warmup),
        global_slot_count=int(routing_slots),
        optimizer_update_count=int(routing_updates),
        target_update_count=int(routing_updates),
        epsilon_decay_start_slot=(1 if routing_updates else None),
        last_optimizer_update_slot=(
            int(routing_slots) if routing_updates else None
        ),
    ).state_dict()
    return {
        "lambda_cost_used_log": [0.0] * int(episode_count),
        "lambda_cost_after_episode_log": after_log,
        "fov_ema_state": initialized_fov_ema_state(
            marker=f"fixture-episode={int(episode_count) - 1}"
        ),
        "sr_route_state": {
            "lifecycle_version": SR_ROUTE_LIFECYCLE_VERSION,
            "teams": [],
            "trajectory": {},
            "checkpoint_scope": "episode_boundary_terminal_snapshot",
            "mid_episode_checkpoint_supported": False,
        },
        "channel_lifecycle_state": initialized_channel_lifecycle_state(),
        **routing_transition_checkpoint_fixture(),
        "routing_lifecycle_state": lifecycle,
        "routing_epsilon_decay_start_slot": lifecycle[
            "routing_epsilon_decay_start_slot"
        ],
        "exploration_schedule_version": EXPLORATION_SCHEDULE_VERSION,
        "movement_exploration_decay_episodes": (
            MOVEMENT_EXPLORATION_DECAY_EPISODES
        ),
        "routing_epsilon_decay_episodes": ROUTING_EPSILON_DECAY_EPISODES,
        "resolved_movement_decay_steps": 60000,
        "resolved_routing_decay_slots": 240000,
        "movement_post_warmup_transition_count": int(movement_post_warmup),
        "movement_noise_start": 0.20,
        "movement_noise_end": 0.05,
        "routing_epsilon_start": 1.0,
        "routing_epsilon_end": 0.05,
    }


def _assert_module_equal(testcase, expected, actual):
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    testcase.assertEqual(expected_state.keys(), actual_state.keys())
    for key in expected_state:
        testcase.assertTrue(torch.equal(expected_state[key], actual_state[key]), key)


class TrainingSeedTest(unittest.TestCase):
    def _models(self):
        return (
            TD3(
                MOVEMENT_STATE_DIM,
                JOINT_ACTION_DIM,
                max_action=1.0,
                gamma=1.0,
                policy_delay=2,
            ),
            DDQN(ROUTING_STATE_DIM, ROUTING_ACTION_DIM),
        )

    def test_same_seed_reproduces_all_td3_and_ddqn_networks(self):
        _seed_training_rng(20260817)
        first_td3, first_ddqn = self._models()
        _seed_training_rng(20260817)
        second_td3, second_ddqn = self._models()

        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_1_target",
            "critic_2",
            "critic_2_target",
        ):
            _assert_module_equal(
                self, getattr(first_td3, name), getattr(second_td3, name)
            )
        for name in (
            "q_network",
            "target_q_network",
            "cost_network",
            "target_cost_network",
        ):
            _assert_module_equal(
                self, getattr(first_ddqn, name), getattr(second_ddqn, name)
            )

    def test_same_seed_reproduces_named_stream_without_mutating_globals(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        first = _seed_training_rng(20260817)
        second = _seed_training_rng(20260817)

        np.testing.assert_array_equal(
            first.numpy("movement_exploration").normal(size=4),
            second.numpy("movement_exploration").normal(size=4),
        )
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(np.random.get_state()[0], numpy_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))


class FullResumeCheckpointTest(unittest.TestCase):
    def _components(self):
        td3 = TD3(
            MOVEMENT_STATE_DIM,
            JOINT_ACTION_DIM,
            max_action=1.0,
            gamma=1.0,
            policy_delay=2,
        )
        ddqn = DDQN(ROUTING_STATE_DIM, ROUTING_ACTION_DIM, hidden_dim=16)
        joint = ReplayBufferJoint(
            MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=8
        )
        routing = ReplayBufferDiscrete(
            ROUTING_STATE_DIM,
            ROUTING_ACTION_DIM,
            max_size=8,
            n_step=1,
            gamma=0.99,
        )
        return td3, ddqn, joint, routing

    def _preflight_checkpoint(self, root, *, planned=1, completed=1):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 77, "c_ref_com": 10.0}
        formal_config = asdict(
            formal_training_config(planned, random_seed=77)
        )
        dinkelbach_state = DinkelbachBlockState.from_config(formal_config)
        used_log = []
        after_log = []
        for _ in range(completed):
            event = dinkelbach_state.record_episode(1.0, 2.0)
            used_log.append(event["dinkelbach_lambda_used"])
            after_log.append(event["dinkelbach_lambda_after_episode"])
        training_state = {
            "completed_episode_index": completed - 1,
            "next_episode_index": completed,
            "full_resume_logging_schema_version": FULL_RESUME_LOGGING_SCHEMA_VERSION,
            "reward_log": [0.0] * completed,
            "delivered_log": [1.0] * completed,
            "energy_log": [2.0] * completed,
            "lambda_used_log": used_log,
            "lambda_after_episode_log": after_log,
            "total_joint_transitions": 0,
            **_lifecycle_training_state(completed),
            **dinkelbach_state.training_state(),
        }
        manifest = generate_manifest("train", 77, planned)
        manifest.save(Path(root) / "scenario_manifest.json")
        checkpoint_dir = Path(root) / "checkpoints" / "full" / f"ep_{completed:04d}"
        save_full_resume_checkpoint(
            checkpoint_dir,
            episode=completed - 1,
            td3=td3,
            ddqn=ddqn,
            joint_replay=joint,
            routing_replay=routing,
            training_state=training_state,
            formal_config=formal_config,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            calibration=calibration,
            experiment_metadata={
                "training_seed": 77,
                "git_sha": "fixture-sha",
                "manifest_hash": manifest.content_hash,
                "formal_config": formal_config,
                **dinkelbach_config_metadata(formal_config),
                "lambda_ee": dinkelbach_state.current_lambda,
                "dinkelbach_state": dinkelbach_state.training_state(),
            },
        )
        return {
            "checkpoint_dir": checkpoint_dir,
            "td3": td3,
            "ddqn": ddqn,
            "joint": joint,
            "routing": routing,
            "calibration": calibration,
            "formal_config": formal_config,
            "manifest": manifest,
        }

    def _inspect_preflight_fixture(self, fixture, current_config, current_manifest):
        return inspect_full_resume_checkpoint(
            fixture["checkpoint_dir"],
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            td3_gamma=1.0,
            ddqn_gamma=0.99,
            calibration=fixture["calibration"],
            expected_formal_config=current_config,
            current_training_manifest=current_manifest,
            require_episode_directory=True,
        )

    def test_full_resume_metadata_incompatibilities_precede_torch_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._preflight_checkpoint(temp_dir, planned=1, completed=1)
            extended, _ = extend_training_manifest(fixture["manifest"], 3)
            extended_config = dict(fixture["formal_config"], total_episodes=3)

            wrong_field = dict(extended_config, batch_size=999)
            with mock.patch("training_checkpoint.torch.load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "batch_size"):
                    self._inspect_preflight_fixture(
                        fixture, wrong_field, extended
                    )
                torch_load.assert_not_called()

            wrong_manifest = generate_manifest("train", 78, 3)
            with mock.patch("training_checkpoint.torch.load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "manifest"):
                    self._inspect_preflight_fixture(
                        fixture, extended_config, wrong_manifest
                    )
                torch_load.assert_not_called()

            with mock.patch(
                "training_checkpoint.torch.load",
                side_effect=RuntimeError("payload-load-marker"),
            ) as torch_load:
                with self.assertRaisesRegex(RuntimeError, "payload-load-marker"):
                    self._inspect_preflight_fixture(
                        fixture, extended_config, extended
                    )
                torch_load.assert_called_once()

    def test_target_below_completed_episode_precedes_torch_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._preflight_checkpoint(temp_dir, planned=3, completed=2)
            shorter = generate_manifest("train", 77, 1)
            shorter_config = dict(fixture["formal_config"], total_episodes=1)
            with mock.patch("training_checkpoint.torch.load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "current run horizon"):
                    self._inspect_preflight_fixture(
                        fixture, shorter_config, shorter
                    )
                torch_load.assert_not_called()

    def test_payload_formal_config_conflict_does_not_mutate_live_agents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._preflight_checkpoint(temp_dir, planned=1, completed=1)
            payload_path = fixture["checkpoint_dir"] / "training_state.pt"
            payload = torch.load(
                payload_path, map_location="cpu", weights_only=False
            )
            payload["formal_config"]["batch_size"] = 999
            torch.save(payload, payload_path)

            live_td3, live_ddqn, live_joint, live_routing = self._components()
            actor_before = {
                key: value.detach().clone()
                for key, value in live_td3.actor.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "batch_size"):
                load_full_resume_checkpoint(
                    fixture["checkpoint_dir"],
                    td3=live_td3,
                    ddqn=live_ddqn,
                    joint_replay=live_joint,
                    routing_replay=live_routing,
                    movement_state_dim=MOVEMENT_STATE_DIM,
                    joint_action_dim=JOINT_ACTION_DIM,
                    routing_state_dim=ROUTING_STATE_DIM,
                    calibration=fixture["calibration"],
                    expected_formal_config=fixture["formal_config"],
                    current_training_manifest=fixture["manifest"],
                )
            for key, expected in actor_before.items():
                self.assertTrue(
                    torch.equal(live_td3.actor.state_dict()[key], expected)
                )

    def _populate_and_train(self, td3, ddqn, joint, routing):
        state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
        state[0] = 1.0
        state[428] = 0.5
        action = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        for index, done in enumerate((False, True)):
            joint.add(
                state,
                action,
                state,
                done=done,
                delivered_mbits=1.0 + index,
                total_mobility_energy=2.0,
                phi_search_t=0.1,
                phi_search_t1=0.2,
                phi_vs_t=0.3,
                phi_vs_t1=0.4,
                phi_com_t=0.5,
                phi_com_t1=0.6,
            )
        td3.update_joint(joint, current_lambda=0.1, batch_size=1)
        td3.update_joint(joint, current_lambda=0.1, batch_size=1)

        routing_state = np.zeros(ROUTING_STATE_DIM, dtype=np.float32)
        routing_state[0] = 1.0
        routing_state[24:41] = 1.0
        routing_state[24] = 0.0
        routing.add(
            routing_state,
            1,
            routing_state,
            reward=3.0,
            cost=0.25,
            done=False,
            tag_gt=4,
            transition_id=100,
        )
        routing.add(
            routing_state,
            2,
            routing_state,
            reward=4.0,
            cost=0.5,
            done=True,
            tag_gt=4,
            transition_id=101,
        )
        ddqn.train(routing, batch_size=1)
        ddqn.update_target()

    def test_full_resume_round_trip_restores_exact_training_state_and_rng(self):
        td3, ddqn, joint, routing = self._components()
        self._populate_and_train(td3, ddqn, joint, routing)
        ddqn.update_cost_multiplier(3.0, 20)
        calibration = {"seed": 20260817, "c_ref_com": 32.5, "unit": "Mbps"}
        training_state = {
            "completed_episode_index": 6,
            "next_episode_index": 7,
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [float(index) for index in range(7)],
            "delivered_log": [1.0] * 7,
            "energy_log": [2.0] * 7,
            "lambda_used_log": [0.0] * 7,
            "lambda_after_episode_log": [0.0] * 7,
            "total_joint_transitions": 37,
            "global_routing_slot": 148,
            "td3_post_warmup_transition": 0,
            "ddqn_schedule_slot": 148,
            "td3_noise_log": [0.2],
            "routing_epsilon_log": [1.0, 0.99],
            **_lifecycle_training_state(
                7,
                lambda_after=ddqn.lambda_cost,
                routing_slots=148,
                routing_updates=1,
                routing_warmup=1,
            ),
            **routing_transition_checkpoint_fixture(102),
        }
        formal_config = asdict(
            formal_training_config(
                100, random_seed=123, routing_warmup_transitions=1
            )
        )
        dinkelbach_state = DinkelbachBlockState.from_config(formal_config)
        for _ in range(7):
            dinkelbach_state.record_episode(1.0, 2.0)
        training_state.update(dinkelbach_state.training_state())

        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(789)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "full"
            save_full_resume_checkpoint(
                checkpoint_dir,
                episode=6,
                td3=td3,
                ddqn=ddqn,
                joint_replay=joint,
                routing_replay=routing,
                training_state=training_state,
                formal_config=formal_config,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                experiment_metadata={
                    "formal_config": formal_config,
                    **dinkelbach_config_metadata(formal_config),
                    "lambda_ee": dinkelbach_state.current_lambda,
                    "dinkelbach_state": dinkelbach_state.training_state(),
                },
            )
            with np.load(checkpoint_dir / "joint_replay.npz") as saved_joint:
                self.assertEqual(saved_joint["state"].shape[0], joint.size)
            with np.load(checkpoint_dir / "routing_replay.npz") as saved_routing:
                self.assertEqual(saved_routing["state"].shape[0], routing.size)

            expected_python = random.random()
            expected_numpy = np.random.random()
            probe_state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
            expected_action = td3.select_action(probe_state, noise_std=0.1)
            expected_batch = joint.sample(1, current_lambda=0.123, gamma=1.0)
            expected_torch = torch.rand(3)

            restored_td3, restored_ddqn, restored_joint, restored_routing = (
                self._components()
            )
            restored = load_full_resume_checkpoint(
                checkpoint_dir,
                td3=restored_td3,
                ddqn=restored_ddqn,
                joint_replay=restored_joint,
                routing_replay=restored_routing,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                expected_formal_config=formal_config,
            )

            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            np.testing.assert_array_equal(
                restored_td3.select_action(probe_state, noise_std=0.1),
                expected_action,
            )
            restored_batch = restored_joint.sample(
                1, current_lambda=0.123, gamma=1.0
            )
            for expected, actual in zip(expected_batch, restored_batch):
                self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))

        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_1_target",
            "critic_2",
            "critic_2_target",
        ):
            _assert_module_equal(
                self, getattr(td3, name), getattr(restored_td3, name)
            )
        for name in (
            "q_network",
            "target_q_network",
            "cost_network",
            "target_cost_network",
        ):
            _assert_module_equal(
                self, getattr(ddqn, name), getattr(restored_ddqn, name)
            )
        self.assertTrue(restored_td3.actor_optimizer.state_dict()["state"])
        self.assertTrue(restored_td3.critic_1_optimizer.state_dict()["state"])
        self.assertTrue(restored_ddqn.optimizer.state_dict()["state"])
        self.assertTrue(restored_ddqn.cost_optimizer.state_dict()["state"])
        self.assertEqual(restored_td3.num_training, td3.num_training)
        self.assertEqual(restored_ddqn.num_training, ddqn.num_training)
        self.assertEqual(restored_td3.gamma, td3.gamma)
        self.assertEqual(restored_td3.policy_delay, td3.policy_delay)
        self.assertEqual(restored_ddqn.constraint_state(), ddqn.constraint_state())
        self.assertEqual(restored_ddqn.loss_log, ddqn.loss_log)
        self.assertEqual(restored_ddqn.cost_loss_log, ddqn.cost_loss_log)
        self.assertEqual(restored_joint.ptr, joint.ptr)
        self.assertEqual(restored_joint.size, joint.size)
        self.assertEqual(restored_routing.ptr, routing.ptr)
        self.assertEqual(restored_routing.size, routing.size)
        self.assertEqual(restored_joint.max_size, joint.max_size)
        self.assertEqual(restored_routing.max_size, routing.max_size)
        self.assertEqual(restored_routing.n_step_buffer, routing.n_step_buffer)
        np.testing.assert_array_equal(
            restored_joint.delivered_mbits[: joint.size],
            joint.delivered_mbits[: joint.size],
        )
        np.testing.assert_array_equal(
            restored_routing.cost[: routing.size], routing.cost[: routing.size]
        )
        np.testing.assert_array_equal(
            restored_routing.transition_id[: routing.size],
            routing.transition_id[: routing.size],
        )
        restored_state = dict(restored["training_state"])
        expected_state = dict(training_state)
        restored_channel = restored_state.pop("channel_lifecycle_state")
        expected_channel = expected_state.pop("channel_lifecycle_state")
        for field in (
            "u2g_los_state",
            "s2u_los_state",
            "u2g_los_probability",
            "s2u_los_probability",
        ):
            np.testing.assert_array_equal(
                restored_channel.pop(field), expected_channel.pop(field)
            )
        self.assertEqual(restored_channel, expected_channel)
        self.assertEqual(restored_state, expected_state)
        self.assertEqual(restored["formal_config"], formal_config)
        self.assertEqual(restored["training_state"]["next_episode_index"], 7)
        self.assertLess(
            restored["training_state"]["total_joint_transitions"],
            formal_config["warmup_joint_transitions"],
        )
        self.assertTrue(_uses_warmup_random_action(37, 1000))
        self.assertFalse(_uses_warmup_random_action(1000, 1000))
        self.assertFalse(_uses_warmup_random_action(1200, 1000))

    def test_model_only_is_complete_for_evaluation_but_refused_for_resume(self):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 1, "c_ref_com": 10.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "model"
            save_model_checkpoint(
                checkpoint_dir,
                episode=1,
                td3=td3,
                ddqn=ddqn,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                experiment_metadata=_model_provenance_fixture(2),
                routing_lifecycle_state=_empty_routing_lifecycle(),
            )
            metadata = json.loads(
                (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["checkpoint_type"], "model-only")
            self.assertEqual(
                metadata["ground_station_position_m"],
                list(GROUND_STATION_POSITION_M),
            )
            self.assertEqual(
                metadata["uav_initial_layout_version"],
                UAV_INITIAL_LAYOUT_VERSION,
            )
            self.assertEqual(
                metadata[
                    "initial_communication_topology_contract_version"
                ],
                INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION,
            )
            self.assertEqual(
                metadata["task_potential_contract_version"],
                TASK_POTENTIAL_CONTRACT_VERSION,
            )
            self.assertEqual(
                metadata["task_potential_configuration"],
                task_potential_contract_metadata(),
            )
            models = torch.load(
                checkpoint_dir / "models.pt", weights_only=False
            )
            self.assertEqual(
                set(models["ddqn"]),
                {
                    "q_network",
                    "target_q_network",
                    "cost_network",
                    "target_cost_network",
                    "constraint_state",
                },
            )
            original_actor = {
                key: value.detach().clone()
                for key, value in td3.actor.state_dict().items()
            }
            with torch.no_grad():
                for parameter in td3.actor.parameters():
                    parameter.zero_()
            load_model_checkpoint(checkpoint_dir, td3, ddqn)
            for key, expected in original_actor.items():
                self.assertTrue(torch.equal(td3.actor.state_dict()[key], expected))
            with self.assertRaisesRegex(RuntimeError, "evaluation, not exact resume"):
                load_full_resume_checkpoint(
                    checkpoint_dir,
                    td3=td3,
                    ddqn=ddqn,
                    joint_replay=joint,
                    routing_replay=routing,
                    movement_state_dim=MOVEMENT_STATE_DIM,
                    joint_action_dim=JOINT_ACTION_DIM,
                    routing_state_dim=ROUTING_STATE_DIM,
                    calibration=calibration,
                )

    def test_split_range_schema_18_checkpoint_is_rejected_before_network_restore(self):
        td3, ddqn, _joint, _routing = self._components()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "model"
            save_model_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration={"fixture": "legacy-safe"},
                experiment_metadata=_model_provenance_fixture(1),
                routing_lifecycle_state=_empty_routing_lifecycle(),
            )
            metadata_path = checkpoint_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_schema_version"] = (
                PRE_UNIFIED_400M_COMMUNICATION_CHECKPOINT_SCHEMA_VERSION
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            with mock.patch(
                "training_checkpoint._load_network_states"
            ) as load_networks:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unified inclusive 400 m S2U/U2G/U2U.*must be retrained",
                ):
                    load_model_checkpoint(checkpoint_dir, td3, ddqn)
                load_networks.assert_not_called()

    def test_soft_projection_schema_19_is_rejected_before_network_restore(self):
        td3, ddqn, _joint, _routing = self._components()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "model"
            save_model_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration={"fixture": "legacy-soft-projection"},
                experiment_metadata=_model_provenance_fixture(1),
                routing_lifecycle_state=_empty_routing_lifecycle(),
            )
            metadata_path = checkpoint_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_schema_version"] = (
                PRE_CONTINUOUS_GATEWAY_PROJECTION_CHECKPOINT_SCHEMA_VERSION
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            with mock.patch(
                "training_checkpoint._load_network_states"
            ) as load_networks:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "continuous hard-only 400 m UAV 0 gateway projection.*"
                    "must be retrained",
                ):
                    load_model_checkpoint(checkpoint_dir, td3, ddqn)
                load_networks.assert_not_called()

    def test_legacy_single_lambda_log_is_rejected_before_network_restore(self):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 1, "c_ref_com": 10.0}
        formal_config = asdict(formal_training_config(1, random_seed=1))
        dinkelbach_state = DinkelbachBlockState.from_config(formal_config)
        event = dinkelbach_state.record_episode(0.0, 1.0)
        training_state = {
            "completed_episode_index": 0,
            "next_episode_index": 1,
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [0.0],
            "delivered_log": [0.0],
            "energy_log": [1.0],
            "lambda_used_log": [event["dinkelbach_lambda_used"]],
            "lambda_after_episode_log": [
                event["dinkelbach_lambda_after_episode"]
            ],
            **_lifecycle_training_state(1),
            **dinkelbach_state.training_state(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "full"
            save_full_resume_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                joint_replay=joint,
                routing_replay=routing,
                training_state=training_state,
                formal_config=formal_config,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
            )
            payload_path = checkpoint_dir / "training_state.pt"
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            legacy_state = payload["training_state"]
            legacy_state.pop("full_resume_logging_schema_version")
            legacy_state["lambda_log"] = legacy_state.pop("lambda_after_episode_log")
            legacy_state.pop("lambda_used_log")
            torch.save(payload, payload_path)

            with mock.patch(
                "training_checkpoint._load_network_states"
            ) as load_networks:
                with self.assertRaisesRegex(
                    RuntimeError, "full-resume logging schema"
                ):
                    load_full_resume_checkpoint(
                        checkpoint_dir,
                        td3=td3,
                        ddqn=ddqn,
                        joint_replay=joint,
                        routing_replay=routing,
                        movement_state_dim=MOVEMENT_STATE_DIM,
                        joint_action_dim=JOINT_ACTION_DIM,
                        routing_state_dim=ROUTING_STATE_DIM,
                        calibration=calibration,
                    )
                load_networks.assert_not_called()

    def test_formal_config_mismatch_preflight_mutates_no_checkpoint_state(self):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 1, "c_ref_com": 10.0}
        formal_config = asdict(formal_training_config(1, random_seed=1))
        dinkelbach_state = DinkelbachBlockState.from_config(formal_config)
        event = dinkelbach_state.record_episode(0.0, 1.0)
        training_state = {
            "completed_episode_index": 0,
            "next_episode_index": 1,
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [0.0],
            "delivered_log": [0.0],
            "energy_log": [1.0],
            "lambda_used_log": [event["dinkelbach_lambda_used"]],
            "lambda_after_episode_log": [
                event["dinkelbach_lambda_after_episode"]
            ],
            **_lifecycle_training_state(1),
            **dinkelbach_state.training_state(),
        }
        experiment_metadata = {
            "formal_config": formal_config,
            **dinkelbach_config_metadata(formal_config),
            "lambda_ee": dinkelbach_state.current_lambda,
            "dinkelbach_state": dinkelbach_state.training_state(),
        }
        incompatible = dict(formal_config)
        incompatible["batch_size"] = int(formal_config["batch_size"]) + 1
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "full"
            save_full_resume_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                joint_replay=joint,
                routing_replay=routing,
                training_state=training_state,
                formal_config=formal_config,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                experiment_metadata=experiment_metadata,
            )

            with (
                mock.patch("training_checkpoint._load_network_states") as networks,
                mock.patch.object(
                    td3.actor_optimizer, "load_state_dict"
                ) as actor_optimizer,
                mock.patch.object(
                    ddqn.optimizer, "load_state_dict"
                ) as ddqn_optimizer,
                mock.patch("training_checkpoint._load_replay") as replay,
                mock.patch("training_checkpoint._restore_rng_state") as rng,
            ):
                with self.assertRaisesRegex(RuntimeError, "batch_size"):
                    load_full_resume_checkpoint(
                        checkpoint_dir,
                        td3=td3,
                        ddqn=ddqn,
                        joint_replay=joint,
                        routing_replay=routing,
                        movement_state_dim=MOVEMENT_STATE_DIM,
                        joint_action_dim=JOINT_ACTION_DIM,
                        routing_state_dim=ROUTING_STATE_DIM,
                        calibration=calibration,
                        expected_formal_config=incompatible,
                    )
                networks.assert_not_called()
                actor_optimizer.assert_not_called()
                ddqn_optimizer.assert_not_called()
                replay.assert_not_called()
                rng.assert_not_called()

    def test_full_resume_rejects_dimension_gamma_schema_and_calibration_mismatch(self):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 1, "c_ref_com": 10.0}
        training_state = {
            "completed_episode_index": 0,
            "next_episode_index": 1,
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [0.0],
            "delivered_log": [0.0],
            "energy_log": [1.0],
            "lambda_used_log": [0.0],
            "lambda_after_episode_log": [0.0],
            "total_joint_transitions": 0,
            **_lifecycle_training_state(1),
        }
        formal_config = asdict(formal_training_config(1, random_seed=1))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "full"
            save_full_resume_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                joint_replay=joint,
                routing_replay=routing,
                training_state=training_state,
                formal_config=formal_config,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
            )

            common = {
                "checkpoint_dir": checkpoint_dir,
                "td3": td3,
                "ddqn": ddqn,
                "joint_replay": joint,
                "routing_replay": routing,
                "movement_state_dim": MOVEMENT_STATE_DIM,
                "joint_action_dim": JOINT_ACTION_DIM,
                "routing_state_dim": ROUTING_STATE_DIM,
                "calibration": calibration,
            }
            for field, value in (
                ("movement_state_dim", 531),
                ("joint_action_dim", JOINT_ACTION_DIM - 1),
                ("routing_state_dim", 122),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, field
                ):
                    load_full_resume_checkpoint(**(common | {field: value}))

            wrong_td3 = TD3(
                MOVEMENT_STATE_DIM,
                JOINT_ACTION_DIM,
                max_action=1.0,
                gamma=0.99,
            )
            with self.assertRaisesRegex(RuntimeError, "centralized_td3_gamma"):
                load_full_resume_checkpoint(**(common | {"td3": wrong_td3}))
            with self.assertRaisesRegex(RuntimeError, "calibration fingerprint"):
                load_full_resume_checkpoint(
                    **(common | {"calibration": {"different": True}})
                )
            with self.assertRaisesRegex(
                RuntimeError, "missing Dinkelbach block state"
            ):
                load_full_resume_checkpoint(**common)

            metadata_path = checkpoint_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_schema_version"] = (
                PRE_ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "incompatible.*must be retrained"
            ):
                load_full_resume_checkpoint(**common)

            metadata["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_schema_version"):
                load_full_resume_checkpoint(**common)


class TrainingCliTest(unittest.TestCase):
    def test_formal_mode_requires_explicit_episodes(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_training_config(
                    ["--mode", "train", "--seed", "20260817"]
                )
        self.assertIn("formal train mode requires --episodes", stderr.getvalue())

    def test_formal_mode_requires_explicit_seed(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_training_config(
                    ["--mode", "train", "--episodes", "1500"]
                )
        self.assertIn("formal train mode requires --seed", stderr.getvalue())

    def test_cli_builds_smoke_and_formal_configs(self):
        smoke = parse_training_config(["--mode", "smoke"])
        self.assertEqual(smoke.total_episodes, 1)
        self.assertEqual(smoke.random_seed, 20260817)
        self.assertFalse(smoke.enable_full_resume)
        formal = parse_training_config(
            [
                "--mode",
                "train",
                "--episodes",
                "1500",
                "--seed",
                "20260817",
            ]
        )
        self.assertEqual(formal.mode, "train")
        self.assertEqual(formal.total_episodes, 1500)
        self.assertEqual(formal.random_seed, 20260817)
        self.assertEqual(formal.warmup_joint_transitions, 10_000)
        self.assertEqual(formal.model_checkpoint_every, 50)
        self.assertEqual(formal.full_resume_every, 50)
        self.assertEqual(formal.full_resume_keep_last, 2)
        self.assertEqual(formal.formal_evaluation_episode, 1500)
        self.assertEqual(formal.dinkelbach_initial_lambda, 0.0)
        self.assertEqual(formal.dinkelbach_update_interval_episodes, 50)
        self.assertEqual(formal.dinkelbach_update_rule, "ratio_of_block_sums")

    def test_smoke_mode_rejects_seed_override(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_training_config(
                    ["--mode", "smoke", "--seed", "123"]
                )
        self.assertIn(
            "smoke mode fixes --seed to 20260817; do not pass --seed",
            stderr.getvalue(),
        )

    def test_checkpoint_schema_is_explicit(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 20)


if __name__ == "__main__":
    unittest.main()
