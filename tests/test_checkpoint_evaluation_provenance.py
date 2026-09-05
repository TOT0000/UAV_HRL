from copy import deepcopy
import unittest

from experiment_config import (
    METHOD_REGISTRY,
    MethodSpec,
    effective_training_config,
)
from HRL_task_aware import (
    _evaluation_provenance_aliases,
    formal_training_config,
)
from routing_lifecycle import RoutingLearnerLifecycle
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_training_provenance,
    validate_model_checkpoint_metadata,
)


class CheckpointEvaluationProvenanceTest(unittest.TestCase):
    @staticmethod
    def _lifecycle():
        return RoutingLearnerLifecycle(
            global_slot_count=240000,
            optimizer_update_count=37,
            target_update_count=37,
            epsilon_decay_start_slot=63,
            last_optimizer_update_slot=240000,
        ).state_dict()

    @classmethod
    def _metadata(cls, method_id):
        method = MethodSpec.parse(method_id)
        formal = effective_training_config(
            formal_training_config(1500, random_seed=17), method
        )
        lifecycle = cls._lifecycle() if method.learns_routing else None
        routing = deepcopy(formal["routing_agent_configuration"])
        routing.update(
            routing_optimizer_update_count=(37 if method.learns_routing else 0),
            routing_target_update_count=(37 if method.learns_routing else 0),
        )
        if method.routing == "safe_ddqn":
            routing.update(
                lambda_cost=2.75,
                initial_lambda_cost=0.0,
                normalized_eta_c=0.01,
                dual_normalization_reference_packets=10_000,
                qos_target_probability=0.05,
                lambda_update_scope="episode_end",
                cost_denominator="fixed_reference_packets",
                mid_episode_checkpoint_supported=False,
            )
        experiment = {
            "method_id": method.method_id,
            "method_spec": method.to_dict(),
            "method_spec_fingerprint": method.fingerprint,
            "training_seed": 17,
            "git_sha": "training-git-sha",
            "formal_config": formal,
        }
        metadata = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": MODEL_CHECKPOINT_TYPE,
            "episode": 1499,
            "movement_state_dim": 519,
            "joint_action_dim": 30,
            "routing_state_dim": 101,
            "movement_agent_kind": method.agent,
            "movement_agent_gamma": 1.0,
            "movement_agent_configuration": deepcopy(
                formal["movement_agent_configuration"]
            ),
            "routing_ddqn_gamma": 0.99,
            "routing_agent_kind": method.routing,
            "routing_agent_configuration": routing,
            "experiment": experiment,
        }
        if method.agent == "td3":
            metadata["centralized_td3_gamma"] = 1.0
        resolved = deepcopy(formal)
        resolved.update(
            {
                "method_key": method.method_id,
                "method_id": method.method_id,
                "method_spec": method.to_dict(),
                "method_spec_fingerprint": method.fingerprint,
                "training_episode_count": 1500,
                "training_seed": 17,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            }
        )
        metadata["training_provenance"] = {
            "training_episode_count": 1500,
            "training_git_sha": "training-git-sha",
            "resolved_training_config": resolved,
            "routing_lifecycle": lifecycle,
            "safe_ddqn_constraint_state": (
                deepcopy(routing) if method.routing == "safe_ddqn" else None
            ),
            "provenance_complete": True,
        }
        return metadata

    def test_all_16_registry_methods_have_strategy_typed_lifecycle(self):
        counts = {"safe_ddqn": 0, "dqn": 0, "random": 0}
        self.assertEqual(len(METHOD_REGISTRY), 16)
        for method_id in METHOD_REGISTRY:
            with self.subTest(method=method_id):
                metadata = self._metadata(method_id)
                validate_model_checkpoint_metadata(metadata)
                provenance = checkpoint_training_provenance(metadata)
                method = MethodSpec.parse(method_id)
                counts[method.routing] += 1
                self.assertEqual(
                    provenance["resolved_training_config"]["method_id"],
                    method_id,
                )
                self.assertEqual(
                    provenance["resolved_training_config"]["routing_policy"],
                    method.routing,
                )
                if method.learns_routing:
                    self.assertEqual(
                        provenance["routing_lifecycle"], self._lifecycle()
                    )
                else:
                    self.assertIsNone(provenance["routing_lifecycle"])
        self.assertEqual(counts, {"safe_ddqn": 12, "dqn": 2, "random": 2})

    def test_schema6_learned_model_requires_explicit_incomplete_opt_in(self):
        metadata = self._metadata("td3_dinkelbach")
        metadata["checkpoint_schema_version"] = (
            ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
        )
        metadata.pop("training_provenance")
        with self.assertRaisesRegex(RuntimeError, "incompatible.*must be retrained"):
            validate_model_checkpoint_metadata(metadata)

    def test_schema6_random_routing_needs_no_learner_lifecycle(self):
        metadata = self._metadata("td3_dinkelbach_random_routing")
        metadata["checkpoint_schema_version"] = (
            ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
        )
        metadata.pop("training_provenance")
        with self.assertRaisesRegex(RuntimeError, "incompatible.*must be retrained"):
            validate_model_checkpoint_metadata(metadata)

    def test_formal_1500_training_and_100_evaluation_aliases_do_not_mix(self):
        training = self._metadata("td3_dinkelbach")["training_provenance"]
        runtime = {
            "evaluation_episode_count": 100,
            "evaluation_git_sha": "evaluation-git-sha",
            "routing_lifecycle": RoutingLearnerLifecycle().state_dict(),
        }
        aliases = _evaluation_provenance_aliases(training, runtime)
        self.assertEqual(aliases["training_episode_count"], 1500)
        self.assertEqual(aliases["checkpoint_training_episode_count"], 1500)
        self.assertEqual(aliases["evaluation_episode_count"], 100)
        self.assertEqual(
            aliases["checkpoint_training_git_sha"], "training-git-sha"
        )
        self.assertEqual(aliases["evaluation_git_sha"], "evaluation-git-sha")
        self.assertEqual(aliases["routing_optimizer_update_count"], 37)
        self.assertEqual(aliases["routing_target_update_count"], 37)
        self.assertEqual(
            runtime["routing_lifecycle"]["routing_optimizer_update_count"], 0
        )


if __name__ == "__main__":
    unittest.main()
