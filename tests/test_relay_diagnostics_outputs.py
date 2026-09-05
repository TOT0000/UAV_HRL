import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_metrics import write_evaluation_outputs
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from relay_diagnostics import (
    RELAY_BITS_SUM_ABS_TOL,
    RELAY_BITS_SUM_REL_TOL,
    RELAY_DIAGNOSTICS_FILENAME,
    RELAY_DIAGNOSTICS_OUTPUT_CONTRACT_VERSION,
    RELAY_FORWARDING_GROUPS,
    aggregate_relay_episode_diagnostics,
    relay_diagnostics_metadata,
    validate_relay_diagnostics,
    write_relay_diagnostics,
)
from scenario_manifest import generate_manifest


class RelayDiagnosticsOutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=1,
            warmup_joint_transitions=10_000,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260817,
        )
        cls.result = train(
            config,
            scenario_manifest=generate_manifest(
                "test", 20260817, 1, num_gt=2
            ),
            method_spec=MethodSpec.parse(
                "kkm_random_action_random_routing"
            ),
            evaluation=True,
        )

    def test_common_evaluation_writer_round_trips_actual_zero_relay_result(self):
        diagnostics = self.result["relay_diagnostics"]
        self.assertEqual(len(diagnostics["episodes"]), 1)
        self.assertEqual(
            diagnostics["episodes"][0]["assignment"]["assigned_relay_count"],
            0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = write_evaluation_outputs(
                temp_dir,
                self.result["episode_metrics"],
                self.result["run_metadata"],
                relay_diagnostics=diagnostics,
            )
            artifact = json.loads(
                (Path(temp_dir) / RELAY_DIAGNOSTICS_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            metadata = json.loads(
                (Path(temp_dir) / "run_metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            artifact,
            json.loads(
                json.dumps(diagnostics, ensure_ascii=False, allow_nan=False)
            ),
        )
        self.assertEqual(
            Path(outputs["relay_diagnostics"]).name,
            RELAY_DIAGNOSTICS_FILENAME,
        )
        self.assertEqual(
            metadata["relay_diagnostics_output_contract_version"],
            RELAY_DIAGNOSTICS_OUTPUT_CONTRACT_VERSION,
        )
        self.assertEqual(metadata["relay_diagnostics_episode_count"], 1)
        self.assertEqual(
            metadata["relay_diagnostics_summary"]["episode_scenarios"],
            [
                {
                    "episode_index": 0,
                    "scenario_id": diagnostics["episodes"][0]["scenario_id"],
                }
            ],
        )

    def test_non_finite_diagnostics_fail_without_writing_artifact(self):
        diagnostics = copy.deepcopy(self.result["relay_diagnostics"])
        diagnostics["forwarding"]["assigned_relay_forwarding"]["bits"] = float(
            "nan"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "summary|non-finite"):
                write_evaluation_outputs(
                    temp_dir,
                    self.result["episode_metrics"],
                    self.result["run_metadata"],
                    relay_diagnostics=diagnostics,
                )
            self.assertFalse(
                (Path(temp_dir) / RELAY_DIAGNOSTICS_FILENAME).exists()
            )

    def test_500_episode_fsum_round_trip_and_meaningful_tamper_rejection(self):
        episodes = []
        for index in range(500):
            forwarding = {
                group: {
                    "bits": 1e9 if index % 2 == 0 else 1e-7,
                    "packets": index % 3,
                    "completed_packet_hops": index % 5,
                }
                for group in RELAY_FORWARDING_GROUPS
            }
            episodes.append(
                {
                    "episode_index": index,
                    "scenario_id": f"scenario-{index}",
                    "assignment": {
                        "relay_assignment_history": [],
                        "relay_candidate_metrics": {},
                        "selected_relay_uav_ids": [],
                        "relay_role_change_count": index % 2,
                    },
                    "forwarding": forwarding,
                }
            )

        diagnostics = aggregate_relay_episode_diagnostics(episodes)
        metadata = relay_diagnostics_metadata(diagnostics)
        self.assertEqual(metadata["relay_diagnostics_episode_count"], 500)
        self.assertEqual(
            metadata["relay_diagnostics_forwarding_semantics"][
                "bits_validation_relative_tolerance"
            ],
            RELAY_BITS_SUM_REL_TOL,
        )
        self.assertEqual(
            metadata["relay_diagnostics_forwarding_semantics"][
                "bits_validation_absolute_tolerance"
            ],
            RELAY_BITS_SUM_ABS_TOL,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _metadata, path = write_relay_diagnostics(temp_dir, diagnostics)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_relay_diagnostics(loaded), loaded)

        tampered = copy.deepcopy(diagnostics)
        tampered["forwarding"]["assigned_relay_forwarding"]["bits"] += 1.0
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            validate_relay_diagnostics(tampered)


if __name__ == "__main__":
    unittest.main()
