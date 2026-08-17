import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from Simulator import Simulator
from scenario_manifest import (
    POLICY_DEPENDENT_KEYS,
    ScenarioManifest,
    generate_manifest,
    sha256_json,
    validate_disjoint_manifests,
)


class ScenarioManifestTest(unittest.TestCase):
    def test_json_round_trip_preserves_content_and_hash(self):
        manifest = generate_manifest("train", manifest_seed=101, episode_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = manifest.save(Path(tmp) / "manifest.json")
            loaded = ScenarioManifest.load(path)

        self.assertEqual(loaded.to_dict(), manifest.to_dict())
        self.assertEqual(loaded.content_hash, manifest.content_hash)

    def test_manifest_hash_is_deterministic_and_content_sensitive(self):
        first = generate_manifest("validation", 202, 3)
        second = generate_manifest("validation", 202, 3)

        self.assertEqual(first.content_hash, second.content_hash)
        changed = second.to_dict()
        changed["episodes"][0]["traffic_primitives"]["load_factor"] = 2.0
        unsigned = {key: value for key, value in changed.items() if key != "content_hash"}
        changed["content_hash"] = sha256_json(unsigned)
        self.assertNotEqual(
            ScenarioManifest.from_dict(changed).content_hash,
            first.content_hash,
        )

    def test_same_entry_reproduces_exogenous_initial_state(self):
        entry = generate_manifest("test", 303, 1).episodes[0]
        first = Simulator(num_UAV=16)
        second = Simulator(num_UAV=16)

        first.apply_scenario_entry(entry)
        second.apply_scenario_entry(entry)

        first_state = {
            "uavs": [uav.get_position() + (uav.energy,) for uav in first.UAVs],
            "gts": [(gt.x, gt.y, gt.z, gt.radius) for gt in first.gts],
            "sr": [sr.get_position() for sr in first.SR_teams],
            "traffic": first.traffic_primitives,
        }
        second_state = {
            "uavs": [uav.get_position() + (uav.energy,) for uav in second.UAVs],
            "gts": [(gt.x, gt.y, gt.z, gt.radius) for gt in second.gts],
            "sr": [sr.get_position() for sr in second.SR_teams],
            "traffic": second.traffic_primitives,
        }
        self.assertEqual(first_state, second_state)

    def test_same_entry_reproduces_sr_motion_primitive_trajectory(self):
        entry = generate_manifest("test", 404, 1).episodes[0]
        trajectories = []
        for _ in range(2):
            env = Simulator(num_UAV=16)
            env.apply_scenario_entry(entry)
            env.SR_team_gogo(env.gts[0])
            for _ in range(5):
                env.advance_sr_teams()
            trajectories.append(
                [list(position) for position in env.sr_trajectory[0]]
            )

        self.assertEqual(trajectories[0], trajectories[1])

    def test_global_rng_consumption_cannot_change_generated_scenarios(self):
        before = generate_manifest("train", 505, 3)
        for _ in range(1000):
            random.random()
            np.random.random()
        after = generate_manifest("train", 505, 3)

        self.assertEqual(before.to_dict(), after.to_dict())

    def test_train_validation_test_ids_and_seeds_are_disjoint(self):
        manifests = [
            generate_manifest(split, 606, 4)
            for split in ("train", "validation", "test")
        ]

        validate_disjoint_manifests(manifests)
        ids = [
            {entry["scenario_id"] for entry in manifest.episodes}
            for manifest in manifests
        ]
        seeds = [
            {entry["scenario_seed"] for entry in manifest.episodes}
            for manifest in manifests
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(ids[left].isdisjoint(ids[right]))
                self.assertTrue(seeds[left].isdisjoint(seeds[right]))

    def test_policy_dependent_outcomes_are_not_in_manifest_entries(self):
        entry = generate_manifest("train", 707, 1).episodes[0]

        def keys(value):
            if isinstance(value, dict):
                return set(value).union(
                    *(keys(child) for child in value.values())
                )
            if isinstance(value, list):
                return set().union(*(keys(child) for child in value))
            return set()

        self.assertTrue(POLICY_DEPENDENT_KEYS.isdisjoint(keys(entry)))

    def test_incompatible_environment_fingerprint_fails_fast(self):
        data = generate_manifest("train", 808, 1).to_dict()
        data["config_fingerprint"] = "incompatible"
        unsigned = {key: value for key, value in data.items() if key != "content_hash"}
        data["content_hash"] = sha256_json(unsigned)

        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            ScenarioManifest.from_dict(data)


if __name__ == "__main__":
    unittest.main()
