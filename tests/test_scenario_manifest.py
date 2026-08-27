from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from Simulator import Simulator
from scenario_manifest import (
    OBSOLETE_SCHEMA_VERSION,
    POLICY_DEPENDENT_KEYS,
    SCENARIO_SCHEMA_VERSION,
    ScenarioManifest,
    extend_training_manifest,
    append_training_manifest_segment,
    generate_manifest,
    initial_training_manifest_segments,
    manifest_prefix,
    resolve_training_manifest_segment,
    validate_training_manifest_segments,
    sha256_json,
    validate_disjoint_manifests,
    validate_manifest_prefix_extension,
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
        first = Simulator(num_UAV=10)
        second = Simulator(num_UAV=10)

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
            env = Simulator(num_UAV=10)
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

    def test_mixed_profile_uses_full_supported_num_gt_range(self):
        manifest = generate_manifest("test", 909, 128)
        values = {int(entry["num_GT"]) for entry in manifest.episodes}

        self.assertEqual(manifest.schema_version, SCENARIO_SCHEMA_VERSION)
        self.assertEqual(manifest.generation_profile["num_gt_mode"], "mixed")
        self.assertEqual(values, set(range(2, 9)))

    def test_fixed_num_gt_profile_applies_to_every_episode_and_environment(self):
        manifest = generate_manifest("test", 910, 5, num_gt=4)

        self.assertEqual(
            manifest.generation_profile,
            {
                "num_gt_mode": "fixed",
                "fixed_num_gt": 4,
                "mixed_num_gt_min": 2,
                "mixed_num_gt_max": 8,
            },
        )
        self.assertEqual({entry["num_GT"] for entry in manifest.episodes}, {4})
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(manifest.episodes[0])
        self.assertEqual(env.num_GT, 4)
        self.assertEqual(len(env.gts), 4)
        self.assertEqual(len(env.SR_teams), 4)

    def test_fixed_num_gt_boundaries_and_invalid_values(self):
        self.assertEqual(
            {entry["num_GT"] for entry in generate_manifest("test", 1, 2, num_gt=2).episodes},
            {2},
        )
        self.assertEqual(
            {entry["num_GT"] for entry in generate_manifest("test", 1, 2, num_gt=8).episodes},
            {8},
        )
        for value in (1, 9):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, r"\[2, 8\]"
            ):
                generate_manifest("test", 1, 1, num_gt=value)

    def test_generation_profiles_have_disjoint_ids_and_seeds(self):
        manifests = [
            generate_manifest("test", 911, 4),
            generate_manifest("test", 911, 4, num_gt=4),
            generate_manifest("test", 911, 4, num_gt=6),
        ]

        validate_disjoint_manifests(manifests)
        for left in range(len(manifests)):
            for right in range(left + 1, len(manifests)):
                left_ids = {
                    entry["scenario_id"] for entry in manifests[left].episodes
                }
                right_ids = {
                    entry["scenario_id"] for entry in manifests[right].episodes
                }
                left_seeds = {
                    entry["scenario_seed"] for entry in manifests[left].episodes
                }
                right_seeds = {
                    entry["scenario_seed"] for entry in manifests[right].episodes
                }
                self.assertTrue(left_ids.isdisjoint(right_ids))
                self.assertTrue(left_seeds.isdisjoint(right_seeds))

    def test_previous_geometry_manifest_is_explicitly_obsolete(self):
        data = generate_manifest("test", 912, 1).to_dict()
        data["schema_version"] = OBSOLETE_SCHEMA_VERSION

        with self.assertRaisesRegex(ValueError, "disconnected-GS.*incompatible"):
            ScenarioManifest.from_dict(data)

    def test_training_manifest_extension_preserves_exact_canonical_prefix(self):
        previous = generate_manifest("train", 913, 3, num_gt=4)
        extended, provenance = extend_training_manifest(previous, 6)

        self.assertEqual(extended.episodes[:3], previous.episodes)
        self.assertEqual(
            manifest_prefix(extended, 3).content_hash,
            previous.content_hash,
        )
        self.assertEqual(provenance["preserved_prefix_length"], 3)
        self.assertEqual(
            provenance["previous_manifest_hash"], previous.content_hash
        )
        self.assertEqual(
            provenance["extended_manifest_hash"], extended.content_hash
        )

    def test_training_manifest_extension_rejects_one_changed_prefix_episode(self):
        previous = generate_manifest("train", 914, 3)
        extended, _ = extend_training_manifest(previous, 6)
        changed = extended.to_dict()
        changed["episodes"][1]["traffic_primitives"]["load_factor"] = 2.0
        unsigned = {
            key: value for key, value in changed.items() if key != "content_hash"
        }
        changed["content_hash"] = sha256_json(unsigned)
        changed_manifest = ScenarioManifest.from_dict(changed)

        with self.assertRaisesRegex(ValueError, "exact scenario prefix"):
            validate_manifest_prefix_extension(previous, changed_manifest)

    def test_manifest_segments_preserve_repeated_extension_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            original_path = run_dir / "scenario_manifest.json"
            original = generate_manifest("train", 915, 1)
            original.save(original_path)
            segments = initial_training_manifest_segments(
                run_dir, original_path, original
            )

            extended_three, _ = extend_training_manifest(original, 3)
            three_path = run_dir / "scenario_manifests" / "train_ep_0003.json"
            extended_three.save_atomic(three_path)
            segments = append_training_manifest_segment(
                run_dir, segments, original, extended_three, three_path
            )

            extended_five, _ = extend_training_manifest(extended_three, 5)
            five_path = run_dir / "scenario_manifests" / "train_ep_0005.json"
            extended_five.save_atomic(five_path)
            segments = append_training_manifest_segment(
                run_dir, segments, extended_three, extended_five, five_path
            )
            canonical = validate_training_manifest_segments(
                run_dir, segments, current_total_episodes=5
            )

            self.assertEqual(
                [(item["episode_start"], item["episode_end"]) for item in canonical],
                [(1, 1), (2, 3), (4, 5)],
            )
            self.assertEqual(
                [item["parent_manifest_hash"] for item in canonical],
                [None, original.content_hash, extended_three.content_hash],
            )
            self.assertEqual(
                [
                    resolve_training_manifest_segment(
                        run_dir, canonical, episode, current_total_episodes=5
                    )["manifest_hash"]
                    for episode in range(1, 6)
                ],
                [
                    original.content_hash,
                    extended_three.content_hash,
                    extended_three.content_hash,
                    extended_five.content_hash,
                    extended_five.content_hash,
                ],
            )

    def test_manifest_segments_fail_on_gap_overlap_hash_and_prefix_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            original = generate_manifest("train", 916, 1)
            extended, _ = extend_training_manifest(original, 3)
            original.save(run_dir / "original.json")
            extended.save(run_dir / "extended.json")
            valid = [
                {
                    "episode_start": 1,
                    "episode_end": 1,
                    "manifest_hash": original.content_hash,
                    "manifest_path": "original.json",
                    "parent_manifest_hash": None,
                },
                {
                    "episode_start": 2,
                    "episode_end": 3,
                    "manifest_hash": extended.content_hash,
                    "manifest_path": "extended.json",
                    "parent_manifest_hash": original.content_hash,
                },
            ]

            for label, mutate, message in (
                ("gap", lambda value: value[1].update(episode_start=3), "gap"),
                ("overlap", lambda value: value[1].update(episode_start=1), "overlap"),
                ("hash", lambda value: value[1].update(manifest_hash="bad"), "hash"),
            ):
                broken = deepcopy(valid)
                mutate(broken)
                with self.subTest(label=label), self.assertRaisesRegex(
                    RuntimeError, message
                ):
                    validate_training_manifest_segments(
                        run_dir, broken, current_total_episodes=3
                    )

            incompatible = generate_manifest("train", 917, 3)
            incompatible.save(run_dir / "incompatible.json")
            bad_prefix = deepcopy(valid)
            bad_prefix[1].update(
                manifest_hash=incompatible.content_hash,
                manifest_path="incompatible.json",
            )
            with self.assertRaisesRegex(RuntimeError, "prefix"):
                validate_training_manifest_segments(
                    run_dir, bad_prefix, current_total_episodes=3
                )

            with self.assertRaisesRegex(RuntimeError, "completely cover"):
                validate_training_manifest_segments(
                    run_dir, valid[:1], current_total_episodes=3
                )


if __name__ == "__main__":
    unittest.main()
