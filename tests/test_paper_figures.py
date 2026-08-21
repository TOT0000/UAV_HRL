import json
import tempfile
import unittest
from pathlib import Path

from experiment_config import MethodSpec
from paper_figure_registry import FIGURE_REGISTRY, LegacyFigureSourceUnavailable
from paper_figures import (
    AmbiguousPaperRunError,
    IncompatiblePaperRunError,
    build_paper_figures,
    causal_trailing_average,
    normalize_episode_ee,
)


class EpisodeEnergyEfficiencyTest(unittest.TestCase):
    def test_causal_average_never_reads_a_future_episode(self):
        values = list(range(1, 52))
        averaged = causal_trailing_average(values, window=50)
        self.assertEqual(averaged[0], 1.0)
        self.assertEqual(averaged[1], 1.5)
        self.assertEqual(averaged[49], sum(range(1, 51)) / 50)
        self.assertEqual(averaged[50], sum(range(2, 52)) / 50)

    def test_adapter_uses_each_episode_bits_and_energy_not_reward(self):
        rows = [
            {
                "method_id": "td3_dinkelbach",
                "episode": 1,
                "reward": 999999.0,
                "timely_goodput_mbits": 2.0,
                "mobility_energy_j": 4.0,
            },
            {
                "method_id": "td3_dinkelbach",
                "episode": 2,
                "reward": -999999.0,
                "timely_goodput_mbits": 9.0,
                "mobility_energy_j": 3.0,
            },
        ]
        normalized = normalize_episode_ee("td3_dinkelbach", rows)
        self.assertEqual(
            [row["raw_energy_efficiency_bit_per_j"] for row in normalized],
            [500000.0, 3000000.0],
        )
        self.assertEqual(
            normalized[1]["trailing_50_energy_efficiency_bit_per_j"],
            1750000.0,
        )


class SyntheticFigureBuildTest(unittest.TestCase):
    def _write_run(self, root, method_id, manifest_hash="m" * 64):
        run_dir = root / method_id
        run_dir.mkdir()
        method = MethodSpec.parse(method_id)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    "method": method_id,
                    "method_spec": method.to_dict(),
                    "seed": 20260817,
                    "training_manifest_hash": manifest_hash,
                    "formal_checkpoint_episode": 2500,
                    "status": "COMPLETED",
                }
            ),
            encoding="utf-8",
        )
        checkpoint = run_dir / "checkpoints" / "models" / "ep_2500"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(
            json.dumps({"episode": 2499, "method": method_id}),
            encoding="utf-8",
        )
        (checkpoint / "models.pt").write_bytes(b"synthetic-model")
        rows = [
            {
                "method_id": method_id,
                "episode": episode,
                "reward": 1e9,
                "timely_goodput_mbits": float(episode),
                "mobility_energy_j": 2.0,
            }
            for episode in range(1, 4)
        ]
        (run_dir / "training_history.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return run_dir

    def _fixture(self, root):
        root.mkdir(parents=True, exist_ok=True)
        methods = FIGURE_REGISTRY["fig2"]["methods"]
        runs = {method: self._write_run(root, method) for method in methods}
        spec = root / "paper_runs.json"
        spec.write_text(
            json.dumps(
                {"methods": {method: {"run_dir": str(path)} for method, path in runs.items()}}
            ),
            encoding="utf-8",
        )
        return spec, runs

    def test_synthetic_fig2_build_writes_png_pdf_csv_json_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _ = self._fixture(root)
            first = build_paper_figures(spec, output_root=root / "output")
            second = build_paper_figures(spec, output_root=root / "output")
            self.assertNotEqual(first["output_directory"], second["output_directory"])
            for result in (first, second):
                output = Path(result["output_directory"])
                for name in (
                    "Total_reward.png",
                    "Total_reward.pdf",
                    "Total_reward.csv",
                    "Total_reward.json",
                    "resolved_figure_spec.json",
                    "paper_figure_build.json",
                    "unavailable_figures.json",
                ):
                    self.assertTrue((output / name).is_file(), name)
                rows = json.loads((output / "Total_reward.json").read_text())
                self.assertEqual(len(rows), 15)
                self.assertNotIn("reward", rows[0])

    def test_ambiguous_and_incompatible_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, runs = self._fixture(root)
            value = json.loads(spec.read_text())
            first_method = FIGURE_REGISTRY["fig2"]["methods"][0]
            value["methods"][first_method] = {
                "candidates": [str(runs[first_method]), str(runs[first_method])]
            }
            spec.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AmbiguousPaperRunError):
                build_paper_figures(spec, output_root=root / "ambiguous")

            spec, runs = self._fixture(root / "other")
            last_method = FIGURE_REGISTRY["fig2"]["methods"][-1]
            resolved = json.loads((runs[last_method] / "resolved_config.json").read_text())
            resolved["training_manifest_hash"] = "x" * 64
            (runs[last_method] / "resolved_config.json").write_text(
                json.dumps(resolved), encoding="utf-8"
            )
            with self.assertRaises(IncompatiblePaperRunError):
                build_paper_figures(spec, output_root=root / "incompatible")

    def test_direct_request_for_missing_legacy_figure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = Path(temp_dir) / "paper_runs.json"
            spec.write_text(json.dumps({"methods": {}}), encoding="utf-8")
            with self.assertRaises(LegacyFigureSourceUnavailable):
                build_paper_figures(spec, figure="fig5", output_root=Path(temp_dir) / "out")


if __name__ == "__main__":
    unittest.main()
