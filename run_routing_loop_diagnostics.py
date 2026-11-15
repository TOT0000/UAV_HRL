"""Run read-only routing foldback diagnostics on a prior fixed-RoI evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from paper_evaluation import run_paper_evaluation
from routing_loop_diagnostics import (
    ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION,
    RoutingLoopDiagnosticsCollector,
    verify_episode_jsonl_prefix,
)
from mutual_next_hop_guard import (
    MUTUAL_NEXT_HOP_GUARD_FINGERPRINT,
    MUTUAL_NEXT_HOP_GUARD_SPEC,
    MUTUAL_NEXT_HOP_GUARD_VERSION,
    normalize_mutual_next_hop_guard,
)
from routing_guard_comparison import build_routing_guard_comparison
from scenario_manifest import ScenarioManifest, manifest_prefix


DEFAULT_OUTPUT_ROOT = (
    r"D:\UAV_HRL_results\ours_ep1400_routing_loop_diagnostics"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse a source fixed-RoI evaluation's exact scenarios and episode-1400 "
            "checkpoint while streaming routing foldback diagnostics."
        )
    )
    parser.add_argument("--source-evaluation-dir", required=True)
    parser.add_argument("--roi-counts", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--episodes-per-roi", type=int, default=10)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mutual-next-hop-guard", choices=("off", "on"), default="off"
    )
    return parser


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _unique_output_directory(output_root, git_sha):
    root = Path(output_root).resolve()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    short_sha = str(git_sha)[:12]
    base = root / f"{stamp}_{short_sha}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return candidate


def _load_source(source_dir, roi_counts, episodes_per_roi):
    source_dir = Path(source_dir).resolve()
    metadata_path = source_dir / "paper_evaluation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing source metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("semantic_suite") != "fixed_roi":
        raise ValueError("source evaluation must be the fixed_roi suite")
    if metadata.get("method_id") != "td3_dinkelbach_wo_c10_penalty":
        raise ValueError("source evaluation is not the requested Ours method")
    if int(metadata.get("checkpoint_episode", -1)) != 1400:
        raise ValueError("source evaluation must identify checkpoint episode 1400")
    if int(metadata.get("training_total_episodes", -1)) != 1400:
        raise ValueError("source evaluation training run is not the 1400-episode run")
    training_run = Path(
        metadata.get("training_run_directory") or metadata.get("training_run") or ""
    ).resolve()
    checkpoint_path = Path(metadata.get("checkpoint_path") or "").resolve()
    if not training_run.is_dir():
        raise FileNotFoundError(f"source training run is unavailable: {training_run}")
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"source checkpoint is unavailable: {checkpoint_path}")
    expected_checkpoint = training_run / "checkpoints" / "models" / "ep_1400"
    if checkpoint_path != expected_checkpoint.resolve():
        raise ValueError(
            "source checkpoint path does not identify the training run's ep_1400"
        )
    if episodes_per_roi <= 0:
        raise ValueError("episodes-per-roi must be positive")
    if len(set(roi_counts)) != len(roi_counts):
        raise ValueError("roi-counts must not contain duplicates")

    manifests = {}
    provenance = {}
    for roi_count in roi_counts:
        if int(roi_count) not in range(2, 9):
            raise ValueError("routing diagnostics support RoI counts 2 through 8")
        point_dir = source_dir / f"roi_{int(roi_count)}"
        manifest_path = point_dir / "scenario_manifest.json"
        source_manifest = ScenarioManifest.load(manifest_path)
        if source_manifest.episode_count < episodes_per_roi:
            raise ValueError(
                f"source RoI {roi_count} has fewer than {episodes_per_roi} episodes"
            )
        prefix = manifest_prefix(source_manifest, episodes_per_roi)
        if list(prefix.episodes) != list(source_manifest.episodes[:episodes_per_roi]):
            raise AssertionError("manifest_prefix changed selected scenario content")
        if any(int(entry["num_GT"]) != int(roi_count) for entry in prefix.episodes):
            raise ValueError(f"source manifest roi_{roi_count} contains another RoI count")
        per_episode_path = point_dir / "per_episode.jsonl"
        if not per_episode_path.is_file():
            raise FileNotFoundError(f"missing source per-episode stream: {per_episode_path}")
        packet_outcomes_path = point_dir / "packet_outcomes.jsonl"
        if not packet_outcomes_path.is_file():
            raise FileNotFoundError(
                f"missing source packet-outcome stream: {packet_outcomes_path}"
            )
        manifests[int(roi_count)] = prefix
        provenance[str(int(roi_count))] = {
            "source_manifest_path": str(manifest_path.resolve()),
            "source_manifest_hash": source_manifest.content_hash,
            "prefix_manifest_hash": prefix.content_hash,
            "selected_scenario_ids": [
                str(entry["scenario_id"]) for entry in prefix.episodes
            ],
            "source_per_episode_jsonl": str(per_episode_path.resolve()),
            "source_packet_outcomes_jsonl": str(packet_outcomes_path.resolve()),
            "selected_entries_exact_source_prefix": True,
        }
    return source_dir, metadata, training_run, checkpoint_path, manifests, provenance


def run_diagnostics(
    *, source_evaluation_dir, roi_counts, episodes_per_roi, output_root,
    mutual_next_hop_guard="off",
):
    roi_counts = tuple(int(value) for value in roi_counts)
    episodes_per_roi = int(episodes_per_roi)
    guard_enabled = normalize_mutual_next_hop_guard(mutual_next_hop_guard)
    (
        source_dir,
        source_metadata,
        training_run,
        checkpoint_path,
        manifests,
        manifest_provenance,
    ) = _load_source(source_evaluation_dir, roi_counts, episodes_per_roi)
    git_sha = _git_sha()
    output_dir = _unique_output_directory(output_root, git_sha)
    collector = RoutingLoopDiagnosticsCollector(
        output_dir,
        metadata={
            "diagnostic_schema_version": ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION,
            "source_evaluation_directory": str(source_dir),
            "source_evaluation_metadata": str(
                (source_dir / "paper_evaluation_metadata.json").resolve()
            ),
            "method_id": source_metadata["method_id"],
            "training_run_directory": str(training_run),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_episode": 1400,
            "roi_counts": list(roi_counts),
            "episodes_per_roi": episodes_per_roi,
            "episode_seconds": 60,
            "manifest_provenance": manifest_provenance,
            "selected_scenario_count": len(roi_counts) * episodes_per_roi,
            "git_sha": git_sha,
            "mutual_next_hop_guard": {
                "enabled": guard_enabled,
                "strategy_version": (
                    MUTUAL_NEXT_HOP_GUARD_VERSION if guard_enabled else None
                ),
                "strategy_fingerprint": (
                    MUTUAL_NEXT_HOP_GUARD_FINGERPRINT if guard_enabled else None
                ),
                "strategy": MUTUAL_NEXT_HOP_GUARD_SPEC if guard_enabled else None,
            },
        },
    )
    try:
        result = run_paper_evaluation(
            source_metadata["method_id"],
            run_directory=str(training_run),
            suite="fixed_roi",
            episodes=episodes_per_roi,
            episode_seconds=60,
            output_directory=str(output_dir),
            checkpoint_episode=1400,
            roi_counts=roi_counts,
            fixed_roi_manifests=manifests,
            allow_registered_fixed_roi_method=True,
            routing_loop_diagnostics=collector,
            mutual_next_hop_guard=guard_enabled,
        )
        verified_episode_count = 0
        actual_manifest_hashes = {}
        for roi_count in roi_counts:
            point_dir = output_dir / f"roi_{roi_count}"
            actual_manifest = ScenarioManifest.load(point_dir / "scenario_manifest.json")
            expected_manifest = manifests[roi_count]
            if actual_manifest.content_hash != expected_manifest.content_hash:
                raise AssertionError("saved diagnostic manifest differs from selected prefix")
            if list(actual_manifest.episodes) != list(expected_manifest.episodes):
                raise AssertionError("saved diagnostic scenario content differs from source")
            selected_ids = manifest_provenance[str(roi_count)]["selected_scenario_ids"]
            if not guard_enabled:
                verified_episode_count += verify_episode_jsonl_prefix(
                    point_dir / "per_episode.jsonl",
                    source_dir / f"roi_{roi_count}" / "per_episode.jsonl",
                    selected_ids,
                )
            actual_manifest_hashes[str(roi_count)] = actual_manifest.content_hash
        diagnostic_metadata = collector.finalize(
            extra_metadata={
                "actual_prefix_manifest_hashes": actual_manifest_hashes,
                "scenario_identity_verification": {
                    "status": "passed",
                    "selected_scenario_count": len(roi_counts) * episodes_per_roi,
                    "saved_manifest_entries_exact": True,
                },
                "source_episode_prefix_comparison": {
                    "status": (
                        "not_applicable_expected_policy_change"
                        if guard_enabled
                        else "passed"
                    ),
                    "compared_episode_count": (
                        None if guard_enabled else verified_episode_count
                    ),
                    "reason": (
                        "guarded routing intentionally changes actions and closed-loop results"
                        if guard_enabled
                        else None
                    ),
                    "discrete_values": None if guard_enabled else "exact",
                    "floating_values": (
                        None
                        if guard_enabled
                        else {"relative_tolerance": 1e-12, "absolute_tolerance": 1e-12}
                    ),
                },
            }
        )
        comparison_outputs = (
            build_routing_guard_comparison(
                source_evaluation_dir=source_dir,
                guarded_evaluation_dir=output_dir,
                roi_counts=roi_counts,
                selected_scenario_ids_by_roi={
                    key: value["selected_scenario_ids"]
                    for key, value in manifest_provenance.items()
                },
            )
            if guard_enabled
            else None
        )
    except Exception:
        collector.close()
        raise
    return {
        "output_directory": str(output_dir),
        "diagnostic_metadata": diagnostic_metadata,
        "evaluation": result,
        "comparison_outputs": comparison_outputs,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_diagnostics(
        source_evaluation_dir=args.source_evaluation_dir,
        roi_counts=args.roi_counts,
        episodes_per_roi=args.episodes_per_roi,
        output_root=args.output_root,
        mutual_next_hop_guard=args.mutual_next_hop_guard,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
