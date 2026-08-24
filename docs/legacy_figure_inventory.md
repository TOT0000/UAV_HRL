# Original paper renderer inventory

This inventory joins four sources of evidence: reachable Git history, the
original testing renderers read from Google Drive, the six supplied paper
screenshots, and current unified-runner artifacts. Google Drive was audited
read-only; no Drive file is a runtime dependency and none of its hard-coded
result arrays is copied into production.

Git history source: commit
`57f6621d248e2917b6a577d872d6c19d298ed006`, `HRL_task_aware.py`, the inline
Episodes--Reward plotting block. The production base audited for this change is
`feature/centralized-td3` at
`465c4690aabf650dce472943f98d429ecd281a58`.

## Google Drive sources

| Role | Original path | Drive file ID | Read-content SHA-256 |
| --- | --- | --- | --- |
| Training renderer | `Training trend chart/Reward_Vs_episode.py` | `1_yX7ObsF1DIYv3Nlue1BIs_oK3VSrwZf` | `65d93a1f92347cfaa0c89cdb08113d9f9a0abd6c5183a18b5c67e5bc09052de4` |
| Assignment collector | `Task_assignment/Plot_curve_EE.py` | `1nPQc8nH8E3pi7hPqqmfghROBKli7lQ0n` | `fb60215c1bbe8e626c6bd868d0f356dbf53534fb95f1a668ee8008de89381057` |
| Assignment renderer | `Task_assignment/EE_Vs_num_gt_task_assignment/EE_Vs_number_of_GT.py` | `1tfMd3Q8kFoTKQfVVtHonN8b5B6rYK7My` | `1f02612d1aef68f0f09e262faf4ab88b2d67d053c0a78b215d342626c40a9192` |
| Trajectory-design collector | `UAV_deployment/Plot_curve_EE.py` | `1L6mFwoTOWSPhf4jJCpAK6LEP7QNCXeG5` | `261277b5d6d884c25db177fc85ddc9153afa1cb96809d05eda4e79fa867de485` |
| Trajectory-design renderer | `UAV_deployment/EE_Vs_num_gt/EE_Vs_number_of_GT.py` | `1S6oxfzELeWvDckAEQhzX1sMWhrAnheHm` | `83b243c80be735e7edd289c121e0559c1858b7b8a7da8e0437ebb96fdd049194` |
| GS-count renderer | `UAV_deployment/EE_Vs_number_of_GS.py` | `1uAAsn_cqY1VNag-ehC9HuGzwRxdAHp1f` | `f51836db4f5b5c81780b349ac00cec4dff9198938bc5c70779a2636ebc1f1f8c` |
| Total collector and `plot_uav_scene` | `Total/Plot_curve_EE.py` | `1sHwN2_3QcOeNiyPUETI-fnMB7yJ_BivU` | `5b8521ac026267509c83cd9736477c5fc447bba17f88b6a10fa3956a2d031afb` |
| Total delay collector | `Total/Plot_curve_delay.py` | `1mODVMjC4N7rG2o2vA2NaVWczznOIZ2L6` | `4699165b3d817757ca53376d041d10f8efd3aae540a9621e42e0cd4a76658443` |
| Hierarchical renderer | `Total/組合比較圖_EE/EE_Vs_number_of_GT.py` | `14bVvMGsCw_UCqN8MEE6yNhGkQTJpw_ZK` | `5c5c427131acfe2d9ca1b0083c5c22a377c7c26ecfdb1cadbf344b944cf00b4d` |
| RoI-delay renderer | `Total/組合比較圖_e2e_delays/Task_type_delay_Vs_number_of_GT.py` | `17eaSEyqwf7cDgy3O-vr5s82RsLogtV6X` | `29807a04626951d358670fa39269f31db906949b800660744c376e099ee6ed19` |
| Arrival-rate renderer | `Total/Delays_Vs_arrival rate/Task_type_delay_Vs_arrival_rate.py` | `1j0vfNcdgjnYb67KXPA9rgfMuWnGedyUH` | `95370a5a2f050220a9265f5406a44cb2b9b6c5e9193a8cee315a3a1ffe6bc690` |
| Violation collector | `Delay_violation/Plot_curve_delay.py` | `17Wx5ii-xTIsBC2PdVffL5DmIIDRyY0V-` | `85d3bceb73d912f115b0cdba86a42a19b48c04fe71e9f0f7239c19606cea4f1e` |
| Violation renderer | `Delay_violation/Delay_violation_Vs_target_delay/Task_type_delay_violation_Vs_target_delay.py` | `1yIA4PI_JOhVks6vfHtodeFb59NcLbmAN` | `d7890c0aae6ecd24d23952467f9117fb527809cdfd0c306cec6d7ad983e832b1` |

## Semantic figure contracts

| Semantic ID / output stem | Original visual contract | Production artifact and intentional changes |
| --- | --- | --- |
| `uav_trajectory_t_5s` / `UAV_trajectory_t_5s` | Standalone t=5 s 3D axes, elevation 20 degrees, azimuth 60 degrees; screenshot `223053` | The t=5 requested snapshot from the shared `trajectory_artifacts.json`; title uses actual artifact time and phase. |
| `uav_trajectory_t_10s` / `UAV_trajectory_t_10s` | Standalone t=10 s 3D axes; screenshot `223053` | The t=10 requested snapshot from the same evaluated artifact. |
| `uav_trajectory_t_15s` / `UAV_trajectory_t_15s` | Standalone t=15 s 3D axes; screenshot `223053` | The t=15 requested snapshot from the same evaluated artifact. |
| `uav_trajectory_t_25s` / `UAV_trajectory_t_25s` | Standalone t=25 s 3D axes; screenshot `223053` | The t=25 requested snapshot from the same evaluated artifact. |
| `training_ee_vs_episode` / `Training_EE_Vs_episode` | Training renderer; red, green, orange, blue, purple raw/smooth pairs; screenshot `223103` | Five `training_history.jsonl` files. Each episode is timely Mbit x 1e6 divided by `max(mobility J, 1e-12 J)`, followed by an exact causal 50-episode trailing mean. Reward and Dinkelbach surrogate values are not plotted. |
| `task_assignment_ee_vs_number_of_rois` / `Task_assignment_EE_Vs_number_of_RoIs` | Red-star K-KM, blue-triangle KM, green-circle Random; screenshot `223109` left | Fixed-RoI pooled evaluation data for `td3_dinkelbach`, `km_td3_dinkelbach`, and `random_assignment_td3_dinkelbach`. |
| `trajectory_design_ee_vs_number_of_rois` / `Trajectory_design_EE_Vs_number_of_RoIs` | Blue-square, brown-diamond, magenta-up-triangle, orange-down-triangle, green-circle sequence; screenshot `223109` middle | Fixed-RoI data for seven registered methods. The two task-potential ablations use documented extra dashed palette entries. |
| `hierarchical_architecture_ee_vs_number_of_rois` / `Hierarchical_architecture_EE_Vs_number_of_RoIs` | Red-star, blue-square, brown-diamond, green-circle; screenshot `223109` right | Fixed-RoI data for task-aware, masked TD3, masked DDPG, and pure-random architectures. The random method has explicit no-checkpoint provenance. |
| `com_task_delay_vs_arrival_rate` / `COM_task_delay_Vs_arrival_rate` | Standalone grouped bars titled `COM task`; screenshot `223114` | COM rates 50/100/150/200 packet/s with VS fixed at 5. Delay is pooled from sums/counts in seconds then converted once to milliseconds. |
| `vs_task_delay_vs_arrival_rate` / `VS_task_delay_Vs_arrival_rate` | Standalone grouped bars titled `VS task`; screenshot `223114` | VS rates 10/20/30/40 packet/s with COM fixed at 50. Delay is pooled from sums/counts in seconds then converted once to milliseconds. |
| `task_type_delay_violation_vs_target_delay` / `Task_type_delay_violation_Vs_target_delay` | Log-y; method color/marker plus VS solid-filled and COM dashed-open; screenshot `223119` | Every 0.5--3.0 s threshold is a real rerun. Probability is pooled violations/generated. A true zero is omitted on log axes and remains zero in source data; no epsilon is fabricated. |
| `task_type_delay_vs_number_of_rois` / `Task_type_delay_Vs_number_of_RoIs` | Method color/marker plus VS solid-filled and COM dashed-open; screenshot `223125` | Fixed RoIs 2--8. Delay is pooled by delivered-packet count and converted from seconds to milliseconds. Zero-delivery points remain `null` with `missing=true`, never zero. |

All twelve standalone contracts, method order, palette, markers, line styles, axes shape,
source provenance, screenshot description, and intentional differences are
machine-readable in `paper_figure_registry.py`.

## Aggregation and provenance rules

- EE comparison points use `sum(timely delivered Mbit) /
  max(sum(mobility J), 1e-12 J)`.
- Task delay uses `sum(delivered E2E delay seconds) / sum(delivered packets)`.
- Violation probability uses `sum(violation packets) / sum(generated packets)`.
- Per-episode values, pooled numerators/denominators, aggregate values, and
  missing flags are persisted together.
- Every non-trajectory point has exactly five rows: EE with no task type, FOV
  and COM delay, and FOV and COM violation probability. Before any
  figure-specific filtering, the builder validates that exact Cartesian set
  and cross-checks top-level rows, point-level rows, and a fresh recomputation
  from each point's per-episode JSONL through the shared `paper_metrics.py`
  implementation. Missing, duplicate, extra, numerically inconsistent, or
  semantically invalid rows fail closed.
- Every learned method requires the formal `ep_1500` checkpoint. The
  `kkm_random_action_random_routing` baseline learns neither component, creates
  no `models.pt`, and records `checkpoint_required=false`.
- Figure builds inspect the actual `metadata.json` and `models.pt` without
  loading weights. They retain the canonical metadata fingerprint, stream a
  SHA-256 over the structurally valid `models.pt`, and compute a versioned
  combined artifact fingerprint over those two hashes. All three are compared
  with top-level, point-level, per-point run metadata, trajectory,
  resolved-spec, method-map, and final-figure provenance. Pure-random methods
  require all checkpoint fields to be null. Existing training checkpoint
  metadata and resume behavior are unchanged; evaluations derive the two new
  hashes from the existing payload.
- Every point reloads its actual `scenario_manifest.json` through
  `ScenarioManifest.load()`, verifies the canonical hash, scenario IDs,
  episode count, 60-second horizon, seeds, 10 UAV entries, fixed-RoI content,
  and resolved arrival/deadline overrides.
- Deprecated `fig*` names exist only as internal compatibility aliases. CLI
  choices, metadata, directories, and filenames use semantic names. Family
  aliases expand to standalone outputs and never render a composite.
- Each trajectory source JSON is a self-contained
  `uav-hrl-standalone-trajectory-v1` scene: all 10 UAV snapshots, assignments
  and time-truncated paths; every SR snapshot/path; GS, targets, links,
  coverage geometry; actual time/phase; complete provenance; and the camera,
  axes, labels, and style contract. The companion long-form CSV distinguishes
  path, snapshot, target, station, link, and coverage records. The standalone
  JSON can redraw the figure after the original evaluation trajectory artifact
  is unavailable, while preserving actual-time titles and the legacy `VS`
  display name for FOV.
