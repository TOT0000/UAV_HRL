"""Semantic paper-figure contracts reconstructed from the original renderers.

The Google Drive scripts are provenance inputs only.  Production rendering is
fully local and consumes artifacts emitted by the unified runner.
"""

from __future__ import annotations

from types import MappingProxyType


GITHUB_HISTORY_SOURCE_COMMIT = "57f6621d248e2917b6a577d872d6c19d298ed006"

DRIVE_SOURCES = MappingProxyType(
    {
        "training_reward": {
            "path": "Training trend chart/Reward_Vs_episode.py",
            "file_id": "1_yX7ObsF1DIYv3Nlue1BIs_oK3VSrwZf",
            "sha256": "65d93a1f92347cfaa0c89cdb08113d9f9a0abd6c5183a18b5c67e5bc09052de4",
            "entry_point": "module-level matplotlib renderer",
        },
        "assignment_renderer": {
            "path": "Task_assignment/EE_Vs_num_gt_task_assignment/EE_Vs_number_of_GT.py",
            "file_id": "1tfMd3Q8kFoTKQfVVtHonN8b5B6rYK7My",
            "sha256": "1f02612d1aef68f0f09e262faf4ab88b2d67d053c0a78b215d342626c40a9192",
            "entry_point": "module-level matplotlib renderer",
        },
        "trajectory_design_renderer": {
            "path": "UAV_deployment/EE_Vs_num_gt/EE_Vs_number_of_GT.py",
            "file_id": "1S6oxfzELeWvDckAEQhzX1sMWhrAnheHm",
            "sha256": "83b243c80be735e7edd289c121e0559c1858b7b8a7da8e0437ebb96fdd049194",
            "entry_point": "module-level matplotlib renderer",
        },
        "total_collector": {
            "path": "Total/Plot_curve_EE.py",
            "file_id": "1sHwN2_3QcOeNiyPUETI-fnMB7yJ_BivU",
            "sha256": "5b8521ac026267509c83cd9736477c5fc447bba17f88b6a10fa3956a2d031afb",
            "entry_point": "plot_uav_scene and legacy EE collector",
        },
        "hierarchical_renderer": {
            "path": "Total/組合比較圖_EE/EE_Vs_number_of_GT.py",
            "file_id": "14bVvMGsCw_UCqN8MEE6yNhGkQTJpw_ZK",
            "sha256": "5c5c427131acfe2d9ca1b0083c5c22a377c7c26ecfdb1cadbf344b944cf00b4d",
            "entry_point": "module-level matplotlib renderer",
        },
        "roi_delay_renderer": {
            "path": "Total/組合比較圖_e2e_delays/Task_type_delay_Vs_number_of_GT.py",
            "file_id": "17eaSEyqwf7cDgy3O-vr5s82RsLogtV6X",
            "sha256": "29807a04626951d358670fa39269f31db906949b800660744c376e099ee6ed19",
            "entry_point": "module-level matplotlib renderer",
        },
        "arrival_renderer": {
            "path": "Total/Delays_Vs_arrival rate/Task_type_delay_Vs_arrival_rate.py",
            "file_id": "1j0vfNcdgjnYb67KXPA9rgfMuWnGedyUH",
            "sha256": "95370a5a2f050220a9265f5406a44cb2b9b6c5e9193a8cee315a3a1ffe6bc690",
            "entry_point": "module-level matplotlib renderer",
        },
        "violation_renderer": {
            "path": "Delay_violation/Delay_violation_Vs_target_delay/Task_type_delay_violation_Vs_target_delay.py",
            "file_id": "1yIA4PI_JOhVks6vfHtodeFb59NcLbmAN",
            "sha256": "d7890c0aae6ecd24d23952467f9117fb527809cdfd0c306cec6d7ad983e832b1",
            "entry_point": "module-level matplotlib renderer",
        },
    }
)


PAPER_METHOD_MAPPINGS = MappingProxyType(
    {
        "training_ee_vs_episode": (
            "td3_dinkelbach",
            "td3_dinkelbach_wo_ta",
            "ddpg_dinkelbach",
            "km_td3_dinkelbach",
            "km_ddpg_dinkelbach",
        ),
        "task_assignment_ee_vs_number_of_rois": {
            "K-KM": "td3_dinkelbach",
            "KM": "km_td3_dinkelbach",
            "Random": "random_assignment_td3_dinkelbach",
        },
        "trajectory_design_ee_vs_number_of_rois": (
            "td3_dinkelbach",
            "ddpg_dinkelbach",
            "td3_ratio",
            "ddpg_ratio",
            "random_action",
            "td3_dinkelbach_no_task_potential",
            "ddpg_dinkelbach_no_task_potential",
        ),
        "hierarchical_architecture_ee_vs_number_of_rois": (
            "td3_dinkelbach",
            "td3_dinkelbach_wo_ta",
            "ddpg_dinkelbach_wo_ta",
            "kkm_random_action_random_routing",
        ),
        "task_type_delay_vs_arrival_rate": {
            "Random": "td3_dinkelbach_random_routing",
            "DQN": "td3_dinkelbach_dqn",
            "Safe-DDQN": "td3_dinkelbach",
        },
        "com_task_delay_vs_arrival_rate": {
            "Random": "td3_dinkelbach_random_routing",
            "DQN": "td3_dinkelbach_dqn",
            "Safe-DDQN": "td3_dinkelbach",
        },
        "vs_task_delay_vs_arrival_rate": {
            "Random": "td3_dinkelbach_random_routing",
            "DQN": "td3_dinkelbach_dqn",
            "Safe-DDQN": "td3_dinkelbach",
        },
        "task_type_delay_violation_vs_target_delay": (
            "td3_dinkelbach_random_routing",
            "td3_dinkelbach_dqn",
            "td3_dinkelbach_wo_ta",
            "td3_dinkelbach",
        ),
        "task_type_delay_vs_number_of_rois": (
            "td3_dinkelbach_random_routing",
            "td3_dinkelbach_dqn_wo_ta",
            "td3_dinkelbach_wo_ta",
            "td3_dinkelbach",
        ),
    }
)


METHOD_DISPLAY_NAMES = MappingProxyType(
    {
        "td3_dinkelbach": "Our method w/ task-aware",
        "td3_dinkelbach_wo_ta": "Our method w/o task-aware",
        "ddpg_dinkelbach": "DDPG with Dinkelbach",
        "km_td3_dinkelbach": "KM+TD3-Dinkelbach",
        "km_ddpg_dinkelbach": "KM+DDPG-Dinkelbach",
        "random_assignment_td3_dinkelbach": "Random",
        "td3_ratio": "TD3",
        "ddpg_ratio": "DDPG",
        "random_action": "Random selected",
        "td3_dinkelbach_no_task_potential": "TD3-Dinkelbach w/o task potential",
        "ddpg_dinkelbach_no_task_potential": "DDPG-Dinkelbach w/o task potential",
        "ddpg_dinkelbach_wo_ta": "DDPG-Dinkelbach w/o task-aware",
        "kkm_random_action_random_routing": "K-KM+Rand+Rand",
        "td3_dinkelbach_random_routing": "Random",
        "td3_dinkelbach_dqn": "DQN",
        "td3_dinkelbach_dqn_wo_ta": "DQN w/o task-aware",
    }
)


PLOT_STYLES = MappingProxyType(
    {
        "uav_trajectory_snapshots": {
            "target_uav": {"color": "green", "current_marker": "open circle", "path": "solid"},
            "other_uavs": {"color": "lightgray", "alpha": 0.45},
            "sr": {"color": "#243BFF", "marker": "square", "path": "dashed"},
            "ground_station": {"color": "red", "marker": "triangle-up"},
            "detected_rois": {"edgecolor": "magenta", "fill": "none"},
            "other_rois": {"edgecolor": "gray", "fill": "none"},
            "u2u_link": {"color": "red", "linestyle": "dashed"},
            "u2g_link": {"color": "purple", "linestyle": "dashed"},
            "sensing_coverage": {"color": "#E8B95A", "alpha": 0.25},
        },
        "training_ee_vs_episode": {
            "td3_dinkelbach": {"label": "Our method w/ task-aware", "color": "#E74C3C", "raw_color": "#FFC4B2", "raw_alpha": 0.4},
            "td3_dinkelbach_wo_ta": {"label": "Our method w/o task-aware", "color": "#27AE60", "raw_color": "#B6E388", "raw_alpha": 0.5},
            "ddpg_dinkelbach": {"label": "K-KM+DDPG-Dinkelbach", "color": "#E67E22", "raw_color": "#FFD1A4", "raw_alpha": 0.5},
            "km_td3_dinkelbach": {"label": "KM+TD3-Dinkelbach", "color": "#2471A3", "raw_color": "#AED6F1", "raw_alpha": 0.5},
            "km_ddpg_dinkelbach": {"label": "KM+DDPG-Dinkelbach", "color": "#4224A3", "raw_color": "#B6AEF1", "raw_alpha": 0.5},
        },
        "task_assignment_ee_vs_number_of_rois": {
            "td3_dinkelbach": {"color": "red", "marker": "*", "markersize": 10},
            "km_td3_dinkelbach": {"color": "blue", "marker": "^", "markersize": 9},
            "random_assignment_td3_dinkelbach": {"color": "#187600", "marker": "o", "markersize": 8},
        },
        "trajectory_design_ee_vs_number_of_rois": {
            "td3_dinkelbach": {"label": "TD3 with Dinkelbach", "color": "#243BFF", "marker": "s"},
            "ddpg_dinkelbach": {"label": "DDPG with Dinkelbach", "color": "#A52A2A", "marker": "D"},
            "td3_ratio": {"label": "TD3", "color": "#C000C0", "marker": "^"},
            "ddpg_ratio": {"label": "DDPG", "color": "#F39C00", "marker": "v"},
            "random_action": {"label": "Random selected", "color": "#187600", "marker": "o"},
            "td3_dinkelbach_no_task_potential": {"label": "TD3-Dink. w/o task potential", "color": "#00A6A6", "marker": "P", "linestyle": "--"},
            "ddpg_dinkelbach_no_task_potential": {"label": "DDPG-Dink. w/o task potential", "color": "#666666", "marker": "X", "linestyle": "--"},
        },
        "hierarchical_architecture_ee_vs_number_of_rois": {
            "td3_dinkelbach": {"label": "Our w/ TA", "color": "red", "marker": "*", "markersize": 10},
            "td3_dinkelbach_wo_ta": {"label": "Our w/o TA", "color": "#243BFF", "marker": "s"},
            "ddpg_dinkelbach_wo_ta": {"label": "DDPG-Dink. w/o TA", "color": "#A52A2A", "marker": "D"},
            "kkm_random_action_random_routing": {"label": "K-KM+Rand+Rand", "color": "#187600", "marker": "o"},
        },
        "task_type_delay_vs_arrival_rate": {
            "td3_dinkelbach_random_routing": {"color": "#187600"},
            "td3_dinkelbach_dqn": {"color": "#F39C00"},
            "td3_dinkelbach": {"color": "red"},
        },
        "com_task_delay_vs_arrival_rate": {
            "td3_dinkelbach_random_routing": {"color": "#187600"},
            "td3_dinkelbach_dqn": {"color": "#F39C00"},
            "td3_dinkelbach": {"color": "red"},
        },
        "vs_task_delay_vs_arrival_rate": {
            "td3_dinkelbach_random_routing": {"color": "#187600"},
            "td3_dinkelbach_dqn": {"color": "#F39C00"},
            "td3_dinkelbach": {"color": "red"},
        },
        "task_type_delay_violation_vs_target_delay": {
            "td3_dinkelbach_random_routing": {"label": "Random", "color": "#187600", "marker": "*"},
            "td3_dinkelbach_dqn": {"label": "DQN", "color": "#F39C00", "marker": "s"},
            "td3_dinkelbach_wo_ta": {"label": "Our w/o task-aware", "color": "#243BFF", "marker": "^"},
            "td3_dinkelbach": {"label": "Our method", "color": "red", "marker": "8"},
        },
        "task_type_delay_vs_number_of_rois": {
            "td3_dinkelbach_random_routing": {"label": "Rand", "color": "#187600", "marker": "^"},
            "td3_dinkelbach_dqn_wo_ta": {"label": "DQN w/o task-aware", "color": "#C000C0", "marker": "s"},
            "td3_dinkelbach_wo_ta": {"label": "Our method w/o task-aware", "color": "#243BFF", "marker": "v"},
            "td3_dinkelbach": {"label": "Our method w/ task-aware", "color": "red", "marker": "*"},
        },
    }
)


def _source(*keys):
    return [dict(DRIVE_SOURCES[key]) for key in keys]


FIGURE_REGISTRY = MappingProxyType(
    {
        "uav_trajectory_t_5s": {
            "output_stem": "UAV_trajectory_t_5s",
            "sources": _source("total_collector"),
            "production_source": "trajectory_artifacts.json from deterministic unified-runner evaluation",
            "screenshot_reference": "standalone t=5 s 3D snapshot; screenshot 2026-08-21 223053",
            "subplots": [1, 1],
            "requested_time_seconds": 5.0,
            "camera": {"elevation_degrees": 20, "azimuth_degrees": 60},
            "methods": ("td3_dinkelbach",),
        },
        "uav_trajectory_t_10s": {
            "output_stem": "UAV_trajectory_t_10s",
            "sources": _source("total_collector"),
            "production_source": "trajectory_artifacts.json from deterministic unified-runner evaluation",
            "screenshot_reference": "standalone t=10 s 3D snapshot; screenshot 2026-08-21 223053",
            "subplots": [1, 1],
            "requested_time_seconds": 10.0,
            "camera": {"elevation_degrees": 20, "azimuth_degrees": 60},
            "methods": ("td3_dinkelbach",),
        },
        "uav_trajectory_t_15s": {
            "output_stem": "UAV_trajectory_t_15s",
            "sources": _source("total_collector"),
            "production_source": "trajectory_artifacts.json from deterministic unified-runner evaluation",
            "screenshot_reference": "standalone t=15 s 3D snapshot; screenshot 2026-08-21 223053",
            "subplots": [1, 1],
            "requested_time_seconds": 15.0,
            "camera": {"elevation_degrees": 20, "azimuth_degrees": 60},
            "methods": ("td3_dinkelbach",),
        },
        "uav_trajectory_t_25s": {
            "output_stem": "UAV_trajectory_t_25s",
            "sources": _source("total_collector"),
            "production_source": "trajectory_artifacts.json from deterministic unified-runner evaluation",
            "screenshot_reference": "standalone t=25 s 3D snapshot; screenshot 2026-08-21 223053",
            "subplots": [1, 1],
            "requested_time_seconds": 25.0,
            "camera": {"elevation_degrees": 20, "azimuth_degrees": 60},
            "methods": ("td3_dinkelbach",),
        },
        "training_ee_vs_episode": {
            "output_stem": "Training_EE_Vs_episode",
            "sources": _source("training_reward"),
            "github_history_source": {
                "commit": GITHUB_HISTORY_SOURCE_COMMIT,
                "path": "HRL_task_aware.py",
                "entry_point": "train inline legacy convergence renderer",
            },
            "production_source": "training_history.jsonl",
            "screenshot_reference": "single convergence axes; screenshot 2026-08-21 223103",
            "subplots": [1, 1],
            "methods": PAPER_METHOD_MAPPINGS["training_ee_vs_episode"],
            "aggregation": "50-episode causal trailing mean",
            "intentional_changes": [
                "reward is replaced with independent episode timely-bit/mobility-energy EE",
                "Drive hard-coded y arrays are never used",
            ],
        },
        "task_assignment_ee_vs_number_of_rois": {
            "output_stem": "Task_assignment_EE_Vs_number_of_RoIs",
            "sources": _source("assignment_renderer"),
            "production_source": "fixed_roi pooled evaluation artifacts",
            "screenshot_reference": "left EE panel; screenshot 2026-08-21 223109",
            "subplots": [1, 1],
            "methods": PAPER_METHOD_MAPPINGS["task_assignment_ee_vs_number_of_rois"],
        },
        "trajectory_design_ee_vs_number_of_rois": {
            "output_stem": "Trajectory_design_EE_Vs_number_of_RoIs",
            "sources": _source("trajectory_design_renderer"),
            "production_source": "fixed_roi pooled evaluation artifacts",
            "screenshot_reference": "middle EE panel; screenshot 2026-08-21 223109",
            "subplots": [1, 1],
            "methods": PAPER_METHOD_MAPPINGS["trajectory_design_ee_vs_number_of_rois"],
            "intentional_changes": [
                "two no-task-potential ablations use additional dashed palette entries"
            ],
        },
        "hierarchical_architecture_ee_vs_number_of_rois": {
            "output_stem": "Hierarchical_architecture_EE_Vs_number_of_RoIs",
            "sources": _source("hierarchical_renderer"),
            "production_source": "fixed_roi pooled evaluation artifacts",
            "screenshot_reference": "right EE panel; screenshot 2026-08-21 223109",
            "subplots": [1, 1],
            "methods": PAPER_METHOD_MAPPINGS["hierarchical_architecture_ee_vs_number_of_rois"],
        },
        "com_task_delay_vs_arrival_rate": {
            "output_stem": "COM_task_delay_Vs_arrival_rate",
            "sources": _source("arrival_renderer"),
            "production_source": "task_type_delay_vs_arrival_rate pooled evaluation artifacts",
            "screenshot_reference": "standalone COM grouped bars; screenshot 2026-08-21 223114",
            "subplots": [1, 1],
            "task_type": "COM",
            "methods": PAPER_METHOD_MAPPINGS["com_task_delay_vs_arrival_rate"],
            "conversion": {"internal": "seconds", "display": "milliseconds", "factor": 1000.0},
        },
        "vs_task_delay_vs_arrival_rate": {
            "output_stem": "VS_task_delay_Vs_arrival_rate",
            "sources": _source("arrival_renderer"),
            "production_source": "task_type_delay_vs_arrival_rate pooled evaluation artifacts",
            "screenshot_reference": "standalone VS grouped bars; screenshot 2026-08-21 223114",
            "subplots": [1, 1],
            "task_type": "FOV",
            "methods": PAPER_METHOD_MAPPINGS["vs_task_delay_vs_arrival_rate"],
            "conversion": {"internal": "seconds", "display": "milliseconds", "factor": 1000.0},
        },
        "task_type_delay_violation_vs_target_delay": {
            "output_stem": "Task_type_delay_violation_Vs_target_delay",
            "sources": _source("violation_renderer"),
            "production_source": "task_type_delay_violation_vs_target_delay pooled evaluation artifacts",
            "screenshot_reference": "log-y line chart; screenshot 2026-08-21 223119",
            "subplots": [1, 1],
            "methods": PAPER_METHOD_MAPPINGS["task_type_delay_violation_vs_target_delay"],
            "intentional_changes": ["deadline threshold is displayed in seconds (0.5 through 3.0)"],
        },
        "task_type_delay_vs_number_of_rois": {
            "output_stem": "Task_type_delay_Vs_number_of_RoIs",
            "sources": _source("roi_delay_renderer"),
            "production_source": "fixed_roi pooled evaluation artifacts",
            "screenshot_reference": "VS/COM line chart; screenshot 2026-08-21 223125",
            "subplots": [1, 1],
            "methods": PAPER_METHOD_MAPPINGS["task_type_delay_vs_number_of_rois"],
            "conversion": {"internal": "seconds", "display": "milliseconds", "factor": 1000.0},
        },
    }
)


# Accepted only at Python compatibility boundaries.  CLIs, output paths, and
# metadata always expose the semantic identifier returned by resolve_figure_id.
DEPRECATED_FIGURE_ALIASES = MappingProxyType(
    {
        "fig2": "training_ee_vs_episode",
        "fig3": (
            "uav_trajectory_t_5s",
            "uav_trajectory_t_10s",
            "uav_trajectory_t_15s",
            "uav_trajectory_t_25s",
        ),
        "uav_trajectory_snapshots": (
            "uav_trajectory_t_5s",
            "uav_trajectory_t_10s",
            "uav_trajectory_t_15s",
            "uav_trajectory_t_25s",
        ),
        "fig4": (
            "task_assignment_ee_vs_number_of_rois",
            "trajectory_design_ee_vs_number_of_rois",
            "hierarchical_architecture_ee_vs_number_of_rois",
        ),
        "energy_efficiency_design_comparisons": (
            "task_assignment_ee_vs_number_of_rois",
            "trajectory_design_ee_vs_number_of_rois",
            "hierarchical_architecture_ee_vs_number_of_rois",
        ),
        "fig4a": "task_assignment_ee_vs_number_of_rois",
        "fig4b": "trajectory_design_ee_vs_number_of_rois",
        "fig4c": "hierarchical_architecture_ee_vs_number_of_rois",
        "fig5": (
            "com_task_delay_vs_arrival_rate",
            "vs_task_delay_vs_arrival_rate",
        ),
        "task_type_delay_vs_arrival_rate": (
            "com_task_delay_vs_arrival_rate",
            "vs_task_delay_vs_arrival_rate",
        ),
        "fig6": "task_type_delay_violation_vs_target_delay",
        "fig7": "task_type_delay_vs_number_of_rois",
    }
)


class LegacyFigureSourceUnavailable(RuntimeError):
    """Compatibility name retained for callers of the old fail-closed API."""


def resolve_figure_id(figure_id, *, allow_deprecated_alias=True):
    resolved = resolve_figure_ids(
        figure_id, allow_deprecated_alias=allow_deprecated_alias
    )
    if len(resolved) != 1:
        raise KeyError(
            f"deprecated paper figure {figure_id!r} expands to standalone figures: "
            f"{', '.join(resolved)}"
        )
    return resolved[0]


def resolve_figure_ids(figure_id, *, allow_deprecated_alias=True):
    key = str(figure_id).strip().lower()
    if key in FIGURE_REGISTRY:
        return (key,)
    if allow_deprecated_alias and key in DEPRECATED_FIGURE_ALIASES:
        value = DEPRECATED_FIGURE_ALIASES[key]
        return tuple(value) if isinstance(value, tuple) else (value,)
    raise KeyError(f"unknown semantic paper figure: {figure_id}")


def require_legacy_figure_contract(figure_id):
    """Return the now-available contract while normalizing old internal aliases."""

    return FIGURE_REGISTRY[resolve_figure_id(figure_id)]
