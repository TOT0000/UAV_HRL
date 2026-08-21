"""Audited legacy paper-figure contracts and production method mappings."""

from __future__ import annotations

from types import MappingProxyType


LEGACY_SOURCE_COMMIT = "57f6621d248e2917b6a577d872d6c19d298ed006"
LEGACY_SEARCH_PATTERNS = (
    "plot",
    "figure",
    "fig",
    "savefig",
    "matplotlib",
    "pyplot",
    "reward",
    "energy efficiency",
    "E2E delay",
    "violation",
    "arrival rate",
    "number of RoIs",
    "trajectory",
    "K-KM",
    "KM",
    "Random",
    "DQN",
    "DDQN",
    "TD3",
    "DDPG",
)


class LegacyFigureSourceUnavailable(RuntimeError):
    """Raised instead of inventing a paper figure absent from Git history."""


FIGURE_REGISTRY = MappingProxyType(
    {
        "fig2": {
            "available": True,
            "legacy_figure_id": "Episodes-Reward convergence",
            "legacy_output_stem": "Total_reward",
            "legacy_source_commit": LEGACY_SOURCE_COMMIT,
            "legacy_source_file": "HRL_task_aware.py",
            "legacy_entry_point": "train() inline plotting block, lines 426-460",
            "legacy_data_source": "per-episode reward_log Python list",
            "x_axis": {"metric": "episode", "label": "Episodes", "unit": None},
            "y_axis": {
                "metric": "per_episode_energy_efficiency",
                "label": "Energy efficiency (bit/J)",
                "unit": "bit/J",
            },
            "subplots": {"rows": 1, "columns": 1, "count": 1},
            "figure_size_inches": [8, 6],
            "methods": [
                "td3_dinkelbach",
                "td3_dinkelbach_wo_ta",
                "ddpg_dinkelbach",
                "km_td3_dinkelbach",
                "km_ddpg_dinkelbach",
            ],
            "style": {
                "raw_line": {"linewidth": 1, "alpha": 0.2},
                "trailing_average_line": {"linewidth": 2, "linestyle": "-"},
                "method_colors": "matplotlib default color cycle",
                "legend": True,
                "grid": True,
                "tight_layout": True,
            },
            "aggregation": {
                "kind": "causal trailing moving average",
                "window_episodes": 50,
            },
            "current_replacement_source": "training_history.jsonl",
            "intentional_changes": [
                "reward is replaced by independent per-episode timely-bit/mobility-energy EE",
                "five method histories are overlaid in one axes",
                "each raw and trailing-average pair shares one default-cycle color",
                "only the trailing-average handle appears in the legend",
            ],
        },
        "fig3": {
            "available": False,
            "legacy_figure_id": "trajectory",
            "missing": "No 2D/3D trajectory plotting function, camera, subplot, or savefig contract exists in any reachable Git tree.",
        },
        "fig4a": {
            "available": False,
            "legacy_figure_id": "assignment comparison",
            "missing": "No comparison plot or x-axis/sweep contract exists in Git history.",
        },
        "fig4b": {
            "available": False,
            "legacy_figure_id": "movement/objective comparison",
            "missing": "No comparison plot, palette, marker, or line-order contract exists in Git history.",
        },
        "fig4c": {
            "available": False,
            "legacy_figure_id": "hierarchical comparison",
            "missing": "No comparison plot or legacy display-label grouping exists in Git history.",
        },
        "fig5": {
            "available": False,
            "legacy_figure_id": "arrival rate versus average E2E delay",
            "missing": "Packet E2E fields exist, but no Fig.5 renderer/layout/style/sweep plot exists in Git history.",
        },
        "fig6": {
            "available": False,
            "legacy_figure_id": "deadline threshold versus violation probability",
            "missing": "Training-time violation curves exist, but no deadline-sweep Fig.6 renderer/layout/grouping exists in Git history.",
        },
        "fig7": {
            "available": False,
            "legacy_figure_id": "number of RoIs versus average E2E delay",
            "missing": "Fixed-RoI manifests exist only after the unified-runner refactor; no legacy Fig.7 renderer exists.",
        },
    }
)


PAPER_METHOD_MAPPINGS = MappingProxyType(
    {
        "fig4a": {
            "K-KM": "td3_dinkelbach",
            "KM": "km_td3_dinkelbach",
            "Random assignment": "random_assignment_td3_dinkelbach",
        },
        "fig4b": (
            "td3_dinkelbach",
            "ddpg_dinkelbach",
            "td3_ratio",
            "ddpg_ratio",
            "random_action",
            "td3_dinkelbach_no_task_potential",
            "ddpg_dinkelbach_no_task_potential",
        ),
        "fig4c": (
            "td3_dinkelbach",
            "td3_dinkelbach_wo_ta",
            "ddpg_dinkelbach_wo_ta",
            "kkm_random_action_random_routing",
        ),
        "fig5": {
            "Random routing": "td3_dinkelbach_random_routing",
            "DQN": "td3_dinkelbach_dqn",
            "Safe-DDQN": "td3_dinkelbach",
        },
        "fig6": (
            "td3_dinkelbach_random_routing",
            "td3_dinkelbach_dqn",
            "td3_dinkelbach_wo_ta",
            "td3_dinkelbach",
        ),
        "fig7": (
            "td3_dinkelbach_random_routing",
            "td3_dinkelbach_dqn_wo_ta",
            "td3_dinkelbach_wo_ta",
            "td3_dinkelbach",
        ),
    }
)


def require_legacy_figure_contract(figure_id):
    key = str(figure_id).strip().lower()
    contract = FIGURE_REGISTRY.get(key)
    if contract is None:
        raise KeyError(f"unknown paper figure: {figure_id}")
    if not contract["available"]:
        raise LegacyFigureSourceUnavailable(
            f"{key} cannot be rendered without guessing: {contract['missing']}"
        )
    return contract
