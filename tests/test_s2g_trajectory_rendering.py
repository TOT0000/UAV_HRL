import matplotlib.pyplot as plt

from paper_figures import render_standalone_trajectory_source
from paper_trajectory import STANDALONE_TRAJECTORY_SCHEMA_VERSION


def _source_with_all_link_endpoint_types():
    uavs = [
        {"uav_id": uid, "x": 100.0 + uid, "y": 200.0, "z": 90.0}
        for uid in range(16)
    ]
    uav_paths = {
        str(row["uav_id"]): [
            {"actual_time_seconds": 1.0, **{key: row[key] for key in ("x", "y", "z")}}
        ]
        for row in uavs
    }
    sr_teams = [
        {"sr_id": 0, "x": 20.0, "y": 30.0, "z": 0.0},
        {"sr_id": 1, "x": 40.0, "y": 50.0, "z": 0.0},
    ]
    sr_paths = {
        str(row["sr_id"]): [
            {"actual_time_seconds": 1.0, **{key: row[key] for key in ("x", "y", "z")}}
        ]
        for row in sr_teams
    }
    digest = "a" * 64
    return {
        "schema_version": STANDALONE_TRAJECTORY_SCHEMA_VERSION,
        "figure_id": "uav_trajectory_t_1s",
        "method_key": "td3_dinkelbach",
        "requested_time_seconds": 1.0,
        "actual_time_seconds": 1.0,
        "actual_phase": "COM",
        "scenario_id": "scenario",
        "scenario_manifest_hash": digest,
        "checkpoint_path": "checkpoint",
        "checkpoint_metadata_fingerprint": digest,
        "checkpoint_models_sha256": digest,
        "checkpoint_artifact_fingerprint": digest,
        "checkpoint_training_provenance": {
            "training_episode_count": 1,
            "training_git_sha": "train-sha",
        },
        "evaluation_runtime_provenance": {
            "evaluation_episode_count": 1,
            "evaluation_git_sha": "eval-sha",
        },
        "checkpoint_training_episode_count": 1,
        "evaluation_episode_count": 1,
        "checkpoint_training_git_sha": "train-sha",
        "evaluation_git_sha": "eval-sha",
        "git_sha": "render-sha",
        "target_uav_id": 0,
        "environment_width_m": 1000.0,
        "environment_height_m": 1000.0,
        "ground_station": {"gs_id": 16, "x": 0.0, "y": 0.0, "z": 0.0},
        "ground_targets": [],
        "uavs": uavs,
        "uav_paths": uav_paths,
        "sr_teams": sr_teams,
        "sr_paths": sr_paths,
        "active_links": [
            {"sender_id": 1, "receiver_id": 16, "link_type": "S2G"},
            {"sender_id": 0, "receiver_id": 2, "link_type": "S2U"},
            {"sender_id": 3, "receiver_id": 4, "link_type": "U2U"},
            {"sender_id": 5, "receiver_id": 16, "link_type": "U2G"},
        ],
        "sensing_coverage": [],
        "render_contract": {
            "camera": {"elevation_degrees": 25.0, "azimuth_degrees": -60.0},
            "axis_limits": {
                "x": [0.0, 1000.0],
                "y": [0.0, 1000.0],
                "z": [0.0, 180.0],
            },
            "axis_labels": {"x": "X(m)", "y": "Y(m)", "z": "Z(m)"},
            "style_reference": "test",
            "style": {},
        },
    }


def test_link_renderer_uses_typed_sr_uav_and_gs_endpoints():
    source = _source_with_all_link_endpoint_types()
    figure = render_standalone_trajectory_source(source)
    try:
        lines = figure.axes[0].lines
        s2g = next(line for line in lines if line.get_color() == "#00A6A6")
        s2u = next(
            line
            for line in lines
            if line.get_color() == "#243BFF"
            and line.get_linewidth() == 1.3
        )
        u2u = next(
            line
            for line in lines
            if line.get_color() == "red" and line.get_linestyle() == "--"
        )
        u2g = next(line for line in lines if line.get_color() == "purple")

        assert tuple(s2g.get_xdata()) == (40.0, 0.0)
        assert tuple(s2u.get_xdata()) == (20.0, 102.0)
        assert tuple(u2u.get_xdata()) == (103.0, 104.0)
        assert tuple(u2g.get_xdata()) == (105.0, 0.0)
    finally:
        plt.close(figure)
