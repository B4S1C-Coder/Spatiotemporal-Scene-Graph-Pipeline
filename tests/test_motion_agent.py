"""Tests for motion-agent enrichment utilities."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from agents.motion_agent import (
    MotionAgent,
    assign_zone,
    classify_movement,
    compute_motion,
    get_centroid,
    load_motion_config,
    normalize_bbox,
)


def write_motion_config(config_path: Path) -> None:
    """Write a minimal motion config for tests."""
    config_path.write_text(
        textwrap.dedent(
            """
            trajectory_buffer_size: 3
            stationary_threshold: 0.003
            erratic_threshold: 800.0
            grid_size: 4
            default_speed: 0.0
            default_heading_deg: 0.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def round_points(points: list[list[float]]) -> list[list[float]]:
    """Round 2D points for stable float comparisons in tests."""
    return [[round(value, 4) for value in point] for point in points]


def test_load_motion_config_reads_yaml_values(tmp_path: Path) -> None:
    """Motion settings should be loaded from the YAML config file."""
    config_path = tmp_path / "motion.yaml"
    write_motion_config(config_path)

    config = load_motion_config(config_path=config_path)

    assert config["trajectory_buffer_size"] == 3
    assert config["stationary_threshold"] == 0.003
    assert config["grid_size"] == 4


def test_normalize_bbox_returns_unit_space_coordinates() -> None:
    """Pixel-space bounding boxes should be normalized by frame size."""
    assert normalize_bbox([10, 20, 110, 220], frame_w=200, frame_h=400) == [0.05, 0.05, 0.55, 0.55]


def test_get_centroid_returns_bbox_center() -> None:
    """Normalized centroids should be computed from bbox corners."""
    assert get_centroid([0.2, 0.1, 0.6, 0.5]) == [0.4, 0.3]


def test_compute_motion_returns_zero_when_history_is_short() -> None:
    """Motion defaults should apply when fewer than two centroids exist."""
    assert compute_motion([[0.4, 0.3]]) == {"speed": 0.0, "heading": 0.0}


def test_compute_motion_returns_speed_and_heading() -> None:
    """Motion should reflect centroid displacement between consecutive frames."""
    motion = compute_motion([[0.4, 0.3], [0.4, 0.5]])

    assert motion["speed"] == 0.2
    assert motion["heading"] == 90.0


def test_classify_movement_handles_stationary_linear_and_erratic() -> None:
    """Movement classification should follow speed and heading variance thresholds."""
    assert classify_movement([0.001, 0.002], [0.0, 0.0], 0.003, 800.0) == "stationary"
    assert classify_movement([0.01, 0.012], [10.0, 12.0], 0.003, 800.0) == "linear"
    assert classify_movement([0.01, 0.012, 0.02], [0.0, 90.0, 180.0], 0.003, 800.0) == "erratic"


def test_assign_zone_maps_centroid_to_grid_cell() -> None:
    """Zone assignment should place centroids into the expected 4x4 grid cell."""
    assert assign_zone([0.74, 0.26], grid_size=4) == "cell_1_2"


def test_motion_agent_enriches_tracks_and_maintains_buffers(tmp_path: Path) -> None:
    """MotionAgent should enrich tracked detections and cap trajectory history."""
    config_path = tmp_path / "motion.yaml"
    write_motion_config(config_path)
    agent = MotionAgent(config_path=config_path)

    frame_one = agent.enrich_tracks(
        [
            {
                "track_id": 7,
                "frame_id": 0,
                "class_name": "car",
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "occlusion": 0,
            }
        ],
        frame_width=200,
        frame_height=200,
    )
    frame_two = agent.enrich_tracks(
        [
            {
                "track_id": 7,
                "frame_id": 1,
                "class_name": "car",
                "bbox": [20.0, 0.0, 120.0, 100.0],
                "occlusion": 1,
            }
        ],
        frame_width=200,
        frame_height=200,
    )
    frame_three = agent.enrich_tracks(
        [
            {
                "track_id": 7,
                "frame_id": 2,
                "class_name": "car",
                "bbox": [40.0, 0.0, 140.0, 100.0],
                "occlusion": 2,
            }
        ],
        frame_width=200,
        frame_height=200,
    )
    frame_four = agent.enrich_tracks(
        [
            {
                "track_id": 7,
                "frame_id": 3,
                "class_name": "car",
                "bbox": [60.0, 0.0, 160.0, 100.0],
                "occlusion": 0,
            }
        ],
        frame_width=200,
        frame_height=200,
    )

    assert frame_one[0]["speed_px_per_frame"] == 0.0
    assert frame_one[0]["heading_deg"] == 0.0
    assert frame_one[0]["movement_pattern"] == "stationary"
    assert frame_one[0]["zone_id"] == "cell_1_1"

    assert frame_two[0]["speed_px_per_frame"] == pytest.approx(0.1)
    assert frame_two[0]["heading_deg"] == 0.0
    assert frame_two[0]["movement_pattern"] == "linear"
    assert round_points(frame_two[0]["trajectory_buffer"]) == [[0.25, 0.25], [0.35, 0.25]]

    assert round_points(frame_three[0]["trajectory_buffer"]) == [[0.25, 0.25], [0.35, 0.25], [0.45, 0.25]]
    assert round_points(frame_four[0]["trajectory_buffer"]) == [[0.35, 0.25], [0.45, 0.25], [0.55, 0.25]]
    assert frame_four[0]["occlusion"] == 0
