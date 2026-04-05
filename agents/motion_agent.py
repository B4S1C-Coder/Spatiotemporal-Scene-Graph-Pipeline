"""
Motion agent utilities for spatiotemporal enrichment.

This module normalizes bounding boxes, computes centroids and motion signals,
maintains per-track trajectory history, classifies movement patterns, and
assigns coarse spatial zones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.loader import MOTION_CONFIG_PATH, load_yaml_config


def load_motion_config(
    config_path: str | Path = MOTION_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load motion-agent settings from configs/motion.yaml.

    Args:
        config_path: YAML config file location.
        config: Optional runtime override mapping.

    Returns:
        Motion configuration dictionary.
    """
    return load_yaml_config(config_path, overrides=config)


def normalize_bbox(bbox: list[float], frame_w: int, frame_h: int) -> list[float]:
    """
    Normalize a pixel-space bounding box into [0, 1] coordinates.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2].
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Normalized bounding box coordinates.
    """
    x1, y1, x2, y2 = bbox
    return [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h]


def get_centroid(bbox_norm: list[float]) -> list[float]:
    """
    Compute the centroid of a normalized bounding box.

    Args:
        bbox_norm: Normalized bbox as [x1, y1, x2, y2].

    Returns:
        Centroid as [cx, cy].
    """
    x1, y1, x2, y2 = bbox_norm
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def compute_motion(centroid_history: list[list[float]]) -> dict[str, float]:
    """
    Compute speed and heading from centroid history.

    Args:
        centroid_history: Ordered centroid history for one track.

    Returns:
        Dictionary with `speed` and `heading`.
    """
    if len(centroid_history) < 2:
        return {"speed": 0.0, "heading": 0.0}

    delta_x = centroid_history[-1][0] - centroid_history[-2][0]
    delta_y = centroid_history[-1][1] - centroid_history[-2][1]
    speed = (delta_x**2 + delta_y**2) ** 0.5
    heading_deg = _heading_from_delta(delta_x, delta_y)
    return {"speed": speed, "heading": heading_deg}


def classify_movement(
    speed_history: list[float],
    heading_history: list[float],
    stationary_threshold: float,
    erratic_threshold: float,
) -> str:
    """
    Classify a track as stationary, linear, or erratic.

    Args:
        speed_history: Recent per-frame speeds.
        heading_history: Recent per-frame headings.
        stationary_threshold: Speed threshold for stationary classification.
        erratic_threshold: Heading variance threshold for erratic motion.

    Returns:
        Movement pattern label.
    """
    if not speed_history:
        return "stationary"

    average_speed = sum(speed_history) / len(speed_history)
    heading_variance = _variance(heading_history)
    if average_speed < stationary_threshold:
        return "stationary"
    if heading_variance > erratic_threshold:
        return "erratic"
    return "linear"


def assign_zone(centroid_norm: list[float], grid_size: int = 4) -> str:
    """
    Assign a normalized centroid to a grid cell.

    Args:
        centroid_norm: Normalized centroid [cx, cy].
        grid_size: Number of rows/columns in the grid.

    Returns:
        Zone identifier such as `cell_2_3`.
    """
    column = min(int(centroid_norm[0] * grid_size), grid_size - 1)
    row = min(int(centroid_norm[1] * grid_size), grid_size - 1)
    return f"cell_{row}_{column}"


class MotionAgent:
    """Stateful motion agent that enriches tracked detections frame by frame."""

    def __init__(
        self,
        config_path: str | Path = MOTION_CONFIG_PATH,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the motion agent with YAML-backed thresholds.

        Args:
            config_path: YAML config file location.
            config: Optional runtime config overrides.
        """
        self.config = load_motion_config(config_path, config=config)
        self.trajectory_history: dict[int, list[list[float]]] = {}
        self.speed_history: dict[int, list[float]] = {}
        self.heading_history: dict[int, list[float]] = {}

    def enrich_tracks(
        self,
        tracked_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        """
        Enrich tracked detections with motion and zone metadata.

        Args:
            tracked_detections: Tracker output dictionaries for one frame.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Motion-enriched detection dictionaries.
        """
        enriched_tracks: list[dict[str, Any]] = []
        for tracked_detection in tracked_detections:
            bbox_norm = normalize_bbox(tracked_detection["bbox"], frame_width, frame_height)
            centroid_norm = get_centroid(bbox_norm)
            track_id = int(tracked_detection["track_id"])

            trajectory_buffer = self._append_history(self.trajectory_history, track_id, centroid_norm)
            motion = compute_motion(trajectory_buffer)
            speed_buffer = self._append_history(self.speed_history, track_id, motion["speed"])
            heading_buffer = self._append_history(self.heading_history, track_id, motion["heading"])
            movement_pattern = classify_movement(
                speed_history=speed_buffer,
                heading_history=heading_buffer,
                stationary_threshold=float(self.config["stationary_threshold"]),
                erratic_threshold=float(self.config["erratic_threshold"]),
            )

            enriched_tracks.append(
                {
                    "track_id": track_id,
                    "frame_id": int(tracked_detection["frame_id"]),
                    "class_name": str(tracked_detection["class_name"]),
                    "centroid_norm": centroid_norm,
                    "bbox_norm": bbox_norm,
                    "speed_px_per_frame": float(motion["speed"]),
                    "heading_deg": float(motion["heading"]),
                    "movement_pattern": movement_pattern,
                    "occlusion": int(tracked_detection["occlusion"]),
                    "zone_id": assign_zone(centroid_norm, grid_size=int(self.config["grid_size"])),
                    "trajectory_buffer": [list(point) for point in trajectory_buffer],
                }
            )
        return enriched_tracks

    def _append_history(
        self,
        history_store: dict[int, list[Any]],
        track_id: int,
        value: Any,
    ) -> list[Any]:
        buffer = history_store.setdefault(track_id, [])
        buffer.append(value)
        max_size = int(self.config["trajectory_buffer_size"])
        if len(buffer) > max_size:
            del buffer[0 : len(buffer) - max_size]
        return buffer


def _heading_from_delta(delta_x: float, delta_y: float) -> float:
    if delta_x == 0.0 and delta_y == 0.0:
        return 0.0
    import math

    return math.degrees(math.atan2(delta_y, delta_x)) % 360


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)
