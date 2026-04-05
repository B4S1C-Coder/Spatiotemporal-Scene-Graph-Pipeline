"""
Tracking agent integration for Ultralytics ByteTrack.

This module currently covers tracker configuration loading, tracker
initialization, and the basic detection-to-track conversion surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from configs.loader import BYTETRACK_CONFIG_PATH, load_yaml_config
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


def load_bytetrack_config(
    config_path: str | Path = BYTETRACK_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load ByteTrack settings from configs/bytetrack.yaml.

    Args:
        config_path: YAML config file location.
        config: Optional runtime override mapping.

    Returns:
        Tracker configuration dictionary.
    """
    return load_yaml_config(config_path, overrides=config)


class TrackingAgent:
    """Tracking agent wrapper that owns a configured ByteTrack instance."""

    def __init__(
        self,
        frame_rate: int = 30,
        config_path: str | Path = BYTETRACK_CONFIG_PATH,
        config: dict[str, Any] | None = None,
        tracker_factory: Callable[..., Any] = BYTETracker,
    ) -> None:
        """
        Initialize the tracking agent with a ByteTrack tracker.

        Args:
            frame_rate: Sequence frame rate used by ByteTrack buffering.
            config_path: YAML config file location.
            config: Optional runtime config overrides.
            tracker_factory: Injectable tracker constructor for testing.
        """
        self.frame_rate = frame_rate
        self.config_path = Path(config_path)
        self.config = load_bytetrack_config(self.config_path, config=config)
        self.tracker_args = IterableSimpleNamespace(**self.config)
        self.tracker = tracker_factory(args=self.tracker_args, frame_rate=self.frame_rate)

    def track_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert detection records into tracker outputs for the current frame.

        Args:
            detections: Detection-agent output dictionaries for one frame.

        Returns:
            Tracked detection dictionaries with tracker-assigned IDs.
        """
        if not detections:
            return []

        tracker_input = _TrackerInput.from_detections(detections)
        tracked_rows = self.tracker.update(tracker_input)
        return self._format_tracked_rows(tracked_rows, detections)

    @staticmethod
    def _format_tracked_rows(
        tracked_rows: Any,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        formatted_tracks: list[dict[str, Any]] = []
        for tracked_row in np.asarray(tracked_rows):
            detection_index = int(tracked_row[7])
            source_detection = detections[detection_index]
            formatted_tracks.append(
                {
                    "track_id": int(tracked_row[4]),
                    "frame_id": int(source_detection["frame_id"]),
                    "class_id": int(source_detection["class_id"]),
                    "class_name": str(source_detection["class_name"]),
                    "confidence": float(tracked_row[5]),
                    "bbox": [float(coordinate) for coordinate in tracked_row[:4]],
                    "occlusion": int(source_detection["occlusion"]),
                    "is_new": False,
                    "is_lost": False,
                }
            )
        return formatted_tracks


class _TrackerInput:
    """Minimal ByteTrack-compatible detection container."""

    def __init__(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_ids: np.ndarray,
    ) -> None:
        self.xyxy = xyxy
        self.conf = confidence
        self.cls = class_ids
        self.xywh = self._xyxy_to_xywh(xyxy)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, item: Any) -> "_TrackerInput":
        return _TrackerInput(
            xyxy=self.xyxy[item],
            confidence=self.conf[item],
            class_ids=self.cls[item],
        )

    @classmethod
    def from_detections(cls, detections: list[dict[str, Any]]) -> "_TrackerInput":
        xyxy = np.asarray([detection["bbox"] for detection in detections], dtype=float)
        confidence = np.asarray([detection["confidence"] for detection in detections], dtype=float)
        class_ids = np.asarray([detection["class_id"] for detection in detections], dtype=float)
        return cls(xyxy=xyxy, confidence=confidence, class_ids=class_ids)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        if xyxy.size == 0:
            return np.empty((0, 4), dtype=float)

        converted = np.asarray(xyxy, dtype=float).copy()
        converted[:, 2] = converted[:, 2] - converted[:, 0]
        converted[:, 3] = converted[:, 3] - converted[:, 1]
        converted[:, 0] = converted[:, 0] + converted[:, 2] / 2.0
        converted[:, 1] = converted[:, 1] + converted[:, 3] / 2.0
        return converted
