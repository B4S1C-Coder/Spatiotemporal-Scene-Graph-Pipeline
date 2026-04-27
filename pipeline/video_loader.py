"""
Video loader for arbitrary MP4/video files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from configs.loader import DETECTION_CONFIG_PATH, load_yaml_config
from pipeline.sequence_loader import SequenceLoader


class VideoLoader:
    """Load and yield frames from a single video file, mocking a SequenceLoader."""

    def __init__(self, sequence_id: str, config: dict[str, Any] | None = None):
        """
        Initialize the video loader.
        
        Args:
            sequence_id: The full path to the video file.
            config: Loader configuration overrides.
        """
        self.video_path = Path(sequence_id)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        self.sequence_id = self.video_path.stem
        self.config = load_yaml_config(DETECTION_CONFIG_PATH, overrides=config or {})
        self.scene_payload = self._build_scene_payload()

    def _build_scene_payload(self) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        scene_defaults = self.config.get("scene_defaults", {})
        
        return {
            "sequence_id": self.sequence_id,
            "total_frames": total_frames,
            "frame_rate": frame_rate,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "altitude_m": float(scene_defaults.get("altitude_fallback_m", 50.0)),
            "altitude_source": "estimated",
            "weather": str(scene_defaults.get("weather", "clear")),
            "weather_source": "default",
            "scene_type": str(scene_defaults.get("scene_type", "urban")),
            "time_of_day": str(scene_defaults.get("time_of_day", "daytime")),
            "split": "custom",
            "frame_skip": int(self.config.get("vision", {}).get("frame_skip", 1)),
            "annotation_available": False,
        }

    def get_scene_payload(self) -> dict[str, Any]:
        return self.scene_payload

    def iter_frames(self, frame_skip: int | None = None) -> Any:
        effective_frame_skip = int(frame_skip if frame_skip is not None else self.config.get("vision", {}).get("frame_skip", 1))
        if effective_frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer.")

        img_size = int(self.config.get("vision", {}).get("img_size", 1280))

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        frame_idx = 0
        yielded_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % effective_frame_skip == 0:
                # Use SequenceLoader's letterbox method for consistency
                frame_letterboxed, scale, pad_w, pad_h = SequenceLoader._letterbox_frame(frame, img_size)

                yield {
                    "frame": frame,
                    "frame_letterboxed": frame_letterboxed,
                    "frame_id": frame_idx,
                    "scale": scale,
                    "pad_w": pad_w,
                    "pad_h": pad_h,
                    "orig_width": self.scene_payload["frame_width"],
                    "orig_height": self.scene_payload["frame_height"],
                    "sequence_id": self.sequence_id,
                    "frame_skip": effective_frame_skip,
                    "is_static": False,
                    "annotations": None,
                    "scene_payload": self.scene_payload if yielded_count == 0 else None,
                }
                yielded_count += 1

            frame_idx += 1

        cap.release()
