"""
Tracking agent integration for Ultralytics ByteTrack.

This module currently covers tracker configuration loading and tracker
initialization only. Detection-to-track conversion is handled in later tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from configs.loader import BYTETRACK_CONFIG_PATH, load_yaml_config
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

