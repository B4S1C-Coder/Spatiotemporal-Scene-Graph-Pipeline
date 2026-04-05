"""Tests for ByteTrack integration setup."""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np

from agents.tracking_agent import TrackingAgent, load_bytetrack_config


class FakeBYTETracker:
    """Simple stand-in for the Ultralytics BYTETracker class."""

    def __init__(self, args: object, frame_rate: int) -> None:
        self.args = args
        self.frame_rate = frame_rate
        self.update_calls: list[object] = []

    def update(self, tracker_input: object) -> np.ndarray:
        self.update_calls.append(tracker_input)
        return np.asarray(
            [
                [10.0, 20.0, 30.0, 40.0, 101.0, 0.91, 0.0, 0.0],
                [50.0, 60.0, 70.0, 80.0, 202.0, 0.76, 3.0, 1.0],
            ],
            dtype=float,
        )


def write_bytetrack_config(config_path: Path) -> None:
    """Write a minimal ByteTrack config for tests."""
    config_path.write_text(
        textwrap.dedent(
            """
            tracker_type: bytetrack
            track_high_thresh: 0.5
            track_low_thresh: 0.1
            new_track_thresh: 0.6
            track_buffer: 30
            match_thresh: 0.8
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_bytetrack_config_reads_yaml_values(tmp_path: Path) -> None:
    """ByteTrack settings should be loaded from the YAML config file."""
    config_path = tmp_path / "bytetrack.yaml"
    write_bytetrack_config(config_path)

    tracker_config = load_bytetrack_config(config_path=config_path)

    assert tracker_config["tracker_type"] == "bytetrack"
    assert tracker_config["track_high_thresh"] == 0.5
    assert tracker_config["track_buffer"] == 30


def test_tracking_agent_initializes_bytetrack_from_yaml_config(tmp_path: Path) -> None:
    """TrackingAgent should instantiate BYTETracker with YAML-backed settings."""
    config_path = tmp_path / "bytetrack.yaml"
    write_bytetrack_config(config_path)

    agent = TrackingAgent(
        frame_rate=24,
        config_path=config_path,
        tracker_factory=FakeBYTETracker,
    )

    assert agent.frame_rate == 24
    assert agent.config["tracker_type"] == "bytetrack"
    assert agent.tracker.frame_rate == 24
    assert agent.tracker.args.track_high_thresh == 0.5
    assert agent.tracker.args.track_low_thresh == 0.1
    assert agent.tracker.args.new_track_thresh == 0.6
    assert agent.tracker.args.track_buffer == 30
    assert agent.tracker.args.match_thresh == 0.8


def test_tracking_agent_applies_runtime_config_overrides(tmp_path: Path) -> None:
    """Runtime overrides should merge into the ByteTrack config before initialization."""
    config_path = tmp_path / "bytetrack.yaml"
    write_bytetrack_config(config_path)

    agent = TrackingAgent(
        config_path=config_path,
        config={"track_buffer": 45, "match_thresh": 0.72},
        tracker_factory=FakeBYTETracker,
    )

    assert agent.config["track_buffer"] == 45
    assert agent.config["match_thresh"] == 0.72
    assert agent.tracker.args.track_buffer == 45
    assert agent.tracker.args.match_thresh == 0.72


def test_tracking_agent_converts_detections_into_tracked_output(tmp_path: Path) -> None:
    """TrackingAgent should adapt detections and map tracker rows back to output records."""
    config_path = tmp_path / "bytetrack.yaml"
    write_bytetrack_config(config_path)
    agent = TrackingAgent(
        config_path=config_path,
        tracker_factory=FakeBYTETracker,
    )
    detections = [
        {
            "frame_id": 12,
            "class_id": 0,
            "class_name": "pedestrian",
            "confidence": 0.9,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "occlusion": 1,
        },
        {
            "frame_id": 12,
            "class_id": 3,
            "class_name": "car",
            "confidence": 0.75,
            "bbox": [50.0, 60.0, 70.0, 80.0],
            "occlusion": 0,
        },
    ]

    tracks = agent.track_detections(detections)

    assert len(agent.tracker.update_calls) == 1
    tracker_input = agent.tracker.update_calls[0]
    assert tracker_input.xyxy.tolist() == [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
    assert tracker_input.conf.tolist() == [0.9, 0.75]
    assert tracker_input.cls.tolist() == [0.0, 3.0]
    assert tracks == [
        {
            "track_id": 101,
            "frame_id": 12,
            "class_id": 0,
            "class_name": "pedestrian",
            "confidence": 0.91,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "occlusion": 1,
            "is_new": False,
            "is_lost": False,
        },
        {
            "track_id": 202,
            "frame_id": 12,
            "class_id": 3,
            "class_name": "car",
            "confidence": 0.76,
            "bbox": [50.0, 60.0, 70.0, 80.0],
            "occlusion": 0,
            "is_new": False,
            "is_lost": False,
        },
    ]


def test_tracking_agent_returns_empty_list_for_empty_detections(tmp_path: Path) -> None:
    """Empty detection batches should not call the tracker."""
    config_path = tmp_path / "bytetrack.yaml"
    write_bytetrack_config(config_path)
    agent = TrackingAgent(
        config_path=config_path,
        tracker_factory=FakeBYTETracker,
    )

    tracks = agent.track_detections([])

    assert tracks == []
    assert agent.tracker.update_calls == []
