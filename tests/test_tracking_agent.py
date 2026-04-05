"""Tests for ByteTrack integration setup."""

from __future__ import annotations

from pathlib import Path
import textwrap

from agents.tracking_agent import TrackingAgent, load_bytetrack_config


class FakeBYTETracker:
    """Simple stand-in for the Ultralytics BYTETracker class."""

    def __init__(self, args: object, frame_rate: int) -> None:
        self.args = args
        self.frame_rate = frame_rate


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
