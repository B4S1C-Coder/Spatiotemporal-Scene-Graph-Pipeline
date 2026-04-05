"""Tests for event-agent rule detection."""

from __future__ import annotations

from pathlib import Path
import textwrap

from agents.event_agent import EventAgent, load_event_config


def write_event_config(config_path: Path) -> None:
    """Write a minimal event config for tests."""
    config_path.write_text(
        textwrap.dedent(
            """
            near_miss:
              distance_threshold: 0.05
              min_speed: 0.004
              dedup_frames: 15
            loiter:
              time_threshold_frames: 3
              classes:
                - pedestrian
                - people
            convoy:
              distance_min: 0.03
              distance_max: 0.12
              heading_diff_max_deg: 20.0
              persistence_frames: 3
            crowd_form:
              density_threshold: 3
              time_window_frames: 4
            jaywalking:
              vehicle_ratio_threshold: 0.8
              classes:
                - pedestrian
                - people
            zones:
              default_vehicle_ratio: 0.0
              default_pedestrian_ratio: 0.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def build_track(
    track_id: int,
    class_name: str,
    centroid_norm: list[float],
    zone_id: str,
    speed_px_per_frame: float = 0.0,
    heading_deg: float = 0.0,
    movement_pattern: str = "stationary",
) -> dict[str, object]:
    """Create a representative motion-enriched track state."""
    return {
        "track_id": track_id,
        "class_name": class_name,
        "centroid_norm": centroid_norm,
        "zone_id": zone_id,
        "speed_px_per_frame": speed_px_per_frame,
        "heading_deg": heading_deg,
        "movement_pattern": movement_pattern,
    }


def test_load_event_config_reads_yaml_values(tmp_path: Path) -> None:
    """Event settings should be loaded from the YAML config file."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)

    config = load_event_config(config_path=config_path)

    assert config["near_miss"]["distance_threshold"] == 0.05
    assert config["loiter"]["time_threshold_frames"] == 3
    assert config["convoy"]["persistence_frames"] == 3


def test_event_agent_detects_near_miss_and_deduplicates(tmp_path: Path) -> None:
    """Near-miss should trigger once and then respect the deduplication window."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)
    agent = EventAgent(config_path=config_path)
    tracks = {
        1: build_track(1, "pedestrian", [0.10, 0.10], "cell_0_0", speed_px_per_frame=0.01),
        2: build_track(2, "car", [0.13, 0.12], "cell_0_0", speed_px_per_frame=0.02),
    }

    first_events = agent.process_tracks(tracks, frame_id=5, sequence_id="seq-1")
    second_events = agent.process_tracks(tracks, frame_id=10, sequence_id="seq-1")

    assert [event["event_type"] for event in first_events] == ["NEAR_MISS"]
    assert second_events == []


def test_event_agent_detects_loiter_after_threshold(tmp_path: Path) -> None:
    """Loiter should trigger once a stationary pedestrian remains in one zone long enough."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)
    agent = EventAgent(config_path=config_path)
    track = {1: build_track(1, "pedestrian", [0.3, 0.3], "cell_1_1")}

    first = agent.process_tracks(track, frame_id=1, sequence_id="seq-1")
    second = agent.process_tracks(track, frame_id=2, sequence_id="seq-1")
    third = agent.process_tracks(track, frame_id=3, sequence_id="seq-1")

    assert first == []
    assert second == []
    assert [event["event_type"] for event in third] == ["LOITER"]


def test_event_agent_detects_convoy_after_persistence(tmp_path: Path) -> None:
    """Convoy should trigger for vehicle pairs that persist with similar heading and spacing."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)
    agent = EventAgent(config_path=config_path)
    tracks = {
        11: build_track(11, "truck", [0.20, 0.20], "cell_0_0", speed_px_per_frame=0.02, heading_deg=10.0),
        12: build_track(12, "truck", [0.28, 0.21], "cell_0_1", speed_px_per_frame=0.02, heading_deg=18.0),
    }

    first = agent.process_tracks(tracks, frame_id=1, sequence_id="seq-1")
    second = agent.process_tracks(tracks, frame_id=2, sequence_id="seq-1")
    third = agent.process_tracks(tracks, frame_id=3, sequence_id="seq-1")

    assert first == []
    assert second == []
    assert [event["event_type"] for event in third] == ["CONVOY"]


def test_event_agent_detects_crowd_form_from_zone_density_growth(tmp_path: Path) -> None:
    """Crowd formation should trigger when pedestrian density jumps above the threshold."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)
    agent = EventAgent(config_path=config_path)

    frame_one_tracks = {
        1: build_track(1, "pedestrian", [0.2, 0.2], "cell_0_0"),
    }
    frame_two_tracks = {
        1: build_track(1, "pedestrian", [0.2, 0.2], "cell_0_0"),
        2: build_track(2, "pedestrian", [0.22, 0.2], "cell_0_0"),
        3: build_track(3, "pedestrian", [0.24, 0.2], "cell_0_0"),
    }

    first = agent.process_tracks(frame_one_tracks, frame_id=1, sequence_id="seq-1")
    second = agent.process_tracks(frame_two_tracks, frame_id=2, sequence_id="seq-1")

    assert first == []
    assert [event["event_type"] for event in second] == ["CROWD_FORM"]


def test_event_agent_detects_jaywalking_from_vehicle_ratio(tmp_path: Path) -> None:
    """Jaywalking should trigger when a pedestrian is in a vehicle-dominated zone."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)
    agent = EventAgent(config_path=config_path)
    tracks = {
        1: build_track(1, "pedestrian", [0.2, 0.2], "cell_0_0"),
        2: build_track(2, "car", [0.22, 0.2], "cell_0_0"),
        3: build_track(3, "truck", [0.24, 0.2], "cell_0_0"),
        4: build_track(4, "bus", [0.26, 0.2], "cell_0_0"),
        5: build_track(5, "van", [0.28, 0.2], "cell_0_0"),
        6: build_track(6, "car", [0.30, 0.2], "cell_0_0"),
    }

    events = agent.process_tracks(tracks, frame_id=4, sequence_id="seq-1")

    assert [event["event_type"] for event in events] == ["JAYWALKING"]


def test_event_agent_updates_zone_class_stats_each_frame(tmp_path: Path) -> None:
    """Zone class stats should track vehicle and pedestrian ratios for the current frame."""
    config_path = tmp_path / "event.yaml"
    write_event_config(config_path)
    agent = EventAgent(config_path=config_path)
    tracks = {
        1: build_track(1, "pedestrian", [0.1, 0.1], "cell_0_0"),
        2: build_track(2, "car", [0.15, 0.1], "cell_0_0"),
        3: build_track(3, "truck", [0.2, 0.1], "cell_0_0"),
    }

    agent.process_tracks(tracks, frame_id=1, sequence_id="seq-1")

    assert agent.zone_class_stats["cell_0_0"]["pedestrian_ratio"] == 1 / 3
    assert agent.zone_class_stats["cell_0_0"]["vehicle_ratio"] == 2 / 3
