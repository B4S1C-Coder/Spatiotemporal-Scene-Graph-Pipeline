"""
Event agent for rule-based scene-graph event detection.

This module emits discrete event records from motion-enriched active tracks and
maintains the minimal cross-frame state required by the project rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.loader import EVENT_CONFIG_PATH, load_yaml_config


PEDESTRIAN_CLASSES = {"pedestrian", "people", "bicycle", "motor", "tricycle"}
VEHICLE_CLASSES = {"car", "van", "truck", "bus", "awning-tricycle"}


def load_event_config(
    config_path: str | Path = EVENT_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load event-agent settings from configs/event.yaml.

    Args:
        config_path: YAML config file location.
        config: Optional runtime override mapping.

    Returns:
        Event configuration dictionary.
    """
    return load_yaml_config(config_path, overrides=config)


class EventAgent:
    """Stateful detector for spatiotemporal scene events."""

    def __init__(
        self,
        config_path: str | Path = EVENT_CONFIG_PATH,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = load_event_config(config_path, config=config)
        self.convoy_candidate_buffer: dict[tuple[int, int], int] = {}
        self.zone_residence_counter: dict[int, int] = {}
        self.track_zone_state: dict[int, str] = {}
        self.near_miss_last_frame: dict[tuple[int, int], int] = {}
        self.follow_candidate_buffer: dict[tuple[int, int], int] = {}
        self.zone_density_history: dict[str, list[dict[str, int]]] = {}
        self.zone_class_stats: dict[str, dict[str, float]] = {}

    def get_zone_stats_snapshot(self) -> dict[str, dict[str, float]]:
        """
        Return a shallow copy of the current per-zone statistics.

        Returns:
            Mapping of zone ID to current zone metrics.
        """
        return {zone_id: dict(stats) for zone_id, stats in self.zone_class_stats.items()}

    def process_tracks(
        self,
        tracks: dict[int, dict[str, Any]],
        frame_id: int,
        sequence_id: str,
    ) -> list[dict[str, Any]]:
        """
        Detect all configured event types for one frame.

        Args:
            tracks: Active track map keyed by track ID.
            frame_id: Current frame ID.
            sequence_id: Current sequence identifier.

        Returns:
            Event dictionaries for the frame.
        """
        self._update_zone_stats(tracks, frame_id)
        events: list[dict[str, Any]] = []
        events.extend(self._detect_near_miss(tracks, frame_id, sequence_id))
        events.extend(self._detect_loiter(tracks, frame_id, sequence_id))
        events.extend(self._detect_convoy(tracks, frame_id, sequence_id))
        events.extend(self._detect_crowd_form(frame_id, sequence_id))
        events.extend(self._detect_jaywalking(tracks, frame_id, sequence_id))
        return events

    def _detect_near_miss(
        self,
        tracks: dict[int, dict[str, Any]],
        frame_id: int,
        sequence_id: str,
    ) -> list[dict[str, Any]]:
        config = self.config["near_miss"]
        events: list[dict[str, Any]] = []
        pedestrians = [track for track in tracks.values() if track["class_name"] in PEDESTRIAN_CLASSES]
        vehicles = [track for track in tracks.values() if track["class_name"] in VEHICLE_CLASSES]
        for pedestrian in pedestrians:
            for vehicle in vehicles:
                pair = (int(pedestrian["track_id"]), int(vehicle["track_id"]))
                distance = _euclidean(pedestrian["centroid_norm"], vehicle["centroid_norm"])
                if distance >= float(config["distance_threshold"]):
                    continue
                if max(_track_speed(pedestrian), _track_speed(vehicle)) <= float(config["min_speed"]):
                    continue
                last_frame = self.near_miss_last_frame.get(pair)
                if last_frame is not None and frame_id - last_frame < int(config["dedup_frames"]):
                    continue
                self.near_miss_last_frame[pair] = frame_id
                events.append(
                    _build_event(
                        event_type="NEAR_MISS",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        primary_track_id=pair[0],
                        secondary_track_id=pair[1],
                        confidence=1.0 - min(distance / float(config["distance_threshold"]), 1.0),
                        metadata={"distance": distance},
                    )
                )
        return events

    def _detect_loiter(
        self,
        tracks: dict[int, dict[str, Any]],
        frame_id: int,
        sequence_id: str,
    ) -> list[dict[str, Any]]:
        config = self.config["loiter"]
        loiter_classes = set(config["classes"])
        events: list[dict[str, Any]] = []
        for track_id, track in tracks.items():
            zone_id = str(track["zone_id"])
            if self.track_zone_state.get(track_id) != zone_id:
                self.zone_residence_counter[track_id] = 0
                self.track_zone_state[track_id] = zone_id
            if track["class_name"] not in loiter_classes or track["movement_pattern"] != "stationary":
                self.zone_residence_counter[track_id] = 0
                continue
            self.zone_residence_counter[track_id] = self.zone_residence_counter.get(track_id, 0) + 1
            if self.zone_residence_counter[track_id] == int(config["time_threshold_frames"]):
                events.append(
                    _build_event(
                        event_type="LOITER",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        primary_track_id=int(track_id),
                        secondary_track_id=None,
                        confidence=1.0,
                        metadata={"zone": zone_id},
                    )
                )
        return events

    def _detect_convoy(
        self,
        tracks: dict[int, dict[str, Any]],
        frame_id: int,
        sequence_id: str,
    ) -> list[dict[str, Any]]:
        config = self.config["convoy"]
        events: list[dict[str, Any]] = []
        vehicles = [track for track in tracks.values() if track["class_name"] in VEHICLE_CLASSES]
        seen_pairs: set[tuple[int, int]] = set()
        for index, first in enumerate(vehicles):
            for second in vehicles[index + 1 :]:
                if first["class_name"] != second["class_name"]:
                    continue
                pair = tuple(sorted((int(first["track_id"]), int(second["track_id"]))))
                seen_pairs.add(pair)
                distance = _euclidean(first["centroid_norm"], second["centroid_norm"])
                heading_diff = _heading_difference(first["heading_deg"], second["heading_deg"])
                if not (
                    float(config["distance_min"]) <= distance <= float(config["distance_max"])
                    and heading_diff <= float(config["heading_diff_max_deg"])
                ):
                    self.convoy_candidate_buffer[pair] = 0
                    continue
                self.convoy_candidate_buffer[pair] = self.convoy_candidate_buffer.get(pair, 0) + 1
                if self.convoy_candidate_buffer[pair] == int(config["persistence_frames"]):
                    events.append(
                        _build_event(
                            event_type="CONVOY",
                            frame_id=frame_id,
                            sequence_id=sequence_id,
                            primary_track_id=pair[0],
                            secondary_track_id=pair[1],
                            confidence=1.0,
                            metadata={"distance": distance, "heading_diff": heading_diff},
                        )
                    )
        for pair in list(self.convoy_candidate_buffer):
            if pair not in seen_pairs:
                self.convoy_candidate_buffer[pair] = 0
        return events

    def _detect_crowd_form(self, frame_id: int, sequence_id: str) -> list[dict[str, Any]]:
        config = self.config["crowd_form"]
        events: list[dict[str, Any]] = []
        for zone_id, history in self.zone_density_history.items():
            recent = [item for item in history if frame_id - item["frame"] < int(config["time_window_frames"])]
            if not recent:
                continue
            current_density = recent[-1]["count"]
            past_density = recent[0]["count"]
            if current_density >= int(config["density_threshold"]) and current_density > past_density * 2:
                events.append(
                    _build_event(
                        event_type="CROWD_FORM",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        primary_track_id=-1,
                        secondary_track_id=None,
                        confidence=1.0,
                        metadata={"zone": zone_id, "density": current_density},
                    )
                )
        return events

    def _detect_jaywalking(
        self,
        tracks: dict[int, dict[str, Any]],
        frame_id: int,
        sequence_id: str,
    ) -> list[dict[str, Any]]:
        config = self.config["jaywalking"]
        jaywalking_classes = set(config["classes"])
        events: list[dict[str, Any]] = []
        for track_id, track in tracks.items():
            if track["class_name"] not in jaywalking_classes:
                continue
            zone_id = str(track["zone_id"])
            stats = self.zone_class_stats.get(zone_id, {})
            vehicle_ratio = float(stats.get("vehicle_ratio", 0.0))
            if vehicle_ratio <= float(config["vehicle_ratio_threshold"]):
                continue
            events.append(
                _build_event(
                    event_type="JAYWALKING",
                    frame_id=frame_id,
                    sequence_id=sequence_id,
                    primary_track_id=int(track_id),
                    secondary_track_id=None,
                    confidence=vehicle_ratio,
                    metadata={"zone": zone_id, "vehicle_ratio": vehicle_ratio},
                )
            )
        return events

    def _update_zone_stats(self, tracks: dict[int, dict[str, Any]], frame_id: int) -> None:
        zone_counts: dict[str, dict[str, int]] = {}
        for track in tracks.values():
            zone_id = str(track["zone_id"])
            counts = zone_counts.setdefault(zone_id, {"pedestrian": 0, "vehicle": 0})
            if track["class_name"] in PEDESTRIAN_CLASSES:
                counts["pedestrian"] += 1
            if track["class_name"] in VEHICLE_CLASSES:
                counts["vehicle"] += 1

        for zone_id, counts in zone_counts.items():
            total = counts["pedestrian"] + counts["vehicle"]
            vehicle_ratio = counts["vehicle"] / total if total else float(self.config["zones"]["default_vehicle_ratio"])
            pedestrian_ratio = counts["pedestrian"] / total if total else float(
                self.config["zones"]["default_pedestrian_ratio"]
            )
            self.zone_class_stats[zone_id] = {
                "last_density": float(total),
                "vehicle_ratio": vehicle_ratio,
                "pedestrian_ratio": pedestrian_ratio,
                "pedestrian_count": float(counts["pedestrian"]),
                "vehicle_count": float(counts["vehicle"]),
            }
            history = self.zone_density_history.setdefault(zone_id, [])
            history.append({"frame": frame_id, "count": counts["pedestrian"]})
            max_window = int(self.config["crowd_form"]["time_window_frames"])
            if len(history) > max_window:
                del history[0 : len(history) - max_window]


def _build_event(
    event_type: str,
    frame_id: int,
    sequence_id: str,
    primary_track_id: int,
    secondary_track_id: int | None,
    confidence: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "frame_id": frame_id,
        "sequence_id": sequence_id,
        "primary_track_id": primary_track_id,
        "secondary_track_id": secondary_track_id,
        "confidence": confidence,
        "metadata": metadata,
    }


def _euclidean(first: list[float], second: list[float]) -> float:
    delta_x = first[0] - second[0]
    delta_y = first[1] - second[1]
    return (delta_x**2 + delta_y**2) ** 0.5


def _heading_difference(first: float, second: float) -> float:
    raw_difference = abs(first - second) % 360.0
    return min(raw_difference, 360.0 - raw_difference)


def _track_speed(track: dict[str, Any]) -> float:
    if "speed_px_per_frame" in track:
        return float(track["speed_px_per_frame"])
    if "speed" in track:
        return float(track["speed"])
    return 0.0
