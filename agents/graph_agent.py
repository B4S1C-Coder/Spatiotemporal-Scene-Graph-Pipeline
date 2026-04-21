"""
Graph agent for buffering and writing object-state data into Neo4j.

This module batches writes for object, frame, zone, and scene updates using the
graph-layer client abstraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs.loader import GRAPH_CONFIG_PATH
from graph.neo4j_client import GraphStatement, Neo4jClient, load_graph_config


class GraphAgent:
    """Batching graph writer for enriched object states and events."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        config_path: str | Path = GRAPH_CONFIG_PATH,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the graph agent.

        Args:
            neo4j_client: Initialized Neo4j client.
            config_path: YAML config file location.
            config: Optional runtime override mapping.
        """
        self.neo4j_client = neo4j_client
        self.config = load_graph_config(config_path, config=config)
        batch_config = self.config["batch"]
        self.frame_batch_size = int(batch_config["frame_batch_size"])
        self.retry_buffer_path = Path(batch_config["retry_buffer_path"])
        self.pending_statements: list[GraphStatement] = []
        self.pending_frame_count = 0
        self.seen_scenes: set[str] = set()
        self.last_frame_by_sequence: dict[str, int] = {}

    def add_frame_data(
        self,
        object_states: list[dict[str, Any]],
        sequence_id: str,
        frame_id: int,
        events: list[dict[str, Any]] | None = None,
        scene_payload: dict[str, Any] | None = None,
        zone_stats: dict[str, dict[str, float]] | None = None,
    ) -> bool:
        """
        Buffer graph writes for one frame and flush when the batch is full.

        Args:
            object_states: Motion-enriched object states for one frame.
            sequence_id: Sequence identifier.
            frame_id: Current frame ID.
            events: Optional event records.
            scene_payload: Optional scene metadata for first-frame setup.
            zone_stats: Optional per-zone metrics for the frame.

        Returns:
            True when a flush happened, otherwise False.
        """
        if scene_payload and sequence_id not in self.seen_scenes:
            self.pending_statements.extend(self._build_scene_statements(scene_payload))
            self.seen_scenes.add(sequence_id)

        self.pending_statements.extend(self._build_frame_statements(sequence_id, frame_id))

        for object_state in object_states:
            self.pending_statements.extend(self._build_object_statements(object_state, sequence_id))

        if zone_stats:
            self.pending_statements.extend(self._build_zone_statements(zone_stats, sequence_id, frame_id))

        for event in events or []:
            self.pending_statements.extend(self._build_event_statements(event))

        self.pending_frame_count += 1
        if self.pending_frame_count >= self.frame_batch_size:
            self.flush()
            return True
        return False

    def flush(self) -> None:
        """Flush buffered statements to Neo4j."""
        if not self.pending_statements:
            return

        statements_to_flush = list(self.pending_statements)
        frame_count = self.pending_frame_count
        try:
            self.neo4j_client.execute_batch(statements_to_flush)
            self.pending_statements.clear()
            self.pending_frame_count = 0
        except Exception:
            self._buffer_failed_batch(statements_to_flush, frame_count)
            raise

    def _buffer_failed_batch(self, statements: list[GraphStatement], frame_count: int) -> None:
        self.retry_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame_count": frame_count,
            "statements": [
                {"query": statement.query, "parameters": statement.parameters}
                for statement in statements
            ],
        }
        with self.retry_buffer_path.open("a", encoding="utf-8") as buffer_file:
            buffer_file.write(json.dumps(payload) + "\n")

    @staticmethod
    def _build_scene_statements(scene_payload: dict[str, Any]) -> list[GraphStatement]:
        return [
            GraphStatement(
                query=(
                    "MERGE (s:Scene {sequence_id: $sequence_id}) "
                    "SET s.altitude_m = $altitude_m, "
                    "s.weather = $weather, "
                    "s.scene_type = $scene_type, "
                    "s.time_of_day = $time_of_day, "
                    "s.total_frames = $total_frames"
                ),
                parameters={
                    "sequence_id": scene_payload["sequence_id"],
                    "altitude_m": scene_payload["altitude_m"],
                    "weather": scene_payload["weather"],
                    "scene_type": scene_payload["scene_type"],
                    "time_of_day": scene_payload["time_of_day"],
                    "total_frames": scene_payload["total_frames"],
                },
            )
        ]

    def _build_frame_statements(self, sequence_id: str, frame_id: int) -> list[GraphStatement]:
        statements = [
            GraphStatement(
                query=(
                    "MERGE (f:Frame {frame_id: $frame_id, sequence_id: $seq_id})"
                ),
                parameters={"frame_id": frame_id, "seq_id": sequence_id},
            )
        ]
        previous_frame_id = self.last_frame_by_sequence.get(sequence_id)
        if previous_frame_id is not None and previous_frame_id != frame_id:
            statements.append(
                GraphStatement(
                    query=(
                        "MATCH (prev:Frame {frame_id: $prev_frame_id, sequence_id: $seq_id}) "
                        "MATCH (curr:Frame {frame_id: $frame_id, sequence_id: $seq_id}) "
                        "MERGE (prev)-[:PRECEDES]->(curr)"
                    ),
                    parameters={
                        "prev_frame_id": previous_frame_id,
                        "frame_id": frame_id,
                        "seq_id": sequence_id,
                    },
                )
            )
        self.last_frame_by_sequence[sequence_id] = frame_id
        return statements

    @staticmethod
    def _build_object_statements(
        object_state: dict[str, Any],
        sequence_id: str,
    ) -> list[GraphStatement]:
        track_parameters = {
            "track_id": object_state["track_id"],
            "seq_id": sequence_id,
            "class_name": object_state["class_name"],
            "frame_id": object_state["frame_id"],
            "centroid_norm": object_state["centroid_norm"],
            "speed": object_state["speed_px_per_frame"],
            "heading": object_state["heading_deg"],
            "movement_pattern": object_state["movement_pattern"],
            "occlusion": object_state["occlusion"],
            "zone_id": object_state["zone_id"],
            "bbox_norm": object_state["bbox_norm"],
            "status": "active",
        }

        return [
            GraphStatement(
                query=(
                    "MERGE (o:Object {track_id: $track_id, sequence_id: $seq_id}) "
                    "ON CREATE SET o.first_seen_frame = $frame_id "
                    "SET o.class = $class_name, "
                    "o.last_seen_frame = $frame_id, "
                    "o.last_centroid = $centroid_norm, "
                    "o.speed = $speed, "
                    "o.heading = $heading, "
                    "o.movement_pattern = $movement_pattern, "
                    "o.occlusion = $occlusion, "
                    "o.status = $status"
                ),
                parameters=track_parameters,
            ),
            GraphStatement(
                query=(
                    "MERGE (z:Zone {zone_id: $zone_id, sequence_id: $seq_id})"
                ),
                parameters={"zone_id": object_state["zone_id"], "seq_id": sequence_id},
            ),
            GraphStatement(
                query=(
                    "MATCH (o:Object {track_id: $track_id, sequence_id: $seq_id}) "
                    "MATCH (z:Zone {zone_id: $zone_id, sequence_id: $seq_id}) "
                    "MERGE (o)-[r:IN_ZONE]->(z) "
                    "SET r.frame_id = $frame_id, "
                    "r.density_contribution = 1"
                ),
                parameters=track_parameters,
            ),
            GraphStatement(
                query=(
                    "MATCH (o:Object {track_id: $track_id, sequence_id: $seq_id}) "
                    "MATCH (f:Frame {frame_id: $frame_id, sequence_id: $seq_id}) "
                    "MERGE (o)-[r:APPEARED_IN]->(f) "
                    "SET r.centroid = $centroid_norm, "
                    "r.bbox_norm = $bbox_norm"
                ),
                parameters=track_parameters,
            ),
            GraphStatement(
                query=(
                    "MATCH (o:Object {track_id: $track_id, sequence_id: $seq_id}) "
                    "MATCH (s:Scene {sequence_id: $seq_id}) "
                    "MERGE (o)-[:DETECTED_IN]->(s)"
                ),
                parameters={"track_id": object_state["track_id"], "seq_id": sequence_id},
            ),
            GraphStatement(
                query=(
                    "MATCH (o:Object {track_id: $track_id, sequence_id: $seq_id}) "
                    "MATCH (c:ObjectClass {name: $class_name}) "
                    "MERGE (o)-[:BELONGS_TO_CLASS]->(c)"
                ),
                parameters={
                    "track_id": object_state["track_id"],
                    "seq_id": sequence_id,
                    "class_name": object_state["class_name"],
                },
            ),
        ]

    @staticmethod
    def _build_zone_statements(
        zone_stats: dict[str, dict[str, float]],
        sequence_id: str,
        frame_id: int,
    ) -> list[GraphStatement]:
        statements: list[GraphStatement] = []
        for zone_id, stats in zone_stats.items():
            statements.append(
                GraphStatement(
                    query=(
                        "MERGE (z:Zone {zone_id: $zone_id, sequence_id: $seq_id}) "
                        "SET z.last_density = $last_density, "
                        "z.vehicle_ratio = $vehicle_ratio, "
                        "z.pedestrian_ratio = $pedestrian_ratio, "
                        "z.updated_at_frame = $frame_id"
                    ),
                    parameters={
                        "zone_id": zone_id,
                        "seq_id": sequence_id,
                        "frame_id": frame_id,
                        "last_density": stats.get("last_density", 0.0),
                        "vehicle_ratio": stats.get("vehicle_ratio", 0.0),
                        "pedestrian_ratio": stats.get("pedestrian_ratio", 0.0),
                    },
                )
            )
        return statements

    @staticmethod
    def _build_event_statements(event: dict[str, Any]) -> list[GraphStatement]:
        event_key = (
            f"{event['sequence_id']}:"
            f"{event['frame_id']}:"
            f"{event['event_type']}:"
            f"{event['primary_track_id']}:"
            f"{event.get('secondary_track_id')}"
        )
        statements = [
            GraphStatement(
                query=(
                    "CREATE (e:Event {"
                    "event_key: $event_key, "
                    "event_type: $event_type, "
                    "frame_id: $frame_id, "
                    "sequence_id: $sequence_id, "
                    "confidence: $confidence, "
                    "primary_track_id: $primary_track_id, "
                    "secondary_track_id: $secondary_track_id, "
                    "metadata: $metadata"
                    "})"
                ),
                parameters={
                    "event_key": event_key,
                    "event_type": event["event_type"],
                    "frame_id": event["frame_id"],
                    "sequence_id": event["sequence_id"],
                    "confidence": event["confidence"],
                    "primary_track_id": event["primary_track_id"],
                    "secondary_track_id": event.get("secondary_track_id"),
                    "metadata": event.get("metadata", {}),
                },
            ),
            GraphStatement(
                query=(
                    "MATCH (e:Event {event_key: $event_key}) "
                    "MATCH (f:Frame {frame_id: $frame_id, sequence_id: $sequence_id}) "
                    "MERGE (e)-[:OCCURRED_IN]->(f)"
                ),
                parameters={
                    "event_key": event_key,
                    "frame_id": event["frame_id"],
                    "sequence_id": event["sequence_id"],
                },
            ),
            GraphStatement(
                query=(
                    "MATCH (e:Event {event_key: $event_key}) "
                    "MATCH (o:Object {track_id: $primary_track_id, sequence_id: $sequence_id}) "
                    "MERGE (e)-[:INVOLVES {role: 'primary'}]->(o)"
                ),
                parameters={
                    "event_key": event_key,
                    "primary_track_id": event["primary_track_id"],
                    "sequence_id": event["sequence_id"],
                },
            ),
        ]
        if event.get("secondary_track_id") is not None:
            statements.append(
                GraphStatement(
                    query=(
                        "MATCH (e:Event {event_key: $event_key}) "
                        "MATCH (o:Object {track_id: $secondary_track_id, sequence_id: $sequence_id}) "
                        "MERGE (e)-[:INVOLVES {role: 'secondary'}]->(o)"
                    ),
                    parameters={
                        "event_key": event_key,
                        "secondary_track_id": event["secondary_track_id"],
                        "sequence_id": event["sequence_id"],
                    },
                )
            )
        statements.extend(_build_semantic_event_edge_statements(event))
        return statements


def _build_semantic_event_edge_statements(event: dict[str, Any]) -> list[GraphStatement]:
    event_type = str(event["event_type"])
    sequence_id = str(event["sequence_id"])
    frame_id = int(event["frame_id"])
    primary_track_id = int(event["primary_track_id"])
    secondary_track_id = event.get("secondary_track_id")
    metadata = event.get("metadata", {})

    if event_type == "NEAR_MISS" and secondary_track_id is not None:
        return [
            GraphStatement(
                query=(
                    "MATCH (a:Object {track_id: $primary_track_id, sequence_id: $seq_id}) "
                    "MATCH (b:Object {track_id: $secondary_track_id, sequence_id: $seq_id}) "
                    "MERGE (a)-[r:NEAR_MISS]->(b) "
                    "SET r.frame_id = $frame_id, "
                    "r.distance = $distance"
                ),
                parameters={
                    "primary_track_id": primary_track_id,
                    "secondary_track_id": int(secondary_track_id),
                    "seq_id": sequence_id,
                    "frame_id": frame_id,
                    "distance": metadata.get("distance"),
                },
            )
        ]

    if event_type == "CONVOY" and secondary_track_id is not None:
        parameters = {
            "primary_track_id": primary_track_id,
            "secondary_track_id": int(secondary_track_id),
            "seq_id": sequence_id,
            "frame_id": frame_id,
            "avg_distance": metadata.get("distance"),
        }
        return [
            GraphStatement(
                query=(
                    "MATCH (a:Object {track_id: $primary_track_id, sequence_id: $seq_id}) "
                    "MATCH (b:Object {track_id: $secondary_track_id, sequence_id: $seq_id}) "
                    "MERGE (a)-[r:CONVOY_WITH]->(b) "
                    "SET r.frame_id = $frame_id, "
                    "r.avg_distance = $avg_distance"
                ),
                parameters=parameters,
            ),
            GraphStatement(
                query=(
                    "MATCH (a:Object {track_id: $primary_track_id, sequence_id: $seq_id}) "
                    "MATCH (b:Object {track_id: $secondary_track_id, sequence_id: $seq_id}) "
                    "MERGE (b)-[r:CONVOY_WITH]->(a) "
                    "SET r.frame_id = $frame_id, "
                    "r.avg_distance = $avg_distance"
                ),
                parameters=parameters,
            ),
        ]

    if event_type == "LOITER":
        return [
            GraphStatement(
                query=(
                    "MATCH (o:Object {track_id: $primary_track_id, sequence_id: $seq_id}) "
                    "MATCH (z:Zone {zone_id: $zone_id, sequence_id: $seq_id}) "
                    "MERGE (o)-[r:LOITERING_IN]->(z) "
                    "SET r.frame_id = $frame_id"
                ),
                parameters={
                    "primary_track_id": primary_track_id,
                    "seq_id": sequence_id,
                    "zone_id": metadata.get("zone"),
                    "frame_id": frame_id,
                },
            )
        ]

    if event_type == "JAYWALKING":
        return [
            GraphStatement(
                query=(
                    "MATCH (o:Object {track_id: $primary_track_id, sequence_id: $seq_id}) "
                    "MATCH (z:Zone {zone_id: $zone_id, sequence_id: $seq_id}) "
                    "MERGE (o)-[r:JAYWALKING_IN]->(z) "
                    "SET r.frame_id = $frame_id, "
                    "r.vehicle_ratio = $vehicle_ratio"
                ),
                parameters={
                    "primary_track_id": primary_track_id,
                    "seq_id": sequence_id,
                    "zone_id": metadata.get("zone"),
                    "frame_id": frame_id,
                    "vehicle_ratio": metadata.get("vehicle_ratio"),
                },
            )
        ]

    return []
