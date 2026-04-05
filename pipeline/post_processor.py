"""
Post-processing utilities for sequence-completion work.

This module wires entity resolution into a sequence-final pass that reads
object histories from Neo4j, derives the fields required by the
EntityResolutionAgent, and writes the resulting graph statements back in one
batch.
"""

from __future__ import annotations

import math
from typing import Any

from agents.entity_resolution_agent import EntityResolutionAgent
from graph.neo4j_client import Neo4jClient


OBJECT_HISTORY_QUERY = """
MATCH (o:Object {sequence_id: $sequence_id})-[r:APPEARED_IN]->(f:Frame)
RETURN o.track_id AS track_id,
       o.class AS class_name,
       o.heading AS heading_deg,
       o.canonical_id AS canonical_id,
       f.frame_id AS frame_id,
       r.centroid AS centroid
ORDER BY track_id ASC, frame_id ASC
"""


class PostProcessor:
    """Run end-of-sequence graph post-processing tasks."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        entity_resolution_agent: EntityResolutionAgent | None = None,
    ) -> None:
        self.neo4j_client = neo4j_client
        self.entity_resolution_agent = entity_resolution_agent or EntityResolutionAgent()

    def run_sequence(self, sequence_id: str) -> dict[str, Any]:
        """
        Run sequence-final post-processing steps.

        Args:
            sequence_id: Sequence to post-process.

        Returns:
            Summary dictionary with derived object and statement counts.
        """
        history_rows = self.neo4j_client.execute_query(
            OBJECT_HISTORY_QUERY,
            {"sequence_id": sequence_id},
        )
        object_snapshots = build_object_snapshots(history_rows)
        statements = self.entity_resolution_agent.process_sequence_objects(sequence_id, object_snapshots)
        if statements:
            self.neo4j_client.execute_batch(statements)
        return {
            "sequence_id": sequence_id,
            "object_count": len(object_snapshots),
            "statement_count": len(statements),
            "post_processed": True,
        }


def build_object_snapshots(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert APPEARED_IN history rows into entity-resolution object payloads.
    """
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for row in history_rows:
        grouped_rows.setdefault(int(row["track_id"]), []).append(row)

    snapshots: list[dict[str, Any]] = []
    for track_id, rows in grouped_rows.items():
        sorted_rows = sorted(rows, key=lambda row: int(row["frame_id"]))
        trajectory_buffer = [list(row["centroid"]) for row in sorted_rows if row.get("centroid") is not None]
        if not trajectory_buffer:
            continue

        frame_ids = [int(row["frame_id"]) for row in sorted_rows]
        fallback_heading = float(sorted_rows[-1].get("heading_deg") or 0.0)
        snapshots.append(
            {
                "track_id": track_id,
                "class": str(sorted_rows[0]["class_name"]),
                "canonical_id": sorted_rows[0].get("canonical_id"),
                "first_seen_frame": frame_ids[0],
                "last_seen_frame": frame_ids[-1],
                "first_centroid": trajectory_buffer[0],
                "last_centroid": trajectory_buffer[-1],
                "trajectory_buffer": trajectory_buffer,
                "heading_at_entry": _heading_from_points(trajectory_buffer[:2], fallback_heading),
                "heading_at_exit": _heading_from_points(trajectory_buffer[-2:], fallback_heading),
            }
        )
    return snapshots


def _heading_from_points(points: list[list[float]], fallback_heading: float) -> float:
    if len(points) < 2:
        return fallback_heading
    delta_x = float(points[-1][0]) - float(points[0][0])
    delta_y = float(points[-1][1]) - float(points[0][1])
    if delta_x == 0.0 and delta_y == 0.0:
        return fallback_heading
    return math.degrees(math.atan2(delta_y, delta_x)) % 360.0
