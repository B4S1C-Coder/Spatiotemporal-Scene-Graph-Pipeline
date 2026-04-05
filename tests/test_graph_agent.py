"""Tests for graph-agent batching and statement generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.graph_agent import GraphAgent
from graph.neo4j_client import GraphStatement


class FakeNeo4jClient:
    """Neo4j client double that records or fails batch writes."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.executed_batches: list[list[GraphStatement]] = []

    def execute_batch(self, statements: list[GraphStatement]) -> None:
        if self.should_fail:
            raise RuntimeError("neo4j write failed")
        self.executed_batches.append(statements)


def build_object_state() -> dict[str, object]:
    """Create a representative motion-enriched object state."""
    return {
        "track_id": 7,
        "frame_id": 12,
        "class_name": "car",
        "centroid_norm": [0.4, 0.3],
        "bbox_norm": [0.2, 0.1, 0.6, 0.5],
        "speed_px_per_frame": 0.1,
        "heading_deg": 90.0,
        "movement_pattern": "linear",
        "occlusion": 1,
        "zone_id": "cell_1_1",
    }


def build_scene_payload() -> dict[str, object]:
    """Create a representative scene payload."""
    return {
        "sequence_id": "uav0000009_04358_v",
        "altitude_m": 45.0,
        "weather": "clear",
        "scene_type": "urban",
        "time_of_day": "daytime",
        "total_frames": 484,
    }


def test_graph_agent_buffers_and_flushes_frame_batches(tmp_path: Path) -> None:
    """GraphAgent should batch statements and flush at the configured frame count."""
    client = FakeNeo4jClient()
    retry_path = tmp_path / "graph_write_failures.jsonl"
    agent = GraphAgent(
        neo4j_client=client,
        config={
            "batch": {
                "frame_batch_size": 2,
                "retry_buffer_path": str(retry_path),
            }
        },
    )

    flushed_first = agent.add_frame_data(
        object_states=[build_object_state()],
        sequence_id="uav0000009_04358_v",
        frame_id=12,
        scene_payload=build_scene_payload(),
    )
    flushed_second = agent.add_frame_data(
        object_states=[{**build_object_state(), "frame_id": 13}],
        sequence_id="uav0000009_04358_v",
        frame_id=13,
    )

    assert flushed_first is False
    assert flushed_second is True
    assert len(client.executed_batches) == 1
    queries = [statement.query for statement in client.executed_batches[0]]
    assert any("MERGE (s:Scene" in query for query in queries)
    assert any("MERGE (o:Object" in query for query in queries)
    assert any("MERGE (f:Frame" in query for query in queries)
    assert any("MERGE (z:Zone" in query for query in queries)
    assert any("MERGE (o)-[r:IN_ZONE]" in query for query in queries)
    assert any("MERGE (o)-[r:APPEARED_IN]" in query for query in queries)


def test_graph_agent_writes_event_statements(tmp_path: Path) -> None:
    """GraphAgent should include Event write statements when events are supplied."""
    client = FakeNeo4jClient()
    agent = GraphAgent(
        neo4j_client=client,
        config={"batch": {"frame_batch_size": 1, "retry_buffer_path": str(tmp_path / "failures.jsonl")}},
    )

    agent.add_frame_data(
        object_states=[build_object_state()],
        sequence_id="uav0000009_04358_v",
        frame_id=12,
        events=[
            {
                "event_type": "NEAR_MISS",
                "frame_id": 12,
                "sequence_id": "uav0000009_04358_v",
                "primary_track_id": 7,
                "secondary_track_id": 11,
                "confidence": 0.92,
                "metadata": {"distance": 0.03},
            }
        ],
    )

    assert len(client.executed_batches) == 1
    queries = [statement.query for statement in client.executed_batches[0]]
    assert any("CREATE (e:Event" in query for query in queries)
    assert any("MERGE (e)-[:OCCURRED_IN]" in query for query in queries)
    assert any("role: 'primary'" in query for query in queries)
    assert any("role: 'secondary'" in query for query in queries)


def test_graph_agent_buffers_failed_batches_to_jsonl(tmp_path: Path) -> None:
    """Failed writes should be persisted to the retry buffer file."""
    retry_path = tmp_path / "graph_write_failures.jsonl"
    client = FakeNeo4jClient(should_fail=True)
    agent = GraphAgent(
        neo4j_client=client,
        config={"batch": {"frame_batch_size": 1, "retry_buffer_path": str(retry_path)}},
    )

    with pytest.raises(RuntimeError, match="neo4j write failed"):
        agent.add_frame_data(
            object_states=[build_object_state()],
            sequence_id="uav0000009_04358_v",
            frame_id=12,
        )

    payload = json.loads(retry_path.read_text(encoding="utf-8").strip())
    assert payload["frame_count"] == 1
    assert any("MERGE (o:Object" in statement["query"] for statement in payload["statements"])
