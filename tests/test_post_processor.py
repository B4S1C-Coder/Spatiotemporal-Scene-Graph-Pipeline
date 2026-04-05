"""Tests for sequence post-processing and entity-resolution wiring."""

from __future__ import annotations

from graph.neo4j_client import GraphStatement
from pipeline.post_processor import PostProcessor, build_object_snapshots


class FakeNeo4jClient:
    """Neo4j client stand-in for post-processor tests."""

    def __init__(self, history_rows: list[dict[str, object]]) -> None:
        self.history_rows = history_rows
        self.query_calls: list[dict[str, object]] = []
        self.batch_calls: list[list[GraphStatement]] = []

    def execute_query(self, query: str, parameters: dict[str, object] | None = None) -> list[dict[str, object]]:
        self.query_calls.append({"query": query, "parameters": parameters or {}})
        return list(self.history_rows)

    def execute_batch(self, statements: list[GraphStatement]) -> None:
        self.batch_calls.append(statements)


class FakeEntityResolutionAgent:
    """Entity-resolution stand-in that records the processed object payloads."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def process_sequence_objects(
        self,
        sequence_id: str,
        object_nodes: list[dict[str, object]],
    ) -> list[GraphStatement]:
        self.calls.append({"sequence_id": sequence_id, "object_nodes": object_nodes})
        return [
            GraphStatement(
                query="MATCH (n) RETURN n",
                parameters={"sequence_id": sequence_id},
            )
        ]


def build_history_rows() -> list[dict[str, object]]:
    """Create representative APPEARED_IN rows for one sequence."""
    return [
        {
            "track_id": 10,
            "class_name": "car",
            "heading_deg": 15.0,
            "canonical_id": None,
            "frame_id": 1,
            "centroid": [0.10, 0.10],
        },
        {
            "track_id": 10,
            "class_name": "car",
            "heading_deg": 15.0,
            "canonical_id": None,
            "frame_id": 2,
            "centroid": [0.20, 0.10],
        },
        {
            "track_id": 11,
            "class_name": "car",
            "heading_deg": 20.0,
            "canonical_id": None,
            "frame_id": 5,
            "centroid": [0.25, 0.10],
        },
        {
            "track_id": 11,
            "class_name": "car",
            "heading_deg": 20.0,
            "canonical_id": None,
            "frame_id": 6,
            "centroid": [0.30, 0.10],
        },
    ]


def test_build_object_snapshots_derives_entity_resolution_fields() -> None:
    """History rows should be grouped into entity-resolution object payloads."""
    snapshots = build_object_snapshots(build_history_rows())

    assert len(snapshots) == 2
    assert snapshots[0]["track_id"] == 10
    assert snapshots[0]["first_seen_frame"] == 1
    assert snapshots[0]["last_seen_frame"] == 2
    assert snapshots[0]["trajectory_buffer"] == [[0.10, 0.10], [0.20, 0.10]]
    assert snapshots[0]["heading_at_entry"] == 0.0
    assert snapshots[0]["heading_at_exit"] == 0.0


def test_post_processor_queries_history_and_executes_entity_resolution_batch() -> None:
    """Running the post-processor should query history and write returned statements."""
    neo4j_client = FakeNeo4jClient(build_history_rows())
    entity_resolution_agent = FakeEntityResolutionAgent()
    post_processor = PostProcessor(
        neo4j_client=neo4j_client,
        entity_resolution_agent=entity_resolution_agent,
    )

    summary = post_processor.run_sequence("seq_a")

    assert summary == {
        "sequence_id": "seq_a",
        "object_count": 2,
        "statement_count": 1,
        "post_processed": True,
    }
    assert neo4j_client.query_calls[0]["parameters"] == {"sequence_id": "seq_a"}
    assert entity_resolution_agent.calls[0]["sequence_id"] == "seq_a"
    assert len(entity_resolution_agent.calls[0]["object_nodes"]) == 2
    assert len(neo4j_client.batch_calls) == 1
