"""Tests for graph schema setup."""

from __future__ import annotations

from pathlib import Path
import textwrap

from graph.neo4j_client import GraphStatement
from graph.schema import initialize_graph_schema


class FakeNeo4jClient:
    """Neo4j client double that records batch statements."""

    def __init__(self) -> None:
        self.executed_batches: list[list[GraphStatement]] = []

    def execute_batch(self, statements: list[GraphStatement]) -> None:
        self.executed_batches.append(statements)


def write_graph_config(config_path: Path) -> None:
    """Write a graph config with indexes and object taxonomy."""
    config_path.write_text(
        textwrap.dedent(
            """
            neo4j:
              uri: bolt://localhost:7687
              user: neo4j
              password: your_password
            batch:
              frame_batch_size: 30
              retry_buffer_path: logs/graph_write_failures.jsonl
            schema:
              indexes:
                - CREATE INDEX object_track IF NOT EXISTS FOR (o:Object) ON (o.track_id, o.sequence_id)
                - CREATE INDEX zone_id IF NOT EXISTS FOR (z:Zone) ON (z.zone_id, z.sequence_id)
              object_classes:
                - name: pedestrian
                  class_group: VulnerableRoadUser
                - name: car
                  class_group: MotorVehicle
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_initialize_graph_schema_creates_indexes_and_taxonomy(tmp_path: Path) -> None:
    """Schema initialization should submit index and ObjectClass statements."""
    config_path = tmp_path / "graph.yaml"
    write_graph_config(config_path)
    client = FakeNeo4jClient()

    initialize_graph_schema(client, config_path=config_path)

    assert len(client.executed_batches) == 1
    statements = client.executed_batches[0]
    assert len(statements) == 4
    assert statements[0].query.startswith("CREATE INDEX object_track")
    assert statements[1].query.startswith("CREATE INDEX zone_id")
    assert statements[2].parameters == {"name": "pedestrian", "class_group": "VulnerableRoadUser"}
    assert statements[3].parameters == {"name": "car", "class_group": "MotorVehicle"}
