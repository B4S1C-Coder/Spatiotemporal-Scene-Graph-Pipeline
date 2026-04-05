"""Tests for the Neo4j client wrapper."""

from __future__ import annotations

from pathlib import Path
import textwrap

from graph.neo4j_client import GraphStatement, Neo4jClient, load_graph_config


class FakeTransaction:
    """Transaction double that records run calls."""

    def __init__(self) -> None:
        self.run_calls: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, parameters: dict[str, object]) -> None:
        self.run_calls.append((query, parameters))


class FakeResult:
    """Result iterable over dictionary-like records."""

    def __iter__(self):
        return iter([{"value": 1}, {"value": 2}])


class FakeSession:
    """Session double that supports run and execute_write."""

    def __init__(self) -> None:
        self.run_calls: list[tuple[str, dict[str, object]]] = []
        self.transaction = FakeTransaction()

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, query: str, parameters: dict[str, object]) -> FakeResult:
        self.run_calls.append((query, parameters))
        return FakeResult()

    def execute_write(self, fn, statements):
        return fn(self.transaction, statements)


class FakeDriver:
    """Driver double that yields a reusable fake session."""

    def __init__(self) -> None:
        self.session_instance = FakeSession()
        self.closed = False

    def session(self) -> FakeSession:
        return self.session_instance

    def close(self) -> None:
        self.closed = True


def write_graph_config(config_path: Path) -> None:
    """Write a minimal graph config for tests."""
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
              object_classes:
                - name: pedestrian
                  class_group: VulnerableRoadUser
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_graph_config_reads_yaml_values(tmp_path: Path) -> None:
    """Graph settings should be loaded from the YAML config file."""
    config_path = tmp_path / "graph.yaml"
    write_graph_config(config_path)

    config = load_graph_config(config_path=config_path)

    assert config["neo4j"]["uri"] == "bolt://localhost:7687"
    assert config["batch"]["frame_batch_size"] == 30


def test_neo4j_client_executes_query_and_returns_records(tmp_path: Path) -> None:
    """execute_query should run Cypher and return record dictionaries."""
    config_path = tmp_path / "graph.yaml"
    write_graph_config(config_path)
    fake_driver = FakeDriver()

    client = Neo4jClient(
        config_path=config_path,
        driver_factory=lambda uri, auth: fake_driver,
    )

    results = client.execute_query("RETURN $value AS value", {"value": 1})

    assert results == [{"value": 1}, {"value": 2}]
    assert fake_driver.session_instance.run_calls == [("RETURN $value AS value", {"value": 1})]


def test_neo4j_client_executes_batch_in_single_transaction(tmp_path: Path) -> None:
    """execute_batch should run all statements via one write transaction."""
    config_path = tmp_path / "graph.yaml"
    write_graph_config(config_path)
    fake_driver = FakeDriver()

    client = Neo4jClient(
        config_path=config_path,
        driver_factory=lambda uri, auth: fake_driver,
    )
    client.execute_batch(
        [
            GraphStatement(query="CREATE (:Node {id: $id})", parameters={"id": 1}),
            GraphStatement(query="CREATE (:Node {id: $id})", parameters={"id": 2}),
        ]
    )

    assert fake_driver.session_instance.transaction.run_calls == [
        ("CREATE (:Node {id: $id})", {"id": 1}),
        ("CREATE (:Node {id: $id})", {"id": 2}),
    ]


def test_neo4j_client_closes_driver(tmp_path: Path) -> None:
    """close should close the underlying driver."""
    config_path = tmp_path / "graph.yaml"
    write_graph_config(config_path)
    fake_driver = FakeDriver()

    client = Neo4jClient(
        config_path=config_path,
        driver_factory=lambda uri, auth: fake_driver,
    )
    client.close()

    assert fake_driver.closed is True
