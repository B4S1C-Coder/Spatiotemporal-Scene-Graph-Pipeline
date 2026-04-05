"""
Neo4j client wrapper for graph-layer operations.

This module provides a small abstraction over the official Neo4j driver for
query execution and batched write transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from configs.loader import GRAPH_CONFIG_PATH, load_yaml_config
from neo4j import GraphDatabase


@dataclass(frozen=True)
class GraphStatement:
    """One Cypher statement plus its parameter map."""

    query: str
    parameters: dict[str, Any]


def load_graph_config(
    config_path: str | Path = GRAPH_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load graph-layer settings from configs/graph.yaml.

    Args:
        config_path: YAML config file location.
        config: Optional runtime override mapping.

    Returns:
        Graph configuration dictionary.
    """
    return load_yaml_config(config_path, overrides=config)


class Neo4jClient:
    """Thin wrapper over the Neo4j driver."""

    def __init__(
        self,
        config_path: str | Path = GRAPH_CONFIG_PATH,
        config: dict[str, Any] | None = None,
        driver_factory: Callable[..., Any] = GraphDatabase.driver,
    ) -> None:
        """
        Initialize the Neo4j client from YAML-backed settings.

        Args:
            config_path: YAML config file location.
            config: Optional runtime override mapping.
            driver_factory: Injectable driver constructor for testing.
        """
        self.config_path = Path(config_path)
        self.config = load_graph_config(self.config_path, config=config)
        neo4j_config = self.config["neo4j"]
        self.driver = driver_factory(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"]),
        )

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a read/query statement and return record dictionaries.

        Args:
            query: Cypher statement.
            parameters: Optional parameter map.

        Returns:
            Query results as dictionaries.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def execute_batch(self, statements: list[GraphStatement]) -> None:
        """
        Execute multiple write statements in a single transaction.

        Args:
            statements: Ordered list of statements to execute.
        """
        if not statements:
            return
        with self.driver.session() as session:
            session.execute_write(self._run_batch, statements)

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self.driver.close()

    @staticmethod
    def _run_batch(transaction: Any, statements: list[GraphStatement]) -> None:
        for statement in statements:
            transaction.run(statement.query, statement.parameters)
