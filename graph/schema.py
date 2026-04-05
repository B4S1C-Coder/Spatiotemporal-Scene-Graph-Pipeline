"""
Schema setup utilities for the Neo4j graph.

This module initializes indexes and ObjectClass taxonomy nodes required by the
rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.loader import GRAPH_CONFIG_PATH
from graph.neo4j_client import GraphStatement, Neo4jClient, load_graph_config


def initialize_graph_schema(
    neo4j_client: Neo4jClient,
    config_path: str | Path = GRAPH_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Create indexes and ObjectClass taxonomy nodes in Neo4j.

    Args:
        neo4j_client: Initialized Neo4j client.
        config_path: YAML config file location.
        config: Optional runtime override mapping.
    """
    graph_config = load_graph_config(config_path, config=config)
    schema_config = graph_config["schema"]

    index_statements = [
        GraphStatement(query=index_query, parameters={})
        for index_query in schema_config["indexes"]
    ]
    taxonomy_statements = [
        GraphStatement(
            query=(
                "MERGE (c:ObjectClass {name: $name}) "
                "SET c.class_group = $class_group"
            ),
            parameters={
                "name": object_class["name"],
                "class_group": object_class["class_group"],
            },
        )
        for object_class in schema_config["object_classes"]
    ]

    neo4j_client.execute_batch(index_statements + taxonomy_statements)
