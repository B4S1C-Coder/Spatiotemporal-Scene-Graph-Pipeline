"""Validation and connectivity tests for the Neo4j Docker Compose configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from neo4j import GraphDatabase


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPO_ROOT / "infra" / "docker-compose.yml"


def load_compose_config() -> dict[str, object]:
    """Load the repository Docker Compose configuration."""
    with COMPOSE_PATH.open("r", encoding="utf-8") as compose_file:
        return yaml.safe_load(compose_file)


def test_docker_compose_file_exists() -> None:
    """The Neo4j compose file must be present in infra/."""
    assert COMPOSE_PATH.exists()


def test_compose_defines_neo4j_service_with_expected_ports() -> None:
    """The compose file should expose both Neo4j web and Bolt ports."""
    compose_config = load_compose_config()

    services = compose_config["services"]
    neo4j_service = services["neo4j"]

    assert neo4j_service["image"] == "neo4j:5-community"
    assert neo4j_service["ports"] == [
        "${NEO4J_HTTP_PORT:-7474}:7474",
        "${NEO4J_BOLT_PORT:-7687}:7687",
    ]


def test_compose_configures_auth_plugins_and_persistent_volumes() -> None:
    """The service should enable auth, APOC, and durable data mounts."""
    compose_config = load_compose_config()
    neo4j_service = compose_config["services"]["neo4j"]

    environment = neo4j_service["environment"]
    volumes = neo4j_service["volumes"]

    assert environment["NEO4J_AUTH"] == "${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:-your_password}"
    assert environment["NEO4J_PLUGINS"] == '["apoc"]'
    assert volumes == [
        "neo4j_data:/data",
        "neo4j_logs:/logs",
        "neo4j_plugins:/plugins",
        "neo4j_import:/import",
    ]

    assert set(compose_config["volumes"].keys()) == {
        "neo4j_data",
        "neo4j_logs",
        "neo4j_plugins",
        "neo4j_import",
    }


def test_neo4j_container_accepts_bolt_connections() -> None:
    """Verify the compose-managed Neo4j container accepts authenticated queries."""
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password"),
    )

    try:
        verified = driver.verify_connectivity()
        assert verified is None

        with driver.session() as session:
            result = session.run("RETURN 1 AS value")
            record = result.single()

        assert record is not None
        assert record["value"] == 1
    finally:
        driver.close()
