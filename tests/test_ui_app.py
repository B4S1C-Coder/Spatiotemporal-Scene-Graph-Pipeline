"""Tests for the Streamlit UI helpers."""

from __future__ import annotations

from typing import Any

from ui.app import run_query, summarize_query_result


class FakeQueryAgent:
    """Query-agent stand-in for UI helper tests."""

    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result or {
            "question": "default",
            "cypher": "MATCH (n) RETURN n",
            "results": [{"value": 1}],
            "answer": "One row returned.",
            "error": None,
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def query(self, natural_language_query: str, sequence_id: str | None = None) -> dict[str, Any]:
        self.calls.append(
            {
                "natural_language_query": natural_language_query,
                "sequence_id": sequence_id,
            }
        )
        if self.error is not None:
            raise self.error
        return dict(self.result)


def test_run_query_rejects_empty_questions() -> None:
    """UI queries should fail fast on blank input."""
    agent = FakeQueryAgent()

    result = run_query(agent, "   ", sequence_id="seq_a")

    assert result["answer"] == "Enter a natural language question before running a query."
    assert result["error"] == "empty_question"
    assert agent.calls == []


def test_run_query_forwards_question_and_sequence_id() -> None:
    """Valid UI queries should pass through to the query agent."""
    agent = FakeQueryAgent()

    result = run_query(agent, "Show all cars", sequence_id="seq_a")

    assert result["answer"] == "One row returned."
    assert agent.calls == [{"natural_language_query": "Show all cars", "sequence_id": "seq_a"}]


def test_run_query_captures_agent_errors() -> None:
    """Unexpected query-agent errors should be turned into UI-safe responses."""
    agent = FakeQueryAgent(error=RuntimeError("neo4j unavailable"))

    result = run_query(agent, "Show all cars", sequence_id="seq_a")

    assert result["answer"] == "The query failed before results could be returned."
    assert result["error"] == "neo4j unavailable"


def test_summarize_query_result_returns_ui_summary() -> None:
    """Result summaries should expose the fields needed by the UI."""
    summary = summarize_query_result(
        {
            "answer": "Two rows returned.",
            "cypher": "MATCH (n) RETURN n",
            "results": [{"value": 1}, {"value": 2}],
            "error": None,
        }
    )

    assert summary == {
        "answer": "Two rows returned.",
        "cypher": "MATCH (n) RETURN n",
        "result_count": 2,
        "has_error": False,
        "error": None,
    }
