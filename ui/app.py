"""
Streamlit UI for the scene-graph query interface.

The module keeps Streamlit imports inside `main()` so helper functions remain
testable in environments where the UI dependency is not installed.
"""

from __future__ import annotations

from typing import Any, Protocol

from agents.llm_agent import LLMQueryAgent
from graph.neo4j_client import Neo4jClient


class QueryAgentProtocol(Protocol):
    """Minimal query-agent contract required by the UI."""

    def query(self, natural_language_query: str, sequence_id: str | None = None) -> dict[str, Any]:
        """Execute a natural-language graph query and return a result payload."""


def run_query(
    query_agent: QueryAgentProtocol,
    natural_language_query: str,
    sequence_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute a UI-issued graph query with input validation and error handling.
    """
    normalized_question = natural_language_query.strip()
    if not normalized_question:
        return {
            "question": normalized_question,
            "cypher": None,
            "results": [],
            "answer": "Enter a natural language question before running a query.",
            "error": "empty_question",
        }

    try:
        return query_agent.query(normalized_question, sequence_id=sequence_id or None)
    except Exception as exc:  # pragma: no cover - exercised via helper tests, not Streamlit
        return {
            "question": normalized_question,
            "cypher": None,
            "results": [],
            "answer": "The query failed before results could be returned.",
            "error": str(exc),
        }


def summarize_query_result(query_result: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact UI summary for one query response.
    """
    result_rows = list(query_result.get("results", []))
    return {
        "answer": str(query_result.get("answer", "")),
        "cypher": query_result.get("cypher"),
        "result_count": len(result_rows),
        "has_error": query_result.get("error") not in (None, ""),
        "error": query_result.get("error"),
    }


def build_default_query_agent() -> LLMQueryAgent:
    """
    Construct the default LLM query agent used by the Streamlit app.
    """
    neo4j_client = Neo4jClient()
    return LLMQueryAgent(neo4j_client=neo4j_client)


def main() -> None:
    """Run the Streamlit application."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - interactive runtime path
        raise RuntimeError(
            "Streamlit is not installed. Install dependencies from requirements.txt "
            "or run `pip install streamlit`."
        ) from exc

    st.set_page_config(page_title="Scene Graph Query UI", layout="wide")
    st.title("Spatiotemporal Scene Graph")
    st.caption("Query the Neo4j scene graph with natural language.")

    with st.sidebar:
        st.header("Status")
        st.write("Pipeline phases 1-12 are implemented in the repository.")
        st.write("The UI currently focuses on graph querying, not live video playback.")
        st.write("Neo4j must be running and reachable before submitting queries.")

    @st.cache_resource
    def _get_query_agent() -> LLMQueryAgent:
        return build_default_query_agent()

    query_agent = _get_query_agent()

    left_column, right_column = st.columns([2, 3])
    with left_column:
        sequence_id = st.text_input("Sequence ID", placeholder="uav0000009_04358_v")
        natural_language_query = st.text_area(
            "Natural Language Query",
            placeholder="Which vehicles were stationary for more than 60 frames?",
            height=180,
        )
        submit = st.button("Run Query", type="primary", use_container_width=True)

    if submit:
        query_result = run_query(query_agent, natural_language_query, sequence_id or None)
        summary = summarize_query_result(query_result)

        with right_column:
            if summary["has_error"]:
                st.error(summary["answer"])
                st.caption(f"Error: {summary['error']}")
            else:
                st.success(summary["answer"])

            st.subheader("Generated Cypher")
            st.code(summary["cypher"] or "No Cypher generated.", language="cypher")

            st.subheader("Raw Result Rows")
            result_rows = list(query_result.get("results", []))
            if result_rows:
                st.json(result_rows)
                st.caption(f"{summary['result_count']} rows returned.")
            else:
                st.info("No result rows returned.")


if __name__ == "__main__":  # pragma: no cover - interactive runtime path
    main()
