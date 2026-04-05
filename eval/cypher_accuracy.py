"""
Cypher generation accuracy helpers and CLI.

This module evaluates syntax validity, retry rate, and semantic correctness for
generated Cypher queries against a labeled set of natural-language prompts.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from graph.validator import normalize_cypher, validate_cypher_syntax


def evaluate_cypher_generation(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate generated Cypher cases for syntax validity and correctness.

    Each case may include:
    - `first_attempt_cypher`
    - `final_cypher`
    - `expected_cypher`
    - `expected_results`
    - `actual_results`
    """
    if not cases:
        return {
            "syntax_validity_first_attempt": 0.0,
            "final_syntax_validity": 0.0,
            "semantic_correctness": 0.0,
            "retry_rate": 0.0,
            "case_count": 0,
        }

    first_attempt_valid = 0
    final_valid = 0
    semantic_correct = 0
    retries_used = 0
    for case in cases:
        first_attempt = normalize_cypher(str(case.get("first_attempt_cypher", "")))
        final_cypher = normalize_cypher(str(case.get("final_cypher", first_attempt)))
        first_valid, _ = validate_cypher_syntax(first_attempt)
        final_is_valid, _ = validate_cypher_syntax(final_cypher)
        if first_valid:
            first_attempt_valid += 1
        if final_is_valid:
            final_valid += 1
        if final_cypher != first_attempt:
            retries_used += 1

        expected_results = case.get("expected_results")
        actual_results = case.get("actual_results")
        if expected_results is not None and actual_results is not None:
            if _normalize_result_rows(expected_results) == _normalize_result_rows(actual_results):
                semantic_correct += 1
            continue

        expected_cypher = normalize_cypher(str(case.get("expected_cypher", "")))
        if final_cypher == expected_cypher and expected_cypher:
            semantic_correct += 1

    case_count = len(cases)
    return {
        "syntax_validity_first_attempt": first_attempt_valid / case_count,
        "final_syntax_validity": final_valid / case_count,
        "semantic_correctness": semantic_correct / case_count,
        "retry_rate": retries_used / case_count,
        "case_count": case_count,
        "target_syntax_validity_first_attempt": 0.90,
        "target_semantic_correctness": 0.75,
        "target_retry_rate": 0.25,
    }


def _normalize_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [{str(key): value for key, value in row.items()} for row in rows],
        key=lambda row: json.dumps(row, sort_keys=True),
    )


def _load_cypher_cases(input_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def main() -> None:
    """CLI entry point for Cypher generation evaluation."""
    parser = ArgumentParser(description="Evaluate generated Cypher query cases from JSON.")
    parser.add_argument("input_path", help="Path to JSON containing a cases list.")
    args = parser.parse_args()

    summary = evaluate_cypher_generation(_load_cypher_cases(args.input_path))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
