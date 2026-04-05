"""
Read-only Cypher validation helpers for the LLM query agent.
"""

from __future__ import annotations

import re
from typing import Any


WRITE_KEYWORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "CALL DBMS",
)
DISALLOWED_SQL_PATTERNS = (
    "GROUP BY",
    "HAVING ",
)
READ_PREFIXES = ("MATCH", "OPTIONAL MATCH", "WITH", "UNWIND", "CALL")


def validate_cypher_syntax(cypher: str) -> tuple[bool, str | None]:
    """
    Perform lightweight validation for generated read-only Cypher.

    Args:
        cypher: Candidate Cypher query.

    Returns:
        Tuple of validity flag and optional error message.
    """
    normalized = normalize_cypher(cypher)
    if not normalized:
        return False, "Cypher query is empty."
    upper_cypher = normalized.upper()
    if any(keyword in upper_cypher for keyword in WRITE_KEYWORDS):
        return False, "Cypher query must be read-only."
    if any(pattern in upper_cypher for pattern in DISALLOWED_SQL_PATTERNS):
        return False, "Cypher query contains SQL-only syntax unsupported by Neo4j."
    if not upper_cypher.startswith(READ_PREFIXES):
        return False, "Cypher query must start with a read clause."
    if "RETURN " not in upper_cypher:
        return False, "Cypher query must contain RETURN."
    if ";" in normalized:
        return False, "Cypher query must not contain semicolons."
    if _has_unbalanced_brackets(normalized):
        return False, "Cypher query contains unbalanced brackets."
    return True, None


def normalize_cypher(cypher: str) -> str:
    """
    Strip markdown fences and surrounding whitespace from generated Cypher.
    """
    trimmed = cypher.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\n?", "", trimmed)
        trimmed = re.sub(r"\n?```$", "", trimmed)
    return trimmed.strip()


def _has_unbalanced_brackets(cypher: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {value: key for key, value in pairs.items()}
    stack: list[str] = []
    for character in cypher:
        if character in pairs:
            stack.append(character)
            continue
        if character in closers:
            if not stack or stack[-1] != closers[character]:
                return True
            stack.pop()
    return bool(stack)
