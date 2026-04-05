"""
Helpers for loading canonical Cypher few-shot examples for the LLM agent.
"""

from __future__ import annotations

from pathlib import Path

from configs.loader import REPO_ROOT


RELATIONSHIPS_PATH = REPO_ROOT / "relationships.md"
PATTERN_SECTION_HEADER = "## 12. Graph Traversal Patterns for LLM Agent"
PATTERN_SECTION_END = "\n## Appendix:"


def load_llm_few_shot_examples(
    relationships_path: str | Path = RELATIONSHIPS_PATH,
) -> str:
    """
    Load the canonical LLM traversal examples from relationships.md.

    Args:
        relationships_path: Path to the project relationships document.

    Returns:
        Markdown section containing the ten traversal patterns.
    """
    content = Path(relationships_path).read_text(encoding="utf-8")
    start_index = content.find(PATTERN_SECTION_HEADER)
    if start_index == -1:
        raise ValueError("Graph traversal pattern section not found in relationships.md")
    end_index = content.find(PATTERN_SECTION_END, start_index)
    if end_index == -1:
        end_index = len(content)
    return content[start_index:end_index].strip()
