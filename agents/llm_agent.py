"""
LLM query agent for natural-language access to the scene graph.

This module builds the Cypher-generation prompt, validates generated queries,
executes them against Neo4j, and performs a second interpretation pass over the
query results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from configs.loader import LLM_CONFIG_PATH, load_yaml_config
from graph.neo4j_client import Neo4jClient
from graph.queries import load_llm_few_shot_examples
from graph.validator import normalize_cypher, validate_cypher_syntax
from openai import OpenAI


class LLMClientProtocol(Protocol):
    """Minimal interface required from an LLM client."""

    def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
        """Return a text completion for the supplied prompts."""


class OpenAILLMClient:
    """Small adapter around the OpenAI Responses API."""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_factory: type[OpenAI] = OpenAI,
    ) -> None:
        self.client = client or openai_factory(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
        )
        return response.output_text.strip()


def load_llm_config(
    config_path: str | Path = LLM_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load LLM-agent settings from YAML.
    """
    return load_yaml_config(config_path, overrides=config)


class LLMQueryAgent:
    """Generate, validate, execute, and interpret Cypher queries."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        llm_client: LLMClientProtocol | None = None,
        config_path: str | Path = LLM_CONFIG_PATH,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.neo4j_client = neo4j_client
        self.config_path = Path(config_path)
        self.config = load_llm_config(self.config_path, config=config)
        self.llm_client = llm_client or OpenAILLMClient(
            base_url=_resolve_llm_base_url(self.config),
            api_key=_resolve_llm_api_key(self.config),
        )

    def build_system_prompt(self) -> str:
        """
        Build the Cypher-generation system prompt with schema and few-shot examples.
        """
        prompt_config = self.config["prompt"]
        few_shot_examples = load_llm_few_shot_examples()
        return (
            f"{prompt_config['system_preamble'].strip()}\n\n"
            f"{few_shot_examples}\n\n"
            f"{prompt_config['grounding_suffix'].strip()}"
        )

    def generate_cypher(
        self,
        natural_language_query: str,
        sequence_id: str | None = None,
        error: str | None = None,
    ) -> str:
        """
        Generate a Cypher query from a natural-language question.
        """
        model = self.config["llm"]["model"]
        user_prompt = self._build_generation_prompt(
            natural_language_query=natural_language_query,
            sequence_id=sequence_id,
            error=error,
        )
        cypher = self.llm_client.generate(
            system_prompt=self.build_system_prompt(),
            user_prompt=user_prompt,
            model=model,
        )
        return normalize_cypher(cypher)

    def validate_cypher(self, cypher: str) -> tuple[bool, str | None]:
        """
        Validate a generated Cypher query.
        """
        return validate_cypher_syntax(cypher)

    def execute_cypher(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a validated Cypher query through the Neo4j client.
        """
        return self.neo4j_client.execute_query(cypher, parameters or {})

    def interpret_results(
        self,
        natural_language_query: str,
        cypher: str,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Turn raw graph results into a concise natural-language answer.
        """
        no_data_response = str(self.config["llm"]["no_data_response"])
        if not results:
            return no_data_response

        prompt_config = self.config["prompt"]
        model = self.config["llm"]["model"]
        user_prompt = (
            f"Original question:\n{natural_language_query}\n\n"
            f"Cypher query:\n{cypher}\n\n"
            f"Results:\n{json.dumps(results, indent=2, sort_keys=True)}"
        )
        return self.llm_client.generate(
            system_prompt=prompt_config["interpretation_preamble"].strip(),
            user_prompt=user_prompt,
            model=model,
        )

    def query(
        self,
        natural_language_query: str,
        sequence_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full NL -> Cypher -> Neo4j -> answer loop with retries.
        """
        max_retries = int(self.config["llm"]["max_retries"])
        invalid_query_response = str(self.config["llm"]["invalid_query_response"])
        last_error: str | None = None
        cypher: str | None = None

        for _attempt in range(max_retries):
            cypher = self.generate_cypher(
                natural_language_query=natural_language_query,
                sequence_id=sequence_id,
                error=last_error,
            )
            is_valid, last_error = self.validate_cypher(cypher)
            if is_valid:
                break
        else:
            return {
                "question": natural_language_query,
                "cypher": cypher,
                "results": [],
                "answer": invalid_query_response,
                "error": last_error,
            }

        parameters = {"seq_id": sequence_id} if sequence_id is not None else {}
        results = self.execute_cypher(cypher, parameters=parameters)
        answer = self.interpret_results(natural_language_query, cypher, results)
        return {
            "question": natural_language_query,
            "cypher": cypher,
            "results": results,
            "answer": answer,
            "error": None,
        }

    def _build_generation_prompt(
        self,
        natural_language_query: str,
        sequence_id: str | None,
        error: str | None,
    ) -> str:
        prompt_lines = [f"Natural language question: {natural_language_query}"]
        if sequence_id is not None:
            prompt_lines.append(f"Sequence scope: {sequence_id}")
            prompt_lines.append("Use the parameter $seq_id for the sequence filter.")
        if error is not None:
            prompt_lines.append(f"Previous validation error: {error}")
            prompt_lines.append("Correct the Cypher and return only the fixed query.")
        return "\n".join(prompt_lines)


def _resolve_llm_base_url(config: dict[str, Any]) -> str | None:
    connection_config = config["llm"].get("connection", {})
    base_url = connection_config.get("base_url")
    if base_url in (None, ""):
        return None
    return str(base_url)


def _resolve_llm_api_key(config: dict[str, Any]) -> str:
    connection_config = config["llm"].get("connection", {})
    configured_api_key = connection_config.get("api_key")
    if configured_api_key not in (None, ""):
        return str(configured_api_key)

    api_key_env_var = str(connection_config.get("api_key_env", "OPENAI_API_KEY"))
    env_api_key = os.getenv(api_key_env_var)
    if env_api_key:
        return env_api_key

    if _resolve_llm_base_url(config) is not None:
        return "llama.cpp"
    return ""
