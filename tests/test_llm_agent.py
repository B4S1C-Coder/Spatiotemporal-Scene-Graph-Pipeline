"""Tests for the LLM query agent."""

from __future__ import annotations

from pathlib import Path
import textwrap

from agents.llm_agent import LLMQueryAgent, OpenAILLMClient, load_llm_config, run_query_cli


class FakeLLMClient:
    """Simple fake LLM client for generation and interpretation tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": model,
            }
        )
        return self.responses.pop(0)


class FakeNeo4jClient:
    """Neo4j client stand-in that records executed queries."""

    def __init__(self, results: list[dict[str, object]] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, object]] = []

    def execute_query(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append({"query": query, "parameters": parameters or {}})
        return list(self.results)


class FakeResponsesAPI:
    """Minimal Responses API stub."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, str]] = []

    def create(self, *, model: str, instructions: str, input: str):
        self.calls.append({"model": model, "instructions": instructions, "input": input})
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeOpenAIClient:
    """OpenAI client stub exposing a responses API."""

    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponsesAPI(output_text)


class RecordingOpenAIFactory:
    """Factory that records OpenAI client construction arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def __call__(self, *, base_url: str | None = None, api_key: str | None = None) -> FakeOpenAIClient:
        self.calls.append({"base_url": base_url, "api_key": api_key})
        return FakeOpenAIClient("MATCH (n) RETURN n")


def write_llm_config(config_path: Path) -> None:
    """Write a minimal LLM config for tests."""
    log_dir = config_path.parent / "logs"
    config_path.write_text(
        textwrap.dedent(
            f"""
            llm:
              model: gpt-4o-mini
              max_retries: 3
              default_limit: 50
              no_data_response: No data found in the graph for this query.
              invalid_query_response: Could not generate a valid query. Please rephrase.
              connection:
                base_url: null
                api_key: null
                api_key_env: OPENAI_API_KEY
            logging:
              validation_failures_path: {log_dir / "validation_failures.jsonl"}
              zero_results_path: {log_dir / "llm_zero_results.jsonl"}
            prompt:
              system_preamble: |
                You are a Cypher query generator.
                IN_ZONE only has frame_id and density_contribution.
                Do not use class='person'; use 'pedestrian' or 'people'.
              grounding_suffix: |
                Never infer facts not present in the graph.
              interpretation_preamble: |
                Interpret the graph results only.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_llm_config_reads_yaml_values(tmp_path: Path) -> None:
    """LLM settings should be loaded from YAML."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)

    config = load_llm_config(config_path=config_path)

    assert config["llm"]["model"] == "gpt-4o-mini"
    assert config["llm"]["max_retries"] == 3
    assert config["llm"]["connection"]["api_key_env"] == "OPENAI_API_KEY"
    assert config["prompt"]["system_preamble"].startswith("You are a Cypher")


def test_openai_client_uses_configured_base_url_and_api_key() -> None:
    """OpenAI-compatible client creation should honor explicit connection settings."""
    factory = RecordingOpenAIFactory()
    client = OpenAILLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="local-token",
        openai_factory=factory,
    )

    output = client.generate(
        system_prompt="System",
        user_prompt="User",
        model="gpt-4o-mini",
    )

    assert factory.calls == [{"base_url": "http://127.0.0.1:8080/v1", "api_key": "local-token"}]
    assert output == "MATCH (n) RETURN n"


def test_llm_query_agent_uses_openai_compatible_connection_settings(tmp_path: Path) -> None:
    """Agent defaults should support OpenAI-compatible endpoints like llama.cpp."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    factory = RecordingOpenAIFactory()
    client = OpenAILLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="llama-local",
        openai_factory=factory,
    )
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=client,
        config_path=config_path,
        config={
            "llm": {
                "connection": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "api_key": "llama-local",
                }
            }
        },
    )

    cypher = agent.generate_cypher("Show all objects", sequence_id="seq_a")

    assert cypher == "MATCH (n) RETURN n"
    assert factory.calls == [{"base_url": "http://127.0.0.1:8080/v1", "api_key": "llama-local"}]


def test_build_system_prompt_includes_schema_and_few_shot_examples(tmp_path: Path) -> None:
    """The system prompt should include the configured preamble and few-shot patterns."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=FakeLLMClient(["RETURN 1"]),
        config_path=config_path,
    )

    prompt = agent.build_system_prompt()

    assert "You are a Cypher query generator." in prompt
    assert "Graph Traversal Patterns for LLM Agent" in prompt
    assert "Pattern 1: Object History" in prompt
    assert "Never infer facts not present in the graph." in prompt
    assert "do not use class='person'" in prompt.lower()
    assert "IN_ZONE only has frame_id and density_contribution" in prompt


def test_validate_cypher_rejects_write_queries(tmp_path: Path) -> None:
    """The validator should reject write-capable Cypher statements."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=FakeLLMClient(["RETURN 1"]),
        config_path=config_path,
    )

    is_valid, error = agent.validate_cypher("MATCH (n) SET n.name = 'bad' RETURN n")

    assert is_valid is False
    assert error == "Cypher query must be read-only."


def test_validate_cypher_rejects_group_by_syntax(tmp_path: Path) -> None:
    """The validator should reject SQL-only GROUP BY syntax."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=FakeLLMClient(["RETURN 1"]),
        config_path=config_path,
    )

    is_valid, error = agent.validate_cypher("MATCH (z:Zone) RETURN z.zone_id, count(*) GROUP BY z.zone_id")

    assert is_valid is False
    assert error == "Cypher query contains SQL-only syntax unsupported by Neo4j."


def test_validate_cypher_rejects_person_class_label(tmp_path: Path) -> None:
    """The validator should reject non-VisDrone person class labels."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=FakeLLMClient(["RETURN 1"]),
        config_path=config_path,
    )

    is_valid, error = agent.validate_cypher("MATCH (o:Object {class: 'person', sequence_id: $seq_id}) RETURN o")

    assert is_valid is False
    assert error == "Cypher query uses class='person'. Use VisDrone class labels such as 'pedestrian' or 'people'."


def test_validate_cypher_rejects_is_active_reference(tmp_path: Path) -> None:
    """The validator should reject unsupported is_active relationship predicates."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=FakeLLMClient(["RETURN 1"]),
        config_path=config_path,
    )

    is_valid, error = agent.validate_cypher(
        "MATCH (o:Object)-[r:IN_ZONE]->(z:Zone) WHERE r.is_active = true RETURN z.zone_id"
    )

    assert is_valid is False
    assert error == (
        "Cypher query references is_active, which is not persisted for the currently supported graph relationships."
    )


def test_query_retries_invalid_cypher_then_executes_valid_query(tmp_path: Path) -> None:
    """The query loop should retry after validation failures and then execute Neo4j."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    llm_client = FakeLLMClient(
        [
            "CREATE (n:Bad) RETURN n",
            "MATCH (o:Object {sequence_id: $seq_id}) RETURN o.track_id AS track_id",
            "Track 7 matched the query.",
        ]
    )
    neo4j_client = FakeNeo4jClient(results=[{"track_id": 7}])
    agent = LLMQueryAgent(
        neo4j_client=neo4j_client,
        llm_client=llm_client,
        config_path=config_path,
    )

    result = agent.query("Which object matched?", sequence_id="seq_a")

    assert result["cypher"] == "MATCH (o:Object {sequence_id: $seq_id}) RETURN o.track_id AS track_id"
    assert result["answer"] == "Track 7 matched the query."
    assert neo4j_client.calls == [
        {
            "query": "MATCH (o:Object {sequence_id: $seq_id}) RETURN o.track_id AS track_id",
            "parameters": {"seq_id": "seq_a"},
        }
    ]
    assert "Previous validation error: Cypher query must be read-only." in llm_client.calls[1]["user_prompt"]


def test_query_returns_invalid_query_response_when_retries_exhausted(tmp_path: Path) -> None:
    """Exhausting all retries should return the configured invalid-query response."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    llm_client = FakeLLMClient(["CREATE (n) RETURN n", "MERGE (n) RETURN n", "DELETE n RETURN n"])
    neo4j_client = FakeNeo4jClient(results=[{"ignored": True}])
    agent = LLMQueryAgent(
        neo4j_client=neo4j_client,
        llm_client=llm_client,
        config_path=config_path,
    )

    result = agent.query("Bad query", sequence_id="seq_a")

    assert result["answer"] == "Could not generate a valid query. Please rephrase."
    assert result["results"] == []
    assert result["error"] == "Cypher query must be read-only."
    assert neo4j_client.calls == []
    validation_log = (tmp_path / "logs" / "validation_failures.jsonl").read_text(encoding="utf-8")
    assert "validator" in validation_log


def test_interpret_results_returns_no_data_message_for_empty_results(tmp_path: Path) -> None:
    """Empty Neo4j results should short-circuit to the configured no-data response."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    llm_client = FakeLLMClient([])
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=llm_client,
        config_path=config_path,
    )

    answer = agent.interpret_results(
        natural_language_query="What happened?",
        cypher="MATCH (n) RETURN n",
        results=[],
    )

    assert answer == "No data found in the graph for this query."
    assert llm_client.calls == []


def test_query_logs_zero_results(tmp_path: Path) -> None:
    """Empty result sets should be logged to the zero-results file."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    llm_client = FakeLLMClient(["MATCH (n) RETURN n"])
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(results=[]),
        llm_client=llm_client,
        config_path=config_path,
    )

    result = agent.query("What happened?", sequence_id="seq_a")

    assert result["answer"] == "No data found in the graph for this query."
    zero_results_log = (tmp_path / "logs" / "llm_zero_results.jsonl").read_text(encoding="utf-8")
    assert "What happened?" in zero_results_log
    assert '"sequence_id": "seq_a"' in zero_results_log


class FailingNeo4jClient(FakeNeo4jClient):
    """Neo4j client stand-in that can fail before succeeding."""

    def __init__(self, side_effects: list[object]) -> None:
        super().__init__(results=[])
        self.side_effects = list(side_effects)

    def execute_query(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append({"query": query, "parameters": parameters or {}})
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def test_query_retries_on_neo4j_execution_error_and_logs_failure(tmp_path: Path) -> None:
    """Neo4j syntax failures should feed back into generation retries."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    llm_client = FakeLLMClient(
        [
            "MATCH (z:Zone) RETURN z.zone_id AS zone_id ORDER z.zone_id",
            "MATCH (z:Zone) RETURN z.zone_id AS zone_id, count(*) AS count",
            "Two zones returned.",
        ]
    )
    neo4j_client = FailingNeo4jClient(
        [
            RuntimeError("Neo.ClientError.Statement.SyntaxError: invalid ORDER clause"),
            [{"zone_id": "cell_0_0", "count": 2}, {"zone_id": "cell_0_1", "count": 1}],
        ]
    )
    agent = LLMQueryAgent(
        neo4j_client=neo4j_client,
        llm_client=llm_client,
        config_path=config_path,
    )

    result = agent.query("how many person are there in zone?", sequence_id="seq_a")

    assert result["answer"] == "Two zones returned."
    assert result["cypher"] == "MATCH (z:Zone) RETURN z.zone_id AS zone_id, count(*) AS count"
    validation_log = (tmp_path / "logs" / "validation_failures.jsonl").read_text(encoding="utf-8")
    assert "neo4j_execute" in validation_log
    assert "invalid ORDER clause" in validation_log
    assert "Previous validation error: Neo.ClientError.Statement.SyntaxError: invalid ORDER clause" in llm_client.calls[1]["user_prompt"]


def test_interpret_results_uses_llm_for_non_empty_results(tmp_path: Path) -> None:
    """Non-empty results should be sent through the interpretation pass."""
    config_path = tmp_path / "llm.yaml"
    write_llm_config(config_path)
    llm_client = FakeLLMClient(["There was one near miss at frame 12."])
    agent = LLMQueryAgent(
        neo4j_client=FakeNeo4jClient(),
        llm_client=llm_client,
        config_path=config_path,
    )

    answer = agent.interpret_results(
        natural_language_query="Show near misses",
        cypher="MATCH (e:Event) RETURN e.frame_id AS frame_id",
        results=[{"frame_id": 12}],
    )

    assert answer == "There was one near miss at frame 12."
    assert llm_client.calls[0]["system_prompt"] == "Interpret the graph results only."
    assert '"frame_id": 12' in llm_client.calls[0]["user_prompt"]


def test_run_query_cli_uses_supplied_query_agent() -> None:
    """The CLI helper should route through an injected query agent."""
    query_agent = type(
        "FakeCLIQueryAgent",
        (),
        {
            "query": lambda self, natural_language_query, sequence_id=None: {
                "question": natural_language_query,
                "cypher": "MATCH (n) RETURN n",
                "results": [{"value": 1}],
                "answer": f"Question for {sequence_id}",
                "error": None,
            }
        },
    )()

    result = run_query_cli("Show all objects", sequence_id="seq_a", query_agent=query_agent)

    assert result["answer"] == "Question for seq_a"
    assert result["cypher"] == "MATCH (n) RETURN n"
