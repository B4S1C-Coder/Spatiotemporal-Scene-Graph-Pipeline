# LLM Agent Phase 10

Phase 10 is implemented across:

- [agents/llm_agent.py](/home/saksham/codebase/deep-learning-project/agents/llm_agent.py)
- [configs/llm.yaml](/home/saksham/codebase/deep-learning-project/configs/llm.yaml)
- [graph/queries.py](/home/saksham/codebase/deep-learning-project/graph/queries.py)
- [graph/validator.py](/home/saksham/codebase/deep-learning-project/graph/validator.py)

The LLM interface now covers the full query loop:

- natural-language question input
- Cypher-generation prompt construction with schema and few-shot patterns
- read-only Cypher validation
- Neo4j query execution
- answer interpretation over graph results
- retry logic for invalid Cypher generations

The implementation is provider-agnostic at the agent boundary through an
injectable LLM client, with a small OpenAI adapter provided for runtime use.

The OpenAI-compatible adapter can now also target a local `llama.cpp` server
through [configs/llm.yaml](/home/saksham/codebase/deep-learning-project/configs/llm.yaml):

- `llm.connection.base_url`
- `llm.connection.api_key`
- `llm.connection.api_key_env`

For `llama.cpp`, point `base_url` at the local OpenAI-compatible `/v1` endpoint.
