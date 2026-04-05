# Usage Guide

This document describes how to start, run, and query the project in its
current state.

## What the project currently has

Implemented layers:

- Neo4j Docker setup and connectivity test
- dataset download and environment bootstrap scripts
- VisDrone sequence loading and metadata parsing
- YOLOv8 detection agent
- ByteTrack tracking agent
- motion enrichment and zone assignment
- Neo4j graph writing
- rule-based event detection
- end-to-end pipeline runner integration
- entity-resolution post-processing helpers
- LLM query agent with OpenAI and OpenAI-compatible endpoint support
- evaluation scripts for detection, tracking, events, and Cypher accuracy
- Streamlit UI for graph querying

What is still incomplete or important to note:

- the pipeline runner does not yet expose a finished CLI entrypoint
- `pipeline/post_processor.py` is still empty, so entity resolution is implemented as a module but not yet wired into a sequence-completion command
- the Streamlit UI is focused on natural-language graph queries, not live frame playback or graph visualization
- the evaluation scripts are lightweight JSON-driven evaluators, not full dataset-integrated benchmarking pipelines like TrackEval
- to use real LLM queries, you still need a reachable LLM endpoint and a populated Neo4j graph

## Setup

From the repository root:

```bash
bash infra/setup_env.sh
```

That creates `venv/`, installs `requirements.txt`, creates core directories,
and prepares `.env` if missing.

If you need dataset assets:

```bash
bash infra/download_visdrone.sh
```

Useful variants:

```bash
bash infra/download_visdrone.sh --weights
bash infra/download_visdrone.sh --det
bash infra/download_visdrone.sh --all
```

## Start Neo4j

Start the local Neo4j service:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Check status:

```bash
docker compose -f infra/docker-compose.yml ps
```

The expected local endpoints are:

- Neo4j Browser: `http://localhost:7474`
- Bolt: `bolt://localhost:7687`

Credentials come from the compose/env configuration in
[infra/docker-compose.yml](/home/saksham/codebase/deep-learning-project/infra/docker-compose.yml)
and [configs/graph.yaml](/home/saksham/codebase/deep-learning-project/configs/graph.yaml).

## LLM configuration

The query agent supports:

- OpenAI
- OpenAI-compatible endpoints such as `llama.cpp`

Connection settings live in
[configs/llm.yaml](/home/saksham/codebase/deep-learning-project/configs/llm.yaml).

OpenAI example:

```yaml
llm:
  model: gpt-4o-mini
  connection:
    base_url: null
    api_key: null
    api_key_env: OPENAI_API_KEY
```

`llama.cpp` example:

```yaml
llm:
  model: qwen2.5-7b-instruct
  connection:
    base_url: http://127.0.0.1:8080/v1
    api_key: llama-local
    api_key_env: OPENAI_API_KEY
```

If `api_key` is omitted, the agent falls back to the configured env var. For a
local OpenAI-compatible server, a placeholder key is acceptable.

## Start the UI

Install dependencies first so `streamlit` is available, then run:

```bash
venv/bin/streamlit run ui/app.py
```

The UI currently provides:

- a sequence ID input
- a natural-language query box
- generated Cypher display
- raw Neo4j result row display
- final answer display from the LLM agent

Before using it:

- Neo4j must be running
- the database should already contain graph data
- the LLM endpoint configured in `configs/llm.yaml` must be reachable

## Running Python tests

Focused suites:

```bash
venv/bin/python -m pytest tests/test_llm_agent.py
venv/bin/python -m pytest tests/test_ui_app.py
venv/bin/python -m pytest tests/test_eval_tools.py
```

Full suite:

```bash
venv/bin/python -m pytest tests/test_*.py
```

Note for the dataset download script tests:

- `tests/test_download_visdrone_script.py` now accepts both dry-run outputs
  when assets are absent or already present

## Using the evaluation tools

All evaluation scripts accept JSON input files.

Detection:

```bash
venv/bin/python eval/detection_metrics.py path/to/detection_eval.json
```

Tracking:

```bash
venv/bin/python eval/tracking_metrics.py path/to/tracking_eval.json
```

Events:

```bash
venv/bin/python eval/event_precision.py path/to/event_eval.json
```

Cypher:

```bash
venv/bin/python eval/cypher_accuracy.py path/to/cypher_eval.json
```

Expected input payloads are documented in
[docs/evaluation_tools_phase11.md](/home/saksham/codebase/deep-learning-project/docs/evaluation_tools_phase11.md).

## Watchouts

- `streamlit` was added as a dependency in `requirements.txt`, but if your venv was created before that change you need to reinstall dependencies
- the UI will not function without a live Neo4j instance and a working LLM endpoint
- the repository does not yet provide a polished command that ingests a sequence, runs the full pipeline, and then runs entity resolution automatically
- some modules are implemented primarily as reusable library surfaces plus tests, so usage still requires a small amount of manual orchestration
- the detection and tracking stack depends on model weights and VisDrone data being present in the expected directories

## What still needs to be done

Remaining practical work outside the completed checklist:

- wire entity resolution into an end-of-sequence post-processing command
- add a production-ready CLI for pipeline execution and querying
- add richer graph result visualization in the UI
- add end-to-end ingestion examples that write a real sequence into Neo4j
- add README-level quickstart coverage if you want a single landing page instead of phase docs
