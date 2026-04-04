# Neo4j Docker Setup

This project's local Neo4j instance is defined in:

`infra/docker-compose.yml`

## What It Provides

- Neo4j Community Edition on HTTP port `7474`
- Neo4j Bolt protocol on port `7687`
- Persistent named volumes for data, logs, plugins, and imports
- APOC plugin enabled for graph utility procedures

## Environment Variables

The compose file supports these optional overrides:

- `NEO4J_USER` default: `neo4j`
- `NEO4J_PASSWORD` default: `your_password`
- `NEO4J_HTTP_PORT` default: `7474`
- `NEO4J_BOLT_PORT` default: `7687`
- `NEO4J_HEAP_INITIAL_SIZE` default: `512m`
- `NEO4J_HEAP_MAX_SIZE` default: `512m`
- `NEO4J_PAGECACHE_SIZE` default: `512m`
- `NEO4J_MIN_PASSWORD_LENGTH` default: `8`

## Run

From the repository root:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Stop the service with:

```bash
docker compose -f infra/docker-compose.yml down
```

## Access

- Browser UI: `http://localhost:7474`
- Bolt URI: `bolt://localhost:7687`

Use the same username and password values passed through the compose environment.
