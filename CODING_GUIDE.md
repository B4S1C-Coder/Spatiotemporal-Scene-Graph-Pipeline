## Purpose

This document defines the **coding standards, repository governance rules, and development workflow** for the *Spatiotemporal Scene Graph Pipeline* project.

All coding agents (human or AI) must follow these rules when modifying the repository.

The system architecture and agent responsibilities are defined in:

```
AGENTS.md
```

This document defines **how code should be written**, not **what the system does**.

### Relationship with AGENTS.md

AGENTS.md defines the system architecture and agent contracts.

CODING_GUIDE.md defines development practices.

If an implementation rule conflicts with the architecture specification,
AGENTS.md takes precedence. In case some directories and files are defined in `CODING_GUIDE.md` but not in `AGENTS.md`, then treat those directories and files as if they were defined in `AGENTS.md`

Constants shown in AGENTS.md examples should be moved to configs/*.yaml
during implementation.

---

# 1. Development Philosophy

This project is developed using **incremental, task-based development**.

Agents must follow these principles:

1. Implement **small modules at a time**
2. Do not modify unrelated components
3. Do not break input/output contracts defined in `AGENTS.md`
4. Always include **tests and documentation with new code**

Agents should avoid making **large multi-module changes in one step**.

---

# 2. Repository Structure Rules

The directory structure defined in `AGENTS.md` must be respected.

New top-level directories should **not be created without explicit approval**.

Current structure:

```
agents/
pipeline/
graph/
data/
weights/
configs/
prompts/
logs/
notebooks/
eval/
ui/
infra/
tests/
docs/
```

### Infrastructure

All infrastructure must live in:

```
infra/
```

Infrastructure includes:

* Docker configuration
* Neo4j setup
* dataset download scripts
* environment bootstrap scripts
* CI pipelines

Example:

```
infra/
docker-compose.yml
neo4j-init.cypher
download_visdrone.sh
setup_env.sh
```

Infrastructure must **not be placed inside agent modules**.

---

# 3. Python Coding Standards

All Python code must follow these rules.

### Type hints required

Every function must include type hints.

Example:

```python
def compute_motion(
    centroid_history: list[tuple[float, float]]
) -> dict[str, float]:
```

Untyped functions are not allowed.

---

### Descriptive names

Avoid short variable names.

Bad:

```python
x, y, z
```

Good:

```
centroid_x
centroid_y
heading_deg
```

---

### Maximum function length

Functions should generally stay under:

```
60 lines
```

If longer, split into helper functions.

---

# 4. Module Documentation

Every Python module must include a **top-level docstring**.

Example:

```python
"""
Motion Agent

Responsible for:
- bounding box normalization
- centroid computation
- speed estimation
- movement classification
"""
```

---

### Function documentation

Public functions must include docstrings.

Example:

```python
def compute_motion(history: list[tuple[float, float]]) -> MotionVector:
    """
    Compute speed and heading from centroid history.

    Args:
        history: list of centroid coordinates

    Returns:
        dict containing speed and heading
    """
```

---

# 5. Testing Requirements

Every module must include tests.

Tests must live in:

```
tests/
```

Example structure:

```
tests/
test_detection_agent.py
test_tracking_agent.py
test_motion_agent.py
test_graph_agent.py
test_event_agent.py
```

---

### Testing framework

Use:

```
pytest
```

---

### Minimum coverage rule

Each public function must have **at least one unit test**.

Integration tests should exist for:

```
pipeline/runner.py
```

---

### Example test

```python
def test_compute_motion_zero_history():
    result = compute_motion([])
    assert result["speed"] == 0
```

---

# 6. Configuration Rules

All constants must live in configuration files.

Allowed locations:

```
configs/
```

Examples:

```
configs/detection.yaml
configs/event.yaml
configs/bytetrack.yaml
configs/llm.yaml
```

Hardcoding thresholds inside code is not allowed.

Bad:

```
NEAR_MISS_DISTANCE_THRESH = 0.05
```

Good:

```
config["near_miss_distance_thresh"]
```

---

# 7. Logging

Agents must use structured logging.

Never use `print()`.

Use:

```python
import logging
logger = logging.getLogger(__name__)
```

Example:

```python
logger.info("Processing frame %s", frame_id)
```

---

# 8. Error Handling

Pipeline execution must be robust.

A single frame failure must **not crash the pipeline**.

Agents should:

* catch recoverable exceptions
* log errors
* continue execution

Example:

```python
try:
    detections = model(frame)
except Exception as e:
    logger.error("Detection failure on frame %s", frame_id)
    detections = []
```

---

# 9. Agent Interface Contracts

Each agent must respect the **input/output schemas defined in `AGENTS.md`**.

Agents must not silently change data formats.

Example contract:

Detection Agent output:

```
{
frame_id
class_id
class_name
confidence
bbox
occlusion
}
```

If a contract must change:

1. update `AGENTS.md`
2. update tests
3. update dependent modules

---

# 10. Dependency Management

Dependencies must be declared in:

```
requirements.txt
```

New dependencies must only be added if necessary.

Heavy ML frameworks should not be duplicated.

Example rule:

Use existing libraries where possible:

```
ultralytics
numpy
pandas
```

---

# 11. Schema and Data Integrity

Graph schema must remain consistent.

Agents writing to Neo4j must:

* use `MERGE` for node creation
* avoid duplicate object nodes
* use indexed fields (`track_id`, `sequence_id`)

Cypher queries must be validated before execution.

---

# 12. Configuration Single Source of Truth

Threshold values must exist in **exactly one place**.

Example:

```
configs/event.yaml
```

Agents must read thresholds from config.

Do not duplicate constants across files.

---

# 13. LLM Query Safety

The LLM Query Agent must:

* only use graph schema defined in prompts
* never invent node labels
* validate Cypher syntax before execution
* handle empty query results gracefully

Empty results must return:

```
"No data found in the graph for this query."
```

---

# 14. Documentation Updates

When code changes:

* update relevant docstrings
* update `README.md` if behavior changes
* update `AGENTS.md` if architecture changes

Documentation must remain synchronized with code.

---

# 15. Development Workflow for Coding Agents

Coding agents should follow this workflow:

1. Read `AGENTS.md`
2. Read `CODING_GUIDE.md`
3. Identify the specific module being implemented
4. Implement minimal functionality
5. Write tests
6. Verify contracts
7. Update documentation

Agents should avoid implementing **multiple agents simultaneously**.

Preferred development order:

1. dataset loader
2. detection agent
3. tracking agent
4. motion agent
5. graph agent
6. event agent
7. LLM agent

---

# 16. Refactoring Policy

Refactoring is allowed only if:

* tests remain passing
* contracts remain unchanged
* architecture defined in `AGENTS.md` is respected

Large refactors should be avoided unless necessary.

---

# 17. Forbidden Practices

The following practices are not allowed:

* using `print()` for debugging
* adding dependencies without updating `requirements.txt`
* modifying agent output schemas silently
* hardcoding configuration values
* writing Neo4j queries without parameterization

---

# 18. Continuous Improvement

If improvements to coding practices are identified:

* update `CODING_GUIDE.md`
* document rationale
* ensure compatibility with `AGENTS.md`

---

# 19. Priority of Documents

If documents conflict, the order of priority is:

```
1. AGENTS.md
2. CODING_GUIDE.md
3. README.md
```

---

# 20. Final Rule

Code should prioritize:

* **clarity**
* **modularity**
* **reproducibility**

over cleverness or premature optimization.
