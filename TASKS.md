Agents must never implement more than one unchecked task in a single step.

## Development Workflow

Coding agents should always follow this workflow:

1. Read `AGENTS.md`
2. Read `CODING_GUIDE.md`
3. Implement **only the next unchecked task**
4. Write tests for the implemented module
5. Stop after completing the task

Agents must **not skip tasks or implement multiple tasks at once**.

---

# Phase 1 — Repository Infrastructure

### Infrastructure


- [x] Add infra/docker-compose.yml for Neo4j
- [x] Add infra/download_visdrone.sh dataset script
- [x] Add infra/setup_env.sh environment setup script


---

# Phase 2 — Dataset Loader

File:

```
pipeline/sequence_loader.py
```

Tasks:


- [ ] Implement VisDrone dataset loader
- [x] Implement VisDrone dataset loader
- [x] Implement sequence metadata parser
- [x] Implement frame iterator interface
- [x] Write tests for sequence_loader

---

# Phase 3 — Detection Agent

File:

```
agents/detection_agent.py
```

Tasks:

- [ ] Implement YOLOv8 model loader
- [x] Implement YOLOv8 model loader
- [x] Implement frame inference function
- [x] Implement detection output formatting
- [x] Load detection configuration from configs/detection.yaml
- [x] Write tests for detection_agent

---

# Phase 4 — Tracking Agent

File:

```
agents/tracking_agent.py
```

Tasks:

- [ ] Integrate ByteTrack tracker
- [x] Integrate ByteTrack tracker
- [x] Implement detection-to-track conversion
- [x] Implement track_id persistence
- [x] Implement lost track handling
- [x] Write tests for tracking_agent

---

# Phase 5 — Motion Agent

File:

```
agents/motion_agent.py
```

Tasks:

- [x] Implement bounding box normalization
- [x] Implement centroid computation
- [x] Implement trajectory buffer
- [x] Implement speed calculation
- [x] Implement heading calculation
- [x] Implement movement pattern classification
- [x] Implement zone assignment
- [x] Write tests for motion_agent

---

# Phase 6 — Graph Layer

### Neo4j client

File:

```
graph/neo4j_client.py
```

Tasks:

- [x] Implement Neo4j connection wrapper
- [x] Implement query execution method
- [x] Implement batch transaction support
- [x] Write tests for neo4j_client

---

### Graph schema

File:

```
graph/schema.py
```

Tasks:

- [x] Implement index creation
- [x] Implement ObjectClass taxonomy initialization
- [x] Write tests for schema setup

---

### Graph agent

File:

```
agents/graph_agent.py
```

Tasks:

- [x] Implement object node MERGE logic
- [x] Implement frame node creation
- [x] Implement zone node creation
- [x] Implement relationship creation (IN_ZONE, APPEARED_IN)
- [x] Implement batch write logic
- [x] Write tests for graph_agent

---

# Phase 7 — Event Detection

File:

```
agents/event_agent.py
```

Tasks:

- [ ] Implement NEAR_MISS detection
- [ ] Implement LOITER detection
- [ ] Implement CONVOY detection
- [ ] Implement CROWD_FORM detection
- [ ] Implement JAYWALKING detection
- [ ] Load thresholds from configs/event.yaml
- [ ] Write tests for event_agent

---

# Phase 8 — Pipeline Runner

File:

```
pipeline/runner.py
```

Tasks:

- [ ] Implement frame processing loop
- [ ] Integrate detection agent
- [ ] Integrate tracking agent
- [ ] Integrate motion agent
- [ ] Integrate graph agent
- [ ] Integrate event agent
- [ ] Implement batch writer integration
- [ ] Write integration tests for pipeline runner

---

# Phase 9 — Entity Resolution

File:

```
agents/entity_resolution_agent.py
```

Tasks:

- [ ] Implement track pair comparison
- [ ] Implement re-identification scoring
- [ ] Implement SAME_ENTITY_AS creation
- [ ] Implement COEXISTS_WITH relationship creation
- [ ] Write tests for entity_resolution_agent

---

# Phase 10 — LLM Query Interface

File:

```
agents/llm_agent.py
```

Tasks:

- [ ] Implement natural language query interface
- [ ] Implement Cypher generation prompt
- [ ] Implement Cypher validation
- [ ] Implement Neo4j query execution
- [ ] Implement answer interpretation pass
- [ ] Write tests for llm_agent

---

# Phase 11 — Evaluation Tools

Files:

```
eval/detection_metrics.py
eval/tracking_metrics.py
eval/event_precision.py
eval/cypher_accuracy.py
```

Tasks:

- [ ] Implement detection evaluation script
- [ ] Implement tracking evaluation script
- [ ] Implement event precision evaluation
- [ ] Implement Cypher generation accuracy evaluation

---

# Phase 12 — Optional UI

File:

```
ui/app.py
```

Tasks:

- [ ] Implement Streamlit interface
- [ ] Add natural language query box
- [ ] Display graph query results
