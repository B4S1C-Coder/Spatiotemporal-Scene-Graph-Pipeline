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

- [x] Implement NEAR_MISS detection
- [x] Implement LOITER detection
- [x] Implement CONVOY detection
- [x] Implement CROWD_FORM detection
- [x] Implement JAYWALKING detection
- [x] Load thresholds from configs/event.yaml
- [x] Write tests for event_agent

---

# Phase 8 — Pipeline Runner

File:

```
pipeline/runner.py
```

Tasks:

- [x] Implement frame processing loop
- [x] Integrate detection agent
- [x] Integrate tracking agent
- [x] Integrate motion agent
- [x] Integrate graph agent
- [x] Integrate event agent
- [x] Implement batch writer integration
- [x] Write integration tests for pipeline runner

---

# Phase 9 — Entity Resolution

File:

```
agents/entity_resolution_agent.py
```

Tasks:

- [ ] Implement track pair comparison
- [x] Implement track pair comparison
- [x] Implement re-identification scoring
- [x] Implement SAME_ENTITY_AS creation
- [x] Implement COEXISTS_WITH relationship creation
- [x] Write tests for entity_resolution_agent

---

# Phase 10 — LLM Query Interface

File:

```
agents/llm_agent.py
```

Tasks:

- [x] Implement natural language query interface
- [x] Implement Cypher generation prompt
- [x] Implement Cypher validation
- [x] Implement Neo4j query execution
- [x] Implement answer interpretation pass
- [x] Write tests for llm_agent

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

- [x] Implement detection evaluation script
- [x] Implement tracking evaluation script
- [x] Implement event precision evaluation
- [x] Implement Cypher generation accuracy evaluation

---

# Phase 12 — Optional UI

File:

```
ui/app.py
```

Tasks:

- [x] Implement Streamlit interface
- [x] Add natural language query box
- [x] Display graph query results

---

# Phase 13 — Follow-up Integration and Usability

Files:

```
pipeline/post_processor.py
pipeline/runner.py
agents/entity_resolution_agent.py
ui/app.py
README.md
docs/usage.md
```

Tasks:

- [x] Wire entity resolution into end-of-sequence post-processing
- [x] Add a production CLI for pipeline execution
- [x] Add a production CLI for natural-language graph querying
- [x] Add richer graph result visualization in the UI
- [x] Add an end-to-end ingestion example for writing a real sequence into Neo4j
- [x] Expand README quickstart to cover setup, ingestion, querying, and UI usage

---

# Phase 14 — Graph Expressiveness Upgrade

Files:

```
agents/graph_agent.py
agents/event_agent.py
pipeline/runner.py
pipeline/batch_writer.py
graph/schema.py
```

Tasks:

- [x] Persist object-to-scene `DETECTED_IN` relationships
- [x] Persist object-to-taxonomy `BELONGS_TO_CLASS` relationships
- [x] Persist `Frame-[:PRECEDES]->Frame` temporal ordering edges
- [x] Persist per-frame zone state on `Zone` nodes (`last_density`, `vehicle_ratio`, `pedestrian_ratio`)
- [x] Materialize semantic graph edges from emitted events (`NEAR_MISS`, `CONVOY_WITH`, `LOITERING_IN`, `JAYWALKING_IN`)
- [ ] Persist raw spatial evidence edges (`NEAR`, `APPROACHING`, `OVERLAPS_WITH`)
- [ ] Persist lifecycle fields for condition-style relationships (`is_active`, `valid_from_frame`, `valid_to_frame`, duration tracking)
- [ ] Persist density history as graph data instead of in-memory only
- [ ] Normalize event metadata into stable first-class properties
- [ ] Materialize crowd membership and event-causality edges (`CROWD_MEMBER_OF`, `TRIGGERED_BY`)
- [ ] Persist motion history, not only latest motion snapshot, for graph-native trajectory queries
- [ ] Write tests for the new graph expressiveness layer
