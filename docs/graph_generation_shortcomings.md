# Graph Generation Shortcomings

This document describes the gap between the intended ontology in
[AGENTS.md](/home/saksham/codebase/deep-learning-project/AGENTS.md) /
[relationships.md](/home/saksham/codebase/deep-learning-project/relationships.md)
and the graph that is actually written today by
[agents/graph_agent.py](/home/saksham/codebase/deep-learning-project/agents/graph_agent.py)
plus the sequence-final post-processing step.

The pipeline is end-to-end runnable, but the current graph is still too thin to
support many meaningful investigative questions reliably.

## What the graph actually contains today

Persisted node types:

- `Object`
- `Frame`
- `Zone`
- `Scene`
- `Event`
- `ObjectClass` taxonomy nodes

Persisted relationship types:

- `IN_ZONE`
- `APPEARED_IN`
- `OCCURRED_IN`
- `INVOLVES`
- `SAME_ENTITY_AS`
- `COEXISTS_WITH`

Partially persisted object state:

- object class
- last seen frame
- last centroid
- latest speed
- latest heading
- latest movement pattern
- latest occlusion
- current status
- optional `canonical_id` from entity resolution

That is enough for:

- per-object frame history via `APPEARED_IN`
- coarse per-frame zone occupancy
- event lookup through `Event` nodes
- post-hoc track stitching via `SAME_ENTITY_AS`

It is not enough for most of the richer spatial, temporal, and semantic queries
described in the ontology docs.

## Main shortcomings

### 1. The graph mostly stores snapshots, not relationship state

`Object` nodes are updated with the latest speed, heading, movement pattern, and
occlusion. Earlier values are not persisted as a queryable history.

Impact:

- you can ask what an object's latest motion state is
- you cannot reliably ask how its speed evolved over time
- you cannot compare motion before and after an event
- you cannot reconstruct heading history from the graph alone

Why this matters:

Meaningful surveillance questions are usually about change, persistence, or
deviation over time. A latest-value overwrite model is weak for all three.

### 2. `IN_ZONE` is too thin for zone analytics

The current `IN_ZONE` edge only stores:

- `frame_id`
- `density_contribution`

It does not persist:

- entry frame
- duration in zone
- validity window
- active/inactive state
- centroid on the relationship

Impact:

- you can count current-ish object-to-zone assignments
- you cannot reliably answer dwell-time questions from `IN_ZONE`
- you cannot query historical zone occupancy as a first-class relationship
- you cannot distinguish "currently in zone" from "was once in zone"

The current implementation also merges one `IN_ZONE` edge per object-zone pair,
so zone presence is effectively collapsed rather than modeled as an explicit
lifecycle.

### 3. Zone nodes do not carry the density layer promised by the ontology

`Zone` nodes are created, but the graph writer does not currently persist:

- `vehicle_ratio`
- `pedestrian_ratio`
- `last_density`
- density history
- per-frame zone counts

Those statistics exist only transiently inside
[agents/event_agent.py](/home/saksham/codebase/deep-learning-project/agents/event_agent.py)
while rules are being evaluated.

Impact:

- analytical zone questions are largely unsupported
- "which zones are vehicle-dominated?" is not graph-grounded
- "which zones consistently have high pedestrian density?" is not answerable
- `JAYWALKING` logic may fire, but the zone evidence is not persisted for later auditing

### 4. Event nodes exist, but the event semantics are not materialized as graph edges

The current graph writes `Event` nodes plus `INVOLVES` and `OCCURRED_IN`.
It does not currently write the ontology-level semantic relationships that make
the graph naturally traversable:

- `NEAR_MISS`
- `CONVOY_WITH`
- `LOITERING_IN`
- `JAYWALKING_IN`
- `FOLLOWING`
- `CROWD_MEMBER_OF`
- `TRIGGERED_BY`

Impact:

- event lookup works only through `Event` node filtering
- semantic traversal patterns from `relationships.md` do not actually exist
- pairwise behavioral questions are much harder to express and validate
- the graph is closer to an event log than a rich scene graph

This is one of the biggest reasons the current graph feels low-impact.

### 5. The raw spatial layer is missing

The ontology expects foundational spatial relationships such as:

- `NEAR`
- `APPROACHING`
- `OVERLAPS_WITH`

These are not currently written.

Impact:

- there is no graph-native proximity layer underneath higher-level events
- you cannot answer "who was near whom" without recomputing geometry externally
- event explanations are weaker because the supporting evidence is not stored

Without this layer, the graph cannot support meaningful "show me nearby objects"
or "who approached this vehicle" style queries with confidence.

### 6. The hierarchical layer is incomplete

The ontology expects:

- `(Object)-[:DETECTED_IN]->(Scene)`
- `(Object)-[:BELONGS_TO_CLASS]->(ObjectClass)`

The schema initializer creates `ObjectClass` nodes, but the graph writer does
not currently link objects to them. `DETECTED_IN` is also not written.

Impact:

- class-group traversal promised to the LLM is not actually grounded in the graph
- object-to-scene traversal is missing
- cross-sequence and scene-context questions are harder than they should be

Right now, class grouping often falls back to string conventions rather than
true ontology traversal.

### 7. Frame-to-frame temporal structure is incomplete

`APPEARED_IN` is written, but `Frame-[:PRECEDES]->Frame` is not.

Impact:

- ordered histories can still be reconstructed by sorting `frame_id`
- explicit temporal traversal patterns from the ontology are unavailable
- event chains and temporal reasoning remain weaker than advertised

This is not the biggest gap, but it contributes to the graph feeling flatter
than intended.

### 8. Event metadata is stored as opaque maps

Event-specific facts are currently placed into `Event.metadata` as a generic map.

Impact:

- ad hoc metadata is easy to write
- querying becomes inconsistent and harder to optimize
- LLM-generated Cypher has to guess metadata keys
- important event evidence is not normalized into stable properties or edges

For example, a near-miss distance is more useful as a stable event property and
possibly as an object-object semantic edge than as opaque metadata alone.

### 9. The graph lacks auditability for why an event fired

The current graph keeps event outputs, but not the full evidence chain:

- no persisted `NEAR` edge for near-miss support
- no persisted zone density timeline for crowd formation
- no persisted loiter duration edge/state
- no persisted convoy relationship state across frames

Impact:

- the graph can tell you that an event was emitted
- it often cannot show why, using only graph-native evidence

That weakens trust and makes the LLM interface brittle, because explanations are
not well grounded in stored structure.

### 10. The ontology docs currently overstate the queryable graph

Several documents describe a graph that includes lifecycle-aware condition
relationships, taxonomy traversal, and richer semantic edges. The current graph
does not yet match that shape.

Impact:

- the LLM prompt is easy to misalign
- users assume more graph semantics exist than are actually stored
- queries can be syntactically valid but semantically empty

This is now a documentation problem as much as an implementation problem.

## What would materially improve impact

The next improvements should focus on graph value, not prompt tuning.

High-value priorities:

1. Persist the missing semantic edges alongside `Event` nodes.
2. Persist the raw spatial evidence layer (`NEAR`, `APPROACHING`).
3. Persist zone density and class-ratio state on `Zone` nodes.
4. Add `DETECTED_IN` and `BELONGS_TO_CLASS` so hierarchy queries are real.
5. Store relationship lifecycle fields for condition-style edges such as
   `IN_ZONE`, `CONVOY_WITH`, and `LOITERING_IN`.
6. Normalize the most important event metadata into stable properties.

If only one area is tackled first, it should be semantic-edge materialization.
That would move the graph from "queryable event log" toward an actual scene
graph.

## Practical bottom line

The current graph is good enough to prove ingestion, persistence, basic history
tracking, event logging, and LLM plumbing.

It is not yet strong enough to behave like a genuinely useful ontology-backed
investigative graph. The main bottleneck is not prompt quality alone. The graph
generation layer is still under-expressive.
