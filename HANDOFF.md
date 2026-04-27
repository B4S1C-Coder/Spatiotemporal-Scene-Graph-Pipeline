# Handoff

This file is for the next coding agent. It captures the current project state,
recent changes, known issues, and the most useful next steps.

## Project summary

This repository implements a research-scale spatiotemporal scene-graph
pipeline:

- VisDrone-style frame sequences in
- YOLO detection
- ByteTrack tracking
- motion enrichment
- rule-based event detection
- Neo4j graph writes
- LLM-backed graph querying
- Streamlit query UI

The graph is usable, but still not at the intended ontology richness described
in [AGENTS.md](/home/saksham/codebase/deep-learning-project/AGENTS.md). The gap
is documented in
[docs/graph_generation_shortcomings.md](/home/saksham/codebase/deep-learning-project/docs/graph_generation_shortcomings.md).

## Important recent changes

### UI visualization

[ui/app.py](/home/saksham/codebase/deep-learning-project/ui/app.py) now has:

- query spinner / in-progress indicator
- frame visualization helpers
- annotated frame preview rendering from graph results
- short GIF clip generation with bounding boxes

Key helper functions:

- `extract_visual_targets(...)`
- `fetch_frame_boxes(...)`
- `resolve_frame_path(...)`
- `render_bounding_boxes(...)`
- `build_preview_frames(...)`
- `build_clip_gif_bytes(...)`
- `build_query_visualization_payload(...)`

Current visualization behavior:

- activates when query results contain frame-like fields, or automatically resolves track-only results to their first_seen_frame
- uses `sequence_id` from the UI input
- fetches `bbox_norm` from `APPEARED_IN`
- renders up to a few preview frames
- builds a short GIF centered on the first matched frame

### LLM query harness hardening

[agents/llm_agent.py](/home/saksham/codebase/deep-learning-project/agents/llm_agent.py)
was recently hardened to compensate for mixed graph states:

- person-like alias expansion for:
  - `person`
  - `pedestrian`
  - `people`
- fallback from `BELONGS_TO_CLASS` queries to direct `o.class` queries for
  basic class inventory/count questions when older ingested sequences do not
  yet have taxonomy edges
- retries on Neo4j execution errors
- logging for validation failures and zero-result queries

Related config:

- [configs/llm.yaml](/home/saksham/codebase/deep-learning-project/configs/llm.yaml)

### Graph layer

[agents/graph_agent.py](/home/saksham/codebase/deep-learning-project/agents/graph_agent.py)
now writes more than the original thin graph:

- `DETECTED_IN`
- `BELONGS_TO_CLASS`
- `Frame-[:PRECEDES]->Frame`
- `Zone` metrics:
  - `last_density`
  - `vehicle_ratio`
  - `pedestrian_ratio`
- semantic edges for:
  - `NEAR_MISS`
  - `CONVOY_WITH`
  - `LOITERING_IN`
  - `JAYWALKING_IN`

Also note:

- event node metadata is now stored as `metadata_json`
- this was changed because Neo4j cannot store a map as a property value

### Runner memory safeguards

[pipeline/runner.py](/home/saksham/codebase/deep-learning-project/pipeline/runner.py)
was recently changed to address WSL2 memory blowups:

- CLI path now uses summary-mode execution instead of retaining every frame’s
  full result payload in memory
- memory cap support:
  - `runtime.max_process_rss_mb`
- progress logging:
  - `runtime.progress_log_every_frames`
- periodic GC:
  - `runtime.gc_every_frames`
- CLI flags:
  - `--max-rss-mb`
  - `--progress-every-frames`

Related config:

- [configs/detection.yaml](/home/saksham/codebase/deep-learning-project/configs/detection.yaml)

## Current graph reality

The graph still does **not** fully match the target ontology.

Reliable current areas:

- object inventory
- class counts
- scene metadata
- frame history via `APPEARED_IN`
- zone-level latest ratios/density
- event lookup through `Event`
- some semantic event edges
- post-hoc entity resolution

Still weak / incomplete:

- raw spatial evidence edges (`NEAR`, `APPROACHING`, `OVERLAPS_WITH`)
- relationship lifecycle fields (`is_active`, validity windows)
- dense historical zone analytics
- normalized event evidence instead of generic metadata JSON
- crowd membership / event causality
- richer motion history in the graph

See
[docs/graph_generation_shortcomings.md](/home/saksham/codebase/deep-learning-project/docs/graph_generation_shortcomings.md)
for the full analysis.

## Known practical caveats

### Ingestion state matters

Older sequences ingested before some graph upgrades may be missing:

- `BELONGS_TO_CLASS`
- `DETECTED_IN`
- `PRECEDES`
- richer event edges

The LLM harness has fallbacks for some of this, but not all of it.

### Detection labels can differ

If the pipeline falls back to base `weights/yolov8m.pt` instead of a
VisDrone-tuned checkpoint, graph labels may include COCO-style `person`
instead of VisDrone-style `pedestrian` / `people`.

The LLM harness currently compensates for this in a narrow way. It is not yet a
general semantic normalization layer.

### Raw arbitrary video ingestion is not complete

The project works best with VisDrone-style extracted frame sequences.

It is **not** yet a polished “ingest any raw video file” system. There is no
generic:

- video decode frontend
- source adapter abstraction
- arbitrary metadata adapter

### xView is not a fit for the current full pipeline

xView is static imagery, not temporal video.

You could reuse the detection / object-node parts, but not the tracking-motion-
event stack as currently designed.

## Tests that matter most right now

Run these first after touching the affected areas:

- `venv/bin/python -m pytest tests/test_pipeline_runner.py`
- `venv/bin/python -m pytest tests/test_graph_agent.py`
- `venv/bin/python -m pytest tests/test_llm_agent.py`
- `venv/bin/python -m pytest tests/test_ui_app.py`

Recent passing status before this handoff:

- `tests/test_pipeline_runner.py` passed
- `tests/test_graph_agent.py` passed
- `tests/test_llm_agent.py` passed
- `tests/test_ui_app.py` was updated for visualization helpers and should be run after UI edits

## Recommended next steps

If continuing from here, the highest-value next steps are:

1. Add graph-backed visualization queries for event-centric playback
   - especially event rows with `frame_id`, `primary_track_id`, `secondary_track_id`
2. Add direct video export support
   - GIF is implemented now
   - MP4 export would be a stronger demo artifact
3. Add a config-driven class alias / synonym layer
   - instead of current targeted person-only heuristics
4. Continue Phase 14 graph work
   - raw spatial edges
   - lifecycle fields
   - density history persistence
   - richer event evidence

## Files worth reading first

- [AGENTS.md](/home/saksham/codebase/deep-learning-project/AGENTS.md)
- [relationships.md](/home/saksham/codebase/deep-learning-project/relationships.md)
- [docs/graph_generation_shortcomings.md](/home/saksham/codebase/deep-learning-project/docs/graph_generation_shortcomings.md)
- [pipeline/runner.py](/home/saksham/codebase/deep-learning-project/pipeline/runner.py)
- [agents/graph_agent.py](/home/saksham/codebase/deep-learning-project/agents/graph_agent.py)
- [agents/llm_agent.py](/home/saksham/codebase/deep-learning-project/agents/llm_agent.py)
- [ui/app.py](/home/saksham/codebase/deep-learning-project/ui/app.py)
- [TASKS.md](/home/saksham/codebase/deep-learning-project/TASKS.md)
