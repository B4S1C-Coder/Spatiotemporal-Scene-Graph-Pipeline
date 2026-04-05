# Tracking Agent: Persistence and Lost-Track Tasks

This step completes the remaining Phase 4 lifecycle behavior in
`agents/tracking_agent.py`.

Implemented in this task:

- track ID first-seen detection via `is_new`
- in-memory persistence of previously seen track IDs
- last-known track snapshots for lifecycle updates
- emission of newly lost tracks with `is_lost=True`
- deduplication so the same lost track is not emitted repeatedly

Current lifecycle behavior:

- `is_new=True` only on the first frame where a given `track_id` is observed
- lost-track records reuse the last known class, bbox, confidence, and occlusion
