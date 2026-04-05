# Entity Resolution Pair Comparison

The first Phase 9 task adds the comparison primitive used by the later
re-identification pass.

`agents/entity_resolution_agent.py` now provides:

- `load_entity_resolution_config(...)`
- `compare_track_pair(track_a, track_b, config)`
- `EntityResolutionAgent.compare_track_pair(...)`

The comparison currently evaluates only the five candidate rules from the
project spec:

- temporal ordering
- spatial continuity
- class match
- bounded temporal gap
- motion continuity

Thresholds live in [configs/entity_resolution.yaml](/home/saksham/codebase/deep-learning-project/configs/entity_resolution.yaml).
