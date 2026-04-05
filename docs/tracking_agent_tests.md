# Tracking Agent Test Coverage

`tests/test_tracking_agent.py` now covers the completed Phase 4 tracking work:

- loading `configs/bytetrack.yaml`
- initializing `BYTETracker` from YAML-backed settings
- applying runtime config overrides
- converting detections into ByteTrack inputs
- mapping tracker rows back into tracking output records
- `is_new` lifecycle behavior for first-seen vs existing track IDs
- one-time emission of newly lost tracks with `is_lost=True`
