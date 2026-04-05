# Tracking Agent: ByteTrack Integration Task

This step adds the ByteTrack integration surface in `agents/tracking_agent.py`.

Implemented in this task:

- `load_bytetrack_config()` for `configs/bytetrack.yaml`
- `TrackingAgent` initialization with Ultralytics `BYTETracker`
- YAML-backed tracker thresholds and buffer settings
- runtime override support for tests and callers

Still deferred to later tasks:

- converting detections into tracked outputs
- track ID persistence logic around emitted records
- lost-track handling in the returned tracking contract
