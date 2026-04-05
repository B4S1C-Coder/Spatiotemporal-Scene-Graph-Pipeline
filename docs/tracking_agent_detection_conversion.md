# Tracking Agent: Detection-to-Track Conversion Task

This step adds the basic conversion layer in `agents/tracking_agent.py`.

Implemented in this task:

- `TrackingAgent.track_detections(detections)`
- adaptation from detection dictionaries to a ByteTrack-compatible input object
- mapping of tracker rows back into the tracking output shape

Current behavior:

- `is_new` is always `False`
- `is_lost` is always `False`

Those lifecycle fields are intentionally left for the next tracking tasks:

- track ID persistence
- lost-track handling
