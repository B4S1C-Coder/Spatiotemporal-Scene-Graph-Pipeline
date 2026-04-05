# Pipeline Runner

Phase 8 now wires the runner through the full frame pipeline:

- `SequenceLoader.iter_frames(...)`
- `DetectionAgent.infer_frame(...)`
- `DetectionAgent.format_detections(...)`
- `TrackingAgent.track_detections(...)`
- `MotionAgent.enrich_tracks(...)`
- `EventAgent.process_tracks(...)`
- `BatchWriter.add_frame_data(...)`

The runner keeps only active tracks for motion and event processing, forwards
`scene_payload` into the graph-write path, and flushes buffered graph writes at
the end of each sequence so partial batches are not left pending.

`pipeline/batch_writer.py` remains intentionally thin. It exists so the runner
can depend on a dedicated batch-write surface while reusing the batching logic
already implemented inside `GraphAgent`.
