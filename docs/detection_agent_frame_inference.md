# Detection Agent: Frame Inference Task

This step adds raw frame inference to `agents/detection_agent.py`.

Implemented in this task:

- `DetectionAgent.infer_frame(frame_packet)`
- inference over `frame_packet["frame_letterboxed"]`
- raw passthrough of `YOLO.predict(...)` results
- default thresholds aligned with the agent spec:
  - `conf=0.35`
  - `iou=0.45`
  - `imgsz=640`

Still deferred to later tasks:

- converting raw YOLO results into the detection output contract
- reading these settings from `configs/detection.yaml`
