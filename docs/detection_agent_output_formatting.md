# Detection Agent: Output Formatting Task

This step adds raw detection formatting to `agents/detection_agent.py`.

Implemented in this task:

- `DetectionAgent.format_detections(raw_results, frame_packet)`
- conversion of YOLO box outputs into the required detection dictionaries
- mapping of class IDs to class names via YOLO result/model names
- stable empty-list behavior when YOLO returns no detections

Current behavior:

- `occlusion` is set to `0` during formatting
- annotation-backed occlusion matching is still deferred

The formatted output now matches the detection contract shape expected by the
tracking stage.
