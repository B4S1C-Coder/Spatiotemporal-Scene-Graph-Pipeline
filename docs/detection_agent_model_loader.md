# Detection Agent: Model Loader Task

This step adds the YOLO model-loading foundation in `agents/detection_agent.py`.

Implemented in this task:

- `load_yolo_model()` helper for checkpoint loading
- `DetectionAgent` initialization with a loaded YOLO model
- fallback from `weights/yolov8m_visdrone.pt` to `weights/yolov8m.pt` when only the base checkpoint exists

Still deferred to later tasks:

- frame inference
- detection output formatting
- loading detection settings from `configs/detection.yaml`
