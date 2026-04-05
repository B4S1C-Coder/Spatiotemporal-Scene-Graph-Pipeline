# Detection Agent Test Coverage

`tests/test_detection_agent.py` now covers the completed Detection Agent work:

- loading config from `configs/detection.yaml`
- resolving preferred and fallback model paths
- loading the YOLO model during agent initialization
- raw frame inference argument forwarding
- missing-input failure paths
- formatting raw YOLO results into detection dictionaries
- empty-result and missing-box fallback behavior
- class-name resolution from both result-level and model-level names
