# Detection Agent: Config Loading Task

This step makes detection configuration loading explicit in
`agents/detection_agent.py`.

Implemented in this task:

- `load_detection_config()` helper
- direct loading from `configs/detection.yaml`
- reuse of that loaded config by model loading and inference setup

The detection agent now gets its runtime settings from YAML for:

- model paths
- confidence threshold
- IoU threshold
- inference image size
