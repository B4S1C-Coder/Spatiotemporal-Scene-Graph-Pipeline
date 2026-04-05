"""Tests for the detection agent model loader."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from agents.detection_agent import DetectionAgent, load_yolo_model


class FakeYOLO:
    """Simple stand-in for the Ultralytics YOLO class."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.predict_calls: list[dict[str, object]] = []
        self.names = {
            0: "pedestrian",
            3: "car",
        }

    def predict(self, **kwargs: object) -> list[dict[str, object]]:
        self.predict_calls.append(kwargs)
        return [{"prediction": "ok", "source": kwargs["source"]}]


def write_detection_config(config_path: Path, model_path: str, fallback_path: str) -> None:
    """Write a minimal detection config for tests."""
    config_path.write_text(
        textwrap.dedent(
            f"""
            model:
              preferred_path: {model_path}
              fallback_path: {fallback_path}
            inference:
              confidence_threshold: 0.35
              iou_threshold: 0.45
              img_size: 640
            vision:
              vision_mode: offline
              data_root: data/visdrone/VisDrone2019-MOT-val/sequences
              frame_skip: 1
              img_size: 1280
            scene_defaults:
              altitude_fallback_m: 50.0
              weather: clear
              weather_source: default
              scene_type: urban
              time_of_day: daytime
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_yolo_model_uses_requested_weights_path(tmp_path: Path) -> None:
    """The loader should instantiate YOLO with the configured model path."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, str(weight_path), str(weight_path))

    model = load_yolo_model(str(weight_path), config_path=config_path, yolo_factory=FakeYOLO)

    assert isinstance(model, FakeYOLO)
    assert model.model_path == str(weight_path)


def test_detection_agent_loads_model_during_initialization(tmp_path: Path) -> None:
    """DetectionAgent should load and retain the YOLO model instance."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, str(weight_path), str(weight_path))

    agent = DetectionAgent(model_path=str(weight_path), config_path=config_path, yolo_factory=FakeYOLO)

    assert agent.model_path == str(weight_path)
    assert isinstance(agent.model, FakeYOLO)
    assert agent.model.model_path == str(weight_path)


def test_detection_agent_uses_fallback_model_path_when_default_is_missing(tmp_path: Path) -> None:
    """The default VisDrone path should fall back to base yolov8m weights when available."""
    fallback_path = tmp_path / "weights" / "yolov8m.pt"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "configs" / "detection.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_config(config_path, "weights/yolov8m_visdrone.pt", "weights/yolov8m.pt")

    agent = DetectionAgent(config_path=config_path, yolo_factory=FakeYOLO)

    assert agent.model_path == str(fallback_path)
    assert agent.model.model_path == str(fallback_path)


def test_load_yolo_model_raises_when_no_supported_weights_exist(tmp_path: Path) -> None:
    """Missing configured and fallback weights should raise FileNotFoundError."""
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, "weights/yolov8m_visdrone.pt", "weights/yolov8m.pt")

    with pytest.raises(FileNotFoundError, match="yolov8m_visdrone.pt"):
        load_yolo_model(config_path=config_path, yolo_factory=FakeYOLO)


def test_detection_agent_infer_frame_calls_model_predict_with_defaults(tmp_path: Path) -> None:
    """Raw frame inference should call YOLO.predict with the agent thresholds."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, str(weight_path), str(weight_path))
    agent = DetectionAgent(model_path=str(weight_path), config_path=config_path, yolo_factory=FakeYOLO)
    frame_packet = {"frame_letterboxed": "frame-bytes"}

    results = agent.infer_frame(frame_packet)

    assert results == [{"prediction": "ok", "source": "frame-bytes"}]
    assert agent.model.predict_calls == [
        {
            "source": "frame-bytes",
            "conf": 0.35,
            "iou": 0.45,
            "imgsz": 640,
            "verbose": False,
        }
    ]


def test_detection_agent_infer_frame_requires_letterboxed_frame(tmp_path: Path) -> None:
    """Inference should fail fast when the expected frame input is missing."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, str(weight_path), str(weight_path))
    agent = DetectionAgent(model_path=str(weight_path), config_path=config_path, yolo_factory=FakeYOLO)

    with pytest.raises(KeyError, match="frame_letterboxed"):
        agent.infer_frame({"frame": "missing-letterboxed"})


def test_detection_agent_reads_thresholds_from_yaml_config(tmp_path: Path) -> None:
    """Inference settings should come from detection.yaml rather than Python constants."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            model:
              preferred_path: {weight_path}
              fallback_path: {weight_path}
            inference:
              confidence_threshold: 0.22
              iou_threshold: 0.61
              img_size: 512
            vision:
              vision_mode: offline
              data_root: data/visdrone/VisDrone2019-MOT-val/sequences
              frame_skip: 1
              img_size: 1280
            scene_defaults:
              altitude_fallback_m: 50.0
              weather: clear
              weather_source: default
              scene_type: urban
              time_of_day: daytime
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    agent = DetectionAgent(config_path=config_path, yolo_factory=FakeYOLO)

    assert agent.conf_threshold == 0.22
    assert agent.iou_threshold == 0.61
    assert agent.img_size == 512


class FakeTensor:
    """Tensor-like object exposing a tolist method."""

    def __init__(self, value: object) -> None:
        self.value = value

    def tolist(self) -> object:
        return self.value


class FakeBoxes:
    """Ultralytics-like boxes container for formatting tests."""

    def __init__(self) -> None:
        self.xyxy = FakeTensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        self.conf = FakeTensor([0.9, 0.75])
        self.cls = FakeTensor([0, 3])


class FakeResult:
    """Ultralytics-like single image result."""

    def __init__(self) -> None:
        self.boxes = FakeBoxes()
        self.names = {0: "pedestrian", 3: "car"}


def test_detection_agent_formats_raw_results_into_detection_contract(tmp_path: Path) -> None:
    """Raw YOLO results should be converted into the expected detection dictionaries."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, str(weight_path), str(weight_path))
    agent = DetectionAgent(model_path=str(weight_path), config_path=config_path, yolo_factory=FakeYOLO)

    detections = agent.format_detections(
        raw_results=[FakeResult()],
        frame_packet={"frame_id": 12, "sequence_id": "uav0000009_04358_v"},
    )

    assert detections == [
        {
            "frame_id": 12,
            "class_id": 0,
            "class_name": "pedestrian",
            "confidence": 0.9,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "occlusion": 0,
        },
        {
            "frame_id": 12,
            "class_id": 3,
            "class_name": "car",
            "confidence": 0.75,
            "bbox": [50.0, 60.0, 70.0, 80.0],
            "occlusion": 0,
        },
    ]


def test_detection_agent_formats_empty_results_as_empty_list(tmp_path: Path) -> None:
    """Formatting should return an empty list when YOLO returns no detections."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path, str(weight_path), str(weight_path))
    agent = DetectionAgent(model_path=str(weight_path), config_path=config_path, yolo_factory=FakeYOLO)

    assert agent.format_detections(raw_results=[], frame_packet={"frame_id": 0}) == []
