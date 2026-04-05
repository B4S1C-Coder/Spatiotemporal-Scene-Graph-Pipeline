"""Tests for the detection agent model loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.detection_agent import DetectionAgent, load_yolo_model


class FakeYOLO:
    """Simple stand-in for the Ultralytics YOLO class."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path


def test_load_yolo_model_uses_requested_weights_path(tmp_path: Path) -> None:
    """The loader should instantiate YOLO with the configured model path."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")

    model = load_yolo_model(str(weight_path), yolo_factory=FakeYOLO)

    assert isinstance(model, FakeYOLO)
    assert model.model_path == str(weight_path)


def test_detection_agent_loads_model_during_initialization(tmp_path: Path) -> None:
    """DetectionAgent should load and retain the YOLO model instance."""
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("stub", encoding="utf-8")

    agent = DetectionAgent(model_path=str(weight_path), yolo_factory=FakeYOLO)

    assert agent.model_path == str(weight_path)
    assert isinstance(agent.model, FakeYOLO)
    assert agent.model.model_path == str(weight_path)


def test_detection_agent_uses_fallback_model_path_when_default_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default VisDrone path should fall back to base yolov8m weights when available."""
    fallback_path = tmp_path / "weights" / "yolov8m.pt"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text("stub", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    agent = DetectionAgent(yolo_factory=FakeYOLO)

    assert agent.model_path == "weights/yolov8m.pt"
    assert agent.model.model_path == "weights/yolov8m.pt"


def test_load_yolo_model_raises_when_no_supported_weights_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing configured and fallback weights should raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="yolov8m_visdrone.pt"):
        load_yolo_model("weights/yolov8m_visdrone.pt", yolo_factory=FakeYOLO)
