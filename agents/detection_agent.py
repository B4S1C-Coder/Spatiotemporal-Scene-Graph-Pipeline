"""
Detection agent model-loading utilities.

This module currently implements only the model-loading portion of the
Detection Agent contract. Frame inference and detection formatting are handled
in later tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ultralytics import YOLO


DEFAULT_MODEL_PATH = "weights/yolov8m_visdrone.pt"
FALLBACK_MODEL_PATH = "weights/yolov8m.pt"


def load_yolo_model(
    model_path: str,
    yolo_factory: Callable[[str], Any] = YOLO,
) -> Any:
    """
    Load a YOLO model from a configured path.

    Args:
        model_path: Preferred model path.
        yolo_factory: Injectable YOLO constructor for testing.

    Returns:
        Instantiated YOLO model object.

    Raises:
        FileNotFoundError: If neither the configured path nor the supported
            fallback path exists on disk.
    """
    resolved_model_path = _resolve_model_path(model_path)
    return yolo_factory(str(resolved_model_path))


class DetectionAgent:
    """Detection agent wrapper that owns a loaded YOLO model."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        yolo_factory: Callable[[str], Any] = YOLO,
    ) -> None:
        """
        Initialize the detection agent with a loaded YOLO model.

        Args:
            model_path: Preferred path to the YOLO checkpoint.
            yolo_factory: Injectable YOLO constructor for testing.
        """
        self.model_path = str(_resolve_model_path(model_path))
        self.model = load_yolo_model(self.model_path, yolo_factory=yolo_factory)


def _resolve_model_path(model_path: str) -> Path:
    preferred_path = Path(model_path)
    if preferred_path.is_file():
        return preferred_path

    if preferred_path.as_posix() == DEFAULT_MODEL_PATH:
        fallback_path = Path(FALLBACK_MODEL_PATH)
        if fallback_path.is_file():
            return fallback_path

    raise FileNotFoundError(f"YOLO model weights not found at {preferred_path}")
