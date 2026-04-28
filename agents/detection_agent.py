"""
Detection agent model-loading utilities.

This module currently implements model loading, raw frame inference, and raw
YOLO detection formatting for the Detection Agent contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from configs.loader import DETECTION_CONFIG_PATH, load_yaml_config
import logging
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_detection_config(
    config_path: str | Path = DETECTION_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load detection-agent settings from configs/detection.yaml.

    Args:
        config_path: YAML config file location.
        config: Optional runtime override mapping.

    Returns:
        Detection configuration dictionary.
    """
    return load_yaml_config(config_path, overrides=config)


def load_yolo_model(
    model_path: str | None = None,
    fallback_model_path: str | None = None,
    config_path: str | Path = DETECTION_CONFIG_PATH,
    config: dict[str, Any] | None = None,
    yolo_factory: Callable[[str], Any] = YOLO,
) -> Any:
    """
    Load a YOLO model from a configured path.

    Args:
        model_path: Preferred model path override.
        fallback_model_path: Fallback model path override.
        config_path: YAML config file location.
        config: Optional runtime config overrides.
        yolo_factory: Injectable YOLO constructor for testing.

    Returns:
        Instantiated YOLO model object.

    Raises:
        FileNotFoundError: If neither the configured path nor the supported
            fallback path exists on disk.
    """
    detection_config = load_detection_config(config_path, config=config)
    model_config = detection_config.get("model", {})
    preferred_model_path = model_path or model_config["preferred_path"]
    fallback_path = fallback_model_path or model_config["fallback_path"]
    resolved_model_path = _resolve_model_path(
        preferred_model_path,
        fallback_path,
        Path(config_path).resolve().parents[1],
    )
    return yolo_factory(str(resolved_model_path))


class DetectionAgent:
    """Detection agent wrapper that owns a loaded YOLO model."""

    def __init__(
        self,
        model_path: str | None = None,
        config_path: str | Path = DETECTION_CONFIG_PATH,
        config: dict[str, Any] | None = None,
        yolo_factory: Callable[[str], Any] = YOLO,
    ) -> None:
        """
        Initialize the detection agent with a loaded YOLO model.

        Args:
            model_path: Preferred path to the YOLO checkpoint.
            config_path: YAML config file location.
            config: Optional runtime config overrides.
            yolo_factory: Injectable YOLO constructor for testing.
        """
        self.config_path = Path(config_path)
        self.config = load_detection_config(self.config_path, config=config)
        inference_config = self.config["inference"]
        model_config = self.config["model"]

        resolved_model_path = _resolve_model_path(
            model_path or model_config["preferred_path"],
            model_config["fallback_path"],
            self.config_path.resolve().parents[1],
        )
        self.model_path = str(resolved_model_path)
        self.model = load_yolo_model(
            model_path=self.model_path,
            fallback_model_path=model_config["fallback_path"],
            config_path=self.config_path,
            config=self.config,
            yolo_factory=yolo_factory,
        )
        self.conf_threshold = float(inference_config["confidence_threshold"])
        self.iou_threshold = float(inference_config["iou_threshold"])
        self.img_size = int(inference_config["img_size"])

        # Resolve compute device: prefer GPU, fall back to CPU with explicit warning
        self.device = self._resolve_device()

    @staticmethod
    def _resolve_device() -> str:
        """Detect the best available compute device."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"\033[92m[GPU] Using CUDA device: {gpu_name}\033[0m")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("\033[92m[GPU] Using Apple MPS device\033[0m")
        else:
            device = "cpu"
            logger.warning(
                "\033[93m[CPU] No GPU detected — running YOLO inference on CPU. "
                "This will be significantly slower.\033[0m"
            )
        return device

    def infer_frame(self, frame_packet: dict[str, Any]) -> Any:
        """
        Run raw YOLO inference for a single frame packet.

        Args:
            frame_packet: Sequence loader frame packet containing `frame_letterboxed`.

        Returns:
            Raw prediction object returned by `YOLO.predict`.

        Raises:
            KeyError: If the frame packet does not include `frame_letterboxed`.
        """
        frame = frame_packet["frame_letterboxed"]
        return self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

    def format_detections(
        self,
        raw_results: Any,
        frame_packet: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Convert raw YOLO results into the detection output contract.

        Args:
            raw_results: Raw results returned by `infer_frame`.
            frame_packet: Sequence loader frame packet with frame metadata.

        Returns:
            List of formatted detection dictionaries.
        """
        frame_id = int(frame_packet["frame_id"])
        result_items = list(raw_results or [])
        if not result_items:
            return []

        names = _extract_class_names(result_items[0], self.model)
        boxes = getattr(result_items[0], "boxes", None)
        if boxes is None:
            return []

        xyxy_rows = _to_rows(getattr(boxes, "xyxy", None))
        confidence_rows = _to_rows(getattr(boxes, "conf", None))
        class_rows = _to_rows(getattr(boxes, "cls", None))

        formatted_detections: list[dict[str, Any]] = []
        for bbox_row, confidence_row, class_row in zip(xyxy_rows, confidence_rows, class_rows):
            class_id = int(class_row[0])
            formatted_detections.append(
                {
                    "frame_id": frame_id,
                    "class_id": class_id,
                    "class_name": str(names[class_id]),
                    "confidence": float(confidence_row[0]),
                    "bbox": [float(coordinate) for coordinate in bbox_row[:4]],
                    "occlusion": 0,
                }
            )
        return formatted_detections


def _resolve_model_path(
    model_path: str,
    fallback_model_path: str,
    base_dir: Path,
) -> Path:
    preferred_path = _resolve_path(model_path, base_dir)
    if preferred_path.is_file():
        return preferred_path

    fallback_path = _resolve_path(fallback_model_path, base_dir)
    if fallback_path.is_file():
        return fallback_path

    raise FileNotFoundError(f"YOLO model weights not found at {preferred_path}")


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def _extract_class_names(result: Any, model: Any) -> dict[int, str] | list[str]:
    names = getattr(result, "names", None)
    if names is not None:
        return names
    return getattr(model, "names", {})


def _to_rows(value: Any) -> list[list[float]]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        python_value = value.tolist()
    else:
        python_value = value

    if isinstance(python_value, np.ndarray):
        python_value = python_value.tolist()

    if not isinstance(python_value, list):
        return [[float(python_value)]]
    if python_value and not isinstance(python_value[0], list):
        return [[float(item)] for item in python_value]
    return [[float(item) for item in row] for row in python_value]
