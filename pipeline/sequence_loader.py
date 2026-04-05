"""
Sequence loader utilities for VisDrone MOT datasets.

This module currently handles dataset-level concerns only:
- locating a sequence on disk
- validating the expected VisDrone MOT directory structure
- enumerating available frame files
- parsing sequence metadata from seqinfo.ini
- reading approved sequence IDs from a JSON manifest
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SequencePaths:
    """Resolved filesystem paths for a single VisDrone MOT sequence."""

    sequence_root: Path
    image_dir: Path
    frame_paths: tuple[Path, ...]
    seqinfo_file: Path
    gt_file: Path | None


class SequenceLoader:
    """Load and validate a VisDrone MOT sequence from disk."""

    def __init__(self, sequence_id: str, config: dict[str, Any]):
        """
        Resolve a sequence from the configured VisDrone data root.

        Args:
            sequence_id: VisDrone sequence directory name.
            config: Loader configuration. Must include `data_root`.

        Raises:
            ValueError: If `data_root` is missing from config.
            FileNotFoundError: If the sequence or required files do not exist.
        """
        self.sequence_id = sequence_id
        self.config = config
        self.data_root = self._resolve_data_root(config)
        self.paths = self._build_sequence_paths(sequence_id, self.data_root)
        self.scene_payload = self._build_scene_payload()

    def get_sequence_paths(self) -> SequencePaths:
        """Return the resolved path bundle for the loaded sequence."""
        return self.paths

    def get_frame_ids(self) -> list[int]:
        """Return zero-indexed frame IDs derived from the sequence filenames."""
        return [int(frame_path.stem) - 1 for frame_path in self.paths.frame_paths]

    def get_scene_payload(self) -> dict[str, Any]:
        """Return the parsed and inferred Scene payload for this sequence."""
        return self.scene_payload

    def iter_frames(self, frame_skip: int = 1) -> Any:
        """
        Yield frame packets in sequence order.

        This task implements the frame iteration interface and basic frame loading.
        Annotation parsing remains deferred, so `annotations` is an empty list when
        gt data exists and `None` otherwise.
        """
        if frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer.")

        img_size = int(self.config.get("img_size", max(
            self.scene_payload["frame_width"],
            self.scene_payload["frame_height"],
        )))

        for packet_index, frame_path in enumerate(self.paths.frame_paths[::frame_skip]):
            frame_id = int(frame_path.stem) - 1
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Could not load frame image: {frame_path}")

            frame_letterboxed, scale, pad_w, pad_h = self._letterbox_frame(frame, img_size)

            yield {
                "frame": frame,
                "frame_letterboxed": frame_letterboxed,
                "frame_id": frame_id,
                "scale": scale,
                "pad_w": pad_w,
                "pad_h": pad_h,
                "orig_width": self.scene_payload["frame_width"],
                "orig_height": self.scene_payload["frame_height"],
                "sequence_id": self.sequence_id,
                "frame_skip": frame_skip,
                "is_static": False,
                "annotations": [] if self.paths.gt_file is not None else None,
                "scene_payload": self.scene_payload if packet_index == 0 else None,
            }

    def get_annotation(self, frame_id: int) -> list[dict[str, Any]]:
        """Annotation parsing is implemented in a later task."""
        raise NotImplementedError("Annotation parsing is implemented in a later task.")

    @staticmethod
    def list_available(sequences_json_path: str) -> list[str]:
        """
        Read approved sequence IDs from a JSON manifest.

        Supported formats:
        - `["seq_a", "seq_b"]`
        - `{"sequence_ids": ["seq_a", "seq_b"]}`
        """
        manifest_path = Path(sequences_json_path)
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        if isinstance(manifest_data, list):
            return [str(sequence_id) for sequence_id in manifest_data]
        if isinstance(manifest_data, dict) and isinstance(manifest_data.get("sequence_ids"), list):
            return [str(sequence_id) for sequence_id in manifest_data["sequence_ids"]]

        raise ValueError("Sequence manifest must be a list or contain a 'sequence_ids' list.")

    @staticmethod
    def _resolve_data_root(config: dict[str, Any]) -> Path:
        data_root = config.get("data_root")
        if not data_root:
            raise ValueError("SequenceLoader config must include 'data_root'.")
        return Path(data_root)

    def _build_scene_payload(self) -> dict[str, Any]:
        seqinfo = self._parse_seqinfo_file(self.paths.seqinfo_file)
        sequence_meta = self._load_sequence_meta_lookup(
            self.config.get("sequence_meta_path"),
            self.data_root,
        ).get(self.sequence_id, {})

        altitude_m, altitude_source = self._resolve_altitude(sequence_meta)

        return {
            "sequence_id": self.sequence_id,
            "total_frames": seqinfo["seqLength"],
            "frame_rate": seqinfo["frameRate"],
            "frame_width": seqinfo["imWidth"],
            "frame_height": seqinfo["imHeight"],
            "altitude_m": altitude_m,
            "altitude_source": altitude_source,
            "weather": str(self.config.get("default_weather", "clear")),
            "weather_source": str(self.config.get("weather_source", "default")),
            "scene_type": str(self.config.get("default_scene_type", "urban")),
            "time_of_day": str(self.config.get("default_time_of_day", "daytime")),
            "split": self._infer_split(self.paths.sequence_root),
            "frame_skip": int(self.config.get("frame_skip", 1)),
            "annotation_available": self.paths.gt_file is not None,
        }

    @staticmethod
    def _parse_seqinfo_file(seqinfo_file: Path) -> dict[str, int]:
        parser = configparser.ConfigParser()
        parser.read(seqinfo_file, encoding="utf-8")

        if "Sequence" not in parser:
            raise ValueError(f"Missing [Sequence] section in {seqinfo_file}")

        sequence_section = parser["Sequence"]
        required_fields = ("seqLength", "frameRate", "imWidth", "imHeight")
        missing_fields = [field_name for field_name in required_fields if field_name not in sequence_section]
        if missing_fields:
            raise ValueError(f"Missing required seqinfo fields in {seqinfo_file}: {', '.join(missing_fields)}")

        return {
            "seqLength": sequence_section.getint("seqLength"),
            "frameRate": sequence_section.getint("frameRate"),
            "imWidth": sequence_section.getint("imWidth"),
            "imHeight": sequence_section.getint("imHeight"),
        }

    @classmethod
    def _load_sequence_meta_lookup(
        cls,
        configured_meta_path: str | None,
        data_root: Path,
    ) -> dict[str, dict[str, Any]]:
        meta_path = cls._resolve_sequence_meta_path(configured_meta_path, data_root)
        if meta_path is None or not meta_path.is_file():
            return {}

        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta_data, dict):
            raise ValueError("Sequence metadata lookup must be a JSON object.")
        return {
            str(sequence_id): sequence_meta
            for sequence_id, sequence_meta in meta_data.items()
            if isinstance(sequence_meta, dict)
        }

    @staticmethod
    def _resolve_sequence_meta_path(configured_meta_path: str | None, data_root: Path) -> Path | None:
        if configured_meta_path:
            return Path(configured_meta_path)

        candidate_paths = [data_root / "visdrone_sequence_meta.json"]
        candidate_paths.extend(parent_path / "visdrone_sequence_meta.json" for parent_path in data_root.parents)

        for candidate_path in candidate_paths:
            if candidate_path.is_file():
                return candidate_path
        return None

    @staticmethod
    def _resolve_altitude(sequence_meta: dict[str, Any]) -> tuple[float, str]:
        altitude_value = sequence_meta.get("altitude_m")
        if altitude_value is not None:
            return float(altitude_value), "lookup"
        return 50.0, "estimated"

    @staticmethod
    def _letterbox_frame(frame: np.ndarray, img_size: int) -> tuple[np.ndarray, float, float, float]:
        original_height, original_width = frame.shape[:2]
        scale = min(img_size / original_width, img_size / original_height)

        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_w = (img_size - resized_width) / 2.0
        pad_h = (img_size - resized_height) / 2.0
        left = int(np.floor(pad_w))
        right = int(np.ceil(pad_w))
        top = int(np.floor(pad_h))
        bottom = int(np.ceil(pad_h))

        letterboxed_frame = cv2.copyMakeBorder(
            resized_frame,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return letterboxed_frame, scale, float(left), float(top)

    @staticmethod
    def _infer_split(sequence_root: Path) -> str:
        relevant_parts = [part_name.lower() for part_name in sequence_root.parts[-4:]]
        for part_name in relevant_parts:
            if part_name in {"train", "val", "test"}:
                return part_name
            if "mot-train" in part_name or "det-train" in part_name:
                return "train"
            if "mot-val" in part_name or "det-val" in part_name:
                return "val"
            if "mot-test" in part_name or "det-test" in part_name:
                return "test"
        return "unknown"

    @classmethod
    def _build_sequence_paths(cls, sequence_id: str, data_root: Path) -> SequencePaths:
        sequence_root = cls._find_sequence_root(sequence_id, data_root)
        image_dir = sequence_root / "img1"
        seqinfo_file = sequence_root / "seqinfo.ini"
        gt_file = sequence_root / "gt" / "gt.txt"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing frame directory: {image_dir}")
        if not seqinfo_file.is_file():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_file}")

        frame_paths = tuple(sorted(image_dir.glob("*.jpg")))
        if not frame_paths:
            raise FileNotFoundError(f"No frame images found in {image_dir}")

        return SequencePaths(
            sequence_root=sequence_root,
            image_dir=image_dir,
            frame_paths=frame_paths,
            seqinfo_file=seqinfo_file,
            gt_file=gt_file if gt_file.is_file() else None,
        )

    @staticmethod
    def _find_sequence_root(sequence_id: str, data_root: Path) -> Path:
        candidate_paths = (
            data_root / sequence_id,
            data_root / "sequences" / sequence_id,
        )
        for candidate_path in candidate_paths:
            if candidate_path.is_dir():
                return candidate_path

        split_sequence_paths = sorted(data_root.glob(f"*/sequences/{sequence_id}"))
        for candidate_path in split_sequence_paths:
            if candidate_path.is_dir():
                return candidate_path

        raise FileNotFoundError(f"Could not find sequence '{sequence_id}' under {data_root}")
