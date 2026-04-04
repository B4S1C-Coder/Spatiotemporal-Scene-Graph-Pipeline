"""
Sequence loader utilities for VisDrone MOT datasets.

This module currently handles dataset-level concerns only:
- locating a sequence on disk
- validating the expected VisDrone MOT directory structure
- enumerating available frame files
- reading approved sequence IDs from a JSON manifest

Metadata parsing and frame iteration are implemented in later tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


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

    def get_sequence_paths(self) -> SequencePaths:
        """Return the resolved path bundle for the loaded sequence."""
        return self.paths

    def get_frame_ids(self) -> list[int]:
        """Return zero-indexed frame IDs derived from the sequence filenames."""
        return [int(frame_path.stem) - 1 for frame_path in self.paths.frame_paths]

    def get_scene_payload(self) -> dict[str, Any]:
        """Metadata parsing is implemented in the next task."""
        raise NotImplementedError("Scene metadata parsing is implemented in a later task.")

    def iter_frames(self, frame_skip: int = 1) -> Any:
        """Frame iteration is implemented in a later task."""
        raise NotImplementedError("Frame iteration is implemented in a later task.")

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
