"""Tests for dataset-level sequence loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.sequence_loader import SequenceLoader


def create_sequence(root: Path, sequence_id: str, with_gt: bool = True) -> Path:
    """Create a minimal VisDrone-style sequence directory for testing."""
    sequence_root = root / sequence_id
    image_dir = sequence_root / "img1"
    gt_dir = sequence_root / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    (sequence_root / "seqinfo.ini").write_text(
        "[Sequence]\nname=test\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=640\nimHeight=480\nimExt=.jpg\n",
        encoding="utf-8",
    )
    (image_dir / "0000001.jpg").write_text("frame-1", encoding="utf-8")
    (image_dir / "0000002.jpg").write_text("frame-2", encoding="utf-8")

    if with_gt:
        (gt_dir / "gt.txt").write_text("1,1,10,10,20,20,1,1,0,0\n", encoding="utf-8")

    return sequence_root


def test_sequence_loader_resolves_sequence_from_sequences_root(tmp_path: Path) -> None:
    """The loader should resolve a sequence directly under a sequences directory."""
    sequences_root = tmp_path / "VisDrone2019-MOT-val" / "sequences"
    sequence_root = create_sequence(sequences_root, "uav0000009_04358_v")

    loader = SequenceLoader(
        sequence_id="uav0000009_04358_v",
        config={"data_root": str(sequences_root)},
    )

    paths = loader.get_sequence_paths()
    assert paths.sequence_root == sequence_root
    assert paths.seqinfo_file == sequence_root / "seqinfo.ini"
    assert paths.gt_file == sequence_root / "gt" / "gt.txt"
    assert loader.get_frame_ids() == [0, 1]


def test_sequence_loader_resolves_sequence_from_visdrone_root(tmp_path: Path) -> None:
    """The loader should find a sequence when data_root points at data/visdrone."""
    visdrone_root = tmp_path / "data" / "visdrone"
    sequence_root = create_sequence(
        visdrone_root / "VisDrone2019-MOT-train" / "sequences",
        "uav0000013_00000_v",
    )

    loader = SequenceLoader(
        sequence_id="uav0000013_00000_v",
        config={"data_root": str(visdrone_root)},
    )

    assert loader.get_sequence_paths().sequence_root == sequence_root


def test_sequence_loader_sets_gt_file_to_none_when_annotations_missing(tmp_path: Path) -> None:
    """Sequences without gt.txt should still load at the dataset level."""
    sequences_root = tmp_path / "VisDrone2019-MOT-val" / "sequences"
    create_sequence(sequences_root, "uav0000149_00000_v", with_gt=False)

    loader = SequenceLoader(
        sequence_id="uav0000149_00000_v",
        config={"data_root": str(sequences_root)},
    )

    assert loader.get_sequence_paths().gt_file is None


def test_sequence_loader_requires_data_root() -> None:
    """The loader should fail fast when config lacks data_root."""
    with pytest.raises(ValueError, match="data_root"):
        SequenceLoader(sequence_id="uav0000009_04358_v", config={})


def test_sequence_loader_requires_frame_directory_and_seqinfo(tmp_path: Path) -> None:
    """Missing required VisDrone files should raise clear filesystem errors."""
    sequence_root = tmp_path / "sequences" / "uav0000073_00600_v"
    sequence_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="frame directory"):
        SequenceLoader(
            sequence_id="uav0000073_00600_v",
            config={"data_root": str(tmp_path / "sequences")},
        )


def test_list_available_supports_list_and_object_manifests(tmp_path: Path) -> None:
    """Sequence manifests may be bare lists or wrapped in a sequence_ids object."""
    list_manifest = tmp_path / "list.json"
    object_manifest = tmp_path / "object.json"

    list_manifest.write_text(json.dumps(["seq_a", "seq_b"]), encoding="utf-8")
    object_manifest.write_text(json.dumps({"sequence_ids": ["seq_c", "seq_d"]}), encoding="utf-8")

    assert SequenceLoader.list_available(str(list_manifest)) == ["seq_a", "seq_b"]
    assert SequenceLoader.list_available(str(object_manifest)) == ["seq_c", "seq_d"]


def test_list_available_rejects_invalid_manifest_shape(tmp_path: Path) -> None:
    """Unsupported manifest shapes should raise a ValueError."""
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text(json.dumps({"items": ["seq_a"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="sequence_ids"):
        SequenceLoader.list_available(str(manifest_path))


def test_scene_payload_parses_seqinfo_and_infers_defaults(tmp_path: Path) -> None:
    """Scene payload should parse core seqinfo fields and default inferred metadata."""
    sequences_root = tmp_path / "VisDrone2019-MOT-val" / "sequences"
    create_sequence(sequences_root, "uav0000009_04358_v")

    loader = SequenceLoader(
        sequence_id="uav0000009_04358_v",
        config={"data_root": str(sequences_root), "frame_skip": 3},
    )

    scene_payload = loader.get_scene_payload()

    assert scene_payload["sequence_id"] == "uav0000009_04358_v"
    assert scene_payload["total_frames"] == 2
    assert scene_payload["frame_rate"] == 30
    assert scene_payload["frame_width"] == 640
    assert scene_payload["frame_height"] == 480
    assert scene_payload["altitude_m"] == 50.0
    assert scene_payload["altitude_source"] == "estimated"
    assert scene_payload["weather"] == "clear"
    assert scene_payload["weather_source"] == "default"
    assert scene_payload["scene_type"] == "urban"
    assert scene_payload["time_of_day"] == "daytime"
    assert scene_payload["split"] == "val"
    assert scene_payload["frame_skip"] == 3
    assert scene_payload["annotation_available"] is True


def test_scene_payload_uses_lookup_metadata_when_available(tmp_path: Path) -> None:
    """Supplemental sequence metadata should populate altitude from the lookup file."""
    visdrone_root = tmp_path / "data" / "visdrone"
    create_sequence(visdrone_root / "VisDrone2019-MOT-train" / "sequences", "uav0000013_00000_v")
    meta_path = tmp_path / "data" / "visdrone_sequence_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"uav0000013_00000_v": {"altitude_m": 60, "altitude_range": [55, 65]}}),
        encoding="utf-8",
    )

    loader = SequenceLoader(
        sequence_id="uav0000013_00000_v",
        config={"data_root": str(visdrone_root)},
    )

    scene_payload = loader.get_scene_payload()
    assert scene_payload["altitude_m"] == 60.0
    assert scene_payload["altitude_source"] == "lookup"
    assert scene_payload["split"] == "train"


def test_scene_payload_rejects_invalid_seqinfo(tmp_path: Path) -> None:
    """Missing required seqinfo fields should raise a clear metadata parsing error."""
    sequences_root = tmp_path / "VisDrone2019-MOT-val" / "sequences"
    sequence_root = create_sequence(sequences_root, "uav0000149_00000_v")
    (sequence_root / "seqinfo.ini").write_text("[Sequence]\nname=test\nframeRate=30\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seqLength"):
        SequenceLoader(
            sequence_id="uav0000149_00000_v",
            config={"data_root": str(sequences_root)},
        )
