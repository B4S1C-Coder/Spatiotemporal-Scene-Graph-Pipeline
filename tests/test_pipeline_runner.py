"""Tests for the pipeline runner and batch-writer integration."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from pipeline.batch_writer import BatchWriter
from pipeline.runner import PipelineRunner, run_pipeline_cli


class FakeSequenceLoader:
    """SequenceLoader stand-in for runner tests."""

    init_calls: list[dict[str, object]] = []
    iter_calls: list[dict[str, object]] = []

    def __init__(self, sequence_id: str, config: dict[str, object]) -> None:
        self.sequence_id = sequence_id
        self.config = config
        FakeSequenceLoader.init_calls.append({"sequence_id": sequence_id, "config": config})

    def iter_frames(self, frame_skip: int | None = None):
        FakeSequenceLoader.iter_calls.append({"sequence_id": self.sequence_id, "frame_skip": frame_skip})
        yield {
            "sequence_id": self.sequence_id,
            "frame_id": 0,
            "frame_skip": frame_skip or 1,
            "orig_width": 1920,
            "orig_height": 1080,
            "frame_letterboxed": f"frame-{self.sequence_id}-0",
            "scene_payload": {"sequence_id": self.sequence_id, "scene_type": "urban"},
        }
        yield {
            "sequence_id": self.sequence_id,
            "frame_id": 1,
            "frame_skip": frame_skip or 1,
            "orig_width": 1920,
            "orig_height": 1080,
            "frame_letterboxed": f"frame-{self.sequence_id}-1",
            "scene_payload": None,
        }


class FakeDetectionAgent:
    """Detection agent stand-in that records inference and formatting calls."""

    def __init__(self) -> None:
        self.infer_calls: list[dict[str, Any]] = []
        self.format_calls: list[dict[str, Any]] = []

    def infer_frame(self, frame_packet: dict[str, Any]) -> list[dict[str, Any]]:
        self.infer_calls.append(frame_packet)
        return [{"frame_id": frame_packet["frame_id"], "raw": True}]

    def format_detections(
        self,
        raw_results: list[dict[str, Any]],
        frame_packet: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.format_calls.append({"raw_results": raw_results, "frame_packet": frame_packet})
        return [
            {
                "frame_id": frame_packet["frame_id"],
                "class_id": 3,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "occlusion": 0,
            }
        ]


class FakeTrackingAgent:
    """Tracking agent stand-in that emits one active and one lost track."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def track_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append(detections)
        frame_id = int(detections[0]["frame_id"])
        return [
            {
                "track_id": 101,
                "frame_id": frame_id,
                "class_id": 3,
                "class_name": "car",
                "confidence": 0.88,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "occlusion": 0,
                "is_new": frame_id == 0,
                "is_lost": False,
            },
            {
                "track_id": 202,
                "frame_id": frame_id,
                "class_id": 3,
                "class_name": "car",
                "confidence": 0.70,
                "bbox": [50.0, 60.0, 70.0, 80.0],
                "occlusion": 1,
                "is_new": False,
                "is_lost": True,
            },
        ]


class FakeMotionAgent:
    """Motion agent stand-in that returns enriched versions of active tracks."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def enrich_tracks(
        self,
        tracked_detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "tracked_detections": tracked_detections,
                "frame_width": frame_width,
                "frame_height": frame_height,
            }
        )
        return [
            {
                "track_id": int(track["track_id"]),
                "frame_id": int(track["frame_id"]),
                "class_name": str(track["class_name"]),
                "centroid_norm": [0.1, 0.2],
                "bbox_norm": [0.01, 0.02, 0.03, 0.04],
                "speed_px_per_frame": 0.02,
                "heading_deg": 45.0,
                "movement_pattern": "linear",
                "occlusion": int(track["occlusion"]),
                "zone_id": "cell_0_0",
                "trajectory_buffer": [[0.1, 0.2]],
            }
            for track in tracked_detections
        ]


class FakeEventAgent:
    """Event agent stand-in that records the active-track map it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.zone_stats = {"cell_0_0": {"last_density": 1.0, "vehicle_ratio": 1.0, "pedestrian_ratio": 0.0}}

    def process_tracks(
        self,
        tracks: dict[int, dict[str, Any]],
        frame_id: int,
        sequence_id: str,
    ) -> list[dict[str, Any]]:
        self.calls.append({"tracks": tracks, "frame_id": frame_id, "sequence_id": sequence_id})
        return [
            {
                "event_type": "NEAR_MISS",
                "frame_id": frame_id,
                "sequence_id": sequence_id,
                "primary_track_id": 101,
                "secondary_track_id": None,
                "confidence": 0.8,
                "metadata": {"source": "fake"},
            }
        ]

    def get_zone_stats_snapshot(self) -> dict[str, dict[str, float]]:
        return self.zone_stats


class FakeBatchWriter:
    """Batch writer stand-in for testing graph integration and final flushes."""

    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.flush_count = 0

    def add_frame_data(
        self,
        object_states: list[dict[str, Any]],
        sequence_id: str,
        frame_id: int,
        events: list[dict[str, Any]] | None = None,
        scene_payload: dict[str, Any] | None = None,
        zone_stats: dict[str, dict[str, float]] | None = None,
    ) -> bool:
        self.add_calls.append(
            {
                "object_states": object_states,
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "events": events,
                "scene_payload": scene_payload,
                "zone_stats": zone_stats,
            }
        )
        return False

    def flush(self) -> None:
        self.flush_count += 1


class FakePostProcessor:
    """Post-processor stand-in for runner post-processing tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_sequence(self, sequence_id: str) -> dict[str, Any]:
        self.calls.append(sequence_id)
        return {
            "sequence_id": sequence_id,
            "object_count": 1,
            "statement_count": 2,
            "post_processed": True,
        }


class FakeGraphAgent:
    """Graph agent stand-in for testing the BatchWriter adapter."""

    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.flush_count = 0

    def add_frame_data(self, **kwargs: Any) -> bool:
        self.add_calls.append(kwargs)
        return True

    def flush(self) -> None:
        self.flush_count += 1


def write_detection_config(config_path: Path) -> None:
    """Write a minimal detection config for runner tests."""
    config_path.write_text(
        textwrap.dedent(
            """
            model:
              preferred_path: weights/yolov8m_visdrone.pt
              fallback_path: weights/yolov8m.pt
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


def reset_fake_loader() -> None:
    """Clear fake-loader call history between tests."""
    FakeSequenceLoader.init_calls = []
    FakeSequenceLoader.iter_calls = []


def build_runner(
    config_path: Path,
    batch_writer: FakeBatchWriter | None = None,
    post_processor: FakePostProcessor | None = None,
) -> tuple[PipelineRunner, dict[str, Any]]:
    """Build a runner with fake agent dependencies for integration tests."""
    dependencies = {
        "detection_agent": FakeDetectionAgent(),
        "tracking_agent": FakeTrackingAgent(),
        "motion_agent": FakeMotionAgent(),
        "event_agent": FakeEventAgent(),
        "batch_writer": batch_writer or FakeBatchWriter(),
        "post_processor": post_processor or FakePostProcessor(),
    }
    runner = PipelineRunner(
        config_path=config_path,
        loader_factory=FakeSequenceLoader,
        detection_agent=dependencies["detection_agent"],
        tracking_agent=dependencies["tracking_agent"],
        motion_agent=dependencies["motion_agent"],
        event_agent=dependencies["event_agent"],
        batch_writer=dependencies["batch_writer"],
        post_processor=dependencies["post_processor"],
    )
    return runner, dependencies


def test_batch_writer_forwards_frame_data_and_flushes() -> None:
    """The BatchWriter adapter should delegate to the graph agent."""
    graph_agent = FakeGraphAgent()
    batch_writer = BatchWriter(graph_agent)

    did_flush = batch_writer.add_frame_data(
        object_states=[{"track_id": 1}],
        sequence_id="seq_a",
        frame_id=3,
        events=[{"event_type": "LOITER"}],
        scene_payload={"sequence_id": "seq_a"},
        zone_stats={"cell_0_0": {"last_density": 1.0}},
    )
    batch_writer.flush()

    assert did_flush is True
    assert graph_agent.add_calls == [
        {
            "object_states": [{"track_id": 1}],
            "sequence_id": "seq_a",
            "frame_id": 3,
            "events": [{"event_type": "LOITER"}],
            "scene_payload": {"sequence_id": "seq_a"},
            "zone_stats": {"cell_0_0": {"last_density": 1.0}},
        }
    ]
    assert graph_agent.flush_count == 1


def test_pipeline_runner_processes_all_frames_in_sequence(tmp_path: Path) -> None:
    """The runner should execute the full agent chain for every frame."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    runner, dependencies = build_runner(config_path)

    results = runner.run_sequence("uav0000009_04358_v")

    assert [result["frame_packet"]["frame_id"] for result in results] == [0, 1]
    assert all(len(result["detections"]) == 1 for result in results)
    assert all(len(result["tracked_detections"]) == 2 for result in results)
    assert all(len(result["active_tracked_detections"]) == 1 for result in results)
    assert all(len(result["enriched_tracks"]) == 1 for result in results)
    assert all(result["events"][0]["event_type"] == "NEAR_MISS" for result in results)
    assert FakeSequenceLoader.init_calls[0]["sequence_id"] == "uav0000009_04358_v"
    assert FakeSequenceLoader.iter_calls == [{"sequence_id": "uav0000009_04358_v", "frame_skip": None}]
    assert len(dependencies["detection_agent"].infer_calls) == 2
    assert len(dependencies["detection_agent"].format_calls) == 2
    assert len(dependencies["tracking_agent"].calls) == 2


def test_pipeline_runner_passes_frame_skip_to_loader(tmp_path: Path) -> None:
    """The runner should forward frame-skip overrides to the loader."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    runner, dependencies = build_runner(config_path)

    results = runner.run_sequence("uav0000013_00000_v", frame_skip=3)

    assert all(result["frame_packet"]["frame_skip"] == 3 for result in results)
    assert FakeSequenceLoader.iter_calls == [{"sequence_id": "uav0000013_00000_v", "frame_skip": 3}]
    assert dependencies["batch_writer"].flush_count == 1


def test_pipeline_runner_filters_lost_tracks_before_motion_and_events(tmp_path: Path) -> None:
    """Lost tracks should not be sent into motion enrichment or event detection."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    runner, dependencies = build_runner(config_path)

    runner.run_sequence("uav0000073_00600_v")

    motion_calls = dependencies["motion_agent"].calls
    event_calls = dependencies["event_agent"].calls
    assert all(len(call["tracked_detections"]) == 1 for call in motion_calls)
    assert all(call["tracked_detections"][0]["track_id"] == 101 for call in motion_calls)
    assert all(list(call["tracks"].keys()) == [101] for call in event_calls)


def test_pipeline_runner_writes_enriched_tracks_events_and_scene_payload(tmp_path: Path) -> None:
    """Graph integration should receive enriched tracks plus per-frame events."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    batch_writer = FakeBatchWriter()
    runner, _ = build_runner(config_path, batch_writer=batch_writer)

    runner.run_sequence("uav0000119_02301_v")

    assert len(batch_writer.add_calls) == 2
    assert batch_writer.add_calls[0]["sequence_id"] == "uav0000119_02301_v"
    assert batch_writer.add_calls[0]["frame_id"] == 0
    assert batch_writer.add_calls[0]["scene_payload"] == {
        "sequence_id": "uav0000119_02301_v",
        "scene_type": "urban",
    }
    assert batch_writer.add_calls[1]["scene_payload"] is None
    assert batch_writer.add_calls[0]["object_states"][0]["track_id"] == 101
    assert batch_writer.add_calls[0]["events"][0]["event_type"] == "NEAR_MISS"
    assert batch_writer.add_calls[0]["zone_stats"] == {
        "cell_0_0": {"last_density": 1.0, "vehicle_ratio": 1.0, "pedestrian_ratio": 0.0}
    }


def test_pipeline_runner_processes_multiple_sequences_and_flushes_each_one(tmp_path: Path) -> None:
    """Each sequence should be processed in order and flushed on completion."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    batch_writer = FakeBatchWriter()
    runner, _ = build_runner(config_path, batch_writer=batch_writer)

    results = runner.run_sequences(["seq_a", "seq_b"], frame_skip=2)

    assert list(results.keys()) == ["seq_a", "seq_b"]
    assert [result["frame_packet"]["frame_id"] for result in results["seq_a"]] == [0, 1]
    assert [result["frame_packet"]["frame_id"] for result in results["seq_b"]] == [0, 1]
    assert FakeSequenceLoader.iter_calls == [
        {"sequence_id": "seq_a", "frame_skip": 2},
        {"sequence_id": "seq_b", "frame_skip": 2},
    ]
    assert batch_writer.flush_count == 2


def test_pipeline_runner_runs_post_processor_when_enabled(tmp_path: Path) -> None:
    """Sequence-final post-processing should run only when explicitly enabled."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    post_processor = FakePostProcessor()
    runner, dependencies = build_runner(config_path, post_processor=post_processor)

    runner.run_sequence("seq_post", post_process=True)

    assert dependencies["batch_writer"].flush_count == 1
    assert post_processor.calls == ["seq_post"]
    assert runner.last_postprocess_summary == {
        "sequence_id": "seq_post",
        "object_count": 1,
        "statement_count": 2,
        "post_processed": True,
    }


def test_run_pipeline_cli_returns_compact_summary(tmp_path: Path) -> None:
    """The runner CLI helper should return frame counts and post-processing state."""
    reset_fake_loader()
    config_path = tmp_path / "detection.yaml"
    write_detection_config(config_path)
    runner, _ = build_runner(config_path)

    summary = run_pipeline_cli(
        ["seq_a", "seq_b"],
        frame_skip=2,
        post_process=False,
        runner=runner,
    )

    assert summary == {
        "sequence_ids": ["seq_a", "seq_b"],
        "frame_counts": {"seq_a": 2, "seq_b": 2},
        "post_process_enabled": False,
        "last_postprocess_summary": None,
    }
