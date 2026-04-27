"""Tests for the Streamlit UI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ui.app import (
    build_clip_gif_bytes,
    build_clip_video_bytes,
    build_preview_frames,
    build_query_visualization_payload,
    build_result_chart_spec,
    build_result_table_payload,
    extract_visual_targets,
    render_bounding_boxes,
    run_query,
    summarize_query_result,
)


class FakeQueryAgent:
    """Query-agent stand-in for UI helper tests."""

    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result or {
            "question": "default",
            "cypher": "MATCH (n) RETURN n",
            "results": [{"value": 1}],
            "answer": "One row returned.",
            "error": None,
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def query(self, natural_language_query: str, sequence_id: str | None = None) -> dict[str, Any]:
        self.calls.append(
            {
                "natural_language_query": natural_language_query,
                "sequence_id": sequence_id,
            }
        )
        if self.error is not None:
            raise self.error
        return dict(self.result)


class FakeNeo4jClient:
    """Neo4j client double for visualization helpers."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[dict[str, Any]] = []

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "parameters": parameters or {}})
        return list(self.rows)


class FakeSequenceLoader:
    """Sequence loader double exposing a fixed frame list."""

    def __init__(self, sequence_id: str, *_args: Any, **_kwargs: Any) -> None:
        self.sequence_id = sequence_id

    def get_sequence_paths(self):
        return type("Paths", (), {"frame_paths": tuple(self.frame_paths)})


class FakeVideoWriter:
    """Video writer double that emits a fake MP4 payload."""

    def __init__(self, output_path: str, *_args: Any, **_kwargs: Any) -> None:
        self.output_path = Path(output_path)
        self.frames: list[np.ndarray] = []

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.output_path.write_bytes(b"fake-mp4")


def test_run_query_rejects_empty_questions() -> None:
    """UI queries should fail fast on blank input."""
    agent = FakeQueryAgent()

    result = run_query(agent, "   ", sequence_id="seq_a")

    assert result["answer"] == "Enter a natural language question before running a query."
    assert result["error"] == "empty_question"
    assert agent.calls == []


def test_run_query_forwards_question_and_sequence_id() -> None:
    """Valid UI queries should pass through to the query agent."""
    agent = FakeQueryAgent()

    result = run_query(agent, "Show all cars", sequence_id="seq_a")

    assert result["answer"] == "One row returned."
    assert agent.calls == [{"natural_language_query": "Show all cars", "sequence_id": "seq_a"}]


def test_run_query_captures_agent_errors() -> None:
    """Unexpected query-agent errors should be turned into UI-safe responses."""
    agent = FakeQueryAgent(error=RuntimeError("neo4j unavailable"))

    result = run_query(agent, "Show all cars", sequence_id="seq_a")

    assert result["answer"] == "The query failed before results could be returned."
    assert result["error"] == "neo4j unavailable"


def test_summarize_query_result_returns_ui_summary() -> None:
    """Result summaries should expose the fields needed by the UI."""
    summary = summarize_query_result(
        {
            "answer": "Two rows returned.",
            "cypher": "MATCH (n) RETURN n",
            "results": [{"value": 1}, {"value": 2}],
            "error": None,
        }
    )

    assert summary == {
        "answer": "Two rows returned.",
        "cypher": "MATCH (n) RETURN n",
        "result_count": 2,
        "has_error": False,
        "error": None,
    }


def test_build_result_table_payload_normalizes_keys() -> None:
    """UI table payloads should preserve rows with string keys."""
    table_rows = build_result_table_payload([{1: "car", "count": 3}])

    assert table_rows == [{"1": "car", "count": 3}]


def test_build_result_chart_spec_extracts_simple_bar_chart_data() -> None:
    """Chart specs should be built when rows contain one label and one numeric column."""
    chart_spec = build_result_chart_spec(
        [
            {"class_name": "car", "count": 4},
            {"class_name": "bus", "count": 2},
        ]
    )

    assert chart_spec == {
        "label_column": "class_name",
        "value_column": "count",
        "rows": [
            {"label": "car", "value": 4.0},
            {"label": "bus", "value": 2.0},
        ],
    }


def test_extract_visual_targets_collects_frame_and_track_ids() -> None:
    """Visualization targets should be inferred from common result-row keys."""
    targets = extract_visual_targets(
        [
            {"frame_id": 12, "track_id": 7},
            {"primary_track_id": 9, "secondary_track_id": 11, "start_frame": 10},
        ]
    )

    assert targets == {
        "frame_ids": [10, 12],
        "track_ids": [7, 9, 11],
        "sequence_id": None,
    }


def test_render_bounding_boxes_draws_on_frame() -> None:
    """Overlay rendering should modify the frame for valid boxes."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    rendered = render_bounding_boxes(
        frame,
        [{"track_id": 7, "class_name": "car", "bbox_norm": [0.1, 0.1, 0.4, 0.4]}],
    )

    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0


def test_build_preview_frames_renders_images_from_sequence_and_graph(tmp_path: Path) -> None:
    """Preview frames should be loaded from disk and rendered with graph overlays."""
    frame_path = tmp_path / "0000001.jpg"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), image)
    FakeSequenceLoader.frame_paths = [frame_path]
    neo4j_client = FakeNeo4jClient(
        rows=[{"track_id": 7, "class_name": "car", "bbox_norm": [0.1, 0.1, 0.5, 0.5]}]
    )

    previews = build_preview_frames(
        sequence_id="seq_a",
        frame_ids=[0],
        track_ids=[7],
        neo4j_client=neo4j_client,
        sequence_loader_factory=FakeSequenceLoader,
    )

    assert len(previews) == 1
    assert previews[0]["frame_id"] == 0
    assert previews[0]["overlay_count"] == 1
    assert previews[0]["image_rgb"].shape == (32, 32, 3)


def test_build_clip_gif_bytes_returns_gif_payload(tmp_path: Path) -> None:
    """Clip rendering should return GIF bytes when surrounding frames exist."""
    frame_paths = []
    for index in range(1, 4):
        frame_path = tmp_path / f"{index:07d}.jpg"
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), image)
        frame_paths.append(frame_path)
    FakeSequenceLoader.frame_paths = frame_paths
    neo4j_client = FakeNeo4jClient(
        rows=[{"track_id": 1, "class_name": "person", "bbox_norm": [0.2, 0.2, 0.6, 0.6]}]
    )

    gif_bytes = build_clip_gif_bytes(
        sequence_id="seq_a",
        center_frame_id=1,
        track_ids=[1],
        neo4j_client=neo4j_client,
        sequence_loader_factory=FakeSequenceLoader,
        radius=1,
    )

    assert gif_bytes is not None
    assert gif_bytes[:6] in (b"GIF87a", b"GIF89a")


def test_build_clip_video_bytes_returns_video_payload(tmp_path: Path) -> None:
    """Clip rendering should return video bytes when a writer is available."""
    frame_paths = []
    for index in range(1, 4):
        frame_path = tmp_path / f"{index:07d}.jpg"
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), image)
        frame_paths.append(frame_path)
    FakeSequenceLoader.frame_paths = frame_paths
    neo4j_client = FakeNeo4jClient(
        rows=[{"track_id": 1, "class_name": "person", "bbox_norm": [0.2, 0.2, 0.6, 0.6]}]
    )

    video_bytes = build_clip_video_bytes(
        sequence_id="seq_a",
        center_frame_id=1,
        track_ids=[1],
        neo4j_client=neo4j_client,
        sequence_loader_factory=FakeSequenceLoader,
        radius=1,
        writer_factory=FakeVideoWriter,
        fourcc_factory=lambda *_args: 0,
    )

    assert video_bytes == b"fake-mp4"


def test_build_query_visualization_payload_defaults_to_frame_0_without_frame_context() -> None:
    """Queries without frame context should default to showing frame 0 of the sequence."""
    payload = build_query_visualization_payload(
        query_result={"results": [{"object_class": "car"}]},
        sequence_id="seq_a",
        neo4j_client=FakeNeo4jClient(),
        sequence_loader_factory=FakeSequenceLoader,
    )

    assert payload is not None
    assert payload["frame_ids"] == [0]
