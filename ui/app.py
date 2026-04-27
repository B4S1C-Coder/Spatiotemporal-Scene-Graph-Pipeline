"""
Streamlit UI for the scene-graph query interface.

The module keeps Streamlit imports inside `main()` so helper functions remain
testable in environments where the UI dependency is not installed.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Protocol

import cv2
import numpy as np
from PIL import Image

from agents.llm_agent import LLMQueryAgent
from graph.neo4j_client import Neo4jClient
from pipeline.sequence_loader import SequenceLoader


class QueryAgentProtocol(Protocol):
    """Minimal query-agent contract required by the UI."""

    def query(self, natural_language_query: str, sequence_id: str | None = None) -> dict[str, Any]:
        """Execute a natural-language graph query and return a result payload."""


class Neo4jClientProtocol(Protocol):
    """Minimal Neo4j client contract required by visualization helpers."""

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return result rows."""


def run_query(
    query_agent: QueryAgentProtocol,
    natural_language_query: str,
    sequence_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute a UI-issued graph query with input validation and error handling.
    """
    normalized_question = natural_language_query.strip()
    if not normalized_question:
        return {
            "question": normalized_question,
            "cypher": None,
            "results": [],
            "answer": "Enter a natural language question before running a query.",
            "error": "empty_question",
        }

    try:
        return query_agent.query(normalized_question, sequence_id=sequence_id or None)
    except Exception as exc:  # pragma: no cover - exercised via helper tests, not Streamlit
        return {
            "question": normalized_question,
            "cypher": None,
            "results": [],
            "answer": "The query failed before results could be returned.",
            "error": str(exc),
        }


def summarize_query_result(query_result: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact UI summary for one query response.
    """
    result_rows = list(query_result.get("results", []))
    return {
        "answer": str(query_result.get("answer", "")),
        "cypher": query_result.get("cypher"),
        "result_count": len(result_rows),
        "has_error": query_result.get("error") not in (None, ""),
        "error": query_result.get("error"),
    }


def build_result_table_payload(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert raw rows into a UI-stable table payload.
    """
    return [{str(key): value for key, value in row.items()} for row in result_rows]


def build_result_chart_spec(result_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Build a simple chart spec from result rows when one categorical and one numeric
    column are available.
    """
    if not result_rows:
        return None

    sample_row = result_rows[0]
    label_column = next(
        (
            key
            for key, value in sample_row.items()
            if isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))
        ),
        None,
    )
    value_column = next(
        (
            key
            for key, value in sample_row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        None,
    )
    if label_column is None or value_column is None or label_column == value_column:
        return None

    chart_rows = [
        {"label": str(row.get(label_column)), "value": float(row.get(value_column))}
        for row in result_rows
        if isinstance(row.get(value_column), (int, float)) and not isinstance(row.get(value_column), bool)
    ]
    if not chart_rows:
        return None
    return {
        "label_column": label_column,
        "value_column": value_column,
        "rows": chart_rows,
    }


def build_default_query_agent() -> LLMQueryAgent:
    """
    Construct the default LLM query agent used by the Streamlit app.
    """
    neo4j_client = Neo4jClient()
    return LLMQueryAgent(neo4j_client=neo4j_client)


def extract_visual_targets(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract candidate frame and track IDs (and sequence ID) from query result rows.
    """
    frame_ids: set[int] = set()
    track_ids: set[int] = set()
    sequence_ids: set[str] = set()

    for row in result_rows:
        for key, value in row.items():
            key_lower = str(key).lower()
            if isinstance(value, str) and "sequence_id" in key_lower:
                sequence_ids.add(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if "frame" in key_lower:
                    frame_ids.add(int(value))
                elif "track" in key_lower:
                    if int(value) >= 0:
                        track_ids.add(int(value))

    return {
        "frame_ids": sorted(frame_ids),
        "track_ids": sorted(track_ids),
        "sequence_id": list(sequence_ids)[0] if sequence_ids else None,
    }


def fetch_frame_boxes(
    neo4j_client: Neo4jClientProtocol,
    sequence_id: str,
    frame_id: int,
    track_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch bounding-box overlays for one frame from the graph.
    """
    query = (
        "MATCH (o:Object {sequence_id: $seq_id})-[r:APPEARED_IN]->"
        "(f:Frame {frame_id: $frame_id, sequence_id: $seq_id}) "
    )
    parameters: dict[str, Any] = {
        "seq_id": sequence_id,
        "frame_id": frame_id,
    }
    if track_ids:
        query += "WHERE o.track_id IN $track_ids "
        parameters["track_ids"] = track_ids
    query += (
        "RETURN o.track_id AS track_id, "
        "o.class AS class_name, "
        "r.bbox_norm AS bbox_norm "
        "ORDER BY o.track_id ASC"
    )
    return neo4j_client.execute_query(query, parameters)


def resolve_frame_path(
    sequence_id: str,
    frame_id: int,
    sequence_loader_factory: Callable[..., Any] = SequenceLoader,
) -> Path | None:
    """
    Resolve an on-disk frame path for the given sequence and frame ID.
    """
    print(f"[DEBUG_VIS] Attempting to load sequence loader for '{sequence_id}' to resolve frame {frame_id}...")
    try:
        loader = sequence_loader_factory(sequence_id=sequence_id)
        paths = loader.get_sequence_paths()
        image_dir = getattr(paths, 'image_dir', 'mocked_dir')
        print(f"[DEBUG_VIS] Successfully loaded sequence paths. Image dir: {image_dir}. Total frames: {len(paths.frame_paths)}")
        for candidate_path in paths.frame_paths:
            if int(candidate_path.stem) - 1 == frame_id:
                print(f"[DEBUG_VIS] Found matching frame path: {candidate_path}")
                return candidate_path
        print(f"[DEBUG_VIS] Could not find frame ID {frame_id} among {len(paths.frame_paths)} candidate paths in {image_dir}")
    except Exception as e:
        print(f"[DEBUG_VIS] Exception resolving frame path for '{sequence_id}': {e}")
    return None


def render_bounding_boxes(
    frame_bgr: np.ndarray,
    overlays: list[dict[str, Any]],
) -> np.ndarray:
    """
    Render graph-derived bounding boxes onto a frame.
    """
    rendered = frame_bgr.copy()
    height, width = rendered.shape[:2]
    for overlay in overlays:
        bbox_norm = overlay.get("bbox_norm")
        if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
            continue
        x1 = max(0, min(width - 1, int(round(float(bbox_norm[0]) * width))))
        y1 = max(0, min(height - 1, int(round(float(bbox_norm[1]) * height))))
        x2 = max(0, min(width - 1, int(round(float(bbox_norm[2]) * width))))
        y2 = max(0, min(height - 1, int(round(float(bbox_norm[3]) * height))))
        label = f"{overlay.get('class_name', 'object')} #{overlay.get('track_id', '?')}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            rendered,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


def build_preview_frames(
    sequence_id: str,
    frame_ids: list[int],
    track_ids: list[int],
    neo4j_client: Neo4jClientProtocol,
    sequence_loader_factory: Callable[..., Any] = SequenceLoader,
    max_frames: int = 3,
) -> list[dict[str, Any]]:
    """
    Build a small set of rendered preview frames from query results.
    """
    previews: list[dict[str, Any]] = []
    for frame_id in frame_ids[:max_frames]:
        frame_path = resolve_frame_path(
            sequence_id=sequence_id,
            frame_id=frame_id,
            sequence_loader_factory=sequence_loader_factory,
        )
        if frame_path is None:
            print(f"[DEBUG_VIS] Skipping preview for frame {frame_id} because resolve_frame_path returned None.")
            continue
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            print(f"[DEBUG_VIS] Skipping preview for frame {frame_id} because cv2.imread failed to load {frame_path}.")
            continue
        overlays = fetch_frame_boxes(neo4j_client, sequence_id, frame_id, track_ids=track_ids or None)
        rendered = render_bounding_boxes(frame_bgr, overlays)
        previews.append(
            {
                "frame_id": frame_id,
                "frame_path": str(frame_path),
                "image_rgb": cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB),
                "overlay_count": len(overlays),
            }
        )
    return previews


def build_clip_gif_bytes(
    sequence_id: str,
    center_frame_id: int,
    track_ids: list[int],
    neo4j_client: Neo4jClientProtocol,
    sequence_loader_factory: Callable[..., Any] = SequenceLoader,
    radius: int = 2,
) -> bytes | None:
    """
    Build a short annotated GIF clip around a frame of interest.
    """
    rendered_frames: list[Image.Image] = []
    for frame_id in range(max(0, center_frame_id - radius), center_frame_id + radius + 1):
        frame_path = resolve_frame_path(
            sequence_id=sequence_id,
            frame_id=frame_id,
            sequence_loader_factory=sequence_loader_factory,
        )
        if frame_path is None:
            continue
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue
        overlays = fetch_frame_boxes(neo4j_client, sequence_id, frame_id, track_ids=track_ids or None)
        rendered = render_bounding_boxes(frame_bgr, overlays)
        rendered_frames.append(Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)))

    if not rendered_frames:
        return None

    buffer = BytesIO()
    rendered_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=rendered_frames[1:],
        duration=250,
        loop=0,
    )
    return buffer.getvalue()


def build_clip_video_bytes(
    sequence_id: str,
    center_frame_id: int,
    track_ids: list[int],
    neo4j_client: Neo4jClientProtocol,
    sequence_loader_factory: Callable[..., Any] = SequenceLoader,
    radius: int = 2,
    fps: int = 4,
    writer_factory: Callable[..., Any] = cv2.VideoWriter,
    fourcc_factory: Callable[..., int] = cv2.VideoWriter_fourcc,
) -> bytes | None:
    """
    Build a short annotated MP4 clip around a frame of interest.
    """
    rendered_frames: list[np.ndarray] = []
    for frame_id in range(max(0, center_frame_id - radius), center_frame_id + radius + 1):
        frame_path = resolve_frame_path(
            sequence_id=sequence_id,
            frame_id=frame_id,
            sequence_loader_factory=sequence_loader_factory,
        )
        if frame_path is None:
            continue
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue
        overlays = fetch_frame_boxes(neo4j_client, sequence_id, frame_id, track_ids=track_ids or None)
        rendered_frames.append(render_bounding_boxes(frame_bgr, overlays))

    if not rendered_frames:
        return None

    height, width = rendered_frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
        writer = writer_factory(
            temp_file.name,
            fourcc_factory(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not getattr(writer, "isOpened", lambda: True)():
            return None
        try:
            for frame in rendered_frames:
                writer.write(frame)
        finally:
            writer.release()

        video_bytes = Path(temp_file.name).read_bytes()
        return video_bytes or None


def build_query_visualization_payload(
    *,
    query_result: dict[str, Any],
    sequence_id: str | None,
    neo4j_client: Neo4jClientProtocol,
    sequence_loader_factory: Callable[..., Any] = SequenceLoader,
) -> dict[str, Any] | None:
    """
    Build UI visualization assets from query results when enough context exists.
    """
    result_rows = list(query_result.get("results", []))
    targets = extract_visual_targets(result_rows)
    print(f"[DEBUG_VIS] Extracted visual targets: {targets}")
    
    active_sequence_id = targets.get("sequence_id") or sequence_id
    print(f"[DEBUG_VIS] active_sequence_id resolved to: {active_sequence_id}")

    if not active_sequence_id:
        print("[DEBUG_VIS] active_sequence_id is None, returning None from payload builder")
        return None

    if not targets["frame_ids"] and targets["track_ids"]:
        query = (
            "MATCH (o:Object {sequence_id: $seq_id})-[r:APPEARED_IN]->(f:Frame) "
            "WHERE o.track_id IN $track_ids "
            "RETURN DISTINCT f.frame_id AS frame_id "
            "LIMIT 5"
        )
        parameters = {"seq_id": active_sequence_id, "track_ids": targets["track_ids"]}
        resolved_rows = neo4j_client.execute_query(query, parameters)
        for row in resolved_rows:
            frame_id = row.get("frame_id")
            if isinstance(frame_id, (int, float)) and not isinstance(frame_id, bool):
                targets["frame_ids"].append(int(frame_id))
        targets["frame_ids"] = sorted(list(set(targets["frame_ids"])))

    if not targets["frame_ids"] and not targets["track_ids"]:
        print("[DEBUG_VIS] No frame or track targets found. Defaulting to frame 0 of the sequence.")
        targets["frame_ids"] = [0]

    if not targets["frame_ids"]:
        return None

    preview_frames = build_preview_frames(
        sequence_id=active_sequence_id,
        frame_ids=targets["frame_ids"],
        track_ids=targets["track_ids"],
        neo4j_client=neo4j_client,
        sequence_loader_factory=sequence_loader_factory,
    )
    print(f"[DEBUG_VIS] preview_frames generated {len(preview_frames)} frames.")
    
    if not preview_frames:
        print("[DEBUG_VIS] preview_frames is empty, returning None from payload builder")
        return None

    clip_gif = build_clip_gif_bytes(
        sequence_id=active_sequence_id,
        center_frame_id=preview_frames[0]["frame_id"],
        track_ids=targets["track_ids"],
        neo4j_client=neo4j_client,
        sequence_loader_factory=sequence_loader_factory,
    )
    clip_video = build_clip_video_bytes(
        sequence_id=active_sequence_id,
        center_frame_id=preview_frames[0]["frame_id"],
        track_ids=targets["track_ids"],
        neo4j_client=neo4j_client,
        sequence_loader_factory=sequence_loader_factory,
    )
    return {
        "preview_frames": preview_frames,
        "clip_gif": clip_gif,
        "clip_video": clip_video,
        "track_ids": targets["track_ids"],
        "frame_ids": targets["frame_ids"],
    }


def main() -> None:
    """Run the Streamlit application."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - interactive runtime path
        raise RuntimeError(
            "Streamlit is not installed. Install dependencies from requirements.txt "
            "or run `pip install streamlit`."
        ) from exc

    st.set_page_config(page_title="Scene Graph Query UI", layout="wide")
    st.title("Spatiotemporal Scene Graph")
    st.caption("Query the Neo4j scene graph with natural language.")

    with st.sidebar:
        st.header("Status")
        st.write("Pipeline phases 1-13 are implemented in the repository.")
        st.write("The UI currently focuses on graph querying, not live video playback.")
        st.write("Neo4j must be running and reachable before submitting queries.")

    @st.cache_resource
    def _get_query_agent() -> LLMQueryAgent:
        return build_default_query_agent()

    query_agent = _get_query_agent()

    left_column, right_column = st.columns([2, 3])
    with left_column:
        sequence_id = st.text_input("Sequence ID", placeholder="uav0000086_00000_v", value="uav0000086_00000_v")
        show_frames = st.checkbox("Show frame visualization", value=True)
        natural_language_query = st.text_area(
            "Natural Language Query",
            placeholder="Which vehicles were stationary for more than 60 frames?",
            height=180,
        )
        submit = st.button("Run Query", type="primary", use_container_width=True)

    if submit:
        with right_column:
            with st.spinner("Generating Cypher, querying Neo4j, and interpreting results..."):
                query_result = run_query(query_agent, natural_language_query, sequence_id or None)
        summary = summarize_query_result(query_result)

        with right_column:
            if summary["has_error"]:
                st.error(summary["answer"])
                st.caption(f"Error: {summary['error']}")
            else:
                st.success(summary["answer"])

            st.subheader("Generated Cypher")
            st.code(summary["cypher"] or "No Cypher generated.", language="cypher")

            st.subheader("Result Table")
            result_rows = build_result_table_payload(list(query_result.get("results", [])))
            if result_rows:
                st.dataframe(result_rows, use_container_width=True)
                st.caption(f"{summary['result_count']} rows returned.")

                chart_spec = build_result_chart_spec(result_rows)
                if chart_spec is not None:
                    st.subheader("Quick Visualization")
                    st.bar_chart(
                        {row["label"]: row["value"] for row in chart_spec["rows"]},
                    )

                with st.expander("Raw JSON"):
                    st.code(json.dumps(result_rows, indent=2, sort_keys=True), language="json")
            else:
                st.info("No result rows returned.")

            visualization_payload = None
            if show_frames:
                visualization_payload = build_query_visualization_payload(
                    query_result=query_result,
                    sequence_id=sequence_id or None,
                    neo4j_client=query_agent.neo4j_client,
                )
            if visualization_payload is not None:
                st.subheader("Frame Visualization")
                for preview in visualization_payload["preview_frames"]:
                    st.image(
                        preview["image_rgb"],
                        caption=(
                            f"Frame {preview['frame_id']} | overlays: {preview['overlay_count']} | "
                            f"{Path(preview['frame_path']).name}"
                        ),
                        use_container_width=True,
                    )

                if visualization_payload["clip_gif"] is not None:
                    st.subheader("Annotated Clip")
                    if visualization_payload["clip_video"] is not None:
                        st.video(visualization_payload["clip_video"])
                    else:
                        st.image(
                            visualization_payload["clip_gif"],
                            caption="GIF fallback clip centered on the first matched frame.",
                            use_container_width=True,
                        )
            elif show_frames:
                st.info("Visualization skipped: No valid visual targets (e.g. track IDs or frame IDs) were found in the query results, or the sequence images could not be loaded from disk.")


if __name__ == "__main__":  # pragma: no cover - interactive runtime path
    main()
