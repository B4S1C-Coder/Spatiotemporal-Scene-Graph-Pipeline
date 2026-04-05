"""Tests for entity-resolution comparison, scoring, and graph statements."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

from agents.entity_resolution_agent import (
    EntityResolutionAgent,
    compare_track_pair,
    load_entity_resolution_config,
    score_track_pair,
)


def write_entity_resolution_config(config_path: Path, ambiguous_log_path: Path | None = None) -> None:
    """Write a minimal entity-resolution config for tests."""
    log_path = ambiguous_log_path or (config_path.parent / "reid_ambiguous.jsonl")
    config_path.write_text(
        textwrap.dedent(
            f"""
            reid:
              spatial_threshold: 0.10
              temporal_gap_frames: 60
              heading_diff_max_deg: 45.0
              confidence_threshold: 0.70
              ambiguity_delta: 0.05
              ambiguous_log_path: {log_path}
              weights:
                spatial: 0.35
                temporal: 0.25
                heading: 0.20
                class: 0.20
            coexists:
              default_min_distance: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def build_track(
    track_id: int,
    *,
    first_seen_frame: int,
    last_seen_frame: int,
    track_class: str = "car",
    first_centroid: list[float] | None = None,
    last_centroid: list[float] | None = None,
    heading_at_entry: float = 0.0,
    heading_at_exit: float = 0.0,
    trajectory_buffer: list[list[float]] | None = None,
) -> dict[str, object]:
    """Build a minimal Object-like payload for entity-resolution tests."""
    return {
        "track_id": track_id,
        "first_seen_frame": first_seen_frame,
        "last_seen_frame": last_seen_frame,
        "class": track_class,
        "first_centroid": first_centroid or [0.20, 0.20],
        "last_centroid": last_centroid or [0.25, 0.25],
        "heading_at_entry": heading_at_entry,
        "heading_at_exit": heading_at_exit,
        "trajectory_buffer": trajectory_buffer or [[0.20, 0.20], [0.25, 0.25]],
    }


def test_load_entity_resolution_config_reads_yaml_values(tmp_path: Path) -> None:
    """Entity-resolution settings should be loaded from YAML."""
    config_path = tmp_path / "entity_resolution.yaml"
    write_entity_resolution_config(config_path)

    config = load_entity_resolution_config(config_path=config_path)

    assert config["reid"]["spatial_threshold"] == 0.10
    assert config["reid"]["confidence_threshold"] == 0.70
    assert config["reid"]["weights"]["spatial"] == 0.35
    assert config["coexists"]["default_min_distance"] == 1.0


def test_compare_track_pair_marks_valid_candidate_pair(tmp_path: Path) -> None:
    """A pair meeting all five rules should be marked as a re-ID candidate."""
    config_path = tmp_path / "entity_resolution.yaml"
    write_entity_resolution_config(config_path)
    agent = EntityResolutionAgent(config_path=config_path)

    comparison = agent.compare_track_pair(
        build_track(
            10,
            first_seen_frame=0,
            last_seen_frame=100,
            last_centroid=[0.30, 0.30],
            heading_at_exit=15.0,
        ),
        build_track(
            11,
            first_seen_frame=120,
            last_seen_frame=180,
            first_centroid=[0.34, 0.32],
            heading_at_entry=30.0,
        ),
    )

    assert comparison["track_a_id"] == 10
    assert comparison["track_b_id"] == 11
    assert comparison["track_b_first_seen_frame"] == 120
    assert comparison["temporal_gap_frames"] == 20
    assert comparison["same_class"] is True
    assert comparison["conditions"] == {
        "temporal_order": True,
        "spatial_continuity": True,
        "class_match": True,
        "temporal_gap": True,
        "motion_continuity": True,
    }
    assert comparison["is_candidate_pair"] is True


def test_compare_track_pair_rejects_overlapping_tracks() -> None:
    """Overlapping lifespans should fail the temporal-order rule."""
    comparison = compare_track_pair(
        build_track(10, first_seen_frame=0, last_seen_frame=100),
        build_track(11, first_seen_frame=90, last_seen_frame=140),
        config={
            "reid": {
                "spatial_threshold": 0.10,
                "temporal_gap_frames": 60,
                "heading_diff_max_deg": 45.0,
            }
        },
    )

    assert comparison["conditions"]["temporal_order"] is False
    assert comparison["is_candidate_pair"] is False


def test_compare_track_pair_supports_trajectory_and_heading_deg_fallbacks() -> None:
    """The comparison helper should fall back to trajectory and heading_deg fields."""
    comparison = compare_track_pair(
        {
            "track_id": 1,
            "first_seen_frame": 0,
            "last_seen_frame": 40,
            "class_name": "bus",
            "trajectory_buffer": [[0.10, 0.10], [0.12, 0.11]],
            "heading_deg": 350.0,
        },
        {
            "track_id": 2,
            "first_seen_frame": 50,
            "last_seen_frame": 80,
            "class_name": "bus",
            "trajectory_buffer": [[0.15, 0.12], [0.18, 0.13]],
            "heading_deg": 5.0,
        },
        config={
            "reid": {
                "spatial_threshold": 0.10,
                "temporal_gap_frames": 60,
                "heading_diff_max_deg": 45.0,
            }
        },
    )

    assert comparison["heading_difference_deg"] == 15.0
    assert comparison["conditions"]["class_match"] is True
    assert comparison["is_candidate_pair"] is True


def test_compare_track_pair_rejects_large_spatial_or_heading_jumps() -> None:
    """Large discontinuities should fail the corresponding candidate rules."""
    comparison = compare_track_pair(
        build_track(7, first_seen_frame=0, last_seen_frame=10, last_centroid=[0.05, 0.05], heading_at_exit=0.0),
        build_track(
            8,
            first_seen_frame=20,
            last_seen_frame=35,
            first_centroid=[0.50, 0.50],
            heading_at_entry=120.0,
        ),
        config={
            "reid": {
                "spatial_threshold": 0.10,
                "temporal_gap_frames": 60,
                "heading_diff_max_deg": 45.0,
            }
        },
    )

    assert comparison["conditions"]["spatial_continuity"] is False
    assert comparison["conditions"]["motion_continuity"] is False
    assert comparison["is_candidate_pair"] is False


def test_score_track_pair_computes_weighted_confidence_for_candidate() -> None:
    """Candidate pairs should receive a weighted confidence score from the config."""
    comparison = score_track_pair(
        build_track(
            10,
            first_seen_frame=0,
            last_seen_frame=100,
            last_centroid=[0.30, 0.30],
            heading_at_exit=10.0,
        ),
        build_track(
            11,
            first_seen_frame=110,
            last_seen_frame=170,
            first_centroid=[0.32, 0.31],
            heading_at_entry=15.0,
        ),
        config={
            "reid": {
                "spatial_threshold": 0.10,
                "temporal_gap_frames": 60,
                "heading_diff_max_deg": 45.0,
                "weights": {
                    "spatial": 0.35,
                    "temporal": 0.25,
                    "heading": 0.20,
                    "class": 0.20,
                },
            }
        },
    )

    assert comparison["score_components"]["class_score"] == 1.0
    assert 0.0 < comparison["score_components"]["spatial_score"] <= 1.0
    assert 0.0 < comparison["score_components"]["temporal_score"] <= 1.0
    assert 0.0 < comparison["score_components"]["heading_score"] <= 1.0
    assert comparison["confidence"] > 0.70


def test_score_track_pair_zeroes_confidence_for_invalid_candidate() -> None:
    """Pairs failing the candidate rules should not receive a usable confidence score."""
    scored = score_track_pair(
        build_track(1, first_seen_frame=0, last_seen_frame=40, track_class="car"),
        build_track(2, first_seen_frame=50, last_seen_frame=80, track_class="bus"),
        config={
            "reid": {
                "spatial_threshold": 0.10,
                "temporal_gap_frames": 60,
                "heading_diff_max_deg": 45.0,
                "weights": {
                    "spatial": 0.35,
                    "temporal": 0.25,
                    "heading": 0.20,
                    "class": 0.20,
                },
            }
        },
    )

    assert scored["conditions"]["class_match"] is False
    assert scored["confidence"] == 0.0


def test_process_sequence_objects_builds_same_entity_and_canonical_id_statements(tmp_path: Path) -> None:
    """Confirmed re-ID matches should produce SAME_ENTITY_AS and canonical-id updates."""
    config_path = tmp_path / "entity_resolution.yaml"
    write_entity_resolution_config(config_path, ambiguous_log_path=tmp_path / "reid_ambiguous.jsonl")
    agent = EntityResolutionAgent(config_path=config_path)

    statements = agent.process_sequence_objects(
        sequence_id="seq_a",
        object_nodes=[
            build_track(
                10,
                first_seen_frame=0,
                last_seen_frame=100,
                last_centroid=[0.30, 0.30],
                heading_at_exit=15.0,
            ),
            build_track(
                11,
                first_seen_frame=110,
                last_seen_frame=180,
                first_centroid=[0.31, 0.30],
                heading_at_entry=18.0,
            ),
        ],
    )

    queries = [statement.query for statement in statements]
    assert any("MERGE (a)-[r:SAME_ENTITY_AS]->(b)" in query for query in queries)
    assert any("SET a.canonical_id = $canonical_id" in query for query in queries)
    assert any("SET b.canonical_id = $canonical_id" in query for query in queries)
    same_entity_statement = next(
        statement for statement in statements if "MERGE (a)-[r:SAME_ENTITY_AS]->(b)" in statement.query
    )
    assert same_entity_statement.parameters["resolved_at_frame"] == 110
    assert same_entity_statement.parameters["confidence"] > 0.70


def test_process_sequence_objects_builds_coexists_with_overlap_stats(tmp_path: Path) -> None:
    """Overlapping track lifespans should produce COEXISTS_WITH relationships."""
    config_path = tmp_path / "entity_resolution.yaml"
    write_entity_resolution_config(config_path)
    agent = EntityResolutionAgent(config_path=config_path)

    statements = agent.process_sequence_objects(
        sequence_id="seq_b",
        object_nodes=[
            build_track(
                20,
                first_seen_frame=10,
                last_seen_frame=40,
                trajectory_buffer=[[0.10, 0.10], [0.15, 0.15], [0.20, 0.20]],
            ),
            build_track(
                21,
                first_seen_frame=25,
                last_seen_frame=55,
                trajectory_buffer=[[0.11, 0.10], [0.16, 0.15], [0.40, 0.40]],
            ),
        ],
    )

    coexists_statement = next(
        statement for statement in statements if "MERGE (a)-[r:COEXISTS_WITH]->(b)" in statement.query
    )
    assert coexists_statement.parameters["overlap_start"] == 25
    assert coexists_statement.parameters["overlap_end"] == 40
    assert coexists_statement.parameters["overlap_frames"] == 16
    assert coexists_statement.parameters["min_distance_ever"] < 0.02


def test_process_sequence_objects_logs_ambiguous_reid_candidates(tmp_path: Path) -> None:
    """Ambiguous best matches should be skipped and logged for review."""
    log_path = tmp_path / "reid_ambiguous.jsonl"
    config_path = tmp_path / "entity_resolution.yaml"
    write_entity_resolution_config(config_path, ambiguous_log_path=log_path)
    agent = EntityResolutionAgent(config_path=config_path)

    statements = agent.process_sequence_objects(
        sequence_id="seq_c",
        object_nodes=[
            build_track(
                30,
                first_seen_frame=0,
                last_seen_frame=100,
                last_centroid=[0.30, 0.30],
                heading_at_exit=10.0,
            ),
            build_track(
                31,
                first_seen_frame=120,
                last_seen_frame=180,
                first_centroid=[0.33, 0.31],
                heading_at_entry=15.0,
            ),
            build_track(
                32,
                first_seen_frame=122,
                last_seen_frame=175,
                first_centroid=[0.34, 0.31],
                heading_at_entry=16.0,
            ),
        ],
    )

    assert not any("SAME_ENTITY_AS" in statement.query for statement in statements)
    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 2
    payloads = [json.loads(line) for line in log_lines]
    assert {payload["track_b_id"] for payload in payloads} == {31, 32}
