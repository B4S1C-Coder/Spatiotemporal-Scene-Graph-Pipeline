"""
Entity-resolution utilities for post-processing track fragments.

This module compares track pairs, computes re-identification confidence scores,
and builds the graph-write statements for `SAME_ENTITY_AS` and
`COEXISTS_WITH` relationships.
"""

from __future__ import annotations

import json
from math import dist
from pathlib import Path
from typing import Any

from configs.loader import ENTITY_RESOLUTION_CONFIG_PATH, load_yaml_config
from graph.neo4j_client import GraphStatement


def load_entity_resolution_config(
    config_path: str | Path = ENTITY_RESOLUTION_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load entity-resolution settings from YAML.

    Args:
        config_path: YAML config file location.
        config: Optional runtime override mapping.

    Returns:
        Entity-resolution configuration dictionary.
    """
    return load_yaml_config(config_path, overrides=config)


class EntityResolutionAgent:
    """Agent wrapper for sequence-level post-processing and graph writes."""

    def __init__(
        self,
        config_path: str | Path = ENTITY_RESOLUTION_CONFIG_PATH,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = load_entity_resolution_config(config_path, config=config)
        self.ambiguous_log_path = Path(self.config["reid"]["ambiguous_log_path"])

    def compare_track_pair(
        self,
        track_a: dict[str, Any],
        track_b: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare two tracks against the re-identification candidate rules.
        """
        return compare_track_pair(track_a=track_a, track_b=track_b, config=self.config)

    def score_track_pair(
        self,
        track_a: dict[str, Any],
        track_b: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute a weighted confidence score for one re-identification candidate pair.
        """
        return score_track_pair(track_a=track_a, track_b=track_b, config=self.config)

    def process_sequence_objects(
        self,
        sequence_id: str,
        object_nodes: list[dict[str, Any]],
    ) -> list[GraphStatement]:
        """
        Build graph-write statements for entity-resolution post-processing.

        Args:
            sequence_id: Sequence identifier shared by all object nodes.
            object_nodes: Object payloads read from Neo4j for the sequence.

        Returns:
            Ordered list of graph statements for confirmed re-ID pairs and
            overlapping co-presence relationships.
        """
        sequence_objects = sorted(object_nodes, key=lambda item: int(item["first_seen_frame"]))
        statements: list[GraphStatement] = []
        statements.extend(_build_coexists_with_statements(sequence_id, sequence_objects, self.config))
        statements.extend(self._build_same_entity_statements(sequence_id, sequence_objects))
        return statements

    def _build_same_entity_statements(
        self,
        sequence_id: str,
        object_nodes: list[dict[str, Any]],
    ) -> list[GraphStatement]:
        scored_candidates: list[dict[str, Any]] = []
        for index, track_a in enumerate(object_nodes):
            for track_b in object_nodes[index + 1 :]:
                comparison = compare_track_pair(track_a, track_b, self.config)
                if not comparison["conditions"]["temporal_order"]:
                    continue
                scored_candidate = score_track_pair(track_a, track_b, self.config, comparison=comparison)
                if scored_candidate["confidence"] >= float(self.config["reid"]["confidence_threshold"]):
                    scored_candidates.append(scored_candidate)

        accepted_candidates = _filter_ambiguous_candidates(
            scored_candidates,
            ambiguity_delta=float(self.config["reid"]["ambiguity_delta"]),
        )
        ambiguous_candidates = [candidate for candidate in scored_candidates if candidate not in accepted_candidates]
        if ambiguous_candidates:
            self._log_ambiguous_candidates(sequence_id, ambiguous_candidates)

        statements: list[GraphStatement] = []
        for candidate in accepted_candidates:
            statements.extend(_build_same_entity_as_statements(sequence_id, candidate))
        return statements

    def _log_ambiguous_candidates(
        self,
        sequence_id: str,
        ambiguous_candidates: list[dict[str, Any]],
    ) -> None:
        self.ambiguous_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ambiguous_log_path.open("a", encoding="utf-8") as log_file:
            for candidate in ambiguous_candidates:
                log_file.write(
                    json.dumps(
                        {
                            "sequence_id": sequence_id,
                            "track_a_id": candidate["track_a_id"],
                            "track_b_id": candidate["track_b_id"],
                            "confidence": candidate["confidence"],
                        }
                    )
                    + "\n"
                )


def compare_track_pair(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two tracks using the Phase 9 candidate rules.

    Args:
        track_a: Earlier candidate track.
        track_b: Later candidate track.
        config: Entity-resolution configuration mapping.

    Returns:
        Derived comparison payload including condition results.
    """
    reid_config = config["reid"]
    end_frame = int(track_a["last_seen_frame"])
    start_frame = int(track_b["first_seen_frame"])
    temporal_gap = start_frame - end_frame
    spatial_distance = dist(_get_last_centroid(track_a), _get_first_centroid(track_b))
    heading_difference = _heading_difference(_get_exit_heading(track_a), _get_entry_heading(track_b))
    class_match = _get_track_class(track_a) == _get_track_class(track_b)

    conditions = {
        "temporal_order": end_frame < start_frame,
        "spatial_continuity": spatial_distance < float(reid_config["spatial_threshold"]),
        "class_match": class_match,
        "temporal_gap": temporal_gap < int(reid_config["temporal_gap_frames"]),
        "motion_continuity": heading_difference < float(reid_config["heading_diff_max_deg"]),
    }

    return {
        "track_a_id": int(track_a["track_id"]),
        "track_b_id": int(track_b["track_id"]),
        "track_a_last_seen_frame": end_frame,
        "track_b_first_seen_frame": start_frame,
        "temporal_gap_frames": temporal_gap,
        "spatial_distance": spatial_distance,
        "heading_difference_deg": heading_difference,
        "same_class": class_match,
        "conditions": conditions,
        "is_candidate_pair": all(conditions.values()),
    }


def score_track_pair(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    config: dict[str, Any],
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Score a compared track pair using the weighted confidence formula.

    Args:
        track_a: Earlier candidate track.
        track_b: Later candidate track.
        config: Entity-resolution configuration mapping.
        comparison: Optional precomputed comparison payload.

    Returns:
        Comparison payload extended with score components and confidence.
    """
    comparison_payload = comparison or compare_track_pair(track_a, track_b, config)
    reid_config = config["reid"]
    weights = reid_config["weights"]

    spatial_score = _normalized_inverse_score(
        comparison_payload["spatial_distance"],
        float(reid_config["spatial_threshold"]),
    )
    temporal_score = _normalized_inverse_score(
        max(float(comparison_payload["temporal_gap_frames"]), 0.0),
        float(reid_config["temporal_gap_frames"]),
    )
    heading_score = _normalized_inverse_score(
        comparison_payload["heading_difference_deg"],
        float(reid_config["heading_diff_max_deg"]),
    )
    class_score = 1.0 if comparison_payload["same_class"] else 0.0

    confidence = (
        spatial_score * float(weights["spatial"])
        + temporal_score * float(weights["temporal"])
        + heading_score * float(weights["heading"])
        + class_score * float(weights["class"])
    )
    if not comparison_payload["is_candidate_pair"]:
        confidence = 0.0

    return {
        **comparison_payload,
        "score_components": {
            "spatial_score": spatial_score,
            "temporal_score": temporal_score,
            "heading_score": heading_score,
            "class_score": class_score,
        },
        "confidence": confidence,
    }


def _build_same_entity_as_statements(
    sequence_id: str,
    scored_candidate: dict[str, Any],
) -> list[GraphStatement]:
    track_a_id = int(scored_candidate["track_a_id"])
    track_b_id = int(scored_candidate["track_b_id"])
    canonical_id = min(track_a_id, track_b_id)
    return [
        GraphStatement(
            query=(
                "MATCH (a:Object {track_id: $track_a_id, sequence_id: $sequence_id}) "
                "MATCH (b:Object {track_id: $track_b_id, sequence_id: $sequence_id}) "
                "MERGE (a)-[r:SAME_ENTITY_AS]->(b) "
                "SET r.confidence = $confidence, "
                "r.method = $method, "
                "r.resolved_at_frame = $resolved_at_frame"
            ),
            parameters={
                "track_a_id": track_a_id,
                "track_b_id": track_b_id,
                "sequence_id": sequence_id,
                "confidence": float(scored_candidate["confidence"]),
                "method": "spatial_temporal_reid",
                "resolved_at_frame": int(scored_candidate["track_b_first_seen_frame"]),
            },
        ),
        GraphStatement(
            query=(
                "MATCH (a:Object {track_id: $track_a_id, sequence_id: $sequence_id}) "
                "SET a.canonical_id = $canonical_id"
            ),
            parameters={
                "track_a_id": track_a_id,
                "sequence_id": sequence_id,
                "canonical_id": canonical_id,
            },
        ),
        GraphStatement(
            query=(
                "MATCH (b:Object {track_id: $track_b_id, sequence_id: $sequence_id}) "
                "SET b.canonical_id = $canonical_id"
            ),
            parameters={
                "track_b_id": track_b_id,
                "sequence_id": sequence_id,
                "canonical_id": canonical_id,
            },
        ),
    ]


def _build_coexists_with_statements(
    sequence_id: str,
    object_nodes: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[GraphStatement]:
    statements: list[GraphStatement] = []
    for index, track_a in enumerate(object_nodes):
        for track_b in object_nodes[index + 1 :]:
            overlap_start = max(int(track_a["first_seen_frame"]), int(track_b["first_seen_frame"]))
            overlap_end = min(int(track_a["last_seen_frame"]), int(track_b["last_seen_frame"]))
            if overlap_start > overlap_end:
                continue
            overlap_frames = overlap_end - overlap_start + 1
            min_distance_ever = _minimum_pair_distance(track_a, track_b, config)
            statements.append(
                GraphStatement(
                    query=(
                        "MATCH (a:Object {track_id: $track_a_id, sequence_id: $sequence_id}) "
                        "MATCH (b:Object {track_id: $track_b_id, sequence_id: $sequence_id}) "
                        "MERGE (a)-[r:COEXISTS_WITH]->(b) "
                        "SET r.overlap_start = $overlap_start, "
                        "r.overlap_end = $overlap_end, "
                        "r.overlap_frames = $overlap_frames, "
                        "r.min_distance_ever = $min_distance_ever"
                    ),
                    parameters={
                        "track_a_id": int(track_a["track_id"]),
                        "track_b_id": int(track_b["track_id"]),
                        "sequence_id": sequence_id,
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_frames": overlap_frames,
                        "min_distance_ever": min_distance_ever,
                    },
                )
            )
    return statements


def _filter_ambiguous_candidates(
    scored_candidates: list[dict[str, Any]],
    ambiguity_delta: float,
) -> list[dict[str, Any]]:
    blocked_track_ids: set[int] = set()
    grouped_candidates: dict[int, list[dict[str, Any]]] = {}
    for candidate in scored_candidates:
        grouped_candidates.setdefault(int(candidate["track_a_id"]), []).append(candidate)

    accepted_candidates: list[dict[str, Any]] = []
    for track_a_id, candidates in grouped_candidates.items():
        ranked_candidates = sorted(candidates, key=lambda item: float(item["confidence"]), reverse=True)
        if len(ranked_candidates) > 1:
            confidence_gap = float(ranked_candidates[0]["confidence"]) - float(ranked_candidates[1]["confidence"])
            if confidence_gap <= ambiguity_delta:
                blocked_track_ids.add(track_a_id)
                continue
        best_candidate = ranked_candidates[0]
        track_b_id = int(best_candidate["track_b_id"])
        if track_b_id in blocked_track_ids:
            continue
        accepted_candidates.append(best_candidate)
        blocked_track_ids.add(track_b_id)
    return accepted_candidates


def _normalized_inverse_score(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (value / threshold)))


def _minimum_pair_distance(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    config: dict[str, Any],
) -> float:
    trajectory_a = _get_trajectory(track_a)
    trajectory_b = _get_trajectory(track_b)
    if not trajectory_a or not trajectory_b:
        return float(config["coexists"]["default_min_distance"])
    pair_count = min(len(trajectory_a), len(trajectory_b))
    distances = [dist(trajectory_a[index], trajectory_b[index]) for index in range(pair_count)]
    return min(distances) if distances else float(config["coexists"]["default_min_distance"])


def _get_track_class(track: dict[str, Any]) -> str:
    if "class" in track:
        return str(track["class"])
    return str(track["class_name"])


def _get_last_centroid(track: dict[str, Any]) -> list[float]:
    centroid = track.get("last_centroid")
    if centroid is None:
        trajectory_buffer = track.get("trajectory_buffer") or []
        if not trajectory_buffer:
            raise KeyError("Track is missing both last_centroid and trajectory_buffer")
        centroid = trajectory_buffer[-1]
    return [float(value) for value in centroid]


def _get_first_centroid(track: dict[str, Any]) -> list[float]:
    centroid = track.get("first_centroid")
    if centroid is None:
        trajectory_buffer = track.get("trajectory_buffer") or []
        if not trajectory_buffer:
            raise KeyError("Track is missing both first_centroid and trajectory_buffer")
        centroid = trajectory_buffer[0]
    return [float(value) for value in centroid]


def _get_exit_heading(track: dict[str, Any]) -> float:
    if "heading_at_exit" in track:
        return float(track["heading_at_exit"])
    if "heading_deg" in track:
        return float(track["heading_deg"])
    raise KeyError("Track is missing heading_at_exit/heading_deg")


def _get_entry_heading(track: dict[str, Any]) -> float:
    if "heading_at_entry" in track:
        return float(track["heading_at_entry"])
    if "heading_deg" in track:
        return float(track["heading_deg"])
    raise KeyError("Track is missing heading_at_entry/heading_deg")


def _get_trajectory(track: dict[str, Any]) -> list[list[float]]:
    if "trajectory_buffer" not in track:
        return []
    return [[float(value) for value in point] for point in track["trajectory_buffer"]]


def _heading_difference(first_heading: float, second_heading: float) -> float:
    difference = abs((second_heading - first_heading) % 360.0)
    return min(difference, 360.0 - difference)
