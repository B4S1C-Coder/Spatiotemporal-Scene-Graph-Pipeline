"""
Tracking evaluation helpers and CLI.

This module computes lightweight MOTA/MOTP-style metrics from per-frame tracking
records without relying on external evaluation packages.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from eval.detection_metrics import bbox_iou


def evaluate_tracking_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate tracking predictions against per-frame ground truth.

    Args:
        predictions: Records with `frame_id`, `track_id`, and `bbox`.
        ground_truths: Records with `frame_id`, `track_id`, and `bbox`.
        iou_threshold: IoU threshold used for frame matching.

    Returns:
        Summary containing `mota`, `motp`, and ID switch counts.
    """
    frame_ids = sorted({int(item["frame_id"]) for item in predictions + ground_truths})
    false_negatives = 0
    false_positives = 0
    id_switches = 0
    matched_ious: list[float] = []
    previous_assignments: dict[int, int] = {}

    for frame_id in frame_ids:
        frame_predictions = [item for item in predictions if int(item["frame_id"]) == frame_id]
        frame_ground_truths = [item for item in ground_truths if int(item["frame_id"]) == frame_id]
        matched_pairs, unmatched_pred_indices, unmatched_gt_indices = _match_frame_tracks(
            frame_predictions,
            frame_ground_truths,
            iou_threshold=iou_threshold,
        )
        false_positives += len(unmatched_pred_indices)
        false_negatives += len(unmatched_gt_indices)

        for pred_index, gt_index, matched_iou in matched_pairs:
            matched_ious.append(matched_iou)
            gt_track_id = int(frame_ground_truths[gt_index]["track_id"])
            predicted_track_id = int(frame_predictions[pred_index]["track_id"])
            previous_track_id = previous_assignments.get(gt_track_id)
            if previous_track_id is not None and previous_track_id != predicted_track_id:
                id_switches += 1
            previous_assignments[gt_track_id] = predicted_track_id

    total_ground_truth_detections = len(ground_truths)
    mota = 0.0
    if total_ground_truth_detections > 0:
        mota = 1.0 - (
            (false_negatives + false_positives + id_switches) / total_ground_truth_detections
        )
    motp = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    total_ground_truth_tracks = len({int(item["track_id"]) for item in ground_truths}) or 1
    return {
        "mota": mota,
        "motp": motp,
        "id_switches": id_switches,
        "id_switch_rate": id_switches / total_ground_truth_tracks,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "target_mota": 0.40,
        "target_motp": 0.65,
    }


def _match_frame_tracks(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, ground_truth in enumerate(ground_truths):
            current_iou = bbox_iou(list(prediction["bbox"]), list(ground_truth["bbox"]))
            if current_iou >= iou_threshold:
                candidate_pairs.append((current_iou, pred_index, gt_index))
    candidate_pairs.sort(reverse=True)

    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []
    for matched_iou, pred_index, gt_index in candidate_pairs:
        if pred_index in matched_pred_indices or gt_index in matched_gt_indices:
            continue
        matched_pred_indices.add(pred_index)
        matched_gt_indices.add(gt_index)
        matched_pairs.append((pred_index, gt_index, matched_iou))

    unmatched_pred_indices = set(range(len(predictions))) - matched_pred_indices
    unmatched_gt_indices = set(range(len(ground_truths))) - matched_gt_indices
    return matched_pairs, unmatched_pred_indices, unmatched_gt_indices


def _load_tracking_payload(input_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return list(payload.get("predictions", [])), list(payload.get("ground_truths", []))


def main() -> None:
    """CLI entry point for tracking metric evaluation."""
    parser = ArgumentParser(description="Evaluate tracking metrics from a JSON payload.")
    parser.add_argument("input_path", help="Path to JSON containing predictions and ground_truths lists.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for frame matching.")
    args = parser.parse_args()

    predictions, ground_truths = _load_tracking_payload(args.input_path)
    summary = evaluate_tracking_metrics(predictions, ground_truths, iou_threshold=args.iou_threshold)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
