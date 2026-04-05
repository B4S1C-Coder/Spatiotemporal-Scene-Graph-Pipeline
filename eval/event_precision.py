"""
Event precision/recall evaluation helpers and CLI.

This module scores predicted event records against manually labeled events and
provides a small threshold-calibration helper for rule-based event tuning.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Callable


def evaluate_event_predictions(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    frame_tolerance: int = 0,
) -> dict[str, Any]:
    """
    Evaluate precision, recall, and F1 for predicted event records.
    """
    matched_ground_truth_indices: set[int] = set()
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        matched_index = None
        for index, ground_truth in enumerate(ground_truths):
            if index in matched_ground_truth_indices:
                continue
            if _events_match(prediction, ground_truth, frame_tolerance=frame_tolerance):
                matched_index = index
                break
        if matched_index is None:
            false_positives += 1
            continue
        matched_ground_truth_indices.add(matched_index)
        true_positives += 1

    false_negatives = len(ground_truths) - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calibrate_event_thresholds(
    candidate_thresholds: list[dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """
    Select the threshold configuration with the best F1 score.
    """
    if not candidate_thresholds:
        raise ValueError("candidate_thresholds must not be empty")
    scored_candidates = []
    for thresholds in candidate_thresholds:
        metrics = evaluator(thresholds)
        scored_candidates.append({"thresholds": thresholds, "metrics": metrics})
    scored_candidates.sort(
        key=lambda item: (
            float(item["metrics"].get("f1", 0.0)),
            float(item["metrics"].get("precision", 0.0)),
            float(item["metrics"].get("recall", 0.0)),
        ),
        reverse=True,
    )
    return scored_candidates[0]


def _events_match(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    frame_tolerance: int,
) -> bool:
    if str(prediction["event_type"]) != str(ground_truth["event_type"]):
        return False
    if str(prediction["sequence_id"]) != str(ground_truth["sequence_id"]):
        return False
    if abs(int(prediction["frame_id"]) - int(ground_truth["frame_id"])) > frame_tolerance:
        return False
    if int(prediction["primary_track_id"]) != int(ground_truth["primary_track_id"]):
        return False
    return prediction.get("secondary_track_id") == ground_truth.get("secondary_track_id")


def _load_event_payload(input_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return list(payload.get("predictions", [])), list(payload.get("ground_truths", []))


def main() -> None:
    """CLI entry point for event precision and recall evaluation."""
    parser = ArgumentParser(description="Evaluate event precision/recall from a JSON payload.")
    parser.add_argument("input_path", help="Path to JSON containing predictions and ground_truths lists.")
    parser.add_argument("--frame-tolerance", type=int, default=0, help="Allowed frame difference when matching events.")
    args = parser.parse_args()

    predictions, ground_truths = _load_event_payload(args.input_path)
    summary = evaluate_event_predictions(predictions, ground_truths, frame_tolerance=args.frame_tolerance)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
