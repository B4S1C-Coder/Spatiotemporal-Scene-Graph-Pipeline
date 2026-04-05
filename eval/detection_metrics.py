"""
Detection evaluation helpers and CLI.

This module computes lightweight mAP-style metrics from JSON-serializable
detection records so validation runs do not depend on external evaluator
frameworks.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for two `[x1, y1, x2, y2]` boxes."""
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def evaluate_detection_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """
    Evaluate mAP-like detection metrics for JSON detection records.

    Args:
        predictions: Detection records with `image_id`, `class_id`, `bbox`, and `confidence`.
        ground_truths: Ground-truth records with `image_id`, `class_id`, and `bbox`.
        iou_thresholds: IoU thresholds used for AP computation.

    Returns:
        Summary dictionary containing `map_50`, `map_50_95`, and per-threshold AP.
    """
    thresholds = iou_thresholds or [round(0.5 + 0.05 * index, 2) for index in range(10)]
    class_ids = sorted({int(item["class_id"]) for item in predictions + ground_truths})
    per_threshold_ap: dict[str, float] = {}
    for threshold in thresholds:
        class_aps = [
            _average_precision_for_class(predictions, ground_truths, class_id, threshold)
            for class_id in class_ids
        ]
        per_threshold_ap[f"{threshold:.2f}"] = sum(class_aps) / len(class_aps) if class_aps else 0.0

    map_50 = per_threshold_ap.get("0.50", 0.0)
    map_50_95 = sum(per_threshold_ap.values()) / len(per_threshold_ap) if per_threshold_ap else 0.0
    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "per_threshold_ap": per_threshold_ap,
        "target_map_50": 0.35,
        "target_map_50_95": 0.20,
    }


def _average_precision_for_class(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    class_id: int,
    iou_threshold: float,
) -> float:
    class_predictions = sorted(
        [item for item in predictions if int(item["class_id"]) == class_id],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    class_ground_truths = [item for item in ground_truths if int(item["class_id"]) == class_id]
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    matched_gt: dict[tuple[str, int], bool] = {}
    for ground_truth in class_ground_truths:
        image_key = str(ground_truth["image_id"])
        gt_by_image.setdefault(image_key, []).append(ground_truth)
    for image_key, gt_items in gt_by_image.items():
        for index in range(len(gt_items)):
            matched_gt[(image_key, index)] = False

    true_positive_count = 0
    false_positive_count = 0
    precision_sum = 0.0
    total_ground_truths = len(class_ground_truths)
    if total_ground_truths == 0:
        return 0.0

    for prediction in class_predictions:
        image_key = str(prediction["image_id"])
        gt_candidates = gt_by_image.get(image_key, [])
        best_match_index = None
        best_iou = 0.0
        for index, ground_truth in enumerate(gt_candidates):
            if matched_gt[(image_key, index)]:
                continue
            current_iou = bbox_iou(list(prediction["bbox"]), list(ground_truth["bbox"]))
            if current_iou > best_iou:
                best_iou = current_iou
                best_match_index = index

        if best_match_index is not None and best_iou >= iou_threshold:
            matched_gt[(image_key, best_match_index)] = True
            true_positive_count += 1
            precision_sum += true_positive_count / (true_positive_count + false_positive_count)
        else:
            false_positive_count += 1

    return precision_sum / total_ground_truths


def _load_detection_payload(input_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return list(payload.get("predictions", [])), list(payload.get("ground_truths", []))


def main() -> None:
    """CLI entry point for detection metric evaluation."""
    parser = ArgumentParser(description="Evaluate detection mAP-style metrics from a JSON payload.")
    parser.add_argument("input_path", help="Path to JSON containing predictions and ground_truths lists.")
    args = parser.parse_args()

    predictions, ground_truths = _load_detection_payload(args.input_path)
    summary = evaluate_detection_metrics(predictions, ground_truths)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
