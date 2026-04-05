"""Tests for Phase 11 evaluation helpers."""

from __future__ import annotations

from eval.cypher_accuracy import evaluate_cypher_generation
from eval.detection_metrics import bbox_iou, evaluate_detection_metrics
from eval.event_precision import calibrate_event_thresholds, evaluate_event_predictions
from eval.tracking_metrics import evaluate_tracking_metrics


def test_detection_metrics_computes_map_values() -> None:
    """Detection evaluation should compute mAP-style summary metrics."""
    predictions = [
        {"image_id": "img-1", "class_id": 0, "bbox": [0.0, 0.0, 10.0, 10.0], "confidence": 0.95},
        {"image_id": "img-1", "class_id": 0, "bbox": [50.0, 50.0, 60.0, 60.0], "confidence": 0.20},
    ]
    ground_truths = [
        {"image_id": "img-1", "class_id": 0, "bbox": [0.0, 0.0, 10.0, 10.0]},
    ]

    metrics = evaluate_detection_metrics(predictions, ground_truths, iou_thresholds=[0.5, 0.75])

    assert bbox_iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) == 1.0
    assert metrics["map_50"] == 1.0
    assert metrics["map_50_95"] == 1.0
    assert metrics["per_threshold_ap"] == {"0.50": 1.0, "0.75": 1.0}


def test_tracking_metrics_computes_mota_motp_and_id_switches() -> None:
    """Tracking evaluation should capture matches, misses, and ID switches."""
    ground_truths = [
        {"frame_id": 1, "track_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"frame_id": 2, "track_id": 1, "bbox": [1.0, 1.0, 11.0, 11.0]},
    ]
    predictions = [
        {"frame_id": 1, "track_id": 101, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"frame_id": 2, "track_id": 202, "bbox": [1.0, 1.0, 11.0, 11.0]},
    ]

    metrics = evaluate_tracking_metrics(predictions, ground_truths, iou_threshold=0.5)

    assert metrics["mota"] == 0.5
    assert metrics["motp"] == 1.0
    assert metrics["id_switches"] == 1
    assert metrics["id_switch_rate"] == 1.0


def test_event_precision_evaluates_precision_recall_and_calibration() -> None:
    """Event evaluation should score matches and select the best threshold set."""
    predictions = [
        {
            "event_type": "NEAR_MISS",
            "frame_id": 10,
            "sequence_id": "seq-1",
            "primary_track_id": 1,
            "secondary_track_id": 2,
        },
        {
            "event_type": "LOITER",
            "frame_id": 30,
            "sequence_id": "seq-1",
            "primary_track_id": 3,
            "secondary_track_id": None,
        },
    ]
    ground_truths = [
        {
            "event_type": "NEAR_MISS",
            "frame_id": 11,
            "sequence_id": "seq-1",
            "primary_track_id": 1,
            "secondary_track_id": 2,
        }
    ]

    metrics = evaluate_event_predictions(predictions, ground_truths, frame_tolerance=1)
    best = calibrate_event_thresholds(
        candidate_thresholds=[{"distance_threshold": 0.03}, {"distance_threshold": 0.05}],
        evaluator=lambda threshold: (
            {"precision": 0.5, "recall": 0.5, "f1": 0.5}
            if threshold["distance_threshold"] == 0.03
            else {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        ),
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert round(metrics["f1"], 4) == 0.6667
    assert best == {
        "thresholds": {"distance_threshold": 0.05},
        "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    }


def test_cypher_accuracy_evaluates_syntax_semantics_and_retry_rate() -> None:
    """Cypher evaluation should summarize syntax validity, correctness, and retries."""
    cases = [
        {
            "first_attempt_cypher": "MATCH (o:Object) RETURN o.track_id AS track_id",
            "final_cypher": "MATCH (o:Object) RETURN o.track_id AS track_id",
            "expected_cypher": "MATCH (o:Object) RETURN o.track_id AS track_id",
        },
        {
            "first_attempt_cypher": "CREATE (n) RETURN n",
            "final_cypher": "MATCH (e:Event) RETURN e.frame_id AS frame_id",
            "expected_results": [{"frame_id": 12}],
            "actual_results": [{"frame_id": 12}],
        },
    ]

    metrics = evaluate_cypher_generation(cases)

    assert metrics["syntax_validity_first_attempt"] == 0.5
    assert metrics["final_syntax_validity"] == 1.0
    assert metrics["semantic_correctness"] == 1.0
    assert metrics["retry_rate"] == 0.5
    assert metrics["case_count"] == 2
