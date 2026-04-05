# Evaluation Tools Phase 11

Phase 11 adds lightweight evaluation scripts for the implemented pipeline:

- [eval/detection_metrics.py](/home/saksham/codebase/deep-learning-project/eval/detection_metrics.py)
- [eval/tracking_metrics.py](/home/saksham/codebase/deep-learning-project/eval/tracking_metrics.py)
- [eval/event_precision.py](/home/saksham/codebase/deep-learning-project/eval/event_precision.py)
- [eval/cypher_accuracy.py](/home/saksham/codebase/deep-learning-project/eval/cypher_accuracy.py)

Each script exposes reusable Python helpers plus a JSON-driven CLI so you can
evaluate saved outputs without extra framework dependencies.

Expected input shapes:

- detection: `{"predictions": [...], "ground_truths": [...]}`
- tracking: `{"predictions": [...], "ground_truths": [...]}`
- events: `{"predictions": [...], "ground_truths": [...]}`
- cypher: `{"cases": [...]}`

The helpers report the project-facing summary metrics:

- detection: `map_50`, `map_50_95`
- tracking: `mota`, `motp`, `id_switches`
- events: `precision`, `recall`, `f1`
- Cypher: first-attempt syntax validity, semantic correctness, retry rate
