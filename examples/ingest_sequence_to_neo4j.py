"""
End-to-end ingestion example for writing a real sequence into Neo4j.

Run this from the repository root after starting Neo4j and downloading the
required dataset assets and weights.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json

from pipeline.runner import run_pipeline_cli


def main() -> None:
    """CLI entry point for the ingestion example."""
    parser = ArgumentParser(description="Example: ingest one VisDrone sequence into Neo4j.")
    parser.add_argument("--sequence", required=True, help="Sequence ID to ingest.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Optional frame skip override.")
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Run entity-resolution post-processing after ingestion.",
    )
    args = parser.parse_args()

    summary = run_pipeline_cli(
        [args.sequence],
        frame_skip=args.frame_skip,
        post_process=args.post_process,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
