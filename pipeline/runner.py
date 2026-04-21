"""
Pipeline runner for sequence-level frame processing.

This module orchestrates the dataset loader and the project agents so one frame
flows through detection, tracking, motion enrichment, event detection, and
buffered graph writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
import gc
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

from agents.detection_agent import DetectionAgent
from agents.event_agent import EventAgent
from agents.graph_agent import GraphAgent
from agents.motion_agent import MotionAgent
from agents.tracking_agent import TrackingAgent
from configs.loader import DETECTION_CONFIG_PATH, GRAPH_CONFIG_PATH, load_yaml_config
from graph.neo4j_client import Neo4jClient
from pipeline.batch_writer import BatchWriter
from pipeline.post_processor import PostProcessor
from pipeline.sequence_loader import SequenceLoader


def current_process_rss_mb() -> float:
    """
    Return the current process resident set size in MiB when available.
    """
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0

    import resource

    max_rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss_kb / 1024.0


class PipelineRunner:
    """Orchestrate sequence iteration and agent execution frame by frame."""

    def __init__(
        self,
        config_path: str | Path = DETECTION_CONFIG_PATH,
        config: dict[str, Any] | None = None,
        loader_factory: Callable[..., Any] = SequenceLoader,
        detection_agent: DetectionAgent | None = None,
        tracking_agent: TrackingAgent | None = None,
        motion_agent: MotionAgent | None = None,
        event_agent: EventAgent | None = None,
        graph_agent: GraphAgent | None = None,
        batch_writer: BatchWriter | None = None,
        neo4j_client: Neo4jClient | None = None,
        post_processor: PostProcessor | None = None,
        current_rss_mb_fn: Callable[[], float] | None = None,
        gc_collect_fn: Callable[[], int] = gc.collect,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the pipeline runner.

        Args:
            config_path: YAML config file location.
            config: Optional runtime override mapping.
            loader_factory: Injectable sequence-loader constructor for testing.
            detection_agent: Optional prebuilt detection agent.
            tracking_agent: Optional prebuilt tracking agent.
            motion_agent: Optional prebuilt motion agent.
            event_agent: Optional prebuilt event agent.
            graph_agent: Optional prebuilt graph agent.
            batch_writer: Optional prebuilt batch writer.
            neo4j_client: Optional prebuilt Neo4j client for default graph-agent setup.
            post_processor: Optional sequence-completion post-processor.
        """
        self.config_path = Path(config_path)
        self.config = load_yaml_config(self.config_path, overrides=config)
        self.loader_factory = loader_factory

        self.detection_agent = detection_agent or DetectionAgent(
            config_path=self.config_path,
            config=self.config,
        )
        self.tracking_agent = tracking_agent or TrackingAgent()
        self.motion_agent = motion_agent or MotionAgent()
        self.event_agent = event_agent or EventAgent()
        self.neo4j_client = neo4j_client
        self.graph_agent = graph_agent or self._build_graph_agent()
        self.batch_writer = batch_writer or BatchWriter(self.graph_agent)
        self.post_processor = post_processor or PostProcessor(self.neo4j_client)
        self.last_postprocess_summary: dict[str, Any] | None = None
        self.current_rss_mb_fn = current_rss_mb_fn or current_process_rss_mb
        self.gc_collect_fn = gc_collect_fn
        self.logger = logger or logging.getLogger(__name__)
        runtime_config = self.config.get("runtime", {})
        self.max_process_rss_mb = float(runtime_config.get("max_process_rss_mb", 3072))
        self.progress_log_every_frames = int(runtime_config.get("progress_log_every_frames", 50))
        self.gc_every_frames = int(runtime_config.get("gc_every_frames", 25))
        self.last_sequence_summary: dict[str, Any] | None = None

    def run_sequence(
        self,
        sequence_id: str,
        frame_skip: int | None = None,
        post_process: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Process all frames in one sequence.

        Args:
            sequence_id: Sequence identifier to process.
            frame_skip: Optional override for frame skipping.
            post_process: Whether to run sequence-final post-processing.

        Returns:
            List of per-frame pipeline outputs.
        """
        processed_packets, summary = self._run_sequence_core(
            sequence_id=sequence_id,
            frame_skip=frame_skip,
            post_process=post_process,
            retain_outputs=True,
        )
        self.last_sequence_summary = summary
        return processed_packets

    def run_sequence_summary(
        self,
        sequence_id: str,
        frame_skip: int | None = None,
        post_process: bool = False,
    ) -> dict[str, Any]:
        """
        Process one sequence without retaining per-frame outputs in memory.

        This path is intended for CLI and long-running ingestion where keeping
        every frame packet and raw detector output would cause memory growth.
        """
        _processed_packets, summary = self._run_sequence_core(
            sequence_id=sequence_id,
            frame_skip=frame_skip,
            post_process=post_process,
            retain_outputs=False,
        )
        self.last_sequence_summary = summary
        return summary

    def run_sequences(
        self,
        sequence_ids: list[str],
        frame_skip: int | None = None,
        post_process: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Process multiple sequences in order.

        Args:
            sequence_ids: Sequence identifiers to process.
            frame_skip: Optional frame skip override for all sequences.
            post_process: Whether to run sequence-final post-processing per sequence.

        Returns:
            Mapping from sequence ID to processed frame packets.
        """
        return {
            sequence_id: self.run_sequence(sequence_id, frame_skip=frame_skip, post_process=post_process)
            for sequence_id in sequence_ids
        }

    def run_sequences_summary(
        self,
        sequence_ids: list[str],
        frame_skip: int | None = None,
        post_process: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Process multiple sequences without retaining per-frame outputs.
        """
        return {
            sequence_id: self.run_sequence_summary(
                sequence_id,
                frame_skip=frame_skip,
                post_process=post_process,
            )
            for sequence_id in sequence_ids
        }

    def process_frame(self, frame_packet: dict[str, Any]) -> dict[str, Any]:
        """
        Process one frame packet through all pipeline agents.

        Args:
            frame_packet: Sequence-loader frame packet.

        Returns:
            Dictionary containing the input packet plus intermediate outputs.
        """
        raw_results = self.detection_agent.infer_frame(frame_packet)
        detections = self.detection_agent.format_detections(raw_results, frame_packet)
        tracked_detections = self.tracking_agent.track_detections(detections)
        active_tracks = [track for track in tracked_detections if not bool(track.get("is_lost", False))]
        enriched_tracks = self.motion_agent.enrich_tracks(
            active_tracks,
            frame_width=int(frame_packet["orig_width"]),
            frame_height=int(frame_packet["orig_height"]),
        )
        active_track_map = {int(track["track_id"]): track for track in enriched_tracks}
        events = self.event_agent.process_tracks(
            tracks=active_track_map,
            frame_id=int(frame_packet["frame_id"]),
            sequence_id=str(frame_packet["sequence_id"]),
        )
        self.batch_writer.add_frame_data(
            object_states=enriched_tracks,
            sequence_id=str(frame_packet["sequence_id"]),
            frame_id=int(frame_packet["frame_id"]),
            events=events,
            scene_payload=frame_packet.get("scene_payload"),
            zone_stats=self.event_agent.get_zone_stats_snapshot(),
        )
        return {
            "frame_packet": frame_packet,
            "raw_results": raw_results,
            "detections": detections,
            "tracked_detections": tracked_detections,
            "active_tracked_detections": active_tracks,
            "enriched_tracks": enriched_tracks,
            "events": events,
        }

    def _run_sequence_core(
        self,
        *,
        sequence_id: str,
        frame_skip: int | None,
        post_process: bool,
        retain_outputs: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        loader = self.loader_factory(sequence_id=sequence_id, config=self.config)
        processed_packets: list[dict[str, Any]] = []
        frame_count = 0
        start_time = time.monotonic()
        peak_rss_mb = self.current_rss_mb_fn()

        self.logger.info(
            "Starting sequence %s | frame_skip=%s | retain_outputs=%s | rss_cap_mb=%.1f",
            sequence_id,
            frame_skip if frame_skip is not None else "config",
            retain_outputs,
            self.max_process_rss_mb,
        )
        try:
            for frame_packet in loader.iter_frames(frame_skip=frame_skip):
                frame_result = self.process_frame(frame_packet)
                frame_count += 1
                if retain_outputs:
                    processed_packets.append(frame_result)

                peak_rss_mb = max(peak_rss_mb, self._maybe_log_and_enforce_memory(sequence_id, frame_count))

                del frame_result
                del frame_packet
        finally:
            self.batch_writer.flush()
            self.gc_collect_fn()

        self.last_postprocess_summary = None
        if post_process:
            self.logger.info("Starting post-processing for sequence %s", sequence_id)
            self.last_postprocess_summary = self.post_processor.run_sequence(sequence_id)

        duration_sec = time.monotonic() - start_time
        summary = {
            "sequence_id": sequence_id,
            "frame_count": frame_count,
            "post_process_enabled": post_process,
            "post_process_summary": self.last_postprocess_summary,
            "duration_sec": round(duration_sec, 3),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "retained_frame_outputs": retain_outputs,
        }
        self.logger.info(
            "Finished sequence %s | frames=%d | duration_sec=%.2f | peak_rss_mb=%.1f",
            sequence_id,
            frame_count,
            duration_sec,
            peak_rss_mb,
        )
        return processed_packets, summary

    def _maybe_log_and_enforce_memory(self, sequence_id: str, frame_count: int) -> float:
        current_rss_mb = self.current_rss_mb_fn()

        should_log_progress = self.progress_log_every_frames > 0 and (
            frame_count == 1 or frame_count % self.progress_log_every_frames == 0
        )
        if should_log_progress:
            self.logger.info(
                "Sequence %s progress | frame=%d | rss_mb=%.1f",
                sequence_id,
                frame_count,
                current_rss_mb,
            )

        should_collect = self.gc_every_frames > 0 and frame_count % self.gc_every_frames == 0
        if should_collect:
            self.gc_collect_fn()
            current_rss_mb = self.current_rss_mb_fn()

        if current_rss_mb <= self.max_process_rss_mb:
            return current_rss_mb

        self.logger.warning(
            "Sequence %s exceeded memory threshold | frame=%d | rss_mb=%.1f | cap_mb=%.1f | forcing flush and GC",
            sequence_id,
            frame_count,
            current_rss_mb,
            self.max_process_rss_mb,
        )
        self.batch_writer.flush()
        self.gc_collect_fn()
        current_rss_mb = self.current_rss_mb_fn()
        if current_rss_mb > self.max_process_rss_mb:
            raise MemoryError(
                f"Process RSS {current_rss_mb:.1f} MiB exceeded configured cap "
                f"{self.max_process_rss_mb:.1f} MiB while processing sequence {sequence_id} "
                f"at frame {frame_count}."
            )
        return current_rss_mb

    def _build_graph_agent(self) -> GraphAgent:
        config_path = GRAPH_CONFIG_PATH
        if self.neo4j_client is None:
            self.neo4j_client = Neo4jClient(config_path=config_path)
        return GraphAgent(neo4j_client=self.neo4j_client, config_path=config_path)

    def close(self) -> None:
        """Close shared resources owned by the runner."""
        if self.neo4j_client is not None:
            self.neo4j_client.close()


def run_pipeline_cli(
    sequence_ids: list[str],
    *,
    frame_skip: int | None = None,
    post_process: bool = False,
    runner: PipelineRunner | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute one or more sequences and return a compact CLI summary.
    """
    active_runner = runner or PipelineRunner(config=config_overrides)
    try:
        results = active_runner.run_sequences_summary(
            sequence_ids,
            frame_skip=frame_skip,
            post_process=post_process,
        )
        return {
            "sequence_ids": sequence_ids,
            "frame_counts": {
                sequence_id: int(sequence_summary["frame_count"])
                for sequence_id, sequence_summary in results.items()
            },
            "sequence_summaries": results,
            "post_process_enabled": post_process,
            "last_postprocess_summary": active_runner.last_postprocess_summary,
        }
    finally:
        if runner is None:
            active_runner.close()


def main() -> None:
    """CLI entry point for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = ArgumentParser(description="Run the spatiotemporal scene-graph pipeline.")
    parser.add_argument("--sequence", action="append", dest="sequences", help="Sequence ID to ingest. Repeatable.")
    parser.add_argument("--manifest", help="JSON manifest containing sequence_ids.", default=None)
    parser.add_argument("--frame-skip", type=int, default=None, help="Optional frame skip override.")
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=None,
        help="Abort safely if process RSS exceeds this many MiB. Default comes from config/runtime.",
    )
    parser.add_argument(
        "--progress-every-frames",
        type=int,
        default=None,
        help="Log progress every N processed frames. Default comes from config/runtime.",
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Run entity-resolution post-processing after each sequence.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args()

    sequence_ids = list(args.sequences or [])
    if args.manifest:
        sequence_ids.extend(SequenceLoader.list_available(args.manifest))
    if not sequence_ids:
        raise SystemExit("Provide at least one --sequence or a --manifest.")

    runtime_overrides: dict[str, Any] = {}
    if args.max_rss_mb is not None:
        runtime_overrides["max_process_rss_mb"] = float(args.max_rss_mb)
    if args.progress_every_frames is not None:
        runtime_overrides["progress_log_every_frames"] = int(args.progress_every_frames)

    summary = run_pipeline_cli(
        sequence_ids,
        frame_skip=args.frame_skip,
        post_process=args.post_process,
        config_overrides={"runtime": runtime_overrides} if runtime_overrides else None,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("Pipeline execution complete.")
    for sequence_id, frame_count in summary["frame_counts"].items():
        print(f"- {sequence_id}: {frame_count} frames processed")
    if summary["post_process_enabled"]:
        print(f"Post-processing enabled. Last summary: {summary['last_postprocess_summary']}")


if __name__ == "__main__":
    main()
