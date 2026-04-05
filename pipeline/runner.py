"""
Pipeline runner for sequence-level frame processing.

This module orchestrates the dataset loader and the project agents so one frame
flows through detection, tracking, motion enrichment, event detection, and
buffered graph writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.detection_agent import DetectionAgent
from agents.event_agent import EventAgent
from agents.graph_agent import GraphAgent
from agents.motion_agent import MotionAgent
from agents.tracking_agent import TrackingAgent
from configs.loader import DETECTION_CONFIG_PATH, GRAPH_CONFIG_PATH, load_yaml_config
from graph.neo4j_client import Neo4jClient
from pipeline.batch_writer import BatchWriter
from pipeline.sequence_loader import SequenceLoader


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

    def run_sequence(
        self,
        sequence_id: str,
        frame_skip: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process all frames in one sequence.

        Args:
            sequence_id: Sequence identifier to process.
            frame_skip: Optional override for frame skipping.

        Returns:
            List of per-frame pipeline outputs.
        """
        loader = self.loader_factory(sequence_id=sequence_id, config=self.config)
        processed_packets: list[dict[str, Any]] = []
        try:
            for frame_packet in loader.iter_frames(frame_skip=frame_skip):
                processed_packets.append(self.process_frame(frame_packet))
        finally:
            self.batch_writer.flush()
        return processed_packets

    def run_sequences(
        self,
        sequence_ids: list[str],
        frame_skip: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Process multiple sequences in order.

        Args:
            sequence_ids: Sequence identifiers to process.
            frame_skip: Optional frame skip override for all sequences.

        Returns:
            Mapping from sequence ID to processed frame packets.
        """
        return {
            sequence_id: self.run_sequence(sequence_id, frame_skip=frame_skip)
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

    def _build_graph_agent(self) -> GraphAgent:
        config_path = GRAPH_CONFIG_PATH
        if self.neo4j_client is None:
            self.neo4j_client = Neo4jClient(config_path=config_path)
        return GraphAgent(neo4j_client=self.neo4j_client, config_path=config_path)
