"""
Batch-writer integration helpers for the pipeline runner.

This module exposes a small adapter around the graph agent so the runner can
append frame data and force a final flush at sequence boundaries.
"""

from __future__ import annotations

from typing import Any

from agents.graph_agent import GraphAgent


class BatchWriter:
    """Thin adapter that forwards frame data into the graph agent batch."""

    def __init__(self, graph_agent: GraphAgent) -> None:
        self.graph_agent = graph_agent

    def add_frame_data(
        self,
        object_states: list[dict[str, Any]],
        sequence_id: str,
        frame_id: int,
        events: list[dict[str, Any]] | None = None,
        scene_payload: dict[str, Any] | None = None,
    ) -> bool:
        """
        Buffer one frame worth of graph data.

        Args:
            object_states: Motion-enriched object states for the frame.
            sequence_id: Sequence identifier.
            frame_id: Current frame ID.
            events: Optional event records for the frame.
            scene_payload: Optional scene metadata for first-frame setup.

        Returns:
            Whether the underlying graph agent flushed during this call.
        """
        return self.graph_agent.add_frame_data(
            object_states=object_states,
            sequence_id=sequence_id,
            frame_id=frame_id,
            events=events,
            scene_payload=scene_payload,
        )

    def flush(self) -> None:
        """Flush any buffered graph writes."""
        self.graph_agent.flush()
