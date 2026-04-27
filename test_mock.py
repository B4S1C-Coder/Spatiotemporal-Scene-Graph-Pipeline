from graph.neo4j_client import Neo4jClient
from ui.app import build_query_visualization_payload

client = Neo4jClient()

# Mock result simulating what an LLM would return
query_result = {
    "results": [
        {"o.track_id": 1, "o.class": "car"}
    ]
}

payload = build_query_visualization_payload(
    query_result=query_result,
    sequence_id="uav0000086_00000_v",
    neo4j_client=client
)
print("Payload keys:", list(payload.keys()) if payload else None)
if payload:
    print("Frame IDs:", payload["frame_ids"])
