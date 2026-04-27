import sys
sys.path.append('.')

from graph.neo4j_client import Neo4jClient
from ui.app import build_query_visualization_payload

neo = Neo4jClient()

payload = build_query_visualization_payload(
    query_result={"results": []}, 
    sequence_id="convoy", 
    neo4j_client=neo
)
print("Payload keys:", payload.keys() if payload else None)
neo.close()
