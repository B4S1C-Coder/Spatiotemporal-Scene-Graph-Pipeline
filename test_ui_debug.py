import sys
from configs.loader import load_yaml_config
from agents.llm_agent import LLMQueryAgent
from graph.neo4j_client import Neo4jClient
from ui.app import run_query, build_query_visualization_payload

client = Neo4jClient()
agent = LLMQueryAgent(client)

query_result = run_query(agent, "Which vehicles were stationary for more than 60 frames?", "uav0000086_00000_v")
print("Query Results:")
print(query_result.get("results"))

payload = build_query_visualization_payload(
    query_result=query_result,
    sequence_id="uav0000086_00000_v",
    neo4j_client=client
)
print("Payload:")
print(payload)
