import sys
sys.path.append('.')
from graph.neo4j_client import Neo4jClient
neo = Neo4jClient()
results = neo.execute_query("MATCH (m:Message {sequence_id: 'convoy'}) RETURN m.sender, m.text")
print("Neo4j Results:")
for r in results:
    print(r)
