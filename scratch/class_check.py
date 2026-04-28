import sys
sys.path.append('.')
from graph.neo4j_client import Neo4jClient

neo = Neo4jClient()

# Check what classes exist in convoy
results = neo.execute_query("MATCH (o:Object {sequence_id:'convoy'}) RETURN DISTINCT o.class AS cls ORDER BY cls")
print("Classes in 'convoy':", [r['cls'] for r in results])

# Check PEDESTRIAN_CLASSES and VEHICLE_CLASSES from event_agent
from agents.event_agent import PEDESTRIAN_CLASSES, VEHICLE_CLASSES
print("Expected pedestrian classes:", PEDESTRIAN_CLASSES)
print("Expected vehicle classes:", VEHICLE_CLASSES)

# Check if "person" is in the pedestrian set
print("'person' in PEDESTRIAN_CLASSES:", 'person' in PEDESTRIAN_CLASSES)
print("'umbrella' in any set:", 'umbrella' in PEDESTRIAN_CLASSES or 'umbrella' in VEHICLE_CLASSES)

neo.close()
