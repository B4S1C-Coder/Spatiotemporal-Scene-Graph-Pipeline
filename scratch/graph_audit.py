import sys
sys.path.append('.')
from graph.neo4j_client import Neo4jClient

neo = Neo4jClient()
queries = [
    ("Total Objects", "MATCH (o:Object {sequence_id:'convoy'}) RETURN count(o) AS cnt"),
    ("Total Frames", "MATCH (f:Frame {sequence_id:'convoy'}) RETURN count(f) AS cnt"),
    ("Total Events", "MATCH (e:Event {sequence_id:'convoy'}) RETURN count(e) AS cnt"),
    ("Total Zones", "MATCH (z:Zone {sequence_id:'convoy'}) RETURN count(z) AS cnt"),
    ("APPEARED_IN edges", "MATCH (:Object {sequence_id:'convoy'})-[r:APPEARED_IN]->() RETURN count(r) AS cnt"),
    ("IN_ZONE edges", "MATCH (:Object {sequence_id:'convoy'})-[r:IN_ZONE]->() RETURN count(r) AS cnt"),
    ("NEAR edges", "MATCH (:Object {sequence_id:'convoy'})-[r:NEAR]->() RETURN count(r) AS cnt"),
    ("COEXISTS_WITH edges", "MATCH (:Object {sequence_id:'convoy'})-[r:COEXISTS_WITH]->() RETURN count(r) AS cnt"),
    ("NEAR_MISS edges", "MATCH (:Object {sequence_id:'convoy'})-[r:NEAR_MISS]->() RETURN count(r) AS cnt"),
    ("CONVOY_WITH edges", "MATCH (:Object {sequence_id:'convoy'})-[r:CONVOY_WITH]->() RETURN count(r) AS cnt"),
    ("SAME_ENTITY_AS edges", "MATCH (:Object {sequence_id:'convoy'})-[r:SAME_ENTITY_AS]->() RETURN count(r) AS cnt"),
    ("INVOLVES edges", "MATCH (e:Event {sequence_id:'convoy'})-[r:INVOLVES]->() RETURN count(r) AS cnt"),
    ("All relationship types", "MATCH (:Object {sequence_id:'convoy'})-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC"),
    ("Object classes", "MATCH (o:Object {sequence_id:'convoy'}) RETURN o.class AS cls, count(o) AS cnt ORDER BY cnt DESC"),
    ("Event types", "MATCH (e:Event {sequence_id:'convoy'}) RETURN e.event_type AS evt, count(e) AS cnt ORDER BY cnt DESC"),
]

for label, q in queries:
    try:
        results = neo.execute_query(q)
        print(f"{label}: {results}")
    except Exception as e:
        print(f"{label}: ERROR - {e}")

neo.close()
