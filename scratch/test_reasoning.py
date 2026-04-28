import sys
sys.path.append('.')
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from agents.reasoning_agent import ReasoningAgent

agent = ReasoningAgent()
goal = "How many trucks are there?"
sequence_id = "convoy"

print(f"\n=== GOAL: {goal} ===\n")
for step in agent.run(goal, sequence_id=sequence_id):
    print(f"\n--- STEP: {step.get('action')} ---")
    print(f"Thought: {step.get('thought')}")
    print(f"Action Input: {step.get('action_input')}")
    if step.get('action') == 'QUERY':
        print(f"Cypher: {step.get('cypher')}")
        results = step.get('results', [])
        print(f"Results ({len(results)} rows): {results[:3]}")
    elif step.get('action') == 'FINAL_ANSWER':
        print("\n*** DONE ***")
        break
