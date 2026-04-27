import sys
sys.path.append('.')
import json
from agents.llm_agent import run_query_cli

result = run_query_cli("Show me the text of the message sent by Operator Bravo.", sequence_id="convoy")
print(json.dumps(result, indent=2))
