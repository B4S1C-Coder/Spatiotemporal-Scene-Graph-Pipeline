import sys
sys.path.append('.')
import json
from agents.llm_agent import run_query_cli

result = run_query_cli("What text messages mention a truck?", sequence_id="convoy")
print(json.dumps(result, indent=2))
