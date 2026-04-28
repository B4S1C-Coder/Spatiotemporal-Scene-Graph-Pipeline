import json
import logging
import re
from typing import Any, Dict, List

from agents.llm_agent import LLMQueryAgent, OpenAILLMClient, load_llm_config, _resolve_llm_base_url, _resolve_llm_api_key
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = """\
You are a master spatiotemporal reasoning agent. Your job is to solve complex queries by breaking them down into steps.
You have access to a Neo4j Knowledge Graph that contains drone surveillance data (Objects, Zones, Events, Text Messages).

You operate in a strict JSON state machine. At each step, you must output a single JSON object (and nothing else) with the following structure:
{
  "thought": "Your internal reasoning about what to do next based on the memory of past steps.",
  "action": "QUERY" | "INFER" | "FINAL_ANSWER",
  "action_input": "The input for your action."
}

Action Types:
1. "QUERY": Use this to query the Neo4j database.
   - action_input: A natural language description of what you want to search for. (e.g. "Find all trucks in the sequence"). A sub-agent will translate this into Cypher and return the results to your memory.
2. "INFER": Use this to reason about intermediate results without running a new query.
   - action_input: Your internal notes or deductions based on the data retrieved so far.
3. "FINAL_ANSWER": Use this when you have gathered enough information to answer the user's original goal.
   - action_input: Your final, comprehensive answer to the user.

RULES:
- DO NOT generate Cypher queries yourself. Always use the "QUERY" action with natural language.
- ALWAYS respond with a valid JSON object. Do not include markdown formatting or conversational text outside the JSON.

Example:
{"thought": "I need to find all trucks first.", "action": "QUERY", "action_input": "Find all trucks in the sequence"}
"""

class ReasoningAgent:
    def __init__(self, neo4j_client: Neo4jClient | None = None, llm_client: OpenAILLMClient | None = None):
        self.neo4j_client = neo4j_client or Neo4jClient()
        config = load_llm_config()
        self.llm_client = llm_client or OpenAILLMClient(
            base_url=_resolve_llm_base_url(config),
            api_key=_resolve_llm_api_key(config),
        )
        self.model = config["llm"]["model"]
        self.query_agent = LLMQueryAgent(neo4j_client=self.neo4j_client, llm_client=self.llm_client)

    def format_memory(self, memory: List[Dict[str, Any]]) -> str:
        """Format the memory of past steps for the LLM prompt."""
        if not memory:
            return "None yet."
        parts = []
        for i, step in enumerate(memory):
            action = step.get('action', '')
            line = f"[{i+1}] {action}: {step.get('action_input','')}"
            if action == 'QUERY':
                results = step.get('results', [])
                top = results[:5]
                line += f" -> {len(results)} rows: {json.dumps(top, separators=(',',':'))}"
            elif action == 'SUMMARIZE':
                # The summary replaces raw data in the model's memory
                line = f"[{i+1}] SUMMARY: {step.get('action_input','')}"
            parts.append(line)
        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any] | None:
        """Try multiple strategies to extract a JSON object from LLM output."""
        cleaned = text.strip()
        # Strip markdown fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Regex fallback: find the first { ... } block
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def decide_next_step(self, goal: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ask the LLM what to do next based on the goal and memory."""
        memory_str = self.format_memory(memory)
        user_prompt = f"Goal: {goal}\nHistory:\n{memory_str}\nNext step? JSON only."
        
        try:
            response = self.llm_client.generate(
                system_prompt=REASONING_SYSTEM_PROMPT.strip(),
                user_prompt=user_prompt,
                model=self.model
            )
            decision = self._extract_json(response)
            if decision and "action" in decision:
                return decision
            logger.warning("LLM returned unparseable response: %s", response[:200])
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return {
            "thought": "Failed to parse decision. Providing best answer with available data.",
            "action": "FINAL_ANSWER",
            "action_input": "I encountered an error while reasoning. Please try a simpler question."
        }

    @staticmethod
    def _digest_results(results: list[dict[str, Any]]) -> str:
        """Deterministically extract a compact structured digest from query results.

        This replaces LLM-based summarization — it's fast, token-safe, and
        preserves all key details (IDs, counts, classes, distances).
        """
        if not results:
            return "0 rows returned."

        total = len(results)
        # Collect all column names
        columns = sorted({k for row in results for k in row})

        # Extract key entity values
        key_extractors = {
            "track_ids": ("track_id", "o.track_id", "tid", "primary_track_id"),
            "frame_ids": ("frame_id", "f.frame_id", "fid"),
            "classes": ("class", "o.class", "cls", "class_name"),
            "zones": ("zone_id", "z.zone_id", "zone"),
            "event_types": ("event_type", "e.event_type", "evt"),
        }
        extracted: dict[str, set] = {k: set() for k in key_extractors}
        for row in results:
            for label, aliases in key_extractors.items():
                for alias in aliases:
                    if alias in row and row[alias] is not None:
                        extracted[label].add(str(row[alias]))

        # Compute numeric stats
        numeric_stats: dict[str, dict[str, float]] = {}
        for col in columns:
            vals = []
            for row in results:
                v = row.get(col)
                if isinstance(v, (int, float)):
                    vals.append(v)
            if vals:
                numeric_stats[col] = {
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "avg": round(sum(vals) / len(vals), 4),
                }

        # Build compact digest string
        lines = [f"Total rows: {total}", f"Columns: {', '.join(columns)}"]
        for label, values in extracted.items():
            if values:
                display = sorted(values)
                if len(display) > 10:
                    display = display[:10] + [f"...+{len(display)-10} more"]
                lines.append(f"{label}: {', '.join(display)}")
        for col, stats in numeric_stats.items():
            lines.append(f"{col}: min={stats['min']}, max={stats['max']}, avg={stats['avg']}")

        # Include first 3 sample rows (compact)
        lines.append(f"Sample rows ({min(3, total)}/{total}):")
        for row in results[:3]:
            lines.append(f"  {json.dumps(row, separators=(',',':'))}")

        return "\n".join(lines)

    def execute_step(self, decision: Dict[str, Any], sequence_id: str | None = None) -> Dict[str, Any]:
        """Execute the chosen action and return the result state to be appended to memory."""
        action = decision.get("action")
        action_input = decision.get("action_input", "")
        
        result_state = {
            "thought": decision.get("thought", ""),
            "action": action,
            "action_input": action_input
        }
        
        if action == "QUERY":
            logger.info(f"Executing QUERY: {action_input}")
            try:
                cypher = self.query_agent.generate_cypher(
                    natural_language_query=action_input,
                    sequence_id=sequence_id,
                )
                is_valid, err = self.query_agent.validate_cypher(cypher)
                if is_valid:
                    params = {"seq_id": sequence_id} if sequence_id else {}
                    results = self.query_agent.execute_cypher(cypher, parameters=params)
                    result_state["cypher"] = cypher
                    result_state["results"] = results
                else:
                    result_state["cypher"] = cypher
                    result_state["results"] = []
                    logger.warning(f"Invalid Cypher: {err}")
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                result_state["cypher"] = None
                result_state["results"] = []
            
        elif action == "INFER":
            logger.info(f"Executing INFER: {action_input}")
            result_state["inference"] = "Noted."
            
        elif action == "FINAL_ANSWER":
            logger.info(f"Executing FINAL_ANSWER: {action_input}")
            
        else:
            logger.warning(f"Unknown action: {action}")
            result_state["action"] = "INFER"
            result_state["inference"] = f"Unknown action {action} attempted."
            
        return result_state

    def run(self, goal: str, sequence_id: str | None = None, max_steps: int = 12):
        """Run the reasoning loop as a generator, yielding each step's state."""
        memory = []
        for step_idx in range(max_steps):
            decision = self.decide_next_step(goal, memory)
            result_state = self.execute_step(decision, sequence_id)
            memory.append(result_state)
            yield result_state

            # After a QUERY with results, inject a deterministic SUMMARIZE step
            if result_state.get("action") == "QUERY" and result_state.get("results"):
                digest_text = self._digest_results(result_state["results"])
                summarize_state = {
                    "thought": f"Digesting {len(result_state['results'])} result rows.",
                    "action": "SUMMARIZE",
                    "action_input": digest_text,
                }
                memory.append(summarize_state)
                yield summarize_state

            if result_state.get("action") == "FINAL_ANSWER":
                break
        else:
            yield {
                "thought": "I have reached the maximum number of steps allowed.",
                "action": "FINAL_ANSWER",
                "action_input": "I was unable to fully complete the goal within the step limit. Please try breaking your query down."
            }

