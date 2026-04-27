import json
import logging
from typing import Any
from pathlib import Path

from agents.llm_agent import OpenAILLMClient, load_llm_config, _resolve_llm_base_url, _resolve_llm_api_key
from graph.neo4j_client import Neo4jClient
from configs.loader import LLM_CONFIG_PATH

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
You are an entity extraction system for a surveillance pipeline.
Your task is to read the following text message and extract ANY physical object classes that might be tracked by a drone.
Valid classes include: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

Return ONLY a valid JSON list of strings containing the exact class names mentioned or implied.
If a "person" or "man" is mentioned, output "pedestrian" or "people".
If no valid objects are mentioned, return [].
Do not include markdown formatting or explanations, just the JSON array.
"""

class MessageAgent:
    def __init__(self, neo4j_client: Neo4jClient | None = None, llm_client: OpenAILLMClient | None = None):
        self.neo4j_client = neo4j_client or Neo4jClient()
        
        config = load_llm_config()
        self.llm_client = llm_client or OpenAILLMClient(
            base_url=_resolve_llm_base_url(config),
            api_key=_resolve_llm_api_key(config),
        )
        self.model = config["llm"]["model"]

    def extract_entities(self, text: str) -> list[str]:
        """Use the LLM to extract object classes from the message."""
        try:
            response = self.llm_client.generate(
                system_prompt=EXTRACTION_PROMPT.strip(),
                user_prompt=f"Message: {text}",
                model=self.model
            )
            # Basic cleanup in case the LLM wrapped it in markdown
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            entities = json.loads(response.strip())
            if isinstance(entities, list):
                return [str(e).lower() for e in entities]
        except Exception as e:
            logger.error(f"Failed to extract entities from text '{text}': {e}")
        return []

    def ingest_messages(self, messages: list[dict[str, Any]], sequence_id: str) -> None:
        """
        Process a list of messages and write them to Neo4j.
        Messages should have: message_id, sender, text, timestamp
        """
        for msg in messages:
            msg_id = msg.get("message_id")
            text = msg.get("text", "")
            
            # 1. Extract entities
            mentioned_classes = self.extract_entities(text)
            logger.info(f"Message {msg_id} mentions: {mentioned_classes}")
            
            # 2. Write to Graph
            query = """
            MERGE (m:Message {message_id: $msg_id})
            SET m.sequence_id = $sequence_id,
                m.sender = $sender,
                m.text = $text,
                m.timestamp = $timestamp
            
            WITH m
            UNWIND $mentioned_classes AS cls
            MATCH (o:Object {sequence_id: $sequence_id})
            WHERE toLower(o.class) = cls
            MERGE (m)-[:MENTIONS]->(o)
            """
            
            parameters = {
                "msg_id": msg_id,
                "sequence_id": sequence_id,
                "sender": msg.get("sender", "Unknown"),
                "text": text,
                "timestamp": msg.get("timestamp", 0),
                "mentioned_classes": mentioned_classes
            }
            
            self.neo4j_client.execute_query(query, parameters)
        
        logger.info(f"Successfully ingested {len(messages)} messages for sequence {sequence_id}.")
