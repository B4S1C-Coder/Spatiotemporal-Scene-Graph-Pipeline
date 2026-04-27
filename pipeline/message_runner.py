import argparse
import json
import logging
from pathlib import Path

from agents.message_agent import MessageAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest text messages into the Spatiotemporal Graph.")
    parser.add_argument("--messages", type=str, required=True, help="Path to JSON file containing messages.")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence ID to ground these messages to.")
    args = parser.parse_args()

    msg_path = Path(args.messages)
    if not msg_path.exists():
        logger.error(f"Message file not found: {msg_path}")
        return

    with open(msg_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    if not isinstance(messages, list):
        logger.error("JSON file must contain a list of message dictionaries.")
        return

    logger.info(f"Loaded {len(messages)} messages. Initializing MessageAgent...")
    agent = MessageAgent()
    
    logger.info(f"Ingesting messages into sequence '{args.sequence}'...")
    agent.ingest_messages(messages, args.sequence)
    
    # Close neo4j connection
    agent.neo4j_client.close()
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
