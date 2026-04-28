import streamlit as st
import sys
import json
from pathlib import Path

# Add project root to sys.path so imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.reasoning_agent import ReasoningAgent
from graph.neo4j_client import Neo4jClient
from ui.app import (
    build_query_visualization_payload, 
    build_result_table_payload
)

st.set_page_config(page_title="Palantir AIP: Reasoning Agent", layout="wide")

@st.cache_resource
def get_reasoning_agent():
    return ReasoningAgent(neo4j_client=Neo4jClient())

def main():
    st.title("Palantir AIP: Guided Reasoning Harness")
    st.markdown("Interact with the Qwen 3.5-4B reasoning agent. It will dynamically decide which queries to run.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Sidebar config
    st.sidebar.header("Configuration")
    sequence_id = st.sidebar.text_input("Sequence ID (optional)", value="convoy")
    if not sequence_id:
        sequence_id = None
        
    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                for step in msg["steps"]:
                    st.markdown(f"**🧠 Thought:** {step.get('thought', '')}")
                    action = step.get('action', '')
                    
                    if action == "QUERY":
                        st.markdown(f"**🔍 Action:** Executed Query Intent: `{step.get('action_input', '')}`")
                        with st.expander("View Query & Results"):
                            st.code(step.get("cypher", ""), language="cypher")
                            results = step.get("results", [])
                            if results:
                                st.dataframe(build_result_table_payload(results))
                            else:
                                st.write("No results.")
                        
                        # Show visualization if payload exists
                        if "vis_payload" in step and step["vis_payload"]:
                            vis = step["vis_payload"]
                            st.subheader("Query Context Video")
                            if vis.get("clip_video"):
                                st.video(vis["clip_video"], start_time=int(vis.get("clip_start_time", 0)))
                            else:
                                st.warning("Video clip not available for this query.")
                                
                    elif action == "INFER":
                        st.markdown(f"**🤔 Inference:** {step.get('action_input', '')}")
                    elif action == "SUMMARIZE":
                        st.info(f"**📋 Summarize:** {step.get('action_input', '')}")
                    elif action == "FINAL_ANSWER":
                        st.success(f"**✅ Final Answer:** {step.get('action_input', '')}")

    # Chat input
    if prompt := st.chat_input("Ask a complex spatiotemporal question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        agent = get_reasoning_agent()
        
        with st.chat_message("assistant"):
            st_placeholder = st.empty()
            
            steps_recorded = []
            with st.spinner("Agent is reasoning..."):
                for step in agent.run(prompt, sequence_id=sequence_id):
                    # If it's a QUERY, we attempt to build visualization
                    if step.get("action") == "QUERY" and step.get("results"):
                        try:
                            vis_payload = build_query_visualization_payload(
                                query_result={"results": step["results"]},
                                sequence_id=sequence_id,
                                neo4j_client=agent.neo4j_client
                            )
                            step["vis_payload"] = vis_payload
                        except Exception as e:
                            st.error(f"Vis error: {e}")
                    
                    steps_recorded.append(step)
                    
                    # Update UI for the current step immediately
                    st.markdown("---")
                    st.markdown(f"**🧠 Thought:** {step.get('thought', '')}")
                    action = step.get('action', '')
                    
                    if action == "QUERY":
                        st.markdown(f"**🔍 Action:** Executed Query Intent: `{step.get('action_input', '')}`")
                        with st.expander("View Query & Results"):
                            st.code(step.get("cypher", ""), language="cypher")
                            results = step.get("results", [])
                            if results:
                                st.dataframe(build_result_table_payload(results))
                            else:
                                st.write("No results.")
                                
                        if "vis_payload" in step and step["vis_payload"]:
                            vis = step["vis_payload"]
                            st.subheader("Query Context Video")
                            if vis.get("clip_video"):
                                st.video(vis["clip_video"], start_time=int(vis.get("clip_start_time", 0)))
                                
                    elif action == "INFER":
                        st.markdown(f"**🤔 Inference:** {step.get('action_input', '')}")
                    elif action == "SUMMARIZE":
                        st.info(f"**📋 Summarize:** {step.get('action_input', '')}")
                    elif action == "FINAL_ANSWER":
                        st.success(f"**✅ Final Answer:** {step.get('action_input', '')}")
            # Save assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "steps": steps_recorded
            })

if __name__ == "__main__":
    main()
