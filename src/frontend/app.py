import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="DDI-AI: The Biomedical Research Copilot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
<style>
    /* Main header style */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Search result summary style */
    .summary-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    /* Entity card style */
    .entity-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease-in-out;
    }
    .entity-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .entity-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f77b4;
    }
    .entity-source {
        font-size: 0.8rem;
        color: #fff;
        background-color: #555;
        padding: 2px 8px;
        border-radius: 10px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


class DDIFrontend:
    """The Streamlit frontend for the Drug-Disease Interaction AI."""
    def __init__(self):
        self.api_base_url = "http://localhost:8000/api/v1"
        if 'api_status' not in st.session_state:
            st.session_state.api_status = self.check_api_health()

    def check_api_health(self) -> Dict[str, Any]:
        """Check the health of the backend API."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            return {"status": "unhealthy", "error": f"API returned status {response.status_code}"}
        except requests.RequestException as e:
            return {"status": "error", "error": f"Connection failed: {e}"}

    def render_sidebar(self):
        """Render the sidebar with controls and status information."""
        with st.sidebar:
            st.header("System Status")
            status_info = st.session_state.api_status
            if status_info["status"] == "healthy":
                st.success("✅ API Connected")
                db_stats = status_info["data"].get("database_stats", {})
                if "neo4j" in db_stats:
                    st.metric("Graph Nodes (Neo4j)", f"{db_stats['neo4j'].get('node_count', 0):,}")
                if "weaviate" in db_stats:
                    total_vectors = sum(info.get("count", 0) for info in db_stats['weaviate'].get("total_objects_by_collection", {}).values())
                    st.metric("Vector Objects (Weaviate)", f"{total_vectors:,}")
            else:
                st.error(f"❌ API Error: {status_info.get('error', 'Check console')}")

            st.markdown("---")
            st.header("🔬 Investigation Mode")
            search_mode = st.selectbox(
                "Select a research focus:",
                ("Exploration", "Hypothesis Testing", "Drug Repurposing", "Drug Discovery"),
                help="Select a mode to tailor the search focus (feature in development)."
            )

            st.header("⚙️ Search Parameters")
            max_results = st.slider("Number of Results", 5, 20, 10)
        
        return {"mode": search_mode, "max_results": max_results}

    def render_search_interface(self):
        """Render the main search bar and example queries."""
        st.markdown('<h1 class="main-header">DDI-AI: The Biomedical Research Copilot</h1>', unsafe_allow_html=True)
        query = st.text_input(
            "Ask a question about drugs, diseases, and their interactions:",
            placeholder="e.g., 'Find drugs that target EGFR for cancer treatment'",
            label_visibility="collapsed"
        )
        return query

    def search_api(self, query: str, settings: Dict) -> Dict:
        """Call the backend search API."""
        payload = {
            "query": query,
            "max_results": settings["max_results"]
        }
        try:
            response = requests.post(f"{self.api_base_url}/search", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Search failed. API returned status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            st.error(f"Failed to connect to backend API: {e}")
        return None

    def render_results(self, results: Dict):
        """Render the search results in a professional format."""
        st.subheader("🤖 AI-Generated Summary")
        st.markdown(f'<div class="summary-box">{results["answer"]}</div>', unsafe_allow_html=True)
        
        st.subheader("🔬 Evidence & Citations")
        for entity in results.get("entities", []):
            with st.container():
                st.markdown('<div class="entity-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<span class="entity-title">{entity["name"]}</span> <span class="entity-source">{entity["type"].upper()}</span>', unsafe_allow_html=True)
                    st.write(f"**Source DB**: `{entity['source']}` | **Relevance Score**: {entity['score']:.3f}")
                    if entity.get("description"):
                        st.markdown(f"> {entity['description'][:250]}...")
                with col2:
                    # Gauge chart for relevance score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=entity['score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 1]}, 'bar': {'color': "#1f77b4"}}
                    ))
                    fig.update_layout(height=100, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application execution loop."""
        settings = self.render_sidebar()
        query = self.render_search_interface()

        if query:
            with st.spinner("Performing hybrid search across graph and vector databases..."):
                results = self.search_api(query, settings)
            if results:
                self.render_results(results)

if __name__ == "__main__":
    frontend = DDIFrontend()
    frontend.run()
