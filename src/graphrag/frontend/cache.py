# src/graphrag/frontend/cache.py
"""Centralized caching for expensive resources and data."""

import time
import logging
import streamlit as st
from pathlib import Path
import json

from ..core.initialization import (
    load_graph_data,
    get_weaviate_connection,
    get_ollama_client,
)
from ..core.query_engine import GraphRAGQueryEngine
from ..core.service_health import get_system_status

LOGGER = logging.getLogger(__name__)

GRAPH_PKL = Path("data/graph/full_mapped/ddi_knowledge_graph.pickle")
SNAPSHOT = GRAPH_PKL.with_suffix(".stats.json")

@st.cache_data(ttl=60, show_spinner=False)
def check_system_status():
    """
    Check and cache the health of backend services (Weaviate, Ollama).
    Cached for 60 seconds to avoid excessive health checks.
    """
    return get_system_status()

@st.cache_resource(ttl=3600, show_spinner=False)
def load_system_resources():
    """
    Load all necessary backend components sequentially with clear status updates.
    This is more robust than parallel loading for debugging.
    """
    with st.status("🚀 Initializing backend...", expanded=True) as status:
        try:
            start_time = time.perf_counter()

            status.update(label="Connecting to Weaviate...")
            vector_store = get_weaviate_connection()
            LOGGER.info("Weaviate connection successful.")

            status.update(label="Connecting to Ollama...")
            llm_client = get_ollama_client()
            LOGGER.info("Ollama client initialized.")

            status.update(label="Loading knowledge graph... (this may take a while)")
            graph = load_graph_data()
            LOGGER.info("Knowledge graph loaded successfully.")

            status.update(label="Initializing Query Engine...")
            engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
            LOGGER.info("Query Engine initialized.")

            end_time = time.perf_counter()
            total_time = end_time - start_time
            LOGGER.info(f"Backend initialization complete in {total_time:.2f} seconds.")
            
            status.update(label=f"Initialization complete in {total_time:.2f}s!", state="complete")
            
            return graph, vector_store, llm_client, engine

        except Exception as e:
            LOGGER.error("Backend initialization failed.", exc_info=True)
            status.update(label="Initialization Failed!", state="error", expanded=True)
            st.error(f"A critical error occurred during backend setup: {e}")
            # Re-raise the exception to stop the app cleanly
            raise

@st.cache_data(show_spinner=False)
def fast_graph_stats() -> dict:
    """
    Instant stats from the JSON snapshot.
    Falls back to full scan if the file is missing.
    """
    if SNAPSHOT.exists():
        return json.loads(SNAPSHOT.read_text())
    
    LOGGER.warning("Stats snapshot not found. Generating from scratch. This will be slow.")
    g = load_graph_data()
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
    }
