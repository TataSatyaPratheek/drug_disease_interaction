# src/graphrag/frontend/cache.py
"""Centralized caching for expensive resources and data."""
import streamlit as st
from ..core.initialization import initialize_graphrag_system
from ..core.service_health import get_system_status
from ..core.metrics import combine_all_metrics

@st.cache_resource(ttl=3600) # Cache for 1 hour
def load_system_resources():
    """Load and cache the core GraphRAG system components."""
    graph, vector_store, llm_client, engine = initialize_graphrag_system()
    return graph, vector_store, llm_client, engine

@st.cache_data(ttl=300) # Cache for 5 minutes
def check_system_status():
    """Check and cache the health of backend services."""
    return get_system_status()

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_system_metrics(graph, vector_store):
    """Calculate and cache combined system metrics."""
    return combine_all_metrics(graph, vector_store)
