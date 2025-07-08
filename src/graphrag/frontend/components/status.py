# src/graphrag/frontend/components/status.py
"""Renders system status and metric components."""
from .. import config
from ..cache import fast_graph_stats
import streamlit as st

def render_status_indicators(system_status):
    """Render system status indicators for Ollama and Weaviate."""
    ollama_status = system_status.get('ollama', {})
    weaviate_status = system_status.get('weaviate', {})
    
    col1, col2 = st.columns(2)
    with col1:
        if ollama_status.get('status') == 'healthy':
            st.success("✅ Ollama running")
        else:
            st.error(config.ERROR_MESSAGES["ollama_not_available"])
    with col2:
        if weaviate_status.get('status') == 'healthy':
            st.success("✅ Weaviate running")
        else:
            st.error(config.ERROR_MESSAGES["weaviate_not_available"])

def render_system_stats(_unused_vector=None):
    stats = fast_graph_stats()
    col1, col2 = st.columns(2)
    col1.metric("Graph Nodes", f"{stats['nodes']:,}")
    col2.metric("Graph Edges", f"{stats['edges']:,}")