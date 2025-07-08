# src/graphrag/frontend/components/status.py
"""Renders system status and metric components."""
import streamlit as st
from .. import config

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

def render_system_stats(stats):
    """Render the main system statistics panel."""
    if not stats:
        st.info("Statistics are not available yet.")
        return
        
    summary = stats.get('summary', {})
    st.markdown("## 📊 System Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entities", f"{summary.get('total_entities', 0):,}")
    col2.metric("Relationships", f"{summary.get('relationships', 0):,}")
    col3.metric("Drugs", f"{summary.get('drugs', 0):,}")
    col4.metric("Diseases", f"{summary.get('diseases', 0):,}")
