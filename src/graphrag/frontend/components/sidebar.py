# src/graphrag/frontend/components/sidebar.py
"""Renders the sidebar controls."""
import streamlit as st
from .. import config, state, cache

def render_sidebar():
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        st.markdown("## 🔧 Configuration")
        
        selected_model = st.selectbox(
            "Ollama Model",
            options=config.AVAILABLE_MODELS,
            index=0
        )
        
        max_results = st.slider(
            "Max Vector Results",
            min_value=5,
            max_value=config.MAX_VECTOR_RESULTS,
            value=config.DEFAULT_VECTOR_RESULTS
        )
        
        temperature = st.slider(
            "AI Temperature",
            min_value=config.AI_TEMPERATURE_RANGE[0],
            max_value=config.AI_TEMPERATURE_RANGE[1],
            value=config.DEFAULT_AI_TEMPERATURE,
            step=0.05
        )
        
        st.markdown("---")
        
        if st.button("🧹 Cleanup Resources", use_container_width=True):
            if state.cleanup_resources():
                st.success(config.SUCCESS_MESSAGES["cleanup_complete"])
                cache.load_system_resources.clear()
            st.rerun()

        if st.button("🔄 Force Restart", use_container_width=True):
            state.cleanup_resources()
            cache.load_system_resources.clear()
            st.rerun()
            
    return {
        "model": selected_model,
        "max_results": max_results,
        "temperature": temperature,
    }
