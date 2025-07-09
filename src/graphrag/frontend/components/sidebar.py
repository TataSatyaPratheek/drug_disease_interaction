import streamlit as st
from graphrag.frontend import config, state

def render_sidebar() -> dict:
    """Render configuration controls and return the user-selected values."""
    st.markdown("## 🔧 Configuration")

    model = st.selectbox(
        "Ollama Model",
        options=config.AVAILABLE_MODELS,
        index=0,
    )

    max_results = st.slider(
        "Max Vector Results",
        5,
        config.MAX_VECTOR_RESULTS,
        value=config.DEFAULT_VECTOR_RESULTS,
    )

    temperature = st.slider(
        "AI Temperature",
        *config.AI_TEMPERATURE_RANGE,
        value=config.DEFAULT_AI_TEMPERATURE,
        step=0.05,
    )

    st.markdown("---")
    if st.button("🧹 Cleanup", use_container_width=True, disabled=state.get_state("busy")):
        state.cleanup_resources()
        st.cache_resource.clear()
        st.experimental_rerun()

    return {
        "model": model,
        "max_results": max_results,
        "temperature": temperature,
    }