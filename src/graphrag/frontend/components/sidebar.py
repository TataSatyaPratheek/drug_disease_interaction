import streamlit as st
from graphrag.frontend import config, state, cache

def render_sidebar() -> dict:
    """Render configuration controls and return the user-selected values."""
    st.markdown("## 🔧 Configuration")

    selected_model = st.selectbox(
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
    
    if st.button("🧹 Cleanup Resources", use_container_width=True):
        try:
            if state.cleanup_resources():
                st.success(config.SUCCESS_MESSAGES["cleanup_complete"])
                if hasattr(cache, 'load_system_resources'):
                    cache.load_system_resources.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Cleanup failed: {e}")

    if st.button("🔄 Force Restart", use_container_width=True):
        try:
            state.cleanup_resources()
            if hasattr(cache, 'load_system_resources'):
                cache.load_system_resources.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Force restart failed: {e}")

    config_dict = {
        "model": selected_model,
        "max_results": max_results,
        "temperature": temperature,
    }
    
    # Store in session state for query processing
    state.set_state("user_config", config_dict)
    return config_dict