# src/graphrag/frontend/state.py
"""Streamlit session state management"""
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables with default values."""
    defaults = {
        'graph': None,
        'vector_store': None,
        'llm_client': None,
        'engine': None,
        'system_initialized': False,
        'last_query': "",
        'last_response': None,
        'page': 'main',
        'busy': False,
    }
    for key, default_value in defaults.items():
        st.session_state.setdefault(key, default_value)

def get_state(key, default=None):
    """Safely get a value from session state."""
    return st.session_state.get(key, default)

def set_state(key, value):
    """Set a value in session state."""
    st.session_state[key] = value

def store_system_components(graph, vector_store, llm_client, engine):
    """Store core system components in the session state."""
    set_state('graph', graph)
    set_state('vector_store', vector_store)
    set_state('llm_client', llm_client)
    set_state('engine', engine)
    set_state('system_initialized', True)
    logger.info("System components stored in session state.")

def cleanup_resources():
    """Clean up resources and reset the session state."""
    try:
        vector_store = get_state('vector_store')
        if vector_store:
            vector_store.close()
            logger.info("Weaviate client closed.")
        
        # Reset all state keys to their defaults
        initialize_session_state()
        
        # Specifically reset component flags
        set_state('system_initialized', False)
        
        logger.info("Session resources cleaned up and state reset.")
        return "cleanup_complete"
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
        return None
