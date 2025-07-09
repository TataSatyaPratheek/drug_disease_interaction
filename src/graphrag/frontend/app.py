"""Main Streamlit application entry point (fragment-friendly)."""
import streamlit as st
import logging
from graphrag.frontend.components.conversation_manager import ConversationManager
from graphrag.frontend import config, state, cache, router
from graphrag.frontend.components import (
    sidebar,                    # ← fragment
    status,
    query_panel,                # ← fragment
    response_panel,
    visualization
)

logger = logging.getLogger(__name__)

@st.fragment
def render_query_interface():
    """Fragmented query interface to prevent unnecessary reruns."""
    
    # Handle pending follow-up
    if st.session_state.get('pending_followup'):
        query_text = st.session_state.pending_followup
        st.session_state.pending_followup = None
        query_type = "auto"
        submitted = True
    else:
        # Regular query panel
        query_text, query_type, submitted = query_panel.render_query_panel()
    
    return query_text, query_type, submitted

@st.fragment  
def render_response_container():
    """Fragmented response container."""
    response_panel.render_response()

def render_fast_metrics():
    """Display fast graph metrics from cache.fast_graph_stats()."""
    stats = cache.fast_graph_stats()
    st.markdown(f"**Graph Stats:**  ")
    st.markdown(f"- **Nodes:** {stats.get('nodes', 'N/A')}")
    st.markdown(f"- **Edges:** {stats.get('edges', 'N/A')}")

def main():
    """Main application with proper fragmentation."""
    st.set_page_config(page_title=config.APP_TITLE, layout="wide", page_icon="🧬")
    state.initialize_session_state()
    
    # Initialize conversation manager
    ConversationManager.initialize_conversation()
    
    st.title(config.APP_TITLE)
    st.markdown(config.APP_SUBTITLE)
    
    # Sidebar with conversation history
    with st.sidebar:
        user_config = sidebar.render_sidebar()
        ConversationManager.render_conversation_history()
    
    # System status
    srv_status = cache.check_system_status()
    status.render_status_indicators(srv_status)
    
    # Fast metrics
    render_fast_metrics()
    st.markdown("---")
    
    # Query interface (fragmented)
    query, query_type, submitted = render_query_interface()
    
    # Process query
    if submitted and query:
        state.set_state("busy", True)
        
        # Initialize system if needed
        if not state.get_state("system_initialized"):
            try:
                graph, vector_store, llm_client, engine = cache.load_system_resources()
                state.store_system_components(graph, vector_store, llm_client, engine)
            except Exception as e:
                st.error(f"System initialization failed: {e}")
                state.set_state("busy", False)
                return
        
        # Process query with conversation context
        engine = state.get_state('engine')
        if engine:
            try:
                # Add conversation context to query
                conversation_context = ConversationManager.get_conversation_context()
                enhanced_query = conversation_context + query if conversation_context else query
                
                result = engine.query(enhanced_query, query_type=query_type)
                
                # Store conversation turn
                ConversationManager.store_conversation_turn(query, result)
                
                state.set_state('last_response', result)
                state.set_state('last_query', query)
                
            except Exception as e:
                st.error(f"Query failed: {e}")
            finally:
                state.set_state('busy', False)
                st.rerun()
    
    # Response display (fragmented)
    render_response_container()

if __name__ == "__main__":
    main()