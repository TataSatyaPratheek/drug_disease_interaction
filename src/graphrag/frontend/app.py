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
    
    st.title(config.APP_TITLE)
    st.markdown(config.APP_SUBTITLE)

    # Sidebar with conversation history
    with st.sidebar:
        user_config = sidebar.render_sidebar()

    # System status
    srv_status = cache.check_system_status()
    status.render_status_indicators(srv_status)

    # Fast metrics
    render_fast_metrics()
    st.markdown("---")

    # Query interface (simplified)
    query, query_type, submitted = query_panel.render_query_panel()

    # Handle pending follow-up questions
    if st.session_state.get('pending_followup'):
        query = st.session_state.pending_followup
        submitted = True
        st.session_state.pending_followup = None

    # Process query (simplified)
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
        engine = state.get_state('engine')
        if engine:
            try:
                result = engine.query(query, query_type=query_type, max_results=user_config.get('max_results', 15))
                state.set_state('last_response', result)
                state.set_state('last_query', query)
            except Exception as e:
                st.error(f"Query failed: {e}")
            finally:
                state.set_state('busy', False)
                st.rerun()

    # Response display (direct)
    if state.get_state('last_response'):
        col1, col2 = st.columns([2, 1])
        with col1:
            response_panel.render_response()
        with col2:
            response_data = state.get_state('last_response')
            entities = response_data.get('retrieved_data', {})
            path_data = response_data.get('path_data')
            community_data = response_data.get('community_data')
            
            if any(entities.values()):
                entity_ids = [e.get('id') for v in entities.values() for e in v if e.get('id')]
                if entity_ids:
                    graph = state.get_state('graph')
                    if graph:
                        subgraph = graph.subgraph([node for node in entity_ids if node in graph]).copy()
                        from graphrag.frontend.components.graph_interactive import render_graph_tabs
                        clicked_node = render_graph_tabs(
                            subgraph,
                            path_data=path_data,
                            communities=community_data,
                            selected_node=state.get_state('selected_node')
                        )

if __name__ == "__main__":
    main()