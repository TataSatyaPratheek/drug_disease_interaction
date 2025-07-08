# src/graphrag/frontend/app.py
"""Main Streamlit application entry point."""
import streamlit as st
import sys
from pathlib import Path

# Ensure the project root is in the path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphrag.frontend import config, state, cache, router
from graphrag.frontend.components import (
    sidebar, status, query_panel, response_panel, visualization
)

def main():
    """Main application flow."""
    st.set_page_config(page_title=config.APP_TITLE, layout="wide", page_icon="🧬")
    state.initialize_session_state()

    st.title(config.APP_TITLE)
    st.markdown(config.APP_SUBTITLE)
    
    # Render sidebar and get user config
    user_config = sidebar.render_sidebar()
    
    # Check system status
    system_status = cache.check_system_status()
    status.render_status_indicators(system_status)
    if system_status.get('overall') != 'healthy':
        st.error("System not ready. Please check backend services.")
        return
        
    # Load system resources if not already initialized
    if not state.get_state('system_initialized'):
        with st.spinner("🚀 Initializing GraphRAG system... This may take a moment."):
            try:
                graph, vector_store, llm_client, engine = cache.load_system_resources()
                state.store_system_components(graph, vector_store, llm_client, engine)
            except Exception as e:
                st.error(f"{config.ERROR_MESSAGES['initialization_failed']}\n\nDetails: {e}")
                st.exception(e)
                return
    
    # Main page vs. sub-pages
    page = state.get_state('page', 'main')
    if page != 'main':
        router.render_page(page)
    else:
        # Display system metrics
        graph = state.get_state('graph')
        vector_store = state.get_state('vector_store')
        metrics = cache.get_system_metrics(graph, vector_store)
        status.render_system_stats(metrics)
        st.markdown("---")
        
        # Render the query panel and process input
        query, query_type = query_panel.render_query_panel()
        if query:
            engine = state.get_state('engine')
            with st.spinner("🔍 Processing query..."):
                try:
                    result = engine.query(
                        query,
                        query_type=query_type,
                        max_results=user_config['max_results']
                    )
                    state.set_state('last_response', result)
                    state.set_state('last_query', query)
                except Exception as e:
                    st.error(f"{config.ERROR_MESSAGES['query_failed']}\n\nDetails: {e}")
                finally:
                    state.set_state('busy', False)
                    st.rerun()

        # Render response and visualization
        if state.get_state('last_response'):
            col1, col2 = st.columns([2, 1])
            with col1:
                response_panel.render_response()
            with col2:
                entities = state.get_state('last_response', {}).get('retrieved_data', {})
                if any(entities.values()):
                    entity_ids = [e.get('id') for v in entities.values() for e in v if e.get('id')]
                    if entity_ids:
                        subgraph = graph.subgraph(entity_ids).copy()
                        visualization.render_graph_visualization(subgraph)

if __name__ == "__main__":
    main()
