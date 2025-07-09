"""Main Streamlit application entry point (fragment-friendly)."""
import streamlit as st
import logging
from graphrag.frontend import config, state, cache, router
from graphrag.frontend.components import (
    sidebar,                    # ← fragment
    status,
    query_panel,                # ← fragment
    response_panel,
    visualization,
)

logger = logging.getLogger(__name__)

def main() -> None:
    st.set_page_config(page_title=config.APP_TITLE, page_icon="🧬", layout="wide")
    state.initialize_session_state()

    st.title(config.APP_TITLE)
    st.markdown(config.APP_SUBTITLE)

    # ── 1. Sidebar ────────────────────────────────────────
    with st.sidebar:
        user_config = sidebar.render_sidebar()

    # ── 2. Service health banner ──────────────────────────
    srv_status = cache.check_system_status()
    status.render_status_indicators(srv_status)
    if srv_status.get("overall") != "healthy":
        st.stop()

    # ── 3. Router for extra pages ─────────────────────────
    if state.get_state("page") != "main":
        router.render_page(state.get_state("page"))
        return

    # ── 4. Fast stats (no graph unpickling) ───────────────
    status.render_system_stats(cache.fast_graph_stats())
    st.divider()

    # ── 5. Query panel (fragment) ─────────────────────────
    query, query_type, submitted = query_panel.render_query_panel()
    
    if submitted and query and not state.get_state("busy"):
        # Set busy state immediately
        state.set_state("busy", True)
        
        # initialise on demand
        if not state.get_state("system_initialized"):
            with st.spinner("🚀 Initialising back-end…"):
                try:
                    graph, vector_store, llm_client, engine = cache.load_system_resources()
                    state.store_system_components(graph, vector_store, llm_client, engine)
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize system: {e}")
                    state.set_state("busy", False)
                    st.stop()

        # Process the query
        engine = state.get_state('engine')
        if engine:
            with st.spinner("🔍 Processing query..."):
                try:
                    user_config = state.get_state("user_config", {"max_results": 10})
                    result = engine.query(
                        query,
                        query_type=query_type,
                        max_results=user_config.get('max_results', 10)
                    )
                    state.set_state('last_response', result)
                    state.set_state('last_query', query)
                    st.success("✅ Query processed successfully!")
                except Exception as e:
                    st.error(f"Query processing failed: {e}")
                    logger.error(f"Query processing error: {e}")
                finally:
                    state.set_state('busy', False)
                    st.rerun()
        else:
            st.error("Engine not initialized")
            state.set_state("busy", False)

    # ── 6. Response & graph ───────────────────────────────
    if state.get_state("last_response"):
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            response_panel.render_response()
        with col2:
            g = state.get_state("graph")
            ent_ids = [
                e["id"]
                for lst in state.get_state("last_response")["retrieved_data"].values()
                for e in lst if e.get("id")
            ]
            if g and ent_ids:
                sub = g.subgraph(ent_ids).copy()
                visualization.render_graph_visualization(sub)

if __name__ == "__main__":
    main()