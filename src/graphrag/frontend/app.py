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
    
    if submitted:
        if not query:
            st.warning("Please enter a query.")
            st.stop()

        state.set_state("busy", True)
        # No spinner here; load_system_resources will create its own st.status
        # Initialize on demand
        if not state.get_state("system_initialized"):
            try:
                graph, vector_store, llm_client, engine = cache.load_system_resources()
                state.store_system_components(graph, vector_store, llm_client, engine)
            except Exception as e:
                # The error is already displayed by the cached function
                state.set_state("busy", False)
                st.stop()

        # Process the query
        engine = state.get_state('engine')
        if engine:
            with st.spinner("🔍 Processing query..."):
                try:
                    result = engine.query(
                        query,
                        query_type=query_type,
                        max_results=user_config.get('max_results', 15)
                    )
                    state.set_state('last_response', result)
                    state.set_state('last_query', query)
                except Exception as e:
                    st.error(f"Query processing failed: {e}")
                    logger.error(f"Query processing error: {e}", exc_info=True)
                finally:
                    state.set_state('busy', False)
                    st.rerun()
        else:
            st.error("❌ Engine not initialized. Cannot process query.")
            state.set_state("busy", False)

    # ── 6. Response & graph ───────────────────────────────
    if state.get_state("last_response"):
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            response_panel.render_response()
        with col2:
            g = state.get_state("graph")
            response_data = state.get_state("last_response")
            if g and response_data:
                retrieved_data = response_data.get("retrieved_data", {})
                ent_ids = [
                    e["id"]
                    for lst in retrieved_data.values()
                    for e in lst if e.get("id")
                ]
                if ent_ids:
                    sub = g.subgraph([node for node in ent_ids if node in g]).copy()
                    visualization.render_graph_visualization(sub)

if __name__ == "__main__":
    main()