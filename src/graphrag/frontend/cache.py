# src/graphrag/frontend/cache.py
"""Centralized caching for expensive resources and data."""
import streamlit as st
import concurrent.futures
import time
from ..core.initialization import initialize_graphrag_system
from ..core.service_health import get_system_status
from ..core.metrics import combine_all_metrics

@st.cache_data(ttl=300) # Cache for 5 minutes
def check_system_status():
    """Check and cache the health of backend services."""
    return get_system_status()

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_system_metrics(graph, vector_store):
    """Calculate and cache combined system metrics."""
    return combine_all_metrics(graph, vector_store)

def _async_init():
    """Run the three costly loaders in parallel threads."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as exe:
        fut_graph   = exe.submit(initialize_graphrag_system.__wrapped__.__globals__['load_graph_data'])
        fut_store   = exe.submit(initialize_graphrag_system.__wrapped__.__globals__['get_weaviate_connection'])
        fut_client  = exe.submit(initialize_graphrag_system.__wrapped__.__globals__['get_ollama_client'])
        graph       = fut_graph.result()
        vector_store= fut_store.result()
        llm_client  = fut_client.result()
    # engine depends on all three
    engine = initialize_graphrag_system.__wrapped__.__globals__['GraphRAGQueryEngine'](
        graph, llm_client, vector_store
    )
    return graph, vector_store, llm_client, engine

@st.cache_resource(ttl=3600)
def load_system_resources():
    """Warm components in parallel, return tuple when ready."""
    start = time.perf_counter()
    result = _async_init()
    st.toast(f"🔥 Warm-up finished in {time.perf_counter() - start:0.1f}s")
    return result
