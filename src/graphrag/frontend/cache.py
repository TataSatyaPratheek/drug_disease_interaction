# src/graphrag/frontend/cache.py
"""Centralized caching for expensive resources and data."""
from ..core.initialization import (
    load_graph_data,
    get_weaviate_connection,
    get_ollama_client,
)
from ..core.query_engine import GraphRAGQueryEngine
from ..core.service_health import get_system_status
from ..core.metrics import combine_all_metrics

import streamlit as st
import concurrent.futures
import time
from pathlib import Path
import json

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
        fut_graph   = exe.submit(load_graph_data)
        fut_store   = exe.submit(get_weaviate_connection)
        fut_client  = exe.submit(get_ollama_client)

        graph        = fut_graph.result()
        vector_store = fut_store.result()
        llm_client   = fut_client.result()

    engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
    return graph, vector_store, llm_client, engine

@st.cache_resource(ttl=3600, show_spinner=False)
def load_system_resources():
    """Warm components in parallel, return tuple when ready."""
    start = time.perf_counter()
    result = _async_init()
    st.toast(f"🔥 Warm-up finished in {time.perf_counter() - start:0.1f}s")
    return result

GRAPH_PKL = Path("data/graph/full_mapped/ddi_knowledge_graph.pickle")
SNAPSHOT  = GRAPH_PKL.with_suffix(".stats.json")

@st.cache_resource(ttl=3600, show_spinner=False)
def fast_graph_stats() -> dict:
    """
    Instant stats from the JSON snapshot.
    Falls back to full scan if the file is missing.
    """
    if SNAPSHOT.exists():
        return json.loads(SNAPSHOT.read_text())
    # one-time heavy fallback
    g = load_graph_data()
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
    }