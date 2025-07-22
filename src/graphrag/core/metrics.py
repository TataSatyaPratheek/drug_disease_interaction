import streamlit as st
import logging
from typing import Dict, Any
import networkx as nx

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)  # Cache for 1 hour - static data
def get_graph_metrics(_graph: nx.MultiDiGraph) -> Dict[str, Any]:  # Add underscore
    """Get static graph topology metrics"""
    try:
        metrics = {
            "nodes": _graph.number_of_nodes(),
            "edges": _graph.number_of_edges(),
            "density": nx.density(_graph),
            "is_directed": _graph.is_directed(),
            "is_multigraph": _graph.is_multigraph()
        }
        
        # Node type distribution
        node_types = {}
        for node, data in _graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        metrics["node_types"] = node_types
        
        # Edge type distribution
        edge_types = {}
        for u, v, data in _graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        metrics["edge_types"] = edge_types
        
        logger.info(f"Graph metrics calculated: {metrics['nodes']} nodes, {metrics['edges']} edges")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate graph metrics: {e}")
        return {}

@st.cache_data(ttl=300)  # Cache for 5 minutes - dynamic data
def get_vector_store_metrics(_vector_store) -> Dict[str, Any]:  # Add underscore
    """Get dynamic vector store statistics"""
    try:
        stats = _vector_store.get_statistics()
        
        performance_metrics = {
            "total_entities": stats.get('total_entities', 0),
            "entity_breakdown": {
                "drugs": stats.get('drugs', 0),
                "diseases": stats.get('diseases', 0),
                "proteins": stats.get('proteins', 0),
                "relationships": stats.get('relationships', 0)
            }
        }
        
        logger.info(f"Vector store metrics: {performance_metrics['total_entities']} total entities")
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Failed to get vector store metrics: {e}")
        return {
            "total_entities": 0,
            "entity_breakdown": {"drugs": 0, "diseases": 0, "proteins": 0, "relationships": 0}
        }

def combine_all_metrics(_graph, _vector_store) -> Dict[str, Any]:  # Add underscores
    """Combine all system metrics into a single view"""
    try:
        graph_metrics = get_graph_metrics(_graph)
        vector_metrics = get_vector_store_metrics(_vector_store)
        performance_metrics = get_system_performance_metrics()
        
        combined = {
            "graph": graph_metrics,
            "vector_store": vector_metrics,
            "performance": performance_metrics,
            "summary": {
                "total_nodes": graph_metrics.get("nodes", 0),
                "total_entities": vector_metrics.get("total_entities", 0),
                "system_health": "operational"
            }
        }
        
        return combined
        
    except Exception as e:
        logger.error(f"Failed to combine metrics: {e}")
        return {"error": "Metrics unavailable"}

@st.cache_data(ttl=60)  # Cache for 1 minute - UI metrics
def get_system_performance_metrics() -> Dict[str, Any]:
    """Get system-wide performance metrics"""
    try:
        # Memory usage (if psutil is available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = "N/A"
        
        metrics = {
            "memory_usage_mb": memory_mb,
            "cache_status": "active",
            "timestamp": st.time()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return {"error": str(e)}
