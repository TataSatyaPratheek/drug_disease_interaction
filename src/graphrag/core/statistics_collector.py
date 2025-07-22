import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def collect_weaviate_statistics(_vector_store) -> Dict[str, Any]:
    """Collect detailed Weaviate statistics with working connection"""
    try:
        # Use direct connection approach that works
        client = _vector_store.client
        
        basic_stats = {}
        collections = ["Drug", "Disease", "Protein", "Relationship"]
        total_count = 0
        
        for collection_name in collections:
            try:
                if client.collections.exists(collection_name):
                    collection = client.collections.get(collection_name)
                    response = collection.aggregate.over_all(total_count=True)
                    count = response.total_count or 0
                    
                    # Map to expected keys
                    key = collection_name.lower() + "s"
                    basic_stats[key] = count
                    total_count += count
                else:
                    key = collection_name.lower() + "s"
                    basic_stats[key] = 0
            except Exception as e:
                logging.error(f"Error getting {collection_name} count: {e}")
                key = collection_name.lower() + "s"
                basic_stats[key] = 0
        
        basic_stats['total_entities'] = total_count
        
        return {
            "basic_stats": basic_stats,
            "collections": {name: {"exists": True, "count": basic_stats.get(name.lower() + "s", 0)} 
                          for name in collections},
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Failed to collect Weaviate statistics: {e}")
        return {
            "basic_stats": {"total_entities": 0, "drugs": 0, "diseases": 0, "proteins": 0, "relationships": 0},
            "status": "error",
            "error": str(e)
        }

def collect_graph_statistics(graph) -> Dict[str, Any]:
    """Collect NetworkX graph statistics"""
    try:
        import networkx as nx
        
        # Basic statistics
        basic_stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph)
        }
        
        # Degree statistics
        degrees = dict(graph.degree())
        degree_values = list(degrees.values())
        
        degree_stats = {
            "average_degree": sum(degree_values) / len(degree_values) if degree_values else 0,
            "max_degree": max(degree_values) if degree_values else 0,
            "min_degree": min(degree_values) if degree_values else 0
        }
        
        # Node type distribution
        node_type_counts = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        return {
            "basic": basic_stats,
            "degree": degree_stats,
            "node_types": node_type_counts,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to collect graph statistics: {e}")
        return {
            "basic": {},
            "degree": {},
            "node_types": {},
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

def get_comprehensive_statistics(graph, vector_store) -> Dict[str, Any]:
    """Get comprehensive system statistics"""
    try:
        graph_stats = collect_graph_statistics(graph)
        weaviate_stats = collect_weaviate_statistics(vector_store)
        
        # Create summary
        summary = {
            "graph_nodes": graph_stats.get("basic", {}).get("nodes", 0),
            "graph_edges": graph_stats.get("basic", {}).get("edges", 0),
            "vector_entities": weaviate_stats.get("basic_stats", {}).get("total_entities", 0),
            "system_status": "operational" if (
                graph_stats.get("status") == "success" and 
                weaviate_stats.get("status") == "success"
            ) else "degraded"
        }
        
        return {
            "summary": summary,
            "graph": graph_stats,
            "vector_store": weaviate_stats,
            "collected_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive statistics: {e}")
        return {
            "summary": {"system_status": "error"},
            "error": str(e),
            "collected_at": datetime.now().isoformat()
        }
