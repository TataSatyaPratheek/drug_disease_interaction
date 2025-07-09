# src/graphrag/retrievers/community_retriever.py

import networkx as nx
from typing import List, Dict, Any
from ..core.graph_analytics import HighPerformanceGraphAnalytics
from ..core.connection_resilience import ConnectionResilience
import streamlit as st
import logging

class CommunityRetriever:
    """Retrieves community information from the graph using high-performance analytics."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.analytics = HighPerformanceGraphAnalytics(graph)
        self.resilience = ConnectionResilience()
        self.logger = logging.getLogger(__name__)

    @ConnectionResilience.with_retry(max_attempts=3, wait_seconds=1)
    def get_communities(self, node_ids: List[str], algorithm: str = 'louvain') -> List[Dict[str, Any]]:
        """Get communities for specified nodes using high-performance algorithms."""
        try:
            # Use high-performance igraph-based community detection
            community_data = self.analytics.detect_communities(algorithm)
            
            # Filter communities that contain target nodes
            relevant_communities = []
            for community in community_data['communities']:
                if any(node in community['nodes'] for node in node_ids):
                    # Add additional metadata
                    community_nodes = community['nodes']
                    
                    # Calculate community centrality
                    centrality_metrics = self.analytics.compute_centrality_metrics()
                    avg_centrality = sum(
                        centrality_metrics['pagerank'].get(node, 0) 
                        for node in community_nodes
                    ) / len(community_nodes)
                    
                    enhanced_community = {
                        'id': community['id'],
                        'nodes': community_nodes,
                        'size': community['size'],
                        'modularity': community_data['modularity'],
                        'avg_centrality': avg_centrality,
                        'algorithm': algorithm,
                        'node_types': self._get_community_node_types(community_nodes),
                        'description': self._generate_community_description(community_nodes)
                    }
                    relevant_communities.append(enhanced_community)
            
            # Sort by relevance (combination of size and centrality)
            relevant_communities.sort(
                key=lambda x: x['size'] * x['avg_centrality'], 
                reverse=True
            )
            
            self.logger.info(f"Found {len(relevant_communities)} relevant communities")
            return relevant_communities
            
        except Exception as e:
            self.logger.error(f"Community retrieval failed: {e}")
            return []

    def _get_community_node_types(self, community_nodes: List[str]) -> Dict[str, int]:
        """Get distribution of node types in community."""
        type_counts = {}
        for node in community_nodes:
            if node in self.graph:
                node_type = self.graph.nodes[node].get('type', 'unknown')
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts

    def _generate_community_description(self, community_nodes: List[str]) -> str:
        """Generate descriptive text for community."""
        type_counts = self._get_community_node_types(community_nodes)
        
        descriptions = []
        for node_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            descriptions.append(f"{count} {node_type}{'s' if count > 1 else ''}")
        
        return f"Community with {', '.join(descriptions)}"

    @st.cache_data(ttl=ConnectionResilience.adaptive_cache_ttl)
    def get_community_statistics(_self) -> Dict[str, Any]:
        """Get comprehensive community statistics."""
        try:
            community_data = _self.analytics.detect_communities('louvain')
            
            stats = {
                'total_communities': len(community_data['communities']),
                'modularity': community_data['modularity'],
                'largest_community_size': max(
                    (c['size'] for c in community_data['communities']), 
                    default=0
                ),
                'average_community_size': sum(
                    c['size'] for c in community_data['communities']
                ) / len(community_data['communities']) if community_data['communities'] else 0,
                'community_type_distribution': _self._get_community_type_distribution(community_data['communities'])
            }
            
            return stats
            
        except Exception as e:
            _self.logger.error(f"Community statistics failed: {e}")
            return {}

    def _get_community_type_distribution(self, communities: List[Dict]) -> Dict[str, int]:
        """Get distribution of community types."""
        type_distribution = {}
        
        for community in communities:
            dominant_type = self._get_dominant_type(community['nodes'])
            type_distribution[dominant_type] = type_distribution.get(dominant_type, 0) + 1
        
        return type_distribution

    def _get_dominant_type(self, nodes: List[str]) -> str:
        """Get the dominant node type in a community."""
        type_counts = self._get_community_node_types(nodes)
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'unknown'
