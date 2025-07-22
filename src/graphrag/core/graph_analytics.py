"""High-performance graph analytics using python-igraph."""


import igraph as ig
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import streamlit as st
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GraphMetrics:
    """Container for graph analysis metrics."""
    node_centrality: Dict[str, float]
    community_structure: Dict[str, int]
    path_importance: Dict[Tuple[str, str], float]
    clustering_coefficient: float
    modularity: float

class HighPerformanceGraphAnalytics:
    """High-performance graph analysis using igraph."""
    
    def __init__(self, nx_graph: nx.MultiDiGraph):
        self.nx_graph = nx_graph
        self._ig_graph = None  # Lazy initialization
        self._node_mapping = None
        self._reverse_mapping = None
        self._initialized = False
        self._precomputed_dir = Path("data/analytics")
        self._centrality_cache = None
        self._community_cache = None

    def _load_precomputed_data(self):
        """Load pre-computed analytics data."""
        try:
            # Load centrality data
            centrality_file = self._precomputed_dir / 'centrality_metrics.json'
            if centrality_file.exists():
                with open(centrality_file, 'r') as f:
                    self._centrality_cache = json.load(f)
                logger.info("Loaded pre-computed centrality metrics")
            # Load community data
            community_file = self._precomputed_dir / 'community_data.json'
            if community_file.exists():
                with open(community_file, 'r') as f:
                    self._community_cache = json.load(f)
                logger.info("Loaded pre-computed community data")
            # Load mappings
            mapping_file = self._precomputed_dir / 'graph_mappings.json'
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    self._node_mapping = mapping_data['node_mapping']
                    self._reverse_mapping = {int(k): v for k, v in mapping_data['reverse_mapping'].items()}
                logger.info("Loaded pre-computed graph mappings")
        except Exception as e:
            logger.warning(f"Failed to load pre-computed data: {e}")
    
    def _ensure_initialized(self):
        """Lazy initialization of igraph conversion."""
        if self._initialized:
            return
        try:
            logger.info("Initializing igraph conversion (lazy loading)...")
            # Create node mapping
            nodes = list(self.nx_graph.nodes())
            self._node_mapping = {node: i for i, node in enumerate(nodes)}
            self._reverse_mapping = {i: node for node, i in self._node_mapping.items()}
            # Create edge list for igraph
            edges = []
            edge_attrs = {'weight': [], 'type': []}
            for u, v, data in self.nx_graph.edges(data=True):
                edges.append((self._node_mapping[u], self._node_mapping[v]))
                edge_attrs['weight'].append(data.get('weight', 1.0))
                edge_attrs['type'].append(data.get('type', 'unknown'))
            # Create igraph
            self._ig_graph = ig.Graph(edges, directed=True)
            # Add node attributes
            self._ig_graph.vs['name'] = [self._reverse_mapping[i] for i in range(len(nodes))]
            self._ig_graph.vs['type'] = [
                self.nx_graph.nodes[self._reverse_mapping[i]].get('type', 'unknown') 
                for i in range(len(nodes))
            ]
            # Add edge attributes
            self._ig_graph.es['weight'] = edge_attrs['weight']
            self._ig_graph.es['type'] = edge_attrs['type']
            self._initialized = True
            logger.info(f"Initialized igraph with {len(nodes)} nodes and {len(edges)} edges")
        except Exception as e:
            logger.error(f"Failed to initialize igraph: {e}")
            raise
    
    @st.cache_data(ttl=3600)
    def compute_centrality_metrics(_self) -> Dict[str, Dict[str, float]]:
        """Compute multiple centrality metrics efficiently."""
        try:
            # Try to use pre-computed data first
            if _self._centrality_cache is None:
                _self._load_precomputed_data()
            if _self._centrality_cache:
                logger.info("Using pre-computed centrality metrics")
                return _self._centrality_cache
            # Fallback to lazy initialization only if needed
            _self._ensure_initialized()
            metrics = {}
            # PageRank centrality
            pagerank = _self._ig_graph.pagerank(weights='weight')
            metrics['pagerank'] = {
                _self._reverse_mapping[i]: score 
                for i, score in enumerate(pagerank)
            }
            # Betweenness centrality
            betweenness = _self._ig_graph.betweenness(weights='weight')
            metrics['betweenness'] = {
                _self._reverse_mapping[i]: score 
                for i, score in enumerate(betweenness)
            }
            logger.info("Computed centrality metrics for all nodes")
            return metrics
        except Exception as e:
            logger.error(f"Failed to compute centrality metrics: {e}")
            return {}
    
    @st.cache_data(ttl=3600)
    def detect_communities(_self, algorithm: str = 'louvain') -> Dict[str, Any]:
        """Detect communities using pre-computed data or algorithms."""
        try:
            # Try to use pre-computed data first
            if _self._community_cache is None:
                _self._load_precomputed_data()
            if _self._community_cache and algorithm == 'louvain':
                logger.info("Using pre-computed community data")
                return _self._community_cache
            # Fallback to computation if needed
            _self._ensure_initialized()
            if algorithm == 'louvain':
                communities = _self._ig_graph.community_multilevel(weights='weight')
            else:
                raise ValueError(f"Algorithm {algorithm} not supported without pre-computation")
            # Convert to node-community mapping
            node_communities = {}
            community_info = []
            for i, community in enumerate(communities):
                community_nodes = [_self._reverse_mapping[node_idx] for node_idx in community]
                community_info.append({
                    'id': i,
                    'nodes': community_nodes,
                    'size': len(community_nodes),
                    'modularity': communities.modularity
                })
                for node in community_nodes:
                    node_communities[node] = i
            result = {
                'node_communities': node_communities,
                'communities': community_info,
                'modularity': communities.modularity,
                'algorithm': algorithm
            }
            logger.info(f"Detected {len(communities)} communities using {algorithm}")
            return result
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {'node_communities': {}, 'communities': [], 'modularity': 0.0}
    
    @st.cache_data(ttl=1800)
    def compute_shortest_paths(_self, source_nodes: List[str], target_nodes: List[str]) -> Dict[str, Any]:
        """Compute shortest paths between source and target nodes."""
        try:
            results = {}
            
            for source in source_nodes:
                if source not in _self._node_mapping:
                    continue
                    
                source_idx = _self._node_mapping[source]
                
                for target in target_nodes:
                    if target not in _self._node_mapping or source == target:
                        continue
                        
                    target_idx = _self._node_mapping[target]
                    
                    try:
                        # Get shortest path
                        path = _self._ig_graph.get_shortest_paths(
                            source_idx, target_idx, weights='weight', output='vpath'
                        )[0]
                        
                        if path:
                            path_nodes = [_self._reverse_mapping[idx] for idx in path]
                            path_length = len(path) - 1
                            
                            # Calculate path score based on centrality
                            centrality_metrics = _self.compute_centrality_metrics()
                            path_score = sum(
                                centrality_metrics['pagerank'].get(node, 0) 
                                for node in path_nodes
                            ) / len(path_nodes)
                            
                            results[f"{source}->{target}"] = {
                                'path': path_nodes,
                                'length': path_length,
                                'score': path_score,
                                'path_names': [
                                    _self.nx_graph.nodes[node].get('name', node) 
                                    for node in path_nodes
                                ],
                                'path_types': [
                                    _self.nx_graph.nodes[node].get('type', 'unknown') 
                                    for node in path_nodes
                                ]
                            }
                            
                    except Exception as e:
                        logger.debug(f"No path found from {source} to {target}: {e}")
                        continue
            
            logger.info(f"Computed {len(results)} shortest paths")
            return results
            
        except Exception as e:
            logger.error(f"Shortest path computation failed: {e}")
            return {}
    
    def get_neighborhood_subgraph(self, nodes: List[str], radius: int = 2) -> nx.MultiDiGraph:
        """Get neighborhood subgraph around specified nodes."""
        try:
            all_neighbors = set(nodes)
            
            for node in nodes:
                if node not in self._node_mapping:
                    continue
                    
                node_idx = self._node_mapping[node]
                
                # Get neighbors within radius
                neighbors = self._ig_graph.neighborhood(node_idx, order=radius)
                neighbor_nodes = [self._reverse_mapping[idx] for idx in neighbors]
                all_neighbors.update(neighbor_nodes)
            
            # Create subgraph
            subgraph = self.nx_graph.subgraph(all_neighbors).copy()
            logger.info(f"Created neighborhood subgraph with {len(subgraph.nodes)} nodes")
            
            return subgraph
            
        except Exception as e:
            logger.error(f"Neighborhood subgraph creation failed: {e}")
            return nx.MultiDiGraph()
    
    def rank_nodes_by_importance(self, node_type: str = None, top_k: int = 20) -> List[Dict[str, Any]]:
        """Rank nodes by importance using multiple centrality measures."""
        try:
            centrality_metrics = self.compute_centrality_metrics()
            
            # Filter by node type if specified
            if node_type:
                nodes = [
                    node for node, data in self.nx_graph.nodes(data=True)
                    if data.get('type', '').lower() == node_type.lower()
                ]
            else:
                nodes = list(self.nx_graph.nodes())
            
            # Compute composite importance score
            ranked_nodes = []
            for node in nodes:
                if node in centrality_metrics['pagerank']:
                    composite_score = (
                        0.4 * centrality_metrics['pagerank'][node] +
                        0.3 * centrality_metrics['betweenness'][node] +
                        0.2 * centrality_metrics['closeness'][node] +
                        0.1 * centrality_metrics['degree'][node] / max(centrality_metrics['degree'].values())
                    )
                    
                    node_data = self.nx_graph.nodes[node]
                    ranked_nodes.append({
                        'id': node,
                        'name': node_data.get('name', node),
                        'type': node_data.get('type', 'unknown'),
                        'importance_score': composite_score,
                        'pagerank': centrality_metrics['pagerank'][node],
                        'betweenness': centrality_metrics['betweenness'][node],
                        'closeness': centrality_metrics['closeness'][node],
                        'degree': centrality_metrics['degree'][node]
                    })
            
            # Sort by importance score
            ranked_nodes.sort(key=lambda x: x['importance_score'], reverse=True)
            
            logger.info(f"Ranked {len(ranked_nodes)} nodes by importance")
            return ranked_nodes[:top_k]
            
        except Exception as e:
            logger.error(f"Node ranking failed: {e}")
            return []
