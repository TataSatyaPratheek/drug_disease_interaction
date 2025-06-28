# src/ddi/analysis/graph_analysis.py
import os
import logging
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import community as community_louvain  # python-louvain package for community detection

class GraphAnalyzer:
    """Analyzer for extracting insights from the drug-disease knowledge graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """Initialize the graph analyzer
        
        Args:
            graph: NetworkX MultiDiGraph to analyze
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph = graph
        
        # Create undirected view for some metrics
        self.undirected_graph = self.graph.to_undirected()
        
        # Cache for computationally intensive metrics
        self._cache = {}
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics
        
        Returns:
            Dictionary of graph statistics
        """
        if "basic_stats" in self._cache:
            return self._cache["basic_stats"]
            
        # Node type counts
        node_types = Counter([data.get("type", "unknown") for _, data in self.graph.nodes(data=True)])
        
        # Edge type counts
        edge_types = Counter([data.get("type", "unknown") for _, _, data in self.graph.edges(data=True)])
        
        # Degree statistics
        degrees = [d for _, d in self.graph.degree()]
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]
        
        # Connected components (using undirected graph)
        connected_components = list(nx.connected_components(self.undirected_graph))
        
        # Calculate statistics
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "degree_stats": {
                "min": min(degrees) if degrees else 0,
                "max": max(degrees) if degrees else 0,
                "mean": np.mean(degrees) if degrees else 0,
                "median": np.median(degrees) if degrees else 0
            },
            "in_degree_stats": {
                "min": min(in_degrees) if in_degrees else 0,
                "max": max(in_degrees) if in_degrees else 0,
                "mean": np.mean(in_degrees) if in_degrees else 0,
                "median": np.median(in_degrees) if in_degrees else 0
            },
            "out_degree_stats": {
                "min": min(out_degrees) if out_degrees else 0,
                "max": max(out_degrees) if out_degrees else 0,
                "mean": np.mean(out_degrees) if out_degrees else 0,
                "median": np.median(out_degrees) if out_degrees else 0
            },
            "density": nx.density(self.graph),
            "num_connected_components": len(connected_components),
            "largest_component_size": len(max(connected_components, key=len)) if connected_components else 0,
            "largest_component_percentage": len(max(connected_components, key=len)) / self.graph.number_of_nodes() * 100 if connected_components else 0
        }
        
        self._cache["basic_stats"] = stats
        return stats
    
    def get_degree_distribution(self, node_type: Optional[str] = None) -> Dict[str, List[int]]:
        """Get degree distribution for nodes
        
        Args:
            node_type: Filter by node type (optional)
            
        Returns:
            Dictionary with degree distributions
        """
        cache_key = f"degree_dist_{node_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Filter nodes by type if specified
        if node_type:
            nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == node_type]
            subgraph = self.graph.subgraph(nodes)
            degrees = [d for _, d in subgraph.degree()]
            in_degrees = [d for _, d in subgraph.in_degree()]
            out_degrees = [d for _, d in subgraph.out_degree()]
        else:
            degrees = [d for _, d in self.graph.degree()]
            in_degrees = [d for _, d in self.graph.in_degree()]
            out_degrees = [d for _, d in self.graph.out_degree()]
        
        distribution = {
            "degree": degrees,
            "in_degree": in_degrees,
            "out_degree": out_degrees
        }
        
        self._cache[cache_key] = distribution
        return distribution
    
    def calculate_centrality(self, centrality_type: str = "degree", node_types: Optional[List[str]] = None, 
                           top_n: int = 10, normalize: bool = True) -> pd.DataFrame:
        """Calculate centrality metrics for nodes
        
        Args:
            centrality_type: Type of centrality to calculate 
                             (degree, in_degree, out_degree, betweenness, eigenvector, pagerank)
            node_types: List of node types to include (optional)
            top_n: Number of top nodes to return
            normalize: Whether to normalize centrality values
            
        Returns:
            DataFrame with centrality scores
        """
        cache_key = f"centrality_{centrality_type}_{str(node_types)}_{normalize}"
        if cache_key in self._cache:
            df = self._cache[cache_key]
            return df.head(top_n) if top_n > 0 else df
        
        # Filter graph by node types if specified
        if node_types:
            nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") in node_types]
            graph = self.graph.subgraph(nodes)
        else:
            graph = self.graph
        
        # Calculate centrality based on type
        if centrality_type == "degree":
            centrality = dict(graph.degree())
        elif centrality_type == "in_degree":
            centrality = dict(graph.in_degree())
        elif centrality_type == "out_degree":
            centrality = dict(graph.out_degree())
        elif centrality_type == "betweenness":
            # Use undirected graph for betweenness to avoid disconnected nodes
            undirected = graph.to_undirected()
            centrality = nx.betweenness_centrality(undirected, normalized=normalize)
        elif centrality_type == "eigenvector":
            try:
                # Use undirected graph for eigenvector centrality
                undirected = graph.to_undirected()
                centrality = nx.eigenvector_centrality(undirected, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                self.logger.warning("Eigenvector centrality failed to converge. Using approximate method.")
                undirected = graph.to_undirected()
                centrality = nx.eigenvector_centrality_numpy(undirected)
        elif centrality_type == "pagerank":
            centrality = nx.pagerank(graph)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
        
        # Create DataFrame with results
        data = []
        for node, score in centrality.items():
            node_data = graph.nodes[node]
            data.append({
                "node_id": node,
                "name": node_data.get("name", node),
                "type": node_data.get("type", "unknown"),
                "score": score
            })
        
        df = pd.DataFrame(data)
        
        # Sort by score in descending order
        df = df.sort_values("score", ascending=False)
        
        self._cache[cache_key] = df
        return df.head(top_n) if top_n > 0 else df
    
    def find_shortest_paths(self, source_type: str, target_type: str, 
                          max_paths: int = 10, cutoff: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find shortest paths between node types
        
        Args:
            source_type: Source node type
            target_type: Target node type
            max_paths: Maximum number of paths to return
            cutoff: Maximum path length to consider (optional)
            
        Returns:
            List of paths with metadata
        """
        # Get nodes of specified types
        source_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == source_type]
        target_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == target_type]
        
        if not source_nodes or not target_nodes:
            self.logger.warning(f"No nodes found for types: {source_type} -> {target_type}")
            return []
        
        # Sample nodes if there are too many
        max_combinations = 1000
        if len(source_nodes) * len(target_nodes) > max_combinations:
            self.logger.info(f"Too many node combinations. Sampling {max_combinations} pairs.")
            
            if len(source_nodes) > len(target_nodes):
                source_nodes = np.random.choice(source_nodes, 
                                              size=max(1, max_combinations // len(target_nodes)),
                                              replace=False).tolist()
            else:
                target_nodes = np.random.choice(target_nodes, 
                                              size=max(1, max_combinations // len(source_nodes)),
                                              replace=False).tolist()
        
        paths = []
        path_count = 0
        
        for source in source_nodes:
            if path_count >= max_paths:
                break
                
            for target in target_nodes:
                if source == target:
                    continue
                    
                if path_count >= max_paths:
                    break
                
                                # Inside the loop for source and target
                try:
                    # Check length against cutoff if provided BEFORE finding the path
                    if cutoff is not None:
                        try:
                            length = nx.shortest_path_length(self.graph, source=source, target=target)
                            if length > cutoff:
                                continue # Skip if path is longer than cutoff
                        except nx.NetworkXNoPath:
                            continue # Skip if no path exists

                    # Try to find the shortest path (now without cutoff argument)
                    path = nx.shortest_path(self.graph, source=source, target=target)

                    if path:
                        # Get path metadata
                        path_data = {
                            "source_id": source,
                            "source_name": self.graph.nodes[source].get("name", source),
                            "target_id": target,
                            "target_name": self.graph.nodes[target].get("name", target),
                            "length": len(path) - 1, # Use actual path length
                            "path": path,
                            "path_names": [self.graph.nodes[n].get("name", n) for n in path],
                            "path_types": [self.graph.nodes[n].get("type", "unknown") for n in path],
                            "edges": []
                        }

                        # Get edge data
                        for i in range(len(path) - 1):
                            source_node = path[i]
                            target_node = path[i + 1]

                            # Get all edges between these nodes
                            edge_data_list = [] # Renamed to avoid conflict
                            # Correctly iterate through edges for MultiDiGraph
                            if self.graph.has_edge(source_node, target_node):
                                for key, data in self.graph[source_node][target_node].items():
                                    edge_data_list.append(data)

                            # Add edge data to path
                            path_data["edges"].append(edge_data_list) # Use the renamed list

                        paths.append(path_data)
                        path_count += 1

                except nx.NetworkXNoPath:
                    # This handles cases where shortest_path_length might pass but shortest_path fails (unlikely)
                    # or if cutoff was None and no path exists
                    continue
                except Exception as e: # Catch potential errors during path finding
                    self.logger.warning(f"Error finding path between {source} and {target}: {e}")
                    continue
        
        # Sort paths by length
        paths.sort(key=lambda x: x["length"])
        
        return paths
    
    def find_common_neighbors(self, node_type_a: str, node_type_b: str, 
                             min_neighbors: int = 3, max_results: int = 100) -> List[Dict[str, Any]]:
        """Find node pairs with common neighbors
        
        Args:
            node_type_a: First node type
            node_type_b: Second node type
            min_neighbors: Minimum number of common neighbors
            max_results: Maximum number of results to return
            
        Returns:
            List of node pairs with common neighbors
        """
        # Get nodes of specified types
        nodes_a = [n for n, d in self.graph.nodes(data=True) if d.get("type") == node_type_a]
        nodes_b = [n for n, d in self.graph.nodes(data=True) if d.get("type") == node_type_b]
        
        if not nodes_a or not nodes_b:
            self.logger.warning(f"No nodes found for types: {node_type_a} or {node_type_b}")
            return []
        
        # Use undirected graph for finding common neighbors
        undirected = self.graph.to_undirected()
        
        results = []
        
        # Sample nodes if there are too many
        max_combinations = 10000
        if len(nodes_a) * len(nodes_b) > max_combinations:
            self.logger.info(f"Too many node combinations. Sampling {max_combinations} pairs.")
            
            if len(nodes_a) > len(nodes_b):
                nodes_a = np.random.choice(nodes_a, 
                                          size=max(1, max_combinations // len(nodes_b)),
                                          replace=False).tolist()
            else:
                nodes_b = np.random.choice(nodes_b, 
                                          size=max(1, max_combinations // len(nodes_a)),
                                          replace=False).tolist()
        
        for node_a in nodes_a:
            for node_b in nodes_b:
                if node_a == node_b:
                    continue
                
                # Find common neighbors
                neighbors_a = set(undirected.neighbors(node_a))
                neighbors_b = set(undirected.neighbors(node_b))
                common = neighbors_a.intersection(neighbors_b)
                
                if len(common) >= min_neighbors:
                    result = {
                        "node_a_id": node_a,
                        "node_a_name": self.graph.nodes[node_a].get("name", node_a),
                        "node_a_type": self.graph.nodes[node_a].get("type", "unknown"),
                        "node_b_id": node_b,
                        "node_b_name": self.graph.nodes[node_b].get("name", node_b),
                        "node_b_type": self.graph.nodes[node_b].get("type", "unknown"),
                        "common_neighbors_count": len(common),
                        "common_neighbors": [
                            {
                                "id": n,
                                "name": self.graph.nodes[n].get("name", n),
                                "type": self.graph.nodes[n].get("type", "unknown")
                            }
                            for n in common
                        ]
                    }
                    
                    results.append(result)
                    
                    if len(results) >= max_results:
                        # Sort by number of common neighbors
                        results.sort(key=lambda x: x["common_neighbors_count"], reverse=True)
                        return results
        
        # Sort by number of common neighbors
        results.sort(key=lambda x: x["common_neighbors_count"], reverse=True)
        return results
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[str, Any]:
        """Detect communities in the graph using Louvain algorithm
        
        Args:
            resolution: Resolution parameter for community detection
            
        Returns:
            Dictionary with community detection results
        """
        cache_key = f"communities_{resolution}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Use undirected graph for community detection
        undirected = self.graph.to_undirected()
        
        # Convert to a simple graph (no multiedges)
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(undirected.nodes(data=True))
        for u, v in undirected.edges():
            if simple_graph.has_edge(u, v):
                continue
            simple_graph.add_edge(u, v)
        
        # Detect communities
        partition = community_louvain.best_partition(simple_graph, resolution=resolution)
        
        # Group nodes by community
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        # Calculate community statistics
        community_stats = []
        for community_id, nodes in communities.items():
            # Count node types in community
            node_types = Counter([self.graph.nodes[n].get("type", "unknown") for n in nodes])
            
            # Create subgraph
            subgraph = self.graph.subgraph(nodes)
            
            stats = {
                "community_id": community_id,
                "size": len(nodes),
                "percentage": len(nodes) / self.graph.number_of_nodes() * 100,
                "node_types": dict(node_types),
                "density": nx.density(subgraph),
                "key_nodes": self._get_key_nodes_in_subgraph(subgraph, top_n=5)
            }
            
            community_stats.append(stats)
        
        # Sort communities by size
        community_stats.sort(key=lambda x: x["size"], reverse=True)
        
        result = {
            "num_communities": len(communities),
            "modularity": community_louvain.modularity(partition, simple_graph),
            "communities": community_stats,
            "node_to_community": partition
        }
        
        self._cache[cache_key] = result
        return result
    
    def _get_key_nodes_in_subgraph(self, subgraph: nx.Graph, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get key nodes in a subgraph based on centrality
        
        Args:
            subgraph: NetworkX Graph or subgraph
            top_n: Number of top nodes to return
            
        Returns:
            List of key nodes with metadata
        """
        # Calculate degree centrality
        centrality = nx.degree_centrality(subgraph)
        
        # Get top nodes
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create result
        result = []
        for node, score in top_nodes:
            node_data = subgraph.nodes[node]
            result.append({
                "id": node,
                "name": node_data.get("name", node),
                "type": node_data.get("type", "unknown"),
                "centrality": score
            })
        
        return result
    
    def extract_subgraph(self, node_types: Optional[List[str]] = None, 
                        edge_types: Optional[List[str]] = None,
                        max_nodes: Optional[int] = None) -> nx.MultiDiGraph:
        """Extract a subgraph based on node and edge types
        
        Args:
            node_types: List of node types to include (optional)
            edge_types: List of edge types to include (optional)
            max_nodes: Maximum number of nodes in the subgraph (optional)
            
        Returns:
            NetworkX MultiDiGraph subgraph
        """
        # Filter nodes by type if specified
        if node_types:
            nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") in node_types]
            
            # Limit number of nodes if specified
            if max_nodes and len(nodes) > max_nodes:
                self.logger.info(f"Limiting subgraph to {max_nodes} nodes (from {len(nodes)})")
                nodes = np.random.choice(nodes, size=max_nodes, replace=False).tolist()
                
            subgraph = self.graph.subgraph(nodes).copy()
        else:
            subgraph = self.graph.copy()
        
        # Filter edges by type if specified
        if edge_types:
            edges_to_remove = []
            for u, v, k, d in subgraph.edges(data=True, keys=True):
                if d.get("type") not in edge_types:
                    edges_to_remove.append((u, v, k))
            
            for u, v, k in edges_to_remove:
                subgraph.remove_edge(u, v, k)
        
        return subgraph
    
    def get_entity_neighborhood(self, entity_id: str, hops: int = 1, 
                              max_nodes: Optional[int] = None) -> nx.MultiDiGraph:
        """Get the neighborhood of an entity
        
        Args:
            entity_id: ID of the entity
            hops: Number of hops from the entity
            max_nodes: Maximum number of nodes in the neighborhood (optional)
            
        Returns:
            NetworkX MultiDiGraph neighborhood subgraph
        """
        if entity_id not in self.graph:
            self.logger.warning(f"Entity {entity_id} not found in graph")
            return nx.MultiDiGraph()
        
        # Get nodes within n hops
        neighborhood = set([entity_id])
        current_shell = set([entity_id])
        
        for _ in range(hops):
            next_shell = set()
            for node in current_shell:
                # Add neighbors
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_shell.update(neighbors)
            
            # Update neighborhood
            neighborhood.update(next_shell)
            current_shell = next_shell
            
            # Check size limit
            if max_nodes and len(neighborhood) >= max_nodes:
                self.logger.info(f"Limiting neighborhood to {max_nodes} nodes")
                neighborhood = set(list(neighborhood)[:max_nodes])
                break
        
        # Create subgraph
        return self.graph.subgraph(neighborhood).copy()
    
    def find_drug_disease_paths(self, drug_id: str, disease_id: str, 
                              max_paths: int = 10) -> List[Dict[str, Any]]:
        """Find all paths between a drug and a disease
        
        Args:
            drug_id: Drug ID
            disease_id: Disease ID
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths with metadata
        """
        if drug_id not in self.graph:
            self.logger.warning(f"Drug {drug_id} not found in graph")
            return []
            
        if disease_id not in self.graph:
            self.logger.warning(f"Disease {disease_id} not found in graph")
            return []
        
        # Find all simple paths
        paths = []
        try:
            # Try to find paths using all_simple_paths (with length limit)
            all_paths = list(nx.all_simple_paths(self.graph, source=drug_id, target=disease_id, cutoff=4))
            
            # Sort paths by length
            all_paths.sort(key=len)
            
            # Take the top paths
            top_paths = all_paths[:max_paths]
            
            for path in top_paths:
                # Get path metadata
                path_data = {
                    "drug_id": drug_id,
                    "drug_name": self.graph.nodes[drug_id].get("name", drug_id),
                    "disease_id": disease_id,
                    "disease_name": self.graph.nodes[disease_id].get("name", disease_id),
                    "length": len(path) - 1,
                    "path": path,
                    "path_names": [self.graph.nodes[n].get("name", n) for n in path],
                    "path_types": [self.graph.nodes[n].get("type", "unknown") for n in path],
                    "edges": []
                }
                
                # Get edge data
                for i in range(len(path) - 1):
                    source_node = path[i]
                    target_node = path[i + 1]
                    
                    # Get all edges between these nodes
                    edge_data = []
                    for _, _, k, data in self.graph.edges(nbunch=[source_node], keys=True, data=True):
                        if k == target_node:  # Check if this edge connects to the target node
                            edge_data.append(data)
                    
                    # Add edge data to path
                    path_data["edges"].append(edge_data)
                
                paths.append(path_data)
                
        except nx.NetworkXNoPath:
            self.logger.warning(f"No path found between drug {drug_id} and disease {disease_id}")
        
        return paths