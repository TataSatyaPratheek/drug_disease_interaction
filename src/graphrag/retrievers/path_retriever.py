import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import concurrent.futures
from collections import defaultdict
from ..core.graph_analytics import HighPerformanceGraphAnalytics
from ..core.connection_resilience import ConnectionResilience
import streamlit as st

logger = logging.getLogger(__name__)

class PathRetriever:
    """Retrieves and analyzes paths between entities with intelligent scoring and performance optimization."""
    
    def __init__(self, graph: nx.MultiDiGraph, max_workers: int = 4):
        self.graph = graph
        self.analytics = HighPerformanceGraphAnalytics(graph)
        self.resilience = ConnectionResilience()
        self.max_workers = max_workers
        
        # Pre-compute node metadata for faster access
        self._node_types = {node: data.get('type', 'unknown') 
                           for node, data in graph.nodes(data=True)}
        self._node_names = {node: data.get('name', node) 
                           for node, data in graph.nodes(data=True)}
        
        # Thread pool for parallel operations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache for centrality metrics
        self._centrality_cache = None

    @ConnectionResilience.with_retry(max_attempts=3, wait_seconds=1)
    def find_drug_disease_paths(self, drug_id: str, disease_id: str, max_paths: int = 5) -> List[Dict[str, Any]]:
        """Find paths between drug and disease with intelligent scoring and parallel processing."""
        try:
            if drug_id not in self.graph or disease_id not in self.graph:
                logger.warning(f"Node not found: {drug_id} or {disease_id}")
                return []

            # Use parallel path finding for better performance
            with self._executor:
                # Submit primary path and alternative paths concurrently
                primary_future = self._executor.submit(
                    self._find_primary_path, drug_id, disease_id
                )
                alternatives_future = self._executor.submit(
                    self._find_alternative_paths_parallel, drug_id, disease_id, max_paths - 1
                )
                
                # Get results
                primary_path = primary_future.result(timeout=10)
                alternative_paths = alternatives_future.result(timeout=15)

            if not primary_path and not alternative_paths:
                logger.info(f"No paths found between {drug_id} and {disease_id}")
                return []

            # Combine all paths
            all_paths = []
            if primary_path:
                all_paths.append(primary_path)
            all_paths.extend(alternative_paths)

            # Enhanced parallel scoring of paths
            scored_paths = self._score_paths_parallel(all_paths)

            # Sort by biological relevance score
            scored_paths.sort(key=lambda x: x['biological_score'], reverse=True)
            logger.info(f"Found {len(scored_paths)} scored paths between {drug_id} and {disease_id}")
            return scored_paths[:max_paths]
            
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return []
    
    def _find_primary_path(self, drug_id: str, disease_id: str) -> Optional[Dict[str, Any]]:
        """Find primary shortest path between drug and disease"""
        try:
            path_data = self.analytics.compute_shortest_paths([drug_id], [disease_id])
            primary_path_key = f"{drug_id}->{disease_id}"
            
            if primary_path_key in path_data:
                return path_data[primary_path_key]
            return None
            
        except Exception as e:
            logger.error(f"Primary path finding failed: {e}")
            return None
    
    def _score_paths_parallel(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score multiple paths in parallel for better performance"""
        if not paths:
            return []
        
        # For small numbers of paths, use sequential processing
        if len(paths) <= 3:
            return [self._enhance_path_with_scoring(path) for path in paths]
        
        # Parallel scoring for larger path sets
        with self._executor:
            futures = [
                self._executor.submit(self._enhance_path_with_scoring, path)
                for path in paths
            ]
            
            scored_paths = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    scored_path = future.result(timeout=8)
                    scored_paths.append(scored_path)
                except Exception as e:
                    logger.warning(f"Path scoring failed: {e}")
        
        return scored_paths

    def _find_alternative_paths_parallel(self, drug_id: str, disease_id: str, max_alternative: int) -> List[Dict[str, Any]]:
        """Find alternative paths through different intermediate nodes with parallel processing."""
        try:
            # Get or compute centrality metrics with caching
            if self._centrality_cache is None:
                self._centrality_cache = self.analytics.compute_centrality_metrics()
            
            # Get high-centrality intermediate nodes
            high_centrality_nodes = sorted(
                self._centrality_cache['betweenness'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:30]  # Top 30 most central nodes for better alternatives
            
            # Filter out source and target nodes
            intermediate_candidates = [
                node for node, _ in high_centrality_nodes 
                if node not in [drug_id, disease_id]
            ][:20]  # Limit to top 20 candidates
            
            # Process intermediate nodes in parallel
            chunk_size = max(1, len(intermediate_candidates) // self.max_workers)
            node_chunks = [
                intermediate_candidates[i:i + chunk_size] 
                for i in range(0, len(intermediate_candidates), chunk_size)
            ]
            
            alternative_paths = []
            
            with self._executor:
                futures = [
                    self._executor.submit(
                        self._find_paths_through_intermediates, 
                        drug_id, disease_id, chunk
                    )
                    for chunk in node_chunks
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_paths = future.result(timeout=12)
                        alternative_paths.extend(chunk_paths)
                        
                        # Early termination if we have enough paths
                        if len(alternative_paths) >= max_alternative:
                            break
                    except Exception as e:
                        logger.warning(f"Alternative path chunk processing failed: {e}")
            
            return alternative_paths[:max_alternative]
            
        except Exception as e:
            logger.error(f"Alternative path finding failed: {e}")
            return []
    
    def _find_paths_through_intermediates(self, drug_id: str, disease_id: str, 
                                        intermediate_nodes: List[str]) -> List[Dict[str, Any]]:
        """Find paths through a set of intermediate nodes"""
        paths = []
        
        for intermediate_node in intermediate_nodes:
            try:
                # Find path through intermediate node
                path1_data = self.analytics.compute_shortest_paths([drug_id], [intermediate_node])
                path2_data = self.analytics.compute_shortest_paths([intermediate_node], [disease_id])

                path1_key = f"{drug_id}->{intermediate_node}"
                path2_key = f"{intermediate_node}->{disease_id}"

                if path1_key in path1_data and path2_key in path2_data:
                    # Combine paths
                    combined_path = path1_data[path1_key]['path'][:-1] + path2_data[path2_key]['path']

                    # Use cached node data for faster access
                    alternative_path = {
                        'path': combined_path,
                        'length': len(combined_path) - 1,
                        'path_names': [self._node_names.get(node, node) for node in combined_path],
                        'path_types': [self._node_types.get(node, 'unknown') for node in combined_path],
                        'score': (path1_data[path1_key]['score'] + path2_data[path2_key]['score']) / 2,
                        'intermediate_node': intermediate_node
                    }
                    paths.append(alternative_path)

            except Exception:
                continue
        
        return paths

    def _enhance_path_with_scoring(self, path: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance path with biological relevance scoring."""
        try:
            path_nodes = path['path']
            path_types = path['path_types']

            # Calculate biological relevance score
            biological_score = self._calculate_biological_relevance(path_nodes, path_types)

            # Calculate confidence score
            confidence_score = self._calculate_path_confidence(path_nodes)

            # Add pathway information
            pathway_info = self._extract_pathway_info(path_nodes)

            enhanced_path = {
                **path,
                'biological_score': biological_score,
                'confidence_score': confidence_score,
                'pathway_info': pathway_info,
                'mechanism_description': self._generate_mechanism_description(path_nodes, path_types)
            }

            return enhanced_path

        except Exception as e:
            logger.error(f"Path enhancement failed: {e}")
            return path

    def _calculate_biological_relevance(self, path_nodes: List[str], path_types: List[str]) -> float:
        """Calculate biological relevance score for a path."""
        try:
            # Weight different edge types
            type_weights = {
                'drug': 0.2,
                'protein': 0.8,  # High importance for proteins
                'disease': 0.2,
                'polypeptide': 0.7,
                'compound': 0.3,
                'unknown': 0.1
            }

            # Calculate weighted score
            total_weight = sum(type_weights.get(node_type, 0.1) for node_type in path_types)
            normalized_score = total_weight / len(path_types) if path_types else 0.0

            # Penalty for very long paths
            length_penalty = 1.0 / (1.0 + (len(path_nodes) - 3) * 0.1)

            return normalized_score * length_penalty

        except Exception as e:
            logger.error(f"Biological relevance calculation failed: {e}")
            return 0.0

    def _calculate_path_confidence(self, path_nodes: List[str]) -> float:
        """Calculate confidence score based on node centrality."""
        try:
            centrality_metrics = self.analytics.compute_centrality_metrics()

            # Average centrality of intermediate nodes
            intermediate_nodes = path_nodes[1:-1]  # Exclude start and end
            if not intermediate_nodes:
                return 0.5

            avg_centrality = sum(
                centrality_metrics['pagerank'].get(node, 0)
                for node in intermediate_nodes
            ) / len(intermediate_nodes)

            return min(avg_centrality * 10, 1.0)  # Scale and cap at 1.0

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0

    def _extract_pathway_info(self, path_nodes: List[str]) -> Dict[str, Any]:
        """Extract pathway information from path nodes."""
        try:
            pathway_info = {
                'drug_targets': [],
                'protein_interactions': [],
                'disease_associations': []
            }

            for i, node in enumerate(path_nodes):
                node_data = self.graph.nodes.get(node, {})
                node_type = node_data.get('type', 'unknown')

                if node_type == 'drug' and i < len(path_nodes) - 1:
                    next_node = path_nodes[i + 1]
                    edge_data = self.graph.get_edge_data(node, next_node)
                    if edge_data:
                        action = list(edge_data.values())[0].get('actions', '')
                        pathway_info['drug_targets'].append({
                            'drug': node_data.get('name', node),
                            'target': self.graph.nodes[next_node].get('name', next_node),
                            'action': action
                        })

            return pathway_info

        except Exception as e:
            logger.error(f"Pathway info extraction failed: {e}")
            return {}

    def _generate_mechanism_description(self, path_nodes: List[str], path_types: List[str]) -> str:
        """Generate human-readable mechanism description."""
        try:
            if len(path_nodes) < 2:
                return "Invalid path"

            descriptions = []

            for i in range(len(path_nodes) - 1):
                current_node = path_nodes[i]
                next_node = path_nodes[i + 1]

                current_name = self.graph.nodes[current_node].get('name', current_node)
                next_name = self.graph.nodes[next_node].get('name', next_node)

                # Get edge information
                edge_data = self.graph.get_edge_data(current_node, next_node)
                if edge_data:
                    edge_type = list(edge_data.values())[0].get('type', 'interacts with')
                    descriptions.append(f"{current_name} {edge_type} {next_name}")
                else:
                    descriptions.append(f"{current_name} connects to {next_name}")

            return " → ".join(descriptions)

        except Exception as e:
            logger.error(f"Mechanism description generation failed: {e}")
            return "Unable to generate mechanism description"
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the path retriever for performance monitoring"""
        return {
            "graph_nodes": len(self.graph.nodes()),
            "graph_edges": len(self.graph.edges()),
            "max_workers": self.max_workers,
            "cached_centrality": self._centrality_cache is not None,
            "node_types_cached": len(self._node_types),
            "node_names_cached": len(self._node_names)
        }
    
    def clear_cache(self):
        """Clear cached data to free memory"""
        self._centrality_cache = None
        logger.info("PathRetriever cache cleared")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)