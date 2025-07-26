# src/graphrag/retrievers/subgraph_retriever.py

import networkx as nx
from typing import List, Set, Dict, Any
import logging
import concurrent.futures
from collections import deque

class SubgraphRetriever:
    """Retrieve relevant subgraphs around entities of interest with performance optimizations"""
    
    def __init__(self, graph: nx.MultiDiGraph, max_workers: int = 4):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        
        # Pre-compute node mappings for faster lookups
        self._node_cache = set(self.graph.nodes())
        self._edge_cache = dict(self.graph.adjacency())
        
        # Thread pool for parallel operations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def get_entity_subgraph(self, entity_ids: List[str], hops: int = 2) -> nx.MultiDiGraph:
        """Get k-hop subgraph around specified entities with parallel processing"""
        if not entity_ids:
            return nx.MultiDiGraph()
        
        # Filter valid entity IDs upfront
        valid_entities = [eid for eid in entity_ids if eid in self._node_cache]
        if not valid_entities:
            return nx.MultiDiGraph()
        
        # Process entities in parallel if there are many
        if len(valid_entities) > 4:
            return self._get_subgraph_parallel(valid_entities, hops)
        else:
            return self._get_subgraph_sequential(valid_entities, hops)
    
    def _get_subgraph_parallel(self, entity_ids: List[str], hops: int) -> nx.MultiDiGraph:
        """Parallel subgraph expansion for multiple entities"""
        chunk_size = max(1, len(entity_ids) // self.max_workers)
        entity_chunks = [entity_ids[i:i + chunk_size] for i in range(0, len(entity_ids), chunk_size)]
        
        subgraph_nodes = set(entity_ids)
        
        with self._executor:
            # Submit parallel neighbor finding tasks
            futures = [
                self._executor.submit(self._get_neighbors_for_chunk, chunk, hops)
                for chunk in entity_chunks
            ]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_neighbors = future.result(timeout=10)
                    subgraph_nodes.update(chunk_neighbors)
                except Exception as e:
                    self.logger.warning(f"Parallel neighbor retrieval failed: {e}")
        
        return self.graph.subgraph(subgraph_nodes).copy()
    
    def _get_subgraph_sequential(self, entity_ids: List[str], hops: int) -> nx.MultiDiGraph:
        """Sequential subgraph expansion for smaller entity sets"""
        subgraph_nodes = set(entity_ids)
        
        for entity_id in entity_ids:
            neighbors = self._get_k_hop_neighbors_fast(entity_id, hops)
            subgraph_nodes.update(neighbors)
        
        return self.graph.subgraph(subgraph_nodes).copy()
    
    def _get_neighbors_for_chunk(self, entity_chunk: List[str], hops: int) -> Set[str]:
        """Get neighbors for a chunk of entities"""
        all_neighbors = set()
        for entity_id in entity_chunk:
            neighbors = self._get_k_hop_neighbors_fast(entity_id, hops)
            all_neighbors.update(neighbors)
        return all_neighbors
    
    def _get_k_hop_neighbors_fast(self, start_node: str, k: int) -> Set[str]:
        """Optimized k-hop neighbor finding using BFS with deque"""
        if start_node not in self._node_cache:
            return set()
        
        neighbors = {start_node}
        queue = deque([(start_node, 0)])  # (node, distance)
        visited = {start_node}
        
        while queue:
            current_node, distance = queue.popleft()
            
            if distance >= k:
                continue
            
            # Use cached adjacency for faster neighbor lookup
            if current_node in self._edge_cache:
                for neighbor in self._edge_cache[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        neighbors.add(neighbor)
                        queue.append((neighbor, distance + 1))
        
        return neighbors
    
    def get_connecting_subgraph(self, source_entities: List[str], 
                               target_entities: List[str], 
                               max_path_length: int = 4) -> nx.MultiDiGraph:
        """Get subgraph connecting two sets of entities with optimized pathfinding"""
        connecting_nodes = set()
        
        # Filter valid entities
        valid_sources = [e for e in source_entities if e in self._node_cache]
        valid_targets = [e for e in target_entities if e in self._node_cache]
        
        if not valid_sources or not valid_targets:
            # Fallback: get subgraphs around both sets
            all_entities = valid_sources + valid_targets
            return self.get_entity_subgraph(all_entities, hops=1)
        
        # Parallel path finding for better performance
        if len(valid_sources) * len(valid_targets) > 16:
            connecting_nodes = self._find_paths_parallel(
                valid_sources, valid_targets, max_path_length
            )
        else:
            connecting_nodes = self._find_paths_sequential(
                valid_sources, valid_targets, max_path_length
            )
        
        if not connecting_nodes:
            # Fallback: get subgraphs around both sets
            all_entities = valid_sources + valid_targets
            return self.get_entity_subgraph(all_entities, hops=1)
            
        return self.graph.subgraph(connecting_nodes).copy()
    
    def _find_paths_parallel(self, sources: List[str], targets: List[str], 
                           max_length: int) -> Set[str]:
        """Find connecting paths in parallel"""
        connecting_nodes = set()
        
        # Create source-target pairs for parallel processing
        pairs = [(s, t) for s in sources for t in targets]
        chunk_size = max(1, len(pairs) // self.max_workers)
        pair_chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        
        with self._executor:
            futures = [
                self._executor.submit(self._find_paths_for_pairs, chunk, max_length)
                for chunk in pair_chunks
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_paths = future.result(timeout=15)
                    connecting_nodes.update(chunk_paths)
                except Exception as e:
                    self.logger.warning(f"Parallel path finding failed: {e}")
        
        return connecting_nodes
    
    def _find_paths_sequential(self, sources: List[str], targets: List[str], 
                             max_length: int) -> Set[str]:
        """Find connecting paths sequentially"""
        connecting_nodes = set()
        
        for source in sources:
            for target in targets:
                try:
                    # Use bidirectional search for better performance
                    if nx.has_path(self.graph, source, target):
                        path = nx.shortest_path(self.graph, source, target)
                        if len(path) <= max_length:
                            connecting_nodes.update(path)
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
                    
                # Limit computation time
                if len(connecting_nodes) > 1000:
                    break
            
            if len(connecting_nodes) > 1000:
                break
        
        return connecting_nodes
    
    def _find_paths_for_pairs(self, pairs: List[tuple], max_length: int) -> Set[str]:
        """Find paths for a set of source-target pairs"""
        connecting_nodes = set()
        
        for source, target in pairs:
            try:
                if nx.has_path(self.graph, source, target):
                    path = nx.shortest_path(self.graph, source, target)
                    if len(path) <= max_length:
                        connecting_nodes.update(path)
            except (nx.NetworkXNoPath, nx.NetworkXError):
                continue
            
            # Prevent excessive computation
            if len(connecting_nodes) > 500:
                break
        
        return connecting_nodes
    
    def get_subgraph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph for optimization insights"""
        return {
            "total_nodes": len(self._node_cache),
            "total_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "is_multigraph": self.graph.is_multigraph(),
            "density": nx.density(self.graph),
            "max_workers": self.max_workers
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
