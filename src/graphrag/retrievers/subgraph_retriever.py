# src/graphrag/retrievers/subgraph_retriever.py

import networkx as nx
from typing import List, Set
import logging

class SubgraphRetriever:
    """Retrieve relevant subgraphs around entities of interest"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
    
    def get_entity_subgraph(self, entity_ids: List[str], hops: int = 2) -> nx.MultiDiGraph:
        """Get k-hop subgraph around specified entities"""
        if not entity_ids:
            return nx.MultiDiGraph()
            
        subgraph_nodes = set(entity_ids)
        
        # Expand k-hops from each seed entity
        for entity_id in entity_ids:
            if entity_id in self.graph:
                neighbors = self._get_k_hop_neighbors(entity_id, hops)
                subgraph_nodes.update(neighbors)
        
        return self.graph.subgraph(subgraph_nodes).copy()
    
    def _get_k_hop_neighbors(self, start_node: str, k: int) -> Set[str]:
        """Get all nodes within k hops of start_node"""
        if start_node not in self.graph:
            return set()
            
        neighbors = {start_node}
        current_frontier = {start_node}
        
        for _ in range(k):
            next_frontier = set()
            for node in current_frontier:
                next_frontier.update(self.graph.neighbors(node))
            neighbors.update(next_frontier)
            current_frontier = next_frontier
            
        return neighbors
    
    def get_connecting_subgraph(self, source_entities: List[str], 
                               target_entities: List[str], 
                               max_path_length: int = 4) -> nx.MultiDiGraph:
        """Get subgraph connecting two sets of entities"""
        connecting_nodes = set()
        
        # Find paths between source and target entities
        for source in source_entities:
            for target in target_entities:
                if source in self.graph and target in self.graph:
                    try:
                        paths = list(nx.all_shortest_paths(
                            self.graph, source, target
                        ))
                        for path in paths[:5]:  # Limit paths
                            if len(path) <= max_path_length:
                                connecting_nodes.update(path)
                    except nx.NetworkXNoPath:
                        continue
        
        if not connecting_nodes:
            # Fallback: get subgraphs around both sets
            all_entities = source_entities + target_entities
            return self.get_entity_subgraph(all_entities, hops=1)
            
        return self.graph.subgraph(connecting_nodes).copy()
