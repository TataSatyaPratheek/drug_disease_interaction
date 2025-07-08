# src/graphrag/retrievers/community_retriever.py

import networkx as nx
from typing import List, Dict
import logging

class CommunityRetriever:
    """Retrieve entities based on graph communities and clustering"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        self._communities = None
    
    def get_entity_community(self, entity_id: str) -> List[str]:
        """Get community members for a given entity"""
        if entity_id not in self.graph:
            return []
            
        # Get 2-hop neighborhood as proxy for community
        neighbors = set([entity_id])
        
        # 1-hop
        first_hop = set(self.graph.neighbors(entity_id))
        neighbors.update(first_hop)
        
        # 2-hop (limited)
        for neighbor in list(first_hop)[:10]:  # Limit to avoid explosion
            second_hop = set(self.graph.neighbors(neighbor))
            neighbors.update(list(second_hop)[:5])  # Further limit
            
        return list(neighbors)
    
    def find_similar_entities(self, entity_id: str, 
                             entity_type: str = None, 
                             top_k: int = 10) -> List[Dict]:
        """Find entities similar based on shared neighbors"""
        if entity_id not in self.graph:
            return []
            
        entity_neighbors = set(self.graph.neighbors(entity_id))
        similar_entities = []
        
        for node, data in self.graph.nodes(data=True):
            if node == entity_id:
                continue
                
            # Filter by type if specified
            if entity_type and data.get('type') != entity_type:
                continue
                
            node_neighbors = set(self.graph.neighbors(node))
            
            # Calculate Jaccard similarity
            intersection = len(entity_neighbors & node_neighbors)
            union = len(entity_neighbors | node_neighbors)
            
            if union > 0 and intersection > 0:
                similarity = intersection / union
                similar_entities.append({
                    'id': node,
                    'name': data.get('name', node),
                    'type': data.get('type', 'unknown'),
                    'similarity': similarity,
                    'shared_neighbors': intersection
                })
        
        return sorted(similar_entities, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def get_functional_module(self, seed_entities: List[str], 
                             expand_steps: int = 2) -> List[str]:
        """Get functional module around seed entities"""
        module_nodes = set(seed_entities)
        
        for step in range(expand_steps):
            new_nodes = set()
            for node in module_nodes:
                if node in self.graph:
                    # Add highly connected neighbors
                    neighbors = list(self.graph.neighbors(node))
                    # Sort by degree and take top connected
                    neighbor_degrees = [(n, self.graph.degree(n)) for n in neighbors]
                    neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add top 3 most connected neighbors
                    for neighbor, _ in neighbor_degrees[:3]:
                        new_nodes.add(neighbor)
            
            module_nodes.update(new_nodes)
            
            # Limit growth
            if len(module_nodes) > 50:
                break
                
        return list(module_nodes)
