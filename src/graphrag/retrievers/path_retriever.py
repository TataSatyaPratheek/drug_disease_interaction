# src/graphrag/retrievers/path_retriever.py

import networkx as nx
from typing import List, Dict
import logging

class PathRetriever:
    """Retrieve meaningful paths between entities"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
    
    def find_drug_disease_paths(self, drug_id: str, disease_id: str, 
                               max_paths: int = 5, max_length: int = 4) -> List[Dict]:
        """Find paths between drug and disease entities"""
        if drug_id not in self.graph or disease_id not in self.graph:
            return []
            
        paths = []
        try:
            # Get shortest paths
            all_paths = list(nx.all_shortest_paths(self.graph, drug_id, disease_id))
            
            for path in all_paths[:max_paths]:
                if len(path) <= max_length:
                    path_info = self._analyze_path(path)
                    paths.append(path_info)
                    
        except nx.NetworkXNoPath:
            self.logger.info(f"No direct path found between {drug_id} and {disease_id}")
            
        return sorted(paths, key=lambda x: x['score'], reverse=True)
    
    def _analyze_path(self, path: List[str]) -> Dict:
        """Analyze a path and extract meaningful information"""
        path_names = []
        path_types = []
        path_relations = []
        score = 0
        
        for i, node in enumerate(path):
            node_data = self.graph.nodes[node]
            path_names.append(node_data.get('name', node))
            path_types.append(node_data.get('type', 'unknown'))
            
            # Get edge information
            if i < len(path) - 1:
                next_node = path[i + 1]
                edge_data = self.graph.get_edge_data(node, next_node)
                if edge_data:
                    # Get first edge (in case of multiple)
                    edge_info = list(edge_data.values())[0]
                    relation = edge_info.get('type', 'connected_to')
                    path_relations.append(relation)
                    
                    # Simple scoring
                    if relation in ['targets', 'treats', 'associated_with']:
                        score += 1
        
        return {
            'path': path,
            'path_names': path_names,
            'path_types': path_types,
            'path_relations': path_relations,
            'length': len(path),
            'score': score
        }
    
    def find_multi_hop_connections(self, entity_id: str, 
                                  target_type: str, 
                                  max_hops: int = 3) -> List[Dict]:
        """Find entities of target_type within max_hops"""
        if entity_id not in self.graph:
            return []
            
        connections = []
        visited = set()
        queue = [(entity_id, [])]
        
        while queue:
            current_node, path = queue.pop(0)
            
            if len(path) >= max_hops:
                continue
                
            if current_node in visited:
                continue
                
            visited.add(current_node)
            current_data = self.graph.nodes[current_node]
            
            # Check if this is a target
            if current_data.get('type') == target_type and current_node != entity_id:
                connections.append({
                    'target_id': current_node,
                    'target_name': current_data.get('name', current_node),
                    'path': path + [current_node],
                    'hops': len(path)
                })
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [current_node]))
        
        return sorted(connections, key=lambda x: x['hops'])[:10]
