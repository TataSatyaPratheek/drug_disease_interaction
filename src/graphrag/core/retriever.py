import networkx as nx
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class GraphRetriever:
    """Enhanced retriever for drug-disease knowledge graphs"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def get_drug_disease_paths(self, drug_id: str, disease_id: str, max_paths: int = 5, max_length: int = 4) -> List[Dict]:
        """Find shortest paths between drug and disease entities"""
        if drug_id not in self.graph or disease_id not in self.graph:
            return []
        
        paths = []
        try:
            all_paths = list(nx.all_shortest_paths(self.graph, drug_id, disease_id))
            for path in all_paths[:max_paths]:
                if len(path) <= max_length:
                    path_info = self._analyze_path(path)
                    paths.append(path_info)
        except nx.NetworkXNoPath:
            logger.info(f"No direct path found between {drug_id} and {disease_id}")
        
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
            
            if i < len(path) - 1:
                next_node = path[i + 1]
                edge_data = self.graph.get_edge_data(node, next_node)
                if edge_data:
                    edge_info = list(edge_data.values())[0]
                    relation = edge_info.get('type', 'connected_to')
                    path_relations.append(relation)
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
    
    def get_disease_associated_drugs(self, disease_id: str, max_drugs: int = 10) -> List[Dict]:
        """Find drugs associated with a disease through direct or indirect connections"""
        if disease_id not in self.graph:
            return []
        
        associated_drugs = []
        visited = set()
        queue = [(disease_id, 0)]
        max_hops = 3
        
        while queue:
            node, hops = queue.pop(0)
            if hops > max_hops or node in visited:
                continue
            
            visited.add(node)
            node_data = self.graph.nodes[node]
            
            if node_data.get('type') == 'drug' and node != disease_id:
                associated_drugs.append({
                    'drug_id': node,
                    'drug_name': node_data.get('name', node),
                    'hops': hops
                })
            
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
        
        return sorted(associated_drugs, key=lambda x: x['hops'])[:max_drugs]
    
    def get_entity_neighbors(self, entity_id: str, max_neighbors: int = 10, edge_types: List[str] = None) -> List[Dict]:
        """Get immediate neighbors of an entity, optionally filtered by edge type"""
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if edge_data:
                for key, data in edge_data.items():
                    if edge_types is None or data.get('type') in edge_types:
                        neighbor_data = self.graph.nodes[neighbor]
                        neighbors.append({
                            'id': neighbor,
                            'name': neighbor_data.get('name', neighbor),
                            'type': neighbor_data.get('type', 'unknown'),
                            'edge_type': data.get('type', 'connected_to'),
                            'score': data.get('score', 1.0)
                        })
        
        return sorted(neighbors, key=lambda x: x['score'], reverse=True)[:max_neighbors]
    
    def get_related_entities(self, entity_id: str, entity_type: str, max_related: int = 10) -> List[Dict]:
        """Find entities related through shared connections"""
        if entity_id not in self.graph:
            return []
        
        entity_neighbors = set(self.graph.neighbors(entity_id))
        related_entities = []
        
        for node, data in self.graph.nodes(data=True):
            if node == entity_id or data.get('type') != entity_type:
                continue
            
            node_neighbors = set(self.graph.neighbors(node))
            intersection = len(entity_neighbors & node_neighbors)
            union = len(entity_neighbors | node_neighbors)
            
            if union > 0 and intersection > 0:
                similarity = intersection / union
                related_entities.append({
                    'id': node,
                    'name': data.get('name', node),
                    'similarity': similarity,
                    'shared_connections': intersection
                })
        
        return sorted(related_entities, key=lambda x: x['similarity'], reverse=True)[:max_related]
