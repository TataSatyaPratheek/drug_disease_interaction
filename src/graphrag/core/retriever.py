# src/graphrag/core/retriever.py
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set

class GraphRetriever:
    """Simplified graph-based retrieval for drug-disease knowledge graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self._build_entity_index()
    
    def _build_entity_index(self):
        """Build searchable index of entities by name and type"""
        self.entity_index = {
            'drugs': {},
            'diseases': {},
            'proteins': {}
        }
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type')
            # FIX: Handle None names properly
            name = (data.get('name') or '').lower()
            
            if node_type == 'drug':
                self.entity_index['drugs'][name] = node
                # Also index synonyms safely
                for synonym in data.get('synonyms', []):
                    if synonym:  # Only if synonym is not None/empty
                        self.entity_index['drugs'][synonym.lower()] = node
            elif node_type == 'disease':
                self.entity_index['diseases'][name] = node
            elif node_type in ['protein', 'polypeptide']:
                self.entity_index['proteins'][name] = node
                gene_name = data.get('gene_name')
                if gene_name:  # Safe check for gene_name
                    self.entity_index['proteins'][gene_name.lower()] = node
    
    def find_entities(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Find entities matching query text"""
        query_lower = query.lower()
        matches = []
        
        search_types = [entity_type] if entity_type else ['drugs', 'diseases', 'proteins']
        
        for etype in search_types:
            for name, node_id in self.entity_index[etype].items():
                if query_lower in name:
                    node_data = self.graph.nodes[node_id]
                    matches.append({
                        'id': node_id,
                        'name': node_data.get('name') or node_id,  # Safe fallback
                        'type': etype[:-1],  # Remove 's'
                        'score': len(query_lower) / len(name) if name else 0
                    })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def get_drug_disease_paths(self, drug_id: str, disease_id: str, max_paths: int = 5) -> List[Dict]:
        """Find paths between drug and disease (simplified version)"""
        paths = []
        
        if drug_id not in self.graph or disease_id not in self.graph:
            return paths
        
        try:
            # Find shortest paths
            shortest_paths = list(nx.all_shortest_paths(self.graph, drug_id, disease_id))
            
            for i, path in enumerate(shortest_paths[:max_paths]):
                path_names = []
                path_types = []
                
                for node in path:
                    node_data = self.graph.nodes[node]
                    path_names.append(node_data.get('name') or node)
                    path_types.append(node_data.get('type', 'unknown'))
                
                paths.append({
                    'path': path,
                    'path_names': path_names,
                    'path_types': path_types,
                    'length': len(path)
                })
        
        except nx.NetworkXNoPath:
            pass
        except Exception as e:
            print(f"Path finding error: {e}")
        
        return paths
    
    def get_entity_neighborhood(self, entity_id: str, hops: int = 2) -> nx.MultiDiGraph:
        """Get k-hop neighborhood of an entity"""
        if entity_id not in self.graph:
            return nx.MultiDiGraph()
        
        neighborhood_nodes = set([entity_id])
        current_nodes = set([entity_id])
        
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(self.graph.neighbors(node))
            neighborhood_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(neighborhood_nodes)
    
    def find_similar_drugs(self, drug_id: str, top_k: int = 10) -> List[Dict]:
        """Find drugs with similar target profiles"""
        if drug_id not in self.graph:
            return []
        
        # Get drug targets
        drug_targets = set()
        for _, target, data in self.graph.out_edges(drug_id, data=True):
            if data.get('type') == 'targets':
                drug_targets.add(target)
        
        if not drug_targets:
            return []
        
        # Find drugs with overlapping targets
        similar_drugs = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'drug' and node != drug_id:
                node_targets = set()
                for _, target, edge_data in self.graph.out_edges(node, data=True):
                    if edge_data.get('type') == 'targets':
                        node_targets.add(target)
                
                if node_targets:
                    overlap = len(drug_targets & node_targets)
                    union_size = len(drug_targets | node_targets)
                    jaccard = overlap / union_size if union_size > 0 else 0
                    
                    if overlap > 0:
                        similar_drugs.append({
                            'id': node,
                            'name': data.get('name') or node,
                            'shared_targets': overlap,
                            'jaccard_similarity': jaccard,
                            'total_targets': len(node_targets)
                        })
        
        return sorted(similar_drugs, key=lambda x: x['jaccard_similarity'], reverse=True)[:top_k]
    
    def get_disease_associated_drugs(self, disease_id: str) -> List[Dict]:
        """Find drugs that target proteins associated with the disease"""
        if disease_id not in self.graph:
            return []
        
        # Find proteins associated with the disease
        associated_proteins = set()
        for source, _, data in self.graph.in_edges(disease_id, data=True):
            if data.get('type') == 'associated_with':
                associated_proteins.add(source)
        
        # Find drugs targeting these proteins
        targeting_drugs = []
        for protein in associated_proteins:
            for source, _, data in self.graph.in_edges(protein, data=True):
                if data.get('type') == 'targets' and self.graph.nodes[source].get('type') == 'drug':
                    drug_data = self.graph.nodes[source]
                    targeting_drugs.append({
                        'drug_id': source,
                        'drug_name': drug_data.get('name') or source,
                        'protein_id': protein,
                        'protein_name': self.graph.nodes[protein].get('name') or protein,
                        'actions': data.get('actions', '')
                    })
        
        return targeting_drugs
