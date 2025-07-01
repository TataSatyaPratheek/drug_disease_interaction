# src/graphrag/core/retriever.py
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from ..ddi.analysis.graph_analysis import GraphAnalyzer
from ..ddi.features.feature_engineering import FeatureExtractor

class GraphRetriever:
    """Advanced graph-based retrieval for drug-disease knowledge graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.analyzer = GraphAnalyzer(graph)
        self.feature_extractor = FeatureExtractor(graph)
        
        # Pre-compute communities for fast retrieval
        self.communities = self.analyzer.detect_communities()
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
            name = data.get('name', '').lower()
            
            if node_type == 'drug':
                self.entity_index['drugs'][name] = node
                # Also index synonyms
                for synonym in data.get('synonyms', []):
                    self.entity_index['drugs'][synonym.lower()] = node
            elif node_type == 'disease':
                self.entity_index['diseases'][name] = node
            elif node_type in ['protein', 'polypeptide']:
                self.entity_index['proteins'][name] = node
                if data.get('gene_name'):
                    self.entity_index['proteins'][data['gene_name'].lower()] = node
    
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
                        'name': node_data.get('name', name),
                        'type': etype[:-1],  # Remove 's'
                        'score': len(query_lower) / len(name)  # Simple relevance score
                    })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def get_drug_disease_paths(self, drug_id: str, disease_id: str, max_paths: int = 5) -> List[Dict]:
        """Find all meaningful paths between drug and disease"""
        return self.analyzer.find_drug_disease_paths(drug_id, disease_id, max_paths)
    
    def get_entity_neighborhood(self, entity_id: str, hops: int = 2) -> nx.MultiDiGraph:
        """Get k-hop neighborhood of an entity"""
        return self.analyzer.get_entity_neighborhood(entity_id, hops)
    
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
                    jaccard = overlap / len(drug_targets | node_targets)
                    
                    if overlap > 0:
                        similar_drugs.append({
                            'id': node,
                            'name': data.get('name', node),
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
                        'drug_name': drug_data.get('name', source),
                        'protein_id': protein,
                        'protein_name': self.graph.nodes[protein].get('name', protein),
                        'actions': data.get('actions', '')
                    })
        
        return targeting_drugs
