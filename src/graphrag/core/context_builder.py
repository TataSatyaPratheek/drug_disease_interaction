# src/graphrag/core/context_builder.py
import networkx as nx
from typing import Dict, List, Any, Optional

class ContextBuilder:
    """Convert graph data into LLM-friendly context"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def build_entity_context(self, entity_id: str) -> str:
        """Build comprehensive context for a single entity"""
        if entity_id not in self.graph:
            return f"Entity {entity_id} not found in knowledge graph."
        
        node_data = self.graph.nodes[entity_id]
        entity_type = node_data.get('type', 'unknown')
        name = node_data.get('name', entity_id)
        
        context_parts = [f"**{name}** ({entity_type.title()})\n"]
        
        if entity_type == 'drug':
            context_parts.append(self._build_drug_context(entity_id, node_data))
        elif entity_type == 'disease':
            context_parts.append(self._build_disease_context(entity_id, node_data))
        elif entity_type in ['protein', 'polypeptide']:
            context_parts.append(self._build_protein_context(entity_id, node_data))
        
        return "\n".join(context_parts)
    
    def _build_drug_context(self, drug_id: str, data: Dict) -> str:
        """Build context for drug entities"""
        context = []
        
        if data.get('description'):
            context.append(f"Description: {data['description'][:500]}...")
        
        if data.get('indication'):
            context.append(f"Indication: {data['indication'][:300]}...")
        
        if data.get('mechanism_of_action'):
            context.append(f"Mechanism: {data['mechanism_of_action'][:300]}...")
        
        # Add target information
        targets = []
        for _, target, edge_data in self.graph.out_edges(drug_id, data=True):
            if edge_data.get('type') == 'targets':
                target_name = self.graph.nodes[target].get('name', target)
                actions = edge_data.get('actions', '')
                targets.append(f"{target_name} ({actions})" if actions else target_name)
        
        if targets:
            context.append(f"Targets: {', '.join(targets[:10])}")
        
        return "\n".join(context)
    
    def _build_disease_context(self, disease_id: str, data: Dict) -> str:
        """Build context for disease entities"""
        context = []
        
        if data.get('annotation'):
            context.append(f"Description: {data['annotation']}")
        
        # Add associated proteins
        proteins = []
        for source, _, edge_data in self.graph.in_edges(disease_id, data=True):
            if edge_data.get('type') == 'associated_with':
                protein_name = self.graph.nodes[source].get('name', source)
                proteins.append(protein_name)
        
        if proteins:
            context.append(f"Associated proteins: {', '.join(proteins[:15])}")
        
        return "\n".join(context)
    
    def build_path_context(self, paths: List[Dict]) -> str:
        """Build context from drug-disease paths"""
        if not paths:
            return "No paths found between the specified entities."
        
        context_parts = ["**Molecular Pathways:**\n"]
        
        for i, path_data in enumerate(paths[:3], 1):  # Limit to top 3 paths
            path_names = path_data['path_names']
            path_types = path_data['path_types']
            
            path_desc = []
            for j, (name, ptype) in enumerate(zip(path_names, path_types)):
                if j == 0:
                    path_desc.append(f"**{name}** ({ptype})")
                elif j == len(path_names) - 1:
                    path_desc.append(f"**{name}** ({ptype})")
                else:
                    path_desc.append(f"{name} ({ptype})")
            
            context_parts.append(f"{i}. {' → '.join(path_desc)}")
        
        return "\n".join(context_parts)
    
    def build_subgraph_context(self, subgraph: nx.MultiDiGraph, focus_entities: List[str] = None) -> str:
        """Build context from a subgraph"""
        if subgraph.number_of_nodes() == 0:
            return "No relevant subgraph found."
        
        # Group nodes by type
        nodes_by_type = {}
        for node, data in subgraph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node, data.get('name', node)))
        
        context_parts = [f"**Relevant Network ({subgraph.number_of_nodes()} entities):**\n"]
        
        for node_type, nodes in nodes_by_type.items():
            if len(nodes) <= 10:
                node_names = [name for _, name in nodes]
                context_parts.append(f"**{node_type.title()}s:** {', '.join(node_names)}")
            else:
                node_names = [name for _, name in nodes[:10]]
                context_parts.append(f"**{node_type.title()}s:** {', '.join(node_names)} ... and {len(nodes)-10} more")
        
        return "\n".join(context_parts)
