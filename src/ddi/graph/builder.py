# src/ddi/graph/builder.py
import os
import logging
import pickle
import json
from pathlib import Path
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from tqdm import tqdm

# Try to import PyTorch Geometric, but handle the case where it's not installed
PYG_AVAILABLE = False
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch Geometric not installed. PyG graph export will not be available.")
    logging.warning("To install PyTorch Geometric: pip install torch torch-geometric")

class KnowledgeGraphBuilder:
    """Builds a knowledge graph from multiple data sources"""
    
    def __init__(self, output_dir: str = None):
        """Initialize the graph builder
        
        Args:
            output_dir: Directory to save outputs
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir or "data/graph/full"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        
        # Track entity types and relationship types
        self.node_types = {}
        self.edge_types = {}
        
        # Node and edge counters
        self.node_counter = 0
        self.edge_counter = 0
        
    def build_graph_from_drugbank(self, drugbank_data: Dict[str, Any]) -> nx.MultiDiGraph:
        """Build knowledge graph from DrugBank data
        
        Args:
            drugbank_data: Parsed DrugBank data
            
        Returns:
            NetworkX MultiDiGraph
        """
        self.logger.info("Building knowledge graph from DrugBank data")
        
        # Add drugs
        self._add_drugs(drugbank_data.get("drugs", []))
        
        self.logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_drugs(self, drugs: List[Dict[str, Any]]) -> None:
        """Add drugs to the graph
        
        Args:
            drugs: List of drug dictionaries
        """
        self.logger.info(f"Adding {len(drugs)} drugs to the graph")
        
        for drug in tqdm(drugs, desc="Processing drugs"):
            # Add drug node
            drug_id = drug.get("drugbank_id")
            if not drug_id:
                self.logger.warning(f"Skipping drug with missing drugbank_id: {drug}")
                continue
                
            # Skip if drug already exists
            if self.graph.has_node(drug_id):
                continue
                
            # Add drug node with properties
            properties = {}
            # Copy basic properties
            for prop in ["type", "name", "description", "cas_number"]:
                if prop in drug and drug[prop] is not None:
                    properties[prop] = drug[prop]
            
            # Add node
            self._add_node(
                node_id=drug_id,
                node_type="drug",
                name=drug.get("name", drug_id),
                properties=properties
            )
            
            # Add drug-category relationships
            if "categories" in drug:
                for category in drug["categories"]:
                    if not category.get("category") or not category.get("mesh_id"):
                        continue
                        
                    category_id = category["mesh_id"]
                    category_name = category["category"]
                    
                    # Add category node if it doesn't exist
                    if not self.graph.has_node(category_id):
                        self._add_node(
                            node_id=category_id,
                            node_type="category",
                            name=category_name,
                            properties={"source": "mesh"}
                        )
                    
                    # Add drug-category relationship
                    self._add_edge(
                        source=drug_id,
                        target=category_id,
                        edge_type="has_category",
                        properties={}
                    )
            
            # Add targets
            if "targets" in drug:
                self._add_protein_relationships(drug_id, drug["targets"], "targets")
            
            # Add enzymes
            if "enzymes" in drug:
                self._add_protein_relationships(drug_id, drug["enzymes"], "metabolized_by")
            
            # Add transporters
            if "transporters" in drug:
                self._add_protein_relationships(drug_id, drug["transporters"], "transported_by")
            
            # Add carriers
            if "carriers" in drug:
                self._add_protein_relationships(drug_id, drug["carriers"], "carried_by")
    
    def _add_protein_relationships(self, drug_id: str, proteins: List[Dict[str, Any]], relationship_type: str) -> None:
        """Add protein relationships (targets, enzymes, carriers, transporters)
        
        Args:
            drug_id: Drug ID
            proteins: List of protein dictionaries
            relationship_type: Type of relationship
        """
        for protein in proteins:
            if not protein.get("id"):
                continue
                
            protein_id = protein["id"]
            protein_name = protein.get("name", protein_id)
            
            # Add protein node if it doesn't exist
            if not self.graph.has_node(protein_id):
                properties = {
                    "organism": protein.get("organism"),
                    "known_action": protein.get("known_action")
                }
                
                self._add_node(
                    node_id=protein_id,
                    node_type="protein",
                    name=protein_name,
                    properties=properties
                )
            
            # Add drug-protein relationship
            actions = protein.get("actions", [])
            action_str = "|".join(actions) if actions else None
            
            self._add_edge(
                source=drug_id,
                target=protein_id,
                edge_type=relationship_type,
                properties={"actions": action_str}
            )
            
            # Process polypeptides
            if "polypeptides" in protein:
                for polypeptide in protein["polypeptides"]:
                    if not polypeptide.get("id"):
                        continue
                        
                    polypeptide_id = polypeptide["id"]
                    
                    # Look for UniProt ID in external identifiers
                    uniprot_id = None
                    if "external_identifiers" in polypeptide:
                        for ext_id in polypeptide["external_identifiers"]:
                            if ext_id.get("resource") == "UniProtKB":
                                uniprot_id = ext_id.get("identifier")
                                break
                    
                    # Use UniProt ID if available, otherwise use polypeptide ID
                    unique_id = uniprot_id if uniprot_id else polypeptide_id
                    
                    # Skip if already processed
                    if self.graph.has_node(unique_id):
                        # Add protein-polypeptide relationship if needed
                        if not self.graph.has_edge(protein_id, unique_id):
                            self._add_edge(
                                source=protein_id,
                                target=unique_id,
                                edge_type="has_polypeptide",
                                properties={}
                            )
                        continue
                    
                    # Prepare properties
                    properties = {}
                    # Copy relevant properties
                    for prop in ["gene_name", "general_function", "specific_function", 
                                "cellular_location", "organism"]:
                        if prop in polypeptide and polypeptide[prop] is not None:
                            properties[prop] = polypeptide[prop]
                    
                    # Add specific identifiers
                    properties["drugbank_polypeptide_id"] = polypeptide_id
                    if uniprot_id:
                        properties["uniprot_id"] = uniprot_id
                    
                    # Add polypeptide node
                    self._add_node(
                        node_id=unique_id,
                        node_type="polypeptide",
                        name=polypeptide.get("name", unique_id),
                        properties=properties
                    )
                    
                    # Add protein-polypeptide relationship
                    self._add_edge(
                        source=protein_id,
                        target=unique_id,
                        edge_type="has_polypeptide",
                        properties={}
                    )
    
    def add_disease_data(self, disease_data: Dict[str, Any]) -> None:
        """Add disease data to the graph
        
        Args:
            disease_data: Disease data dictionary
        """
        if not disease_data:
            self.logger.warning("No disease data provided")
            return
            
        diseases = []
        # Handle different formats of disease data
        if isinstance(disease_data, dict):
            if "descriptors" in disease_data:
                # Handle MeSH format
                for d_id, desc in disease_data["descriptors"].items():
                    if desc.get("is_disease", False):
                        diseases.append(desc)
            else:
                # Handle other dictionary formats
                diseases = list(disease_data.values())
        elif isinstance(disease_data, list):
            # Handle list format
            diseases = disease_data
        else:
            self.logger.warning(f"Unsupported disease data format: {type(disease_data)}")
            return
            
        self.logger.info(f"Adding {len(diseases)} diseases to the graph")
        
        for disease in tqdm(diseases, desc="Processing diseases"):
            disease_id = disease.get("id")
            if not disease_id:
                continue
                
            # Skip if disease already exists
            if self.graph.has_node(disease_id):
                continue
                
            # Prepare properties
            properties = {}
            for prop in ["description", "synonyms", "tree_numbers"]:
                if prop in disease and disease[prop] is not None:
                    properties[prop] = disease[prop]
            
            # Add disease node
            self._add_node(
                node_id=disease_id,
                node_type="disease",
                name=disease.get("name", disease_id),
                properties=properties
            )
            
            # Add parent-child relationships
            if "parents" in disease:
                for parent in disease["parents"]:
                    parent_id = parent.get("id")
                    if not parent_id or not self.graph.has_node(parent_id):
                        continue
                        
                    # Add relationship
                    self._add_edge(
                        source=parent_id,
                        target=disease_id,
                        edge_type="has_child",
                        properties={}
                    )
    
    def add_drug_disease_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add drug-disease associations
        
        Args:
            associations: List of drug-disease association dictionaries
        """
        if not associations:
            self.logger.warning("No drug-disease associations provided")
            return
            
        self.logger.info(f"Adding {len(associations)} drug-disease associations")
        added_count = 0
        
        for assoc in tqdm(associations, desc="Processing drug-disease associations"):
            drug_id = assoc.get("drug_id")
            disease_id = assoc.get("disease_id")
            
            # Skip if invalid data
            if not drug_id or not disease_id:
                continue
                
            # Skip if drug or disease doesn't exist
            if not self.graph.has_node(drug_id) or not self.graph.has_node(disease_id):
                continue
            
            # Prepare properties
            properties = {}
            for prop in ["source", "evidence_level", "mechanism", "score", "confidence"]:
                if prop in assoc and assoc[prop] is not None:
                    properties[prop] = assoc[prop]
            
            # Add drug-disease relationship
            self._add_edge(
                source=drug_id,
                target=disease_id,
                edge_type="treats",
                properties=properties
            )
            
            added_count += 1
            
        self.logger.info(f"Added {added_count} drug-disease associations to the graph")
    
    def add_drug_target_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add drug-target associations
        
        Args:
            associations: List of drug-target association dictionaries
        """
        if not associations:
            self.logger.warning("No drug-target associations provided")
            return
            
        self.logger.info(f"Adding {len(associations)} drug-target associations")
        added_count = 0
        
        for assoc in tqdm(associations, desc="Processing drug-target associations"):
            drug_id = assoc.get("drug_id")
            target_id = assoc.get("target_id")
            
            # Skip if invalid data
            if not drug_id or not target_id:
                continue
                
            # Skip if drug or target doesn't exist
            if not self.graph.has_node(drug_id) or not self.graph.has_node(target_id):
                continue
            
            # Prepare properties
            properties = {}
            for prop in ["score", "confidence", "mechanism", "action"]:
                if prop in assoc and assoc[prop] is not None:
                    properties[prop] = assoc[prop]
            
            # Add drug-target relationship
            self._add_edge(
                source=drug_id,
                target=target_id,
                edge_type="targets",
                properties=properties
            )
            
            added_count += 1
            
        self.logger.info(f"Added {added_count} drug-target associations to the graph")
    
    def add_target_disease_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add target-disease associations
        
        Args:
            associations: List of target-disease association dictionaries
        """
        if not associations:
            self.logger.warning("No target-disease associations provided")
            return
            
        self.logger.info(f"Adding {len(associations)} target-disease associations")
        added_count = 0
        
        for assoc in tqdm(associations, desc="Processing target-disease associations"):
            target_id = assoc.get("target_id")
            disease_id = assoc.get("disease_id")
            
            # Skip if invalid data
            if not target_id or not disease_id:
                continue
                
            # Skip if target or disease doesn't exist
            if not self.graph.has_node(target_id) or not self.graph.has_node(disease_id):
                continue
            
            # Prepare properties
            properties = {}
            for prop in ["score", "confidence", "evidence"]:
                if prop in assoc and assoc[prop] is not None:
                    properties[prop] = assoc[prop]
            
            # Add target-disease relationship
            self._add_edge(
                source=target_id,
                target=disease_id,
                edge_type="associated_with",
                properties=properties
            )
            
            added_count += 1
            
        self.logger.info(f"Added {added_count} target-disease associations to the graph")
    
    def _add_node(self, node_id: str, node_type: str, name: str, properties: Dict[str, Any]) -> None:
        """Add a node to the graph
        
        Args:
            node_id: Node ID
            node_type: Node type
            name: Node name
            properties: Node properties
        """
        # Add node to graph
        node_attrs = {
            "type": node_type,
            "name": name,
            **properties
        }
        
        self.graph.add_node(node_id, **node_attrs)
        
        # Update node type counter
        if node_type not in self.node_types:
            self.node_types[node_type] = 0
        self.node_types[node_type] += 1
        
        # Update node counter
        self.node_counter += 1
    
    def _add_edge(self, source: str, target: str, edge_type: str, properties: Dict[str, Any]) -> None:
        """Add an edge to the graph
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Edge type
            properties: Edge properties
        """
        # Add edge to graph
        edge_attrs = {
            "type": edge_type,
            **properties
        }
        
        self.graph.add_edge(source, target, **edge_attrs)
        
        # Update edge type counter
        if edge_type not in self.edge_types:
            self.edge_types[edge_type] = 0
        self.edge_types[edge_type] += 1
        
        # Update edge counter
        self.edge_counter += 1
    
    def _normalize_id(self, text: str) -> str:
        """Normalize text for use as an ID
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        try:
            return text.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")
        except:
            return str(text)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics
        
        Returns:
            Dictionary of graph statistics
        """
        # Basic graph statistics
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self.node_types,
            "edge_types": self.edge_types
        }
        
        # Additional statistics if graph is large enough
        if self.graph.number_of_nodes() > 0:
            # Calculate average degree
            degrees = [d for _, d in self.graph.degree()]
            stats["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
            
            # Get connected components in undirected graph
            undirected = self.graph.to_undirected()
            connected_components = list(nx.connected_components(undirected))
            stats["num_connected_components"] = len(connected_components)
            
            if connected_components:
                largest_cc = max(connected_components, key=len)
                stats["largest_component_size"] = len(largest_cc)
                stats["largest_component_percentage"] = len(largest_cc) / self.graph.number_of_nodes() * 100
        
        return stats
    
    def save_graph(self, formats: List[str] = ["graphml", "pickle"]) -> Dict[str, str]:
        """Save graph to files

        Args:
            formats: List of output formats

        Returns:
            Dictionary of output file paths
        """
        output_files = {}

        for fmt in formats:
            output_path = os.path.join(self.output_dir, f"knowledge_graph.{fmt}")

            if fmt == "graphml":
                self.logger.info("Preparing simplified graph for GraphML export...")
                # Create a separate, simplified graph for GraphML export
                clean_graph_ml = nx.MultiDiGraph()

                # Add nodes with minimal attributes
                for n, attrs in self.graph.nodes(data=True):
                    ml_attrs = {
                        'type': str(attrs.get('type', '')),
                        'name': str(attrs.get('name', n)) # Use node ID as fallback name
                    }
                    # Ensure attributes are simple strings for GraphML
                    ml_attrs = {k: v for k, v in ml_attrs.items() if isinstance(v, (str, int, float, bool))}
                    clean_graph_ml.add_node(n, **ml_attrs)

                # Add edges with minimal attributes
                for u, v, key, attrs in self.graph.edges(keys=True, data=True):
                    # Ensure source and target nodes exist in the simplified graph
                    if clean_graph_ml.has_node(u) and clean_graph_ml.has_node(v):
                        ml_attrs = {
                            'type': str(attrs.get('type', ''))
                        }
                        # Ensure attributes are simple strings for GraphML
                        ml_attrs = {k: v for k, v in ml_attrs.items() if isinstance(v, (str, int, float, bool))}
                        clean_graph_ml.add_edge(u, v, key=key, **ml_attrs)
                    else:
                        self.logger.warning(f"Skipping edge ({u}, {v}) for GraphML due to missing node in simplified graph.")

                try:
                    nx.write_graphml(clean_graph_ml, output_path)
                    self.logger.info(f"Saved simplified graph in GraphML format to {output_path}")
                except Exception as e:
                    self.logger.error(f"Error writing GraphML file: {e}")
                    continue # Skip adding to output_files if saving failed

            elif fmt == "pickle":
                try:
                    with open(output_path, "wb") as f:
                        pickle.dump(self.graph, f)
                    self.logger.info(f"Saved graph in pickle format to {output_path}")
                except Exception as e:
                    self.logger.error(f"Error writing pickle file: {e}")
                    continue

            elif fmt == "pyg":
                if PYG_AVAILABLE:
                    pyg_graph = self._convert_to_pyg()
                    if pyg_graph is not None:
                        try:
                            # Use pickle for saving PyG data object
                            with open(output_path, "wb") as f:
                                pickle.dump(pyg_graph, f)
                            self.logger.info(f"Saved graph in PyG format (via pickle) to {output_path}")
                        except Exception as e:
                            self.logger.error(f"Error writing PyG file: {e}")
                            continue
                    else:
                        self.logger.warning("Failed to create PyG graph. Skipping PyG format export.")
                        continue
                else:
                    self.logger.warning("PyTorch Geometric not available. Skipping PyG format export.")
                    continue
            else:
                 self.logger.warning(f"Unsupported save format: {fmt}")
                 continue

            # Only add to output_files if saving was successful (or attempted for unsupported)
            output_files[fmt] = output_path


        # Save node and edge type mappings (remains the same)
        mappings = {
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "statistics": self.get_statistics()
        }
        mappings_path = os.path.join(self.output_dir, "graph_mappings.json")
        try:
            with open(mappings_path, "w") as f:
                json.dump(mappings, f, indent=2)
            output_files["mappings"] = mappings_path
            self.logger.info(f"Saved graph mappings to {mappings_path}")
        except Exception as e:
             self.logger.error(f"Error writing mappings JSON file: {e}")


        return output_files

    def _convert_to_pyg(self):
        """Convert NetworkX graph to PyTorch Geometric Data object
        
        Returns:
            PyTorch Geometric Data object
        """
        if not PYG_AVAILABLE:
            self.logger.error("PyTorch Geometric not available. Cannot convert to PyG format.")
            return None
            
        self.logger.info("Converting NetworkX graph to PyTorch Geometric format")
        
        try:
            import torch
            from torch_geometric.data import Data
            
            # Create node and edge type mappings
            node_type_to_id = {t: i for i, t in enumerate(sorted(self.node_types.keys()))}
            edge_type_to_id = {t: i for i, t in enumerate(sorted(self.edge_types.keys()))}
            
            # Create node features
            nodes = list(self.graph.nodes)
            node_type_dict = {n: self.graph.nodes[n]["type"] for n in nodes}
            node_types = [node_type_to_id[node_type_dict[n]] for n in nodes]
            
            # Create mapping from node ID to index
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Create edge index and edge attributes
            edge_index = [[], []]  # [src_nodes, dst_nodes]
            edge_types_list = []
            
            for u, v, data in self.graph.edges(data=True):
                edge_index[0].append(node_to_idx[u])
                edge_index[1].append(node_to_idx[v])
                edge_types_list.append(edge_type_to_id[data["type"]])
            
            # Convert to tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_type = torch.tensor(node_types, dtype=torch.long)
            edge_type = torch.tensor(edge_types_list, dtype=torch.long)
            
            # Create PyG Data object
            data = Data(
                x=node_type.view(-1, 1),  # Node features (just type for now)
                edge_index=edge_index,
                edge_attr=edge_type.view(-1, 1),  # Edge features (just type for now)
                num_nodes=len(nodes)
            )
            
            # Add metadata as attributes
            data.node_type_to_id = node_type_to_id
            data.edge_type_to_id = edge_type_to_id
            data.id_to_node_type = {v: k for k, v in node_type_to_id.items()}
            data.id_to_edge_type = {v: k for k, v in edge_type_to_id.items()}
            data.node_to_idx = node_to_idx
            data.idx_to_node = nodes
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error converting to PyTorch Geometric graph: {str(e)}")
            return None