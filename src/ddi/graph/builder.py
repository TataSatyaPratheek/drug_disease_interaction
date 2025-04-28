# src/ddi/graph/builder.py
import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from tqdm import tqdm
import dgl

class KnowledgeGraphBuilder:
    """Builds a knowledge graph from DrugBank and other data sources"""
    
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
        self._add_drugs(drugbank_data["drugs"])
        
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
            drug_id = drug["drugbank_id"]
            
            # Skip if drug already exists
            if self.graph.has_node(drug_id):
                continue
                
            # Add drug node
            self._add_node(
                node_id=drug_id,
                node_type="drug",
                name=drug["name"],
                properties={
                    "type": drug["type"],
                    "description": drug["description"],
                    "cas_number": drug["cas_number"],
                    "groups": drug["groups"],
                    "indication": drug["indications"],
                    "pharmacodynamics": drug["pharmacodynamics"],
                    "mechanism_of_action": drug["mechanism_of_action"],
                    "toxicity": drug["toxicity"],
                    "metabolism": drug["metabolism"],
                    "absorption": drug["absorption"],
                    "half_life": drug["half_life"],
                    "protein_binding": drug["protein_binding"],
                    "route_of_elimination": drug["route_of_elimination"],
                    "volume_of_distribution": drug["volume_of_distribution"],
                    "clearance": drug["clearance"]
                }
            )
            
            # Add drug-category relationships
            for category in drug.get("categories", []):
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
            
            # Add ATC codes
            for atc in drug.get("atc_codes", []):
                if not atc.get("code"):
                    continue
                    
                atc_id = atc["code"]
                
                # Add ATC node if it doesn't exist
                if not self.graph.has_node(atc_id):
                    atc_name = None
                    for level in atc.get("levels", []):
                        if level.get("code") == atc_id:
                            atc_name = level.get("name")
                            break
                            
                    self._add_node(
                        node_id=atc_id,
                        node_type="atc_code",
                        name=atc_name or atc_id,
                        properties={}
                    )
                
                # Add drug-ATC relationship
                self._add_edge(
                    source=drug_id,
                    target=atc_id,
                    edge_type="has_atc_code",
                    properties={}
                )
            
            # Add classification
            if drug.get("classification"):
                # Add classification nodes and relationships
                for class_type in ["kingdom", "superclass", "class", "subclass", "direct_parent"]:
                    class_value = drug["classification"].get(class_type)
                    if not class_value:
                        continue
                        
                    class_id = f"{class_type}_{self._normalize_id(class_value)}"
                    
                    # Add classification node if it doesn't exist
                    if not self.graph.has_node(class_id):
                        self._add_node(
                            node_id=class_id,
                            node_type=f"classification_{class_type}",
                            name=class_value,
                            properties={}
                        )
                    
                    # Add drug-classification relationship
                    self._add_edge(
                        source=drug_id,
                        target=class_id,
                        edge_type=f"has_{class_type}",
                        properties={}
                    )
            
            # Add calculated properties as node attributes
            for prop in drug.get("calculated_properties", []):
                if not prop.get("kind") or not prop.get("value"):
                    continue
                    
                prop_kind = prop["kind"]
                prop_value = prop["value"]
                
                # Update drug node with property
                # Convert to appropriate data type if possible
                try:
                    if "logP" in prop_kind or "logS" in prop_kind or "pKa" in prop_kind:
                        prop_value = float(prop_value)
                    elif prop_kind in ["Molecular Weight", "Monoisotopic Weight"]:
                        prop_value = float(prop_value.split()[0])  # Extract number from e.g. "309.3256 g/mol"
                except:
                    pass  # Keep as string if conversion fails
                
                # Add property to drug node
                self.graph.nodes[drug_id][f"property_{self._normalize_id(prop_kind)}"] = prop_value
            
            # Add pathways
            for pathway in drug.get("pathways", []):
                if not pathway.get("smpdb_id") or not pathway.get("name"):
                    continue
                    
                pathway_id = pathway["smpdb_id"]
                pathway_name = pathway["name"]
                
                # Add pathway node if it doesn't exist
                if not self.graph.has_node(pathway_id):
                    self._add_node(
                        node_id=pathway_id,
                        node_type="pathway",
                        name=pathway_name,
                        properties={"category": pathway.get("category")}
                    )
                
                # Add drug-pathway relationship
                self._add_edge(
                    source=drug_id,
                    target=pathway_id,
                    edge_type="involved_in_pathway",
                    properties={}
                )
                
                # Add pathway-enzyme relationships
                for enzyme_id in pathway.get("enzymes", []):
                    if not enzyme_id:
                        continue
                        
                    # Add enzyme node if it doesn't exist
                    if not self.graph.has_node(enzyme_id):
                        self._add_node(
                            node_id=enzyme_id,
                            node_type="enzyme",
                            name=enzyme_id,  # Use ID as name until we get more info
                            properties={"source": "uniprot"}
                        )
                    
                    # Add pathway-enzyme relationship
                    self._add_edge(
                        source=pathway_id,
                        target=enzyme_id,
                        edge_type="has_enzyme",
                        properties={}
                    )
            
            # Add targets
            self._add_protein_relationships(drug_id, drug.get("targets", []), "targets")
            
            # Add enzymes
            self._add_protein_relationships(drug_id, drug.get("enzymes", []), "metabolized_by")
            
            # Add carriers
            self._add_protein_relationships(drug_id, drug.get("carriers", []), "carried_by")
            
            # Add transporters
            self._add_protein_relationships(drug_id, drug.get("transporters", []), "transported_by")
            
            # Add drug interactions
            for interaction in drug.get("drug_interactions", []):
                if not interaction.get("drugbank_id"):
                    continue
                    
                target_drug_id = interaction["drugbank_id"]
                
                # Add target drug node as a placeholder if it doesn't exist
                if not self.graph.has_node(target_drug_id):
                    self._add_node(
                        node_id=target_drug_id,
                        node_type="drug",
                        name=interaction.get("name", target_drug_id),
                        properties={"placeholder": True}
                    )
                
                # Add drug-drug interaction
                self._add_edge(
                    source=drug_id,
                    target=target_drug_id,
                    edge_type="interacts_with",
                    properties={"description": interaction.get("description")}
                )
    
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
                self._add_node(
                    node_id=protein_id,
                    node_type="protein",
                    name=protein_name,
                    properties={
                        "organism": protein.get("organism"),
                        "known_action": protein.get("known_action")
                    }
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
            for polypeptide in protein.get("polypeptides", []):
                if not polypeptide.get("id"):
                    continue
                    
                polypeptide_id = polypeptide["id"]
                
                # Look for UniProt ID
                uniprot_id = None
                for ext_id in polypeptide.get("external_identifiers", []):
                    if ext_id.get("resource") == "UniProtKB":
                        uniprot_id = ext_id.get("identifier")
                        break
                
                # Use UniProt ID if available, otherwise use polypeptide ID
                if uniprot_id:
                    unique_id = uniprot_id
                else:
                    unique_id = polypeptide_id
                
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
                
                # Add polypeptide node
                self._add_node(
                    node_id=unique_id,
                    node_type="polypeptide",
                    name=polypeptide.get("name", unique_id),
                    properties={
                        "gene_name": polypeptide.get("gene_name"),
                        "general_function": polypeptide.get("general_function"),
                        "specific_function": polypeptide.get("specific_function"),
                        "cellular_location": polypeptide.get("cellular_location"),
                        "organism": polypeptide.get("organism"),
                        "drugbank_polypeptide_id": polypeptide_id,
                        "uniprot_id": uniprot_id
                    }
                )
                
                # Add protein-polypeptide relationship
                self._add_edge(
                    source=protein_id,
                    target=unique_id,
                    edge_type="has_polypeptide",
                    properties={}
                )
                
                # Add GO terms
                for go_term in polypeptide.get("go_classifiers", []):
                    if not go_term.get("description"):
                        continue
                        
                    go_id = f"GO_{self._normalize_id(go_term['description'])}"
                    go_category = go_term.get("category")
                    
                    # Add GO term node if it doesn't exist
                    if not self.graph.has_node(go_id):
                        self._add_node(
                            node_id=go_id,
                            node_type="go_term",
                            name=go_term["description"],
                            properties={"category": go_category}
                        )
                    
                    # Add polypeptide-GO relationship
                    self._add_edge(
                        source=unique_id,
                        target=go_id,
                        edge_type="has_go_term",
                        properties={}
                    )
                
                # Add Pfam domains
                for pfam in polypeptide.get("pfams", []):
                    if not pfam.get("identifier"):
                        continue
                        
                    pfam_id = pfam["identifier"]
                    
                    # Add Pfam node if it doesn't exist
                    if not self.graph.has_node(pfam_id):
                        self._add_node(
                            node_id=pfam_id,
                            node_type="pfam",
                            name=pfam.get("name", pfam_id),
                            properties={}
                        )
                    
                    # Add polypeptide-Pfam relationship
                    self._add_edge(
                        source=unique_id,
                        target=pfam_id,
                        edge_type="has_pfam",
                        properties={}
                    )
    
    def _add_node(self, node_id: str, node_type: str, name: str, properties: Dict[str, Any]) -> None:
        """Add a node to the graph
        
        Args:
            node_id: Node ID
            node_type: Node type
            name: Node name
            properties: Node properties
        """
        # Add node to graph
        self.graph.add_node(
            node_id,
            type=node_type,
            name=name,
            **properties
        )
        
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
        self.graph.add_edge(
            source,
            target,
            type=edge_type,
            **properties
        )
        
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
            
        return text.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")
    
    def add_disease_data(self, disease_data: Dict[str, Any]) -> None:
        """Add disease data to the graph
        
        Args:
            disease_data: Disease data
        """
        self.logger.info(f"Adding disease data to the graph")
        
        diseases = disease_data.get("diseases", [])
        self.logger.info(f"Processing {len(diseases)} diseases")
        
        for disease in tqdm(diseases, desc="Processing diseases"):
            disease_id = disease["id"]
            
            # Add disease node
            self._add_node(
                node_id=disease_id,
                node_type="disease",
                name=disease["name"],
                properties={
                    "description": disease.get("description"),
                    "synonyms": disease.get("synonyms", []),
                    "xrefs": disease.get("xrefs", {})
                }
            )
            
            # Add disease-gene associations
            for gene_association in disease.get("gene_associations", []):
                gene_id = gene_association["gene_id"]
                
                # Add gene node if it doesn't exist
                if not self.graph.has_node(gene_id):
                    self._add_node(
                        node_id=gene_id,
                        node_type="gene",
                        name=gene_association.get("gene_symbol", gene_id),
                        properties={"source": "entrez"}
                    )
                
                # Add disease-gene relationship
                self._add_edge(
                    source=disease_id,
                    target=gene_id,
                    edge_type="associated_with_gene",
                    properties={
                        "score": gene_association.get("score"),
                        "evidence": gene_association.get("evidence")
                    }
                )
    
    def add_drug_disease_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add known drug-disease associations
        
        Args:
            associations: List of drug-disease association dictionaries
        """
        self.logger.info(f"Adding {len(associations)} drug-disease associations")
        
        for assoc in tqdm(associations, desc="Processing drug-disease associations"):
            drug_id = assoc["drug_id"]
            disease_id = assoc["disease_id"]
            
            # Skip if drug or disease doesn't exist
            if not self.graph.has_node(drug_id) or not self.graph.has_node(disease_id):
                continue
            
            # Add drug-disease relationship
            self._add_edge(
                source=drug_id,
                target=disease_id,
                edge_type="treats",
                properties={
                    "source": assoc.get("source"),
                    "evidence_level": assoc.get("evidence_level"),
                    "mechanism": assoc.get("mechanism")
                }
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics
        
        Returns:
            Dictionary of graph statistics
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self.node_types,
            "edge_types": self.edge_types
        }
    
    def save_graph(self, formats: List[str] = ["graphml", "pickle", "dgl"]) -> Dict[str, str]:
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
                # Remove complex attributes that can't be serialized to GraphML
                clean_graph = nx.MultiDiGraph()
                for n, attrs in self.graph.nodes(data=True):
                    clean_attrs = {}
                    for k, v in attrs.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            clean_attrs[k] = v
                        elif isinstance(v, list):
                            # Convert lists to strings
                            clean_attrs[k] = "|".join(str(x) for x in v)
                        else:
                            # Skip complex objects
                            continue
                    clean_graph.add_node(n, **clean_attrs)
                
                for u, v, key, attrs in self.graph.edges(keys=True, data=True):
                    clean_attrs = {}
                    for k, v in attrs.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            clean_attrs[k] = v
                        elif isinstance(v, list):
                            # Convert lists to strings
                            clean_attrs[k] = "|".join(str(x) for x in v)
                        else:
                            # Skip complex objects
                            continue
                    clean_graph.add_edge(u, v, key=key, **clean_attrs)
                
                nx.write_graphml(clean_graph, output_path)
            
            elif fmt == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(self.graph, f)
            
            elif fmt == "dgl":
                # Convert to DGL graph
                dgl_graph = self._convert_to_dgl()
                
                # Save DGL graph
                dgl.save_graphs(output_path, [dgl_graph])
            
            output_files[fmt] = output_path
            self.logger.info(f"Saved graph in {fmt} format to {output_path}")
        
        # Save node and edge type mappings
        mappings = {
            "node_types": self.node_types,
            "edge_types": self.edge_types
        }
        
        mappings_path = os.path.join(self.output_dir, "graph_mappings.json")
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=2)
        
        output_files["mappings"] = mappings_path
        self.logger.info(f"Saved graph mappings to {mappings_path}")
        
        return output_files
    
    def _convert_to_dgl(self) -> dgl.DGLGraph:
        """Convert NetworkX graph to DGL graph
        
        Returns:
            DGL graph
        """
        self.logger.info("Converting NetworkX graph to DGL format")
        
        # Create node and edge type mappings
        node_type_to_id = {t: i for i, t in enumerate(sorted(self.node_types.keys()))}
        edge_type_to_id = {t: i for i, t in enumerate(sorted(self.edge_types.keys()))}
        
        # Create node features
        nodes = list(self.graph.nodes)
        node_type_dict = {n: self.graph.nodes[n]["type"] for n in nodes}
        node_types = [node_type_to_id[node_type_dict[n]] for n in nodes]
        
        # Create mapping from node ID to index
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create edge lists
        src_nodes = []
        dst_nodes = []
        edge_types = []
        
        for u, v, data in self.graph.edges(data=True):
            src_nodes.append(node_to_idx[u])
            dst_nodes.append(node_to_idx[v])
            edge_types.append(edge_type_to_id[data["type"]])
        
        # Create DGL graph
        dgl_graph = dgl.graph((src_nodes, dst_nodes))
        
        # Add node features
        dgl_graph.ndata["type"] = torch.tensor(node_types)
        dgl_graph.ndata["idx"] = torch.arange(len(nodes))
        
        # Add edge features
        dgl_graph.edata["type"] = torch.tensor(edge_types)
        
        # Add node and edge type mappings
        dgl_graph.node_type_to_id = node_type_to_id
        dgl_graph.edge_type_to_id = edge_type_to_id
        dgl_graph.id_to_node_type = {v: k for k, v in node_type_to_id.items()}
        dgl_graph.id_to_edge_type = {v: k for k, v in edge_type_to_id.items()}
        
        # Add node ID to index mapping
        dgl_graph.node_to_idx = node_to_idx
        dgl_graph.idx_to_node = nodes
        
        return dgl_graph


# Example usage in build_graph.py script
def main():
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description="Build knowledge graph from DrugBank and other data")
    parser.add_argument("--drugbank", required=True, help="Path to parsed DrugBank data (pickle or JSON)")
    parser.add_argument("--disease", help="Path to disease data (pickle or JSON)")
    parser.add_argument("--associations", help="Path to drug-disease associations (pickle or JSON)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load DrugBank data
    if args.drugbank.endswith(".pickle") or args.drugbank.endswith(".pkl"):
        with open(args.drugbank, "rb") as f:
            drugbank_data = pickle.load(f)
    else:
        with open(args.drugbank, "r") as f:
            drugbank_data = json.load(f)
    
    # Initialize graph builder
    graph_builder = KnowledgeGraphBuilder(output_dir=args.output)
    
    # Build graph from DrugBank data
    graph_builder.build_graph_from_drugbank(drugbank_data)
    
    # Add disease data if provided
    if args.disease:
        if args.disease.endswith(".pickle") or args.disease.endswith(".pkl"):
            with open(args.disease, "rb") as f:
                disease_data = pickle.load(f)
        else:
            with open(args.disease, "r") as f:
                disease_data = json.load(f)
        
        graph_builder.add_disease_data(disease_data)
    
    # Add drug-disease associations if provided
    if args.associations:
        if args.associations.endswith(".pickle") or args.associations.endswith(".pkl"):
            with open(args.associations, "rb") as f:
                associations = pickle.load(f)
        else:
            with open(args.associations, "r") as f:
                associations = json.load(f)
        
        graph_builder.add_drug_disease_associations(associations)
    
    # Print graph statistics
    stats = graph_builder.get_statistics()
    logging.info(f"Graph statistics: {stats}")
    
    # Save graph
    output_files = graph_builder.save_graph(formats=["graphml", "pickle", "dgl"])
    logging.info(f"Graph saved to: {output_files}")

if __name__ == "__main__":
    main()


# src/ddi/graph/features.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import logging
from tqdm import tqdm

class FeatureExtractor:
    """Extract features for different node types in the knowledge graph"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_drug_features(self, drugs_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Extract features for drug nodes
        
        Args:
            drugs_data: List of drug dictionaries
            
        Returns:
            Dictionary of drug ID to feature tensor
        """
        self.logger.info(f"Extracting features for {len(drugs_data)} drugs")
        
        features = {}
        
        for drug in tqdm(drugs_data, desc="Extracting drug features"):
            drug_id = drug["drugbank_id"]
            
            # Get SMILES if available
            smiles = None
            for prop in drug.get("calculated_properties", []):
                if prop.get("kind") == "SMILES":
                    smiles = prop.get("value")
                    break
            
            # Extract features from SMILES
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Generate Morgan fingerprint
                        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        fp_array = np.zeros((1,))
                        DataStructs.ConvertToNumpyArray(fingerprint, fp_array)
                        
                        # Calculate basic descriptors
                        descriptors = [
                            Descriptors.MolWt(mol),  # Molecular weight
                            Descriptors.MolLogP(mol),  # LogP
                            Descriptors.NumHDonors(mol),  # H-bond donors
                            Descriptors.NumHAcceptors(mol),  # H-bond acceptors
                            Descriptors.TPSA(mol),  # Topological polar surface area
                            Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
                            mol.GetNumAtoms(),  # Number of atoms
                            Descriptors.FractionCSP3(mol),  # Fraction of sp3 carbon atoms
                            Descriptors.NumAromaticRings(mol),  # Number of aromatic rings
                            Descriptors.NumAliphaticRings(mol)  # Number of aliphatic rings
                        ]
                        
                        # Combine fingerprint and descriptors
                        feature_vector = np.concatenate([fp_array, descriptors])
                        features[drug_id] = torch.tensor(feature_vector, dtype=torch.float32)
                    else:
                        self.logger.warning(f"Could not create molecule from SMILES for drug {drug_id}")
                        features[drug_id] = torch.zeros(1034, dtype=torch.float32)  # Default empty feature vector
                except Exception as e:
                    self.logger.warning(f"Error extracting features for drug {drug_id}: {str(e)}")
                    features[drug_id] = torch.zeros(1034, dtype=torch.float32)  # Default empty feature vector
            else:
                self.logger.warning(f"No SMILES available for drug {drug_id}")
                features[drug_id] = torch.zeros(1034, dtype=torch.float32)  # Default empty feature vector
        
        self.logger.info(f"Extracted features for {len(features)} drugs")
        return features
    
    def extract_protein_features(self, proteins_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Extract features for protein nodes
        
        Args:
            proteins_data: List of protein dictionaries
            
        Returns:
            Dictionary of protein ID to feature tensor
        """
        self.logger.info(f"Extracting features for {len(proteins_data)} proteins")
        
        features = {}
        
        # For proteins, we'll use one-hot encoding of amino acids in the sequence
        # This is a simplified approach; in practice, you might want to use pre-trained protein embeddings
        
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        for protein in tqdm(proteins_data, desc="Extracting protein features"):
            protein_id = protein["id"]
            
            # Look for sequence in polypeptides
            sequence = None
            for polypeptide in protein.get("polypeptides", []):
                if polypeptide.get("amino_acid_sequence") and polypeptide["amino_acid_sequence"].get("sequence"):
                    sequence = polypeptide["amino_acid_sequence"]["sequence"]
                    break
            
            if sequence:
                # Count amino acids
                aa_counts = np.zeros(len(amino_acids))
                valid_count = 0
                
                for aa in sequence:
                    if aa in aa_to_idx:
                        aa_counts[aa_to_idx[aa]] += 1
                        valid_count += 1
                
                # Normalize by sequence length
                if valid_count > 0:
                    aa_counts = aa_counts / valid_count
                
                # Add basic sequence properties
                seq_properties = np.array([
                    len(sequence),  # Sequence length
                    sum(1 for aa in sequence if aa in "ACGPSTWY") / len(sequence),  # Hydrophobic ratio
                    sum(1 for aa in sequence if aa in "RKHDENQ") / len(sequence),  # Charged ratio
                    sequence.count("C") / len(sequence),  # Cysteine ratio (for disulfide bonds)
                    (sequence.count("K") + sequence.count("R")) / len(sequence),  # Positive charge ratio
                    (sequence.count("D") + sequence.count("E")) / len(sequence)  # Negative charge ratio
                ])
                
                # Combine features
                feature_vector = np.concatenate([aa_counts, seq_properties])
                features[protein_id] = torch.tensor(feature_vector, dtype=torch.float32)
            else:
                self.logger.warning(f"No sequence available for protein {protein_id}")
                features[protein_id] = torch.zeros(len(amino_acids) + 6, dtype=torch.float32)  # Default empty feature vector
        
        self.logger.info(f"Extracted features for {len(features)} proteins")
        return features
    
    def extract_disease_features(self, diseases_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Extract features for disease nodes
        
        Args:
            diseases_data: List of disease dictionaries
            
        Returns:
            Dictionary of disease ID to feature tensor
        """
        self.logger.info(f"Extracting features for {len(diseases_data)} diseases")
        
        features = {}
        
        # For diseases, we'll create embeddings based on their associations
        # This is a placeholder; in a real implementation, you might use text embeddings from descriptions
        
        for disease in tqdm(diseases_data, desc="Extracting disease features"):
            disease_id = disease["id"]
            
            # Create a simple feature vector based on available properties
            feature_list = []
            
            # Number of synonyms
            feature_list.append(len(disease.get("synonyms", [])))
            
            # Number of gene associations
            feature_list.append(len(disease.get("gene_associations", [])))
            
            # Placeholder for other features
            feature_list.extend([0] * 8)  # Add 8 placeholder features to make a 10-dimensional vector
            
            features[disease_id] = torch.tensor(feature_list, dtype=torch.float32)
        
        self.logger.info(f"Extracted features for {len(features)} diseases")
        return features
    
    def normalize_features(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize features
        
        Args:
            features_dict: Dictionary of node ID to feature tensor
            
        Returns:
            Dictionary of normalized features
        """
        self.logger.info(f"Normalizing features for {len(features_dict)} nodes")
        
        # Extract all feature vectors
        feature_tensors = list(features_dict.values())
        
        # Stack into a single tensor
        all_features = torch.stack(feature_tensors)
        
        # Compute mean and std
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        
        # Replace zeros in std with ones to avoid division by zero
        std[std == 0] = 1.0
        
        # Normalize features
        normalized_features = {}
        for node_id, feature in features_dict.items():
            normalized_features[node_id] = (feature - mean) / std
        
        return normalized_features