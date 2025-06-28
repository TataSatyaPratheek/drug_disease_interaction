# src/ddi/graph/builder.py

import os
import logging
import pickle
import json
from pathlib import Path
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import argparse

# --- PyG imports ---
PYG_AVAILABLE = False
try:
    import torch
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch or PyTorch Geometric not installed. PyG graph export will not be available.")
    logging.warning("To install: pip install torch torch-geometric")

class KnowledgeGraphBuilder:
    """Builds a knowledge graph from multiple data sources, handling ID mapping."""

    def __init__(self, output_dir: str, disease_mapping_path: str, protein_mapping_path: str):
        """
        Initialize the graph builder.
        
        Args:
            output_dir: Directory to save graph outputs.
            disease_mapping_path: Path to the precomputed disease mapping JSON file (EFO/MONDO -> MeSH).
            protein_mapping_path: Path to the precomputed protein mapping CSV file (Ensembl -> UniProt).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.graph = nx.MultiDiGraph()
        self.node_types: Dict[str, int] = {}
        self.edge_types: Dict[str, int] = {}

        # Load identifier mappings
        self.disease_mapping = self._load_disease_mapping(disease_mapping_path)
        self.protein_mapping = self._load_protein_mapping(protein_mapping_path)

    def _load_disease_mapping(self, mapping_path: str) -> Dict[str, str]:
        """Loads the disease ID mapping file (EFO/MONDO -> MeSH)."""
        self.logger.info(f"Loading disease mapping from {mapping_path}")
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            self.logger.info(f"Loaded {len(mapping)} disease mappings.")
            return mapping
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load disease mapping file: {e}. Proceeding without disease mappings.")
            return {}

    def _load_protein_mapping(self, mapping_path: str) -> Dict[str, str]:
        """Loads the protein ID mapping file (Ensembl -> UniProt)."""
        self.logger.info(f"Loading protein mapping from {mapping_path}")
        try:
            df = pd.read_csv(mapping_path)
            # Create a dictionary for fast lookup
            mapping = pd.Series(df.UniProt_ID.values, index=df.Ensembl_ID).to_dict()
            self.logger.info(f"Loaded {len(mapping)} protein mappings.")
            return mapping
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load protein mapping file: {e}. Proceeding without protein mappings.")
            return {}
        except Exception as e:
            self.logger.error(f"An error occurred loading the protein mapping CSV: {e}")
            return {}

    def _add_node(self, node_id: str, node_type: str, name: str, properties: Dict[str, Any]):
        """Adds a node to the graph if it doesn't already exist."""
        if self.graph.has_node(node_id):
            return
        node_attrs = {"type": node_type, "name": name, **properties}
        node_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        self.graph.add_node(node_id, **node_attrs)
        self.node_types[node_type] = self.node_types.get(node_type, 0) + 1

    def _add_edge(self, source: str, target: str, edge_type: str, properties: Dict[str, Any]):
        """Adds a directed edge to the graph."""
        edge_attrs = {"type": edge_type, **properties}
        edge_attrs = {k: v for k, v in edge_attrs.items() if v is not None}
        self.graph.add_edge(source, target, **edge_attrs)
        self.edge_types[edge_type] = self.edge_types.get(edge_type, 0) + 1

    def build_graph_from_drugbank(self, drugbank_data: List[Dict[str, Any]]):
        """Builds the initial graph structure from DrugBank data."""
        self.logger.info(f"Adding {len(drugbank_data)} drugs and their relationships from DrugBank.")
        for drug in tqdm(drugbank_data, desc="Processing DrugBank Drugs"):
            drug_id = drug.get("drugbank_id")
            if not drug_id:
                continue

            self._add_node(
                node_id=drug_id,
                node_type="drug",
                name=drug.get("name", drug_id),
                properties={k: drug[k] for k in ["description", "indication", "pharmacodynamics", "mechanism_of_action", "classification"] if k in drug}
            )

            # Add protein relationships (targets, enzymes, etc.)
            for rel_type in ["targets", "enzymes", "transporters", "carriers"]:
                if rel_type in drug:
                    self._add_protein_relationships(drug_id, drug[rel_type], rel_type)

    def _add_protein_relationships(self, drug_id: str, proteins: List[Dict[str, Any]], relationship_type: str):
        """Adds protein nodes and their relationships to a drug."""
        for protein in proteins:
            # Prioritize UniProt ID as the canonical node ID
            uniprot_id = None
            if protein.get("polypeptides"):
                for ext_id in protein["polypeptides"][0].get("external_identifiers", []):
                    if ext_id.get("resource") == "UniProtKB":
                        uniprot_id = ext_id.get("identifier")
                        break
            
            protein_node_id = uniprot_id if uniprot_id else protein.get("id")
            if not protein_node_id:
                continue

            self._add_node(
                node_id=protein_node_id,
                node_type="protein",
                name=protein.get("name", protein_node_id),
                properties={
                    "organism": protein.get("organism"),
                    "gene_name": protein["polypeptides"][0].get("gene_name") if protein.get("polypeptides") else None,
                    "db_id": protein.get("id")
                }
            )

            self._add_edge(
                source=drug_id,
                target=protein_node_id,
                edge_type=relationship_type,
                properties={"actions": "|".join(protein.get("actions", []))}
            )

    def add_disease_data(self, disease_data: Dict[str, Any]):
        """Adds disease nodes from MeSH data."""
        descriptors = disease_data.get('descriptors', {})
        self.logger.info(f"Adding {len(descriptors)} disease descriptors from MeSH.")
        for mesh_id, disease_info in tqdm(descriptors.items(), desc="Processing MeSH Diseases"):
            prefixed_id = f"MESH:{mesh_id}"
            self._add_node(
                node_id=prefixed_id,
                node_type="disease",
                name=disease_info.get("name", prefixed_id),
                properties={
                    "description": disease_info.get("description"),
                    "tree_numbers": disease_info.get("tree_numbers"),
                    "source": "mesh"
                }
            )
    
    def add_target_disease_associations(self, associations: pd.DataFrame):
        """Adds target-disease associations from OpenTargets, using pre-loaded mappings."""
        self.logger.info(f"Adding {len(associations)} target-disease associations from OpenTargets.")
        
        added_count = 0
        skipped_protein_map = 0
        skipped_disease_map = 0
        skipped_missing_node = 0

        for _, row in tqdm(associations.iterrows(), total=len(associations), desc="Processing Target-Disease Associations"):
            target_id_ot = row['targetId']  # Ensembl ID
            disease_id_ot = row['diseaseId'] # EFO ID

            # Step 1: Map Protein ID
            protein_node_id = self.protein_mapping.get(target_id_ot)
            if not protein_node_id:
                skipped_protein_map += 1
                continue
            
            if not self.graph.has_node(protein_node_id):
                skipped_missing_node += 1
                continue

            # Step 2: Map Disease ID
            disease_node_id = self.disease_mapping.get(disease_id_ot)
            if not disease_node_id:
                skipped_disease_map += 1
                continue # Skip if no mapping is found

            if not self.graph.has_node(disease_node_id):
                # This case is less likely if MeSH data was added first
                skipped_missing_node += 1
                continue
            
            # Step 3: Add the edge
            self._add_edge(
                source=protein_node_id,
                target=disease_node_id,
                edge_type="associated_with",
                properties={"score": row.get('score'), "source": "opentargets"}
            )
            added_count += 1
        
        self.logger.info(f"Successfully added {added_count} target-disease associations.")
        if skipped_protein_map > 0:
            self.logger.warning(f"Skipped {skipped_protein_map} associations due to missing protein ID mapping.")
        if skipped_disease_map > 0:
            self.logger.warning(f"Skipped {skipped_disease_map} associations due to missing disease ID mapping.")
        if skipped_missing_node > 0:
            self.logger.warning(f"Skipped {skipped_missing_node} associations because the mapped protein/disease node was not in the graph.")

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates and returns statistics about the graph."""
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self.node_types,
            "edge_types": self.edge_types
        }
        if self.graph.number_of_nodes() > 0:
            try:
                undirected_g = self.graph.to_undirected(as_view=True)
                components = list(nx.connected_components(undirected_g))
                stats["num_connected_components"] = len(components)
                if components:
                    largest_cc = max(components, key=len)
                    stats["largest_component_size"] = len(largest_cc)
            except Exception as e:
                self.logger.warning(f"Could not compute connected components: {e}")
        return stats

    def save_graph(self, formats: List[str] = ["graphml", "pickle"]):
        """Saves the graph in specified formats."""
        for fmt in formats:
            output_path = os.path.join(self.output_dir, f"ddi_knowledge_graph.{fmt}")
            self.logger.info(f"Saving graph to {output_path}...")
            try:
                if fmt == "graphml":
                    nx.write_graphml(self.graph, output_path)
                elif fmt == "pickle":
                    with open(output_path, "wb") as f:
                        pickle.dump(self.graph, f)
                elif fmt == "pyg" and PYG_AVAILABLE:
                    pyg_data = self._convert_to_pyg()
                    with open(output_path, "wb") as f:
                        pickle.dump(pyg_data, f)
                else:
                    self.logger.warning(f"Unsupported format or missing libraries for: {fmt}")
                self.logger.info(f"Successfully saved graph as {fmt}.")
            except Exception as e:
                self.logger.error(f"Error saving graph as {fmt}: {e}")

    def _convert_to_pyg(self) -> Optional[Data]:
        """Converts the NetworkX graph to a PyTorch Geometric Data object."""
        if not PYG_AVAILABLE:
            return None
        # Implementation for PyG conversion (can be detailed if needed)
        self.logger.info("PyG conversion is a placeholder in this version.")
        return None

def load_data(file_path: str) -> Any:
    """Helper function to load data from pickle or JSON files."""
    logger = logging.getLogger("load_data")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        if file_path.endswith('.pickle'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Build an integrated drug-disease knowledge graph.")
    parser.add_argument("--drugbank", required=True, help="Path to processed DrugBank data pickle file.")
    parser.add_argument("--mesh", required=True, help="Path to processed MeSH disease data pickle file.")
    parser.add_argument("--opentargets", required=True, help="Path to processed OpenTargets associations pickle file.")
    parser.add_argument("--disease_mapping", required=True, help="Path to the disease mapping JSON file (EFO -> MeSH).")
    parser.add_argument("--protein_mapping", required=True, help="Path to the protein mapping CSV file (Ensembl -> UniProt).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final graph files.")
    parser.add_argument("--log_file", help="Path to log file for the build process.")
    args = parser.parse_args()

    # Setup logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger("graph_build_main")
    logger.info("Starting knowledge graph construction...")

    # Initialize the builder with all necessary mapping paths
    builder = KnowledgeGraphBuilder(
        output_dir=args.output_dir,
        disease_mapping_path=args.disease_mapping,
        protein_mapping_path=args.protein_mapping
    )

    # --- Load Data Sequentially ---
    drugbank_data = load_data(args.drugbank)
    if not drugbank_data:
        logger.error("Failed to load DrugBank data. Aborting.")
        return

    mesh_data = load_data(args.mesh)
    if not mesh_data:
        logger.error("Failed to load MeSH data. Aborting.")
        return

    opentargets_data = load_data(args.opentargets)
    if opentargets_data is None:
        logger.error("Failed to load OpenTargets data. Aborting.")
        return

    # --- Build Graph Sequentially ---
    # 1. Add nodes from DrugBank (drugs and their protein targets)
    builder.build_graph_from_drugbank(drugbank_data)
    
    # 2. Add disease nodes from MeSH
    builder.add_disease_data(mesh_data)

    # 3. Add target-disease associations, which connects the two sets of nodes
    builder.add_target_disease_associations(opentargets_data)

    # --- Finalize and Save ---
    logger.info("Graph construction finished. Generating statistics...")
    stats = builder.get_statistics()
    logger.info(f"Final Graph Statistics: {json.dumps(stats, indent=2)}")

    builder.save_graph(formats=["graphml", "pickle"])
    logger.info("Graph build process complete.")

if __name__ == "__main__":
    main()
