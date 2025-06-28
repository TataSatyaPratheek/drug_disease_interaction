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
import argparse
from datetime import datetime

# PyG imports remain the same
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

    def __init__(self, output_dir: str = None, disease_mapping_path: Optional[str] = None): # Added mapping path
        """Initialize the graph builder

        Args:
            output_dir: Directory to save outputs
            disease_mapping_path: Path to the precomputed disease mapping JSON file (optional)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir or "data/graph/full"
        os.makedirs(self.output_dir, exist_ok=True)

        self.graph = nx.MultiDiGraph()
        self.node_types = {}
        self.edge_types = {}
        self.node_counter = 0
        self.edge_counter = 0

        # Load disease mapping
        self.disease_mapping: Dict[str, str] = {} # EFO/MONDO -> MeSH
        if disease_mapping_path:
            self._load_disease_mapping(disease_mapping_path)

    def _load_disease_mapping(self, mapping_path: str):
        """Loads the disease ID mapping file."""
        self.logger.info(f"Loading disease mapping from {mapping_path}")
        try:
            with open(mapping_path, 'r') as f:
                self.disease_mapping = json.load(f)
            self.logger.info(f"Loaded {len(self.disease_mapping)} disease mappings.")
        except FileNotFoundError:
            self.logger.error(f"Disease mapping file not found: {mapping_path}. Proceeding without mapping.")
        except json.JSONDecodeError:
             self.logger.error(f"Error decoding JSON from mapping file: {mapping_path}. Proceeding without mapping.")
        except Exception as e:
            self.logger.error(f"Error loading disease mapping file: {e}. Proceeding without mapping.")

    # --- build_graph_from_drugbank, _add_drugs, _add_protein_relationships remain the same ---
    def build_graph_from_drugbank(self, drugbank_data: Dict[str, Any]) -> nx.MultiDiGraph:
        """Build knowledge graph from DrugBank data

        Args:
            drugbank_data: Parsed DrugBank data

        Returns:
            NetworkX MultiDiGraph
        """
        self.logger.info("Building knowledge graph from DrugBank data")
        self._add_drugs(drugbank_data.get("drugs", []))
        self.logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges after DrugBank processing")
        return self.graph

    def _add_drugs(self, drugs: List[Dict[str, Any]]) -> None:
        """Add drugs to the graph"""
        self.logger.info(f"Adding {len(drugs)} drugs to the graph")
        for drug in tqdm(drugs, desc="Processing drugs"):
            drug_id = drug.get("drugbank_id")
            if not drug_id: continue
            if self.graph.has_node(drug_id): continue

            properties = {}
            for prop in ["type", "name", "description", "cas_number", "unii", "state", "indication", "pharmacodynamics", "mechanism_of_action", "classification", "synonyms"]: # Added more props
                if prop in drug and drug[prop] is not None:
                    properties[prop] = drug[prop]

            self._add_node(
                node_id=drug_id,
                node_type="drug",
                name=drug.get("name", drug_id),
                properties=properties
            )

            # Categories (MeSH IDs)
            if "categories" in drug:
                for category in drug.get("categories", []):
                    mesh_id = category.get("mesh_id")
                    if mesh_id and mesh_id.startswith("D"): # Ensure it looks like a MeSH ID
                        mesh_id_prefixed = f"MESH:{mesh_id}" # Use prefixed ID
                        category_name = category.get("category")
                        if not self.graph.has_node(mesh_id_prefixed):
                            self._add_node(
                                node_id=mesh_id_prefixed,
                                node_type="disease", # Treat MeSH categories as potential diseases
                                name=category_name or mesh_id_prefixed,
                                properties={"source": "mesh_category", "alt_ids": []} # Initialize alt_ids
                            )
                        self._add_edge(drug_id, mesh_id_prefixed, "has_category", {})

            # Targets, Enzymes, Transporters, Carriers
            if "targets" in drug: self._add_protein_relationships(drug_id, drug["targets"], "targets")
            if "enzymes" in drug: self._add_protein_relationships(drug_id, drug["enzymes"], "metabolized_by")
            if "transporters" in drug: self._add_protein_relationships(drug_id, drug["transporters"], "transported_by")
            if "carriers" in drug: self._add_protein_relationships(drug_id, drug["carriers"], "carried_by")

    def _add_protein_relationships(self, drug_id: str, proteins: List[Dict[str, Any]], relationship_type: str) -> None:
        """Add protein relationships (targets, enzymes, carriers, transporters)"""
        for protein in proteins:
            protein_id = protein.get("id") # DrugBank protein ID
            if not protein_id: continue

            # --- Normalize to UniProt ID if possible ---
            polypeptides = protein.get("polypeptides", [])
            primary_polypeptide_id = None # Will hold the preferred ID (UniProt > DB Polypeptide > DB Protein)
            uniprot_id = None

            if polypeptides:
                # Try to find UniProt ID from the first polypeptide's external IDs
                first_poly = polypeptides[0]
                poly_id = first_poly.get("id") # DrugBank polypeptide ID
                ext_ids = first_poly.get("external_identifiers", [])
                for ext_id in ext_ids:
                    if ext_id.get("resource") == "UniProtKB":
                        uniprot_id = ext_id.get("identifier")
                        break
                primary_polypeptide_id = uniprot_id if uniprot_id else poly_id
            # -------------------------------------------

            # Use UniProt ID if found, otherwise DrugBank protein ID as fallback node ID
            node_id_to_use = primary_polypeptide_id if primary_polypeptide_id else protein_id
            node_type = "protein" # Keep generic type 'protein'

            if not self.graph.has_node(node_id_to_use):
                properties = {
                    "organism": protein.get("organism"),
                    "known_action": protein.get("known_action"),
                    "db_protein_id": protein_id, # Store original DB protein ID
                    "db_polypeptide_ids": [p.get("id") for p in polypeptides if p.get("id")],
                    "uniprot_id": uniprot_id, # Store UniProt if found
                    # Add gene name from first polypeptide if available
                    "gene_name": polypeptides[0].get("gene_name") if polypeptides else None,
                    # Add function info
                    "general_function": polypeptides[0].get("general_function") if polypeptides else None,
                }
                # Remove None properties
                properties = {k: v for k, v in properties.items() if v is not None}

                self._add_node(
                    node_id=node_id_to_use,
                    node_type=node_type,
                    name=protein.get("name", node_id_to_use),
                    properties=properties
                )
            else:
                # Update existing node with UniProt ID if found later
                if uniprot_id and 'uniprot_id' not in self.graph.nodes[node_id_to_use]:
                     self.graph.nodes[node_id_to_use]['uniprot_id'] = uniprot_id
                     self.logger.debug(f"Updated node {node_id_to_use} with UniProt ID {uniprot_id}")


            # Add drug -> protein edge
            actions = protein.get("actions", [])
            action_str = "|".join(actions) if actions else None
            self._add_edge(
                source=drug_id,
                target=node_id_to_use, # Use the normalized ID
                edge_type=relationship_type,
                properties={"actions": action_str}
            )

    # --- MODIFIED add_disease_data ---
    def add_disease_data(self, disease_data: Dict[str, Any]) -> None:
        """Add disease data to the graph (primarily from MeSH)"""
        if not disease_data:
            self.logger.warning("No disease data provided")
            return

        # Assuming MeSH format from the parser: {'descriptors': {...}, 'version': '2025', ...}
        descriptors = disease_data.get('descriptors', {})
        if not descriptors:
             self.logger.warning("No 'descriptors' found in disease data.")
             return

        self.logger.info(f"Adding {len(descriptors)} potential disease descriptors from MeSH data")

        added_count = 0
        for mesh_id_plain, disease in tqdm(descriptors.items(), desc="Processing MeSH diseases"):
            if not isinstance(disease, dict): continue # Skip if format is wrong

            # Use prefixed ID
            mesh_id = f"MESH:{mesh_id_plain}"

            # Skip if node already exists (might have been added via DrugBank category)
            if self.graph.has_node(mesh_id):
                # Ensure alt_ids list exists if node was added previously without it
                if 'alt_ids' not in self.graph.nodes[mesh_id]:
                    self.graph.nodes[mesh_id]['alt_ids'] = []
                continue

            properties = {}
            for prop in ["description", "synonyms", "tree_numbers"]:
                if prop in disease and disease[prop] is not None:
                    properties[prop] = disease[prop]

            # Initialize alt_ids list
            properties['alt_ids'] = []
            properties['source'] = 'mesh' # Mark source

            self._add_node(
                node_id=mesh_id,
                node_type="disease",
                name=disease.get("name", mesh_id),
                properties=properties
            )
            added_count += 1

            # Add parent-child relationships (using prefixed IDs)
            # Note: MeSH parser's extract_disease_taxonomy already creates 'parents' list
            # We need to ensure those parent IDs are also prefixed if adding edges here.
            # Alternatively, rely on tree numbers for hierarchy if needed later.
            # Let's skip adding hierarchy edges here for simplicity, assuming MeSH parser handles it
            # or it can be inferred from tree numbers later.

        self.logger.info(f"Added {added_count} new disease nodes from MeSH data.")

    # --- add_drug_disease_associations remains the same ---
    def add_drug_disease_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add drug-disease associations (e.g., from indications file)"""
        if not associations:
            self.logger.warning("No explicit drug-disease associations provided (e.g., indications)")
            return

        self.logger.info(f"Adding {len(associations)} explicit drug-disease associations")
        added_count = 0
        for assoc in tqdm(associations, desc="Processing drug-disease indications"):
            drug_id = assoc.get("drug_id") # Expecting DrugBank ID
            disease_id = assoc.get("disease_id") # Expecting MeSH ID (plain or prefixed?)
            source = assoc.get("source", "indication_file")

            if not drug_id or not disease_id: continue

            # --- Normalize disease ID (assume input might be plain MeSH ID) ---
            if disease_id.startswith("D") and not disease_id.startswith("MESH:"):
                disease_node_id = f"MESH:{disease_id}"
            else:
                disease_node_id = disease_id # Assume already prefixed or is EFO/MONDO
            # --------------------------------------------------------------------

            # Check if nodes exist
            if not self.graph.has_node(drug_id):
                self.logger.debug(f"Skipping indication: Drug {drug_id} not in graph.")
                continue
            if not self.graph.has_node(disease_node_id):
                 # Try mapping if it's an EFO/MONDO ID
                 if disease_node_id in self.disease_mapping:
                     mesh_id = self.disease_mapping[disease_node_id]
                     if self.graph.has_node(mesh_id):
                         disease_node_id = mesh_id # Use the mapped MeSH ID
                     else:
                         self.logger.debug(f"Skipping indication: Mapped MeSH ID {mesh_id} for {disease_node_id} not in graph.")
                         continue
                 else:
                    self.logger.debug(f"Skipping indication: Disease {disease_node_id} not in graph and no mapping found.")
                    continue

            # Add edge
            properties = {"source": source}
            self._add_edge(drug_id, disease_node_id, "treats", properties)
            added_count += 1
        self.logger.info(f"Added {added_count} 'treats' edges from indications.")


    # --- add_drug_target_associations remains the same ---
    def add_drug_target_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add drug-target associations (e.g., from OpenTargets)"""
        # This function might need adjustment based on the IDs used in the input associations
        # Assuming drug_id is ChEMBL and target_id is Ensembl/UniProt
        # Requires mapping ChEMBL -> DrugBank and Ensembl -> UniProt
        # For now, let's assume the input is already mapped or skip if not feasible
        self.logger.warning("add_drug_target_associations currently assumes input IDs match graph IDs (DrugBank/UniProt). Mapping might be needed.")
        if not associations:
            self.logger.warning("No drug-target associations provided")
            return

        self.logger.info(f"Adding {len(associations)} drug-target associations")
        added_count = 0
        for assoc in tqdm(associations, desc="Processing drug-target associations"):
            drug_id = assoc.get("drug_id") # Assumed to be DrugBank ID for now
            target_id = assoc.get("target_id") # Assumed to be UniProt ID for now

            if not drug_id or not target_id: continue
            if not self.graph.has_node(drug_id) or not self.graph.has_node(target_id): continue

            properties = {}
            for prop in ["score", "confidence", "mechanism", "action", "source"]: # Added source
                if prop in assoc and assoc[prop] is not None:
                    properties[prop] = assoc[prop]

            self._add_edge(drug_id, target_id, "targets", properties)
            added_count += 1
        self.logger.info(f"Added {added_count} 'targets' edges.")


    # --- MODIFIED add_target_disease_associations ---
    def add_target_disease_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Add target-disease associations (e.g., from OpenTargets), using disease mapping."""
        if not associations:
            self.logger.warning("No target-disease associations provided")
            return

        self.logger.info(f"Adding {len(associations)} target-disease associations")
        added_count = 0
        skipped_unmapped_disease = 0
        skipped_missing_nodes = 0

        for assoc in tqdm(associations, desc="Processing target-disease associations"):
            target_id_ot = assoc.get("target_id") # Expecting Ensembl or UniProt
            disease_id_ot = assoc.get("disease_id") # Expecting EFO or MONDO

            if not target_id_ot or not disease_id_ot:
                continue

            # --- Normalize Target ID (Assume UniProt if possible, else keep original) ---
            # This part depends heavily on how proteins were added. Assuming node IDs are UniProt if available.
            # A more robust way would be to build a mapping from Ensembl -> UniProt if needed.
            # For now, we just check if the target_id_ot exists directly.
            target_node_id = target_id_ot # Use the ID directly for now
            if not self.graph.has_node(target_node_id):
                 # Maybe check if it's stored as a property if normalization happened differently?
                 # This highlights the importance of consistent protein ID handling.
                 # self.logger.debug(f"Skipping TD assoc: Target {target_node_id} not in graph.")
                 skipped_missing_nodes += 1
                 continue
            # --------------------------------------------------------------------------

            # --- Map Disease ID ---
            disease_node_id = None
            mapped_mesh_id = self.disease_mapping.get(disease_id_ot)

            if mapped_mesh_id:
                if self.graph.has_node(mapped_mesh_id):
                    disease_node_id = mapped_mesh_id
                    # Add the original OT ID as an alternative ID to the MeSH node
                    if 'alt_ids' in self.graph.nodes[disease_node_id] and disease_id_ot not in self.graph.nodes[disease_node_id]['alt_ids']:
                        self.graph.nodes[disease_node_id]['alt_ids'].append(disease_id_ot)
                    elif 'alt_ids' not in self.graph.nodes[disease_node_id]:
                         self.graph.nodes[disease_node_id]['alt_ids'] = [disease_id_ot]
                else:
                    # Mapped MeSH ID doesn't exist in graph (shouldn't happen if MeSH data added first)
                    self.logger.warning(f"Mapped MeSH ID {mapped_mesh_id} for {disease_id_ot} not found in graph. Skipping edge.")
                    skipped_missing_nodes += 1
                    continue
            else:
                # No mapping found for EFO/MONDO ID
                skipped_unmapped_disease += 1
                # Option 1: Skip the association
                # continue
                # Option 2: Add the EFO/MONDO node if it doesn't exist
                if not self.graph.has_node(disease_id_ot):
                    self._add_node(
                        node_id=disease_id_ot,
                        node_type="disease",
                        name=disease_id_ot, # Name is unknown without OT disease entities
                        properties={"source": "opentargets", "alt_ids": []}
                    )
                disease_node_id = disease_id_ot # Use the original OT ID

            # ----------------------

            # Add target-disease relationship if we have a valid disease node ID
            if disease_node_id:
                properties = {}
                for prop in ["score", "datasource"]: # Keep relevant props from lean parser
                    if prop in assoc and assoc[prop] is not None:
                        properties[prop] = assoc[prop]
                properties['source'] = 'opentargets' # Add source explicitly

                self._add_edge(
                    source=target_node_id,
                    target=disease_node_id, # Use the mapped or original disease ID
                    edge_type="associated_with",
                    properties=properties
                )
                added_count += 1
            # else: node was skipped due to mapping issues

        self.logger.info(f"Added {added_count} target-disease associations to the graph.")
        if skipped_missing_nodes > 0:
             self.logger.warning(f"Skipped {skipped_missing_nodes} TD associations due to missing target or mapped disease nodes.")
        if skipped_unmapped_disease > 0:
            self.logger.info(f"Encountered {skipped_unmapped_disease} unique OpenTargets disease IDs without a MeSH mapping (these were added as separate nodes or skipped).")


    # --- _add_node, _add_edge, _normalize_id, get_statistics, save_graph, _convert_to_pyg remain the same ---
    def _add_node(self, node_id: str, node_type: str, name: str, properties: Dict[str, Any]) -> None:
        """Add a node to the graph"""
        node_attrs = {"type": node_type, "name": name, **properties}
        # Ensure all properties are suitable for storage (e.g., convert complex objects if needed)
        node_attrs = {k: v for k, v in node_attrs.items() if v is not None} # Remove None values
        self.graph.add_node(node_id, **node_attrs)
        if node_type not in self.node_types: self.node_types[node_type] = 0
        self.node_types[node_type] += 1
        self.node_counter += 1

    def _add_edge(self, source: str, target: str, edge_type: str, properties: Dict[str, Any]) -> None:
        """Add an edge to the graph"""
        edge_attrs = {"type": edge_type, **properties}
        edge_attrs = {k: v for k, v in edge_attrs.items() if v is not None} # Remove None values
        self.graph.add_edge(source, target, **edge_attrs)
        if edge_type not in self.edge_types: self.edge_types[edge_type] = 0
        self.edge_types[edge_type] += 1
        self.edge_counter += 1

    def _normalize_id(self, text: str) -> str:
        """Normalize text for use as an ID"""
        # This might not be needed if IDs are handled carefully
        if not text: return ""
        try:
            return text.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")
        except:
            return str(text)

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self.node_types,
            "edge_types": self.edge_types
        }
        if self.graph.number_of_nodes() > 0:
            degrees = [d for _, d in self.graph.degree()]
            stats["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
            try: # Add try-except for component calculation
                undirected = self.graph.to_undirected()
                connected_components = list(nx.connected_components(undirected))
                stats["num_connected_components"] = len(connected_components)
                if connected_components:
                    largest_cc = max(connected_components, key=len)
                    stats["largest_component_size"] = len(largest_cc)
                    stats["largest_component_percentage"] = len(largest_cc) / self.graph.number_of_nodes() * 100
            except Exception as e:
                 self.logger.warning(f"Could not calculate connected components: {e}")
                 stats["num_connected_components"] = "Error"

        return stats

    def save_graph(self, formats: List[str] = ["graphml", "pickle"]) -> Dict[str, str]:
        """Save graph to files"""
        output_files = {}
        for fmt in formats:
            output_path = os.path.join(self.output_dir, f"knowledge_graph.{fmt}")
            try:
                if fmt == "graphml":
                    self.logger.info("Preparing simplified graph for GraphML export...")
                    clean_graph_ml = nx.MultiDiGraph()
                    for n, attrs in self.graph.nodes(data=True):
                        ml_attrs = {'type': str(attrs.get('type', '')), 'name': str(attrs.get('name', n))}
                        ml_attrs = {k: v for k, v in ml_attrs.items() if isinstance(v, (str, int, float, bool))}
                        clean_graph_ml.add_node(n, **ml_attrs)
                    for u, v, key, attrs in self.graph.edges(keys=True, data=True):
                        if clean_graph_ml.has_node(u) and clean_graph_ml.has_node(v):
                            ml_attrs = {'type': str(attrs.get('type', ''))}
                            ml_attrs = {k: v for k, v in ml_attrs.items() if isinstance(v, (str, int, float, bool))}
                            clean_graph_ml.add_edge(u, v, key=key, **ml_attrs)
                    nx.write_graphml(clean_graph_ml, output_path)
                    self.logger.info(f"Saved simplified graph in GraphML format to {output_path}")
                elif fmt == "pickle":
                    with open(output_path, "wb") as f: pickle.dump(self.graph, f)
                    self.logger.info(f"Saved graph in pickle format to {output_path}")
                elif fmt == "pyg":
                    if PYG_AVAILABLE:
                        pyg_graph = self._convert_to_pyg()
                        if pyg_graph:
                            with open(output_path, "wb") as f: pickle.dump(pyg_graph, f)
                            self.logger.info(f"Saved graph in PyG format (via pickle) to {output_path}")
                        else: continue # Skip adding to output_files if conversion failed
                    else:
                        self.logger.warning("PyTorch Geometric not available. Skipping PyG format export.")
                        continue
                else:
                    self.logger.warning(f"Unsupported save format: {fmt}")
                    continue
                output_files[fmt] = output_path # Add path only if save was successful or format unsupported
            except Exception as e:
                self.logger.error(f"Error writing {fmt} file: {e}", exc_info=True)

        # Save mappings
        mappings = {"node_types": self.node_types, "edge_types": self.edge_types, "statistics": self.get_statistics()}
        mappings_path = os.path.join(self.output_dir, "graph_mappings.json")
        try:
            with open(mappings_path, "w") as f: json.dump(mappings, f, indent=2)
            output_files["mappings"] = mappings_path
            self.logger.info(f"Saved graph mappings to {mappings_path}")
        except Exception as e: self.logger.error(f"Error writing mappings JSON file: {e}")

        return output_files

    def _convert_to_pyg(self):
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        # --- This function remains the same ---
        if not PYG_AVAILABLE:
            self.logger.error("PyTorch Geometric not available. Cannot convert to PyG format.")
            return None
        self.logger.info("Converting NetworkX graph to PyTorch Geometric format")
        try:
            node_type_to_id = {t: i for i, t in enumerate(sorted(self.node_types.keys()))}
            edge_type_to_id = {t: i for i, t in enumerate(sorted(self.edge_types.keys()))}
            nodes = list(self.graph.nodes)
            node_type_dict = {n: self.graph.nodes[n]["type"] for n in nodes}
            node_types_tensor = torch.tensor([node_type_to_id[node_type_dict[n]] for n in nodes], dtype=torch.long)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            edge_index = [[], []]
            edge_types_list = []
            for u, v, data in self.graph.edges(data=True):
                edge_index[0].append(node_to_idx[u])
                edge_index[1].append(node_to_idx[v])
                edge_types_list.append(edge_type_to_id[data["type"]])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_type = torch.tensor(edge_types_list, dtype=torch.long)
            data = Data(
                x=node_types_tensor.view(-1, 1), # Use plural 'node_types_tensor'
                edge_index=edge_index,
                edge_attr=edge_type.view(-1, 1),
                num_nodes=len(nodes)
            )
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


# --- Add this helper function (if not already present) ---
def load_data(file_path: str) -> Any:
    """Load data from file with error handling
    
    Args:
        file_path: Path to data file
        
    Returns:
        Loaded data
    """
    logger = logging.getLogger("load_data") # Use a specific logger
    if not file_path:
        logger.error("No file path provided to load_data.")
        return None
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        if file_path.endswith('.pickle') or file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling file {file_path}: {e}. File might be corrupted or not a pickle file.", exc_info=True)
        return None
    except EOFError as e:
         logger.error(f"EOFError loading pickle file {file_path}: {e}. File might be empty or truncated.", exc_info=True)
         return None
    except json.JSONDecodeError as e:
         logger.error(f"Error decoding JSON from file: {file_path}. Error: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading file {file_path}: {e}", exc_info=True)
        return None


# --- Replace the existing main function with this one ---
def main():
    parser = argparse.ArgumentParser(description="Build integrated knowledge graph")
    parser.add_argument("--drugbank", required=True, help="Path to processed DrugBank data")
    # --- Corrected Arguments ---
    parser.add_argument("--mesh", required=True, help="Path to processed MeSH disease data (e.g., mesh_data_2025.pickle)")
    parser.add_argument("--opentargets_td_assoc", required=True, help="Path to processed OpenTargets target-disease associations")
    parser.add_argument("--disease_mapping", required=True, help="Path to the precomputed disease mapping JSON file (EFO/MONDO -> MeSH)")
    # --- End Corrected Arguments ---
    # Optional arguments for other associations if needed later
    # parser.add_argument("--indications", help="Path to drug-disease indications file (optional)")
    # parser.add_argument("--ot_dt_assoc", help="Path to OpenTargets drug-target associations (optional)")
    parser.add_argument("--output", required=True, help="Output directory for graph")
    parser.add_argument("--log_file", help="Path to log file")
    parser.add_argument("--formats", default="graphml,pickle", help="Comma-separated list of output formats (pyg optional)")
    args = parser.parse_args()

    # Set up logging using the utility function if available, otherwise basic config
    try:
        from ddi.utils.logging import setup_logging
        setup_logging(args.log_file, level=logging.INFO)
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if args.log_file:
             file_handler = logging.FileHandler(args.log_file)
             file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
             logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger("build_graph")

    logger.info("Starting knowledge graph construction with ID mapping")

    # Initialize graph builder WITH mapping
    graph_builder = KnowledgeGraphBuilder(output_dir=args.output, disease_mapping_path=args.disease_mapping)

    # Load and add DrugBank data
    logger.info(f"Loading DrugBank data from {args.drugbank}")
    drugbank_data = load_data(args.drugbank)
    if drugbank_data:
        graph_builder.build_graph_from_drugbank(drugbank_data) # Adds drugs, proteins, MeSH categories
    else:
        logger.error("Failed to load DrugBank data. Cannot proceed.")
        return # Exit if critical data is missing

    # Load and add MeSH disease data (ensures MeSH nodes exist before OT processing)
    logger.info(f"Loading MeSH disease data from {args.mesh}")
    mesh_data = load_data(args.mesh)
    if mesh_data:
        graph_builder.add_disease_data(mesh_data) # Adds MeSH disease nodes
    else:
        logger.error("Failed to load MeSH data. Cannot proceed.")
        return # Exit if critical data is missing

    # Load and add OpenTargets target-disease associations (using mapping)
    logger.info(f"Loading OpenTargets target-disease associations from {args.opentargets_td_assoc}")
    ot_td_associations = load_data(args.opentargets_td_assoc)
    if ot_td_associations is not None: # Check for None, allow empty list
        graph_builder.add_target_disease_associations(ot_td_associations) # This method now uses the mapping
    else:
        logger.warning("Failed to load OpenTargets associations. Continuing without them.")


    # --- Add other association types if files are provided ---
    # Example:
    # indications_path = getattr(args, 'indications', None) # Check if arg exists
    # if indications_path:
    #     logger.info(f"Loading indications from {indications_path}")
    #     indications = load_data(indications_path)
    #     if indications:
    #         graph_builder.add_drug_disease_associations(indications) # Assumes MeSH IDs

    # Get graph statistics
    stats = graph_builder.get_statistics()
    logger.info("Graph statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items(): logger.info(f"    {k}: {v}")
        else: logger.info(f"  {key}: {value}")

    # Save graph
    formats = args.formats.split(',')
    logger.info(f"Saving graph in formats: {formats}")
    output_files = graph_builder.save_graph(formats=formats)
    for fmt, file_path in output_files.items(): logger.info(f"Graph saved in {fmt} format: {file_path}")

    logger.info("Knowledge graph construction complete")

if __name__ == "__main__":
    # Ensure the load_data function is defined before calling main
    main()
