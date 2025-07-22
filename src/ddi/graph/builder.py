# src/ddi/graph/builder.py

import os
import logging
import pickle
import json
import networkx as nx
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
import argparse

class KnowledgeGraphBuilder:
    """Builds a knowledge graph from multiple data sources, handling ID mapping."""

    def __init__(self, output_dir: str, disease_mapping_path: str, protein_mapping_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph = nx.MultiDiGraph()
        self.node_types: Dict[str, int] = {}
        self.edge_types: Dict[str, int] = {}
        self.disease_mapping = self._load_disease_mapping(disease_mapping_path)
        self.protein_mapping = self._load_protein_mapping(protein_mapping_path)

    def _load_disease_mapping(self, mapping_path: str) -> Dict[str, str]:
        self.logger.info(f"Loading disease mapping from {mapping_path}")
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.error(f"Failed to load or parse disease mapping file: {mapping_path}", exc_info=True)
            return {}

    def _load_protein_mapping(self, mapping_path: str) -> Dict[str, str]:
        self.logger.info(f"Loading protein mapping from {mapping_path}")
        try:
            df = pd.read_csv(mapping_path)
            return pd.Series(df.UniProt_ID.values, index=df.Ensembl_ID).to_dict()
        except FileNotFoundError:
            self.logger.error(f"Protein mapping file not found: {mapping_path}")
            return {}
        except Exception as e:
            self.logger.error(f"An error occurred loading the protein mapping CSV: {e}")
            return {}

    def _add_node(self, node_id: str, node_type: str, **properties):
        # Deduplicate properties by normalized key (Cypher is case-insensitive)
        clean_props = _clean_dupe_keys(properties, node_id=node_id, context="Node")
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=node_type, **clean_props)
            self.node_types[node_type] = self.node_types.get(node_type, 0) + 1

    def _add_edge(self, source: str, target: str, edge_type: str, **properties):
        clean_props = _clean_dupe_keys(properties, node_id=f"{source}->{target}", context="Edge")
        self.graph.add_edge(source, target, key=edge_type, type=edge_type, **clean_props)
        self.edge_types[edge_type] = self.edge_types.get(edge_type, 0) + 1


    def build_graph_from_drugbank(self, drugbank_data: List[Dict[str, Any]]):
        self.logger.info(f"Adding {len(drugbank_data)} drugs and their relationships from DrugBank.")
        for drug in tqdm(drugbank_data, desc="Processing DrugBank Drugs"):
            drug_id = drug.get("drugbank_id")
            if not drug_id: continue
            self._add_node(drug_id, "drug", **drug)

            for rel_type in ["targets", "enzymes", "transporters", "carriers"]:
                self._add_protein_relationships(drug_id, drug.get(rel_type, []), rel_type)

    def _add_protein_relationships(self, drug_id: str, proteins: List[Dict[str, Any]], relationship_type: str):
        for protein in proteins:
            # <<< FIX: Enforce UniProt ID as the only valid identifier for protein nodes >>>
            # This ensures consistency with the OpenTargets mapping.
            uniprot_id = protein.get("uniprot_id")
            if not uniprot_id:
                continue # Skip this relationship if no UniProt ID is present.
            
            protein_node_id = uniprot_id
            self._add_node(
                protein_node_id,
                "protein",
                name=protein.get("name"),
                gene_name=protein.get("gene_name"),
                organism=protein.get("organism")
            )
            self._add_edge(
                drug_id,
                protein_node_id,
                relationship_type,
                actions="|".join(protein.get("actions", []))
            )

    def add_disease_data(self, disease_data: Dict[str, Any]):
        descriptors = disease_data.get('descriptors', {})
        self.logger.info(f"Adding {len(descriptors)} disease descriptors from MeSH.")
        for mesh_id, disease_info in tqdm(descriptors.items(), desc="Processing MeSH Diseases"):
            prefixed_id = f"MESH:{mesh_id}"
            self._add_node(prefixed_id, "disease", **disease_info)
    
    def add_target_disease_associations(self, associations: pd.DataFrame):
        self.logger.info(f"Adding {len(associations)} target-disease associations from OpenTargets.")
        added_count = 0
        skipped_protein_map = 0
        skipped_disease_map = 0
        skipped_missing_node = 0

        for _, row in tqdm(associations.iterrows(), total=len(associations), desc="Processing Associations"):
            target_id_ot = row['targetId']
            disease_id_ot = row['diseaseId']

            protein_node_id = self.protein_mapping.get(target_id_ot)
            if not protein_node_id:
                skipped_protein_map += 1
                continue

            disease_node_id = self.disease_mapping.get(disease_id_ot)
            if not disease_node_id:
                skipped_disease_map += 1
                continue

            #  create the node on-the-fly when it is missing
            if not self.graph.has_node(protein_node_id):
                self._add_node(protein_node_id, "protein", name=None)

            if not self.graph.has_node(disease_node_id):
                self._add_node(disease_node_id, "disease", name=None)

            if self.graph.has_node(protein_node_id) and self.graph.has_node(disease_node_id):
                self._add_edge(
                    protein_node_id,
                    disease_node_id,
                    "associated_with",
                    score=row.get('score'),
                    data_source="opentargets"
                )
                added_count += 1
            else:
                skipped_missing_node += 1
        
        self.logger.info(f"Successfully added {added_count} target-disease associations.")
        if skipped_protein_map > 0: self.logger.warning(f"Skipped {skipped_protein_map} due to missing protein mapping.")
        if skipped_disease_map > 0: self.logger.warning(f"Skipped {skipped_disease_map} due to missing disease mapping.")
        if skipped_missing_node > 0: self.logger.warning(f"Skipped {skipped_missing_node} because mapped node was not in graph.")

    def save_graph(self, formats: List[str] = ["graphml", "pickle"]):
        def _clean_attrs(attrs: dict):
            """Helper to convert list attributes to strings for GraphML."""
            return {k: ("|".join(map(str, v)) if isinstance(v, list) else v)
                    for k, v in attrs.items() if v is not None}

        for fmt in formats:
            output_path = os.path.join(self.output_dir, f"ddi_knowledge_graph.{fmt}")
            self.logger.info(f"Saving graph to {output_path}...")
            try:
                if fmt == "graphml":
                    # Create a new graph with cleaned attributes for GraphML export
                    graph_for_export = nx.MultiDiGraph()
                    for n, d in self.graph.nodes(data=True):
                        graph_for_export.add_node(n, **_clean_attrs(d))
                    for u, v, k, d in self.graph.edges(keys=True, data=True):
                        graph_for_export.add_edge(u, v, key=k, **_clean_attrs(d))
                    nx.write_graphml(graph_for_export, output_path)
                elif fmt == "pickle":
                    # For pickle, save the original graph with Python objects
                    with open(output_path, "wb") as f:
                        pickle.dump(self.graph, f)
                self.logger.info(f"Successfully saved graph as {fmt}.")
            except Exception as e:
                self.logger.error(f"Error saving graph as {fmt}: {e}")

    def get_statistics(self):
        # ... (rest of the functions remain the same)
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self.node_types,
            "edge_types": self.edge_types,
        }
        if self.graph.number_of_nodes() > 0:
            stats["num_connected_components"] = nx.number_connected_components(self.graph.to_undirected())
            largest_cc = max(nx.connected_components(self.graph.to_undirected()), key=len)
            stats["largest_component_size"] = len(largest_cc)
        return stats

def _clean_dupe_keys(properties: dict, node_id:str = None, context:str = "") -> dict:
        """Ensure no duplicate keys for Cypher/Memgraph import (case-insensitive!)"""
        seen = set()
        new_props = {}
        dups = set()
        # If properties come from a merge of multiple dicts, keys might collide
        for k, v in properties.items():
            k_norm = k.lower()
            if k_norm in seen:
                dups.add(k)
            else:
                seen.add(k_norm)
                new_props[k] = v
        if dups:
            import logging
            logging.getLogger("dupe_clean").warning(
                f"[Dupe WARNING] {context} {node_id} duplicate keys ignored: {dups}"
            )
        return new_props

def load_data(file_path: str):
    logger = logging.getLogger("load_data")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}"); return None
    if file_path.endswith('.pickle'):
        with open(file_path, 'rb') as f: return pickle.load(f)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f: return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Build an integrated drug-disease knowledge graph.")
    # Add arguments
    parser.add_argument("--drugbank", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--opentargets", required=True)
    parser.add_argument("--disease_mapping", required=True)
    parser.add_argument("--protein_mapping", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_file", default="graph_build.log")
    args = parser.parse_args()

    # Setup logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file)])
    
    logger = logging.getLogger("graph_build_main")
    logger.info("Starting knowledge graph construction...")

    # Load data
    drugbank_data = load_data(args.drugbank)
    mesh_data = load_data(args.mesh)
    opentargets_data = load_data(args.opentargets)
    
    if not all([drugbank_data, mesh_data, opentargets_data is not None]):
        logger.error("Failed to load one or more data files. Aborting.")
        return

    # Build graph
    builder = KnowledgeGraphBuilder(args.output_dir, args.disease_mapping, args.protein_mapping)
    builder.build_graph_from_drugbank(drugbank_data)
    builder.add_disease_data(mesh_data)
    builder.add_target_disease_associations(opentargets_data)

    # Finalize and Save
    logger.info("Graph construction finished. Generating statistics...")
    stats = builder.get_statistics()
    logger.info(f"Final Graph Statistics: {json.dumps(stats, indent=2)}")
    builder.save_graph()
    logger.info("Graph build process complete.")

if __name__ == "__main__":
    main()
