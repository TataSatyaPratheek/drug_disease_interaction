# src/ddi/features/feature_engineering.py
import os
import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import json
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds

class FeatureExtractor:
    """Feature extraction for knowledge graph nodes and edges"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """Initialize the feature extractor
        
        Args:
            graph: NetworkX MultiDiGraph to extract features from
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph = graph
        
        # Cache for computed features
        self._cache = {}
        
        # Mappings for categorical features
        self.categorical_mappings = {}
    
    def extract_graph_features(self, output_dir: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract features for all nodes in the graph
        
        Args:
            output_dir: Directory to save extracted features (optional)
            
        Returns:
            Dictionary mapping node types to feature dictionaries
        """
        # Group nodes by type
        nodes_by_type = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        # Extract features for each node type
        features_by_type = {}
        for node_type, nodes in nodes_by_type.items():
            self.logger.info(f"Extracting features for {len(nodes)} {node_type} nodes")
            
            if node_type == "drug":
                features = self.extract_drug_features(nodes)
            elif node_type == "disease":
                features = self.extract_disease_features(nodes)
            elif node_type in ["protein", "polypeptide"]:
                features = self.extract_protein_features(nodes)
            else:
                # Default features for other node types
                features = self.extract_default_features(nodes)
            
            features_by_type[node_type] = features
        
        # Save features if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each feature set separately
            for node_type, features in features_by_type.items():
                output_path = os.path.join(output_dir, f"{node_type}_features.pickle")
                with open(output_path, "wb") as f:
                    pickle.dump(features, f)
                self.logger.info(f"Saved {node_type} features to {output_path}")
                
            # Save categorical mappings
            if self.categorical_mappings:
                mapping_path = os.path.join(output_dir, "categorical_mappings.json")
                with open(mapping_path, "w") as f:
                    # Convert any non-serializable keys to strings
                    serializable_mappings = {}
                    for k, v in self.categorical_mappings.items():
                        serializable_mappings[k] = {str(kk): vv for kk, vv in v.items()}
                    json.dump(serializable_mappings, f, indent=2)
                self.logger.info(f"Saved categorical mappings to {mapping_path}")
        
        return features_by_type
    
    def extract_drug_features(self, drug_nodes: List[str]) -> Dict[str, np.ndarray]:
        """Extract features for drug nodes
        
        Args:
            drug_nodes: List of drug node IDs
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        if not drug_nodes:
            return {}
            
        # Check cache
        cache_key = "drug_features"
        if cache_key in self._cache:
            # Filter cached features to keep only the requested nodes
            cached_features = self._cache[cache_key]
            return self._filter_features(cached_features, drug_nodes)
        
        self.logger.info(f"Extracting features for {len(drug_nodes)} drug nodes")
        
        # Initialize feature dictionaries
        features = {
            "node_id": np.array(drug_nodes),
            "network_features": np.zeros((len(drug_nodes), 5)),  # Basic network metrics
            "molecular_features": np.zeros((len(drug_nodes), 10)), # Placeholder for molecular features
            "categorical_features": np.zeros((len(drug_nodes), 3), dtype=int)  # Groups, categories, etc.
        }
        
        # 1. Extract basic network features
        for i, node in enumerate(drug_nodes):
            # Degree features
            features["network_features"][i, 0] = self.graph.degree(node)
            features["network_features"][i, 1] = self.graph.in_degree(node)
            features["network_features"][i, 2] = self.graph.out_degree(node)
            
            # Neighbor diversity (number of different node types among neighbors)
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            neighbor_types = [self.graph.nodes[n].get("type", "unknown") for n in neighbors]
            features["network_features"][i, 3] = len(set(neighbor_types))
            
            # Clustering coefficient (local)
            try:
                # Convert to undirected for clustering coefficient
                subgraph = self.graph.to_undirected().subgraph(neighbors + [node])
                if len(subgraph) > 1:  # Need at least 2 nodes for clustering
                    features["network_features"][i, 4] = nx.clustering(subgraph, node)
            except Exception as e:
                self.logger.debug(f"Error calculating clustering for node {node}: {e}")
        
        # Scale network features
        scaler = StandardScaler()
        features["network_features"] = scaler.fit_transform(features["network_features"])
        
        # 2. Extract molecular features (from properties if available)
        molecular_props = [
            "smiles", "inchi", "inchikey", "molecular_weight", "molecular_formula",
            "logp", "h_bond_donors", "h_bond_acceptors", "rotatable_bonds", "tpsa"
        ]
        
        for i, node in enumerate(drug_nodes):
            node_data = self.graph.nodes[node]
            
            # Check for molecular properties in node attributes
            for j, prop in enumerate(molecular_props):
                if prop in node_data:
                    try:
                        # Try to convert to float
                        value = float(node_data[prop])
                        features["molecular_features"][i, j] = value
                    except (ValueError, TypeError):
                        # Non-numeric property, leave as zero
                        pass
        
        # Replace missing molecular features with mean values
        for j in range(features["molecular_features"].shape[1]):
            col = features["molecular_features"][:, j]
            mean_val = np.mean(col[col != 0]) if np.any(col != 0) else 0
            col[col == 0] = mean_val
            features["molecular_features"][:, j] = col
        
        # Scale molecular features
        scaler = StandardScaler()
        features["molecular_features"] = scaler.fit_transform(features["molecular_features"])
        
        # 3. Extract categorical features (groups, categories)
        # 3.1. Drug groups (approved, investigational, etc.)
        group_types = set()
        for node in drug_nodes:
            node_data = self.graph.nodes[node]
            if "groups" in node_data:
                if isinstance(node_data["groups"], list):
                    group_types.update(node_data["groups"])
                elif isinstance(node_data["groups"], str):
                    group_types.add(node_data["groups"])
        
        group_mapping = {group: i for i, group in enumerate(sorted(group_types))}
        self.categorical_mappings["drug_groups"] = group_mapping
        
        # 3.2. Categories (from has_category edges)
        category_nodes = set()
        for node in drug_nodes:
            for _, target, data in self.graph.out_edges(node, data=True):
                if data.get("type") == "has_category":
                    category_nodes.add(target)
                    
        category_mapping = {cat: i for i, cat in enumerate(sorted(category_nodes))}
        self.categorical_mappings["drug_categories"] = category_mapping
        
        # 3.3. ATC codes (if available)
        atc_codes = set()
        for node in drug_nodes:
            node_data = self.graph.nodes[node]
            if "atc_codes" in node_data:
                if isinstance(node_data["atc_codes"], list):
                    for atc in node_data["atc_codes"]:
                        if isinstance(atc, dict) and "code" in atc:
                            atc_codes.add(atc["code"])
                        else:
                            atc_codes.add(str(atc))
        
        atc_mapping = {atc: i for i, atc in enumerate(sorted(atc_codes))}
        self.categorical_mappings["drug_atc_codes"] = atc_mapping
        
        # Fill in categorical features
        for i, node in enumerate(drug_nodes):
            node_data = self.graph.nodes[node]
            
            # Groups
            if "groups" in node_data:
                groups = node_data["groups"]
                if isinstance(groups, list):
                    for group in groups:
                        if group in group_mapping:
                            features["categorical_features"][i, 0] = group_mapping[group]
                elif isinstance(groups, str) and groups in group_mapping:
                    features["categorical_features"][i, 0] = group_mapping[groups]
            
            # Categories
            for _, target, data in self.graph.out_edges(node, data=True):
                if data.get("type") == "has_category" and target in category_mapping:
                    features["categorical_features"][i, 1] = category_mapping[target]
            
            # ATC codes
            if "atc_codes" in node_data:
                atc_list = node_data["atc_codes"]
                if isinstance(atc_list, list):
                    for atc in atc_list:
                        if isinstance(atc, dict) and "code" in atc and atc["code"] in atc_mapping:
                            features["categorical_features"][i, 2] = atc_mapping[atc["code"]]
                            break
                        elif atc in atc_mapping:
                            features["categorical_features"][i, 2] = atc_mapping[atc]
                            break
        
        # 4. Add graph embedding features using SVD on the adjacency matrix
        embedding_dim = min(16, len(drug_nodes) - 1) if len(drug_nodes) > 1 else 1
        
        try:
            # Create an adjacency matrix for the drug nodes
            drug_adj = nx.adjacency_matrix(self.graph.subgraph(drug_nodes))
            
            # Apply SVD to get embeddings
            u, s, vt = svds(drug_adj, k=embedding_dim)
            
            # Compute embeddings
            embeddings = u * s[np.newaxis, :]
            
            features["embedding_features"] = embeddings
        except Exception as e:
            self.logger.warning(f"Error computing drug embeddings: {e}")
            # Fall back to zeros
            features["embedding_features"] = np.zeros((len(drug_nodes), embedding_dim))
        
        # Cache features
        self._cache[cache_key] = features
        
        return features
    
    def extract_disease_features(self, disease_nodes: List[str]) -> Dict[str, np.ndarray]:
        """Extract features for disease nodes
        
        Args:
            disease_nodes: List of disease node IDs
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        if not disease_nodes:
            return {}
            
        # Check cache
        cache_key = "disease_features"
        if cache_key in self._cache:
            # Filter cached features to keep only the requested nodes
            cached_features = self._cache[cache_key]
            return self._filter_features(cached_features, disease_nodes)
        
        self.logger.info(f"Extracting features for {len(disease_nodes)} disease nodes")
        
        # Initialize feature dictionaries
        features = {
            "node_id": np.array(disease_nodes),
            "network_features": np.zeros((len(disease_nodes), 5)),  # Basic network metrics
            "tree_features": np.zeros((len(disease_nodes), 3)),  # Hierarchy-based features
            "categorical_features": np.zeros((len(disease_nodes), 2), dtype=int)  # Top-level category, etc.
        }
        
        # 1. Extract basic network features
        for i, node in enumerate(disease_nodes):
            # Degree features
            features["network_features"][i, 0] = self.graph.degree(node)
            features["network_features"][i, 1] = self.graph.in_degree(node)
            features["network_features"][i, 2] = self.graph.out_degree(node)
            
            # Neighbor diversity (number of different node types among neighbors)
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            neighbor_types = [self.graph.nodes[n].get("type", "unknown") for n in neighbors]
            features["network_features"][i, 3] = len(set(neighbor_types))
            
            # Clustering coefficient (local)
            try:
                # Convert to undirected for clustering coefficient
                subgraph = self.graph.to_undirected().subgraph(neighbors + [node])
                if len(subgraph) > 1:  # Need at least 2 nodes for clustering
                    features["network_features"][i, 4] = nx.clustering(subgraph, node)
            except Exception as e:
                self.logger.debug(f"Error calculating clustering for node {node}: {e}")
        
        # Scale network features
        scaler = StandardScaler()
        features["network_features"] = scaler.fit_transform(features["network_features"])
        
        # 2. Extract tree/hierarchy features
        for i, node in enumerate(disease_nodes):
            node_data = self.graph.nodes[node]
            
            # Tree numbers (from MeSH)
            if "tree_numbers" in node_data:
                tree_nums = node_data["tree_numbers"]
                if isinstance(tree_nums, list):
                    # Tree level (depth in hierarchy)
                    depths = [tn.count('.') + 1 for tn in tree_nums]
                    features["tree_features"][i, 0] = np.mean(depths) if depths else 0
                    
                    # Is top-level category
                    features["tree_features"][i, 1] = 1 if any(tn.count('.') == 0 for tn in tree_nums) else 0
                    
                    # Number of tree locations
                    features["tree_features"][i, 2] = len(tree_nums)
            
            # Parent-child relationships
            parent_count = 0
            child_count = 0
            
            for _, target, data in self.graph.out_edges(node, data=True):
                if data.get("type") == "has_child":
                    child_count += 1
                    
            for source, _, data in self.graph.in_edges(node, data=True):
                if data.get("type") == "has_child":
                    parent_count += 1
            
            # If we don't have tree numbers but have hierarchy edges
            if "tree_numbers" not in node_data:
                features["tree_features"][i, 0] = parent_count  # Used as proxy for depth
                features["tree_features"][i, 1] = 1 if parent_count == 0 else 0  # Top-level if no parents
                features["tree_features"][i, 2] = child_count  # Number of children
        
        # Scale tree features
        scaler = StandardScaler()
        features["tree_features"] = scaler.fit_transform(features["tree_features"])
        
        # 3. Extract categorical features
        # 3.1. Top-level category (e.g., C01, C02, etc.)
        top_categories = set()
        for node in disease_nodes:
            node_data = self.graph.nodes[node]
            if "tree_numbers" in node_data:
                tree_nums = node_data["tree_numbers"]
                if isinstance(tree_nums, list):
                    for tn in tree_nums:
                        parts = tn.split('.')
                        if parts:
                            top_categories.add(parts[0])
        
        category_mapping = {cat: i for i, cat in enumerate(sorted(top_categories))}
        self.categorical_mappings["disease_categories"] = category_mapping
        
        # 3.2. Custom categories (from edges)
        custom_categories = set()
        for node in disease_nodes:
            for source, _, data in self.graph.in_edges(node, data=True):
                if data.get("type") == "has_category":
                    custom_categories.add(source)
        
        custom_mapping = {cat: i for i, cat in enumerate(sorted(custom_categories))}
        self.categorical_mappings["disease_custom_categories"] = custom_mapping
        
        # Fill in categorical features
        for i, node in enumerate(disease_nodes):
            node_data = self.graph.nodes[node]
            
            # Top-level category
            if "tree_numbers" in node_data:
                tree_nums = node_data["tree_numbers"]
                if isinstance(tree_nums, list):
                    for tn in tree_nums:
                        parts = tn.split('.')
                        if parts and parts[0] in category_mapping:
                            features["categorical_features"][i, 0] = category_mapping[parts[0]]
                            break
            
            # Custom categories
            for source, _, data in self.graph.in_edges(node, data=True):
                if data.get("type") == "has_category" and source in custom_mapping:
                    features["categorical_features"][i, 1] = custom_mapping[source]
                    break
        
        # 4. Add graph embedding features
        embedding_dim = min(16, len(disease_nodes) - 1) if len(disease_nodes) > 1 else 1
        
        try:
            # Create an adjacency matrix for the disease nodes
            disease_adj = nx.adjacency_matrix(self.graph.subgraph(disease_nodes))
            
            # Apply SVD to get embeddings
            u, s, vt = svds(disease_adj, k=embedding_dim)
            
            # Compute embeddings
            embeddings = u * s[np.newaxis, :]
            
            features["embedding_features"] = embeddings
        except Exception as e:
            self.logger.warning(f"Error computing disease embeddings: {e}")
            # Fall back to zeros
            features["embedding_features"] = np.zeros((len(disease_nodes), embedding_dim))
        
        # Cache features
        self._cache[cache_key] = features
        
        return features
    
    def extract_protein_features(self, protein_nodes: List[str]) -> Dict[str, np.ndarray]:
        """Extract features for protein/polypeptide nodes
        
        Args:
            protein_nodes: List of protein node IDs
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        if not protein_nodes:
            return {}
            
        # Check cache
        cache_key = "protein_features"
        if cache_key in self._cache:
            # Filter cached features to keep only the requested nodes
            cached_features = self._cache[cache_key]
            return self._filter_features(cached_features, protein_nodes)
        
        self.logger.info(f"Extracting features for {len(protein_nodes)} protein nodes")
        
        # Initialize feature dictionaries
        features = {
            "node_id": np.array(protein_nodes),
            "network_features": np.zeros((len(protein_nodes), 5)),  # Basic network metrics
            "sequence_features": np.zeros((len(protein_nodes), 5)),  # Sequence-based features
            "categorical_features": np.zeros((len(protein_nodes), 3), dtype=int)  # Organism, location, GO terms
        }
        
        # 1. Extract basic network features
        for i, node in enumerate(protein_nodes):
            # Degree features
            features["network_features"][i, 0] = self.graph.degree(node)
            features["network_features"][i, 1] = self.graph.in_degree(node)
            features["network_features"][i, 2] = self.graph.out_degree(node)
            
            # Neighbor diversity (number of different node types among neighbors)
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            neighbor_types = [self.graph.nodes[n].get("type", "unknown") for n in neighbors]
            features["network_features"][i, 3] = len(set(neighbor_types))
            
            # Clustering coefficient (local)
            try:
                # Convert to undirected for clustering coefficient
                subgraph = self.graph.to_undirected().subgraph(neighbors + [node])
                if len(subgraph) > 1:  # Need at least 2 nodes for clustering
                    features["network_features"][i, 4] = nx.clustering(subgraph, node)
            except Exception as e:
                self.logger.debug(f"Error calculating clustering for node {node}: {e}")
        
        # Scale network features
        scaler = StandardScaler()
        features["network_features"] = scaler.fit_transform(features["network_features"])
        
        # 2. Extract sequence features
        for i, node in enumerate(protein_nodes):
            node_data = self.graph.nodes[node]
            
            # Protein sequences
            if "amino_acid_sequence" in node_data and node_data["amino_acid_sequence"]:
                sequence = node_data["amino_acid_sequence"]
                if isinstance(sequence, str):
                    # Sequence length
                    features["sequence_features"][i, 0] = len(sequence)
                    
                    # Amino acid composition features (simple)
                    # Hydrophobic residues
                    hydrophobic = sum(1 for aa in sequence if aa in "AILMFWYV")
                    features["sequence_features"][i, 1] = hydrophobic / len(sequence) if len(sequence) > 0 else 0
                    
                    # Charged residues
                    charged = sum(1 for aa in sequence if aa in "DEKHR")
                    features["sequence_features"][i, 2] = charged / len(sequence) if len(sequence) > 0 else 0
                    
                    # Polar residues
                    polar = sum(1 for aa in sequence if aa in "STNQ")
                    features["sequence_features"][i, 3] = polar / len(sequence) if len(sequence) > 0 else 0
                    
                    # Special residues
                    special = sum(1 for aa in sequence if aa in "CGP")
                    features["sequence_features"][i, 4] = special / len(sequence) if len(sequence) > 0 else 0
        
        # Scale sequence features
        scaler = StandardScaler()
        non_zero_idxs = ~np.all(features["sequence_features"] == 0, axis=1)
        if np.any(non_zero_idxs):
            features["sequence_features"][non_zero_idxs] = scaler.fit_transform(features["sequence_features"][non_zero_idxs])
        
        # 3. Extract categorical features
        # 3.1. Organism
        organisms = set()
        for node in protein_nodes:
            node_data = self.graph.nodes[node]
            if "organism" in node_data and node_data["organism"]:
                organisms.add(str(node_data["organism"]))
        
        organism_mapping = {org: i for i, org in enumerate(sorted(organisms))}
        self.categorical_mappings["protein_organisms"] = organism_mapping
        
        # 3.2. Cellular location
        locations = set()
        for node in protein_nodes:
            node_data = self.graph.nodes[node]
            if "cellular_location" in node_data and node_data["cellular_location"]:
                locations.add(str(node_data["cellular_location"]))
        
        location_mapping = {loc: i for i, loc in enumerate(sorted(locations))}
        self.categorical_mappings["protein_locations"] = location_mapping
        
        # 3.3. GO terms (function)
        functions = set()
        for node in protein_nodes:
            node_data = self.graph.nodes[node]
            
            # Check for GO classifiers
            if "go_classifiers" in node_data:
                go_terms = node_data["go_classifiers"]
                if isinstance(go_terms, list):
                    for go_term in go_terms:
                        if isinstance(go_term, dict) and "category" in go_term:
                            functions.add(go_term["category"])
        
        function_mapping = {func: i for i, func in enumerate(sorted(functions))}
        self.categorical_mappings["protein_functions"] = function_mapping
        
        # Fill in categorical features
        for i, node in enumerate(protein_nodes):
            node_data = self.graph.nodes[node]
            
            # Organism
            if "organism" in node_data and str(node_data["organism"]) in organism_mapping:
                features["categorical_features"][i, 0] = organism_mapping[str(node_data["organism"])]
            
            # Cellular location
            if "cellular_location" in node_data and str(node_data["cellular_location"]) in location_mapping:
                features["categorical_features"][i, 1] = location_mapping[str(node_data["cellular_location"])]
            
            # GO terms
            if "go_classifiers" in node_data:
                go_terms = node_data["go_classifiers"]
                if isinstance(go_terms, list):
                    for go_term in go_terms:
                        if isinstance(go_term, dict) and "category" in go_term:
                            category = go_term["category"]
                            if category in function_mapping:
                                features["categorical_features"][i, 2] = function_mapping[category]
                                break
        
        # 4. Add graph embedding features
        embedding_dim = min(16, len(protein_nodes) - 1) if len(protein_nodes) > 1 else 1
        
        try:
            # Create an adjacency matrix for the protein nodes
            protein_adj = nx.adjacency_matrix(self.graph.subgraph(protein_nodes))
            
            # Apply SVD to get embeddings
            u, s, vt = svds(protein_adj, k=embedding_dim)
            
            # Compute embeddings
            embeddings = u * s[np.newaxis, :]
            
            features["embedding_features"] = embeddings
        except Exception as e:
            self.logger.warning(f"Error computing protein embeddings: {e}")
            # Fall back to zeros
            features["embedding_features"] = np.zeros((len(protein_nodes), embedding_dim))
        
        # Cache features
        self._cache[cache_key] = features
        
        return features
    
    def extract_default_features(self, nodes: List[str]) -> Dict[str, np.ndarray]:
        """Extract features for nodes without a specific extraction method
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        if not nodes:
            return {}
            
        # Check cache
        node_type = self.graph.nodes[nodes[0]].get("type", "unknown") if nodes else "unknown"
        cache_key = f"{node_type}_features"
        if cache_key in self._cache:
            # Filter cached features to keep only the requested nodes
            cached_features = self._cache[cache_key]
            return self._filter_features(cached_features, nodes)
        
        self.logger.info(f"Extracting default features for {len(nodes)} {node_type} nodes")
        
        # Initialize feature dictionaries
        features = {
            "node_id": np.array(nodes),
            "network_features": np.zeros((len(nodes), 5))  # Basic network metrics
        }
        
        # Extract network features
        for i, node in enumerate(nodes):
            # Degree features
            features["network_features"][i, 0] = self.graph.degree(node)
            features["network_features"][i, 1] = self.graph.in_degree(node)
            features["network_features"][i, 2] = self.graph.out_degree(node)
            
            # Neighbor diversity (number of different node types among neighbors)
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            neighbor_types = [self.graph.nodes[n].get("type", "unknown") for n in neighbors]
            features["network_features"][i, 3] = len(set(neighbor_types))
            
            # Clustering coefficient (local)
            try:
                # Convert to undirected for clustering coefficient
                subgraph = self.graph.to_undirected().subgraph(neighbors + [node])
                if len(subgraph) > 1:  # Need at least 2 nodes for clustering
                    features["network_features"][i, 4] = nx.clustering(subgraph, node)
            except Exception as e:
                self.logger.debug(f"Error calculating clustering for node {node}: {e}")
        
        # Scale network features
        scaler = StandardScaler()
        features["network_features"] = scaler.fit_transform(features["network_features"])
        
        # Add graph embedding features
        embedding_dim = min(16, len(nodes) - 1) if len(nodes) > 1 else 1
        
        try:
            # Create an adjacency matrix
            adj = nx.adjacency_matrix(self.graph.subgraph(nodes))
            
            # Apply SVD to get embeddings
            u, s, vt = svds(adj, k=embedding_dim)
            
            # Compute embeddings
            embeddings = u * s[np.newaxis, :]
            
            features["embedding_features"] = embeddings
        except Exception as e:
            self.logger.warning(f"Error computing embeddings for {node_type} nodes: {e}")
            # Fall back to zeros
            features["embedding_features"] = np.zeros((len(nodes), embedding_dim))
        
        # Cache features
        self._cache[cache_key] = features
        
        return features
    
    def extract_edge_features(self, edge_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Extract features for edges
        
        Args:
            edge_types: List of edge types to extract features for (optional)
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        # Get all edges
        all_edges = list(self.graph.edges(data=True, keys=True))
        
        # Filter by type if specified
        if edge_types:
            edges = [(u, v, k, d) for u, v, k, d in all_edges if d.get("type") in edge_types]
        else:
            edges = all_edges
        
        self.logger.info(f"Extracting features for {len(edges)} edges")
        
        # Initialize feature arrays
        features = {
            "edge_id": np.array([(u, v, k) for u, v, k, _ in edges]),
            "numeric_features": np.zeros((len(edges), 3)),  # Score, confidence, etc.
            "categorical_features": np.zeros((len(edges), 1), dtype=int)  # Edge type
        }
        
        # Extract categorical features
        edge_type_set = set()
        for _, _, _, data in edges:
            if "type" in data:
                edge_type_set.add(data["type"])
                
        edge_type_mapping = {t: i for i, t in enumerate(sorted(edge_type_set))}
        self.categorical_mappings["edge_types"] = edge_type_mapping
        
        # Extract features
        for i, (u, v, k, data) in enumerate(edges):
            # Numeric features
            if "score" in data:
                try:
                    features["numeric_features"][i, 0] = float(data["score"])
                except (ValueError, TypeError):
                    pass
            
            if "confidence" in data:
                try:
                    features["numeric_features"][i, 1] = float(data["confidence"])
                except (ValueError, TypeError):
                    pass
            
            # Connection strength (e.g., number of parallel edges)
            # Check if there are multiple edges between the same nodes
            parallel_edges = 0
            for _, _, k2, _ in self.graph.edges(nbunch=[u], data=True, keys=True):
                if k2 == v:  # Same target node
                    parallel_edges += 1
            features["numeric_features"][i, 2] = parallel_edges
                
            # Categorical features
            if "type" in data and data["type"] in edge_type_mapping:
                features["categorical_features"][i, 0] = edge_type_mapping[data["type"]]
        
        # Scale numeric features
        scaler = StandardScaler()
        non_zero_idxs = ~np.all(features["numeric_features"] == 0, axis=1)
        if np.any(non_zero_idxs):
            features["numeric_features"][non_zero_idxs] = scaler.fit_transform(features["numeric_features"][non_zero_idxs])
        
        return features
    
    def extract_node_pairs_features(self, pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """Extract features for node pairs (e.g., drug-disease pairs)
        
        Args:
            pairs: List of node ID pairs (source, target)
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        self.logger.info(f"Extracting features for {len(pairs)} node pairs")
        
        # Initialize feature arrays
        features = {
            "pair_id": np.array(pairs),
            "path_features": np.zeros((len(pairs), 5)),  # Path-based features
            "neighborhood_features": np.zeros((len(pairs), 3)),  # Common neighbor features
            "connectivity_features": np.zeros((len(pairs), 2), dtype=int)  # Direct connection, etc.
        }
        
        # Extract features for each pair
        for i, (source, target) in enumerate(pairs):
            # Check if source and target are in the graph
            if source not in self.graph or target not in self.graph:
                self.logger.warning(f"Node pair ({source}, {target}) not fully in graph")
                continue
            
            # 1. Path features
            try:
                # Check if there's a direct edge
                has_direct_edge = self.graph.has_edge(source, target)
                features["connectivity_features"][i, 0] = int(has_direct_edge)
                
                # Shortest path length
                if has_direct_edge:
                    shortest_path_length = 1
                else:
                    try:
                        shortest_path_length = nx.shortest_path_length(self.graph, source=source, target=target)
                    except nx.NetworkXNoPath:
                        shortest_path_length = -1  # No path exists
                
                features["path_features"][i, 0] = shortest_path_length if shortest_path_length > 0 else 10
                
                # Number of different paths (with cutoff)
                if has_direct_edge:
                    num_paths = 1
                else:
                    try:
                        paths = list(nx.all_simple_paths(self.graph, source=source, target=target, cutoff=3))
                        num_paths = len(paths)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        num_paths = 0
                
                features["path_features"][i, 1] = num_paths
                
                # Path diversity (number of different node types along paths)
                if num_paths > 0:
                    path_node_types = set()
                    for path in paths:
                        for node in path:
                            node_type = self.graph.nodes[node].get("type", "unknown")
                            path_node_types.add(node_type)
                    features["path_features"][i, 2] = len(path_node_types)
                
                # Edge type diversity in paths
                if num_paths > 0:
                    edge_types = set()
                    for path in paths:
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            for _, _, data in self.graph.edges(nbunch=[u], data=True):
                                if data.get("type"):
                                    edge_types.add(data["type"])
                    features["path_features"][i, 3] = len(edge_types)
                
                # Any protein in shortest path
                if shortest_path_length > 0 and shortest_path_length < 10:
                    try:
                        shortest_path = nx.shortest_path(self.graph, source=source, target=target)
                        has_protein = any(self.graph.nodes[n].get("type") in ["protein", "polypeptide"] for n in shortest_path)
                        features["path_features"][i, 4] = int(has_protein)
                    except nx.NetworkXNoPath:
                        pass
                    
            except Exception as e:
                self.logger.warning(f"Error calculating path features for pair ({source}, {target}): {e}")
            
            # 2. Neighborhood features
            try:
                # Common neighbors
                source_neighbors = set(self.graph.successors(source)) | set(self.graph.predecessors(source))
                target_neighbors = set(self.graph.successors(target)) | set(self.graph.predecessors(target))
                common_neighbors = source_neighbors & target_neighbors
                
                features["neighborhood_features"][i, 0] = len(common_neighbors)
                
                # Jaccard similarity
                if len(source_neighbors) > 0 or len(target_neighbors) > 0:
                    jaccard = len(common_neighbors) / len(source_neighbors | target_neighbors)
                    features["neighborhood_features"][i, 1] = jaccard
                
                # Common neighbor types
                common_neighbor_types = set()
                for node in common_neighbors:
                    node_type = self.graph.nodes[node].get("type", "unknown")
                    common_neighbor_types.add(node_type)
                
                features["neighborhood_features"][i, 2] = len(common_neighbor_types)
                
                # Any protein in common neighbors
                has_protein_neighbor = any(self.graph.nodes[n].get("type") in ["protein", "polypeptide"] for n in common_neighbors)
                features["connectivity_features"][i, 1] = int(has_protein_neighbor)
                
            except Exception as e:
                self.logger.warning(f"Error calculating neighborhood features for pair ({source}, {target}): {e}")
        
        # Scale features
        scaler = StandardScaler()
        non_zero_idxs = ~np.all(features["path_features"] == 0, axis=1)
        if np.any(non_zero_idxs):
            features["path_features"][non_zero_idxs] = scaler.fit_transform(features["path_features"][non_zero_idxs])
        
        non_zero_idxs = ~np.all(features["neighborhood_features"] == 0, axis=1)
        if np.any(non_zero_idxs):
            features["neighborhood_features"][non_zero_idxs] = scaler.fit_transform(features["neighborhood_features"][non_zero_idxs])
        
        return features
    
    def extract_drug_disease_features(self, pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """Extract features specifically for drug-disease pairs
        
        Args:
            pairs: List of (drug_id, disease_id) pairs
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        self.logger.info(f"Extracting features for {len(pairs)} drug-disease pairs")
        
        # Get node pair features
        pair_features = self.extract_node_pairs_features(pairs)
        
        # Extract drug and disease features
        drug_ids = [drug_id for drug_id, _ in pairs]
        disease_ids = [disease_id for _, disease_id in pairs]
        
        unique_drug_ids = list(set(drug_ids))
        unique_disease_ids = list(set(disease_ids))
        
        drug_features = self.extract_drug_features(unique_drug_ids)
        disease_features = self.extract_disease_features(unique_disease_ids)
        
        # Map node features to pairs
        drug_to_idx = {drug_id: i for i, drug_id in enumerate(unique_drug_ids)}
        disease_to_idx = {disease_id: i for i, disease_id in enumerate(unique_disease_ids)}
        
        # Create mapping dictionaries for drug and disease network features
        drug_network_features = {}
        for i, drug_id in enumerate(unique_drug_ids):
            if "network_features" in drug_features:
                drug_network_features[drug_id] = drug_features["network_features"][i]
        
        disease_network_features = {}
        for i, disease_id in enumerate(unique_disease_ids):
            if "network_features" in disease_features:
                disease_network_features[disease_id] = disease_features["network_features"][i]
        
        # Initialize additional feature arrays
        drug_network_array = np.zeros((len(pairs), drug_features["network_features"].shape[1]))
        disease_network_array = np.zeros((len(pairs), disease_features["network_features"].shape[1]))
        
        # Fill in features for each pair
        for i, (drug_id, disease_id) in enumerate(pairs):
            if drug_id in drug_network_features:
                drug_network_array[i] = drug_network_features[drug_id]
            
            if disease_id in disease_network_features:
                disease_network_array[i] = disease_network_features[disease_id]
        
        # Add to pair features
        pair_features["drug_network_features"] = drug_network_array
        pair_features["disease_network_features"] = disease_network_array
        
        # Add target-based features (if drug targets are connected to the disease)
        target_based_features = np.zeros((len(pairs), 3))
        
        for i, (drug_id, disease_id) in enumerate(pairs):
            # Find drug targets
            drug_targets = []
            for _, target, data in self.graph.out_edges(drug_id, data=True):
                if data.get("type") == "targets":
                    drug_targets.append(target)
            
            # Check if any targets are associated with the disease
            target_disease_associations = 0
            for target in drug_targets:
                for _, d_target, data in self.graph.out_edges(target, data=True):
                    if d_target == disease_id and data.get("type") == "associated_with":
                        target_disease_associations += 1
            
            target_based_features[i, 0] = len(drug_targets)
            target_based_features[i, 1] = target_disease_associations
            target_based_features[i, 2] = target_disease_associations / len(drug_targets) if drug_targets else 0
        
        # Scale target-based features
        scaler = StandardScaler()
        non_zero_idxs = ~np.all(target_based_features == 0, axis=1)
        if np.any(non_zero_idxs):
            target_based_features[non_zero_idxs] = scaler.fit_transform(target_based_features[non_zero_idxs])
        
        pair_features["target_based_features"] = target_based_features
        
        return pair_features
    
    def prepare_training_data(self, positive_pairs: List[Tuple[str, str]], 
                             negative_pairs: Optional[List[Tuple[str, str]]] = None,
                             generate_negatives: bool = True,
                             neg_pos_ratio: int = 1,
                             output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Prepare training data for drug-disease interaction prediction
        
        Args:
            positive_pairs: List of positive (drug_id, disease_id) pairs
            negative_pairs: List of negative pairs (optional)
            generate_negatives: Whether to generate negative samples if not provided
            neg_pos_ratio: Ratio of negative to positive samples (if generating)
            output_dir: Directory to save prepared data (optional)
            
        Returns:
            Dictionary with prepared training data
        """
        self.logger.info(f"Preparing training data with {len(positive_pairs)} positive pairs")
        
        # Generate negative pairs if not provided
        if negative_pairs is None and generate_negatives:
            self.logger.info(f"Generating negative pairs with ratio {neg_pos_ratio}:1")
            negative_pairs = self._generate_negative_pairs(positive_pairs, neg_pos_ratio)
        elif negative_pairs is None:
            self.logger.warning("No negative pairs provided and generate_negatives is False")
            negative_pairs = []
            
        self.logger.info(f"Using {len(negative_pairs)} negative pairs")
        
        # Extract features for positive and negative pairs
        pos_features = self.extract_drug_disease_features(positive_pairs)
        
        if negative_pairs:
            neg_features = self.extract_drug_disease_features(negative_pairs)
            
            # Combine positive and negative features
            combined_features = {}
            for key in pos_features:
                if key == "pair_id":
                    combined_features[key] = np.concatenate([pos_features[key], neg_features[key]])
                else:
                    combined_features[key] = np.vstack([pos_features[key], neg_features[key]])
                    
            # Create labels (1 for positive, 0 for negative)
            labels = np.zeros(len(positive_pairs) + len(negative_pairs))
            labels[:len(positive_pairs)] = 1
                
        else:
            combined_features = pos_features
            labels = np.ones(len(positive_pairs))
        
        # Save prepared data if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save features and labels
            features_path = os.path.join(output_dir, "drug_disease_features.pickle")
            with open(features_path, "wb") as f:
                pickle.dump(combined_features, f)
                
            labels_path = os.path.join(output_dir, "drug_disease_labels.npy")
            np.save(labels_path, labels)
            
            self.logger.info(f"Saved features to {features_path} and labels to {labels_path}")
            
            # Save pairs for reference
            pairs_path = os.path.join(output_dir, "drug_disease_pairs.json")
            with open(pairs_path, "w") as f:
                json.dump({
                    "positive_pairs": [{"drug_id": p[0], "disease_id": p[1]} for p in positive_pairs],
                    "negative_pairs": [{"drug_id": n[0], "disease_id": n[1]} for n in negative_pairs]
                }, f, indent=2)
                
            self.logger.info(f"Saved pairs to {pairs_path}")
        
        return {
            "features": combined_features,
            "labels": labels,
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs
        }
    
    def _generate_negative_pairs(self, positive_pairs: List[Tuple[str, str]], 
                                neg_pos_ratio: int = 1) -> List[Tuple[str, str]]:
        """Generate negative pairs for training
        
        Args:
            positive_pairs: List of positive (drug_id, disease_id) pairs
            neg_pos_ratio: Ratio of negative to positive samples
            
        Returns:
            List of negative pairs
        """
        # Get all unique drug and disease IDs
        drug_ids = set()
        disease_ids = set()
        for drug_id, disease_id in positive_pairs:
            drug_ids.add(drug_id)
            disease_ids.add(disease_id)
            
        # Create a set of positive pairs for fast lookup
        positive_set = set(positive_pairs)
        
        # Generate negative pairs
        negative_pairs = []
        num_to_generate = len(positive_pairs) * neg_pos_ratio
        
        # Strategy 1: Random sampling
        max_attempts = num_to_generate * 10
        attempts = 0
        
        while len(negative_pairs) < num_to_generate and attempts < max_attempts:
            # Randomly select a drug and disease
            drug_id = np.random.choice(list(drug_ids))
            disease_id = np.random.choice(list(disease_ids))
            
            # Check if this is a positive pair
            if (drug_id, disease_id) not in positive_set and (drug_id, disease_id) not in negative_pairs:
                # Check if there's an edge in the graph
                if not self.graph.has_edge(drug_id, disease_id):
                    negative_pairs.append((drug_id, disease_id))
            
            attempts += 1
        
        # If we couldn't generate enough negatives, try a different strategy
        if len(negative_pairs) < num_to_generate:
            self.logger.warning(f"Could only generate {len(negative_pairs)} negatives randomly. Using strategic sampling.")
            
            # Strategy 2: Targeted sampling - for each positive drug, sample diseases it doesn't treat
            for drug_id, _ in positive_pairs:
                treated_diseases = set(target for _, target, data in self.graph.out_edges(drug_id, data=True)
                                    if data.get("type") == "treats")
                
                # Sample diseases not treated by this drug
                untreated_diseases = list(disease_ids - treated_diseases)
                
                if untreated_diseases:
                    num_to_sample = min(5, len(untreated_diseases))  # Limit samples per drug
                    sampled_diseases = np.random.choice(untreated_diseases, size=num_to_sample, replace=False)
                    
                    for disease_id in sampled_diseases:
                        pair = (drug_id, disease_id)
                        if pair not in positive_set and pair not in negative_pairs:
                            negative_pairs.append(pair)
                            
                            if len(negative_pairs) >= num_to_generate:
                                break
                
                if len(negative_pairs) >= num_to_generate:
                    break
        
        self.logger.info(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs
    
    def _filter_features(self, features: Dict[str, np.ndarray], nodes: List[str]) -> Dict[str, np.ndarray]:
        """Filter features to keep only the requested nodes
        
        Args:
            features: Dictionary of feature arrays
            nodes: List of nodes to keep
            
        Returns:
            Filtered feature dictionary
        """
        if "node_id" not in features:
            return features
            
        # Create node set for fast lookup
        node_set = set(nodes)
        
        # Check if all requested nodes are in features
        feature_node_set = set(features["node_id"])
        missing_nodes = node_set - feature_node_set
        
        if missing_nodes:
            self.logger.warning(f"{len(missing_nodes)} requested nodes are not in cached features")
            # Return None to force recalculation
            return None
            
        # Get indices of nodes to keep
        indices = [i for i, node in enumerate(features["node_id"]) if node in node_set]
        
        # Filter features
        filtered = {
            "node_id": features["node_id"][indices]
        }
        
        # Filter numeric arrays
        for key, array in features.items():
            if key != "node_id" and isinstance(array, np.ndarray):
                filtered[key] = array[indices]
        
        return filtered