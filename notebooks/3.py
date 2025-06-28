import json 

nc = {
    "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {},
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feature Engineering for Drug-Disease Interaction Prediction\n",
    "\n",
    "This notebook demonstrates the feature engineering process for our drug-disease interaction prediction model. We'll extract and preprocess features for different node types in the knowledge graph and prepare training data for the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup\n",
    "\n",
    "First, let's import the necessary libraries and load the knowledge graph."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import pickle\n",
    "import networkx as nx\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.manifold import TSNE\n",
    "from sklearn.decomposition import PCA\n",
    "from typing import Dict, List, Tuple, Any, Optional, Set\n",
    "\n",
    "# Ensure we can import from src\n",
    "sys.path.append(str(Path.cwd().parent))\n",
    "\n",
    "# Import our modules\n",
    "from src.ddi.features.feature_engineering import FeatureExtractor\n",
    "from src.ddi.analysis.graph_analysis import GraphAnalyzer\n",
    "from src.ddi.visualization.graph_viz import GraphVisualizer\n",
    "\n",
    "# Set up plotting defaults\n",
    "sns.set_style(\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 8)\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load the Knowledge Graph\n",
    "\n",
    "We'll load the knowledge graph that we built in the previous steps."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to the saved graph file\n",
    "graph_path = \"../data/graph/full/knowledge_graph.pickle\"\n",
    "\n",
    "# Load the graph\n",
    "with open(graph_path, \"rb\") as f:\n",
    "    graph = pickle.load(f)\n",
    "\n",
    "print(f\"Loaded knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges\")\n",
    "\n",
    "# Create feature extractor, analyzer, and visualizer\n",
    "extractor = FeatureExtractor(graph)\n",
    "analyzer = GraphAnalyzer(graph)\n",
    "visualizer = GraphVisualizer(output_dir=\"../figures\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Exploring Node Types\n",
    "\n",
    "Let's first explore the different types of nodes in our knowledge graph to understand what features we need to extract."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count node types\n",
    "node_types = {}\n",
    "for node, data in graph.nodes(data=True):\n",
    "    node_type = data.get(\"type\", \"unknown\")\n",
    "    if node_type not in node_types:\n",
    "        node_types[node_type] = 0\n",
    "    node_types[node_type] += 1\n",
    "\n",
    "# Display node type counts\n",
    "node_type_df = pd.DataFrame({\n",
    "    \"Node Type\": list(node_types.keys()),\n",
    "    \"Count\": list(node_types.values())\n",
    "}).sort_values(\"Count\", ascending=False).reset_index(drop=True)\n",
    "\n",
    "print(\"Node types in the knowledge graph:\")\n",
    "display(node_type_df)\n",
    "\n",
    "# Visualize node type distribution\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=\"Node Type\", y=\"Count\", data=node_type_df)\n",
    "plt.title(\"Distribution of Node Types\")\n",
    "plt.xticks(rotation=45, ha=\"right\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's examine the attributes available for each node type to see what features we can extract."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to analyze node attributes for a given type\n",
    "def analyze_node_attributes(graph, node_type, max_examples=5):\n",
    "    # Get nodes of this type\n",
    "    nodes = [n for n, d in graph.nodes(data=True) if d.get(\"type\") == node_type]\n",
    "    if not nodes:\n",
    "        return f\"No nodes of type '{node_type}' found.\"\n",
    "    \n",
    "    # Sample nodes\n",
    "    sample_nodes = np.random.choice(nodes, size=min(max_examples, len(nodes)), replace=False)\n",
    "    \n",
    "    # Collect all attributes\n",
    "    all_attrs = set()\n",
    "    for node in sample_nodes:\n",
    "        all_attrs.update(graph.nodes[node].keys())\n",
    "    \n",
    "    # Count attribute prevalence\n",
    "    attr_counts = {attr: 0 for attr in all_attrs}\n",
    "    for node in nodes:\n",
    "        for attr in all_attrs:\n",
    "            if attr in graph.nodes[node]:\n",
    "                attr_counts[attr] += 1\n",
    "    \n",
    "    # Calculate percentages\n",
    "    attr_percentages = {attr: (count / len(nodes)) * 100 for attr, count in attr_counts.items()}\n",
    "    \n",
    "    # Create result DataFrame\n",
    "    result_df = pd.DataFrame({\n",
    "        \"Attribute\": list(attr_counts.keys()),\n",
    "        \"Count\": list(attr_counts.values()),\n",
    "        \"Percentage\": [f\"{attr_percentages[attr]:.1f}%\" for attr in attr_counts.keys()]\n",
    "    }).sort_values(\"Count\", ascending=False).reset_index(drop=True)\n",
    "    \n",
    "    # Show examples for each attribute\n",
    "    examples = {}\n",
    "    for attr in all_attrs:\n",
    "        attr_examples = []\n",
    "        for node in sample_nodes:\n",
    "            if attr in graph.nodes[node]:\n",
    "                value = graph.nodes[node][attr]\n",
    "                # Truncate long values\n",
    "                if isinstance(value, str) and len(value) > 50:\n",
    "                    value = value[:50] + \"...\"\n",
    "                # Handle list values\n",
    "                elif isinstance(value, list) and len(value) > 3:\n",
    "                    value = value[:3] + [\"...\"]\n",
    "                attr_examples.append(str(value))\n",
    "        examples[attr] = \", \".join(attr_examples) if attr_examples else \"N/A\"\n",
    "    \n",
    "    examples_df = pd.DataFrame({\n",
    "        \"Attribute\": list(examples.keys()),\n",
    "        \"Examples\": list(examples.values())\n",
    "    }).sort_values(\"Attribute\").reset_index(drop=True)\n",
    "    \n",
    "    return result_df, examples_df"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze attributes for each key node type\n",
    "key_node_types = [\"drug\", \"disease\", \"protein\", \"polypeptide\"]\n",
    "\n",
    "for node_type in key_node_types:\n",
    "    print(f\"\\n{node_type.upper()} NODE ATTRIBUTES:\")\n",
    "    result = analyze_node_attributes(graph, node_type)\n",
    "    \n",
    "    if isinstance(result, tuple):\n",
    "        counts_df, examples_df = result\n",
    "        print(f\"Attribute prevalence:\")\n",
    "        display(counts_df)\n",
    "        print(f\"Example values:\")\n",
    "        display(examples_df)\n",
    "    else:\n",
    "        print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Engineering for Drug Nodes\n",
    "\n",
    "Let's extract features for drug nodes, which will include:\n",
    "\n",
    "1. Network features (degree, centrality, etc.)\n",
    "2. Molecular features (properties from DrugBank)\n",
    "3. Categorical features (groups, categories, ATC codes)"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get all drug nodes\n",
    "drug_nodes = [n for n, d in graph.nodes(data=True) if d.get(\"type\") == \"drug\"]\n",
    "print(f\"Extracting features for {len(drug_nodes)} drug nodes\")\n",
    "\n",
    "# Extract drug features\n",
    "drug_features = extractor.extract_drug_features(drug_nodes)\n",
    "\n",
    "# Show available feature types\n",
    "print(\"\\nAvailable feature types:\")\n",
    "for feature_type, feature_array in drug_features.items():\n",
    "    if feature_type != \"node_id\":\n",
    "        print(f\"  - {feature_type}: {feature_array.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a DataFrame with node IDs and names\n",
    "drug_names = [graph.nodes[node].get(\"name\", node) for node in drug_features[\"node_id\"]]\n",
    "drug_df = pd.DataFrame({\"node_id\": drug_features[\"node_id\"], \"name\": drug_names})\n",
    "\n",
    "# Add network features\n",
    "network_columns = [f\"network_{i}\" for i in range(drug_features[\"network_features\"].shape[1])]\n",
    "network_df = pd.DataFrame(drug_features[\"network_features\"], columns=network_columns)\n",
    "drug_df = pd.concat([drug_df, network_df], axis=1)\n",
    "\n",
    "# Display a sample of the data\n",
    "print(\"Sample of drug features:\")\n",
    "display(drug_df.head())\n",
    "\n",
    "# Calculate feature correlations\n",
    "plt.figure(figsize=(10, 8))\n",
    "corr = network_df.corr()\n",
    "sns.heatmap(corr, annot=True, cmap=\"coolwarm\", vmin=-1, vmax=1, fmt=\".2f\")\n",
    "plt.title(\"Correlation Matrix of Drug Network Features\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize drug embeddings using PCA\n",
    "if \"embedding_features\" in drug_features:\n",
    "    # Apply PCA to reduce to 2D\n",
    "    pca = PCA(n_components=2)\n",
    "    drug_pca = pca.fit_transform(drug_features[\"embedding_features\"])\n",
    "    \n",
    "    # Create DataFrame for plotting\n",
    "    pca_df = pd.DataFrame({\n",
    "        \"PCA1\": drug_pca[:, 0],\n",
    "        \"PCA2\": drug_pca[:, 1],\n",
    "        \"Drug\": drug_names\n",
    "    })\n",
    "    \n",
    "    # Plot\n",
    "    plt.figure(figsize=(12, 10))\n",
    "    sns.scatterplot(x=\"PCA1\", y=\"PCA2\", data=pca_df, alpha=0.7)\n",
    "    \n",
    "    # Add labels for some points\n",
    "    n_labels = 20\n",
    "    for i in range(min(n_labels, len(pca_df))):\n",
    "        plt.text(pca_df.iloc[i][\"PCA1\"], pca_df.iloc[i][\"PCA2\"], pca_df.iloc[i][\"Drug\"],\n",
    "                fontsize=8, ha=\"center\", va=\"center\")\n",
    "    \n",
    "    plt.title(f\"PCA of Drug Embeddings (explained variance: {pca.explained_variance_ratio_.sum():.2f})\")\n",
    "    plt.grid(alpha=0.3)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Engineering for Disease Nodes\n",
    "\n",
    "Now let's extract features for disease nodes, which will include:\n",
    "\n",
    "1. Network features (degree, centrality, etc.)\n",
    "2. Tree features (hierarchy information from MeSH)\n",
    "3. Categorical features (top-level categories)"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get all disease nodes\n",
    "disease_nodes = [n for n, d in graph.nodes(data=True) if d.get(\"type\") == \"disease\"]\n",
    "print(f\"Extracting features for {len(disease_nodes)} disease nodes\")\n",
    "\n",
    "# Extract disease features\n",
    "disease_features = extractor.extract_disease_features(disease_nodes)\n",
    "\n",
    "# Show available feature types\n",
    "print(\"\\nAvailable feature types:\")\n",
    "for feature_type, feature_array in disease_features.items():\n",
    "    if feature_type != \"node_id\":\n",
    "        print(f\"  - {feature_type}: {feature_array.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a DataFrame with node IDs and names\n",
    "disease_names = [graph.nodes[node].get(\"name\", node) for node in disease_features[\"node_id\"]]\n",
    "disease_df = pd.DataFrame({\"node_id\": disease_features[\"node_id\"], \"name\": disease_names})\n",
    "\n",
    "# Add tree features\n",
    "tree_columns = [f\"tree_{i}\" for i in range(disease_features[\"tree_features\"].shape[1])]\n",
    "tree_df = pd.DataFrame(disease_features[\"tree_features\"], columns=tree_columns)\n",
    "disease_df = pd.concat([disease_df, tree_df], axis=1)\n",
    "\n",
    "# Display a sample of the data\n",
    "print(\"Sample of disease features:\")\n",
    "display(disease_df.head())\n",
    "\n",
    "# Calculate feature correlations\n",
    "plt.figure(figsize=(10, 8))\n",
    "corr = tree_df.corr()\n",
    "sns.heatmap(corr, annot=True, cmap=\"coolwarm\", vmin=-1, vmax=1, fmt=\".2f\")\n",
    "plt.title(\"Correlation Matrix of Disease Tree Features\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize disease embeddings using t-SNE\n",
    "if \"embedding_features\" in disease_features:\n",
    "    # Apply t-SNE to reduce to 2D\n",
    "    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)\n",
    "    disease_tsne = tsne.fit_transform(disease_features[\"embedding_features\"])\n",
    "    \n",
    "    # Create DataFrame for plotting\n",
    "    tsne_df = pd.DataFrame({\n",
    "        \"TSNE1\": disease_tsne[:, 0],\n",
    "        \"TSNE2\": disease_tsne[:, 1],\n",
    "        \"Disease\": disease_names\n",
    "    })\n",
    "    \n",
    "    # Plot\n",
    "    plt.figure(figsize=(12, 10))\n",
    "    sns.scatterplot(x=\"TSNE1\", y=\"TSNE2\", data=tsne_df, alpha=0.7)\n",
    "    \n",
    "    # Add labels for some points\n",
    "    n_labels = 20\n",
    "    for i in range(min(n_labels, len(tsne_df))):\n",
    "        plt.text(tsne_df.iloc[i][\"TSNE1\"], tsne_df.iloc[i][\"TSNE2\"], tsne_df.iloc[i][\"Disease\"],\n",
    "                fontsize=8, ha=\"center\", va=\"center\")\n",
    "    \n",
    "    plt.title(\"t-SNE of Disease Embeddings\")\n",
    "    plt.grid(alpha=0.3)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Engineering for Protein/Polypeptide Nodes\n",
    "\n",
    "Let's extract features for protein and polypeptide nodes, which will include:\n",
    "\n",
    "1. Network features (degree, centrality, etc.)\n",
    "2. Sequence features (amino acid composition)\n",
    "3. Categorical features (organism, cellular location, GO terms)"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get all protein nodes\n",
    "protein_nodes = [n for n, d in graph.nodes(data=True) if d.get(\"type\") in [\"protein\", \"polypeptide\"]]\n",
    "print(f\"Extracting features for {len(protein_nodes)} protein nodes\")\n",
    "\n",
    "# Extract protein features\n",
    "protein_features = extractor.extract_protein_features(protein_nodes)\n",
    "\n",
    "# Show available feature types\n",
    "print(\"\\nAvailable feature types:\")\n",
    "for feature_type, feature_array in protein_features.items():\n",
    "    if feature_type != \"node_id\":\n",
    "        print(f\"  - {feature_type}: {feature_array.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a DataFrame with node IDs and names\n",
    "protein_names = [graph.nodes[node].get(\"name\", node) for node in protein_features[\"node_id\"]]\n",
    "protein_df = pd.DataFrame({\"node_id\": protein_features[\"node_id\"], \"name\": protein_names})\n",
    "\n",
    "# Check for sequence features\n",
    "if \"sequence_features\" in protein_features:\n",
    "    # Add sequence features\n",
    "    seq_columns = [f\"seq_{i}\" for i in range(protein_features[\"sequence_features\"].shape[1])]\n",
    "    seq_df = pd.DataFrame(protein_features[\"sequence_features\"], columns=seq_columns)\n",
    "    protein_df = pd.concat([protein_df, seq_df], axis=1)\n",
    "\n",
    "    # Display a sample of the data\n",
    "    print(\"Sample of protein features:\")\n",
    "    display(protein_df.head())\n",
    "\n",
    "    # Calculate feature correlations\n",
    "    plt.figure(figsize=(10, 8))\n",
    "    corr = seq_df.corr()\n",
    "    sns.heatmap(corr, annot=True, cmap=\"coolwarm\", vmin=-1, vmax=1, fmt=\".2f\")\n",
    "    plt.title(\"Correlation Matrix of Protein Sequence Features\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Extracting Pairwise Features for Drug-Disease Interactions\n",
    "\n",
    "Now let's extract features for drug-disease pairs, which will include:\n",
    "\n",
    "1. Path-based features (shortest path length, number of paths, path diversity)\n",
    "2. Neighborhood features (common neighbors, Jaccard similarity)\n",
    "3. Connectivity features (direct connection, protein-mediated connection)"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find existing drug-disease interactions (treats edges)\n",
    "drug_disease_pairs = []\n",
    "for u, v, data in graph.edges(data=True):\n",
    "    if data.get(\"type\") == \"treats\":\n",
    "        # Verify node types\n",
    "        if graph.nodes[u].get(\"type\") == \"drug\" and graph.nodes[v].get(\"type\") == \"disease\":\n",
    "            drug_disease_pairs.append((u, v))\n",
    "\n",
    "print(f\"Found {len(drug_disease_pairs)} drug-disease interaction pairs\")\n",
    "\n",
    "# Extract features for these pairs\n",
    "pair_features = extractor.extract_drug_disease_features(drug_disease_pairs)\n",
    "\n",
    "# Show available feature types\n",
    "print(\"\\nAvailable feature types:\")\n",
    "for feature_type, feature_array in pair_features.items():\n",
    "    if feature_type != \"pair_id\":\n",
    "        print(f\"  - {feature_type}: {feature_array.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a DataFrame with pair IDs and names\n",
    "pair_df = pd.DataFrame({\n",
    "    \"drug_id\": [pair[0] for pair in pair_features[\"pair_id\"]],\n",
    "    \"disease_id\": [pair[1] for pair in pair_features[\"pair_id\"]],\n",
    "    \"drug_name\": [graph.nodes[pair[0]].get(\"name\", pair[0]) for pair in pair_features[\"pair_id\"]],\n",
    "    \"disease_name\": [graph.nodes[pair[1]].get(\"name\", pair[1]) for pair in pair_features[\"pair_id\"]]\n",
    "})\n",
    "\n",
    "# Add path features\n",
    "path_columns = [f\"path_{i}\" for i in range(pair_features[\"path_features\"].shape[1])]\n",
    "path_df = pd.DataFrame(pair_features[\"path_features\"], columns=path_columns)\n",
    "pair_df = pd.concat([pair_df, path_df], axis=1)\n",
    "\n",
    "# Add neighborhood features\n",
    "neighbor_columns = [f\"neighbor_{i}\" for i in range(pair_features[\"neighborhood_features\"].shape[1])]\n",
    "neighbor_df = pd.DataFrame(pair_features[\"neighborhood_features\"], columns=neighbor_columns)\n",
    "pair_df = pd.concat([pair_df, neighbor_df], axis=1)\n",
    "\n",
    "# Display a sample of the data\n",
    "print(\"Sample of drug-disease pair features:\")\n",
    "display(pair_df.head())\n",
    "\n",
    "# Calculate feature correlations\n",
    "feature_df = pd.concat([path_df, neighbor_df], axis=1)\n",
    "plt.figure(figsize=(10, 8))\n",
    "corr = feature_df.corr()\n",
    "sns.heatmap(corr, annot=True, cmap=\"coolwarm\", vmin=-1, vmax=1, fmt=\".2f\")\n",
    "plt.title(\"Correlation Matrix of Drug-Disease Pair Features\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Preparing Training Data for ML Model\n",
    "\n",
    "Let's prepare training data for our machine learning model by generating positive and negative samples."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use existing drug-disease pairs as positive samples\n",
    "positive_pairs = drug_disease_pairs\n",
    "\n",
    "# Prepare training data (this will generate negative samples)\n",
    "training_data = extractor.prepare_training_data(\n",
    "    positive_pairs=positive_pairs,\n",
    "    generate_negatives=True,\n",
    "    neg_pos_ratio=2,\n",
    "    output_dir=\"../data/features\"\n",
    ")\n",
    "\n",
    "# Print statistics\n",
    "print(f\"Prepared training data with:\")\n",
    "print(f\"  - {len(training_data['positive_pairs'])} positive pairs\")\n",
    "print(f\"  - {len(training_data['negative_pairs'])} negative pairs\")\n",
    "print(f\"  - Total of {len(training_data['labels'])} samples\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Examine the features\n",
    "features = training_data[\"features\"]\n",
    "labels = training_data[\"labels\"]\n",
    "\n",
    "# Calculate feature importance using a simple approach (mean difference between classes)\n",
    "def calculate_feature_importance(features, labels, feature_type):\n",
    "    if feature_type not in features:\n",
    "        return None\n",
    "    \n",
    "    # Get feature array\n",
    "    feature_array = features[feature_type]\n",
    "    \n",
    "    # Calculate mean for each class\n",
    "    pos_mean = np.mean(feature_array[labels == 1], axis=0)\n",
    "    neg_mean = np.mean(feature_array[labels == 0], axis=0)\n",
    "    \n",
    "    # Calculate absolute difference as importance\n",
    "    importance = np.abs(pos_mean - neg_mean)\n",
    "    \n",
    "    return importance\n",
    "\n",
    "# Calculate importance for each feature type\n",
    "feature_types = [k for k in features.keys() if k != \"pair_id\"]\n",
    "\n",
    "for feature_type in feature_types:\n",
    "    importance = calculate_feature_importance(features, labels, feature_type)\n",
    "    \n",
    "    if importance is not None:\n",
    "        # Plot feature importance\n",
    "        plt.figure(figsize=(10, 5))\n",
    "        plt.bar(range(len(importance)), importance)\n",
    "        plt.title(f\"Feature Importance: {feature_type}\")\n",
    "        plt.xlabel(\"Feature Index\")\n",
    "        plt.ylabel(\"Importance (|Pos Mean - Neg Mean|)\")\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "        \n",
    "        # Print top 3 most important features\n",
    "        top_indices = np.argsort(importance)[::-1][:3]\n",
    "        print(f\"Top 3 important features for {feature_type}:\")\n",
    "        for i, idx in enumerate(top_indices):\n",
    "            print(f\"  {i+1}. Feature {idx}: {importance[idx]:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Split Training and Testing Data\n",
    "\n",
    "Finally, let's split our prepared data into training and testing sets."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into training and testing sets\n",
    "X_indices = np.arange(len(labels))\n",
    "X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(\n",
    "    X_indices, labels, X_indices, test_size=0.2, random_state=42, stratify=labels\n",
    ")\n",
    "\n",
    "# Function to split features\n",
    "def split_features(features, indices_train, indices_test):\n",
    "    train_features = {}\n",
    "    test_features = {}\n",
    "    \n",
    "    for key, value in features.items():\n",
    "        if key == \"pair_id\":\n",
    "            train_features[key] = value[indices_train]\n",
    "            test_features[key] = value[indices_test]\n",
    "        else:\n",
    "            train_features[key] = value[indices_train]\n",
    "            test_features[key] = value[indices_test]\n",
    "    \n",
    "    return train_features, test_features\n",
    "\n",
    "# Split features\n",
    "train_features, test_features = split_features(features, indices_train, indices_test)\n",
    "\n",
    "# Print statistics\n",
    "print(f\"Training set: {len(y_train)} samples\")\n",
    "print(f\"  - Positive: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)\")\n",
    "print(f\"  - Negative: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)\")\n",
    "print(f\"Testing set: {len(y_test)} samples\")\n",
    "print(f\"  - Positive: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)\")\n",
    "print(f\"  - Negative: {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's save the train/test split for later use\n",
    "output_dir = \"../data/features\"\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "# Save train/test indices\n",
    "np.save(os.path.join(output_dir, \"train_indices.npy\"), indices_train)\n",
    "np.save(os.path.join(output_dir, \"test_indices.npy\"), indices_test)\n",
    "\n",
    "# Save train/test labels\n",
    "np.save(os.path.join(output_dir, \"train_labels.npy\"), y_train)\n",
    "np.save(os.path.join(output_dir, \"test_labels.npy\"), y_test)\n",
    "\n",
    "print(f\"Saved train/test split to {output_dir}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Summary and Conclusions\n",
    "\n",
    "In this notebook, we've engineered features for our drug-disease interaction prediction model:\n",
    "\n",
    "1. **Node Features**: We extracted features for drug, disease, and protein nodes based on their network properties, attributes, and structural information.\n",
    "\n",
    "2. **Pairwise Features**: We created features for drug-disease pairs based on path analysis, common neighborhoods, and connectivity patterns.\n",
    "\n",
    "3. **Training Data Preparation**: We prepared a balanced dataset with positive and negative examples for training machine learning models.\n",
    "\n",
    "These features will serve as the input for our graph neural network model in the next phase of the project."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\n",
    "\n",
    "1. **Model Implementation**: Implement the graph neural network architecture as defined in the project plan.\n",
    "\n",
    "2. **Model Training**: Train the GNN model using our prepared features and evaluate its performance.\n",
    "\n",
    "3. **Explainability**: Develop methods to explain model predictions using the features we've engineered.\n",
    "\n",
    "4. **API Development**: Create an API for making drug-disease interaction predictions based on our model."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("feature_engineering.ipynb", "w") as f:
    json.dump(nc, f)