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
    "# Graph Analysis for Drug-Disease Interaction Prediction\n",
    "\n",
    "This notebook demonstrates the use of graph analysis techniques on the knowledge graph built from biomedical data sources. We'll explore the graph structure, calculate centrality measures, find important paths between drugs and diseases, and visualize the results."
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
    "\n",
    "# Ensure we can import from src\n",
    "sys.path.append(str(Path.cwd().parent))\n",
    "\n",
    "# Import our modules\n",
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
    "# Create analyzer and visualizer\n",
    "analyzer = GraphAnalyzer(graph)\n",
    "visualizer = GraphVisualizer(output_dir=\"../figures\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Basic Graph Statistics\n",
    "\n",
    "Let's start by examining the basic statistics of our knowledge graph."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get basic statistics\n",
    "stats = analyzer.get_basic_statistics()\n",
    "\n",
    "# Display node and edge counts\n",
    "print(f\"Number of nodes: {stats['num_nodes']}\")\n",
    "print(f\"Number of edges: {stats['num_edges']}\")\n",
    "print(f\"Graph density: {stats['density']:.6f}\")\n",
    "print(f\"Connected components: {stats['num_connected_components']}\")\n",
    "print(f\"Largest component size: {stats['largest_component_size']} nodes \"\n",
    "      f\"({stats['largest_component_percentage']:.2f}% of total)\")\n",
    "\n",
    "# Display degree statistics\n",
    "print(\"\\nDegree statistics:\")\n",
    "for stat, value in stats[\"degree_stats\"].items():\n",
    "    print(f\"  {stat}: {value:.2f}\")\n",
    "\n",
    "# Create a table of node types\n",
    "node_types_df = pd.DataFrame({\n",
    "    \"Node Type\": list(stats[\"node_types\"].keys()),\n",
    "    \"Count\": list(stats[\"node_types\"].values())\n",
    "}).sort_values(\"Count\", ascending=False).reset_index(drop=True)\n",
    "\n",
    "# Display node types\n",
    "print(\"\\nNode types:\")\n",
    "display(node_types_df)\n",
    "\n",
    "# Create a table of edge types\n",
    "edge_types_df = pd.DataFrame({\n",
    "    \"Edge Type\": list(stats[\"edge_types\"].keys()),\n",
    "    \"Count\": list(stats[\"edge_types\"].values())\n",
    "}).sort_values(\"Count\", ascending=False).reset_index(drop=True)\n",
    "\n",
    "# Display edge types\n",
    "print(\"\\nEdge types:\")\n",
    "display(edge_types_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize Node and Edge Type Distributions"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot node type distribution\n",
    "node_type_fig = visualizer.plot_node_type_distribution(\n",
    "    stats[\"node_types\"], \n",
    "    title=\"Distribution of Node Types in Knowledge Graph\",\n",
    "    save_path=\"node_type_distribution.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot edge type distribution\n",
    "edge_type_fig = visualizer.plot_edge_type_distribution(\n",
    "    stats[\"edge_types\"], \n",
    "    title=\"Distribution of Edge Types in Knowledge Graph\",\n",
    "    save_path=\"edge_type_distribution.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Degree Distribution"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get degree distribution\n",
    "degree_dist = analyzer.get_degree_distribution()\n",
    "\n",
    "# Plot overall degree distribution (log-log scale)\n",
    "degree_fig = visualizer.plot_degree_distribution(\n",
    "    degree_dist[\"degree\"],\n",
    "    log_scale=True,\n",
    "    title=\"Degree Distribution (log-log scale)\",\n",
    "    save_path=\"degree_distribution_log.png\"\n",
    ")\n",
    "plt.show()\n",
    "\n",
    "# Plot in-degree distribution\n",
    "in_degree_fig = visualizer.plot_degree_distribution(\n",
    "    degree_dist[\"in_degree\"],\n",
    "    log_scale=True,\n",
    "    title=\"In-Degree Distribution (log-log scale)\",\n",
    "    save_path=\"in_degree_distribution_log.png\"\n",
    ")\n",
    "plt.show()\n",
    "\n",
    "# Plot out-degree distribution\n",
    "out_degree_fig = visualizer.plot_degree_distribution(\n",
    "    degree_dist[\"out_degree\"],\n",
    "    log_scale=True,\n",
    "    title=\"Out-Degree Distribution (log-log scale)\",\n",
    "    save_path=\"out_degree_distribution_log.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's also look at the degree distribution for specific node types."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Node types to analyze\n",
    "key_node_types = [\"drug\", \"disease\", \"protein\"]\n",
    "\n",
    "for node_type in key_node_types:\n",
    "    # Get degree distribution for this node type\n",
    "    type_dist = analyzer.get_degree_distribution(node_type=node_type)\n",
    "    \n",
    "    # Plot degree distribution\n",
    "    fig = visualizer.plot_degree_distribution(\n",
    "        type_dist[\"degree\"],\n",
    "        log_scale=True,\n",
    "        title=f\"{node_type.capitalize()} Degree Distribution (log-log scale)\",\n",
    "        save_path=f\"{node_type}_degree_distribution_log.png\"\n",
    "    )\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Centrality Analysis\n",
    "\n",
    "Now let's identify the most central nodes in our knowledge graph using different centrality measures."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Degree Centrality\n",
    "\n",
    "Let's find the most connected nodes in the graph."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate degree centrality for all nodes\n",
    "degree_cent = analyzer.calculate_centrality(centrality_type=\"degree\", top_n=20)\n",
    "\n",
    "# Display results\n",
    "display(degree_cent)\n",
    "\n",
    "# Visualize top results\n",
    "fig = visualizer.plot_centrality_distribution(\n",
    "    degree_cent, \n",
    "    top_n=20,\n",
    "    title=\"Top 20 Nodes by Degree Centrality\",\n",
    "    save_path=\"degree_centrality_top20.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's look at the most central drugs and diseases specifically."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate degree centrality for drugs\n",
    "drug_cent = analyzer.calculate_centrality(centrality_type=\"degree\", node_types=[\"drug\"], top_n=10)\n",
    "\n",
    "# Display results\n",
    "print(\"Top 10 drugs by degree centrality:\")\n",
    "display(drug_cent)\n",
    "\n",
    "# Visualize\n",
    "fig = visualizer.plot_centrality_distribution(\n",
    "    drug_cent, \n",
    "    title=\"Top 10 Drugs by Degree Centrality\",\n",
    "    save_path=\"drug_degree_centrality.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate degree centrality for diseases\n",
    "disease_cent = analyzer.calculate_centrality(centrality_type=\"degree\", node_types=[\"disease\"], top_n=10)\n",
    "\n",
    "# Display results\n",
    "print(\"Top 10 diseases by degree centrality:\")\n",
    "display(disease_cent)\n",
    "\n",
    "# Visualize\n",
    "fig = visualizer.plot_centrality_distribution(\n",
    "    disease_cent, \n",
    "    title=\"Top 10 Diseases by Degree Centrality\",\n",
    "    save_path=\"disease_degree_centrality.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Betweenness Centrality\n",
    "\n",
    "Let's identify nodes that act as bridges in the network."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate betweenness centrality\n",
    "between_cent = analyzer.calculate_centrality(centrality_type=\"betweenness\", top_n=20)\n",
    "\n",
    "# Display results\n",
    "print(\"Top 20 nodes by betweenness centrality:\")\n",
    "display(between_cent)\n",
    "\n",
    "# Visualize\n",
    "fig = visualizer.plot_centrality_distribution(\n",
    "    between_cent, \n",
    "    title=\"Top 20 Nodes by Betweenness Centrality\",\n",
    "    save_path=\"betweenness_centrality.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### PageRank\n",
    "\n",
    "Let's use PageRank to identify influential nodes."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate PageRank centrality\n",
    "pagerank = analyzer.calculate_centrality(centrality_type=\"pagerank\", top_n=20)\n",
    "\n",
    "# Display results\n",
    "print(\"Top 20 nodes by PageRank:\")\n",
    "display(pagerank)\n",
    "\n",
    "# Visualize\n",
    "fig = visualizer.plot_centrality_distribution(\n",
    "    pagerank, \n",
    "    title=\"Top 20 Nodes by PageRank\",\n",
    "    save_path=\"pagerank.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Community Detection\n",
    "\n",
    "Let's identify communities in the knowledge graph."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detect communities\n",
    "communities = analyzer.detect_communities(resolution=1.0)\n",
    "\n",
    "# Print summary\n",
    "print(f\"Detected {communities['num_communities']} communities with modularity {communities['modularity']:.4f}\")\n",
    "\n",
    "# Display statistics for top 5 communities\n",
    "top_communities = communities[\"communities\"][:5]\n",
    "\n",
    "print(\"\\nTop 5 communities:\")\n",
    "for i, comm in enumerate(top_communities):\n",
    "    print(f\"\\nCommunity {i+1} (ID: {comm['community_id']})\")\n",
    "    print(f\"Size: {comm['size']} nodes ({comm['percentage']:.2f}% of total)\")\n",
    "    print(f\"Density: {comm['density']:.4f}\")\n",
    "    print(f\"Node types: {comm['node_types']}\")\n",
    "    print(f\"Key nodes: {[n['name'] for n in comm['key_nodes'][:3]]}\")"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize community size distribution\n",
    "fig = visualizer.plot_community_distribution(\n",
    "    communities, \n",
    "    top_n=10,\n",
    "    title=\"Top 10 Communities by Size\",\n",
    "    save_path=\"community_sizes.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize community composition\n",
    "fig = visualizer.plot_community_composition(\n",
    "    communities, \n",
    "    top_n=5,\n",
    "    title=\"Node Type Composition of Top 5 Communities\",\n",
    "    save_path=\"community_composition.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Path Analysis\n",
    "\n",
    "Let's analyze paths between drugs and diseases in the knowledge graph."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Find Shortest Paths between Drugs and Diseases"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find shortest paths from drugs to diseases\n",
    "paths = analyzer.find_shortest_paths(source_type=\"drug\", target_type=\"disease\", max_paths=10)\n",
    "\n",
    "# Display summary of paths\n",
    "print(f\"Found {len(paths)} paths from drugs to diseases\")\n",
    "\n",
    "# Create a table of paths\n",
    "path_data = []\n",
    "for path in paths:\n",
    "    path_data.append({\n",
    "        \"Source\": path[\"source_name\"],\n",
    "        \"Target\": path[\"target_name\"],\n",
    "        \"Length\": path[\"length\"],\n",
    "        \"Path\": \" -> \".join(path[\"path_names\"])\n",
    "    })\n",
    "\n",
    "path_df = pd.DataFrame(path_data)\n",
    "display(path_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize a Specific Path"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select a path to visualize (the shortest one)\n",
    "if paths:\n",
    "    shortest_path = min(paths, key=lambda x: x[\"length\"])\n",
    "    print(f\"Visualizing path from {shortest_path['source_name']} to {shortest_path['target_name']}\")\n",
    "    print(f\"Path length: {shortest_path['length']}\")\n",
    "    print(f\"Path: {' -> '.join(shortest_path['path_names'])}\")\n",
    "    \n",
    "    # Visualize the path\n",
    "    fig = visualizer.visualize_path(\n",
    "        shortest_path,\n",
    "        save_path=\"shortest_drug_disease_path.png\"\n",
    "    )\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"No paths found\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Find Paths for a Specific Drug-Disease Pair\n",
    "\n",
    "Let's pick one of the most central drugs and diseases and find all paths between them."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get top drug and disease IDs\n",
    "if len(drug_cent) > 0 and len(disease_cent) > 0:\n",
    "    top_drug_id = drug_cent.iloc[0][\"node_id\"]\n",
    "    top_drug_name = drug_cent.iloc[0][\"name\"]\n",
    "    top_disease_id = disease_cent.iloc[0][\"node_id\"]\n",
    "    top_disease_name = disease_cent.iloc[0][\"name\"]\n",
    "    \n",
    "    print(f\"Finding paths between {top_drug_name} and {top_disease_name}\")\n",
    "    \n",
    "    # Find all paths between them\n",
    "    drug_disease_paths = analyzer.find_drug_disease_paths(\n",
    "        drug_id=top_drug_id, \n",
    "        disease_id=top_disease_id,\n",
    "        max_paths=5\n",
    "    )\n",
    "    \n",
    "    # Display paths\n",
    "    print(f\"Found {len(drug_disease_paths)} paths\")\n",
    "    \n",
    "    for i, path in enumerate(drug_disease_paths):\n",
    "        print(f\"\\nPath {i+1}:\")\n",
    "        print(f\"Length: {path['length']}\")\n",
    "        print(f\"Path: {' -> '.join(path['path_names'])}\")\n",
    "        print(f\"Types: {' -> '.join(path['path_types'])}\")\n",
    "    \n",
    "    # Visualize the first path\n",
    "    if drug_disease_paths:\n",
    "        fig = visualizer.visualize_path(\n",
    "            drug_disease_paths[0],\n",
    "            save_path=\"top_drug_disease_path.png\"\n",
    "        )\n",
    "        plt.show()\n",
    "else:\n",
    "    print(\"No drugs or diseases with centrality data\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Common Neighbor Analysis\n",
    "\n",
    "Let's find drug-disease pairs that share common neighbors (potential indirect connections)."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find drug-disease pairs with common neighbors\n",
    "common_neighbors = analyzer.find_common_neighbors(\n",
    "    node_type_a=\"drug\", \n",
    "    node_type_b=\"disease\",\n",
    "    min_neighbors=2,  # At least 2 common neighbors\n",
    "    max_results=20\n",
    ")\n",
    "\n",
    "# Display results\n",
    "print(f\"Found {len(common_neighbors)} drug-disease pairs with at least 2 common neighbors\")\n",
    "\n",
    "# Create a table of results\n",
    "neighbor_data = []\n",
    "for result in common_neighbors:\n",
    "    neighbor_data.append({\n",
    "        \"Drug\": result[\"node_a_name\"],\n",
    "        \"Disease\": result[\"node_b_name\"],\n",
    "        \"Common Neighbors\": result[\"common_neighbors_count\"],\n",
    "        \"Neighbor Names\": \", \".join([n[\"name\"] for n in result[\"common_neighbors\"][:3]]) + \n",
    "                          (\"...\" if len(result[\"common_neighbors\"]) > 3 else \"\")\n",
    "    })\n",
    "\n",
    "neighbor_df = pd.DataFrame(neighbor_data)\n",
    "display(neighbor_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Network Visualization\n",
    "\n",
    "Let's visualize parts of our knowledge graph."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize a Drug-Protein-Disease Subgraph"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract a subgraph with drugs, proteins, and diseases\n",
    "subgraph = analyzer.extract_subgraph(\n",
    "    node_types=[\"drug\", \"protein\", \"disease\"],\n",
    "    edge_types=[\"targets\", \"associated_with\", \"treats\"],\n",
    "    max_nodes=50\n",
    ")\n",
    "\n",
    "print(f\"Extracted subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges\")\n",
    "\n",
    "# Visualize the subgraph\n",
    "fig = visualizer.visualize_graph(\n",
    "    subgraph,\n",
    "    node_color_attribute=\"type\",\n",
    "    node_size_attribute=\"degree\",\n",
    "    edge_color_attribute=\"type\",\n",
    "    layout=\"spring\",\n",
    "    title=\"Drug-Protein-Disease Subgraph\",\n",
    "    save_path=\"drug_protein_disease_subgraph.png\"\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize a Specific Drug's Neighborhood"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the neighborhood of the top drug\n",
    "if len(drug_cent) > 0:\n",
    "    top_drug_id = drug_cent.iloc[0][\"node_id\"]\n",
    "    top_drug_name = drug_cent.iloc[0][\"name\"]\n",
    "    \n",
    "    # Get 1-hop neighborhood\n",
    "    hood = analyzer.get_entity_neighborhood(entity_id=top_drug_id, hops=1)\n",
    "    print(f\"1-hop neighborhood of {top_drug_name} has {hood.number_of_nodes()} nodes and {hood.number_of_edges()} edges\")\n",
    "    \n",
    "    # Visualize\n",
    "    fig = visualizer.visualize_graph(\n",
    "        hood,\n",
    "        node_color_attribute=\"type\",\n",
    "        node_size_attribute=\"degree\",\n",
    "        edge_color_attribute=\"type\",\n",
    "        layout=\"spring\",\n",
    "        title=f\"1-hop Neighborhood of {top_drug_name}\",\n",
    "        save_path=\"top_drug_neighborhood.png\"\n",
    "    )\n",
    "    plt.show()\n",
    "    \n",
    "    # Get 2-hop neighborhood if 1-hop is small\n",
    "    if hood.number_of_nodes() < 30:\n",
    "        hood2 = analyzer.get_entity_neighborhood(entity_id=top_drug_id, hops=2, max_nodes=50)\n",
    "        print(f\"2-hop neighborhood of {top_drug_name} has {hood2.number_of_nodes()} nodes and {hood2.number_of_edges()} edges\")\n",
    "        \n",
    "        # Visualize\n",
    "        fig = visualizer.visualize_graph(\n",
    "            hood2,\n",
    "            node_color_attribute=\"type\",\n",
    "            node_size_attribute=\"degree\",\n",
    "            edge_color_attribute=\"type\",\n",
    "            layout=\"spring\",\n",
    "            title=f\"2-hop Neighborhood of {top_drug_name}\",\n",
    "            save_path=\"top_drug_neighborhood_2hop.png\"\n",
    "        )\n",
    "        plt.show()\n",
    "else:\n",
    "    print(\"No drugs with centrality data\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Drug Repurposing Candidates\n",
    "\n",
    "Let's explore potential drug repurposing candidates by looking at drugs and diseases that are not directly connected but share similar network patterns."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to find potential drug repurposing candidates\n",
    "def find_repurposing_candidates(analyzer, top_n=20):\n",
    "    # Get all drugs and diseases\n",
    "    drugs = [n for n, d in analyzer.graph.nodes(data=True) if d.get(\"type\") == \"drug\"]\n",
    "    diseases = [n for n, d in analyzer.graph.nodes(data=True) if d.get(\"type\") == \"disease\"]\n",
    "    \n",
    "    # Find pairs with common neighbors but no direct connection\n",
    "    candidates = []\n",
    "    \n",
    "    # Get common neighbors\n",
    "    common_neighbors = analyzer.find_common_neighbors(\n",
    "        node_type_a=\"drug\", \n",
    "        node_type_b=\"disease\",\n",
    "        min_neighbors=2\n",
    "    )\n",
    "    \n",
    "    # Filter to keep only pairs without direct connection\n",
    "    for result in common_neighbors:\n",
    "        drug_id = result[\"node_a_id\"]\n",
    "        disease_id = result[\"node_b_id\"]\n",
    "        \n",
    "        # Check if there's a direct edge\n",
    "        if not analyzer.graph.has_edge(drug_id, disease_id):\n",
    "            # Calculate a score based on common neighbors\n",
    "            score = result[\"common_neighbors_count\"]\n",
    "            \n",
    "            # Enhance score based on common neighbor types\n",
    "            protein_neighbors = sum(1 for n in result[\"common_neighbors\"] if n[\"type\"] == \"protein\")\n",
    "            score += protein_neighbors * 0.5  # Give extra weight to protein neighbors\n",
    "            \n",
    "            candidates.append({\n",
    "                \"drug_id\": drug_id,\n",
    "                \"drug_name\": result[\"node_a_name\"],\n",
    "                \"disease_id\": disease_id,\n",
    "                \"disease_name\": result[\"node_b_name\"],\n",
    "                \"common_neighbors\": result[\"common_neighbors_count\"],\n",
    "                \"protein_neighbors\": protein_neighbors,\n",
    "                \"repurposing_score\": score,\n",
    "                \"common_neighbor_names\": [n[\"name\"] for n in result[\"common_neighbors\"]]\n",
    "            })\n",
    "    \n",
    "    # Sort by score\n",
    "    candidates.sort(key=lambda x: x[\"repurposing_score\"], reverse=True)\n",
    "    \n",
    "    # Return top N\n",
    "    return candidates[:top_n] if top_n else candidates"
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find repurposing candidates\n",
    "candidates = find_repurposing_candidates(analyzer, top_n=20)\n",
    "\n",
    "# Display candidates\n",
    "print(f\"Found {len(candidates)} potential drug repurposing candidates\")\n",
    "\n",
    "# Create a table\n",
    "candidate_data = []\n",
    "for cand in candidates:\n",
    "    candidate_data.append({\n",
    "        \"Drug\": cand[\"drug_name\"],\n",
    "        \"Disease\": cand[\"disease_name\"],\n",
    "        \"Repurposing Score\": round(cand[\"repurposing_score\"], 2),\n",
    "        \"Common Neighbors\": cand[\"common_neighbors\"],\n",
    "        \"Protein Neighbors\": cand[\"protein_neighbors\"],\n",
    "        \"Common Neighbor Examples\": \", \".join(cand[\"common_neighbor_names\"][:3]) + \n",
    "                                  (\"...\" if len(cand[\"common_neighbor_names\"]) > 3 else \"\")\n",
    "    })\n",
    "\n",
    "candidate_df = pd.DataFrame(candidate_data)\n",
    "display(candidate_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's visualize the network of one of the top repurposing candidates to understand the common neighbors and indirect connections."
   ]
  },
  {
   "cell_type": "code",
   
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize a top repurposing candidate\n",
    "if candidates:\n",
    "    top_candidate = candidates[0]\n",
    "    drug_id = top_candidate[\"drug_id\"]\n",
    "    disease_id = top_candidate[\"disease_id\"]\n",
    "    \n",
    "    print(f\"Visualizing network for {top_candidate['drug_name']} and {top_candidate['disease_name']}\")\n",
    "    print(f\"Common neighbors: {top_candidate['common_neighbors']}\")\n",
    "    print(f\"Common neighbor examples: {', '.join(top_candidate['common_neighbor_names'][:5])}\")\n",
    "    \n",
    "    # Create a subgraph with the drug, disease, and their common neighbors\n",
    "    nodes = [drug_id, disease_id] + [n[\"id\"] for n in top_candidate[\"common_neighbor_names\"]]\n",
    "    subgraph = analyzer.graph.subgraph(nodes).copy()\n",
    "    \n",
    "    # If subgraph is small, add 1-hop neighbors\n",
    "    if subgraph.number_of_nodes() < 10:\n",
    "        hood1 = analyzer.get_entity_neighborhood(entity_id=drug_id, hops=1)\n",
    "        hood2 = analyzer.get_entity_neighborhood(entity_id=disease_id, hops=1)\n",
    "        \n",
    "        # Combine nodes\n",
    "        all_nodes = set(hood1.nodes()) | set(hood2.nodes())\n",
    "        subgraph = analyzer.graph.subgraph(all_nodes).copy()\n",
    "    \n",
    "    # Visualize\n",
    "    fig = visualizer.visualize_graph(\n",
    "        subgraph,\n",
    "        node_color_attribute=\"type\",\n",
    "        node_size_attribute=\"degree\",\n",
    "        layout=\"spring\",\n",
    "        title=f\"Network for Repurposing Candidate: {top_candidate['drug_name']} → {top_candidate['disease_name']}\",\n",
    "        save_path=\"top_repurposing_candidate_network.png\"\n",
    "    )\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"No repurposing candidates found\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Summary and Conclusions\n",
    "\n",
    "In this notebook, we've analyzed the structure of our knowledge graph and applied various graph analysis techniques to gain insights. Here's a summary of our findings:\n",
    "\n",
    "1. **Graph Statistics**: We examined the basic properties of the graph, including node and edge counts, degree distributions, and connectivity metrics.\n",
    "\n",
    "2. **Centrality Analysis**: We identified the most central nodes in the graph using different centrality measures, highlighting the most important drugs, diseases, and proteins.\n",
    "\n",
    "3. **Community Detection**: We detected communities in the graph, which can reveal groups of related entities and therapeutic areas.\n",
    "\n",
    "4. **Path Analysis**: We analyzed paths between drugs and diseases, which can help understand potential mechanisms of action.\n",
    "\n",
    "5. **Common Neighbor Analysis**: We found drug-disease pairs with common neighbors, which can suggest indirect relationships.\n",
    "\n",
    "6. **Network Visualization**: We visualized various subgraphs to better understand the relationships between entities.\n",
    "\n",
    "7. **Drug Repurposing Candidates**: We identified potential drug repurposing candidates based on network patterns.\n",
    "\n",
    "These analyses provide a foundation for the next phase of the project: feature engineering for our graph neural network model. By understanding the structure and properties of the knowledge graph, we can design more effective features and model architectures for drug-disease interaction prediction."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\n",
    "\n",
    "1. **Feature Engineering**: Use insights from graph analysis to design node and edge features for the graph neural network model.\n",
    "\n",
    "2. **Model Implementation**: Implement the graph neural network architecture as defined in the project plan.\n",
    "\n",
    "3. **Training & Evaluation**: Train the model on the knowledge graph and evaluate its performance.\n",
    "\n",
    "4. **Explainability**: Develop methods to explain model predictions using path and subgraph analysis.\n",
    "\n",
    "5. **API Development**: Create an API for making drug-disease interaction predictions."
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

with open("graph_analysis.ipynb", "w") as f:
    json.dump(nc, f, indent=2)