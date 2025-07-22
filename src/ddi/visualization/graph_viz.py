# src/ddi/visualization/graph_viz.py
import os
import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import matplotlib.colors as mcolors
from collections import Counter

class GraphVisualizer:
    """Utilities for visualizing drug-disease knowledge graph"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the graph visualizer
        
        Args:
            output_dir: Directory to save visualizations (optional)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up default style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
    
    def plot_degree_distribution(self, degrees: List[int], log_scale: bool = True, 
                                title: str = "Degree Distribution", 
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot degree distribution
        
        Args:
            degrees: List of node degrees
            log_scale: Whether to use log scale for axes
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate degree frequency
        degree_counts = Counter(degrees)
        unique_degrees = sorted(degree_counts.keys())
        frequencies = [degree_counts[d] for d in unique_degrees]
        
        # Create plot
        bars = ax.bar(unique_degrees, frequencies, alpha=0.7, width=0.8, color='steelblue')
        
        # Set axes limits and labels
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        
        # Apply log scale if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Ensure x-axis shows integer ticks
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
            ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        
        # Add annotations for highest degrees
        top_n = 3
        for i, (degree, freq) in enumerate(sorted(degree_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]):
            ax.annotate(f"k={degree}, n={freq}", 
                      xy=(degree, freq),
                      xytext=(10, (-1)**i * 20),
                      textcoords="offset points",
                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved degree distribution plot to {full_path}")
        
        return fig
    
    def plot_node_type_distribution(self, node_types: Dict[str, int], 
                                    title: str = "Node Type Distribution",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of node types
        
        Args:
            node_types: Dictionary mapping node types to counts
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort node types by count
        sorted_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)
        types = [t[0] for t in sorted_types]
        counts = [t[1] for t in sorted_types]
        
        # Create plot
        bars = ax.bar(types, counts, alpha=0.7, color='skyblue')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        # Set axes labels and title
        ax.set_xlabel("Node Type")
        ax.set_ylabel("Count")
        ax.set_title(title)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved node type distribution plot to {full_path}")
        
        return fig
    
    def plot_edge_type_distribution(self, edge_types: Dict[str, int], 
                                    title: str = "Edge Type Distribution",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of edge types
        
        Args:
            edge_types: Dictionary mapping edge types to counts
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort edge types by count
        sorted_types = sorted(edge_types.items(), key=lambda x: x[1], reverse=True)
        types = [t[0] for t in sorted_types]
        counts = [t[1] for t in sorted_types]
        
        # Create plot
        bars = ax.bar(types, counts, alpha=0.7, color='lightcoral')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        # Set axes labels and title
        ax.set_xlabel("Edge Type")
        ax.set_ylabel("Count")
        ax.set_title(title)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved edge type distribution plot to {full_path}")
        
        return fig
    
    def plot_centrality_distribution(self, centrality_scores: pd.DataFrame, top_n: int = 20,
                                    node_type: Optional[str] = None, log_scale: bool = False,
                                    title: Optional[str] = None, 
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of centrality scores
        
        Args:
            centrality_scores: DataFrame with centrality scores
            top_n: Number of top nodes to display
            node_type: Filter by node type (optional)
            log_scale: Whether to use log scale for y-axis
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Filter by node type if specified
        if node_type:
            df = centrality_scores[centrality_scores["type"] == node_type].copy()
            if df.empty:
                self.logger.warning(f"No nodes of type {node_type} found in centrality scores")
                return None
        else:
            df = centrality_scores.copy()
        
        # Take top N nodes
        df = df.head(top_n)
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(df["name"], df["score"], alpha=0.7, color='lightseagreen')
        
        # Add node type as color
        if "type" in df.columns and node_type is None:
            # Get unique node types
            types = df["type"].unique()
            
            # Create color map
            cmap = plt.cm.get_cmap("tab10", len(types))
            type_to_color = {t: mcolors.rgb2hex(cmap(i)) for i, t in enumerate(types)}
            
            # Set bar colors
            for i, bar in enumerate(bars):
                bar.set_color(type_to_color[df.iloc[i]["type"]])
            
            # Add legend
            handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in type_to_color.values()]
            ax.legend(handles, type_to_color.keys(), title="Node Type")
        
        # Set axes labels and title
        ax.set_xlabel("Centrality Score")
        ax.set_ylabel("Node Name")
        
        if title is None:
            title_str = "Centrality Distribution"
            if node_type:
                title_str += f" ({node_type} nodes)"
        else:
            title_str = title
            
        ax.set_title(title_str)
        
        # Apply log scale if requested
        if log_scale:
            ax.set_xscale("log")
        
        # Adjust layout
        plt.tight_layout()
        
        # Add grid
        ax.grid(True, axis="x", alpha=0.3)
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved centrality distribution plot to {full_path}")
        
        return fig
    
    def plot_community_distribution(self, communities: Dict[str, Any], 
                                   top_n: int = 10,
                                   title: str = "Community Size Distribution",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of community sizes
        
        Args:
            communities: Dictionary with community detection results
            top_n: Number of top communities to display
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top N communities
        top_communities = communities["communities"][:top_n]
        
        # Get community IDs and sizes
        community_ids = [f"C{comm['community_id']}" for comm in top_communities]
        sizes = [comm["size"] for comm in top_communities]
        
        # Create plot
        bars = ax.bar(community_ids, sizes, alpha=0.7, color='mediumpurple')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        # Set axes labels and title
        ax.set_xlabel("Community ID")
        ax.set_ylabel("Size (Number of Nodes)")
        ax.set_title(title)
        
        # Add metadata in text box
        textstr = f"Number of Communities: {communities['num_communities']}\n"
        textstr += f"Modularity: {communities['modularity']:.4f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment="top", bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved community distribution plot to {full_path}")
        
        return fig

    def plot_community_composition(self, communities: Dict[str, Any], 
                                  top_n: int = 5,
                                  title: str = "Community Composition",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot composition of top communities by node type
        
        Args:
            communities: Dictionary with community detection results
            top_n: Number of top communities to display
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top N communities
        top_communities = communities["communities"][:top_n]
        
        # Prepare data for stacked bar chart
        community_ids = [f"C{comm['community_id']}" for comm in top_communities]
        
        # Get all node types across communities
        all_node_types = set()
        for comm in top_communities:
            all_node_types.update(comm["node_types"].keys())
        all_node_types = sorted(all_node_types)
        
        # Create data matrix for stacked bars
        data = np.zeros((len(top_communities), len(all_node_types)))
        for i, comm in enumerate(top_communities):
            for j, node_type in enumerate(all_node_types):
                data[i, j] = comm["node_types"].get(node_type, 0)
        
        # Plot stacked bars
        bottom = np.zeros(len(top_communities))
        
        # Create color map
        cmap = plt.cm.get_cmap("tab10", len(all_node_types))
        colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(all_node_types))]
        
        for i, node_type in enumerate(all_node_types):
            ax.bar(community_ids, data[:, i], bottom=bottom, label=node_type, alpha=0.7, color=colors[i])
            bottom += data[:, i]
        
        # Set axes labels and title
        ax.set_xlabel("Community ID")
        ax.set_ylabel("Number of Nodes")
        ax.set_title(title)
        
        # Add legend
        ax.legend(title="Node Type")
        
        # Add percentage labels
        for i, comm_id in enumerate(community_ids):
            total = sum(data[i, :])
            y_pos = 0
            for j, count in enumerate(data[i, :]):
                if count / total > 0.05:  # Only label segments that are at least 5% of the total
                    y_pos += count / 2
                    ax.annotate(f"{count:.0f} ({count/total*100:.1f}%)",
                             xy=(i, y_pos),
                             xytext=(0, 0),
                             textcoords="offset points",
                             ha='center', va='center')
                y_pos += count / 2
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved community composition plot to {full_path}")
        
        return fig
    
    def visualize_graph(self, graph: nx.Graph, node_color_attribute: str = "type",
                      node_size_attribute: Optional[str] = None,
                      edge_color_attribute: Optional[str] = None,
                      title: str = "Graph Visualization",
                      layout: str = "spring",
                      save_path: Optional[str] = None) -> plt.Figure:
        """Visualize a graph
        
        Args:
            graph: NetworkX graph to visualize
            node_color_attribute: Node attribute to use for coloring
            node_size_attribute: Node attribute to use for sizing (optional)
            edge_color_attribute: Edge attribute to use for coloring (optional)
            title: Plot title
            layout: Layout algorithm to use (spring, circular, kamada_kawai, spectral)
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Check graph size
        if graph.number_of_nodes() > 500:
            self.logger.warning(f"Graph is very large ({graph.number_of_nodes()} nodes). Visualization may be cluttered.")
            
            # Offer to sample a smaller subgraph
            if graph.number_of_nodes() > 1000:
                self.logger.info("Sampling 500 nodes for visualization...")
                nodes = list(graph.nodes())
                sampled_nodes = np.random.choice(nodes, size=500, replace=False)
                graph = graph.subgraph(sampled_nodes).copy()
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        else:
            self.logger.warning(f"Unknown layout: {layout}. Using spring layout.")
            pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=42)
        
        # Set up node colors
        if node_color_attribute and node_color_attribute in graph.nodes[list(graph.nodes())[0]]:
            # Get unique values for the attribute
            attr_values = set()
            for n, data in graph.nodes(data=True):
                if node_color_attribute in data:
                    attr_values.add(data[node_color_attribute])
            attr_values = sorted(attr_values)
            
            # Create color map
            cmap = plt.cm.get_cmap("tab10", len(attr_values))
            value_to_color = {val: mcolors.rgb2hex(cmap(i)) for i, val in enumerate(attr_values)}
            
            # Map node attribute to color
            node_colors = [value_to_color.get(graph.nodes[n].get(node_color_attribute), "grey") for n in graph.nodes()]
            
            # Add legend
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10) 
                     for color in value_to_color.values()]
            ax.legend(handles, value_to_color.keys(), title=node_color_attribute, loc="upper right")
        else:
            # Use default color
            node_colors = "skyblue"
            
        # Set up node sizes
        if node_size_attribute:
            # Use attribute for sizing
            if node_size_attribute == "degree":
                node_sizes = [10 + graph.degree(n) * 2 for n in graph.nodes()]
            elif node_size_attribute in graph.nodes[list(graph.nodes())[0]]:
                # Get min/max values for scaling
                attr_values = [graph.nodes[n].get(node_size_attribute, 0) for n in graph.nodes()]
                min_val = min(attr_values)
                max_val = max(attr_values)
                
                # Scale to 5-50 size range
                if min_val == max_val:
                    node_sizes = [30] * len(graph.nodes())
                else:
                    node_sizes = [5 + 45 * (graph.nodes[n].get(node_size_attribute, 0) - min_val) / 
                                (max_val - min_val) for n in graph.nodes()]
            else:
                # Use default size
                node_sizes = 30
        else:
            # Use default size
            node_sizes = 30
            
        # Set up edge colors
        if edge_color_attribute and isinstance(graph, nx.MultiDiGraph):
            # Handle edge attributes for MultiDiGraph
            edge_colors = []
            edges = []
            
            # Get unique values for the attribute
            attr_values = set()
            for u, v, data in graph.edges(data=True):
                if edge_color_attribute in data:
                    attr_values.add(data[edge_color_attribute])
            attr_values = sorted(attr_values)
            
            # Create color map
            cmap = plt.cm.get_cmap("tab20", len(attr_values))
            value_to_color = {val: mcolors.rgb2hex(cmap(i)) for i, val in enumerate(attr_values)}
            
            # Create edge list and colors
            for u, v, data in graph.edges(data=True):
                edges.append((u, v))
                edge_colors.append(value_to_color.get(data.get(edge_color_attribute), "grey"))
                
            # Add edge legend
            edge_handles = [plt.Line2D([0], [0], color=color, linewidth=2) 
                          for color in value_to_color.values()]
            edge_labels = list(value_to_color.keys())
            
            # Add second legend for edges
            ax.legend(edge_handles, edge_labels, title=edge_color_attribute, loc="lower right")
        else:
            # Use default edge color
            edge_colors = "grey"
            edges = list(graph.edges())
        
        # Draw the graph
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges, edge_color=edge_colors, alpha=0.5, arrows=True)
        
        # Add labels for small graphs
        if graph.number_of_nodes() <= 50:
            # Get node names
            labels = {n: graph.nodes[n].get("name", n) for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color="black")
        
        # Set title and remove axes
        plt.title(title)
        plt.axis("off")
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved graph visualization to {full_path}")
        
        return fig
    
    def visualize_path(self, path: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Visualize a path between nodes
        
        Args:
            path: Path dictionary with nodes and edges
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Get path information
        node_ids = path["path"]
        node_names = path["path_names"]
        node_types = path["path_types"]
        
        # Create positions for nodes (horizontal line)
        pos = {node_id: (i, 0) for i, node_id in enumerate(node_ids)}
        
        # Create a new graph just for this path
        G = nx.DiGraph()
        
        # Add nodes with properties
        for i, node_id in enumerate(node_ids):
            G.add_node(node_id, name=node_names[i], type=node_types[i])
        
        # Add edges
        for i in range(len(node_ids) - 1):
            G.add_edge(node_ids[i], node_ids[i + 1])
        
        # Set up node colors by type
        node_type_to_color = {
            "drug": "skyblue",
            "disease": "salmon",
            "protein": "lightgreen",
            "polypeptide": "gold",
            "category": "violet",
            "unknown": "grey"
        }
        
        node_colors = [node_type_to_color.get(node_type, "grey") for node_type in node_types]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.7, arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, labels={node_id: name for node_id, name in zip(node_ids, node_names)},
                             font_size=10, font_color="black")
        
        # Add node type legend
        unique_types = set(node_types)
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=node_type_to_color.get(t, "grey"), 
                           markersize=10) for t in unique_types]
        ax.legend(handles, unique_types, title="Node Type", loc="upper center", 
                bbox_to_anchor=(0.5, 1.15), ncol=len(unique_types))
        
        # Set title and remove axes
        source_name = node_names[0]
        target_name = node_names[-1]
        plt.title(f"Path from {source_name} to {target_name} (length: {len(node_ids) - 1})")
        plt.axis("off")
        
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            if not save_path.endswith((".png", ".jpg", ".pdf", ".svg")):
                save_path += ".png"
                
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved path visualization to {full_path}")
        
        return fig