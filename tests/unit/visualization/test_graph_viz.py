# tests/unit/visualization/test_graph_viz.py
import pytest
import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile

# Add parent directory to path to allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ddi.visualization.graph_viz import GraphVisualizer
from ddi.analysis.graph_analysis import GraphAnalyzer

class TestGraphVisualizer:
    """Test the GraphVisualizer class"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing"""
        # Create a directed multigraph
        G = nx.MultiDiGraph()
        
        # Add drug nodes
        G.add_node("DB00001", type="drug", name="Drug A")
        G.add_node("DB00002", type="drug", name="Drug B")
        G.add_node("DB00003", type="drug", name="Drug C")
        
        # Add disease nodes
        G.add_node("D000001", type="disease", name="Disease X")
        G.add_node("D000002", type="disease", name="Disease Y")
        
        # Add protein nodes
        G.add_node("P12345", type="protein", name="Protein Alpha")
        G.add_node("P67890", type="protein", name="Protein Beta")
        
        # Add category node
        G.add_node("C00001", type="category", name="Category Z")
        
        # Add edges
        G.add_edge("DB00001", "P12345", type="targets", score=0.8)
        G.add_edge("DB00002", "P12345", type="targets", score=0.7)
        G.add_edge("DB00002", "P67890", type="targets", score=0.9)
        G.add_edge("DB00003", "P67890", type="targets", score=0.6)
        
        G.add_edge("P12345", "D000001", type="associated_with", score=0.5)
        G.add_edge("P67890", "D000001", type="associated_with", score=0.6)
        G.add_edge("P67890", "D000002", type="associated_with", score=0.7)
        
        G.add_edge("DB00001", "D000001", type="treats", score=0.4)
        G.add_edge("DB00002", "D000002", type="treats", score=0.5)
        
        G.add_edge("DB00001", "C00001", type="has_category")
        G.add_edge("DB00002", "C00001", type="has_category")
        
        return G
    
    @pytest.fixture
    def sample_centrality(self, sample_graph):
        """Create sample centrality data for testing"""
        analyzer = GraphAnalyzer(sample_graph)
        return analyzer.calculate_centrality(centrality_type="degree")
    
    @pytest.fixture
    def sample_communities(self, sample_graph):
        """Create sample community detection results for testing"""
        analyzer = GraphAnalyzer(sample_graph)
        return analyzer.detect_communities()
    
    @pytest.fixture
    def sample_path(self, sample_graph):
        """Create a sample path for testing"""
        analyzer = GraphAnalyzer(sample_graph)
        paths = analyzer.find_drug_disease_paths(drug_id="DB00001", disease_id="D000001")
        if not paths:
            pytest.skip("No path found in sample graph")
        return paths[0]
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for output files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_init(self, temp_output_dir):
        """Test initialization of GraphVisualizer"""
        # Test with output directory
        viz = GraphVisualizer(output_dir=temp_output_dir)
        assert viz.output_dir == temp_output_dir
        
        # Test without output directory
        viz2 = GraphVisualizer()
        assert viz2.output_dir is None
    
    def test_plot_degree_distribution(self, sample_graph, temp_output_dir):
        """Test plotting degree distribution"""
        # Get degree distribution
        analyzer = GraphAnalyzer(sample_graph)
        degrees = analyzer.get_degree_distribution()["degree"]
        
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.plot_degree_distribution(degrees)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.plot_degree_distribution(degrees, save_path="degree_dist.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "degree_dist.png"))
        plt.close(fig)
        
        # Test with log scale false
        fig = viz.plot_degree_distribution(degrees, log_scale=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_node_type_distribution(self, sample_graph, temp_output_dir):
        """Test plotting node type distribution"""
        # Get node types
        analyzer = GraphAnalyzer(sample_graph)
        node_types = analyzer.get_basic_statistics()["node_types"]
        
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.plot_node_type_distribution(node_types)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.plot_node_type_distribution(node_types, save_path="node_types.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "node_types.png"))
        plt.close(fig)
    
    def test_plot_edge_type_distribution(self, sample_graph, temp_output_dir):
        """Test plotting edge type distribution"""
        # Get edge types
        analyzer = GraphAnalyzer(sample_graph)
        edge_types = analyzer.get_basic_statistics()["edge_types"]
        
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.plot_edge_type_distribution(edge_types)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.plot_edge_type_distribution(edge_types, save_path="edge_types.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "edge_types.png"))
        plt.close(fig)
    
    def test_plot_centrality_distribution(self, sample_centrality, temp_output_dir):
        """Test plotting centrality distribution"""
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.plot_centrality_distribution(sample_centrality)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.plot_centrality_distribution(sample_centrality, save_path="centrality.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "centrality.png"))
        plt.close(fig)
        
        # Test with node type filter
        fig = viz.plot_centrality_distribution(sample_centrality, node_type="drug")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with log scale
        fig = viz.plot_centrality_distribution(sample_centrality, log_scale=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with custom title
        fig = viz.plot_centrality_distribution(sample_centrality, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_community_distribution(self, sample_communities, temp_output_dir):
        """Test plotting community distribution"""
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.plot_community_distribution(sample_communities)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.plot_community_distribution(sample_communities, save_path="communities.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "communities.png"))
        plt.close(fig)
    
    def test_plot_community_composition(self, sample_communities, temp_output_dir):
        """Test plotting community composition"""
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.plot_community_composition(sample_communities)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.plot_community_composition(sample_communities, save_path="composition.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "composition.png"))
        plt.close(fig)
    
    def test_visualize_graph(self, sample_graph, temp_output_dir):
        """Test graph visualization"""
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.visualize_graph(sample_graph)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.visualize_graph(sample_graph, save_path="graph.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "graph.png"))
        plt.close(fig)
        
        # Test with different layouts
        for layout in ["spring", "circular", "kamada_kawai", "spectral"]:
            fig = viz.visualize_graph(sample_graph, layout=layout)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        
        # Test with node size attribute
        fig = viz.visualize_graph(sample_graph, node_size_attribute="degree")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with edge color attribute
        fig = viz.visualize_graph(sample_graph, edge_color_attribute="type")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_visualize_path(self, sample_path, temp_output_dir):
        """Test path visualization"""
        # Create visualizer
        viz = GraphVisualizer(output_dir=temp_output_dir)
        
        # Test without saving
        fig = viz.visualize_path(sample_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with saving
        fig = viz.visualize_path(sample_path, save_path="path.png")
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(os.path.join(temp_output_dir, "path.png"))
        plt.close(fig)