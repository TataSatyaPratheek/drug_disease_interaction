# tests/unit/analysis/test_graph_analysis.py
import pytest
import os
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ddi.analysis.graph_analysis import GraphAnalyzer

class TestGraphAnalyzer:
    """Test the GraphAnalyzer class"""
    
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
    
    def test_init(self, sample_graph):
        """Test initialization of GraphAnalyzer"""
        analyzer = GraphAnalyzer(sample_graph)
        
        assert analyzer.graph is sample_graph
        assert isinstance(analyzer.undirected_graph, nx.Graph)
        assert analyzer._cache == {}
    
    def test_get_basic_statistics(self, sample_graph):
        """Test getting basic graph statistics"""
        analyzer = GraphAnalyzer(sample_graph)
        
        stats = analyzer.get_basic_statistics()
        
        assert stats["num_nodes"] == 8
        assert stats["num_edges"] == 11
        assert set(stats["node_types"].keys()) == {"drug", "disease", "protein", "category"}
        assert stats["node_types"]["drug"] == 3
        assert stats["node_types"]["disease"] == 2
        assert stats["node_types"]["protein"] == 2
        assert stats["node_types"]["category"] == 1
        
        assert set(stats["edge_types"].keys()) == {"targets", "associated_with", "treats", "has_category"}
        assert stats["edge_types"]["targets"] == 4
        assert stats["edge_types"]["associated_with"] == 3
        assert stats["edge_types"]["treats"] == 2
        assert stats["edge_types"]["has_category"] == 2
        
        assert "degree_stats" in stats
        assert "in_degree_stats" in stats
        assert "out_degree_stats" in stats
        assert "density" in stats
        assert "num_connected_components" in stats
        assert stats["num_connected_components"] == 1  # Graph should be connected
    
    def test_get_degree_distribution(self, sample_graph):
        """Test getting degree distribution"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Test overall distribution
        dist = analyzer.get_degree_distribution()
        assert "degree" in dist
        assert "in_degree" in dist
        assert "out_degree" in dist
        assert len(dist["degree"]) == 8  # Number of nodes
        
        # Test distribution for specific node type
        drug_dist = analyzer.get_degree_distribution(node_type="drug")
        assert len(drug_dist["degree"]) == 3  # Number of drug nodes
        assert len(drug_dist["out_degree"]) == 3 # Check if the correct number of drug nodes were processed
    
    def test_calculate_centrality(self, sample_graph):
        """Test calculating centrality metrics"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Test degree centrality
        degree_cent = analyzer.calculate_centrality(centrality_type="degree")
        assert isinstance(degree_cent, pd.DataFrame)
        assert set(degree_cent.columns) == {"node_id", "name", "type", "score"}
        assert len(degree_cent) == 8  # All nodes
        
        # Test degree centrality for specific node types
        drug_cent = analyzer.calculate_centrality(centrality_type="degree", node_types=["drug"])
        assert len(drug_cent) == 3  # Only drug nodes
        
        # Test top N results
        top_cent = analyzer.calculate_centrality(centrality_type="degree", top_n=2)
        assert len(top_cent) == 2
        
        # Test other centrality types
        for cent_type in ["in_degree", "out_degree", "betweenness", "pagerank"]:
            cent = analyzer.calculate_centrality(centrality_type=cent_type)
            assert isinstance(cent, pd.DataFrame)
            assert len(cent) == 8
    
    def test_find_shortest_paths(self, sample_graph):
        """Test finding shortest paths between node types"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Find paths from drugs to diseases
        paths = analyzer.find_shortest_paths(source_type="drug", target_type="disease")
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        
        # Check path structure
        path = paths[0]
        assert "source_id" in path
        assert "source_name" in path
        assert "target_id" in path
        assert "target_name" in path
        assert "length" in path
        assert "path" in path
        assert "path_names" in path
        assert "path_types" in path
        
        # Verify paths start with drug and end with disease
        for path in paths:
            assert path["path_types"][0] == "drug"
            assert path["path_types"][-1] == "disease"
    
    def test_find_common_neighbors(self, sample_graph):
        """Test finding common neighbors between node types"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Find common neighbors between drugs and diseases
        common_neighbors = analyzer.find_common_neighbors(
            node_type_a="drug", 
            node_type_b="disease", 
            min_neighbors=1
        )
        
        assert isinstance(common_neighbors, list)
        assert len(common_neighbors) > 0
        
        # Check result structure
        result = common_neighbors[0]
        assert "node_a_id" in result
        assert "node_a_name" in result
        assert "node_a_type" in result
        assert "node_b_id" in result
        assert "node_b_name" in result
        assert "node_b_type" in result
        assert "common_neighbors_count" in result
        assert "common_neighbors" in result
        
        # Verify types
        for result in common_neighbors:
            assert result["node_a_type"] == "drug"
            assert result["node_b_type"] == "disease"
            assert result["common_neighbors_count"] >= 1
    
    def test_detect_communities(self, sample_graph):
        """Test community detection"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Detect communities
        communities = analyzer.detect_communities()
        
        assert isinstance(communities, dict)
        assert "num_communities" in communities
        assert "modularity" in communities
        assert "communities" in communities
        assert "node_to_community" in communities
        
        # Check community structure
        assert communities["num_communities"] > 0
        assert len(communities["communities"]) == communities["num_communities"]
        assert isinstance(communities["modularity"], float)
        
        # Check individual community
        community = communities["communities"][0]
        assert "community_id" in community
        assert "size" in community
        assert "percentage" in community
        assert "node_types" in community
        assert "density" in community
        assert "key_nodes" in community
    
    def test_extract_subgraph(self, sample_graph):
        """Test extracting subgraph by node and edge types"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Extract drug-protein subgraph
        subgraph = analyzer.extract_subgraph(
            node_types=["drug", "protein"], 
            edge_types=["targets"]
        )
        
        assert isinstance(subgraph, nx.MultiDiGraph)
        assert subgraph.number_of_nodes() <= 5  # 3 drugs + 2 proteins
        
        # Check node types in subgraph
        node_types = [data["type"] for _, data in subgraph.nodes(data=True)]
        assert set(node_types) <= {"drug", "protein"}
        
        # Check edge types in subgraph
        edge_types = [data["type"] for _, _, data in subgraph.edges(data=True)]
        assert set(edge_types) == {"targets"}
    
    def test_get_entity_neighborhood(self, sample_graph):
        """Test getting entity neighborhood"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Get neighborhood for a drug
        hood = analyzer.get_entity_neighborhood(entity_id="DB00001", hops=1)
        
        assert isinstance(hood, nx.MultiDiGraph)
        assert hood.number_of_nodes() > 1
        assert "DB00001" in hood.nodes()
        
        # Check 2-hop neighborhood
        hood2 = analyzer.get_entity_neighborhood(entity_id="DB00001", hops=2)
        assert hood2.number_of_nodes() >= hood.number_of_nodes()
    
    def test_find_drug_disease_paths(self, sample_graph):
        """Test finding paths between a drug and a disease"""
        analyzer = GraphAnalyzer(sample_graph)
        
        # Find paths between a specific drug and disease
        paths = analyzer.find_drug_disease_paths(
            drug_id="DB00001", 
            disease_id="D000001"
        )
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        
        # Check path structure
        path = paths[0]
        assert "drug_id" in path
        assert "drug_name" in path
        assert "disease_id" in path
        assert "disease_name" in path
        assert "length" in path
        assert "path" in path
        assert "path_names" in path
        assert "path_types" in path
        
        # Verify path
        assert path["drug_id"] == "DB00001"
        assert path["disease_id"] == "D000001"
        assert path["path"][0] == "DB00001"
        assert path["path"][-1] == "D000001"