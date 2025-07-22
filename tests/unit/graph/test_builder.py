# tests/unit/graph/test_builder.py
import pytest
import os
import pickle
import json
import networkx as nx
from pathlib import Path
from ddi.graph.builder import KnowledgeGraphBuilder

class TestKnowledgeGraphBuilder:
    """Test the KnowledgeGraphBuilder class"""
    
    def test_init(self, test_output_dir):
        """Test initialization of KnowledgeGraphBuilder"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        assert builder.output_dir == str(test_output_dir)
        assert isinstance(builder.graph, nx.MultiDiGraph)
        assert builder.node_types == {}
        assert builder.edge_types == {}
        assert builder.node_counter == 0
        assert builder.edge_counter == 0
    
    def test_build_graph_from_drugbank(self, sample_drugbank_data, test_output_dir):
        """Test building graph from DrugBank data"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Build graph
        graph = builder.build_graph_from_drugbank(sample_drugbank_data)
        
        # Check graph structure
        assert isinstance(graph, nx.MultiDiGraph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Check that drug nodes were added
        assert graph.has_node("DB00001")
        assert graph.has_node("DB00002")
        assert graph.nodes["DB00001"]["type"] == "drug"
        assert graph.nodes["DB00001"]["name"] == "Test Drug 1"
        
        # Check that category nodes were added
        assert any(graph.nodes[n]["type"] == "category" for n in graph.nodes())
        
        # Check that target nodes were added
        assert graph.has_node("BE0000001")
        assert graph.nodes["BE0000001"]["type"] == "protein"
        
        # Check that relationships were added
        assert any(data["type"] == "targets" for _, _, data in graph.edges(data=True))
    
    def test_add_disease_data(self, sample_mesh_data, test_output_dir):
        """Test adding disease data to graph"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Add disease data
        builder.add_disease_data(sample_mesh_data)
        
        # Check graph structure
        assert builder.graph.number_of_nodes() > 0
        
        # Check that disease nodes were added
        assert builder.graph.has_node("D000001")
        assert builder.graph.has_node("D000002")
        assert builder.graph.nodes["D000001"]["type"] == "disease"
        assert builder.graph.nodes["D000001"]["name"] == "Test Disease 1"
    
    def test_add_drug_disease_associations(self, sample_drugbank_data, sample_mesh_data, test_output_dir):
        """Test adding drug-disease associations"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Build initial graph
        builder.build_graph_from_drugbank(sample_drugbank_data)
        builder.add_disease_data(sample_mesh_data)
        
        # Create associations
        associations = [
            {
                "drug_id": "DB00001",
                "disease_id": "D000001",
                "score": 0.8,
                "source": "manual"
            },
            {
                "drug_id": "DB00002",
                "disease_id": "D000002",
                "score": 0.7,
                "source": "manual"
            },
            {
                "drug_id": "NONEXISTENT",  # This should be skipped
                "disease_id": "D000001",
                "score": 0.6,
                "source": "manual"
            }
        ]
        
        # Add associations
        builder.add_drug_disease_associations(associations)
        
        # Check that relationships were added
        edges = list(builder.graph.edges(data=True))
        treat_edges = [e for e in edges if e[2]["type"] == "treats"]
        
        assert len(treat_edges) == 2
        
        # Check specific edge
        edge_data = None
        for s, t, d in edges:
            if s == "DB00001" and t == "D000001" and d["type"] == "treats":
                edge_data = d
                break
                
        assert edge_data is not None
        assert edge_data["score"] == 0.8
        assert edge_data["source"] == "manual"
    
    def test_add_drug_target_associations(self, test_output_dir):
        """Test adding drug-target associations"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Add nodes manually
        builder._add_node("DB00001", "drug", "Test Drug 1", {})
        builder._add_node("ENSG00000123456", "protein", "Test Target", {})
        
        # Create associations
        associations = [
            {
                "drug_id": "DB00001",
                "target_id": "ENSG00000123456",
                "score": 0.8,
                "mechanism": "inhibitor"
            },
            {
                "drug_id": "NONEXISTENT",  # This should be skipped
                "target_id": "ENSG00000123456",
                "score": 0.7,
                "mechanism": "activator"
            }
        ]
        
        # Add associations
        builder.add_drug_target_associations(associations)
        
        # Check that relationships were added
        edges = list(builder.graph.edges(data=True))
        target_edges = [e for e in edges if e[2]["type"] == "targets"]
        
        assert len(target_edges) == 1
        
        # Check specific edge
        edge_data = target_edges[0][2]
        assert edge_data["score"] == 0.8
        assert edge_data["mechanism"] == "inhibitor"
    
    def test_add_target_disease_associations(self, test_output_dir):
        """Test adding target-disease associations"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Add nodes manually
        builder._add_node("ENSG00000123456", "protein", "Test Target", {})
        builder._add_node("D000001", "disease", "Test Disease", {})
        
        # Create associations
        associations = [
            {
                "target_id": "ENSG00000123456",
                "disease_id": "D000001",
                "score": 0.7,
                "evidence": "genetic_association"
            }
        ]
        
        # Add associations
        builder.add_target_disease_associations(associations)
        
        # Check that relationships were added
        edges = list(builder.graph.edges(data=True))
        assoc_edges = [e for e in edges if e[2]["type"] == "associated_with"]
        
        assert len(assoc_edges) == 1
        
        # Check specific edge
        edge_data = assoc_edges[0][2]
        assert edge_data["score"] == 0.7
        assert edge_data["evidence"] == "genetic_association"
    
    def test_get_statistics(self, sample_graph, test_output_dir):
        """Test getting graph statistics"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Set graph
        builder.graph = sample_graph
        
        # Update counters
        builder.node_types = {
            "drug": 2,
            "protein": 1,
            "polypeptide": 1,
            "disease": 2
        }
        
        builder.edge_types = {
            "targets": 1,
            "has_polypeptide": 1,
            "treats": 1,
            "associated_with": 1
        }
        
        # Get statistics
        stats = builder.get_statistics()
        
        assert stats["num_nodes"] == 6
        assert stats["num_edges"] == 4
        assert stats["node_types"]["drug"] == 2
        assert stats["edge_types"]["treats"] == 1
        assert "avg_degree" in stats
        assert "num_connected_components" in stats
        assert "largest_component_size" in stats
    
    def test_save_graph(self, sample_graph, test_output_dir):
        """Test saving graph in different formats"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        # Set graph
        builder.graph = sample_graph
        
        # Set type counters
        builder.node_types = {
            "drug": 2,
            "protein": 1,
            "polypeptide": 1,
            "disease": 2
        }
        
        builder.edge_types = {
            "targets": 1,
            "has_polypeptide": 1,
            "treats": 1,
            "associated_with": 1
        }
        
        # Save graph
        output_files = builder.save_graph(formats=["graphml", "pickle"])
        
        # Check that files were created
        assert os.path.exists(output_files["graphml"])
        assert os.path.exists(output_files["pickle"])
        assert os.path.exists(output_files["mappings"])
        
        # Verify GraphML file
        g_graphml = nx.read_graphml(output_files["graphml"])
        # assert g_graphml.number_of_nodes() == sample_graph.number_of_nodes() # Original assertion
        # More robust check: compare the set of node IDs
        assert set(g_graphml.nodes()) == set(sample_graph.nodes())

        
        # Verify pickle file
        with open(output_files["pickle"], "rb") as f:
            g_pickle = pickle.load(f)
            
        assert g_pickle.number_of_nodes() == sample_graph.number_of_nodes()
        
        # Verify mappings file
        with open(output_files["mappings"], "r") as f:
            mappings = json.load(f)
            
        assert "node_types" in mappings
        assert "edge_types" in mappings
        assert "statistics" in mappings
    
    def test_normalize_id(self, test_output_dir):
        """Test ID normalization"""
        builder = KnowledgeGraphBuilder(output_dir=str(test_output_dir))
        
        assert builder._normalize_id("Test String") == "test_string"
        assert builder._normalize_id("Test-String") == "test_string"
        assert builder._normalize_id("Test (String)") == "test_string"
        assert builder._normalize_id("Test, String.") == "test_string"
        assert builder._normalize_id("") == ""
        assert builder._normalize_id(None) == ""