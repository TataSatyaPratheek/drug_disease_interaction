# tests/unit/graph/test_conversion.py
import pytest
import os
import pickle
import networkx as nx
import torch
from pathlib import Path
from src.ddi.graph.builder import KnowledgeGraphBuilder

# Skip this test module if PyTorch Geometric is not available
try:
    import torch_geometric
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
class TestGraphConversion:
    """Test graph conversion functions in KnowledgeGraphBuilder"""
    
    def test_convert_to_pyg(self, sample_graph, test_output_dir):
        """Test conversion from NetworkX to PyTorch Geometric"""
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
        
        # Convert to PyG
        pyg_graph = builder._convert_to_pyg()
        
        # Check PyG graph
        assert pyg_graph is not None
        assert isinstance(pyg_graph, Data)
        assert pyg_graph.num_nodes == sample_graph.number_of_nodes()
        assert pyg_graph.num_edges == sample_graph.number_of_edges()
        
        # Check node features
        assert pyg_graph.x is not None
        assert isinstance(pyg_graph.x, torch.Tensor)
        assert pyg_graph.x.shape[0] == pyg_graph.num_nodes
        
        # Check edge features
        assert pyg_graph.edge_attr is not None
        assert isinstance(pyg_graph.edge_attr, torch.Tensor)
        assert pyg_graph.edge_attr.shape[0] == pyg_graph.num_edges
        
        # Check mappings
        assert hasattr(pyg_graph, "node_type_to_id")
        assert hasattr(pyg_graph, "edge_type_to_id")
        assert hasattr(pyg_graph, "id_to_node_type")
        assert hasattr(pyg_graph, "id_to_edge_type")
        assert hasattr(pyg_graph, "node_to_idx")
        assert hasattr(pyg_graph, "idx_to_node")
        
        # Test mappings correctness
        assert len(pyg_graph.node_type_to_id) == len(builder.node_types)
        assert len(pyg_graph.edge_type_to_id) == len(builder.edge_types)
    
    def test_save_graph_with_pyg(self, sample_graph, test_output_dir):
        """Test saving graph with PyG format"""
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
        
        # Save graph with PyG format
        output_files = builder.save_graph(formats=["pyg"])
        
        # Check that PyG file was created
        assert "pyg" in output_files
        assert os.path.exists(output_files["pyg"])
        
        # Verify PyG file using pickle directly
        with open(output_files["pyg"], "rb") as f:
            pyg_graph = pickle.load(f)
        assert pyg_graph.num_nodes == sample_graph.number_of_nodes()
        assert pyg_graph.num_edges == sample_graph.number_of_edges()
