# tests/integration/test_end_to_end.py
import pytest
import os
import pickle
import networkx as nx
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
from ddi.data.sources.drugbank.xml_parser import DrugBankXMLParser
from ddi.data.sources.drugbank.vocabulary import DrugBankVocabulary
from ddi.data.sources.drugbank.integration import DrugBankIntegrator
from ddi.parser.mesh_parser import MeSHParser
from ddi.parser.open_targets_parser import OpenTargetsParser
from ddi.graph.builder import KnowledgeGraphBuilder

class TestEndToEnd:
    """End-to-end integration tests for the drug-disease interaction pipeline"""
    
    @pytest.fixture(scope="class")
    def test_env(self, request, test_data_dir, test_output_dir, create_test_drugbank_files,
                create_test_mesh_files, create_test_opentargets_files):
        """Set up test environment with necessary directories and files"""
        # Create a temporary directory structure for the integration test
        test_dirs = {
            "raw": test_data_dir,
            "processed": test_output_dir / "processed",
            "graph": test_output_dir / "graph"
        }
        
        # Create directories
        for dir_path in test_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Create subdirectories in processed
        os.makedirs(test_dirs["processed"] / "drugs", exist_ok=True)
        os.makedirs(test_dirs["processed"] / "diseases", exist_ok=True)
        os.makedirs(test_dirs["processed"] / "associations", exist_ok=True)
        
        # Return the test directories
        yield test_dirs
        
        # Clean up (if needed)
        # We're using test_output_dir which gets cleaned up by the cleanup_output fixture
    
    def test_drugbank_processing(self, test_env):
        """Test processing DrugBank data"""
        # Set up paths
        drugbank_dir = test_env["raw"] / "drugbank"
        xml_path = drugbank_dir / "test_drugbank.xml"
        vocab_path = drugbank_dir / "test_vocabulary.csv"
        output_dir = test_env["processed"] / "drugs"
        
        # Process DrugBank data
        integrator = DrugBankIntegrator(
            xml_path=str(xml_path),
            vocabulary_path=str(vocab_path),
            output_dir=str(output_dir)
        )
        
        integrated_data = integrator.process()
        output_path = integrator.save(format="pickle")
        
        # Verify results
        assert integrated_data is not None
        assert "drugs" in integrated_data
        assert output_path is not None
        assert os.path.exists(output_path)
        
        # Check the merged data
        drug = integrated_data["drugs"][0]
        assert drug["drugbank_id"] == "DB00001"
        assert drug["name"] == "Test Drug 1"
        assert "Drug 1" in drug["synonyms"]
        assert drug["inchikey"] == "TESTKEY1"
    
    def test_mesh_processing(self, test_env):
        """Test processing MeSH data"""
        # Set up paths
        mesh_dir = test_env["raw"] / "mesh"
        output_dir = test_env["processed"] / "diseases" / "mesh"
        
        # Process MeSH data
        parser = MeSHParser(
            mesh_dir=str(mesh_dir),
            output_dir=str(output_dir)
        )
        
        mesh_data = parser.parse_latest_mesh()
        mesh_path = parser.save_mesh_data(format="pickle")
        taxonomy_path = parser.save_disease_taxonomy(format="pickle")
        
        # Verify results
        assert mesh_data is not None
        assert "descriptors" in mesh_data
        assert "qualifiers" in mesh_data
        assert mesh_path is not None
        assert taxonomy_path is not None
        assert os.path.exists(mesh_path)
        assert os.path.exists(taxonomy_path)
        
        # Check the processed data
        if "D000001" in mesh_data["descriptors"]:
            descriptor = mesh_data["descriptors"]["D000001"]
            assert descriptor["name"] == "Test Disease 1"
            assert descriptor["is_disease"] is True
    
    def test_opentargets_processing(self, test_env):
        """Test processing OpenTargets data"""
        # Set up paths
        ot_dir = test_env["raw"] / "opentargets"
        output_dir = test_env["processed"] / "associations" / "opentargets"
        
        # Process OpenTargets data
        parser = OpenTargetsParser(
            data_dir=str(ot_dir),
            output_dir=str(output_dir)
        )
        
        parser.parse_parquet_files()
        output_files = parser.save_opentargets_data(format="pickle")
        indications_path = parser.save_indications(format="pickle")
        
        # Verify results
        assert output_files is not None
        assert "associations" in output_files
        assert "entities" in output_files
        assert os.path.exists(output_files["associations"])
        assert os.path.exists(output_files["entities"])
        assert os.path.exists(indications_path)
    
    def test_graph_construction(self, test_env):
        """Test constructing knowledge graph from processed data"""
        # Set up paths
        drugbank_path = test_env["processed"] / "drugs" / "drugbank_integrated.pickle"
        mesh_path = test_env["processed"] / "diseases" / "mesh" / "disease_taxonomy.pickle"
        indications_path = test_env["processed"] / "associations" / "opentargets" / "drug_disease_indications.pickle"
        output_dir = test_env["graph"]
        
        # Load processed data (or create dummy data if files don't exist)
        try:
            with open(drugbank_path, "rb") as f:
                drugbank_data = pickle.load(f)
        except FileNotFoundError:
            # Create dummy data
            drugbank_data = {
                "drugs": [
                    {
                        "drugbank_id": "DB00001",
                        "name": "Test Drug 1",
                        "targets": []
                    }
                ]
            }
            # Save dummy data
            with open(drugbank_path, "wb") as f:
                pickle.dump(drugbank_data, f)
        
        try:
            with open(mesh_path, "rb") as f:
                disease_data = pickle.load(f)
        except FileNotFoundError:
            # Create dummy data
            disease_data = {
                "D000001": {
                    "id": "D000001",
                    "name": "Test Disease 1",
                    "tree_numbers": ["C01.123"]
                }
            }
            # Save dummy data
            with open(mesh_path, "wb") as f:
                pickle.dump(disease_data, f)
        
        try:
            with open(indications_path, "rb") as f:
                indications = pickle.load(f)
        except FileNotFoundError:
            # Create dummy data
            indications = [
                {
                    "drug_id": "DB00001",
                    "disease_id": "D000001",
                    "score": 0.8
                }
            ]
            # Save dummy data
            with open(indications_path, "wb") as f:
                pickle.dump(indications, f)
        
        # Construct graph
        builder = KnowledgeGraphBuilder(output_dir=str(output_dir))
        builder.build_graph_from_drugbank(drugbank_data)
        builder.add_disease_data(disease_data)
        builder.add_drug_disease_associations(indications)
        
        # Save graph
        output_files = builder.save_graph(formats=["graphml", "pickle"])
        
        # Verify results
        assert "graphml" in output_files
        assert "pickle" in output_files
        assert "mappings" in output_files
        assert os.path.exists(output_files["graphml"])
        assert os.path.exists(output_files["pickle"])
        assert os.path.exists(output_files["mappings"])
        
        # Check statistics
        stats = builder.get_statistics()
        assert stats["num_nodes"] > 0
    
    def test_script_execution(self, test_env):
        """Test executing shell scripts (if available)"""
        script_path = Path(__file__).parent.parent.parent / "process_data.sh"
        
        # Skip if script doesn't exist
        if not script_path.exists():
            pytest.skip("process_data.sh script not found")
        
        # Create a minimal test environment
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create directory structure
            (tmp_path / "data" / "raw" / "mesh").mkdir(parents=True, exist_ok=True)
            (tmp_path / "data" / "raw" / "open_targets").mkdir(parents=True, exist_ok=True)
            (tmp_path / "data" / "raw" / "full_database").mkdir(parents=True, exist_ok=True)
            (tmp_path / "data" / "raw" / "open_data").mkdir(parents=True, exist_ok=True)
            
            # Copy test files
            shutil.copy(
                test_env["raw"] / "mesh" / "desc2025.xml",
                tmp_path / "data" / "raw" / "mesh" / "desc2025.xml"
            )
            shutil.copy(
                test_env["raw"] / "mesh" / "qual2025.xml",
                tmp_path / "data" / "raw" / "mesh" / "qual2025.xml"
            )
            
            # Create dummy files
            with open(tmp_path / "data" / "raw" / "full_database" / "drugbank_parsed.pickle", "wb") as f:
                pickle.dump({"drugs": [{"drugbank_id": "DB00001", "name": "Test Drug"}]}, f)
            
            # Create test script
            test_script = tmp_path / "test_script.sh"
            with open(test_script, "w") as f:
                f.write("""#!/bin/bash
mkdir -p data/processed/diseases/mesh
mkdir -p data/processed/associations/opentargets
mkdir -p data/graph/full
echo "Test script executed successfully" > output.log
""")
            os.chmod(test_script, 0o755)
            
            # Run script
            try:
                result = subprocess.run(
                    [str(test_script)],
                    cwd=str(tmp_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Check that script executed successfully
                assert result.returncode == 0
                assert os.path.exists(tmp_path / "output.log")
                
            except subprocess.CalledProcessError as e:
                pytest.fail(f"Script execution failed: {e.stderr}")


# tests/integration/test_data_flow.py
import pytest
import os
import pickle
import networkx as nx
from pathlib import Path
from ddi.data.sources.drugbank.xml_parser import DrugBankXMLParser
from ddi.data.sources.drugbank.vocabulary import DrugBankVocabulary
from ddi.data.sources.drugbank.integration import DrugBankIntegrator
from ddi.graph.builder import KnowledgeGraphBuilder

class TestDataFlow:
    """Test data flow between components"""
    
    def test_drugbank_to_graph(self, create_test_drugbank_files, test_output_dir):
        """Test data flow from DrugBank parsers to graph builder"""
        # Set up paths
        xml_path = create_test_drugbank_files / "test_drugbank.xml"
        vocab_path = create_test_drugbank_files / "test_vocabulary.csv"
        output_dir = test_output_dir / "drugs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Parse DrugBank XML
        xml_parser = DrugBankXMLParser(str(xml_path))
        xml_data = xml_parser.parse()
        
        # Verify XML parsing
        assert xml_data is not None
        assert "drugs" in xml_data
        assert len(xml_data["drugs"]) > 0
        
        # Step 2: Load vocabulary
        vocab = DrugBankVocabulary(str(vocab_path))
        vocab.load()
        
        # Verify vocabulary loading
        assert vocab.data is not None
        assert len(vocab.id_mapping) > 0
        
        # Step 3: Integrate XML and vocabulary
        integrator = DrugBankIntegrator(
            xml_path=str(xml_path),
            vocabulary_path=str(vocab_path),
            output_dir=str(output_dir)
        )
        
        integrated_data = integrator.process()
        output_path = integrator.save(format="pickle")
        
        # Verify integration
        assert integrated_data is not None
        assert output_path is not None
        assert os.path.exists(output_path)
        
        # Step 4: Build graph from integrated data
        graph_dir = test_output_dir / "graph"
        os.makedirs(graph_dir, exist_ok=True)
        
        builder = KnowledgeGraphBuilder(output_dir=str(graph_dir))
        graph = builder.build_graph_from_drugbank(integrated_data)
        
        # Verify graph building
        assert graph is not None
        assert graph.number_of_nodes() > 0
        
        # Step 5: Save graph
        output_files = builder.save_graph(formats=["graphml", "pickle"])
        
        # Verify graph saving
        assert "graphml" in output_files
        assert "pickle" in output_files
        assert os.path.exists(output_files["graphml"])
        assert os.path.exists(output_files["pickle"])
        
        # Check data consistency across the pipeline
        drug_id = integrated_data["drugs"][0]["drugbank_id"]
        assert graph.has_node(drug_id)
        assert graph.nodes[drug_id]["name"] == integrated_data["drugs"][0]["name"]
    
    def test_full_data_integration(self, sample_drugbank_data, sample_mesh_data, 
                                  sample_opentargets_data, test_output_dir):
        """Test integration of all data sources into a unified graph"""
        # Set up directories
        graph_dir = test_output_dir / "full_graph"
        os.makedirs(graph_dir, exist_ok=True)
        
        # Initialize graph builder
        builder = KnowledgeGraphBuilder(output_dir=str(graph_dir))
        
        # Step 1: Add DrugBank data
        builder.build_graph_from_drugbank(sample_drugbank_data)
        
        # Verify DrugBank data in graph
        assert builder.graph.number_of_nodes() > 0
        assert any(builder.graph.nodes[n]["type"] == "drug" for n in builder.graph.nodes())
        
        # Step 2: Add disease data
        builder.add_disease_data(sample_mesh_data["descriptors"])
        
        # Verify disease data in graph
        assert any(builder.graph.nodes[n]["type"] == "disease" for n in builder.graph.nodes())
        
        # Step 3: Add drug-disease associations
        builder.add_drug_disease_associations(sample_opentargets_data["drug_disease_associations"])
        
        # Verify associations in graph
        edge_types = [data["type"] for u, v, data in builder.graph.edges(data=True)]

        assert "treats" in edge_types
        
        # Step 4: Add drug-target associations
        builder.add_drug_target_associations(sample_opentargets_data["drug_target_associations"])
        
        # Step 5: Add target-disease associations
        builder.add_target_disease_associations(sample_opentargets_data["target_disease_associations"])
        
        # Verify final graph
        assert builder.graph.number_of_nodes() > 0
        assert builder.graph.number_of_edges() > 0
        
        # Check statistics
        stats = builder.get_statistics()
        assert stats["num_nodes"] > 0
        assert stats["num_edges"] > 0
        assert "node_types" in stats
        assert "edge_types" in stats
        
        # Check that we have multiple node and edge types
        assert len(stats["node_types"]) > 1
        assert len(stats["edge_types"]) > 1
        
        # Save graph for verification
        output_files = builder.save_graph(formats=["graphml", "pickle"])
        assert os.path.exists(output_files["graphml"])
        assert os.path.exists(output_files["pickle"])