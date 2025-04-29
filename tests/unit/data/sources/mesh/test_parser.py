# tests/unit/data/sources/mesh/test_parser.py
import pytest
import os
import pickle
import json
from pathlib import Path
from src.ddi.data.sources.mesh.parser import MeSHParser

class TestMeSHParser:
    """Test the MeSHParser class"""
    
    def test_init(self, test_data_dir):
        """Test initialization of MeSHParser"""
        mesh_dir = test_data_dir / "mesh"
        parser = MeSHParser(str(mesh_dir))
        
        assert parser.mesh_dir == str(mesh_dir)
        assert parser.descriptors == {}
        assert parser.qualifiers == {}
        assert parser.disease_hierarchy == {}
        assert parser.term_to_id == {}
        assert parser.disease_categories == {"C"}
    
    def test_parse_descriptor_file(self, create_test_mesh_files):
        """Test parsing MeSH descriptor file"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir))
        
        descriptor_file = mesh_dir / "desc2025.xml"
        result = parser.parse_descriptor_file(str(descriptor_file))
        
        assert len(result) == 1  # Only one disease descriptor
        assert "D000001" in result
        
        descriptor = result["D000001"]
        assert descriptor["id"] == "D000001"
        assert descriptor["name"] == "Test Disease 1"
        assert descriptor["tree_numbers"] == ["C01.123"]
        assert descriptor["description"] == "A test disease"
        assert set(descriptor["synonyms"]) == {"Disease 1", "Condition 1"}
        assert descriptor["allowed_qualifiers"] == ["Q000139"]
        assert descriptor["is_disease"] is True
    
    def test_parse_qualifier_file(self, create_test_mesh_files):
        """Test parsing MeSH qualifier file"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir))
        
        qualifier_file = mesh_dir / "qual2025.xml"
        result = parser.parse_qualifier_file(str(qualifier_file))
        
        assert len(result) == 1
        assert "Q000139" in result
        
        qualifier = result["Q000139"]
        assert qualifier["id"] == "Q000139"
        assert qualifier["name"] == "drug therapy"
        assert qualifier["tree_numbers"] == ["E02.319"]
        assert "administration of drugs" in qualifier["description"]
    
    def test_parse_latest_mesh(self, create_test_mesh_files):
        """Test parsing latest MeSH files"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir))
        
        result = parser.parse_latest_mesh()
        
        assert "descriptors" in result
        assert "qualifiers" in result
        assert "disease_hierarchy" in result
        assert "term_to_id" in result
        assert "version" in result
        
        assert "D000001" in result["descriptors"]
        assert "Q000139" in result["qualifiers"]
        assert "test disease 1" in result["term_to_id"]
        assert result["term_to_id"]["test disease 1"] == "D000001"
    
    def test_extract_disease_taxonomy(self, create_test_mesh_files):
        """Test extracting disease taxonomy"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir))
        
        parser.parse_latest_mesh()
        taxonomy = parser.extract_disease_taxonomy()
        
        assert "D000001" in taxonomy
        
        disease = taxonomy["D000001"]
        assert disease["id"] == "D000001"
        assert disease["name"] == "Test Disease 1"
        assert disease["tree_numbers"] == ["C01.123"]
        assert set(disease["synonyms"]) == {"Disease 1", "Condition 1"}
    
    def test_save_mesh_data(self, create_test_mesh_files, test_output_dir):
        """Test saving MeSH data"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir), output_dir=str(test_output_dir))
        
        parser.parse_latest_mesh()
        
        # Test saving as pickle
        pickle_path = parser.save_mesh_data(format="pickle")
        assert os.path.exists(pickle_path)
        
        # Verify the saved file
        with open(pickle_path, "rb") as f:
            saved_data = pickle.load(f)
            
        assert "descriptors" in saved_data
        assert "qualifiers" in saved_data
        assert "disease_hierarchy" in saved_data
        assert "term_to_id" in saved_data
        
        # Test saving as JSON
        json_path = parser.save_mesh_data(format="json")
        assert os.path.exists(json_path)
    
    def test_save_disease_taxonomy(self, create_test_mesh_files, test_output_dir):
        """Test saving disease taxonomy"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir), output_dir=str(test_output_dir))
        
        parser.parse_latest_mesh()
        
        # Test saving as pickle
        pickle_path = parser.save_disease_taxonomy(format="pickle")
        assert os.path.exists(pickle_path)
        
        # Verify the saved file
        with open(pickle_path, "rb") as f:
            taxonomy = pickle.load(f)
            
        assert "D000001" in taxonomy
        
        # Test saving as JSON
        json_path = parser.save_disease_taxonomy(format="json")
        assert os.path.exists(json_path)
    
    def test_handle_malformed_xml(self, tmpdir, test_output_dir):
        """Test handling of malformed XML"""
        mesh_dir = tmpdir / "mesh"
        mesh_dir.mkdir()
        
        # Create malformed XML files
        with open(mesh_dir / "desc2025.xml", "w") as f:
            f.write("<DescriptorRecordSet><DescriptorRecord><DescriptorUI>D000001</DescriptorUI>")
        
        with open(mesh_dir / "qual2025.xml", "w") as f:
            f.write("<QualifierRecordSet><QualifierRecord><QualifierUI>Q000139</QualifierUI>")
        
        parser = MeSHParser(str(mesh_dir), output_dir=str(test_output_dir))
        
        # Parsing should handle errors gracefully
        result = parser.parse_latest_mesh()
        
        assert "descriptors" in result
        assert "qualifiers" in result
        assert len(result["descriptors"]) == 0
        assert len(result["qualifiers"]) == 0
    
    def test_process_disease_hierarchy(self, create_test_mesh_files):
        """Test processing disease hierarchy"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir))
        
        parser.parse_latest_mesh()
        
        # Check that disease hierarchy was processed
        assert "C01" in parser.disease_hierarchy
        
        hierarchy_node = parser.disease_hierarchy["C01"]
        assert hierarchy_node["tree_number"] == "C01"
        assert hierarchy_node["name"] == "Test Category"
        assert hierarchy_node["descriptor_id"] == "D000003"


# tests/unit/data/sources/opentargets/test_parser.py
import pytest
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from src.ddi.data.sources.opentargets.parser import OpenTargetsParser

class MockParquetFile:
    """Mock ParquetFile for testing"""
    
    def __init__(self, schema, data):
        self.schema = schema
        self.data = data
    
    def read_table(self):
        return self.data
    
    def to_pandas(self):
        return pd.DataFrame(self.data)

class TestOpenTargetsParser:
    """Test the OpenTargetsParser class"""
    
    def test_init(self, test_data_dir):
        """Test initialization of OpenTargetsParser"""
        ot_dir = test_data_dir / "opentargets"
        parser = OpenTargetsParser(str(ot_dir))
        
        assert parser.data_dir == str(ot_dir)
        assert parser.drug_target_associations == []
        assert parser.target_disease_associations == []
        assert parser.drug_disease_associations == []
        assert parser.targets == {}
        assert parser.diseases == {}
        assert parser.drugs == {}
    
    def test_process_target_disease_associations(self, monkeypatch):
        """Test processing target-disease associations"""
        # Create a mock DataFrame with target-disease associations
        df = pd.DataFrame({
            'targetId': ['ENSG00000123456', 'ENSG00000789012'],
            'diseaseId': ['EFO:0000001', 'EFO:0000002'],
            'score': [0.7, 0.8],
            'targetName': ['Target 1', 'Target 2'],
            'diseaseName': ['Disease 1', 'Disease 2']
        })
        
        parser = OpenTargetsParser("dummy_dir")
        
        # Call the internal method directly
        parser._process_target_disease_associations(df)
        
        # Check that associations were extracted
        assert len(parser.target_disease_associations) == 2
        
        assoc1 = parser.target_disease_associations[0]
        assert assoc1['target_id'] == 'ENSG00000123456'
        assert assoc1['disease_id'] == 'EFO:0000001'
        assert assoc1['score'] == 0.7
        
        # Check that entities were extracted
        assert 'ENSG00000123456' in parser.targets
        assert parser.targets['ENSG00000123456']['name'] == 'Target 1'
        
        assert 'EFO:0000001' in parser.diseases
        assert parser.diseases['EFO:0000001']['name'] == 'Disease 1'
    
    def test_process_drug_target_associations(self, monkeypatch):
        """Test processing drug-target associations"""
        # Create a mock DataFrame with drug-target associations
        df = pd.DataFrame({
            'drugId': ['CHEMBL1234', 'CHEMBL5678'],
            'targetId': ['ENSG00000123456', 'ENSG00000789012'],
            'score': [0.8, 0.9],
            'drugName': ['Drug 1', 'Drug 2'],
            'targetName': ['Target 1', 'Target 2'],
            'mechanism': ['inhibitor', 'activator']
        })
        
        parser = OpenTargetsParser("dummy_dir")
        
        # Call the internal method directly
        parser._process_drug_target_associations(df)
        
        # Check that associations were extracted
        assert len(parser.drug_target_associations) == 2
        
        assoc1 = parser.drug_target_associations[0]
        assert assoc1['drug_id'] == 'CHEMBL1234'
        assert assoc1['target_id'] == 'ENSG00000123456'
        assert assoc1['score'] == 0.8
        assert assoc1['mechanism'] == 'inhibitor'
        
        # Check that entities were extracted
        assert 'CHEMBL1234' in parser.drugs
        assert parser.drugs['CHEMBL1234']['name'] == 'Drug 1'
        
        assert 'ENSG00000123456' in parser.targets
        assert parser.targets['ENSG00000123456']['name'] == 'Target 1'
    
    def test_process_drug_disease_associations(self, monkeypatch):
        """Test processing drug-disease associations"""
        # Create a mock DataFrame with drug-disease associations
        df = pd.DataFrame({
            'drugId': ['CHEMBL1234', 'CHEMBL5678'],
            'diseaseId': ['EFO:0000001', 'EFO:0000002'],
            'score': [0.9, 0.7],
            'drugName': ['Drug 1', 'Drug 2'],
            'diseaseName': ['Disease 1', 'Disease 2'],
            'clinicalPhase': [3, 2]
        })
        
        parser = OpenTargetsParser("dummy_dir")
        
        # Call the internal method directly
        parser._process_drug_disease_associations(df)
        
        # Check that associations were extracted
        assert len(parser.drug_disease_associations) == 2
        
        assoc1 = parser.drug_disease_associations[0]
        assert assoc1['drug_id'] == 'CHEMBL1234'
        assert assoc1['disease_id'] == 'EFO:0000001'
        assert assoc1['score'] == 0.9
        assert assoc1['clinical_phase'] == 3
        
        # Check that entities were extracted
        assert 'CHEMBL1234' in parser.drugs
        assert parser.drugs['CHEMBL1234']['name'] == 'Drug 1'
        
        assert 'EFO:0000001' in parser.diseases
        assert parser.diseases['EFO:0000001']['name'] == 'Disease 1'
    
    def test_extract_drug_disease_indications(self):
        """Test extracting drug-disease indications"""
        parser = OpenTargetsParser("dummy_dir")
        
        # Add some drug-disease associations
        parser.drug_disease_associations = [
            {
                'drug_id': 'CHEMBL1234',
                'disease_id': 'EFO:0000001',
                'score': 0.9,
                'clinical_phase': 3
            },
            {
                'drug_id': 'CHEMBL5678',
                'disease_id': 'EFO:0000002',
                'score': 0.4,  # Low score should be filtered out
                'clinical_phase': 2
            }
        ]
        
        # Add entity information
        parser.drugs = {
            'CHEMBL1234': {'id': 'CHEMBL1234', 'name': 'Drug 1'},
            'CHEMBL5678': {'id': 'CHEMBL5678', 'name': 'Drug 2'}
        }
        
        parser.diseases = {
            'EFO:0000001': {'id': 'EFO:0000001', 'name': 'Disease 1'},
            'EFO:0000002': {'id': 'EFO:0000002', 'name': 'Disease 2'}
        }
        
        # Extract indications
        indications = parser.extract_drug_disease_indications()
        
        # Only the high-scoring association should be included
        assert len(indications) == 1
        
        indication = indications[0]
        assert indication['drug_id'] == 'CHEMBL1234'
        assert indication['disease_id'] == 'EFO:0000001'
        assert indication['drug_name'] == 'Drug 1'
        assert indication['disease_name'] == 'Disease 1'
        assert indication['score'] == 0.9
        assert indication['clinical_phase'] == 3
        assert indication['source'] == 'opentargets'
    
    def test_save_opentargets_data(self, test_output_dir):
        """Test saving OpenTargets data"""
        parser = OpenTargetsParser("dummy_dir", output_dir=str(test_output_dir))
        
        # Add some test data
        parser.target_disease_associations = [
            {'target_id': 'ENSG00000123456', 'disease_id': 'EFO:0000001', 'score': 0.7}
        ]
        
        parser.targets = {'ENSG00000123456': {'id': 'ENSG00000123456', 'name': 'Target 1'}}
        parser.diseases = {'EFO:0000001': {'id': 'EFO:0000001', 'name': 'Disease 1'}}
        
        # Save data
        output_files = parser.save_opentargets_data(format="pickle")
        
        # Check that files were created
        assert os.path.exists(output_files['associations'])
        assert os.path.exists(output_files['entities'])
        
        # Verify association data
        with open(output_files['associations'], 'rb') as f:
            assoc_data = pickle.load(f)
            
        assert 'target_disease_associations' in assoc_data
        assert len(assoc_data['target_disease_associations']) == 1
        
        # Verify entity data
        with open(output_files['entities'], 'rb') as f:
            entity_data = pickle.load(f)
            
        assert 'targets' in entity_data
        assert 'diseases' in entity_data
        assert 'ENSG00000123456' in entity_data['targets']
        assert 'EFO:0000001' in entity_data['diseases']
    
    def test_save_indications(self, test_output_dir):
        """Test saving drug-disease indications"""
        parser = OpenTargetsParser("dummy_dir", output_dir=str(test_output_dir))
        
        # Add some drug-disease associations
        parser.drug_disease_associations = [
            {'drug_id': 'CHEMBL1234', 'disease_id': 'EFO:0000001', 'score': 0.9}
        ]
        
        # Add entity information
        parser.drugs = {'CHEMBL1234': {'id': 'CHEMBL1234', 'name': 'Drug 1'}}
        parser.diseases = {'EFO:0000001': {'id': 'EFO:0000001', 'name': 'Disease 1'}}
        
        # Save indications
        output_path = parser.save_indications(format="pickle")
        
        # Check that file was created
        assert os.path.exists(output_path)
        
        # Verify indication data
        with open(output_path, 'rb') as f:
            indications = pickle.load(f)
            
        assert len(indications) == 1
        assert indications[0]['drug_id'] == 'CHEMBL1234'
        assert indications[0]['disease_id'] == 'EFO:0000001'