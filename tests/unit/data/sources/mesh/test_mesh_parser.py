# /Users/vi/Documents/drug_disease_interaction/tests/unit/data/sources/mesh/test_mesh_parser.py
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

        assert len(result) == 2
        assert "D000001" in result
        assert "D000003" in result

        descriptor = result["D000001"]
        assert descriptor["id"] == "D000001"
        assert descriptor["name"] == "Test Disease 1"
        assert descriptor["tree_numbers"] == ["C01.123"]
        assert descriptor["description"] == "A test disease"
        assert set(descriptor["synonyms"]) == {"Disease 1", "Condition 1"}
        assert descriptor["allowed_qualifiers"] == ["Q000139"]
        assert descriptor["is_disease"] is True

        category_descriptor = result["D000003"]
        assert category_descriptor["id"] == "D000003"
        assert category_descriptor["name"] == "Test Category"
        assert category_descriptor["tree_numbers"] == ["C01"]
        assert category_descriptor["is_disease"] is True

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

        pickle_path = parser.save_mesh_data(format="pickle")
        assert os.path.exists(pickle_path)
        with open(pickle_path, "rb") as f:
            saved_data = pickle.load(f)
        assert "descriptors" in saved_data

        json_path = parser.save_mesh_data(format="json")
        assert os.path.exists(json_path)

    def test_save_disease_taxonomy(self, create_test_mesh_files, test_output_dir):
        """Test saving disease taxonomy"""
        mesh_dir = create_test_mesh_files
        parser = MeSHParser(str(mesh_dir), output_dir=str(test_output_dir))

        parser.parse_latest_mesh()

        pickle_path = parser.save_disease_taxonomy(format="pickle")
        assert os.path.exists(pickle_path)
        with open(pickle_path, "rb") as f:
            taxonomy = pickle.load(f)
        assert "D000001" in taxonomy

        json_path = parser.save_disease_taxonomy(format="json")
        assert os.path.exists(json_path)

    def test_handle_malformed_xml(self, tmpdir, test_output_dir):
        """Test handling of malformed XML"""
        mesh_dir = tmpdir / "mesh"
        mesh_dir.mkdir()

        with open(mesh_dir / "desc2025.xml", "w") as f:
            f.write("<DescriptorRecordSet><DescriptorRecord><DescriptorUI>D000001</DescriptorUI>")
        with open(mesh_dir / "qual2025.xml", "w") as f:
            f.write("<QualifierRecordSet><QualifierRecord><QualifierUI>Q000139</QualifierUI>")

        parser = MeSHParser(str(mesh_dir), output_dir=str(test_output_dir))
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

        assert "C01" in parser.disease_hierarchy
        hierarchy_node = parser.disease_hierarchy["C01"]
        assert hierarchy_node["tree_number"] == "C01"
        assert hierarchy_node["name"] == "Test Category"
        assert hierarchy_node["descriptor_id"] == "D000003"

# --- REMOVED TestOpenTargetsParser from this file ---
