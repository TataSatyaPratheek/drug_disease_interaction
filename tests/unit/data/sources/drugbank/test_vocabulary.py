# tests/unit/data/sources/drugbank/test_vocabulary.py
import pytest
import pandas as pd
import numpy as np
from src.ddi.data.sources.drugbank.vocabulary import DrugBankVocabulary

class TestDrugBankVocabulary:
    """Test the DrugBankVocabulary class"""
    
    def test_init(self, test_data_dir):
        """Test initialization of DrugBankVocabulary"""
        test_csv = test_data_dir / "drugbank" / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        
        assert vocab.csv_path == str(test_csv)
        assert vocab.data is None
        assert vocab.id_mapping == {}
        assert vocab.cas_to_id == {}
        assert vocab.name_to_id == {}
        assert vocab.inchi_to_id == {}
        assert vocab.unii_to_id == {}
    
    def test_load(self, create_test_drugbank_files):
        """Test loading vocabulary data"""
        test_csv = create_test_drugbank_files / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        df = vocab.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['drugbank_id', 'name', 'cas_number', 'unii', 'synonyms', 'inchikey']
        assert df.iloc[0]['drugbank_id'] == 'DB00001'
        assert df.iloc[0]['name'] == 'Test Drug 1'
    
    def test_build_mappings(self, create_test_drugbank_files):
        """Test building mappings from vocabulary data"""
        test_csv = create_test_drugbank_files / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        vocab.load()
        
        # Check ID mapping
        assert 'DB00001' in vocab.id_mapping
        assert 'DB00002' in vocab.id_mapping
        
        # Check CAS mapping
        assert '12345-67-8' in vocab.cas_to_id
        assert vocab.cas_to_id['12345-67-8'] == 'DB00001'
        
        # Check name mapping (case-insensitive)
        assert 'test drug 1' in vocab.name_to_id
        assert vocab.name_to_id['test drug 1'] == 'DB00001'
        
        # Check synonym mapping
        assert 'drug 1' in vocab.name_to_id
        assert 'medicine 1' in vocab.name_to_id
        assert vocab.name_to_id['drug 1'] == 'DB00001'
        
        # Check InChIKey mapping
        assert 'TESTKEY1' in vocab.inchi_to_id
        assert vocab.inchi_to_id['TESTKEY1'] == 'DB00001'
        
        # Check UNII mapping
        assert 'ABC123' in vocab.unii_to_id
        assert vocab.unii_to_id['ABC123'] == 'DB00001'
    
    def test_get_drug_by_id(self, create_test_drugbank_files):
        """Test retrieving drug by ID"""
        test_csv = create_test_drugbank_files / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        vocab.load()
        
        # Test getting existing drug
        drug = vocab.get_drug_by_id('DB00001')
        assert drug is not None
        assert drug['drugbank_id'] == 'DB00001'
        assert drug['name'] == 'Test Drug 1'
        assert drug['cas_number'] == '12345-67-8'
        assert 'Drug 1' in drug['synonyms']
        assert 'Medicine 1' in drug['synonyms']
        
        # Test getting non-existent drug
        drug = vocab.get_drug_by_id('DB99999')
        assert drug is None
    
    def test_get_id_by_name(self, create_test_drugbank_files):
        """Test retrieving drug ID by name"""
        test_csv = create_test_drugbank_files / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        vocab.load()
        
        # Test case-insensitive name lookup
        assert vocab.get_id_by_name('Test Drug 1') == 'DB00001'
        assert vocab.get_id_by_name('test drug 1') == 'DB00001'
        
        # Test synonym lookup
        assert vocab.get_id_by_name('Drug 1') == 'DB00001'
        assert vocab.get_id_by_name('Medicine 1') == 'DB00001'
        
        # Test non-existent name
        assert vocab.get_id_by_name('Nonexistent Drug') is None
    
    def test_validate_drug_id(self, create_test_drugbank_files):
        """Test validating drug ID"""
        test_csv = create_test_drugbank_files / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        vocab.load()
        
        assert vocab.validate_drug_id('DB00001') is True
        assert vocab.validate_drug_id('DB00002') is True
        assert vocab.validate_drug_id('DB99999') is False
    
    def test_enrich_drug_data(self, create_test_drugbank_files):
        """Test enriching drug data with vocabulary information"""
        test_csv = create_test_drugbank_files / "test_vocabulary.csv"
        vocab = DrugBankVocabulary(str(test_csv))
        vocab.load()
        
        # Test enriching with partial data
        drug_data = {
            'drugbank_id': 'DB00001',
            'name': 'Test Drug 1',
            'synonyms': ['Old Synonym']
        }
        
        enriched = vocab.enrich_drug_data(drug_data)
        assert enriched['drugbank_id'] == 'DB00001'
        assert enriched['name'] == 'Test Drug 1'
        assert 'Old Synonym' in enriched['synonyms']
        assert 'Drug 1' in enriched['synonyms']
        assert 'Medicine 1' in enriched['synonyms']
        assert enriched['inchikey'] == 'TESTKEY1'
        assert enriched['unii'] == 'ABC123'
    
    def test_handle_invalid_input(self):
        """Test handling of invalid input"""
        # Test with non-existent file
        vocab = DrugBankVocabulary("nonexistent_file.csv")
        df = vocab.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        # Test with invalid drug data
        assert vocab.enrich_drug_data({}) == {}
    
    def test_handle_non_string_values(self, tmpdir):
        """Test handling of non-string values in the CSV"""
        # Create a CSV with a numeric name
        test_csv = tmpdir / "test_numeric.csv"
        
        # CSV with an integer in the name column
        csv_content = """DrugBank ID,Common name,CAS,UNII,Synonyms,Standard InChI Key
DB00001,12345,12345-67-8,ABC123,"Drug 1|Medicine 1",TESTKEY1
DB00002,Test Drug 2,23456-78-9,DEF456,"Drug 2|Medicine 2",TESTKEY2
"""
        
        with open(test_csv, "w") as f:
            f.write(csv_content)
        
        # Test parsing
        vocab = DrugBankVocabulary(str(test_csv))
        df = vocab.load()
        
        # Check that mappings were created properly
        assert '12345' in vocab.name_to_id
        assert vocab.name_to_id['12345'] == 'DB00001'
        
        # Get drug by ID should work
        drug = vocab.get_drug_by_id('DB00001')
        assert drug is not None
        assert drug['name'] == '12345'


# tests/unit/data/sources/drugbank/test_xml_parser.py
import pytest
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from src.ddi.data.sources.drugbank.xml_parser import DrugBankXMLParser

class TestDrugBankXMLParser:
    """Test the DrugBankXMLParser class"""
    
    def test_init(self, create_test_drugbank_files):
        """Test initialization of DrugBankXMLParser"""
        test_xml = create_test_drugbank_files / "test_drugbank.xml"
        parser = DrugBankXMLParser(str(test_xml))
        
        assert parser.xml_file_path == str(test_xml)
        assert parser.ns == {}
    
    @pytest.mark.parametrize("limit", [None, 1])
    def test_parse(self, create_test_drugbank_files, limit):
        """Test parsing DrugBank XML"""
        test_xml = create_test_drugbank_files / "test_drugbank.xml"
        parser = DrugBankXMLParser(str(test_xml))
        
        result = parser.parse(limit=limit)
        
        assert "version" in result
        assert "drugs" in result
        assert isinstance(result["drugs"], list)
        assert len(result["drugs"]) == 1
        
        drug = result["drugs"][0]
        assert drug["drugbank_id"] == "DB00001"
        assert drug["name"] == "Test Drug 1"
        assert drug["description"] == "This is a test drug"
        assert drug["cas_number"] == "12345-67-8"
        assert "approved" in drug["groups"]
        assert "investigational" in drug["groups"]
        
        assert len(drug["categories"]) == 1
        assert drug["categories"][0]["category"] == "Test Category"
        assert drug["categories"][0]["mesh_id"] == "D000001"
        
        assert len(drug["targets"]) == 1
        target = drug["targets"][0]
        assert target["id"] == "BE0000001"
        assert target["name"] == "Test Target"
        assert target["organism"] == "Humans"
        assert target["actions"] == ["inhibitor"]
        assert target["known_action"] == "yes"
        
        assert len(target["polypeptides"]) == 1
        polypeptide = target["polypeptides"][0]
        assert polypeptide["id"] == "PP00001"
        assert polypeptide["name"] == "Test Polypeptide"
        assert polypeptide["gene_name"] == "TEST1"
        
        assert len(polypeptide["external_identifiers"]) == 1
        ext_id = polypeptide["external_identifiers"][0]
        assert ext_id["resource"] == "UniProtKB"
        assert ext_id["identifier"] == "P12345"
    
    def test_handle_invalid_xml(self, tmpdir):
        """Test handling of invalid XML"""
        # Create an invalid XML file
        test_xml = tmpdir / "invalid.xml"
        with open(test_xml, "w") as f:
            f.write("<invalid>This is not valid XML")
        
        parser = DrugBankXMLParser(str(test_xml))
        
        # Parse should handle errors gracefully
        result = parser.parse()
        assert result == {"version": "unknown", "drugs": []}
    
    def test_handle_nonexistent_file(self):
        """Test handling of nonexistent file"""
        parser = DrugBankXMLParser("nonexistent_file.xml")
        
        # Parse should handle errors gracefully
        result = parser.parse()
        assert result == {"version": "unknown", "drugs": []}


# tests/unit/data/sources/drugbank/test_integration.py
import pytest
import os
import pickle
from pathlib import Path
from src.ddi.data.sources.drugbank.integration import DrugBankIntegrator

class TestDrugBankIntegrator:
    """Test the DrugBankIntegrator class"""
    
    def test_init(self, create_test_drugbank_files, test_output_dir):
        """Test initialization of DrugBankIntegrator"""
        xml_path = create_test_drugbank_files / "test_drugbank.xml"
        vocab_path = create_test_drugbank_files / "test_vocabulary.csv"
        
        integrator = DrugBankIntegrator(
            xml_path=str(xml_path), 
            vocabulary_path=str(vocab_path), 
            output_dir=str(test_output_dir)
        )
        
        assert integrator.xml_path == str(xml_path)
        assert integrator.vocabulary_path == str(vocab_path)
        assert integrator.output_dir == str(test_output_dir)
        assert integrator.xml_data is None
        assert integrator.integrated_data is None
    
    def test_process(self, create_test_drugbank_files, test_output_dir):
        """Test processing DrugBank data"""
        xml_path = create_test_drugbank_files / "test_drugbank.xml"
        vocab_path = create_test_drugbank_files / "test_vocabulary.csv"
        
        integrator = DrugBankIntegrator(
            xml_path=str(xml_path), 
            vocabulary_path=str(vocab_path), 
            output_dir=str(test_output_dir)
        )
        
        result = integrator.process()
        
        assert result is not None
        assert "version" in result
        assert "drugs" in result
        assert len(result["drugs"]) == 1
        
        # Check that drug was enriched with vocabulary data
        drug = result["drugs"][0]
        assert drug["drugbank_id"] == "DB00001"
        assert drug["name"] == "Test Drug 1"
        assert "Drug 1" in drug["synonyms"]
        assert "Medicine 1" in drug["synonyms"]
        assert drug["inchikey"] == "TESTKEY1"
        assert drug["unii"] == "ABC123"
    
    def test_save(self, create_test_drugbank_files, test_output_dir):
        """Test saving integrated data"""
        xml_path = create_test_drugbank_files / "test_drugbank.xml"
        vocab_path = create_test_drugbank_files / "test_vocabulary.csv"
        
        integrator = DrugBankIntegrator(
            xml_path=str(xml_path), 
            vocabulary_path=str(vocab_path), 
            output_dir=str(test_output_dir)
        )
        
        integrator.process()
        
        # Test saving as pickle
        pickle_path = integrator.save(format="pickle")
        assert os.path.exists(pickle_path)
        
        # Verify the saved file
        with open(pickle_path, "rb") as f:
            saved_data = pickle.load(f)
            
        assert saved_data["version"] == integrator.integrated_data["version"]
        assert len(saved_data["drugs"]) == len(integrator.integrated_data["drugs"])
        
        # Test saving as JSON
        json_path = integrator.save(format="json")
        assert os.path.exists(json_path)
    
    def test_handle_missing_vocabulary(self, create_test_drugbank_files, test_output_dir):
        """Test handling of missing vocabulary file"""
        xml_path = create_test_drugbank_files / "test_drugbank.xml"
        
        integrator = DrugBankIntegrator(
            xml_path=str(xml_path), 
            vocabulary_path="nonexistent_file.csv", 
            output_dir=str(test_output_dir)
        )
        
        # Process should still work without vocabulary
        result = integrator.process()
        
        assert result is not None
        assert "drugs" in result
        assert len(result["drugs"]) == 1