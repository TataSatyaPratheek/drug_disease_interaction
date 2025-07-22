# /Users/vi/Documents/drug_disease_interaction/tests/unit/data/sources/opentargets/test_opentargets_parser.py
import pytest
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from ddi.parser.open_targets_parser import OpenTargetsParser

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
        df = pd.DataFrame({
            'targetId': ['ENSG00000123456', 'ENSG00000789012'],
            'diseaseId': ['EFO:0000001', 'EFO:0000002'], # Use 6 zeros
            'score': [0.7, 0.8],
            'targetName': ['Target 1', 'Target 2'],
            'diseaseName': ['Disease 1', 'Disease 2']
        })
        parser = OpenTargetsParser("dummy_dir")

        # Call association AND entity extraction methods
        parser._process_target_disease_associations(df)
        parser._extract_target_information(df) # Ensure this is called
        parser._extract_disease_information(df) # Ensure this is called

        assert len(parser.target_disease_associations) == 2
        assoc1 = parser.target_disease_associations[0]
        assert assoc1['target_id'] == 'ENSG00000123456'
        assert assoc1['disease_id'] == 'EFO:0000001' # Check 6 zeros
        assert assoc1['score'] == 0.7

        # Check entities
        assert 'ENSG00000123456' in parser.targets
        assert parser.targets['ENSG00000123456']['name'] == 'Target 1'
        assert 'EFO:0000001' in parser.diseases # Check 6 zeros
        assert parser.diseases['EFO:0000001']['name'] == 'Disease 1'

    def test_process_drug_target_associations(self, monkeypatch):
        """Test processing drug-target associations"""
        df = pd.DataFrame({
            'drugId': ['CHEMBL1234', 'CHEMBL5678'],
            'targetId': ['ENSG00000123456', 'ENSG00000789012'],
            'score': [0.8, 0.9],
            'drugName': ['Drug 1', 'Drug 2'],
            'targetName': ['Target 1', 'Target 2'],
            'mechanism': ['inhibitor', 'activator']
        })
        parser = OpenTargetsParser("dummy_dir")

        # Call association AND entity extraction methods
        parser._process_drug_target_associations(df)
        parser._extract_drug_information(df)
        parser._extract_target_information(df)

        assert len(parser.drug_target_associations) == 2
        assoc1 = parser.drug_target_associations[0]
        assert assoc1['drug_id'] == 'CHEMBL1234'
        assert assoc1['target_id'] == 'ENSG00000123456'
        assert assoc1['score'] == 0.8
        assert assoc1['mechanism'] == 'inhibitor'

        # Check entities
        assert 'CHEMBL1234' in parser.drugs
        assert parser.drugs['CHEMBL1234']['name'] == 'Drug 1'
        assert 'ENSG00000123456' in parser.targets
        assert parser.targets['ENSG00000123456']['name'] == 'Target 1'

    def test_process_drug_disease_associations(self, monkeypatch):
        """Test processing drug-disease associations"""
        df = pd.DataFrame({
            'drugId': ['CHEMBL1234', 'CHEMBL5678'],
            'diseaseId': ['EFO:0000001', 'EFO:0000002'], # Use 6 zeros
            'score': [0.9, 0.7],
            'drugName': ['Drug 1', 'Drug 2'],
            'diseaseName': ['Disease 1', 'Disease 2'],
            'clinicalPhase': [3, 2]
        })
        parser = OpenTargetsParser("dummy_dir")

        # Call association AND entity extraction methods
        parser._process_drug_disease_associations(df)
        parser._extract_drug_information(df)
        parser._extract_disease_information(df)

        assert len(parser.drug_disease_associations) == 2
        assoc1 = parser.drug_disease_associations[0]
        assert assoc1['drug_id'] == 'CHEMBL1234'
        assert assoc1['disease_id'] == 'EFO:0000001' # Check 6 zeros
        assert assoc1['score'] == 0.9
        assert assoc1['clinical_phase'] == 3

        # Check entities
        assert 'CHEMBL1234' in parser.drugs
        assert parser.drugs['CHEMBL1234']['name'] == 'Drug 1'
        assert 'EFO:0000001' in parser.diseases # Check 6 zeros
        assert parser.diseases['EFO:0000001']['name'] == 'Disease 1'

    def test_extract_drug_disease_indications(self):
        """Test extracting drug-disease indications"""
        parser = OpenTargetsParser("dummy_dir")

        parser.drug_disease_associations = [
            {
                'drug_id': 'CHEMBL1234',
                'disease_id': 'EFO:0000001', # Use 6 zeros
                'score': 0.9,
                'clinical_phase': 3
            },
            {
                'drug_id': 'CHEMBL5678',
                'disease_id': 'EFO:0000002', # Use 6 zeros
                'score': 0.4,
                'clinical_phase': 2
            }
        ]
        parser.drugs = {
            'CHEMBL1234': {'id': 'CHEMBL1234', 'name': 'Drug 1'},
            'CHEMBL5678': {'id': 'CHEMBL5678', 'name': 'Drug 2'}
        }
        # FIX: Use consistent IDs (6 zeros) as keys
        parser.diseases = {
            'EFO:0000001': {'id': 'EFO:0000001', 'name': 'Disease 1'},
            'EFO:0000002': {'id': 'EFO:0000002', 'name': 'Disease 2'}
        }

        indications = parser.extract_drug_disease_indications()

        assert len(indications) == 1
        indication = indications[0]
        assert indication['drug_id'] == 'CHEMBL1234'
        assert indication['disease_id'] == 'EFO:0000001' # Check 6 zeros
        assert indication['drug_name'] == 'Drug 1'
        assert indication['disease_name'] == 'Disease 1' # Should now pass
        assert indication['score'] == 0.9
        assert indication['clinical_phase'] == 3
        assert indication['source'] == 'opentargets'

    def test_save_opentargets_data(self, test_output_dir):
        """Test saving OpenTargets data"""
        parser = OpenTargetsParser("dummy_dir", output_dir=str(test_output_dir))
        parser.target_disease_associations = [
            {'target_id': 'ENSG00000123456', 'disease_id': 'EFO:0000001', 'score': 0.7} # Use 6 zeros
        ]
        parser.targets = {'ENSG00000123456': {'id': 'ENSG00000123456', 'name': 'Target 1'}}
        parser.diseases = {'EFO:0000001': {'id': 'EFO:0000001', 'name': 'Disease 1'}} # Use 6 zeros

        output_files = parser.save_opentargets_data(format="pickle")

        assert os.path.exists(output_files['associations'])
        assert os.path.exists(output_files['entities'])
        with open(output_files['associations'], 'rb') as f:
            assoc_data = pickle.load(f)
        assert len(assoc_data['target_disease_associations']) == 1
        with open(output_files['entities'], 'rb') as f:
            entity_data = pickle.load(f)
        assert 'ENSG00000123456' in entity_data['targets']
        assert 'EFO:0000001' in entity_data['diseases'] # Check 6 zeros

    def test_save_indications(self, test_output_dir):
        """Test saving drug-disease indications"""
        parser = OpenTargetsParser("dummy_dir", output_dir=str(test_output_dir))
        parser.drug_disease_associations = [
            {'drug_id': 'CHEMBL1234', 'disease_id': 'EFO:0000001', 'score': 0.9} # Use 6 zeros
        ]
        parser.drugs = {'CHEMBL1234': {'id': 'CHEMBL1234', 'name': 'Drug 1'}}
        parser.diseases = {'EFO:0000001': {'id': 'EFO:0000001', 'name': 'Disease 1'}} # Use 6 zeros

        output_path = parser.save_indications(format="pickle")

        assert os.path.exists(output_path)
        with open(output_path, 'rb') as f:
            indications = pickle.load(f)
        assert len(indications) == 1
        assert indications[0]['drug_id'] == 'CHEMBL1234'
        assert indications[0]['disease_id'] == 'EFO:0000001' # Check 6 zeros
