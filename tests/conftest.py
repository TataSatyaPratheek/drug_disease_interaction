# tests/conftest.py
import os
import sys
import pytest
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path

# Add the src directory to the path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define fixture directories
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Create directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory"""
    return TEST_DATA_DIR

@pytest.fixture
def test_output_dir():
    """Return the path to the test output directory"""
    return TEST_OUTPUT_DIR

@pytest.fixture
def cleanup_output():
    """Clean up the test output directory after tests"""
    yield
    for f in TEST_OUTPUT_DIR.glob("**/*"):
        if f.is_file():
            f.unlink()

@pytest.fixture
def sample_drugbank_data():
    """Create a small sample of DrugBank data"""
    return {
        "version": "5.1.9",
        "drugs": [
            {
                "drugbank_id": "DB00001",
                "name": "Test Drug 1",
                "description": "This is a test drug",
                "cas_number": "12345-67-8",
                "groups": ["approved", "investigational"],
                "categories": [
                    {
                        "category": "Test Category",
                        "mesh_id": "D000001"
                    }
                ],
                "targets": [
                    {
                        "id": "BE0000001",
                        "name": "Test Target",
                        "organism": "Humans",
                        "actions": ["inhibitor"],
                        "known_action": "yes",
                        "polypeptides": [
                            {
                                "id": "PP00001",
                                "name": "Test Polypeptide",
                                "gene_name": "TEST1",
                                "external_identifiers": [
                                    {
                                        "resource": "UniProtKB",
                                        "identifier": "P12345"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "drugbank_id": "DB00002",
                "name": "Test Drug 2",
                "description": "Another test drug",
                "cas_number": "23456-78-9",
                "groups": ["approved"],
                "categories": [],
                "targets": []
            }
        ]
    }

@pytest.fixture
def sample_vocabulary_data():
    """Create a sample DrugBank vocabulary DataFrame"""
    data = {
        'drugbank_id': ['DB00001', 'DB00002'],
        'name': ['Test Drug 1', 'Test Drug 2'],
        'cas_number': ['12345-67-8', '23456-78-9'],
        'unii': ['ABC123', 'DEF456'],
        'synonyms': ['Drug 1|Medicine 1', 'Drug 2|Medicine 2'],
        'inchikey': ['TESTKEY1', 'TESTKEY2']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_mesh_data():
    """Create a sample of MeSH disease data"""
    return {
        "descriptors": {
            "D000001": {
                "id": "D000001",
                "name": "Test Disease 1",
                "tree_numbers": ["C01.123"],
                "description": "A test disease",
                "synonyms": ["Disease 1", "Condition 1"],
                "allowed_qualifiers": ["Q000139"],
                "is_disease": True
            },
            "D000002": {
                "id": "D000002",
                "name": "Test Disease 2",
                "tree_numbers": ["C01.456"],
                "description": "Another test disease",
                "synonyms": ["Disease 2", "Condition 2"],
                "allowed_qualifiers": ["Q000139"],
                "is_disease": True
            },
            "D000003": {
                "id": "D000003",
                "name": "Test Category",
                "tree_numbers": ["C01"],
                "description": "A test category",
                "synonyms": [],
                "allowed_qualifiers": [],
                "is_disease": False
            }
        },
        "qualifiers": {
            "Q000139": {
                "id": "Q000139",
                "name": "drug therapy",
                "tree_numbers": ["E02.319"],
                "description": "Used with disease headings for the treatment of disease by the administration of drugs"
            }
        },
        "disease_hierarchy": {
            "C01": {
                "tree_number": "C01",
                "name": "Test Category",
                "descriptor_id": "D000003",
                "children": ["C01.123", "C01.456"]
            }
        },
        "term_to_id": {
            "test disease 1": "D000001",
            "disease 1": "D000001",
            "condition 1": "D000001",
            "test disease 2": "D000002",
            "disease 2": "D000002",
            "condition 2": "D000002",
            "test category": "D000003"
        },
        "version": "2025"
    }

@pytest.fixture
def sample_opentargets_data():
    """Create a sample of OpenTargets data"""
    return {
        "drug_target_associations": [
            {
                "drug_id": "CHEMBL1234",
                "target_id": "ENSG00000123456",
                "score": 0.8,
                "mechanism": "inhibitor"
            }
        ],
        "target_disease_associations": [
            {
                "target_id": "ENSG00000123456",
                "disease_id": "EFO:0000001",
                "score": 0.7,
                "evidence": "genetic_association"
            }
        ],
        "drug_disease_associations": [
            {
                "drug_id": "CHEMBL1234",
                "disease_id": "EFO:0000001",
                "score": 0.9,
                "clinical_phase": 3
            }
        ],
        "targets": {
            "ENSG00000123456": {
                "id": "ENSG00000123456",
                "name": "Test Target",
                "symbol": "TEST1"
            }
        },
        "diseases": {
            "EFO:0000001": {
                "id": "EFO:0000001",
                "name": "Test Disease"
            }
        },
        "drugs": {
            "CHEMBL1234": {
                "id": "CHEMBL1234",
                "name": "Test Drug",
                "type": "small molecule"
            }
        }
    }

@pytest.fixture
def sample_graph():
    """Create a sample knowledge graph"""
    G = nx.MultiDiGraph()
    
    # Add drug nodes
    G.add_node("DB00001", type="drug", name="Test Drug 1")
    G.add_node("DB00002", type="drug", name="Test Drug 2")
    
    # Add target nodes
    G.add_node("BE0000001", type="protein", name="Test Target")
    G.add_node("P12345", type="polypeptide", name="Test Polypeptide", gene_name="TEST1")
    
    # Add disease nodes
    G.add_node("D000001", type="disease", name="Test Disease 1")
    G.add_node("D000002", type="disease", name="Test Disease 2")
    
    # Add edges
    G.add_edge("DB00001", "BE0000001", type="targets", actions="inhibitor")
    G.add_edge("BE0000001", "P12345", type="has_polypeptide")
    G.add_edge("DB00001", "D000001", type="treats", score=0.8)
    G.add_edge("P12345", "D000001", type="associated_with", score=0.7)
    
    return G

# Create a directory with test XML files for MeSH
@pytest.fixture
def create_test_mesh_files(test_data_dir):
    """Create test MeSH XML files"""
    mesh_dir = test_data_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    
    # Create a minimal descriptor XML file
    desc_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <DescriptorRecordSet>
        <DescriptorRecord>
            <DescriptorUI>D000001</DescriptorUI>
            <DescriptorName>
                <String>Test Disease 1</String>
            </DescriptorName>
            <TreeNumberList>
                <TreeNumber>C01.123</TreeNumber>
            </TreeNumberList>
            <ScopeNote>A test disease</ScopeNote>
            <ConceptList>
                <Concept>
                    <TermList>
                        <Term>
                            <String>Disease 1</String>
                        </Term>
                        <Term>
                            <String>Condition 1</String>
                        </Term>
                    </TermList>
                </Concept>
            </ConceptList>
            <AllowableQualifierList>
                <AllowableQualifier>
                    <QualifierReferredTo>
                        <QualifierUI>Q000139</QualifierUI>
                    </QualifierReferredTo>
                </AllowableQualifier>
            </AllowableQualifierList>
        </DescriptorRecord>
        <DescriptorRecord>
            <DescriptorUI>D000003</DescriptorUI>
            <DescriptorName>
                <String>Test Category</String>
            </DescriptorName>
            <TreeNumberList>
                <TreeNumber>C01</TreeNumber>
            </TreeNumberList>
            <ScopeNote>A test category</ScopeNote>
        </DescriptorRecord>
    </DescriptorRecordSet>
    """
    
    with open(mesh_dir / "desc2025.xml", "w") as f:
        f.write(desc_xml)
    
    # Create a minimal qualifier XML file
    qual_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <QualifierRecordSet>
        <QualifierRecord>
            <QualifierUI>Q000139</QualifierUI>
            <QualifierName>
                <String>drug therapy</String>
            </QualifierName>
            <TreeNumberList>
                <TreeNumber>E02.319</TreeNumber>
            </TreeNumberList>
            <ScopeNote>Used with disease headings for the treatment of disease by the administration of drugs</ScopeNote>
        </QualifierRecord>
    </QualifierRecordSet>
    """
    
    with open(mesh_dir / "qual2025.xml", "w") as f:
        f.write(qual_xml)
    
    return mesh_dir

# Create a directory with test parquet files for OpenTargets
@pytest.fixture
def create_test_opentargets_files(test_data_dir):
    """Create test OpenTargets parquet files"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow import csv
    
    ot_dir = test_data_dir / "opentargets"
    ot_dir.mkdir(exist_ok=True)
    
    # Create a CSV with target-disease associations to convert to parquet
    csv_data = """
    targetId,diseaseId,score,evidence
    ENSG00000123456,EFO:0000001,0.7,genetic_association
    """
    
    csv_path = ot_dir / "temp.csv"
    with open(csv_path, "w") as f:
        f.write(csv_data.strip())
    
    # Convert to parquet
    table = csv.read_csv(csv_path)
    pq.write_table(table, ot_dir / "part-00000.snappy.parquet")
    
    # Remove temporary CSV
    csv_path.unlink()
    
    return ot_dir

@pytest.fixture
def create_test_drugbank_files(test_data_dir):
    """Create test DrugBank files"""
    drugbank_dir = test_data_dir / "drugbank"
    drugbank_dir.mkdir(exist_ok=True)
    
    # Create a minimal DrugBank XML snippet
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <drugbank xmlns="http://www.drugbank.ca" version="5.1.9">
        <drug type="small molecule" created="2005-06-13" updated="2022-05-16">
            <drugbank-id primary="true">DB00001</drugbank-id>
            <name>Test Drug 1</name>
            <description>This is a test drug</description>
            <cas-number>12345-67-8</cas-number>
            <groups>
                <group>approved</group>
                <group>investigational</group>
            </groups>
            <categories>
                <category>
                    <category>Test Category</category>
                    <mesh-id>D000001</mesh-id>
                </category>
            </categories>
            <targets>
                <target>
                    <id>BE0000001</id>
                    <name>Test Target</name>
                    <organism>Humans</organism>
                    <actions>
                        <action>inhibitor</action>
                    </actions>
                    <known-action>yes</known-action>
                    <polypeptide id="PP00001">
                        <name>Test Polypeptide</name>
                        <gene-name>TEST1</gene-name>
                        <external-identifiers>
                            <external-identifier>
                                <resource>UniProtKB</resource>
                                <identifier>P12345</identifier>
                            </external-identifier>
                        </external-identifiers>
                    </polypeptide>
                </target>
            </targets>
        </drug>
    </drugbank>
    """
    
    with open(drugbank_dir / "test_drugbank.xml", "w") as f:
        f.write(xml_content)
    
    # Create a minimal vocabulary CSV
    csv_content = """DrugBank ID,Common name,CAS,UNII,Synonyms,Standard InChI Key
DB00001,Test Drug 1,12345-67-8,ABC123,"Drug 1|Medicine 1",TESTKEY1
DB00002,Test Drug 2,23456-78-9,DEF456,"Drug 2|Medicine 2",TESTKEY2
"""
    
    with open(drugbank_dir / "test_vocabulary.csv", "w") as f:
        f.write(csv_content)
    
    return drugbank_dir