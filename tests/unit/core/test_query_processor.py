import pytest
from src.graphrag.core import query_processor as qp

def test_classify_query_type():
    assert qp.classify_query_type("Compare aspirin vs ibuprofen") == "comparison"
    assert qp.classify_query_type("What is the mechanism of action of metformin?") == "mechanism"
    assert qp.classify_query_type("What are the side effects of statins?") == "safety"
    assert qp.classify_query_type("Can we repurpose metformin for cancer?") == "repurposing"
    assert qp.classify_query_type("How do these drugs interact?") == "interaction"
    assert qp.classify_query_type("What is the best treatment for hypertension?") == "treatment"
    assert qp.classify_query_type("Is diabetes associated with obesity?") == "association"
    assert qp.classify_query_type("Tell me about aspirin") == "general"

def test_enhanced_entity_extraction():
    q = "Compare aspirin vs ibuprofen for treating hypertension and COVID-19."
    entities = qp.enhanced_entity_extraction(q)
    assert "aspirin" in [e.lower() for e in entities["drugs"]]
    assert "ibuprofen" in [e.lower() for e in entities["drugs"]]
    assert "hypertension" in [e.lower() for e in entities["diseases"]]
    assert "COVID-19" in entities["diseases"] or "covid-19" in [e.lower() for e in entities["diseases"]]

def test_preprocess_query():
    q = "Compare aspirin vs ibuprofen for treating hypertension."
    pq = qp.preprocess_query(q)
    assert pq.query_type == "comparison"
    assert pq.strategy == qp.QueryStrategy.COMPARATIVE_ANALYSIS
    assert "aspirin" in [e.lower() for e in pq.extracted_entities["drugs"]]
    assert "ibuprofen" in [e.lower() for e in pq.extracted_entities["drugs"]]
    assert "hypertension" in [e.lower() for e in pq.extracted_entities["diseases"]]
    assert pq.confidence > 0.5
    assert "compare" in pq.reasoning or "comparison" in pq.reasoning

def test_determine_enhanced_strategy():
    # Comparison with 2 drugs
    entities = {"drugs": ["aspirin", "ibuprofen"], "diseases": [], "proteins": []}
    assert qp.determine_enhanced_strategy("comparison", entities) == qp.QueryStrategy.COMPARATIVE_ANALYSIS
    # Mechanism with 1 drug
    entities = {"drugs": ["metformin"], "diseases": [], "proteins": []}
    assert qp.determine_enhanced_strategy("mechanism", entities) == qp.QueryStrategy.PATHWAY_TRAVERSAL
    # Repurposing
    entities = {"drugs": ["metformin"], "diseases": [], "proteins": []}
    assert qp.determine_enhanced_strategy("repurposing", entities) == qp.QueryStrategy.COMMUNITY_ANALYSIS
    # Only 1 entity
    entities = {"drugs": ["aspirin"], "diseases": [], "proteins": []}
    assert qp.determine_enhanced_strategy("general", entities) == qp.QueryStrategy.ENTITY_EXPANSION
    # Multiple entities, ambiguous
    entities = {"drugs": ["aspirin"], "diseases": ["hypertension"], "proteins": []}
    assert qp.determine_enhanced_strategy("general", entities) == qp.QueryStrategy.MULTI_ENTITY_ANALYSIS

def test_generate_processing_reasoning():
    entities = {"drugs": ["aspirin", "ibuprofen"], "diseases": [], "proteins": []}
    s = qp.QueryStrategy.COMPARATIVE_ANALYSIS
    r = qp.generate_processing_reasoning("comparison", entities, s)
    assert "compare" in r or "comparison" in r
    s = qp.QueryStrategy.PATHWAY_TRAVERSAL
    r = qp.generate_processing_reasoning("mechanism", entities, s)
    assert "mechanism" in r or "pathway" in r
    s = qp.QueryStrategy.COMMUNITY_ANALYSIS
    r = qp.generate_processing_reasoning("repurposing", entities, s)
    assert "repurposing" in r or "community" in r
