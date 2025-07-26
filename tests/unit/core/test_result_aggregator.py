import pytest
from src.graphrag.core import result_aggregator
from datetime import datetime

def make_entity(eid, name, score=1.0, desc=None, source=None):
    e = {"id": eid, "name": name, "score": score}
    if desc:
        e["description"] = desc
    if source:
        e["source"] = source
    return e

def test_hybrid_merge_and_metadata():
    vector_results = {
        "drugs": [make_entity("d1", "DrugA", 0.9, "descA")],
        "diseases": [make_entity("ds1", "DiseaseX", 0.8)],
        "proteins": [],
        "relationships": []
    }
    graph_results = {
        "drugs": [make_entity("d1", "DrugA", 0.95, "descA graph")],
        "diseases": [make_entity("ds2", "DiseaseY", 0.7)],
        "proteins": [make_entity("p1", "Protein1", 0.6)],
        "relationships": []
    }
    query_context = {"query_type": "mechanism"}
    agg = result_aggregator.aggregate_search_results(vector_results, graph_results, query_context)
    # Check merged drug entity is hybrid
    drugs = agg["entities"]["drugs"]
    assert len(drugs) == 1
    assert drugs[0]["id"] == "d1"
    assert "hybrid" == drugs[0]["source"]
    assert set(drugs[0]["sources"]) == {"vector_search", "graph_traversal"}
    # Check all entity types present
    assert set(agg["entities"].keys()) == {"drugs", "diseases", "proteins", "relationships"}
    # Check metadata
    meta = agg["metadata"]
    assert meta["vector_search_count"] == 2
    assert meta["graph_traversal_count"] == 3
    assert meta["total_unique_entities"] == 4
    assert "vector_search" in meta["sources"] and "graph_traversal" in meta["sources"]
    assert meta["success"] if "success" in meta else agg["success"]
    # Check context string
    assert "DrugA" in agg["context"] and "DiseaseX" in agg["context"] and "Protein1" in agg["context"]
    # Check citations
    citations = agg["citations"]
    assert any(c["source"] == "hybrid" for c in citations if c["name"] == "DrugA")
    assert any(c["source"] == "graph_traversal" for c in citations if c["name"] == "Protein1")

def test_format_results_for_llm_truncation():
    agg = {
        "context": "A" * 5000,
        "metadata": {"total_unique_entities": 2}
    }
    out = result_aggregator.format_results_for_llm(agg, max_context_length=100)
    assert out.startswith("A" * 100)
    assert "truncated" in out
    assert "entities analyzed" in out

def test_generate_followup_questions():
    agg = {"entities": {"drugs": [{"id": "d1", "name": "DrugA"}], "proteins": []}}
    ctx = {"query_type": "comparison"}
    followups = result_aggregator.generate_followup_questions(agg, ctx)
    assert any("mechanisms" in f for f in followups)
    assert len(followups) <= 3
    # Entity-based
    agg2 = {"entities": {"drugs": [{"id": "d1", "name": "DrugA"}], "proteins": [{"id": "p1", "name": "P1"}]}}
    ctx2 = {"query_type": "safety"}
    followups2 = result_aggregator.generate_followup_questions(agg2, ctx2)
    assert any("conditions" in f for f in followups2)
    assert any("proteins" in f or "drugs" in f for f in followups2)
