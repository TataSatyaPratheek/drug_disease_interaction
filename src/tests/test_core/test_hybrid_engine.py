import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from llama_index.core import QueryBundle, Response

# Patch CrossEncoder globally for all tests to avoid model loading errors
@pytest.fixture(autouse=True, scope="module")
def patch_cross_encoder():
    with patch("src.core.hybrid_engine.CrossEncoder") as mock_ce:
        mock_instance = MagicMock()
        # The predict method returns a list of dummy scores
        mock_instance.predict.return_value = [0.5] * 100
        mock_ce.return_value = mock_instance
        yield

from src.core.hybrid_engine import HybridRAGEngine

@pytest.fixture
def mock_services():
    """Provides mocked services for the hybrid engine."""
    mock_neo4j = Mock()
    mock_weaviate = Mock()
    mock_llm = Mock()

    # Create realistic mock data for the Neo4j result.
    realistic_graph_result = [
        {
            "node_details": [
                {"id": "DB_001", "name": "DrugA"},
                {"id": "D_001", "name": "DiseaseX"}
            ],
            "rel_types": ["TREATS"]
        }
    ]
    mock_neo4j.search_drug_disease_paths = AsyncMock(return_value=realistic_graph_result)
    mock_weaviate.hybrid_search = AsyncMock(return_value=[{"source": "weaviate_result", "score": 0.9, "id": "vec1", "name": "Weaviate Result", "description": "Vector DB result", "collection": "vector",}])
    mock_llm.generate_response = AsyncMock(return_value="This is the LLM answer.")
    return mock_neo4j, mock_weaviate, mock_llm

@pytest.fixture
def hybrid_engine(mock_services):
    """Provides a HybridRAGEngine instance with mocked services."""
    mock_neo4j, mock_weaviate, mock_llm = mock_services
    return HybridRAGEngine(mock_neo4j, mock_weaviate, mock_llm, config={})

@pytest.mark.asyncio
async def test_aquery_success_path(hybrid_engine, mock_services):
    """Tests the successful query path of the hybrid engine."""
    _, mock_weaviate, mock_llm = mock_services
    query_bundle = QueryBundle("What is aspirin?")
    
    response = await hybrid_engine.aquery(query_bundle)

    # Assertions
    assert isinstance(response, Response)
    assert response.response == "This is the LLM answer."
    assert len(response.metadata['retrieved_results']) > 0
    mock_llm.generate_response.assert_called_once()
    mock_weaviate.hybrid_search.assert_called_once_with("What is aspirin?")

@pytest.mark.asyncio
async def test_retrieval_failure_graceful_handling(hybrid_engine, mock_services):
    """Tests that the engine handles failures in one of the retrieval services."""
    mock_neo4j, mock_weaviate, mock_llm = mock_services
    
    # Simulate Neo4j failing
    mock_neo4j.search_drug_disease_paths.side_effect = Exception("Neo4j connection failed")
    
    query_bundle = QueryBundle("Test query")
    response = await hybrid_engine.aquery(query_bundle)
    
    # Should still produce a response using Weaviate results
    assert response.response == "This is the LLM answer."
    assert "weaviate_result" in [r.get('source') for r in response.metadata['retrieved_results']]
    assert "neo4j_result" not in [r.get('source') for r in response.metadata['retrieved_results']]

@pytest.mark.asyncio
async def test_vector_failure_graceful_handling(hybrid_engine, mock_services):
    """Tests that the engine handles failures in the vector retrieval service."""
    mock_neo4j, mock_weaviate, mock_llm = mock_services

    # Simulate Weaviate failing
    mock_weaviate.hybrid_search.side_effect = Exception("Weaviate connection failed")

    query_bundle = QueryBundle("Test query")
    response = await hybrid_engine.aquery(query_bundle)

    # Should still produce a response using Neo4j results
    assert response.response == "This is the LLM answer."
    assert any("DrugA" in r.get("node_details", [{}])[0].get("name", "") for r in response.metadata['retrieved_results'])
    assert not any(r.get('source') == "weaviate_result" for r in response.metadata['retrieved_results'])

@pytest.mark.asyncio
async def test_empty_results_handling(hybrid_engine, mock_services):
    """Tests that the engine handles empty results gracefully."""
    mock_neo4j, mock_weaviate, mock_llm = mock_services

    # Both services return empty
    mock_neo4j.search_drug_disease_paths = AsyncMock(return_value=[])
    mock_weaviate.hybrid_search = AsyncMock(return_value=[])

    query_bundle = QueryBundle("No results query")
    response = await hybrid_engine.aquery(query_bundle)

    assert response.response == "This is the LLM answer."
    assert response.metadata['retrieved_results'] == []

@pytest.mark.asyncio
async def test_merge_and_rerank_called(monkeypatch, hybrid_engine, mock_services):
    """Test that _merge_and_rerank is called with correct arguments."""
    mock_neo4j, mock_weaviate, mock_llm = mock_services
    query = "Check rerank"
    called = {}

    def fake_merge_and_rerank(q, graph, vector):
        called['query'] = q
        called['graph'] = graph
        called['vector'] = vector
        return [{"source": "dummy"}]

    monkeypatch.setattr(hybrid_engine, "_merge_and_rerank", fake_merge_and_rerank)
    query_bundle = QueryBundle(query)
    await hybrid_engine.aquery(query_bundle)
    assert called['query'] == query
    assert isinstance(called['graph'], list)
    assert isinstance(called['vector'], list)

@pytest.mark.asyncio
async def test_aquery_success_path(hybrid_engine, mock_services):
    """Tests the successful query path of the hybrid engine."""
    _, mock_weaviate, mock_llm = mock_services
    query_bundle = QueryBundle("What is aspirin?")
    
    response = await hybrid_engine.aquery(query_bundle)

    # Assertions
    assert isinstance(response, Response)
    assert response.response == "This is the LLM answer."
    assert len(response.metadata['retrieved_results']) > 0
    mock_llm.generate_response.assert_called_once()
    mock_weaviate.hybrid_search.assert_called_once_with("What is aspirin?")

@pytest.mark.asyncio
async def test_retrieval_failure_graceful_handling(hybrid_engine, mock_services):
    """Tests that the engine handles failures in one of the retrieval services."""
    mock_neo4j, mock_weaviate, mock_llm = mock_services
    
    # Simulate Neo4j failing
    mock_neo4j.search_drug_disease_paths.side_effect = Exception("Neo4j connection failed")
    
    query_bundle = QueryBundle("Test query")
    response = await hybrid_engine.aquery(query_bundle)
    
    # Should still produce a response using Weaviate results
    assert response.response == "This is the LLM answer."
    assert "weaviate_result" in [r.get('source') for r in response.metadata['retrieved_results']]
    assert "neo4j_result" not in [r.get('source') for r in response.metadata['retrieved_results']]
