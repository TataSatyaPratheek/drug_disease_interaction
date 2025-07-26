import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from llama_index.core import QueryBundle, Response

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
