
# Enhanced API tests with dependency overrides and async mocks
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
from llama_index.core.response import Response

from src.api.main import app
from src.api.dependencies import get_hybrid_engine, get_neo4j_service

# Mock engine that will be injected
mock_engine = Mock()
mock_neo4j = Mock()

def override_get_hybrid_engine():
    return mock_engine

def override_get_neo4j_service():
    return mock_neo4j

# Apply the overrides to the app
app.dependency_overrides[get_hybrid_engine] = override_get_hybrid_engine
app.dependency_overrides[get_neo4j_service] = override_get_neo4j_service

client = TestClient(app)

def test_search_endpoint_success():
    """Test the /search endpoint on a successful query."""
    # Configure the mock engine's async query method
    mock_engine.aquery = AsyncMock(return_value=Response(
        response="Aspirin is a drug.",
        metadata={
            "retrieved_results": [{
                "id": "DB00945", "name": "Aspirin", "type": "Drug", 
                "score": 0.95, "description": "...", "source": "weaviate"
            }],
            "sources": ["neo4j", "weaviate"]
        }
    ))
    
    response = client.post("/api/v1/search", json={"query": "What is Aspirin?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Aspirin is a drug."
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "Aspirin"
    mock_engine.aquery.assert_called_once()

def test_search_endpoint_validation_error():
    """Test for validation errors on invalid requests."""
    response = client.post("/api/v1/search", json={"query": ""})
    assert response.status_code == 422 # Unprocessable Entity
    
def test_suggest_endpoint_success():
    """Test the /suggest endpoint."""
    mock_neo4j.search_drug_disease_paths = AsyncMock(return_value=[{"name": "Aspirin"}])
    
    response = client.get("/api/v1/suggest?query=asp")
    
    assert response.status_code == 200
    assert response.json() == {"suggestions": [{"name": "Aspirin"}]}
    mock_neo4j.search_drug_disease_paths.assert_called_once_with("asp", 10)

# Add a fixture to reset mocks between tests
@pytest.fixture(autouse=True)
def reset_mocks():
    mock_engine.reset_mock()
    mock_neo4j.reset_mock()
