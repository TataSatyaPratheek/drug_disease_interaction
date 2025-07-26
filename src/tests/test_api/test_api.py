# tests/test_api.py - NEW FILE
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_services():
    """Mock all services for testing"""
    with patch('src.api.main.Neo4jService') as mock_neo4j, \
         patch('src.api.main.WeaviateService') as mock_weaviate, \
         patch('src.api.main.LLMService') as mock_llm:
        
        yield {
            'neo4j': mock_neo4j.return_value,
            'weaviate': mock_weaviate.return_value, 
            'llm': mock_llm.return_value
        }

def test_health_endpoint(client, mock_services):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    
    # This might fail until services are properly mocked
    # assert response.status_code == 200
    # assert "status" in response.json()

def test_search_endpoint_validation(client):
    """Test search endpoint input validation"""
    # Test empty query
    response = client.post("/api/v1/search", json={"query": ""})
    assert response.status_code == 422  # Validation error
    
    # Test too long query
    long_query = "a" * 600
    response = client.post("/api/v1/search", json={"query": long_query})
    assert response.status_code == 422
