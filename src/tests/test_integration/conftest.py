
# src/tests/test_integration/conftest.py - RELIABLE VERSION

import pytest
import time
import httpx
import asyncio
from neo4j import GraphDatabase
import weaviate
from typing import AsyncGenerator

def check_service_health(url: str, timeout: int = 5) -> bool:
    """Check if a service is responsive."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            return response.status_code == 200
    except Exception:
        return False

@pytest.fixture(scope="session")
def ensure_services_running():
    """
    Ensure all required services are running before tests.
    This fixture fails fast if services aren't available.
    """
    services = {
        "Neo4j": "http://localhost:7474",
        "Weaviate": "http://localhost:8080/v1/.well-known/ready", 
        "Ollama": "http://localhost:11434",
    }
    
    failed_services = []
    for name, url in services.items():
        if not check_service_health(url, timeout=10):
            failed_services.append(name)
    
    if failed_services:
        pytest.skip(f"Required services not running: {', '.join(failed_services)}. "
                   f"Start them with: docker-compose -f docker/docker-compose.yml up -d")
    
    return True

@pytest.fixture(scope="session")
def neo4j_driver(ensure_services_running):
    """Provides a driver to the local Neo4j instance."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123lol123"))
    try:
        driver.verify_connectivity()
        yield driver
    finally:
        driver.close()

@pytest.fixture(scope="session") 
def weaviate_client(ensure_services_running):
    """Provides a client to the local Weaviate instance."""
    client = weaviate.connect_to_local(host="localhost", port=8080)
    try:
        if not client.is_ready():
            pytest.skip("Weaviate is not ready")
        yield client
    finally:
        client.close()

@pytest.fixture(scope="session")
async def api_client(ensure_services_running) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an HTTP client for testing the API."""
    async with httpx.AsyncClient(
        base_url="http://localhost:8000", 
        timeout=30.0
    ) as client:
        yield client

@pytest.fixture(scope="function")
def clean_test_data(neo4j_driver, weaviate_client):
    """Setup and cleanup test data for each test."""
    # Setup test data
    with neo4j_driver.session() as session:
        # Clean any existing test data
        session.run("MATCH (n:TestDrug) DETACH DELETE n")
        session.run("MATCH (n:TestDisease) DETACH DELETE n")
        
        # Create test data
        session.run("""
            CREATE (d:TestDrug {id: 'TEST_DRUG_001', name: 'TestDrug'})
            CREATE (dis:TestDisease {id: 'TEST_DISEASE_001', name: 'TestDisease'}) 
            CREATE (d)-[:TREATS]->(dis)
        """)
    
    yield  # Test runs here
    
    # Cleanup after test
    with neo4j_driver.session() as session:
        session.run("MATCH (n:TestDrug) DETACH DELETE n")
        session.run("MATCH (n:TestDisease) DETACH DELETE n")
