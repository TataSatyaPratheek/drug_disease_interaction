# src/tests/test_integration/conftest.py - ROBUST VERSION

import pytest
import time
import httpx
from neo4j import GraphDatabase
import weaviate
from typing import AsyncGenerator

# Helper function to poll a service until it's ready
def wait_for_service(name: str, url: str, timeout: int = 60):
    """Polls a service URL until it returns a 200 OK or the timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with httpx.Client() as client:
                response = client.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"✅ {name} service is ready.")
                    return True
        except httpx.ConnectError:
            pass  # Service not up yet
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(f"🔴 {name} service did not become ready within {timeout} seconds.")

@pytest.fixture(scope="session", autouse=True)
def ensure_services_are_running_before_tests():
    """
    Blocks test execution until all Docker services are responsive.
    This runs automatically for the entire test session.
    """
    print("\n--- Ensuring all services are running for integration tests ---")
    wait_for_service("Neo4j", "http://localhost:7474")
    wait_for_service("Weaviate", "http://localhost:8080/v1/.well-known/ready")
    wait_for_service("Ollama", "http://localhost:11434")
    print("--- All services are responsive. Starting tests. ---\n")

@pytest.fixture(scope="session")
def neo4j_driver():
    """Provides a driver to the local Neo4j instance."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123lol123"))
    try:
        with driver.session() as session:
            session.run("RETURN 1").single()
        yield driver
    except Exception as e:
        pytest.skip(f"Neo4j connection failed: {e}")
    finally:
        driver.close()

@pytest.fixture(scope="session") 
def weaviate_client():
    """Provides a client to the local Weaviate instance."""
    try:
        client = weaviate.connect_to_local(host="localhost", port=8080)
        if not client.is_ready():
            pytest.skip("Weaviate is not ready")
        yield client
    except Exception as e:
        pytest.skip(f"Weaviate connection failed: {e}")
    finally:
        if 'client' in locals():
            client.close()

@pytest.fixture(scope="session")
async def api_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an HTTP client for testing the API."""
    async with httpx.AsyncClient(
        base_url="http://localhost:8000", 
        timeout=30.0
    ) as client:
        yield client

@pytest.fixture(scope="function")
def clean_test_data(neo4j_driver):
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
