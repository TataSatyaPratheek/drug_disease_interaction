
# src/tests/test_integration/conftest.py - CORRECTED
import pytest
import time
from neo4j import GraphDatabase
import weaviate
from httpx import AsyncClient
# DO NOT import DockerCompose

# This fixture will start up the docker-compose services
@pytest.fixture(scope="session")
def docker_services(docker_compose_up):
    """
    Ensure that services are up and responsive.
    The `docker_compose_up` fixture is provided by the pytest-docker plugin.
    """
    # Wait for services to be healthy
    # This assumes you have healthchecks in your docker-compose.yml
    print("Waiting for Docker services to become healthy...")
    time.sleep(30) # Generous wait time for services to initialize
    
    # You could add more specific waits here by polling the /health endpoint
    
    yield
    
    # The plugin handles shutdown automatically
    print("Docker services will be shut down by the plugin.")

@pytest.fixture(scope="session")
def neo4j_driver(docker_services):
    """Provides a driver to the integration test Neo4j instance."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123lol123"))
    yield driver
    driver.close()

@pytest.fixture(scope="session")
def weaviate_client(docker_services):
    """Provides a client to the integration test Weaviate instance."""
    client = weaviate.connect_to_local(host="localhost", port=8080)
    yield client
    client.close()

@pytest.fixture(scope="session")
def api_client(docker_services):
    """Provides an httpx client to the running API service."""
    return AsyncClient(base_url="http://localhost:8000")
