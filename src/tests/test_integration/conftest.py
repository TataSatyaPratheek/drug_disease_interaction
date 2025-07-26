# src/tests/test_integration/conftest.py - FINAL CORRECTED VERSION

import pytest
import time
from neo4j import GraphDatabase
import weaviate
from httpx import AsyncClient

def is_ready(services):
    """Helper function to check if all docker services are responsive."""
    try:
        # Check Neo4j
        neo4j_port = services.port_for("neo4j", 7687)
        with GraphDatabase.driver(f"bolt://localhost:{neo4j_port}", auth=("neo4j", "123lol123")) as driver:
            driver.verify_connectivity()
        
        # Check Weaviate
        weaviate_port = services.port_for("weaviate", 8080)
        with weaviate.connect_to_local(host="localhost", port=weaviate_port) as client:
            if not client.is_ready():
                return False
        
        # All checks passed
        return True
    except Exception as e:
        print(f"Health check during test setup failed: {e}")
        return False

@pytest.fixture(scope="session")
def live_services(docker_services):
    """
    The main fixture that uses the 'docker_services' provided by pytest-docker.
    It waits until all services are fully responsive before yielding.
    """
    print("Waiting for Docker services to become healthy...")
    docker_services.wait_until_responsive(
        timeout=60.0, pause=1.0, check=lambda: is_ready(docker_services)
    )
    return docker_services

@pytest.fixture(scope="session")
def neo4j_driver(live_services):
    """Provides a driver to the integration test Neo4j instance."""
    port = live_services.port_for("neo4j", 7687)
    driver = GraphDatabase.driver(f"bolt://localhost:{port}", auth=("neo4j", "123lol123"))
    yield driver
    driver.close()

@pytest.fixture(scope="session")
def weaviate_client(live_services):
    """Provides a client to the integration test Weaviate instance."""
    port = live_services.port_for("weaviate", 8080)
    client = weaviate.connect_to_local(host="localhost", port=port)
    yield client
    # The 'with' statement in 'is_ready' handles closing, but this is good practice
    client.close()

@pytest.fixture(scope="session")
def api_client(live_services):
    """Provides an httpx client to the running API service."""
    port = live_services.port_for("api", 8000)
    return AsyncClient(base_url=f"http://localhost:{port}", timeout=30)

# The data setup fixture now depends on the new live_services fixtures
@pytest.fixture(scope="module", autouse=True)
def setup_test_data(neo4j_driver, weaviate_client):
    """Populates the databases with a small, consistent test dataset."""
    # Setup Neo4j data
    try:
        with neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n") # Clear previous data
            session.run("""
                CREATE (d:Drug {id: 'DB_TEST', name: 'Testodrug'})
                CREATE (t:Disease {id: 'D_TEST', name: 'Testitis'})
                CREATE (d)-[:TREATS]->(t)
            """)
    except Exception as e:
        pytest.fail(f"Failed to set up Neo4j test data: {e}")
        
    # Setup Weaviate data
    try:
        if weaviate_client.collections.exists("Drug"):
            weaviate_client.collections.delete("Drug")
        if weaviate_client.collections.exists("Disease"):
            weaviate_client.collections.delete("Disease")
        
        drug_collection = weaviate_client.collections.create(
            name="Drug",
            properties=[
                weaviate.classes.config.Property(name="name", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="neo4j_id", data_type=weaviate.classes.config.DataType.TEXT)
            ]
        )
        drug_collection.data.insert({
            "name": "Testodrug",
            "neo4j_id": "DB_TEST"
        })
    except Exception as e:
        pytest.fail(f"Failed to set up Weaviate test data: {e}")
        
    yield # Tests run here
