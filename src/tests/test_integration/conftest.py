# src/tests/test_integration/conftest.py - FINAL CORRECTED VERSION


import pytest
import time
import httpx
from neo4j import GraphDatabase
import weaviate

def is_responsive(url):
    """Check if a service is responsive."""
    try:
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        return True
    except (httpx.ConnectError, httpx.HTTPStatusError):
        return False


@pytest.fixture(scope="session")
def live_services(docker_ip, docker_services):
    """
    Ensure that all services are up and responsive.
    The `docker_services` fixture is provided by the pytest-docker plugin.
    """
    api_url = f"http://{docker_ip}:{docker_services.port_for('api', 8000)}/api/v1/health"
    # Use the wait_until_responsive utility provided by the plugin
    docker_services.wait_until_responsive(
        timeout=120.0, pause=2.0, check=lambda: is_responsive(api_url)
    )
    return docker_services


@pytest.fixture(scope="session")
def neo4j_driver(docker_ip, live_services):
    """Provides a driver to the integration test Neo4j instance."""
    port = live_services.port_for("neo4j", 7687)
    driver = GraphDatabase.driver(f"bolt://{docker_ip}:{port}", auth=("neo4j", "123lol123"))
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def weaviate_client(docker_ip, live_services):
    """Provides a client to the integration test Weaviate instance."""
    port = live_services.port_for("weaviate", 8080)
    client = weaviate.connect_to_local(host=docker_ip, port=port)
    yield client
    client.close()


@pytest.fixture(scope="session")
def api_client(docker_ip, live_services):
    """Provides an httpx client to the running API service."""
    port = live_services.port_for("api", 8000)
    return httpx.AsyncClient(base_url=f"http://{docker_ip}:{port}", timeout=30)

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
