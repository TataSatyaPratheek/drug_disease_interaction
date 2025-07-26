import pytest
import weaviate

@pytest.fixture(scope="module", autouse=True)
def setup_test_data(neo4j_driver, weaviate_client):
    """Populates the databases with a small, consistent test dataset."""
    # Setup Neo4j data
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n") # Clear previous data
        session.run("""
            CREATE (d:Drug {id: 'DB_TEST', name: 'Testodrug'})
            CREATE (t:Disease {id: 'D_TEST', name: 'Testitis'})
            CREATE (d)-[:TREATS]->(t)
        """)
        
    # Setup Weaviate data
    if weaviate_client.collections.exists("Drug"):
        weaviate_client.collections.delete("Drug")
    
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
    
    yield # Tests run here
    
    # Teardown (optional, as docker will be torn down anyway)
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    if weaviate_client.collections.exists("Drug"):
        weaviate_client.collections.delete("Drug")

@pytest.mark.asyncio
async def test_api_health_endpoint(api_client):
    """Tests if the API health check responds correctly against live services."""
    response = await api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["services"]["neo4j"] == "ok"
    assert data["services"]["weaviate"] == "ok"

@pytest.mark.asyncio
async def test_end_to_end_search(api_client):
    """Performs a full end-to-end search query."""
    response = await api_client.post("/api/v1/search", json={"query": "Testodrug"})
    
    assert response.status_code == 200
    data = response.json()
    assert "Testodrug" in data["answer"]
    assert len(data["entities"]) > 0
    assert data["entities"][0]["name"] == "Testodrug"
