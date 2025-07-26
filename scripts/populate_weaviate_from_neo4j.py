import weaviate
import neo4j
import logging
import os
from weaviate.util import get_valid_uuid
from uuid import uuid4

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "123lol123") # Ensure this matches your actual password
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# --- Neo4j Connection ---
try:
    neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    logging.info("Successfully connected to Neo4j.")
except neo4j.exceptions.AuthError as e:
    logging.error(f"Neo4j authentication failed: {e}. Please check your credentials.")
    exit()

# --- Weaviate Connection ---
try:
    # Use the modern connect_to_local() for easy connection to a local instance
    client = weaviate.connect_to_local()
    logging.info(f"Successfully connected to Weaviate. Client is ready: {client.is_ready()}")
except Exception as e:
    logging.error(f"Could not connect to Weaviate: {e}")
    exit()


def create_weaviate_schema():
    """Creates the necessary classes in Weaviate if they don't exist."""
    classes_to_create = ["Drug", "Disease", "Target", "Pathway"]
    existing_schema = client.collections.list_all()
    existing_class_names = {details.name for details in existing_schema.values()}

    for class_name in classes_to_create:
        if class_name not in existing_class_names:
            try:
                client.collections.create(name=class_name)
                logging.info(f"Created Weaviate class: {class_name}")
            except Exception as e:
                logging.error(f"Failed to create class {class_name}: {e}")
        else:
            logging.info(f"Class '{class_name}' already exists in Weaviate.")


def populate_weaviate_from_neo4j():
    """
    Fetches data from Neo4j and batch-indexes it into Weaviate using the v4 client.
    """
    create_weaviate_schema()

    with neo4j_driver.session() as session:
        # Define the nodes to extract
        node_queries = {
            "Drug": "MATCH (n:Drug) RETURN n.id AS id, n.name AS name, n.description AS description",
            "Disease": "MATCH (n:Disease) RETURN n.id AS id, n.name AS name, n.description AS description",
            "Target": "MATCH (n:Target) RETURN n.id AS id, n.name AS name, n.description AS description",
            "Pathway": "MATCH (n:Pathway) RETURN n.id AS id, n.name AS name, n.description AS description",
        }

        # Use Weaviate's context manager for efficient, auto-batching
        with client.batch.dynamic() as batch:
            for class_name, query in node_queries.items():
                logging.info(f"Fetching and indexing nodes for class: {class_name}")
                results = session.run(query).data()

                for record in results:
                    properties = {
                        "neo4j_id": record.get("id"),
                        "name": record.get("name"),
                        "description": record.get("description", "") # Use empty string if description is null
                    }
                    
                    # Generate a consistent UUID from the Neo4j ID to prevent duplicates
                    generated_uuid = get_valid_uuid(uuid4())

                    batch.add_object(
                        collection=class_name,
                        properties=properties,
                        uuid=generated_uuid
                    )
        
        logging.info(f"Batch import finished. Total objects failed: {len(client.batch.failed_objects)}")


if __name__ == "__main__":
    try:
        populate_weaviate_from_neo4j()
        logging.info("✅ Weaviate population from Neo4j complete.")
    finally:
        # Ensure connections are closed
        neo4j_driver.close()
        client.close()

