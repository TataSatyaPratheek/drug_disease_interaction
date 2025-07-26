"""
Script to populate Weaviate vector database directly from Neo4j using optimized batch processing.
Hardware-optimized for Ryzen 4800H + GTX 1650Ti + 16GB RAM.
"""

from src.graphrag.core.optimized_vector_store import OptimizedWeaviateGraphStore
from src.graphrag.core.connection_manager import get_weaviate_manager
from src.graphrag.core.neo4j_analytics import Neo4jGraphAnalytics
import neo4j

# Neo4j connection (update password as needed)
driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
analytics = Neo4jGraphAnalytics(driver)

# Hardware-optimized Weaviate vector store
vector_store = OptimizedWeaviateGraphStore(
    client=get_weaviate_manager().get_client(),
    max_workers=4,  # 8-core CPU, use 4 for optimal throughput
    cache_size=1000  # 16GB RAM optimization
)

def populate_weaviate_from_neo4j():
    with driver.session() as session:
        # Extract nodes by type (adjust limits as needed)
        drugs = session.run("MATCH (n:Drug) RETURN n LIMIT 10000").data()
        diseases = session.run("MATCH (n:Disease) RETURN n LIMIT 10000").data()
        targets = session.run("MATCH (n:Target) RETURN n LIMIT 10000").data()

        # Batch index into Weaviate
        vector_store._batch_index_entities_optimized(drugs, "Drug", None)
        vector_store._batch_index_entities_optimized(diseases, "Disease", None)
        vector_store._batch_index_entities_optimized(targets, "Protein", None)

if __name__ == "__main__":
    populate_weaviate_from_neo4j()
    print("Weaviate population from Neo4j complete.")
