import pickle
import sys
from pathlib import Path
import logging
import weaviate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def migrate_with_direct_connection():
    """Migrate using direct Docker Weaviate connection"""
    print("🔄 Starting migration to Docker Weaviate...")
    
    # Load graph
    graph_path = project_root / "data/graph/full_mapped/ddi_knowledge_graph.pickle"
    print(f"📊 Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    print(f"✅ Loaded graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")

    # Direct connection to Docker Weaviate
    print("🔌 Connecting directly to Docker Weaviate...")
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        client.is_ready()
        print("✅ Connected to Docker Weaviate on port 8080")
    except Exception as e:
        print(f"❌ Failed to connect to Docker Weaviate: {e}")
        return

    # Create vector store with direct client
    print("🔧 Initializing vector store...")
    from graphrag.core.vector_store import WeaviateGraphStore
    
    # Temporarily patch the connection manager to use our client
    vector_store = WeaviateGraphStore.__new__(WeaviateGraphStore)
    vector_store.client = client
    vector_store.persist_directory = Path("data/weaviate_db")
    vector_store.collections = {}
    
    # Initialize embedding model
    from sentence_transformers import SentenceTransformer
    vector_store.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_store.logger = logging.getLogger("WeaviateGraphStore")
    
    print("🚀 Starting data migration...")
    try:
        vector_store.initialize_from_graph(graph, force_rebuild=True)
        
        # Show statistics
        stats = vector_store.get_statistics()
        print("✅ Migration complete!")
        print("📊 Final statistics:")
        for key, value in stats.items():
            if isinstance(value, int):
                print(f"   {key}: {value:,}")
                
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    migrate_with_direct_connection()
