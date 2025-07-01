# scripts/migrate_to_weaviate.py
import pickle
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.graphrag.core.vector_store import WeaviateGraphStore

def migrate_graph_to_weaviate():
    """Migrate your existing graph to Weaviate v4"""
    
    print("🔄 Starting migration to Weaviate v4...")
    
    # Load your graph
    graph_path = project_root / "data/graph/full_mapped/ddi_knowledge_graph.pickle"
    
    if not graph_path.exists():
        print(f"❌ Graph file not found at {graph_path}")
        return
    
    print(f"📊 Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    
    print(f"✅ Loaded graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
    
    # Initialize Weaviate store
    try:
        vector_store = WeaviateGraphStore()
        print("✅ Weaviate v4 client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Weaviate: {e}")
        return
    
    # Migrate graph (this will take a few minutes)
    print("🚀 Migrating to Weaviate... (this may take 5-10 minutes)")
    try:
        vector_store.initialize_from_graph(graph, force_rebuild=True)
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        vector_store.close()
        return
    
    # Show final statistics
    stats = vector_store.get_statistics()
    print("✅ Migration complete!")
    print("📊 Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # Test search
    print("\n🔍 Testing search functionality...")
    try:
        test_results = vector_store.search_entities("cancer drug", n_results=3)
        print(f"Found {len(test_results)} results for 'cancer drug':")
        for result in test_results:
            print(f"  - {result['name']} ({result['type']}) - Score: {result['similarity_score']:.3f}")
    except Exception as e:
        print(f"❌ Search test failed: {e}")
    
    vector_store.close()
    print("🎉 Migration successful! You can now use Weaviate v4 with your GraphRAG system.")

if __name__ == "__main__":
    migrate_graph_to_weaviate()
