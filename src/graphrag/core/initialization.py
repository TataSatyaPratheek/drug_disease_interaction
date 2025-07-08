import streamlit as st
import pickle
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Get project root dynamically
project_root = Path(__file__).parent.parent.parent.parent  

@st.cache_resource
def load_graph_data():
    """Load NetworkX graph from pickle file"""
    try:
        graph_path = project_root / "data/graph/full_mapped/ddi_knowledge_graph.pickle"
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        
        logger.info(f"✅ Loaded graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
        return graph
    except Exception as e:
        logger.error(f"Failed to load graph data: {e}")
        raise

@st.cache_resource
def get_weaviate_connection():
    """Initialize and cache Weaviate vector store connection"""
    try:
        from graphrag.core.vector_store import WeaviateGraphStore
        vector_store = WeaviateGraphStore()
        
        # Store in session state for cleanup
        st.session_state.vector_store = vector_store
        
        logger.info("✅ Weaviate connection established")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise

@st.cache_resource
def get_ollama_client(model_name: str = "qwen3:1.7b"):
    """Initialize and cache Ollama LLM client"""
    try:
        from graphrag.generators.llm_client import OllamaClient
        client = OllamaClient(model_name=model_name)
        
        logger.info(f"✅ Ollama client initialized with model: {model_name}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        raise

@st.cache_resource
def verify_weaviate_collections(_vector_store) -> bool:
    """Verify that required Weaviate collections exist"""
    try:
        required_collections = ["Drug", "Disease", "Protein", "Relationship"]
        
        for collection_name in required_collections:
            if not _vector_store.client.collections.exists(collection_name):
                logger.error(f"Missing collection: {collection_name}")
                return False
        
        logger.info("✅ All required Weaviate collections verified")
        return True
    except Exception as e:
        logger.error(f"Collection verification failed: {e}")
        return False

def initialize_graphrag_system() -> Tuple:
    """Initialize the complete GraphRAG system with caching"""
    try:
        # Load components in dependency order
        vector_store = get_weaviate_connection()
        graph = load_graph_data()
        llm_client = get_ollama_client()
        
        # Verify system readiness
        if not verify_weaviate_collections(vector_store):
            raise ValueError("Weaviate collections missing! Run migration script.")
        
        # Initialize query engine
        from graphrag.core.query_engine import GraphRAGQueryEngine
        engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
        
        logger.info("✅ GraphRAG system initialization complete")
        return graph, vector_store, llm_client, engine
        
    except Exception as e:
        logger.error(f"GraphRAG system initialization failed: {e}")
        # Cleanup on failure
        if 'vector_store' in locals():
            try:
                vector_store.close()
            except:
                pass
        raise
