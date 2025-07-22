# src/graphrag/frontend/config.py
"""Application configuration constants and settings"""

# Model Configuration
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
AVAILABLE_MODELS = ["qwen3:1.7b", "llama3.2:latest", "nomic-embed-text:latest"]

# UI Configuration
MAX_VECTOR_RESULTS = 50
DEFAULT_VECTOR_RESULTS = 15
AI_TEMPERATURE_RANGE = (0.0, 1.0)
DEFAULT_AI_TEMPERATURE = 0.3

# Query Configuration
QUERY_TYPES = {
    "auto": "Auto-detect",
    "drug_comparison": "Drug Comparison",
    "mechanism": "Mechanism Analysis",
    "safety": "Safety Profile",
    "repurposing": "Drug Repurposing",
    "interaction": "Drug Interactions",
}

# UI Constants
APP_TITLE = "🧬 Drug-Disease GraphRAG System"
APP_SUBTITLE = "**Powered by Weaviate, Ollama, and NetworkX**"

# Example Queries
EXAMPLE_QUERIES = [
    "Compare the side effects of ACE inhibitors vs ARBs",
    "What is the mechanism of action of metformin?",
    "Find potential repurposing opportunities for aspirin",
    "What proteins does ibuprofen target?",
    "Explain the relationship between diabetes and cardiovascular disease",
]

# Visualization Settings
MAX_NODES_VISUALIZATION = 100
DEFAULT_GRAPH_LAYOUT = "spring"
VISUALIZATION_WIDTH = 800
VISUALIZATION_HEIGHT = 600

# Error Messages
ERROR_MESSAGES = {
    "query_processing_failed": "❌ Query processing failed. Please check your input and try again.",
    "engine_not_ready": "❌ Query engine not ready. Please wait for system initialization.",
    "backend_error": "❌ Backend processing error. Please try again.",
    "ollama_not_available": "❌ Ollama server not available. Please ensure the service is running.",
    "weaviate_not_available": "❌ Weaviate database not available. Please ensure the service is running.",
    "initialization_failed": "❌ System initialization failed. Check logs for details.",
    "query_failed": "❌ Query processing failed. Please check the inputs and try again.",
}

# Success Messages
SUCCESS_MESSAGES = {
    "system_ready": "✅ System ready! Graph: {nodes:,} nodes, Vector DB: {entities:,} entities.",
    "cleanup_complete": "🧹 Resources cleaned up successfully.",
    "query_complete": "✅ Query processed successfully.",
    "query_submitted": "📤 Query submitted for processing...",
    "backend_ready": "🚀 Backend services are ready.",
}

# Add privacy notice
PRIVACY_NOTICE = """
🔒 **Complete Local Privacy**: This system operates entirely on your local device. 
No data leaves your machine. All models run locally without internet connections.
"""