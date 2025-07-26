# src/api/main.py - USE FASTAPI (DON'T REINVENT WEB FRAMEWORK)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from src.api.routes.search import router as search_router
from src.api.routes.health import router as health_router
from src.utils.config import settings
from src.utils.logging import setup_logging
from src.core.hybrid_engine import HybridRAGEngine
from src.core.neo4j_service import Neo4jService
from src.core.weaviate_service import WeaviateService
from src.core.llm_service import LLMService

# Global services
hybrid_engine = None
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global hybrid_engine
    
    # Startup
    logger.info("🚀 Starting Hybrid RAG API...")
    
    # Initialize services using your existing optimized setup
    neo4j_service = Neo4jService(
        settings.database.neo4j_uri,
        settings.database.neo4j_user,
        settings.database.neo4j_password,
        max_workers=settings.hardware['threading_config']['max_workers']
    )
    
    weaviate_service = WeaviateService(settings.database.weaviate_url)
    llm_service = LLMService(settings.hardware)
    
    hybrid_engine = HybridRAGEngine(
        neo4j_service, weaviate_service, llm_service, settings.hardware
    )
    
    logger.info("✅ All services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down services...")
    neo4j_service.close()
    weaviate_service.close()

app = FastAPI(
    title="Drug-Disease Interaction Hybrid RAG API",
    description="Production API using Neo4j + Weaviate + LlamaIndex + Ollama",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router, prefix="/api/v1", tags=["Search"])
app.include_router(health_router, prefix="/api/v1", tags=["Health"])

def get_hybrid_engine():
    """Dependency to get hybrid engine"""
    if hybrid_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return hybrid_engine

# Make it available for routes
app.dependency_overrides[HybridRAGEngine] = get_hybrid_engine

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        workers=1  # Single worker for your setup
    )
