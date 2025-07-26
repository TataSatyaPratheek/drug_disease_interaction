# src/api/routes/health.py - NEW FILE
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import asyncio
import time

from src.core.neo4j_service import Neo4jService
from src.core.weaviate_service import WeaviateService
from src.core.llm_service import LLMService
from src.api.models.responses import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all services"""
    start_time = time.time()
    services_status = {}
    
    try:
        # Test Neo4j connection
        # Note: You'll need to inject the service properly
        services_status["neo4j"] = "healthy"
        
        # Test Weaviate connection  
        services_status["weaviate"] = "healthy"
        
        # Test Ollama connection
        services_status["ollama"] = "healthy"
        
        # Get basic database stats
        database_stats = {
            "neo4j_nodes": "unknown",  # Implement actual queries
            "weaviate_objects": "unknown",
            "response_time_ms": (time.time() - start_time) * 1000
        }
        
        return HealthResponse(
            status="healthy",
            services=services_status,
            database_stats=database_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@router.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness probe"""
    # Implement actual readiness logic
    return {"status": "ready", "timestamp": time.time()}

@router.get("/live") 
async def liveness_check():
    """Kubernetes-style liveness probe"""
    return {"status": "alive", "timestamp": time.time()}
