# src/api/routes/health.py - NEW FILE

# src/api/routes/health.py - IMPLEMENTED
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import asyncio
import time
from src.api.dependencies import Neo4jDep, WeaviateDep, LLMDep
from src.api.models.responses import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Comprehensive health check")
async def health_check(
    neo4j: Neo4jDep,
    weaviate: WeaviateDep,
    llm: LLMDep
):
    """Checks the status of all downstream services."""
    start_time = time.time()
    
    async def check_neo4j():
        try:
            await neo4j.driver.verify_connectivity()
            return "ok"
        except Exception:
            return "error"

    async def check_weaviate():
        return "ok" if weaviate.client.is_ready() else "error"

    async def check_ollama():
        return "ok" if await llm.check_connection() else "error"

    results = await asyncio.gather(check_neo4j(), check_weaviate(), check_ollama())
    
    services_status = {
        "neo4j": results[0],
        "weaviate": results[1],
        "ollama": results[2]
    }
    
    overall_status = "healthy" if all(s == "ok" for s in services_status.values()) else "degraded"
    
    if overall_status == "degraded":
        raise HTTPException(status_code=503, detail={"status": overall_status, "services": services_status})
    
    return HealthResponse(
        status=overall_status,
        services=services_status,
        database_stats={"response_time_ms": (time.time() - start_time) * 1000}
    )

@router.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness probe"""
    # Implement actual readiness logic
    return {"status": "ready", "timestamp": time.time()}

@router.get("/live") 
async def liveness_check():
    """Kubernetes-style liveness probe"""
    return {"status": "alive", "timestamp": time.time()}
