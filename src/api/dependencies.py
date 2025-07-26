# src/api/dependencies.py - NEW FILE  
from fastapi import Depends, HTTPException
from functools import lru_cache
from typing import Annotated

from src.core.hybrid_engine import HybridRAGEngine
from src.core.neo4j_service import Neo4jService
from src.core.weaviate_service import WeaviateService
from src.core.llm_service import LLMService
from src.utils.config import settings

# Global service instances (initialized in lifespan)
_neo4j_service: Neo4jService = None
_weaviate_service: WeaviateService = None  
_llm_service: LLMService = None
_hybrid_engine: HybridRAGEngine = None

def set_services(neo4j: Neo4jService, weaviate: WeaviateService, llm: LLMService, engine: HybridRAGEngine):
    """Called during FastAPI lifespan to set service instances"""
    global _neo4j_service, _weaviate_service, _llm_service, _hybrid_engine
    _neo4j_service = neo4j
    _weaviate_service = weaviate
    _llm_service = llm
    _hybrid_engine = engine

def get_neo4j_service() -> Neo4jService:
    if _neo4j_service is None:
        raise HTTPException(status_code=503, detail="Neo4j service not ready")
    return _neo4j_service

def get_weaviate_service() -> WeaviateService:
    if _weaviate_service is None:
        raise HTTPException(status_code=503, detail="Weaviate service not ready")
    return _weaviate_service

def get_llm_service() -> LLMService:
    if _llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not ready")
    return _llm_service

def get_hybrid_engine() -> HybridRAGEngine:
    if _hybrid_engine is None:
        raise HTTPException(status_code=503, detail="Hybrid engine not ready")
    return _hybrid_engine

# Type aliases for cleaner dependency injection
Neo4jDep = Annotated[Neo4jService, Depends(get_neo4j_service)]
WeaviateDep = Annotated[WeaviateService, Depends(get_weaviate_service)]
LLMDep = Annotated[LLMService, Depends(get_llm_service)]
HybridEngineDep = Annotated[HybridRAGEngine, Depends(get_hybrid_engine)]
