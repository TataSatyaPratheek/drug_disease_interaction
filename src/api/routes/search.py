# src/api/routes/search.py - USE FASTAPI ROUTER
from fastapi import APIRouter, Depends, HTTPException
from llama_index.core import QueryBundle
import time
import logging

from src.api.models.requests import HybridSearchRequest, EntityDetailsRequest
from src.api.models.responses import HybridSearchResponse, EntityResult
from src.core.hybrid_engine import HybridRAGEngine

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    engine: HybridRAGEngine = Depends()
):
    """
    Main hybrid search endpoint using your Neo4j + Weaviate data
    Optimized for Ryzen 4800H + GTX 1650Ti
    """
    try:
        start_time = time.time()
        
        # Create LlamaIndex QueryBundle
        query_bundle = QueryBundle(query_str=request.query)
        
        # Execute hybrid search
        response = await engine.aquery(query_bundle)
        
        # Extract entities from metadata
        entities = []
        retrieved_results = response.metadata.get('retrieved_results', [])
        
        for result in retrieved_results[:request.max_results]:
            entities.append(EntityResult(
                id=result['metadata'].get('id', 'unknown'),
                name=result['metadata'].get('name', 'Unknown'),
                type=result['metadata'].get('collection', result['metadata'].get('type', 'Entity')),
                score=result['score'],
                description=result['metadata'].get('description', ''),
                source=result['source']
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return HybridSearchResponse(
            query=request.query,
            answer=response.response,
            entities=entities,
            total_found=len(entities),
            processing_time_ms=processing_time,
            sources_used=response.metadata.get('sources', [])
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/suggest")
async def get_suggestions(
    query: str,
    limit: int = 10,
    engine: HybridRAGEngine = Depends()
):
    """Get search suggestions from your Neo4j data"""
    try:
        # Use Neo4j for fast entity name lookups
        suggestions = await engine.neo4j.search_drug_disease_paths(query, limit)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
