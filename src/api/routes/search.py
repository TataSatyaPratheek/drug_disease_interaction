# src/api/routes/search.py - USE FASTAPI ROUTER

# src/api/routes/search.py - UPDATED
from fastapi import APIRouter, HTTPException
from llama_index.core import QueryBundle
import time
import logging

from src.api.models.requests import HybridSearchRequest, EntityDetailsRequest
from src.api.models.responses import HybridSearchResponse, EntityResult
from src.api.dependencies import HybridEngineDep, Neo4jDep


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    engine: HybridEngineDep
):
    """Main hybrid search endpoint using Neo4j + Weaviate + LlamaIndex"""
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
                id=result.get('id', 'unknown'),
                name=result.get('name', 'Unknown'),
                type=result.get('collection', result.get('source', 'Entity')),
                score=result.get('score', 0.0),
                description=result.get('description', ''),
                source=result.get('source', 'unknown')
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
    neo4j_service: Neo4jDep,
    limit: int = 10
):
    """Get search suggestions from Neo4j data"""
    try:
        suggestions = await neo4j_service.search_drug_disease_paths(query, limit)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
