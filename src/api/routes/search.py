# src/api/routes/search.py - USE FASTAPI ROUTER

# src/api/routes/search.py - UPDATED

from fastapi import APIRouter, HTTPException, Depends
from llama_index.core import QueryBundle
import time
import logging

from fastapi_cache.decorator import cache
from fastapi.responses import StreamingResponse

# ...existing code...

router = APIRouter()

@router.post("/search/stream")
async def hybrid_search_stream(
    request: HybridSearchRequest,
    engine = Depends(HybridEngineDep)
):
    """
    ✅ NEW FEATURE: Performs a hybrid search and streams the LLM's response token-by-token.
    """
    try:
        # Step 1: Retrieve context (this part is not streamed)
        retrieved_results = await engine._retrieve_async(request.query)
        context = "\n".join([r.get('name', '') + ": " + r.get('description', '') for r in retrieved_results[:10]])

        # Step 2: Define an async generator for the streaming response
        async def stream_generator():
            prompt = f"""
            Based on the following context, answer the question.
            Context: {context}
            Question: {request.query}
            Answer:"""
            # Use the streaming method of the LlamaIndex LLM
            stream = await engine.llm.llm.astream_complete(prompt)
            async for delta in stream:
                yield delta.delta

        # Step 3: Return a StreamingResponse
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Streaming search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Streaming search failed.")

from src.api.models.requests import HybridSearchRequest, EntityDetailsRequest
from src.api.models.responses import HybridSearchResponse, EntityResult
from src.api.dependencies import HybridEngineDep, Neo4jDep


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/search", response_model=HybridSearchResponse)
@cache(expire=3600)  # Cache results for this endpoint for 1 hour
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
