from sentence_transformers import CrossEncoder
# src/core/hybrid_engine.py - UPDATED with proper model loading

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core import QueryBundle
from llama_index.core.response import Response
from typing import List, Dict, Any, Optional
import asyncio
import logging

from src.utils.model_manager import model_manager

logger = logging.getLogger(__name__)

class HybridRAGEngine(BaseQueryEngine):
    """Production Hybrid RAG Engine using LlamaIndex patterns"""
    
    def __init__(
        self,
        neo4j_service,
        weaviate_service,
        llm_service,
        config: Dict[str, Any],
        **kwargs,
    ):
        self.neo4j = neo4j_service
        self.weaviate = weaviate_service
        self.llm = llm_service  
        self.config = config
        super().__init__(callback_manager=kwargs.get("callback_manager"), **kwargs)
        
        # ✅ FIX: Use model manager to handle downloading
        logger.info("Initializing reranker model...")
        self.reranker = model_manager.ensure_reranker_model()
        logger.info("Reranker model ready")

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Sync query method. Standard practice is to wrap the async version."""
        return asyncio.run(self._aquery(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Async query method. This is where the main logic lives."""
        retrieved_results = await self._retrieve_async(query_bundle.query_str)
        
        # Build context string  
        context = "\n".join([r.get('name', '') + ": " + r.get('description', '') for r in retrieved_results[:10]])
        
        # Generate response
        response_text = await self.llm.generate_response(context, query_bundle.query_str)
        
        # Create LlamaIndex Response object
        return Response(
            response=response_text,
            source_nodes=[],
            metadata={
                'retrieved_results': retrieved_results,
                'sources': ['neo4j', 'weaviate'],
                'query': query_bundle.query_str
            }
        )

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Return prompt modules. For a custom engine, this can often be empty."""
        return {}

    async def _retrieve_async(self, query: str) -> List[Dict[str, Any]]:
        """Asynchronously retrieves data from both databases."""
        graph_task = self.neo4j.search_drug_disease_paths(query)
        vector_task = self.weaviate.hybrid_search(query)

        results = await asyncio.gather(graph_task, vector_task, return_exceptions=True)

        graph_results = results[0] if not isinstance(results[0], Exception) else []
        vector_results = results[1] if not isinstance(results[1], Exception) else []

        if isinstance(results[0], Exception):
            logger.error(f"Graph search failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.error(f"Vector search failed: {results[1]}")

        return self._merge_and_rerank(query, graph_results, vector_results)

    def _merge_and_rerank(self, query: str, graph_results: List, vector_results: List) -> List[Dict]:
        """Merges results and uses a CrossEncoder to rerank for relevance."""
        all_candidates = []

        # Prepare Weaviate results
        for res in vector_results:
            all_candidates.append({
                "text": f"{res.get('name', '')}: {res.get('description', '')}",
                "original_result": res
            })

        # Prepare Neo4j path results  
        for res in graph_results:
            if 'node_details' in res:
                path_text = " -> ".join([node['name'] for node in res.get('node_details', [])])
                all_candidates.append({
                    "text": f"Graph Path: {path_text}",
                    "original_result": res
                })

        if not all_candidates:
            return []

        try:
            # Create pairs of [query, passage] for the reranker
            rerank_pairs = [[query, candidate['text']] for candidate in all_candidates]

            # Predict relevance scores
            scores = self.reranker.predict(rerank_pairs)

            # Add scores to the original results and sort
            for i, candidate in enumerate(all_candidates):
                candidate['original_result']['rerank_score'] = scores[i]
                
            reranked_results = sorted(
                [candidate['original_result'] for candidate in all_candidates],
                key=lambda x: x.get('rerank_score', 0.0),
                reverse=True
            )
            return reranked_results[:20]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original results")
            # Fallback: return original results with default scores
            for res in vector_results:
                res['rerank_score'] = res.get('score', 0.5)
            for res in graph_results:
                res['rerank_score'] = 0.8  # Graph results get high default score
            
            all_results = vector_results + graph_results
            return sorted(all_results, key=lambda x: x.get('rerank_score', 0.0), reverse=True)[:20]
