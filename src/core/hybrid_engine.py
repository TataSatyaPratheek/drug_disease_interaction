# src/core/hybrid_engine.py - USE LLAMAINDEX QUERY ENGINE PATTERN
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.response import Response
from typing import List, Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class HybridRAGEngine(BaseQueryEngine):
    """
    Production Hybrid RAG Engine using LlamaIndex patterns
    Orchestrates Neo4j (graph) + Weaviate (vector) + Ollama (LLM)
    """

    def __init__(
        self,
        neo4j_service,
        weaviate_service,
        llm_service,
        config: Dict[str, Any]
    ):
        self.neo4j = neo4j_service
        self.weaviate = weaviate_service
        self.llm = llm_service
        self.config = config

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

        return self._merge_and_rerank(graph_results, vector_results)

    def _merge_and_rerank(self, graph_results: List, vector_results: List) -> List[Dict]:
        """
        Merges results, normalizes scores, and re-ranks.
        This is a simple example; a real implementation might use a more advanced re-ranking model.
        """
        merged = []

        # Normalize Weaviate scores (assuming they are between 0 and 1)
        for result in vector_results:
            result['normalized_score'] = result.get('score', 0.0)
            merged.append(result)

        # Assign a high base score to graph results and add them
        graph_base_score = 0.8
        for result in graph_results:
            # Create a unique ID for the path to avoid duplicates
            path_id = "-".join(sorted([node['id'] for node in result.get('node_details', [])]))
            if not any(item.get('id') == path_id for item in merged):
                merged.append({
                    'id': path_id,
                    'name': f"Interaction Path: {result['node_details'][0]['name']} to {result['node_details'][-1]['name']}",
                    'normalized_score': graph_base_score,
                    'metadata': result,
                    'source': 'neo4j'
                })

        # Sort by the new normalized score
        return sorted(merged, key=lambda x: x['normalized_score'], reverse=True)[:20]

    async def aquery(self, query_bundle: QueryBundle) -> Response:
        retrieved_results = await self._retrieve_async(query_bundle.query_str)
        # Build context string
        context = "\n".join([r.get('name', r.get('content', '')) for r in retrieved_results[:10]])
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
