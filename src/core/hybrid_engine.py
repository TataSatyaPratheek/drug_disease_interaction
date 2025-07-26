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
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[Dict[str, Any]]:
        """Hybrid retrieval using both graph and vector databases"""
        query = query_bundle.query_str
        
        # Run concurrent retrieval (DON'T REINVENT CONCURRENCY)
        loop = asyncio.get_event_loop()
        graph_task = self.neo4j.search_drug_disease_paths(query)
        vector_task = self.weaviate.hybrid_search(query)
        
        graph_results, vector_results = loop.run_until_complete(
            asyncio.gather(graph_task, vector_task, return_exceptions=True)
        )
        
        # Handle exceptions
        if isinstance(graph_results, Exception):
            logger.warning(f"Graph search failed: {graph_results}")
            graph_results = []
        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed: {vector_results}")
            vector_results = []
        
        return self._merge_results(graph_results, vector_results)
    
    def _merge_results(self, graph_results: List, vector_results: List) -> List[Dict]:
        """Merge and rank results from both sources"""
        merged = []
        
        # Process graph results
        for result in graph_results:
            merged.append({
                'content': f"Graph path: {result.get('node_details', [])}",
                'source': 'neo4j',
                'score': 0.8,  # Graph results get high relevance
                'metadata': result
            })
        
        # Process vector results
        for result in vector_results:
            merged.append({
                'content': f"{result['name']}: {result['description']}",
                'source': 'weaviate',
                'score': result['score'],
                'metadata': result
            })
        
        # Sort by score
        return sorted(merged, key=lambda x: x['score'], reverse=True)[:20]
    
    def _query(self, query_bundle: QueryBundle) -> Response:
        """Execute hybrid query using LlamaIndex Response pattern"""
        # Retrieve relevant context
        retrieved_results = self._retrieve(query_bundle)
        
        # Build context string
        context = "\n".join([r['content'] for r in retrieved_results[:10]])
        
        # Generate response
        loop = asyncio.get_event_loop()
        response_text = loop.run_until_complete(
            self.llm.generate_response(context, query_bundle.query_str)
        )
        
        # Create LlamaIndex Response object
        return Response(
            response=response_text,
            source_nodes=[],  # Could create NodeWithScore objects if needed
            metadata={
                'retrieved_results': retrieved_results,
                'sources': ['neo4j', 'weaviate'],
                'query': query_bundle.query_str
            }
        )
    
    async def aquery(self, query_bundle: QueryBundle) -> Response:
        """Async version of query"""
        return self._query(query_bundle)
