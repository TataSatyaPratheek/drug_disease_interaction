"""
Neo4j-native GraphRAG Query Engine
Streamlined implementation optimized for Ryzen 4800H + GTX 1650Ti

Removes dependencies on NetworkX, igraph, and complex retrievers.
Uses Neo4j's built-in graph algorithms and optimized threading.
"""

import logging
import time
import concurrent.futures
from typing import Dict, List, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class Neo4jGraphRAGEngine:
    """Streamlined GraphRAG engine using Neo4j native capabilities"""
    
    def __init__(
        self, 
        neo4j_driver=None,
        vector_store=None,
        llm_client=None,
        max_workers: int = 4,
        enable_caching: bool = True,
        cache_size: int = 128
    ):
        """Initialize with Neo4j driver, Weaviate store, and optimized LLM client"""
        self.neo4j_driver = neo4j_driver
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        # Performance optimizations for Ryzen 4800H
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Response cache for repeated queries
        if enable_caching:
            self._cached_query = lru_cache(maxsize=cache_size)(self._execute_query)
        
        logger.info(f"✅ Neo4jGraphRAGEngine initialized with {max_workers} workers, caching: {enable_caching}")
    
    def query(self, user_query: str, max_results: int = 10) -> Dict[str, Any]:
        """Main query interface - handles caching and routing"""
        if self.enable_caching:
            return self._cached_query(user_query, max_results)
        else:
            return self._execute_query(user_query, max_results)
    
    def _execute_query(self, user_query: str, max_results: int) -> Dict[str, Any]:
        """Execute the actual query processing"""
        start_time = time.time()
        logger.info(f"Processing query: '{user_query}'")
        
        try:
            # Step 1: Parallel entity retrieval
            with self._executor:
                # Submit parallel tasks for better performance
                vector_future = self._executor.submit(
                    self._search_vector_entities, user_query, max_results
                )
                neo4j_future = self._executor.submit(
                    self._search_neo4j_context, user_query
                )
                
                # Collect results
                entities = vector_future.result(timeout=10)
                neo4j_context = neo4j_future.result(timeout=8)
            
            # Step 2: Build enriched context
            enriched_context = self._build_context(entities, neo4j_context, user_query)
            
            # Step 3: Generate LLM response
            response_text = self.llm_client.generate_response(
                self._create_prompt(user_query, enriched_context)
            )
            
            # Step 4: Build final response
            processing_time = time.time() - start_time
            result = {
                'response': response_text,
                'entities': entities,
                'neo4j_context': neo4j_context,
                'confidence': self._calculate_confidence(entities),
                'processing_time': round(processing_time, 3),
                'query': user_query
            }
            
            logger.info(f"Query completed in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'response': f"I encountered an error processing your query: {str(e)}",
                'entities': [],
                'confidence': 0.0,
                'error': str(e),
                'query': user_query
            }
    
    def _search_vector_entities(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for entities using Weaviate vector similarity"""
        try:
            results = self.vector_store.search_entities(
                query,
                entity_types=["Drug", "Disease", "Protein"],
                n_results=max_results
            )
            logger.info(f"Vector search found {len(results)} entities")
            return results
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
    
    def _search_neo4j_context(self, query: str) -> Dict[str, Any]:
        """Get contextual information from Neo4j using native queries"""
        try:
            with self.neo4j_driver.session() as session:
                # Use Neo4j's native full-text search and graph traversal
                cypher_query = """
                CALL db.index.fulltext.queryNodes('entitySearch', $query)
                YIELD node, score
                WITH node, score
                LIMIT 10
                OPTIONAL MATCH (node)-[r]-(connected)
                RETURN node, collect(distinct {rel: type(r), connected: connected.name})[..5] as relationships,
                       score
                ORDER BY score DESC
                """
                
                result = session.run(cypher_query, query=query)
                
                context_data = []
                for record in result:
                    node_data = dict(record["node"])
                    node_data["relationships"] = record["relationships"]
                    node_data["search_score"] = record["score"]
                    context_data.append(node_data)
                
                logger.info(f"Neo4j context search found {len(context_data)} nodes")
                return {"nodes": context_data}
                
        except Exception as e:
            logger.warning(f"Neo4j context search failed: {e}")
            return {"nodes": []}
    
    def _build_context(self, entities: List[Dict], neo4j_context: Dict, query: str) -> str:
        """Build enriched context from both vector and graph data"""
        context_parts = [f"Query: {query}\n"]
        
        # Add vector entities
        if entities:
            context_parts.append(f"Found {len(entities)} relevant entities:")
            for entity in entities[:5]:  # Top 5
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'unknown')
                similarity = entity.get('similarity', 0)
                context_parts.append(f"- {name} ({entity_type}) - Relevance: {similarity:.3f}")
        
        # Add Neo4j graph context
        neo4j_nodes = neo4j_context.get("nodes", [])
        if neo4j_nodes:
            context_parts.append(f"\nGraph relationships found:")
            for node in neo4j_nodes[:3]:  # Top 3
                node_name = node.get('name', 'Unknown')
                relationships = node.get('relationships', [])
                if relationships:
                    rel_text = ", ".join([f"{r['rel']} {r['connected']}" for r in relationships[:3]])
                    context_parts.append(f"- {node_name}: {rel_text}")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for the LLM"""
        return f"""Based on the biomedical knowledge graph data below, please answer the query.

Query: {query}

Knowledge Graph Context:
{context}

Please provide a comprehensive answer based on the entities and relationships found in the graph. Focus on the specific data retrieved rather than general knowledge.

Answer:"""
    
    def _calculate_confidence(self, entities: List[Dict]) -> float:
        """Calculate confidence score based on entity relevance"""
        if not entities:
            return 0.0
        
        # Average similarity scores
        similarities = [e.get('similarity', 0) for e in entities]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Boost confidence with entity count (more entities = higher confidence)
        entity_boost = min(len(entities) / 10, 0.3)  # Max 0.3 boost
        
        return min(avg_similarity + entity_boost, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics"""
        stats = {
            "max_workers": self.max_workers,
            "caching_enabled": self.enable_caching,
            "executor_active": hasattr(self, '_executor') and not self._executor._shutdown
        }
        
        if self.enable_caching and hasattr(self, '_cached_query'):
            cache_info = self._cached_query.cache_info()
            stats.update({
                "cache_hits": cache_info.hits,
                "cache_misses": cache_info.misses,
                "cache_size": cache_info.currsize,
                "cache_maxsize": cache_info.maxsize
            })
        
        return stats
    
    def clear_cache(self):
        """Clear the query cache"""
        if self.enable_caching and hasattr(self, '_cached_query'):
            self._cached_query.cache_clear()
            logger.info("Query cache cleared")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logger.info("Neo4jGraphRAGEngine executor shut down")
