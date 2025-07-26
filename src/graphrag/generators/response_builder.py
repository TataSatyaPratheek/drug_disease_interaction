# src/graphrag/generators/response_builder.py
from typing import Dict, List, Any, Optional
import concurrent.futures
import threading
import time
import hashlib
from functools import lru_cache

class ResponseBuilder:
    """Enhanced response builder optimized for Ryzen 4800H performance"""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 128):
        self.max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._cache_lock = threading.Lock()
        
        # LRU cache for response components
        self._followup_cache = {}
        self._citation_cache = {}
        
    @lru_cache(maxsize=128)
    def _get_cached_followups(self, query_hash: str, query_type: str) -> tuple:
        """Cached follow-up question generation."""
        return tuple(self._generate_followup_questions_uncached(query_hash, query_type))
    
    def build_response(self,
                   query: str,
                   llm_response: str,
                   retrieved_data: Dict[str, Any],
                   subgraph_context: str,
                   query_type: str = "general",
                   confidence_score: float = 0.5,
                   reasoning: str = "",
                   suggested_followups: List[str] = None) -> Dict[str, Any]:
        """Build comprehensive response with parallel processing"""
        
        # Create query hash for caching
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Process components in parallel for better CPU utilization
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit parallel tasks
            citation_future = executor.submit(self._extract_citations, retrieved_data)
            entity_future = executor.submit(self._extract_entities, retrieved_data)
            
            if suggested_followups is None:
                followup_future = executor.submit(self._generate_followup_questions, query, query_type)
            else:
                followup_future = None
            
            # Collect results
            citations = citation_future.result()
            entities = entity_future.result()
            
            if followup_future:
                suggested_followups = followup_future.result()
        
        # Build response object
        response = {
            "response": llm_response,
            "reasoning": reasoning,
            "retrieved_data": retrieved_data,
            "citations": citations,
            "entities": entities,  # Add entities to response
            "suggested_followups": suggested_followups,
            "subgraph_context": subgraph_context,
            "confidence_score": confidence_score,
            "query_type": query_type,
            "metadata": {
                "entities_found": len(entities),
                "citations_count": len(citations),
                "context_length": len(subgraph_context),
                "processing_time": time.time(),
                "query_hash": query_hash
            }
        }
        
        return response

    def build_responses_batch(self, 
                            queries_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build multiple responses in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.build_response, **query_data) 
                for query_data in queries_data
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "error": f"Response building failed: {e}",
                        "response": "Error in response generation",
                        "metadata": {"error": True}
                    })
            
            return results

    def _generate_followup_questions(self, query: str, query_type: str) -> List[str]:
        """Generate follow-up questions with caching for performance."""
        query_hash = hashlib.md5(f"{query}_{query_type}".encode()).hexdigest()
        
        # Check cache first
        cached = self._get_cached_followups(query_hash, query_type)
        if cached:
            return list(cached)
        
        return self._generate_followup_questions_uncached(query, query_type)

    def _generate_followup_questions_uncached(self, query: str, query_type: str) -> List[str]:
        """Generate follow-up questions based on query type - uncached version"""
        base_followups = {
            "comparison": [
                "What are the mechanisms behind these differences?",
                "Which patient populations benefit most from each option?",
                "Are there any contraindications to consider?",
                "What are the cost-effectiveness differences?",
                "How do biomarkers influence treatment choice?"
            ],
            "mechanism": [
                "What are the downstream effects of this mechanism?",
                "Are there alternative pathways involved?",
                "How does genetic variation affect this pathway?",
                "What are the off-target effects?",
                "How does this mechanism relate to drug resistance?"
            ],
            "safety": [
                "What is the frequency of these adverse effects?",
                "Are there drug interactions to be aware of?",
                "What monitoring is recommended?",
                "How do adverse effects vary by population?",
                "What are the long-term safety considerations?"
            ],
            "drug_comparison": [
                "How do efficacy profiles compare?",
                "What are the pharmacokinetic differences?",
                "Which has better tolerability?",
                "Are there biomarkers for response prediction?",
                "What are the economic considerations?"
            ],
            "repurposing": [
                "What is the biological rationale for repurposing?",
                "Are there clinical trials supporting this use?",
                "What are the regulatory considerations?",
                "How does dosing differ for the new indication?",
                "What are the potential risks and benefits?"
            ],
            "general": [
                "What are the clinical implications of these findings?",
                "Are there any recent research developments?",
                "What additional factors should be considered?",
                "How does this apply to different patient populations?",
                "What are the practical implementation considerations?"
            ]
        }
        
        followups = base_followups.get(query_type, base_followups["general"])
        
        # Return 3-4 most relevant questions instead of just 3
        return followups[:4]

    def _extract_citations(self, retrieved_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract citations from retrieved data with optimized processing"""
        citations = []
        
        # Process entity types in parallel if dataset is large
        entity_types = ['drugs', 'diseases', 'proteins', 'pathways', 'targets']
        
        for entity_type in entity_types:
            if entity_type in retrieved_data and retrieved_data[entity_type]:
                entities = retrieved_data[entity_type]
                
                # Batch process entities for better performance
                for entity in entities[:10]:  # Limit to top 10 for performance
                    citations.append({
                        "type": entity_type[:-1] if entity_type.endswith('s') else entity_type,
                        "id": entity.get('id', ''),
                        "name": entity.get('name', entity.get('title', '')),
                        "source": entity.get('source', 'Knowledge Graph'),
                        "confidence": float(entity.get('similarity_score', entity.get('score', 0.0))),
                        "context": entity.get('description', '')[:200] if entity.get('description') else ''
                    })
        
        # Sort by confidence and return top results
        return sorted(citations, key=lambda x: x['confidence'], reverse=True)[:15]
    
    def _extract_entities(self, retrieved_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key entities with enhanced metadata"""
        entities = []
        
        entity_types = ['drugs', 'diseases', 'proteins', 'pathways', 'targets']
        
        for entity_type in entity_types:
            if entity_type in retrieved_data and retrieved_data[entity_type]:
                for entity in retrieved_data[entity_type][:8]:  # Top 8 per type
                    entity_info = {
                        "type": entity_type[:-1] if entity_type.endswith('s') else entity_type,
                        "id": entity.get('id', ''),
                        "name": entity.get('name', entity.get('title', '')),
                        "relevance_score": float(entity.get('similarity_score', entity.get('score', 0.0))),
                        "properties": {
                            "description": entity.get('description', '')[:300],
                            "synonyms": entity.get('synonyms', [])[:5],  # Limit synonyms
                            "external_ids": entity.get('external_ids', {}),
                        }
                    }
                    
                    # Add type-specific properties
                    if entity_type == 'drugs':
                        entity_info["properties"].update({
                            "mechanism": entity.get('mechanism_of_action', ''),
                            "indication": entity.get('indication', ''),
                            "drug_type": entity.get('drug_type', '')
                        })
                    elif entity_type == 'diseases':
                        entity_info["properties"].update({
                            "category": entity.get('category', ''),
                            "symptoms": entity.get('symptoms', [])[:3]
                        })
                    
                    entities.append(entity_info)
        
        return sorted(entities, key=lambda x: x['relevance_score'], reverse=True)[:20]

    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
    