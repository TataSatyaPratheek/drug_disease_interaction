# src/graphrag/generators/response_builder.py
from typing import Dict, List, Any

class ResponseBuilder:
    """Enhanced response builder with qwen3 reasoning support"""
    
    def build_response(self,
                   query: str,
                   llm_response: str,
                   retrieved_data: Dict[str, Any],
                   subgraph_context: str,
                   query_type: str = "general",
                   confidence_score: float = 0.5,
                   reasoning: str = "",
                   suggested_followups: List[str] = None) -> Dict[str, Any]:  # Add this parameter
        """Build comprehensive response with all components"""
        
        # Extract citations from retrieved data
        citations = self._extract_citations(retrieved_data)
        
        # Generate follow-up questions if not provided
        if suggested_followups is None:
            suggested_followups = self._generate_followup_questions(query, query_type)
        
        response = {
            "response": llm_response,
            "reasoning": reasoning,
            "retrieved_data": retrieved_data,
            "citations": citations,
            "suggested_followups": suggested_followups,  # Include the parameter
            "subgraph_context": subgraph_context,
            "confidence_score": confidence_score,
            "query_type": query_type,
            "metadata": {
                "entities_found": sum(len(v) if isinstance(v, list) else 0 for v in retrieved_data.values()),
                "citations_count": len(citations),
                "context_length": len(subgraph_context)
            }
        }
        
        return response

    def _generate_followup_questions(self, query: str, query_type: str) -> List[str]:
        """Generate follow-up questions based on query type"""
        base_followups = {
            "comparison": [
                "What are the mechanisms behind these differences?",
                "Which patient populations benefit most from each option?",
                "Are there any contraindications to consider?"
            ],
            "mechanism": [
                "What are the downstream effects of this mechanism?",
                "Are there alternative pathways involved?",
                "How does genetic variation affect this pathway?"
            ],
            "safety": [
                "What is the frequency of these adverse effects?",
                "Are there drug interactions to be aware of?",
                "What monitoring is recommended?"
            ],
            "general": [
                "What are the clinical implications of these findings?",
                "Are there any recent research developments?",
                "What additional factors should be considered?"
            ]
        }
        
        return base_followups.get(query_type, base_followups["general"])[:3]

    def _extract_citations(self, retrieved_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract citations from retrieved data"""
        citations = []
        
        for entity_type in ['drugs', 'diseases', 'proteins']:
            if entity_type in retrieved_data:
                for entity in retrieved_data[entity_type]:
                    citations.append({
                        "type": entity_type[:-1],
                        "id": entity.get('id', ''),
                        "name": entity.get('name', ''),
                        "source": "Knowledge Graph",
                        "confidence": entity.get('similarity_score', 0.0)
                    })
        
        return citations
    
    def _extract_entities(self, retrieved_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key entities mentioned in the response"""
        entities = []
        
        for entity_type in ['drugs', 'diseases', 'proteins']:
            if entity_type in retrieved_data:
                for entity in retrieved_data[entity_type]:
                    entities.append({
                        "type": entity_type[:-1],
                        "id": entity.get('id', ''),
                        "name": entity.get('name', ''),
                        "relevance_score": entity.get('similarity_score', 0.0)
                    })
        
        return sorted(entities, key=lambda x: x['relevance_score'], reverse=True)
    