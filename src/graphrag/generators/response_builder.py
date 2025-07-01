# src/graphrag/generators/response_builder.py
from typing import Dict, List, Any, Optional
import json

class ResponseBuilder:
    """Enhanced response builder with qwen3 reasoning support"""
    
    def build_response(self, 
                      query: str,
                      llm_response: str, 
                      retrieved_data: Dict[str, Any],
                      subgraph_context: str,
                      query_type: str,
                      confidence_score: float = 0.0,
                      reasoning: str = None) -> Dict[str, Any]:
        """Build a structured response with reasoning"""
        
        response = {
            "query": query,
            "query_type": query_type,
            "response": llm_response,
            "reasoning": reasoning,  # Add qwen3 reasoning output
            "confidence_score": confidence_score,
            "retrieved_data": retrieved_data,
            "subgraph_context": subgraph_context,
            "citations": self._extract_citations(retrieved_data),
            "related_entities": self._extract_entities(retrieved_data),
            "suggested_followups": self._generate_followups(query, query_type)
        }
        
        return response
    
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
    
    def _generate_followups(self, query: str, query_type: str) -> List[str]:
        """Generate intelligent follow-up questions"""
        
        followups = {
            "drug_repurposing": [
                "What are the potential side effects of this repurposing approach?",
                "What clinical trials would be needed to validate this?",
                "Are there any contraindications to consider?",
                "What biomarkers could predict success?"
            ],
            "mechanism_explanation": [
                "What are the downstream effects of this mechanism?",
                "Are there alternative pathways involved?",
                "How does this compare to other drugs in the same class?",
                "What resistance mechanisms might develop?"
            ],
            "hypothesis_testing": [
                "What additional experiments could strengthen this evidence?",
                "What are the statistical limitations of this analysis?",
                "How could we validate this in clinical studies?",
                "What confounding factors should we consider?"
            ]
        }
        
        return followups.get(query_type, [
            "Can you provide more specific details?",
            "What are the clinical implications?",
            "How does this relate to current treatments?",
            "What are the next research steps?"
        ])
