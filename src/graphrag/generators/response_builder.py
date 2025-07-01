# src/graphrag/generators/response_builder.py
from typing import Dict, List, Any, Optional
import json

class ResponseBuilder:
    """Build structured responses with citations and confidence scores"""
    
    def __init__(self):
        pass
    
    def build_response(self, 
                      query: str,
                      llm_response: str, 
                      retrieved_data: Dict[str, Any],
                      subgraph_context: str,
                      query_type: str,
                      confidence_score: float = 0.0) -> Dict[str, Any]:
        """Build a structured response with all components"""
        
        return {
            "query": query,
            "query_type": query_type,
            "response": llm_response,
            "confidence_score": confidence_score,
            "retrieved_data": retrieved_data,
            "subgraph_context": subgraph_context,
            "citations": self._extract_citations(retrieved_data),
            "related_entities": self._extract_entities(retrieved_data),
            "suggested_followups": self._generate_followups(query, query_type)
        }
    
    def _extract_citations(self, retrieved_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract citations from retrieved data"""
        citations = []
        
        # Extract drug citations
        if 'drugs' in retrieved_data:
            for drug in retrieved_data['drugs']:
                citations.append({
                    "type": "drug",
                    "id": drug.get('id', ''),
                    "name": drug.get('name', ''),
                    "source": "Knowledge Graph"
                })
        
        # Extract disease citations
        if 'diseases' in retrieved_data:
            for disease in retrieved_data['diseases']:
                citations.append({
                    "type": "disease", 
                    "id": disease.get('id', ''),
                    "name": disease.get('name', ''),
                    "source": "Knowledge Graph"
                })
        
        # Extract pathway citations
        if 'paths' in retrieved_data:
            citations.append({
                "type": "pathway",
                "count": len(retrieved_data['paths']),
                "source": "Graph Analysis"
            })
        
        return citations
    
    def _extract_entities(self, retrieved_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key entities mentioned in the response"""
        entities = []
        
        for entity_type in ['drugs', 'diseases', 'proteins']:
            if entity_type in retrieved_data:
                for entity in retrieved_data[entity_type]:
                    entities.append({
                        "type": entity_type[:-1],  # Remove 's'
                        "id": entity.get('id', ''),
                        "name": entity.get('name', ''),
                        "relevance_score": entity.get('score', 0.0)
                    })
        
        return sorted(entities, key=lambda x: x['relevance_score'], reverse=True)
    
    def _generate_followups(self, query: str, query_type: str) -> List[str]:
        """Generate suggested follow-up questions"""
        
        followups = {
            "drug_repurposing": [
                "What are the clinical trial requirements for this repurposing?",
                "Are there any safety concerns with this approach?",
                "What biomarkers could predict success?",
                "Which patient populations would benefit most?"
            ],
            "mechanism_explanation": [
                "What other drugs work through similar mechanisms?", 
                "Are there any drug resistance pathways?",
                "What are the downstream effects of this mechanism?",
                "How does this compare to standard treatments?"
            ],
            "drug_comparison": [
                "What are the relative side effect profiles?",
                "Which drug is more effective for specific patient types?",
                "Are there combination therapy opportunities?",
                "What are the cost-effectiveness differences?"
            ],
            "target_discovery": [
                "What chemical scaffolds could target these proteins?",
                "Are there any existing tool compounds?",
                "What assays could be used for screening?",
                "What are the structural requirements for binding?"
            ]
        }
        
        return followups.get(query_type, [
            "Can you provide more specific details?",
            "What are the clinical implications?",
            "Are there any recent research developments?",
            "How does this relate to current treatments?"
        ])
