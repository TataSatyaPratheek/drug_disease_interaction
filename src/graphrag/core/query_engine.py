# src/graphrag/core/query_engine.py
import re
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from .retriever import GraphRetriever
from .context_builder import ContextBuilder
from .prompt_templates import PromptTemplates

class GraphRAGQueryEngine:
    """Main GraphRAG orchestrator for drug-disease knowledge graph"""
    
    def __init__(self, graph, llm_client, vector_store=None):
        self.graph = graph
        self.retriever = GraphRetriever(graph)
        self.context_builder = ContextBuilder(graph)
        self.prompt_templates = PromptTemplates()
        self.llm_client = llm_client
        self.vector_store = vector_store
    
    def query(self, user_query: str, query_type: str = "auto", max_results: int = 10) -> Dict[str, Any]:
        """Main query processing pipeline"""
        
        # 1. Classify query type if not specified
        if query_type == "auto":
            query_type = self._classify_query(user_query)
        
        # 2. Use vector store for entity retrieval if available
        if self.vector_store:
            entities = self._vector_entity_extraction(user_query, max_results)
        else:
            entities = self._extract_entities(user_query)
        
        # 3. Route to appropriate handler
        if query_type == "drug_repurposing":
            return self._handle_drug_repurposing(user_query, entities)
        elif query_type == "mechanism_explanation":
            return self._handle_mechanism_explanation(user_query, entities)
        elif query_type == "drug_comparison":
            return self._handle_drug_comparison(user_query, entities)
        elif query_type == "target_discovery":
            return self._handle_target_discovery(user_query, entities)
        else:
            return self._handle_general_query(user_query, entities)
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query using patterns"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["repurpos", "repositioning", "new indication"]):
            return "drug_repurposing"
        elif any(word in query_lower for word in ["mechanism", "how does", "pathway", "why"]):
            return "mechanism_explanation"
        elif any(word in query_lower for word in ["compare", "difference", "similar", "versus", "vs"]):
            return "drug_comparison"
        elif any(word in query_lower for word in ["target", "protein", "biomarker"]):
            return "target_discovery"
        else:
            return "general"
    
    def _vector_entity_extraction(self, query: str, max_results: int) -> Dict[str, List[Dict]]:
        """Extract entities using vector similarity search"""
        entities = {
            'drugs': self.vector_store.search_entities(query, ['drug'], max_results//3),
            'diseases': self.vector_store.search_entities(query, ['disease'], max_results//3),
            'proteins': self.vector_store.search_entities(query, ['protein'], max_results//3)
        }
        return entities
    
    def _extract_entities(self, query: str) -> Dict[str, List[Dict]]:
        """Fallback entity extraction without vector store"""
        entities = {
            'drugs': self.retriever.find_entities(query, 'drugs'),
            'diseases': self.retriever.find_entities(query, 'diseases'),
            'proteins': self.retriever.find_entities(query, 'proteins')
        }
        return entities
    
    def _handle_drug_repurposing(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle drug repurposing queries"""
        
        if entities['diseases']:
            disease_id = entities['diseases'][0]['id']
            
            # Get drugs targeting disease-associated proteins
            candidate_drugs = self.retriever.get_disease_associated_drugs(disease_id)
            
            # Build context
            disease_context = self.context_builder.build_entity_context(disease_id)
            
            # Build context for top candidate drugs
            drug_contexts = []
            for drug_info in candidate_drugs[:5]:
                drug_context = self.context_builder.build_entity_context(drug_info['drug_id'])
                drug_contexts.append(drug_context)
            
            # Generate response
            context = f"""
Disease Information:
{disease_context}

Candidate Drugs for Repurposing:
{chr(10).join(drug_contexts)}
"""
            
            prompt = self.prompt_templates.drug_repurposing_prompt(query, context)
            response = self.llm_client.generate(prompt)
            
            return {
                "query": query,
                "query_type": "drug_repurposing",
                "response": response,
                "retrieved_data": {
                    "disease": entities['diseases'][0],
                    "candidate_drugs": candidate_drugs[:10]
                },
                "subgraph_context": context
            }
        
        return {"error": "Could not identify disease in query"}
    
    def _handle_general_query(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle general queries about the knowledge graph"""
        
        # Build context from all found entities
        contexts = []
        all_entities = []
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list[:2]:  # Top 2 per type
                context = self.context_builder.build_entity_context(entity['id'])
                contexts.append(context)
                all_entities.append(entity)
        
        if contexts:
            full_context = "\n\n".join(contexts)
            prompt = self.prompt_templates.general_query_prompt(query, full_context)
            response = self.llm_client.generate(prompt)
            
            return {
                "query": query,
                "query_type": "general",
                "response": response,
                "retrieved_data": {
                    "entities": all_entities
                },
                "subgraph_context": full_context
            }
        
        return {"error": "Could not find relevant entities in the knowledge graph"}
    
    def _handle_mechanism_explanation(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle mechanism queries"""
        return self._handle_general_query(query, entities)
    
    def _handle_drug_comparison(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle drug comparison queries"""
        return self._handle_general_query(query, entities)
    
    def _handle_target_discovery(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle target discovery queries"""
        return self._handle_general_query(query, entities)
