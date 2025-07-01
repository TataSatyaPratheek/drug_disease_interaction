# src/graphrag/core/query_engine.py
import re
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from .retriever import GraphRetriever
from .context_builder import ContextBuilder
from .prompt_templates import PromptTemplates
from .vector_store import WeaviateGraphStore
from ..generators.llm_client import OllamaClient
from ..generators.response_builder import ResponseBuilder

class GraphRAGQueryEngine:
    """Enhanced GraphRAG orchestrator with qwen3 reasoning capabilities"""
    
    def __init__(self, graph, llm_client: OllamaClient, vector_store: WeaviateGraphStore):
        self.graph = graph
        self.retriever = GraphRetriever(graph)
        self.context_builder = ContextBuilder(graph)
        self.prompt_templates = PromptTemplates()
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.response_builder = ResponseBuilder()
    
    def query(self, user_query: str, query_type: str = "auto", max_results: int = 10) -> Dict[str, Any]:
        """Enhanced GraphRAG pipeline with qwen3 reasoning output"""
        
        # 1. Classify query type
        if query_type == "auto":
            query_type = self._classify_query(user_query)
        
        # 2. Vector search for relevant entities
        entities = self._vector_entity_search(user_query, max_results)
        
        # 3. Graph-based context enrichment
        enriched_context = self._enrich_with_graph_context(entities, user_query)
        
        # 4. Route to appropriate handler with reasoning
        if query_type == "drug_repurposing":
            return self._handle_drug_repurposing_with_reasoning(user_query, entities, enriched_context)
        elif query_type == "mechanism_explanation":
            return self._handle_mechanism_with_reasoning(user_query, entities, enriched_context)
        elif query_type == "hypothesis_testing":
            return self._handle_hypothesis_with_reasoning(user_query, entities, enriched_context)
        else:
            return self._handle_general_with_reasoning(user_query, entities, enriched_context)
    
    def _classify_query(self, query: str) -> str:
        """Intelligent query classification"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["repurpos", "repositioning", "new indication", "alternative use"]):
            return "drug_repurposing"
        elif any(word in query_lower for word in ["mechanism", "how does", "pathway", "why", "works"]):
            return "mechanism_explanation"
        elif any(word in query_lower for word in ["hypothesis", "test", "prove", "evidence", "study"]):
            return "hypothesis_testing"
        elif any(word in query_lower for word in ["compare", "difference", "similar", "versus", "vs"]):
            return "drug_comparison"
        else:
            return "general"
    
    def _vector_entity_search(self, query: str, max_results: int) -> Dict[str, List[Dict]]:
        """Use Weaviate for vector-based entity retrieval"""
        
        drugs = self.vector_store.search_entities(query, ["Drug"], max_results//3)
        diseases = self.vector_store.search_entities(query, ["Disease"], max_results//3)
        proteins = self.vector_store.search_entities(query, ["Protein"], max_results//3)
        
        return {
            'drugs': drugs,
            'diseases': diseases,
            'proteins': proteins
        }
    
    def _enrich_with_graph_context(self, entities: Dict, query: str) -> str:
        """Enrich vector results with graph neighborhood context"""
        
        contexts = []
        
        # For each found entity, get its graph neighborhood
        for entity_type, entity_list in entities.items():
            for entity in entity_list[:3]:  # Top 3 per type
                entity_id = entity['id']
                entity_context = self.context_builder.build_entity_context(entity_id)
                contexts.append(entity_context)
                
                # Get paths to other relevant entities
                if entity_type == 'drugs':
                    for disease in entities['diseases'][:2]:
                        paths = self.retriever.get_drug_disease_paths(entity_id, disease['id'], max_paths=2)
                        if paths:
                            path_context = self.context_builder.build_path_context(paths)
                            contexts.append(path_context)
        
        # Get relationship context from vector store
        try:
            relationships = self.vector_store.search_relationships(query, n_results=5)
        except AttributeError:
            relationships = []

        if relationships:
            rel_context = self._build_relationship_context(relationships)
            contexts.append(rel_context)
        
        return "\n\n".join(contexts)
    
    def _build_relationship_context(self, relationships: List[Dict]) -> str:
        """Build context from relationship search results"""
        if not relationships:
            return ""
        
        context_parts = ["**Relevant Relationships:**"]
        for rel in relationships:
            context_parts.append(f"- {rel['source_name']} ({rel['edge_type']}) {rel['target_name']}")
        
        return "\n".join(context_parts)
    
    def _handle_drug_repurposing_with_reasoning(self, query: str, entities: Dict, context: str) -> Dict[str, Any]:
        """Handle drug repurposing with qwen3 reasoning output"""
        
        # Build specialized context
        repurposing_context = context
        
        if entities['diseases']:
            disease_id = entities['diseases'][0]['id']
            candidate_drugs = self.retriever.get_disease_associated_drugs(disease_id)
            
            if candidate_drugs:
                repurposing_context += "\n\n**Potential Repurposing Candidates:**\n"
                for drug in candidate_drugs[:5]:
                    drug_context = self.context_builder.build_entity_context(drug['drug_id'])
                    repurposing_context += f"\n{drug_context}"
        
        # Generate response with reasoning using qwen3
        prompt = self.prompt_templates.drug_repurposing_prompt(query, repurposing_context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=repurposing_context,
            query_type="drug_repurposing",
            confidence_score=0.8,
            reasoning=reasoning  # Add reasoning to response
        )
    
    def _handle_mechanism_with_reasoning(self, query: str, entities: Dict, context: str) -> Dict[str, Any]:
        """Handle mechanism explanation with reasoning"""
        
        prompt = self.prompt_templates.mechanism_explanation_prompt(query, context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=context,
            query_type="mechanism_explanation",
            confidence_score=0.7,
            reasoning=reasoning
        )
    
    def _handle_hypothesis_with_reasoning(self, query: str, entities: Dict, context: str) -> Dict[str, Any]:
        """Handle hypothesis testing with reasoning"""
        
        prompt = self.prompt_templates.hypothesis_testing_prompt(query, context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=context,
            query_type="hypothesis_testing",
            confidence_score=0.6,
            reasoning=reasoning
        )
    
    def _handle_general_with_reasoning(self, query: str, entities: Dict, context: str) -> Dict[str, Any]:
        """Handle general queries with reasoning"""
        
        prompt = self.prompt_templates.general_query_prompt(query, context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=context,
            query_type="general",
            confidence_score=0.5,
            reasoning=reasoning
        )
