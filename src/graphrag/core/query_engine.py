from typing import Dict, List, Any
import logging
from .llama_integration import LlamaGraphRAGEngine
from .retriever import GraphRetriever
from .context_builder import ContextBuilder
from .prompt_templates import PromptTemplates
from .vector_store import WeaviateGraphStore
from ..generators.llm_client import OllamaClient
from ..generators.response_builder import ResponseBuilder
from ..retrievers import SubgraphRetriever, PathRetriever, CommunityRetriever
from .query_processor import preprocess_query, classify_query_type, QueryStrategy
from .result_aggregator import aggregate_search_results, format_results_for_llm, generate_followup_questions

logger = logging.getLogger(__name__)


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
        self.subgraph_retriever = SubgraphRetriever(graph)
        self.path_retriever = PathRetriever(graph)
        self.community_retriever = CommunityRetriever(graph)

        logger.info("GraphRAGQueryEngine initialized successfully")
        # Initialize LlamaIndex integration
        self.llama_engine = LlamaGraphRAGEngine(
            nx_graph=graph,
            vector_store=vector_store,
            llm_client=llm_client,
            path_retriever=self.path_retriever,
            community_retriever=self.community_retriever,
            subgraph_retriever=self.subgraph_retriever
        )
    
    def query(self, user_query: str, query_type: str = "auto", max_results: int = 10) -> Dict[str, Any]:
        # All queries now go through the centralized, graph-native LlamaIndex engine
        logger.info(f"Routing query to LlamaGraphRAGEngine: '{user_query}'")
        return self.llama_engine.query(user_query, query_type=query_type, max_results=max_results)
    
    def _vector_entity_search(self, query: str, max_results: int) -> Dict[str, List[Dict]]:
        """Use Weaviate for vector-based entity retrieval"""
        try:
            logger.info(f"Starting vector search for: {query}")
            
            # Search entities using the vector store
            all_entities = self.vector_store.search_entities(
                query, 
                entity_types=["Drug", "Disease", "Protein"], 
                n_results=max_results
            )
            
            logger.info(f"Vector search returned {len(all_entities)} entities")
            
            # Group by type
            result = {'drugs': [], 'diseases': [], 'proteins': []}
            for entity in all_entities:
                entity_type = entity.get('type', 'unknown')
                if entity_type == 'drug':
                    result['drugs'].append(entity)
                elif entity_type == 'disease':
                    result['diseases'].append(entity)
                elif entity_type == 'protein':
                    result['proteins'].append(entity)
            
            logger.info(f"Grouped entities: {len(result['drugs'])} drugs, {len(result['diseases'])} diseases, {len(result['proteins'])} proteins")
            return result
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {'drugs': [], 'diseases': [], 'proteins': []}
    
    def _enrich_with_graph_context(self, entities: Dict, query: str) -> str:
        """Enrich vector results with graph neighborhood context"""
        contexts = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list[:3]:  # Top 3 per type
                entity_id = entity['id']
                entity_context = self.context_builder.build_entity_context(entity_id)
                contexts.append(entity_context)
            if entity_type == 'drugs':
                for disease in entities['diseases'][:2]:
                    paths = self.retriever.get_drug_disease_paths(entity_id, disease['id'], max_paths=2)
                    if paths:
                        path_context = self.context_builder.build_path_context(paths)
                        contexts.append(path_context)
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
    
    def _handle_drug_repurposing_with_reasoning(self, query: str, entities: Dict, context: str, aggregated_results: Dict) -> Dict[str, Any]:
        """Handle drug repurposing with qwen3 reasoning output"""
        repurposing_context = context
        if entities['diseases']:
            disease_id = entities['diseases'][0]['id']
            candidate_drugs = self.retriever.get_disease_associated_drugs(disease_id)
            if candidate_drugs:
                repurposing_context += "\n\n**Potential Repurposing Candidates:**\n"
                for drug in candidate_drugs[:5]:
                    drug_context = self.context_builder.build_entity_context(drug['drug_id'])
                    repurposing_context += f"\n{drug_context}"
        prompt = self.prompt_templates.drug_repurposing_prompt(query, repurposing_context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        followups = generate_followup_questions(aggregated_results, {"query_type": "drug_repurposing"})
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=repurposing_context,
            query_type="drug_repurposing",
            confidence_score=0.8,
            reasoning=reasoning,
            suggested_followups=followups
        )
    
    def _handle_mechanism_with_reasoning(self, query: str, entities: Dict, context: str, aggregated_results: Dict) -> Dict[str, Any]:
        """Handle mechanism explanation with reasoning"""
        prompt = self.prompt_templates.mechanism_explanation_prompt(query, context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        followups = generate_followup_questions(aggregated_results, {"query_type": "mechanism_explanation"})
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=context,
            query_type="mechanism_explanation",
            confidence_score=0.7,
            reasoning=reasoning,
            suggested_followups=followups
        )
    
    def _handle_hypothesis_with_reasoning(self, query: str, entities: Dict, context: str, aggregated_results: Dict) -> Dict[str, Any]:
        """Handle hypothesis testing with reasoning"""
        prompt = self.prompt_templates.hypothesis_testing_prompt(query, context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        followups = generate_followup_questions(aggregated_results, {"query_type": "hypothesis_testing"})
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=context,
            query_type="hypothesis_testing",
            confidence_score=0.6,
            reasoning=reasoning,
            suggested_followups=followups
        )
    
    def _handle_general_with_reasoning(self, query: str, entities: Dict, context: str, aggregated_results: Dict) -> Dict[str, Any]:
        """Handle general queries with reasoning"""
        prompt = self.prompt_templates.general_query_prompt(query, context)
        reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
        followups = generate_followup_questions(aggregated_results, {"query_type": "general"})
        return self.response_builder.build_response(
            query=query,
            llm_response=final_answer,
            retrieved_data=entities,
            subgraph_context=context,
            query_type="general",
            confidence_score=0.5,
            reasoning=reasoning,
            suggested_followups=followups
        )
    
    def _classify_query(self, query: str) -> str:
        """Intelligent query classification - fallback method if preprocess_query fails"""
        return classify_query_type(query)

    def _handle_comparative_analysis(self, processed_query) -> Dict[str, Any]:
        """Handle comparative analysis queries."""
        entities = processed_query.extracted_entities

        # Get entities for comparison
        comparison_entities = []
        for entity_type, entity_list in entities.items():
            for entity_name in entity_list:
                matches = self.vector_store.search_entities(
                    entity_name,
                    entity_types=[entity_type.title()],
                    n_results=1
                )
                comparison_entities.extend(matches)

        # Use LlamaIndex for comparison
        return self.llama_engine.query(processed_query.cleaned_query)

    def _handle_pathway_traversal(self, processed_query) -> Dict[str, Any]:
        """Handle pathway traversal queries."""
        return self.llama_engine.query(processed_query.cleaned_query)

    def _handle_community_analysis(self, processed_query) -> Dict[str, Any]:
        """Handle community-based analysis."""
        return self.llama_engine.query(processed_query.cleaned_query)

    def _handle_multi_entity_analysis(self, processed_query) -> Dict[str, Any]:
        """Handle multi-entity analysis."""
        return self.llama_engine.query(processed_query.cleaned_query)
    
    def _handle_basic_query(self, processed_query) -> Dict[str, Any]:
        """Handle basic queries using vector search + graph context."""
        logger.info("Using basic query handler")
        
        try:
            # 1. Vector search for entities
            entities = self._vector_entity_search(processed_query.cleaned_query, 10)
            logger.info(f"Vector search found: {sum(len(v) for v in entities.values())} entities")
            
            # 2. Build graph context
            context = self._enrich_with_graph_context(entities, processed_query.cleaned_query)
            logger.info(f"Built context length: {len(context)}")
            
            # 3. Generate response using LLM
            prompt = self.prompt_templates.general_query_prompt(processed_query.cleaned_query, context)
            reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
            logger.info("LLM response generated")
            
            # 4. Generate follow-ups
            followups = self._generate_followups(processed_query.cleaned_query, entities)
            
            # 5. Build final response
            response = self.response_builder.build_response(
                query=processed_query.cleaned_query,
                llm_response=final_answer,
                retrieved_data=entities,
                subgraph_context=context,
                query_type=processed_query.query_type,
                confidence_score=0.7,
                reasoning=reasoning,
                suggested_followups=followups
            )
            
            logger.info("Response built successfully")
            return response
            
        except Exception as e:
            logger.error(f"Basic query processing failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _generate_followups(self, query: str, entities: Dict) -> List[str]:
        """Generate follow-up questions based on entities found."""
        followups = []
        
        if entities.get('drugs'):
            followups.append("What are the side effects of these drugs?")
        
        if entities.get('diseases'):
            followups.append("What other treatments are available?")
        
        if entities.get('proteins'):
            followups.append("What other drugs target these proteins?")
        
        if len(entities.get('drugs', [])) > 1:
            followups.append("How do these drugs compare?")
        
        return followups[:3]
