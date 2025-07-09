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
    
    def query(self, user_query: str, query_type: str = "auto", max_results: int = 15) -> Dict[str, Any]:
        """Fixed query processing that ensures query-specific responses."""
        logger.info(f"Processing query: '{user_query}'")
        
        # Add debug call
        debug_info = self.debug_retrieval(user_query)
        
        try:
            # 1. Enhanced entity retrieval
            entities = self._vector_entity_search(user_query, max_results)
            total_entities = sum(len(v) for v in entities.values())
            logger.info(f"Retrieved {total_entities} entities total")
            
            # 2. Build query-specific context
            graph_context = self._build_graph_context(entities, user_query)
            
            # 3. Create query-specific prompt
            prompt = f"""You are analyzing a biomedical knowledge graph for this specific query: "{user_query}"

    RETRIEVED GRAPH DATA:
    {graph_context}

    INSTRUCTIONS:
    1. If specific entities were found, explain how they relate to the query
    2. If relationships exist, describe the pathways or mechanisms
    3. If no specific data was found, acknowledge this limitation
    4. Focus ONLY on the actual graph data provided above
    5. Do NOT provide general knowledge - only use the graph data shown

    Answer the specific question: {user_query}"""

            # 4. Generate response
            reasoning, final_answer = self.llm_client.generate_with_reasoning(prompt)
            
            # 5. Build response
            response = {
                'response': final_answer,
                'reasoning': reasoning,
                'retrieved_data': entities,
                'citations': self._build_citations(entities),
                'suggested_followups': self._generate_query_specific_followups(user_query, entities),
                'query_type': query_type,
                'confidence_score': 0.8 if total_entities > 0 else 0.3,
                'subgraph_context': graph_context,
                'debug_info': debug_info
            }
            
            logger.info("Query processing completed")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise


    def _build_graph_context(self, entities: Dict, query: str) -> str:
        """Build specific context from retrieved entities and graph relationships."""
        if not any(entities.values()):
            return f"No specific entities found for query: '{query}'. Searching broader context..."
        
        context_parts = [f"GRAPH ANALYSIS FOR QUERY: '{query}'\n"]
        
        # Document found entities
        total_entities = sum(len(v) for v in entities.values())
        context_parts.append(f"Found {total_entities} relevant entities in the knowledge graph:")
        
        all_entity_ids = []
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                context_parts.append(f"\n{entity_type.upper()} ({len(entity_list)}):")
                for entity in entity_list[:5]:  # Top 5 per type
                    name = entity.get('name', 'Unknown')
                    entity_id = entity.get('id', '')
                    similarity = entity.get('similarity', 0)
                    context_parts.append(f"- {name} (Relevance: {similarity:.3f})")
                    
                    if entity_id:
                        all_entity_ids.append(entity_id)
                        
                        # Get immediate neighbors from graph
                        if entity_id in self.graph:
                            neighbors = list(self.graph.neighbors(entity_id))[:3]
                            if neighbors:
                                neighbor_names = []
                                for neighbor_id in neighbors:
                                    neighbor_data = self.graph.nodes.get(neighbor_id, {})
                                    neighbor_name = neighbor_data.get('name', neighbor_id)
                                    neighbor_names.append(neighbor_name)
                                context_parts.append(f"  Connected to: {', '.join(neighbor_names)}")
        
        # Find relationships between entities
        if len(all_entity_ids) >= 2:
            context_parts.append(f"\nGRAPH RELATIONSHIPS:")
            paths_found = 0
            for i, entity1 in enumerate(all_entity_ids[:3]):
                for entity2 in all_entity_ids[i+1:4]:
                    if entity1 in self.graph and entity2 in self.graph:
                        try:
                            if self.graph.has_edge(entity1, entity2):
                                edge_data = self.graph.get_edge_data(entity1, entity2)
                                relation_type = list(edge_data.values())[0].get('type', 'connected')
                                entity1_name = self.graph.nodes[entity1].get('name', entity1)
                                entity2_name = self.graph.nodes[entity2].get('name', entity2)
                                context_parts.append(f"- {entity1_name} --[{relation_type}]--> {entity2_name}")
                                paths_found += 1
                        except Exception as e:
                            continue
            
            if paths_found == 0:
                context_parts.append("- No direct relationships found between retrieved entities")
        
        return "\n".join(context_parts)


    def _build_prompt(self, query: str, context: str, entities: Dict) -> str:
        """Build prompt that connects query to actual retrieved data."""
        entity_summary = []
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_summary.append(f"{len(entity_list)} {entity_type}")
        
        prompt = f"""Based on the drug-disease knowledge graph, I found {', '.join(entity_summary)} relevant to your query.

    QUERY: {query}

    KNOWLEDGE GRAPH CONTEXT:
    {context}

    Please analyze this information and provide:
    1. How the retrieved entities relate to your query
    2. What connections exist between them in the knowledge graph
    3. A comprehensive answer based on this specific graph data

    Focus on the actual entities and relationships found, not general knowledge."""
        
        return prompt

    
    def _vector_entity_search(self, query: str, max_results: int) -> Dict[str, List[Dict]]:
        """Enhanced vector search with broader entity matching."""
        try:
            logger.info(f"Starting vector search for: '{query}'")
            
            # Expand search terms
            search_terms = [query]
            
            all_entities = []

            for term in search_terms:
                try:
                    results = self.vector_store.search_entities(
                        term,
                        entity_types=["Drug", "Disease", "Protein"],  # Only these exist in schema
                        n_results=15
                    )
                    all_entities.extend(results)
                    logger.info(f"Term '{term}' found {len(results)} entities")
                except Exception as e:
                    logger.warning(f"Search for '{term}' failed: {e}")
            
            # Remove duplicates and group by type
            seen_ids = set()
            unique_entities = []
            for entity in all_entities:
                entity_id = entity.get('id')
                if entity_id and entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    unique_entities.append(entity)
            
            # Group by type
            result = {'drugs': [], 'diseases': [], 'proteins': []}
            for entity in unique_entities[:max_results]:
                entity_type = entity.get('type', 'unknown').lower()
                if entity_type in ['drug', 'compound']:
                    result['drugs'].append(entity)
                elif entity_type in ['disease', 'disorder', 'syndrome']:
                    result['diseases'].append(entity)
                elif entity_type in ['protein', 'polypeptide', 'enzyme']:
                    result['proteins'].append(entity)
            
            total_found = sum(len(v) for v in result.values())
            logger.info(f"Final grouped results: {total_found} entities")
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

    def debug_retrieval(self, query: str) -> Dict:
        """Debug what entities are actually being retrieved."""
        print(f"\n=== DEBUGGING QUERY: '{query}' ===")
        
        # Test vector search
        try:
            vector_results = self.vector_store.search_entities(
                query, 
                entity_types=["Drug", "Disease", "Protein", "Polypeptide"],
                n_results=10
            )
            print(f"Vector search found: {len(vector_results)} entities")
            for i, entity in enumerate(vector_results[:5]):
                print(f"  {i+1}. {entity.get('name', 'Unknown')} ({entity.get('type', 'unknown')}) - Score: {entity.get('similarity', 0):.3f}")
        except Exception as e:
            print(f"Vector search failed: {e}")
            vector_results = []
        
        # Test individual term searches
        terms = ["protein",  "disease", "drug"]
        for term in terms:
            try:
                term_results = self.vector_store.search_entities(term, n_results=3)
                print(f"Term '{term}' found: {len(term_results)} entities")
            except Exception as e:
                print(f"Term '{term}' search failed: {e}")
        
        return {"vector_results": vector_results}
    def _build_citations(self, entities):
        """Build citations from retrieved entities."""
        citations = []
        count = 1
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                citation = {
                    'id': count,
                    'name': entity.get('name', 'Unknown'),
                    'type': entity_type,
                    'entity_id': entity.get('id', ''),
                    'score': entity.get('similarity', 0.0)
                }
                citations.append(citation)
                count += 1
        return citations

    def _generate_query_specific_followups(self, query: str, entities: Dict) -> List[str]:
        """Generate follow-up questions based on the query and retrieved entities."""
        followups = []
        
        if not any(entities.values()):
            return [
                "Try searching for more specific terms",
                "Check if your data is properly indexed",
                "Browse the knowledge graph manually"
            ]
        
        if entities.get('proteins'):
            followups.append("What diseases are associated with these proteins?")
        
        if entities.get('diseases'):
            followups.append("What treatments are available for these conditions?")
        
        if entities.get('drugs'):
            followups.append("What are the side effects of these drugs?")
        
        return followups[:3]
