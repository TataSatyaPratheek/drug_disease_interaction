
from typing import Dict, List, Any, Optional
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
from .graph_analytics import HighPerformanceGraphAnalytics
from .connection_resilience import ConnectionResilience
import streamlit as st

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """Enhanced GraphRAG orchestrator with high-performance analytics."""
    
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
        self.analytics = HighPerformanceGraphAnalytics(graph)
        self.resilience = ConnectionResilience()
        # Initialize LlamaIndex integration
        self.llama_engine = LlamaGraphRAGEngine(
            nx_graph=graph,
            vector_store=vector_store,
            llm_client=llm_client,
            path_retriever=self.path_retriever,
            community_retriever=self.community_retriever,
            subgraph_retriever=self.subgraph_retriever
        )
    
    @ConnectionResilience.with_retry(max_attempts=3, wait_seconds=2)
    def query(self, user_query: str, query_type: str = "auto", max_results: int = 15) -> Dict[str, Any]:
        """Enhanced query processing with graph analytics."""
        logger.info(f"Processing query: '{user_query}'")
        try:
            # 1. Entity retrieval (enhanced with analytics)
            entities = self._enhanced_vector_entity_search(user_query, max_results)
            total_entities = sum(len(v) for v in entities.values())
            logger.info(f"Retrieved {total_entities} entities total")
            # 2. Build graph-aware context
            graph_context = self._build_enhanced_graph_context(entities, user_query)
            # 3. Create prompt with graph intelligence
            detailed_prompt = self._create_graph_aware_prompt(user_query, graph_context, entities)
            # 4. Generate response with robust LLM call
            logger.info("Calling LLM for response generation...")
            try:
                reasoning, final_answer = self.resilience.robust_ollama_call(
                    self.llm_client, detailed_prompt
                )
                logger.info(f"LLM response generated - Reasoning: {len(reasoning)} chars, Answer: {len(final_answer)} chars")
            except Exception as llm_error:
                logger.error(f"LLM generation failed: {llm_error}")
                reasoning = f"LLM generation failed: {str(llm_error)}"
                final_answer = f"Unable to generate response due to LLM error: {str(llm_error)}"
            # 5. Extract advanced graph features
            path_data = self._extract_enhanced_path_data(entities, user_query)
            community_data = self._extract_enhanced_community_data(entities)
            node_importance = self._get_node_importance_rankings(entities)
            # 6. Build comprehensive response
            response = {
                'response': final_answer,
                'reasoning': reasoning,
                'retrieved_data': entities,
                'citations': self._build_citations(entities),
                'suggested_followups': self._generate_graph_aware_followups(user_query, entities),
                'query_type': query_type,
                'confidence_score': self._calculate_response_confidence(entities, total_entities),
                'subgraph_context': graph_context,
                'path_data': path_data,
                'community_data': community_data,
                'node_importance': node_importance,
                'graph_metrics': self._get_query_specific_metrics(entities)
            }
            logger.info("Enhanced query processing completed")
            return response
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

    def _enhanced_vector_entity_search(self, query: str, max_results: int) -> Dict[str, List[Dict]]:
        """Enhanced vector search with graph analytics ranking."""
        try:
            # Get basic vector search results
            base_results = self._vector_entity_search(query, max_results)
            # Enhance with graph analytics
            for entity_type, entities in base_results.items():
                for entity in entities:
                    entity_id = entity.get('id')
                    if entity_id:
                        # Add importance ranking
                        importance_rankings = self.analytics.rank_nodes_by_importance(
                            entity_type, top_k=100
                        )
                        # Find this entity in rankings
                        for rank, ranked_entity in enumerate(importance_rankings):
                            if ranked_entity['id'] == entity_id:
                                entity['importance_rank'] = rank + 1
                                entity['importance_score'] = ranked_entity['importance_score']
                                entity['centrality_metrics'] = {
                                    'pagerank': ranked_entity['pagerank'],
                                    'betweenness': ranked_entity['betweenness'],
                                    'closeness': ranked_entity['closeness'],
                                    'degree': ranked_entity['degree']
                                }
                                break
            return base_results
        except Exception as e:
            logger.error(f"Enhanced vector search failed: {e}")
            return self._vector_entity_search(query, max_results)

    def _build_enhanced_graph_context(self, entities: Dict, query: str) -> str:
        """Build enhanced context with graph analytics insights."""
        try:
            context_parts = [f"ENHANCED GRAPH ANALYSIS FOR QUERY: '{query}'\n"]
            total_entities = sum(len(v) for v in entities.values())
            context_parts.append(f"Retrieved {total_entities} entities from knowledge graph with graph analytics:\n")
            # Add entity information with importance rankings
            for entity_type, entity_list in entities.items():
                if entity_list:
                    context_parts.append(f"\n{entity_type.upper()} ({len(entity_list)}):")
                    for entity in entity_list[:3]:  # Top 3 per type
                        name = entity.get('name', 'Unknown')
                        importance_rank = entity.get('importance_rank', 'N/A')
                        importance_score = entity.get('importance_score', 0)
                        context_parts.append(
                            f"- {name} (Importance Rank: {importance_rank}, "
                            f"Score: {importance_score:.4f})"
                        )
            # Add community insights
            if total_entities > 0:
                community_stats = self.community_retriever.get_community_statistics()
                if community_stats:
                    context_parts.append(f"\nCOMMUNITY INSIGHTS:")
                    context_parts.append(f"- {community_stats['total_communities']} communities detected")
                    context_parts.append(f"- Modularity: {community_stats['modularity']:.3f}")
                    context_parts.append(f"- Average community size: {community_stats['average_community_size']:.1f}")
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Enhanced context building failed: {e}")
            return self._build_graph_context(entities, query)

    def _create_graph_aware_prompt(self, query: str, context: str, entities: Dict) -> str:
        """Create prompt with graph-aware intelligence."""
        try:
            entity_summary = []
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_summary.append(f"{len(entity_list)} {entity_type}")
            if not entity_summary:
                entity_summary = ["no specific entities"]
            prompt = f"""You are a senior biomedical researcher with access to a comprehensive drug-disease knowledge graph enhanced with graph analytics.

RESEARCH QUERY: {query}

GRAPH ANALYTICS RESULTS:
Found {', '.join(entity_summary)} using advanced graph traversal and importance ranking.

ENHANCED CONTEXT WITH GRAPH INTELLIGENCE:
{context}

ANALYSIS FRAMEWORK:
1. **Network Analysis**: Use the importance rankings and centrality metrics provided
2. **Community Structure**: Consider the community insights for therapeutic groupings
3. **Pathway Analysis**: Leverage the graph topology for mechanism discovery
4. **Evidence Integration**: Synthesize findings from multiple graph perspectives

RESPONSE FORMAT:
- **Executive Summary**: 2-3 sentences with key findings and graph-based insights
- **Detailed Analysis**: Comprehensive reasoning including:
  • Graph topology insights and node importance
  • Community structure and therapeutic implications
  • Pathway analysis and mechanism discovery
  • Integration of graph-based evidence

GRAPH-NATIVE REASONING:
Use the graph structure, node importance, and community insights to provide deeper analysis than simple entity matching. Consider network effects and topological relationships.

Please provide a comprehensive analysis leveraging the graph intelligence."""
            return prompt
        except Exception as e:
            logger.error(f"Graph-aware prompt creation failed: {e}")
            return self._create_comprehensive_prompt(query, context, entities)

    def _calculate_response_confidence(self, entities: Dict, total_entities: int) -> float:
        """Calculate confidence score based on graph analytics."""
        try:
            if total_entities == 0:
                return 0.1
            # Base confidence on entity count
            entity_confidence = min(total_entities / 10, 1.0)
            # Enhance with importance scores
            total_importance = 0
            importance_count = 0
            for entity_list in entities.values():
                for entity in entity_list:
                    importance_score = entity.get('importance_score', 0)
                    if importance_score > 0:
                        total_importance += importance_score
                        importance_count += 1
            importance_confidence = (total_importance / importance_count) if importance_count > 0 else 0.5
            # Combine scores
            final_confidence = (entity_confidence * 0.6) + (importance_confidence * 0.4)
            return min(final_confidence, 1.0)
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _get_query_specific_metrics(self, entities: Dict) -> Dict[str, Any]:
        """Get query-specific graph metrics."""
        try:
            all_entity_ids = [
                entity['id'] for entity_list in entities.values() 
                for entity in entity_list if entity.get('id')
            ]
            if not all_entity_ids:
                return {}
            # Get subgraph metrics
            subgraph = self.analytics.get_neighborhood_subgraph(all_entity_ids, radius=2)
            import networkx as nx
            metrics = {
                'subgraph_size': subgraph.number_of_nodes(),
                'subgraph_edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph) if subgraph.number_of_nodes() > 0 else 0,
                'connected_components': nx.number_connected_components(subgraph.to_undirected()),
                'entity_types_found': list(entities.keys()),
                'total_entities': sum(len(v) for v in entities.values())
            }
            return metrics
        except Exception as e:
            logger.error(f"Query metrics calculation failed: {e}")
            return {}

    def _extract_path_data(self, entities: Dict, query: str) -> 'Optional[Dict]':
        """Extract path data for visualization."""
        try:
            # Get entity IDs for path finding
            all_ids = [e['id'] for entity_list in entities.values() for e in entity_list if e.get('id')]
            
            if len(all_ids) >= 2:
                # Use existing path retriever
                paths = self.path_retriever.find_drug_disease_paths(all_ids[0], all_ids[1], max_paths=3)
                if paths:
                    return {
                        'path': paths[0].get('path', []),
                        'path_names': paths[0].get('path_names', []),
                        'path_types': paths[0].get('path_types', [])
                    }
        except Exception as e:
            logger.warning(f"Path extraction failed: {e}")
        
        return None

    def _extract_community_data(self, entities: Dict) -> 'Optional[List[Dict]]':
        """Extract community data for visualization."""
        try:
            # Get entity IDs for community analysis
            all_ids = [e['id'] for entity_list in entities.values() for e in entity_list if e.get('id')]
            
            if all_ids:
                communities = self.community_retriever.get_communities(all_ids[:5])
                return communities
        except Exception as e:
            logger.warning(f"Community extraction failed: {e}")
        
        return None

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
                        
                        # Get immediate neighbors from graph - FIX HERE
                        if entity_id in self.graph:
                            neighbors = list(self.graph.neighbors(entity_id))[:3]
                            if neighbors:
                                neighbor_names = []
                                for neighbor_id in neighbors:
                                    neighbor_data = self.graph.nodes.get(neighbor_id, {})
                                    neighbor_name = neighbor_data.get('name', neighbor_id)  # Use ID as fallback
                                    # Ensure we have a string value
                                    if neighbor_name:
                                        neighbor_names.append(str(neighbor_name))
                                    else:
                                        neighbor_names.append(str(neighbor_id))
                                
                                # Only add if we have valid names
                                if neighbor_names:
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

    def _create_comprehensive_prompt(self, query: str, graph_context: str, entities: Dict) -> str:
        """Create a comprehensive prompt for the LLM."""
        
        # Count entities by type
        entity_counts = {k: len(v) for k, v in entities.items() if v}
        
        if not entity_counts:
            return f"""Based on the drug-disease knowledge graph, I could not find specific entities for the query: "{query}"

    Please provide a general response about this topic and suggest more specific terms that might yield better results."""

        # Build entity summary
        entity_summary = []
        for entity_type, count in entity_counts.items():
            entity_summary.append(f"{count} {entity_type}")
        
        prompt = f"""You are analyzing a biomedical knowledge graph to answer this query: "{query}"

    RETRIEVED ENTITIES:
    Found {', '.join(entity_summary)} relevant to your query.

    DETAILED GRAPH CONTEXT:
    {graph_context}

    ANALYSIS TASK:
    1. Examine the retrieved entities and their relationships
    2. Focus on how they relate to the specific query
    3. If comparing drugs (like the query), discuss their mechanisms, efficacy, and applications
    4. Base your analysis on the graph data provided above
    5. Provide specific insights about the entities found

    Please analyze this information systematically and provide a comprehensive answer."""

        return prompt

    def _create_professional_prompt(self, query: str, graph_context: str, entities: Dict) -> str:
        """Create professional scientific prompt."""
        
        from .prompt_templates import ScientificPromptTemplates
        
        # Determine query type for specialized prompts
        if any(word in query.lower() for word in ["compare", "vs", "versus", "efficacy"]):
            return ScientificPromptTemplates.comparative_drug_prompt(query, graph_context, entities)
        else:
            return ScientificPromptTemplates.biomedical_analysis_prompt(query, graph_context, entities)
