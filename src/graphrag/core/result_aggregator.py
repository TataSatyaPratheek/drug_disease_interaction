import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def aggregate_search_results(vector_results: Dict, graph_results: Dict, query_context: Dict) -> Dict[str, Any]:
    """Aggregate results from vector search and graph traversal"""
    """Aggregate results from vector search (Weaviate) and graph traversal (Neo4j) for hybrid RAG."""
    try:
        # Combine entity results with hybrid awareness
        combined_entities = merge_entity_results_hybrid(vector_results, graph_results)
        # Build unified context
        unified_context = build_unified_context(combined_entities, query_context)
        # Extract citations (with hybrid source info)
        citations = extract_citations_from_results(combined_entities)
        # Hybrid metadata
        hybrid_sources = set()
        for entity_list in combined_entities.values():
            for entity in entity_list:
                if "sources" in entity:
                    hybrid_sources.update(entity["sources"])
                elif "source" in entity:
                    hybrid_sources.add(entity["source"])
        aggregated_result = {
            "entities": combined_entities,
            "context": unified_context,
            "citations": citations,
            "metadata": {
                "vector_search_count": count_vector_results(vector_results),
                "graph_traversal_count": count_graph_results(graph_results),
                "total_unique_entities": count_unique_entities(combined_entities),
                "aggregation_timestamp": datetime.now().isoformat(),
                "sources": sorted(list(hybrid_sources))
            },
            "success": True
        }
        logger.info(f"Results aggregated: {aggregated_result['metadata']['total_unique_entities']} unique entities from {aggregated_result['metadata']['sources']}")
        return aggregated_result
    except Exception as e:
        logger.error(f"Result aggregation failed: {e}")
        return {
            "entities": {"drugs": [], "diseases": [], "proteins": [], "relationships": []},
            "context": "",
            "citations": [],
            "metadata": {"error": str(e)},
            "success": False
        }

def merge_entity_results_hybrid(vector_results: Dict[str, List[Dict]], graph_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Merge entity results from Weaviate (vector) and Neo4j (graph) for hybrid RAG."""
    merged: Dict[str, List[Dict]] = {
        "drugs": [],
        "diseases": [],
        "proteins": [],
        "relationships": []
    }
    # Index by ID for fast lookup
    id_index: Dict[str, Dict[str, Dict]] = {k: {} for k in merged.keys()}
    # Add vector search results
    for entity_type in merged.keys():
        for entity in vector_results.get(entity_type, []):
            eid = entity.get("id")
            entity = dict(entity)  # Copy to avoid mutating input
            entity["sources"] = ["vector_search"]
            entity["source"] = "vector_search"
            id_index[entity_type][eid] = entity
            merged[entity_type].append(entity)
    # Add graph traversal results
    for entity_type in merged.keys():
        for entity in graph_results.get(entity_type, []):
            eid = entity.get("id")
            if eid in id_index[entity_type]:
                # Merge attributes and sources
                existing = id_index[entity_type][eid]
                # Merge sources
                if "sources" in existing:
                    if "graph_traversal" not in existing["sources"]:
                        existing["sources"].append("graph_traversal")
                else:
                    existing["sources"] = [existing.get("source", "vector_search"), "graph_traversal"]
                # Merge fields (prefer higher score, longer description, etc.)
                for k, v in entity.items():
                    if k in ["similarity", "score"]:
                        if v > existing.get(k, 0):
                            existing[k] = v
                    elif k == "description":
                        if v and (not existing.get(k) or len(v) > len(existing.get(k, ""))):
                            existing[k] = v
                    elif k not in existing or not existing[k]:
                        existing[k] = v
                existing["source"] = "hybrid"
            else:
                entity = dict(entity)
                entity["sources"] = ["graph_traversal"]
                entity["source"] = "graph_traversal"
                id_index[entity_type][eid] = entity
                merged[entity_type].append(entity)
    # Sort by relevance score if available
    for entity_type in merged.keys():
        merged[entity_type].sort(
            key=lambda x: x.get("similarity", x.get("score", 0)),
            reverse=True
        )
    return merged

def merge_entity_information(existing_entities: List[Dict], new_entity: Dict):
    """Merge information from duplicate entities"""
    entity_id = new_entity.get("id")
    
    for existing in existing_entities:
        if existing.get("id") == entity_id:
            # Merge sources
            existing_sources = existing.get("sources", [existing.get("source", "")])
            new_source = new_entity.get("source", "")
            if new_source not in existing_sources:
                existing_sources.append(new_source)
            existing["sources"] = existing_sources
            
            # Take higher score
            existing_score = existing.get("similarity", existing.get("score", 0))
            new_score = new_entity.get("similarity", new_entity.get("score", 0))
            if new_score > existing_score:
                existing["similarity"] = new_score
                existing["score"] = new_score
            
            break

def build_unified_context(entities: Dict[str, List[Dict]], query_context: Dict) -> str:
    """Build unified context string from aggregated entities"""
    context_parts = []
    
    # Add query context
    query_type = query_context.get("query_type", "general")
    context_parts.append(f"Query Type: {query_type}")
    
    # Add entity summaries
    for entity_type, entity_list in entities.items():
        if entity_list:
            context_parts.append(f"\n=== {entity_type.upper()} ===")
            for entity in entity_list[:5]:  # Limit to top 5 per type
                name = entity.get("name", "Unknown")
                description = entity.get("description", "")
                score = entity.get("similarity", entity.get("score", 0))
                
                context_parts.append(f"- {name} (relevance: {score:.3f})")
                if description:
                    context_parts.append(f"  Description: {description[:200]}...")
    
    return "\n".join(context_parts)

def extract_citations_from_results(entities: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract citation information from entity results"""
    citations = []
    citation_id = 1
    
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            citation = {
                "id": citation_id,
                "name": entity.get("name", "Unknown"),
                "type": entity_type,
                "entity_id": entity.get("id", ""),
                "source": entity.get("source", "unknown"),
                "relevance": entity.get("similarity", entity.get("score", 0))
            }
            citations.append(citation)
            citation_id += 1
    
    return citations

def count_vector_results(vector_results: Dict) -> int:
    """Count total entities from vector search"""
    return sum(len(entities) for entities in vector_results.values())

def count_graph_results(graph_results: Dict) -> int:
    """Count total entities from graph traversal"""
    return sum(len(entities) for entities in graph_results.values())

def count_unique_entities(entities: Dict[str, List[Dict]]) -> int:
    """Count total unique entities across all types"""
    unique_ids = set()
    for entity_list in entities.values():
        for entity in entity_list:
            entity_id = entity.get("id")
            if entity_id:
                unique_ids.add(entity_id)
    return len(unique_ids)

def format_results_for_llm(aggregated_results: Dict[str, Any], max_context_length: int = 4000) -> str:
    """Format aggregated results for LLM consumption"""
    try:
        context = aggregated_results.get("context", "")
        
        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... (truncated)"
        
        # Add metadata
        metadata = aggregated_results.get("metadata", {})
        context += f"\n\nMetadata: {metadata.get('total_unique_entities', 0)} entities analyzed"
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to format results for LLM: {e}")
        return "Error formatting results for analysis"

def generate_followup_questions(aggregated_results: Dict[str, Any], query_context: Dict) -> List[str]:
    """Generate relevant follow-up questions based on results"""
    try:
        query_type = query_context.get("query_type", "general")
        entities = aggregated_results.get("entities", {})
        followups = []
        # Type-specific followups
        if query_type == "comparison":
            followups.extend([
                "What are the underlying mechanisms that cause these differences?",
                "Which patient populations would benefit most from each option?",
                "Are there any contraindications to consider?"
            ])
        elif query_type == "mechanism":
            followups.extend([
                "What are the downstream effects of this mechanism?",
                "Are there any alternative pathways involved?",
                "How does genetic variation affect this mechanism?"
            ])
        elif query_type == "safety":
            followups.extend([
                "What is the frequency of these adverse effects?",
                "Are there any drug interactions to be aware of?",
                "What monitoring is recommended?"
            ])
        # Entity-based followups (always add, even if over 3)
        entity_followups = []
        if entities.get("drugs"):
            entity_followups.append("What other conditions could these drugs potentially treat?")
        if entities.get("proteins"):
            entity_followups.append("What other drugs target these same proteins?")
        # Combine and limit to 3, but always prefer at least one entity-based if present
        combined = followups + entity_followups
        # If entity-based present and not in first 3, swap in
        if entity_followups:
            # Ensure at least one entity-based in the first 3
            for ef in entity_followups:
                if ef not in combined[:3]:
                    combined = combined[:2] + [ef]
                    break
        return combined[:3]
    except Exception as e:
        logger.error(f"Failed to generate followup questions: {e}")
        return ["What additional information would be helpful?"]
