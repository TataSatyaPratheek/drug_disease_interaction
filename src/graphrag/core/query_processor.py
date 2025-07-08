import streamlit as st
import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

def classify_query_type(query: str) -> str:
    """Classify query type based on content analysis"""
    query_lower = query.lower()
    
    # Drug comparison queries
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    
    # Mechanism queries
    if any(word in query_lower for word in ["mechanism", "how does", "pathway", "targets"]):
        return "mechanism"
    
    # Side effect queries
    if any(word in query_lower for word in ["side effect", "adverse", "toxicity", "safety"]):
        return "safety"
    
    # Drug discovery queries
    if any(word in query_lower for word in ["repurpose", "new use", "alternative"]):
        return "discovery"
    
    # Interaction queries
    if any(word in query_lower for word in ["interaction", "combine", "together"]):
        return "interaction"
    
    return "general"

def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
    """Extract potential entity names from query text"""
    # This is a simple implementation - could be enhanced with NER
    entities = {
        "drugs": [],
        "diseases": [],
        "proteins": []
    }
    
    # Common drug patterns
    drug_patterns = [
        r'\b[A-Z][a-z]+(?:mab|ine|ib|ol|pril|sartan|ide|in)\b',  # Common drug suffixes
        r'\b(?:ACE inhibitor|ARB|beta.?blocker|statin|NSAID)s?\b'  # Drug classes
    ]
    
    for pattern in drug_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["drugs"].extend(matches)
    
    # Disease patterns (basic)
    disease_patterns = [
        r'\b[A-Z][a-z]+(?:osis|itis|emia|oma|pathy|trophy)\b',
        r'\b(?:diabetes|hypertension|cancer|disease|syndrome|disorder)\b'
    ]
    
    for pattern in disease_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["diseases"].extend(matches)
    
    # Remove duplicates and clean
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))
    
    return entities

@st.cache_data(hash_funcs={"builtins.str": str})
def preprocess_query(query: str, max_results: int = 15) -> Dict[str, Any]:
    """Preprocess query and extract parameters"""
    try:
        # Basic cleaning
        cleaned_query = query.strip()
        
        # Classify query
        query_type = classify_query_type(cleaned_query)
        
        # Extract entities
        extracted_entities = extract_entities_from_query(cleaned_query)
        
        # Determine search strategy
        search_strategy = determine_search_strategy(query_type, extracted_entities)
        
        preprocessing_result = {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "query_type": query_type,
            "extracted_entities": extracted_entities,
            "search_strategy": search_strategy,
            "max_results": max_results,
            "preprocessing_success": True
        }
        
        logger.info(f"Query preprocessed: type={query_type}, entities={sum(len(v) for v in extracted_entities.values())}")
        return preprocessing_result
        
    except Exception as e:
        logger.error(f"Query preprocessing failed: {e}")
        return {
            "original_query": query,
            "cleaned_query": query,
            "query_type": "general",
            "extracted_entities": {"drugs": [], "diseases": [], "proteins": []},
            "search_strategy": "vector_only",
            "max_results": max_results,
            "preprocessing_success": False,
            "error": str(e)
        }

def determine_search_strategy(query_type: str, extracted_entities: Dict[str, List[str]]) -> str:
    """Determine the best search strategy based on query characteristics"""
    total_entities = sum(len(entities) for entities in extracted_entities.values())
    
    if query_type == "comparison" and total_entities >= 2:
        return "comparative_analysis"
    elif query_type == "mechanism" and total_entities >= 1:
        return "pathway_traversal"
    elif total_entities == 0:
        return "vector_only"
    elif total_entities == 1:
        return "entity_expansion"
    else:
        return "multi_entity_analysis"

def route_query(preprocessing_result: Dict[str, Any]) -> Dict[str, Any]:
    """Route query to appropriate processing pipeline"""
    strategy = preprocessing_result.get("search_strategy", "vector_only")
    query_type = preprocessing_result.get("query_type", "general")
    
    routing_plan = {
        "strategy": strategy,
        "query_type": query_type,
        "use_vector_search": True,
        "use_graph_traversal": strategy in ["pathway_traversal", "comparative_analysis", "multi_entity_analysis"],
        "use_community_detection": strategy == "multi_entity_analysis",
        "use_path_finding": strategy in ["comparative_analysis", "pathway_traversal"],
        "expected_response_sections": get_expected_response_sections(query_type)
    }
    
    return routing_plan

def get_expected_response_sections(query_type: str) -> List[str]:
    """Define expected sections in response based on query type"""
    base_sections = ["reasoning", "response"]
    
    type_specific_sections = {
        "comparison": ["comparative_table", "key_differences"],
        "mechanism": ["pathway_diagram", "molecular_targets"],
        "safety": ["adverse_effects", "contraindications"],
        "discovery": ["repurposing_opportunities", "evidence_level"],
        "interaction": ["interaction_effects", "clinical_significance"]
    }
    
    return base_sections + type_specific_sections.get(query_type, [])
