import streamlit as st
import logging
from typing import Dict, Any, List
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryStrategy(Enum):
    """Enhanced query processing strategies."""
    VECTOR_ONLY = "vector_only"
    ENTITY_EXPANSION = "entity_expansion"
    PATHWAY_TRAVERSAL = "pathway_traversal"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    MULTI_ENTITY_ANALYSIS = "multi_entity_analysis"
    COMMUNITY_ANALYSIS = "community_analysis"

@dataclass
class ProcessedQuery:
    """Structured query processing result."""
    original_query: str
    cleaned_query: str
    query_type: str
    strategy: QueryStrategy
    extracted_entities: Dict[str, List[str]]
    confidence: float
    reasoning: str

def classify_query_type(query: str) -> str:
    """Classify query type based on content analysis"""
    query_lower = query.lower()

    # Enhanced classification with confidence scoring
    classification_rules = [
        (["compare", "vs", "versus", "difference", "better", "prefer"], "comparison", 0.9),
        (["mechanism", "how does", "pathway", "targets", "works"], "mechanism", 0.8),
        (["side effect", "adverse", "toxicity", "safety", "risk"], "safety", 0.8),
        (["repurpose", "new use", "alternative", "off-label"], "repurposing", 0.7),
        (["interaction", "combine", "together", "with"], "interaction", 0.7),
        (["treat", "therapy", "treatment", "cure"], "treatment", 0.6),
        (["cause", "associated", "related", "linked"], "association", 0.6)
    ]

    best_match = ("general", 0.0)

    for keywords, category, confidence in classification_rules:
        if any(keyword in query_lower for keyword in keywords):
            if confidence > best_match[1]:
                best_match = (category, confidence)

    return best_match[0]

def enhanced_entity_extraction(query: str) -> Dict[str, List[str]]:
    """Enhanced entity extraction with medical terminology."""
    entities = {"drugs": [], "diseases": [], "proteins": []}

    # Enhanced drug patterns
    drug_patterns = [
        r'\b[A-Z][a-z]+(?:mab|ine|ib|ol|pril|sartan|ide|in|cin|stat)\b',
        r'\b(?:ACE inhibitor|ARB|beta.?blocker|statin|NSAID|antibiotic)s?\b',
        r'\b(?:aspirin|ibuprofen|metformin|insulin|warfarin|prednisone)\b'
    ]

    # Enhanced disease patterns
    disease_patterns = [
        r'\b[A-Z][a-z]+(?:osis|itis|emia|oma|pathy|trophy|ism)\b',
        r'\b(?:diabetes|hypertension|cancer|disease|syndrome|disorder)\b',
        r'\b(?:COVID-19|HIV|AIDS|COPD|ADHD|PTSD)\b'
    ]

    # Enhanced protein patterns
    protein_patterns = [
        r'\b[A-Z][A-Z0-9]+\b',  # Protein abbreviations
        r'\b(?:protein|enzyme|receptor|kinase|antibody)\b'
    ]

    # Extract with patterns
    for pattern in drug_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["drugs"].extend(matches)

    for pattern in disease_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["diseases"].extend(matches)

    for pattern in protein_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["proteins"].extend(matches)

    # Clean and deduplicate
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))

    return entities

@st.cache_data(hash_funcs={"builtins.str": str})
def preprocess_query(query: str, max_results: int = 15) -> ProcessedQuery:
    """Enhanced query preprocessing with strategy determination."""
    try:
        # Basic cleaning
        cleaned_query = query.strip()

        # Enhanced classification
        query_type = classify_query_type(cleaned_query)

        # Enhanced entity extraction
        extracted_entities = enhanced_entity_extraction(cleaned_query)

        # Determine optimal strategy
        strategy = determine_enhanced_strategy(query_type, extracted_entities)

        # Generate reasoning
        reasoning = generate_processing_reasoning(query_type, extracted_entities, strategy)

        preprocessing_result = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            query_type=query_type,
            strategy=strategy,
            extracted_entities=extracted_entities,
            confidence=0.8,
            reasoning=reasoning
        )

        logger.info(f"Enhanced query processing: type={query_type}, strategy={strategy}")
        return preprocessing_result

    except Exception as e:
        logger.error(f"Query preprocessing failed: {e}")
        return ProcessedQuery(
            original_query=query,
            cleaned_query=query,
            query_type="general",
            strategy=QueryStrategy.VECTOR_ONLY,
            extracted_entities={"drugs": [], "diseases": [], "proteins": []},
            confidence=0.1,
            reasoning=f"Processing failed: {str(e)}"
        )

def determine_enhanced_strategy(query_type: str, extracted_entities: Dict[str, List[str]]) -> QueryStrategy:
    """Determine optimal search strategy with enhanced logic."""
    total_entities = sum(len(entities) for entities in extracted_entities.values())

    # Enhanced strategy mapping
    if query_type == "comparison" and total_entities >= 2:
        return QueryStrategy.COMPARATIVE_ANALYSIS
    elif query_type in ["mechanism", "interaction"] and total_entities >= 1:
        return QueryStrategy.PATHWAY_TRAVERSAL
    elif query_type == "repurposing":
        return QueryStrategy.COMMUNITY_ANALYSIS
    elif total_entities == 0:
        return QueryStrategy.VECTOR_ONLY
    elif total_entities == 1:
        return QueryStrategy.ENTITY_EXPANSION
    else:
        return QueryStrategy.MULTI_ENTITY_ANALYSIS

def generate_processing_reasoning(query_type: str, entities: Dict, strategy: QueryStrategy) -> str:
    """Generate reasoning explanation for query processing decisions."""
    entity_count = sum(len(v) for v in entities.values())

    reasoning = f"Query classified as '{query_type}' with {entity_count} extracted entities. "
    reasoning += f"Selected strategy: {strategy.value}. "

    if strategy == QueryStrategy.COMPARATIVE_ANALYSIS:
        reasoning += "Will retrieve both entities and compare their properties."
    elif strategy == QueryStrategy.PATHWAY_TRAVERSAL:
        reasoning += "Will explore molecular pathways and mechanisms."
    elif strategy == QueryStrategy.COMMUNITY_ANALYSIS:
        reasoning += "Will analyze community structure for repurposing opportunities."

    return reasoning