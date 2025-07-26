
import logging
import re
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

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
    """Classify query type based on content analysis"""
    query_lower = query.lower()
    classification_rules = [
        (re.compile(r"\b(compare|vs|versus|difference|better|prefer)\b", re.I), "comparison", 0.9),
        (re.compile(r"\b(mechanism|how does|pathway|targets|works)\b", re.I), "mechanism", 0.8),
        (re.compile(r"\b(side effect(s)?|adverse|toxicity|safety|risk)\b", re.I), "safety", 0.8),
        (re.compile(r"\b(repurpose|new use|alternative|off-label)\b", re.I), "repurposing", 0.7),
        (re.compile(r"\b(cause|associated|related|linked)\b", re.I), "association", 0.8),
        (re.compile(r"\b(interact|interaction|combine|together|with)\b", re.I), "interaction", 0.7),
        (re.compile(r"\b(treat|therapy|treatment|cure)\b", re.I), "treatment", 0.6)
    ]
    best_match = ("general", 0.0)
    for pattern, category, confidence in classification_rules:
        if pattern.search(query_lower):
            if confidence > best_match[1]:
                best_match = (category, confidence)
    return best_match[0]

def enhanced_entity_extraction(query: str) -> Dict[str, List[str]]:
    """Enhanced entity extraction with medical terminology."""
    """Enhanced entity extraction with medical terminology."""
    entities = {"drugs": [], "diseases": [], "proteins": []}
    # Compile regex patterns once
    drug_patterns = [
        re.compile(r'\b[A-Z][a-z]+(?:mab|ine|ib|ol|pril|sartan|ide|in|cin|stat)\b', re.I),
        re.compile(r'\b(?:ACE inhibitor|ARB|beta.?blocker|statin|NSAID|antibiotic)s?\b', re.I),
        re.compile(r'\b(?:aspirin|ibuprofen|metformin|insulin|warfarin|prednisone)\b', re.I)
    ]
    disease_patterns = [
        re.compile(r'\b[A-Z][a-z]+(?:osis|itis|emia|oma|pathy|trophy|ism)\b', re.I),
        re.compile(r'\b(?:diabetes|hypertension|cancer|disease|syndrome|disorder)\b', re.I),
        re.compile(r'\b(?:COVID-19|HIV|AIDS|COPD|ADHD|PTSD)\b', re.I)
    ]
    protein_patterns = [
        re.compile(r'\b[A-Z][A-Z0-9]+\b', re.I),  # Protein abbreviations
        re.compile(r'\b(?:protein|enzyme|receptor|kinase|antibody)\b', re.I)
    ]
    for pattern in drug_patterns:
        entities["drugs"].extend(pattern.findall(query))
    for pattern in disease_patterns:
        entities["diseases"].extend(pattern.findall(query))
    for pattern in protein_patterns:
        entities["proteins"].extend(pattern.findall(query))
    # Clean and deduplicate
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))
    return entities


def preprocess_query(query: str, max_results: int = 15) -> ProcessedQuery:
    """Enhanced query preprocessing with strategy determination."""
    try:
        cleaned_query = query.strip()
        query_type = classify_query_type(cleaned_query)
        extracted_entities = enhanced_entity_extraction(cleaned_query)
        strategy = determine_enhanced_strategy(query_type, extracted_entities)
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
    if query_type == "comparison" and total_entities >= 2:
        return QueryStrategy.COMPARATIVE_ANALYSIS
    if query_type in ["mechanism", "interaction"] and total_entities >= 1:
        return QueryStrategy.PATHWAY_TRAVERSAL
    if query_type == "repurposing":
        return QueryStrategy.COMMUNITY_ANALYSIS
    if total_entities == 1:
        return QueryStrategy.ENTITY_EXPANSION
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