
# src/api/models/requests.py - ENHANCED WITH VALIDATION
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class QueryType(str, Enum):
    GENERAL = "general"
    DRUG_SEARCH = "drug_search"
    DISEASE_SEARCH = "disease_search"
    INTERACTION = "drug_disease_interaction"
    PATHWAY = "pathway_analysis"

class HybridSearchRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="Search query",
        example="What drugs treat hypertension?"
    )
    query_type: Optional[QueryType] = Field(
        default=QueryType.GENERAL,
        description="Type of search query"
    )
    max_results: int = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Maximum results to return"
    )
    include_paths: bool = Field(
        default=True, 
        description="Include graph paths in results"
    )
    include_similarity: bool = Field(
        default=True, 
        description="Include vector similarity search"
    )
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v.strip()

class EntityDetailsRequest(BaseModel):
    entity_id: str = Field(
        ..., 
        min_length=1,
        description="Entity ID to get details for",
        example="DB00001"
    )
