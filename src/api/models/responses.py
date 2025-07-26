# src/api/models/responses.py - USE PYDANTIC RESPONSE MODELS
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EntityResult(BaseModel):
    id: str
    name: str
    type: str
    score: float
    description: Optional[str] = None
    source: str  # 'neo4j' or 'weaviate'

class HybridSearchResponse(BaseModel):
    query: str
    answer: str
    entities: List[EntityResult]
    total_found: int
    processing_time_ms: float
    sources_used: List[str]
    
class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    database_stats: Dict[str, Any]
