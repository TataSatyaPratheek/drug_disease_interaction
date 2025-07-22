import streamlit as st
import hashlib
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def ttl_cache(ttl_seconds: int):
    """TTL cache decorator wrapper for Streamlit cache_data"""
    def decorator(func):
        return st.cache_data(ttl=ttl_seconds)(func)
    return decorator

def resource_cache():
    """Resource cache decorator wrapper for Streamlit cache_resource"""
    def decorator(func):
        return st.cache_resource()(func)
    return decorator

def hash_query_params(query: str, max_results: int, query_type: str = "auto") -> str:
    """Generate consistent hash for query parameters"""
    param_string = f"{query}|{max_results}|{query_type}"
    return hashlib.md5(param_string.encode()).hexdigest()

def clear_cache_by_pattern(pattern: str):
    """Clear cached data matching a pattern"""
    try:
        st.cache_data.clear()
        logger.info(f"Cleared cache data matching pattern: {pattern}")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache usage statistics"""
    # Placeholder - Streamlit doesn't expose cache stats directly
    return {
        "cache_data_enabled": True,
        "cache_resource_enabled": True,
        "note": "Detailed cache stats not available in Streamlit"
    }

class CacheConfig:
    """Cache configuration constants"""
    TTL_SERVICE_HEALTH = 300  # 5 minutes
    TTL_VECTOR_STATS = 600    # 10 minutes
    TTL_GRAPH_METRICS = 3600  # 1 hour (static data)
    TTL_QUERY_RESULTS = 1800  # 30 minutes
    TTL_UI_COMPONENTS = 60    # 1 minute
