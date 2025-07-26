"""
Graph Analytics - Neo4j Native Implementation
Clean import interface for the new Neo4j-based analytics
"""

# Import the new Neo4j-native implementations
from .neo4j_analytics import (
    Neo4jGraphAnalytics,
    HighPerformanceGraphAnalytics,
    GraphMetrics
)

# Maintain compatibility with existing imports
__all__ = [
    'Neo4jGraphAnalytics',
    'HighPerformanceGraphAnalytics', 
    'GraphMetrics'
]
