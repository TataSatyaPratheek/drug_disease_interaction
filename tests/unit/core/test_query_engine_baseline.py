"""
Unit tests for Neo4j-native GraphRAG Core Query Engine
Tests the new streamlined implementation optimized for Ryzen 4800H + GTX 1650Ti
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class TestNeo4jGraphRAGQueryEngine:
    """Test the new streamlined Neo4j-native query engine"""
    
    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver"""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value = session
        
        # Mock some sample data
        session.run.return_value = [
            {"n": {"id": "drug_1", "name": "Aspirin", "type": "Drug"}},
            {"n": {"id": "disease_1", "name": "Arthritis", "type": "Disease"}}
        ]
        return driver
    
    @pytest.fixture
    def mock_weaviate_store(self):
        """Mock Weaviate vector store"""
        store = MagicMock()
        store.search_entities.return_value = [
            {"id": "drug_1", "name": "Aspirin", "type": "drug", "similarity": 0.9}
        ]
        return store
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock optimized LLM client"""
        client = MagicMock()
        client.generate_response.return_value = "Test response from optimized client"
        return client
    
    def test_streamlined_query_engine_creation(self, mock_neo4j_driver, mock_weaviate_store, mock_llm_client):
        """Test that we can create a streamlined query engine"""
        # This test will drive the design of our new streamlined engine
        from graphrag.core.neo4j_query_engine import Neo4jGraphRAGEngine
        
        engine = Neo4jGraphRAGEngine(
            neo4j_driver=mock_neo4j_driver,
            vector_store=mock_weaviate_store,
            llm_client=mock_llm_client
        )
        
        assert engine is not None
        assert hasattr(engine, 'neo4j_driver')
        assert hasattr(engine, 'vector_store')
        assert hasattr(engine, 'llm_client')
    
    def test_simple_query_processing(self, mock_neo4j_driver, mock_weaviate_store, mock_llm_client):
        """Test streamlined query processing"""
        from graphrag.core.neo4j_query_engine import Neo4jGraphRAGEngine
        
        engine = Neo4jGraphRAGEngine(
            neo4j_driver=mock_neo4j_driver,
            vector_store=mock_weaviate_store,
            llm_client=mock_llm_client
        )
        
        result = engine.query("What is aspirin used for?")
        
        assert result is not None
        assert 'response' in result
        assert 'entities' in result
        assert 'confidence' in result
    
    def test_performance_optimizations_applied(self, mock_neo4j_driver, mock_weaviate_store, mock_llm_client):
        """Test that performance optimizations are properly applied"""
        from graphrag.core.neo4j_query_engine import Neo4jGraphRAGEngine
        
        engine = Neo4jGraphRAGEngine(
            neo4j_driver=mock_neo4j_driver,
            vector_store=mock_weaviate_store,
            llm_client=mock_llm_client,
            max_workers=4,  # Ryzen 4800H optimization
            enable_caching=True
        )
        
        # Should have threading capabilities
        assert hasattr(engine, 'max_workers')
        assert engine.max_workers == 4
        
        # Should have caching
        assert hasattr(engine, 'enable_caching')
        assert engine.enable_caching is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
