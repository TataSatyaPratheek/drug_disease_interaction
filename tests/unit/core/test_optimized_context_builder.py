"""
Test suite for OptimizedContextBuilder  
Testing LRU caching, parallel context assembly, and memory management
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import psutil

from src.graphrag.core.optimized_context_builder import (
    ContextFragment,
    MemoryAwareContextCache,
    OptimizedContextBuilder,
    create_optimized_context_builder
)


class TestContextFragment:
    """Test context fragment functionality"""
    
    def test_fragment_creation(self):
        """Test fragment creation and properties"""
        fragment = ContextFragment(
            content="Test content",
            source="test_source",
            relevance=0.8
        )
        
        assert fragment.content == "Test content"
        assert fragment.source == "test_source" 
        assert fragment.relevance == 0.8
        assert fragment.size > 0
        assert fragment.hash is not None
        assert fragment.timestamp > 0
    
    def test_fragment_hash_consistency(self):
        """Test fragment hashing for deduplication"""
        fragment1 = ContextFragment("Same content", "source1")
        fragment2 = ContextFragment("Same content", "source2")  # Different source
        fragment3 = ContextFragment("Different content", "source1")
        
        # Same content should have same hash
        assert fragment1.hash == fragment2.hash
        
        # Different content should have different hash
        assert fragment1.hash != fragment3.hash
    
    def test_fragment_equality(self):
        """Test fragment equality based on content"""
        fragment1 = ContextFragment("Test content", "source1")
        fragment2 = ContextFragment("Test content", "source2") 
        fragment3 = ContextFragment("Different content", "source1")
        
        # Same content should be equal
        assert fragment1 == fragment2
        
        # Different content should not be equal
        assert fragment1 != fragment3
    
    def test_fragment_size_calculation(self):
        """Test fragment size calculation"""
        short_fragment = ContextFragment("short", "test")
        long_fragment = ContextFragment("x" * 1000, "test")
        
        assert short_fragment.size < long_fragment.size
        assert long_fragment.size >= 1000


class TestMemoryAwareContextCache:
    """Test memory-aware context caching"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = MemoryAwareContextCache(max_size=100, max_memory_mb=10)
        
        assert cache.max_size == 100
        assert cache.max_memory_bytes == 10 * 1024 * 1024
        assert len(cache.cache) == 0
        assert cache.current_memory == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_get(self):
        """Test basic cache operations"""
        cache = MemoryAwareContextCache(max_size=100, max_memory_mb=10)
        fragment = ContextFragment("Test content", "test_source")
        
        # Put fragment
        cache.put("key1", fragment)
        
        # Get fragment
        result = cache.get("key1")
        assert result == fragment
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Get non-existent key
        result = cache.get("key2")
        assert result is None
        assert cache.hits == 1
        assert cache.misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = MemoryAwareContextCache(max_size=2, max_memory_mb=100)
        
        fragment1 = ContextFragment("Content 1", "source1")
        fragment2 = ContextFragment("Content 2", "source2")
        fragment3 = ContextFragment("Content 3", "source3")
        
        # Fill cache
        cache.put("key1", fragment1)
        cache.put("key2", fragment2)
        
        # Access key1 to make it more recent
        cache.get("key1")
        
        # Add key3, should evict key2 (LRU)
        cache.put("key3", fragment3)
        
        assert cache.get("key1") == fragment1  # Still there
        assert cache.get("key2") is None       # Evicted
        assert cache.get("key3") == fragment3  # New item
    
    def test_cache_memory_eviction(self):
        """Test memory-based eviction"""
        cache = MemoryAwareContextCache(max_size=100, max_memory_mb=0.001)  # Very small
        
        large_content = "x" * 1000  # 1KB content
        fragment1 = ContextFragment(large_content, "source1")
        fragment2 = ContextFragment(large_content, "source2") 
        
        cache.put("key1", fragment1)
        cache.put("key2", fragment2)  # Should trigger memory eviction
        
        # First item should be evicted due to memory pressure
        assert cache.get("key1") is None
        assert cache.get("key2") == fragment2
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = MemoryAwareContextCache(max_size=100, max_memory_mb=10)
        fragment = ContextFragment("Test content", "test_source")
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['hit_rate'] == 0
        assert stats['cache_size'] == 0
        
        # Add and access items
        cache.put("key1", fragment)
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.5
        assert stats['cache_size'] == 1
        assert stats['memory_usage_mb'] > 0


class TestOptimizedContextBuilder:
    """Test optimized context builder"""
    
    @pytest.fixture
    def context_builder(self):
        """Create context builder for testing"""
        return OptimizedContextBuilder(
            max_workers=2,  # Small for testing
            cache_size=100,
            max_context_length=1000
        )
    
    def test_builder_initialization(self, context_builder):
        """Test context builder initialization"""
        assert context_builder.max_workers == 2
        assert context_builder.max_context_length == 1000
        assert context_builder.fragment_cache is not None
        assert context_builder.entity_cache is not None
        assert context_builder.executor is not None
    
    def test_build_context_parallel_basic(self, context_builder):
        """Test basic parallel context building"""
        entities = ["entity1", "entity2"]
        relationships = [
            {"source": "entity1", "target": "entity2", "type": "interacts_with"}
        ]
        vector_results = [
            {"content": "Vector result 1", "score": 0.9},
            {"content": "Vector result 2", "score": 0.8}
        ]
        
        context = context_builder.build_context_parallel(
            entities=entities,
            relationships=relationships,
            vector_results=vector_results,
            max_length=500
        )
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert len(context) <= 500 + 100  # Allow for format overhead
        assert "CONTEXT" in context
    
    def test_build_context_with_caching(self, context_builder):
        """Test context building with caching"""
        entities = ["cached_entity"]
        relationships = []
        vector_results = []
        
        # First build
        context1 = context_builder.build_context_parallel(
            entities=entities,
            relationships=relationships, 
            vector_results=vector_results
        )
        
        # Second build (should use cache)
        context2 = context_builder.build_context_parallel(
            entities=entities,
            relationships=relationships,
            vector_results=vector_results
        )
        
        assert context1 == context2
        
        # Check cache was used
        cache_stats = context_builder.entity_cache.get_stats()
        assert cache_stats['hit_rate'] > 0
    
    def test_fragment_deduplication(self, context_builder):
        """Test fragment deduplication"""
        # Create duplicate entities
        entities = ["duplicate_entity", "duplicate_entity", "unique_entity"]
        relationships = []
        vector_results = []
        
        context = context_builder.build_context_parallel(
            entities=entities,
            relationships=relationships,
            vector_results=vector_results
        )
        
        # Should not contain duplicates
        assert isinstance(context, str)
        # Count occurrences of entity in context (rough check)
        entity_count = context.count("duplicate_entity")
        assert entity_count <= 2  # Should appear at most twice due to deduplication
    
    def test_relevance_based_selection(self, context_builder):
        """Test relevance-based fragment selection"""
        entities = []
        relationships = []
        vector_results = [
            {"content": "High relevance", "score": 0.9, "entity": "entity1"},
            {"content": "Low relevance", "score": 0.1, "entity": "entity2"}
        ]
        
        # Build with very small max length to force selection
        context = context_builder.build_context_parallel(
            entities=entities,
            relationships=relationships,
            vector_results=vector_results,
            max_length=100
        )
        
        # High relevance content should be preferred
        assert "High relevance" in context
    
    def test_concurrent_context_building(self, context_builder):
        """Test thread safety of context building"""
        results = []
        errors = []
        
        def build_worker(worker_id):
            try:
                entities = [f"entity_{worker_id}"]
                relationships = []
                vector_results = [{"content": f"Result {worker_id}", "score": 0.8}]
                
                context = context_builder.build_context_parallel(
                    entities=entities,
                    relationships=relationships,
                    vector_results=vector_results
                )
                results.append(context)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=build_worker, args=(i,))
            threads.append(thread)
        
        # Start and wait for threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify no errors and all results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(isinstance(result, str) for result in results)
    
    def test_performance_tracking(self, context_builder):
        """Test performance tracking"""
        entities = ["test_entity"]
        relationships = []
        vector_results = []
        
        # Build context multiple times
        for i in range(3):
            context_builder.build_context_parallel(
                entities=entities,
                relationships=relationships,
                vector_results=vector_results
            )
        
        # Check performance stats
        stats = context_builder.get_performance_stats()
        assert stats['total_builds'] == 3
        assert stats['avg_build_time'] > 0
        assert stats['avg_fragments'] >= 0
    
    def test_cleanup(self, context_builder):
        """Test resource cleanup"""
        # Verify executor exists
        assert context_builder.executor is not None
        
        # Cleanup
        context_builder.cleanup()
        
        # Should not raise exception
        assert True


class TestContextBuilderFactory:
    """Test context builder factory function"""
    
    def test_factory_creation(self):
        """Test factory function creates valid builder"""
        builder = create_optimized_context_builder()
        
        assert isinstance(builder, OptimizedContextBuilder)
        assert builder.max_workers >= 2
        assert builder.cache_size > 0
        assert builder.max_context_length > 0
    
    def test_factory_with_custom_workers(self):
        """Test factory with custom worker count"""
        builder = create_optimized_context_builder(max_workers=8)
        
        assert builder.max_workers == 8
    
    @patch('src.graphrag.core.optimized_context_builder.psutil.cpu_count')
    @patch('src.graphrag.core.optimized_context_builder.psutil.virtual_memory')
    def test_factory_hardware_adaptation(self, mock_memory, mock_cpu):
        """Test factory adapts to hardware"""
        # Mock hardware specs
        mock_cpu.return_value = 16
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        
        builder = create_optimized_context_builder()
        
        # Should use reasonable values for high-end hardware
        assert builder.max_workers >= 2
        assert builder.cache_size > 100


class TestContextBuilderIntegration:
    """Integration tests for context builder"""
    
    def test_end_to_end_context_building(self):
        """Test complete end-to-end context building"""
        builder = create_optimized_context_builder(max_workers=2)
        
        # Realistic test data
        entities = ["Aspirin", "Heart Disease", "COX-2"]
        relationships = [
            {
                "source": "Aspirin",
                "target": "COX-2", 
                "type": "inhibits",
                "description": "Aspirin inhibits COX-2 enzyme"
            },
            {
                "source": "COX-2",
                "target": "Heart Disease",
                "type": "associated_with",
                "description": "COX-2 is associated with cardiovascular disease"
            }
        ]
        vector_results = [
            {
                "content": "Aspirin is a widely used anti-inflammatory drug",
                "entity": "Aspirin",
                "score": 0.9
            },
            {
                "content": "Heart disease is a leading cause of mortality",
                "entity": "Heart Disease", 
                "score": 0.8
            }
        ]
        
        context = builder.build_context_parallel(
            entities=entities,
            relationships=relationships,
            vector_results=vector_results
        )
        
        # Verify context structure and content
        assert isinstance(context, str)
        assert "CONTEXT" in context
        assert "ENTITIES" in context or "Aspirin" in context
        assert "RELATIONSHIPS" in context or "inhibits" in context
        assert "ADDITIONAL CONTEXT" in context or len(vector_results) == 0
        
        # Cleanup
        builder.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
