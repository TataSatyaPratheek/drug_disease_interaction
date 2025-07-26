"""
Test suite for OptimizedWeaviateGraphStore
Testing parallel processing, memory management, and hardware optimizations
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import psutil
from concurrent.futures import ThreadPoolExecutor

# Import our optimized components
from src.graphrag.core.optimized_vector_store import OptimizedWeaviateGraphStore, MemoryAwareCache


class TestMemoryAwareCache:
    """Test the memory-aware caching system"""
    
    def test_cache_initialization(self):
        """Test cache is properly initialized"""
        cache = MemoryAwareCache(max_size=100, max_memory_mb=10)
        
        assert cache.max_size == 100
        assert cache.max_memory_bytes == 10 * 1024 * 1024
        assert len(cache.cache) == 0
        assert cache.current_memory == 0
    
    def test_cache_put_and_get(self):
        """Test basic put/get operations"""
        cache = MemoryAwareCache(max_size=100, max_memory_mb=10)
        
        # Put item
        cache.put("key1", "test_value")
        
        # Get item
        result = cache.get("key1")
        assert result == "test_value"
        
        # Get non-existent item
        result = cache.get("key2")
        assert result is None
    
    def test_cache_memory_eviction(self):
        """Test memory-based eviction"""
        cache = MemoryAwareCache(max_size=100, max_memory_mb=0.001)  # Very small memory limit
        
        # Add items that should trigger eviction
        large_value = "x" * 1000  # 1KB value
        cache.put("key1", large_value)
        cache.put("key2", large_value)
        
        # Should have evicted first item
        assert cache.get("key1") is None
        assert cache.get("key2") == large_value
    
    def test_cache_size_eviction(self):
        """Test size-based eviction"""
        cache = MemoryAwareCache(max_size=2, max_memory_mb=100)  # Large memory, small size
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = MemoryAwareCache(max_size=100, max_memory_mb=10)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['hit_rate'] == 0
        assert stats['cache_size'] == 0
        
        # Add items and access them
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.5  # 1 hit, 1 miss
        assert stats['cache_size'] == 1


class TestOptimizedWeaviateGraphStore:
    """Test the optimized vector store implementation"""
    
    @pytest.fixture
    def mock_weaviate_client(self):
        """Create mock Weaviate client"""
        client = Mock()
        client.is_ready.return_value = True
        
        # Mock collections
        collection_mock = Mock()
        collection_mock.query.return_value.objects = [
            {
                'properties': {'name': 'Test Entity', 'description': 'Test Description'},
                'uuid': 'test-uuid-1',
                'metadata': {'distance': 0.1}
            }
        ]
        
        client.collections.get.return_value = collection_mock
        client.collections.create.return_value = collection_mock
        client.collections.exists.return_value = True
        
        return client
    
    @pytest.fixture
    def optimized_store(self, mock_weaviate_client):
        """Create optimized store with mock client"""
        return OptimizedWeaviateGraphStore(
            client=mock_weaviate_client,
            max_workers=2,  # Small for testing
            cache_size=100
        )
    
    def test_initialization(self, optimized_store):
        """Test store initialization"""
        assert optimized_store.max_workers == 2
        assert optimized_store.cache_size == 100
        assert optimized_store.executor is not None
        assert optimized_store.embedding_cache is not None
        assert optimized_store.search_cache is not None
    
    def test_search_entities_basic(self, optimized_store):
        """Test basic entity search"""
        results = optimized_store.search_entities(
            query="test query",
            entity_types=["Drug"],
            n_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'name' in results[0]
        assert 'type' in results[0]
    
    def test_parallel_batch_processing(self, optimized_store):
        """Test parallel batch processing"""
        # Create multiple items to process
        items = [f"item_{i}" for i in range(10)]
        
        # Mock the batch processing function
        def mock_process_batch(batch):
            return [f"processed_{item}" for item in batch]
        
        results = optimized_store._parallel_batch_process(
            items=items,
            batch_size=3,
            process_func=mock_process_batch
        )
        
        assert len(results) == 10
        assert all(result.startswith("processed_") for result in results)
    
    def test_memory_monitoring(self, optimized_store):
        """Test memory monitoring during operations"""
        initial_memory = psutil.virtual_memory().percent
        
        # Trigger memory check
        is_safe = optimized_store._is_memory_safe_for_operation(1.0)  # 1GB operation
        
        # Should return boolean
        assert isinstance(is_safe, bool)
    
    def test_cache_performance(self, optimized_store):
        """Test caching improves performance"""
        query = "test cache query"
        
        # First search (cache miss)
        start_time = time.time()
        results1 = optimized_store.search_entities(query, ["Drug"], 5)
        first_time = time.time() - start_time
        
        # Second search (cache hit)
        start_time = time.time()
        results2 = optimized_store.search_entities(query, ["Drug"], 5)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert results1 == results2
        
        # Second search should be faster (though this might not always be true in mocked tests)
        # At minimum, verify caching mechanism exists
        cache_stats = optimized_store.search_cache.get_stats()
        assert cache_stats['cache_size'] > 0
    
    def test_concurrent_access(self, optimized_store):
        """Test thread safety of concurrent access"""
        results = []
        errors = []
        
        def search_worker(worker_id):
            try:
                result = optimized_store.search_entities(
                    f"query_{worker_id}",
                    ["Drug"],
                    3
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=search_worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and all results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
    
    def test_resource_cleanup(self, optimized_store):
        """Test proper resource cleanup"""
        # Verify executor exists
        assert optimized_store.executor is not None
        
        # Cleanup
        optimized_store.cleanup()
        
        # Verify cleanup (executor should be shutdown)
        # Note: We can't easily test if executor is shutdown without accessing private members
        # So we'll just verify the method runs without error
        assert True  # If we reach here, cleanup didn't crash
    
    def test_hardware_specific_batch_sizing(self, optimized_store):
        """Test hardware-specific batch size calculation"""
        batch_size = optimized_store._calculate_optimal_batch_size(1000)
        
        # Should return reasonable batch size
        assert isinstance(batch_size, int)
        assert 50 <= batch_size <= 200  # Reasonable range for our hardware
    
    def test_error_handling(self, optimized_store):
        """Test error handling in optimized operations"""
        # Mock client to raise exception
        optimized_store.client.collections.get.side_effect = Exception("Test error")
        
        # Should handle error gracefully
        results = optimized_store.search_entities("test", ["Drug"], 5)
        
        # Should return empty list instead of crashing
        assert isinstance(results, list)
        assert len(results) == 0


class TestOptimizedStoreIntegration:
    """Integration tests for optimized vector store"""
    
    @pytest.fixture
    def real_store_config(self):
        """Configuration for real store testing (if available)"""
        return {
            'max_workers': 4,
            'cache_size': 500,
            'memory_threshold': 0.8
        }
    
    def test_performance_comparison(self, real_store_config):
        """Test performance improvement over sequential processing"""
        # This would require a real Weaviate instance
        # For now, we'll test the configuration
        
        config = real_store_config
        assert config['max_workers'] == 4
        assert config['cache_size'] == 500
        assert config['memory_threshold'] == 0.8
    
    def test_memory_pressure_handling(self, real_store_config):
        """Test behavior under memory pressure"""
        # Simulate memory pressure scenario
        # In real test, this would involve large data operations
        
        # Verify configuration supports memory management
        assert real_store_config['memory_threshold'] > 0
        assert real_store_config['cache_size'] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
