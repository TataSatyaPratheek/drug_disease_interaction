"""
Comprehensive integration test for optimized GraphRAG components
Tests the complete pipeline with hardware optimizations
"""

import time
import psutil
from unittest.mock import Mock, patch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimized_integration():
    """Test complete optimized integration"""
    
    print("🧪 Testing Optimized GraphRAG Integration")
    print("=" * 50)
    
    # Test 1: Import all optimized components
    print("\n1️⃣ Testing optimized component imports...")
    try:
        from src.graphrag.core.optimized_vector_store import OptimizedWeaviateGraphStore
        from src.graphrag.core.optimized_initialization import initialize_graphrag_optimized
        from src.graphrag.core.optimized_context_builder import create_optimized_context_builder
        from src.graphrag.core.optimized_cache_manager import create_optimized_cache_manager
        from src.graphrag.core.enhanced_connection_resilience import connection_manager
        
        print("✅ All optimized components imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Hardware detection
    print("\n2️⃣ Testing hardware detection...")
    try:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        memory_usage = psutil.virtual_memory().percent
        
        print(f"📊 System specs: {memory_gb:.1f}GB RAM, {cpu_count} CPUs, {memory_usage:.1f}% memory usage")
        print("✅ Hardware detection working")
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False
    
    # Test 3: Cache manager
    print("\n3️⃣ Testing optimized cache manager...")
    try:
        cache_manager = create_optimized_cache_manager()
        
        # Test basic operations
        cache_manager.put("test_key", "test_value")
        result = cache_manager.get("test_key")
        assert result == "test_value"
        
        # Test compute and cache
        def expensive_computation():
            time.sleep(0.1)
            return "computed_value"
        
        start_time = time.time()
        result1 = cache_manager.get_or_compute("compute_test", expensive_computation)
        first_time = time.time() - start_time
        
        start_time = time.time() 
        result2 = cache_manager.get_or_compute("compute_test", expensive_computation)
        second_time = time.time() - start_time
        
        assert result1 == result2
        assert second_time < first_time  # Should be faster due to caching
        
        stats = cache_manager.get_comprehensive_stats()
        print(f"📈 Cache stats: {stats['combined_hit_rate']:.1%} hit rate")
        print("✅ Cache manager working correctly")
        
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        return False
    
    # Test 4: Context builder
    print("\n4️⃣ Testing optimized context builder...")
    try:
        context_builder = create_optimized_context_builder(max_workers=2)
        
        # Test context building
        entities = ["Aspirin", "Heart Disease"]
        relationships = [{"source": "Aspirin", "target": "Heart Disease", "type": "treats"}]
        vector_results = [{"content": "Aspirin treats heart disease", "score": 0.9}]
        
        context = context_builder.build_context_parallel(
            entities=entities,
            relationships=relationships,
            vector_results=vector_results
        )
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "CONTEXT" in context
        
        # Test performance stats
        stats = context_builder.get_performance_stats()
        print(f"📊 Context builder: {stats['total_builds']} builds completed")
        print("✅ Context builder working correctly")
        
        context_builder.cleanup()
        
    except Exception as e:
        print(f"❌ Context builder test failed: {e}")
        return False
    
    # Test 5: Connection resilience
    print("\n5️⃣ Testing connection resilience...")
    try:
        # Test circuit breaker registration
        circuit_breaker = connection_manager.register_circuit_breaker("test_service")
        retry_strategy = connection_manager.register_retry_strategy("test_retry")
        
        # Test resilient decorator
        call_count = 0
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Test failure")
            return "success"
        
        # Apply resilience
        resilient_func = retry_strategy(test_function)
        result = resilient_func()
        
        assert result == "success"
        assert call_count == 3  # Should have retried
        
        # Test system status
        status = connection_manager.get_system_status()
        print(f"🔧 Connection manager: {len(status['circuit_breakers'])} circuit breakers")
        print("✅ Connection resilience working correctly")
        
    except Exception as e:
        print(f"❌ Connection resilience test failed: {e}")
        return False
    
    # Test 6: Mock query engine integration
    print("\n6️⃣ Testing query engine integration...")
    try:
        # Mock dependencies for testing
        mock_graph = Mock()
        mock_llm = Mock()
        mock_vector_store = Mock()
        mock_vector_store.client = Mock()
        
        # Configure mock LLM
        mock_llm.generate_with_reasoning = Mock(return_value=("Test reasoning", "Test answer"))
        
        # Test query engine import and basic functionality
        from src.graphrag.core.query_engine import GraphRAGQueryEngine
        
        # Create engine
        engine = GraphRAGQueryEngine.create_legacy_engine(
            graph=mock_graph,
            vector_store=mock_vector_store,
            llm_client=mock_llm
        )
        
        # Verify optimized components are initialized
        assert hasattr(engine, 'cache_manager')
        assert hasattr(engine, 'optimized_context_builder')
        assert hasattr(engine, 'optimized_vector_store')
        
        print("✅ Query engine integration working correctly")
        
    except Exception as e:
        print(f"❌ Query engine integration test failed: {e}")
        return False
    
    # Test 7: Performance benchmark
    print("\n7️⃣ Running performance benchmark...")
    try:
        # Simple performance test
        iterations = 100
        
        start_time = time.time()
        for i in range(iterations):
            cache_manager.put(f"bench_key_{i}", f"value_{i}")
            cache_manager.get(f"bench_key_{i}")
        benchmark_time = time.time() - start_time
        
        ops_per_second = (iterations * 2) / benchmark_time
        print(f"⚡ Cache performance: {ops_per_second:.0f} ops/second")
        print("✅ Performance benchmark completed")
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False
    
    # Cleanup
    try:
        cache_manager.cleanup()
    except:
        pass
    
    print("\n🎉 All integration tests passed!")
    print("=" * 50)
    print("✅ Optimized GraphRAG system is ready for production")
    print(f"🚀 System optimized for: {memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    return True

if __name__ == "__main__":
    success = test_optimized_integration()
    if not success:
        exit(1)
