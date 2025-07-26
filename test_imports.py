"""
Simple test to verify our optimized components can be imported
"""

def test_optimized_imports():
    """Test that optimized components can be imported"""
    try:
        from src.graphrag.core.optimized_vector_store import OptimizedWeaviateGraphStore, MemoryAwareCache
        from src.graphrag.core.optimized_initialization import HardwareResourceMonitor, OptimizedComponentLoader
        from src.graphrag.core.optimized_context_builder import OptimizedContextBuilder, ContextFragment
        from src.graphrag.core.enhanced_connection_resilience import CircuitBreaker, AdaptiveRetryStrategy
        
        # Test basic instantiation
        cache = MemoryAwareCache(max_size=10, max_memory_mb=1)
        monitor = HardwareResourceMonitor()
        fragment = ContextFragment("test", "test")
        
        print("✅ All optimized components imported successfully!")
        assert True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False

if __name__ == "__main__":
    test_optimized_imports()
