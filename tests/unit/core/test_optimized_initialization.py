"""
Test suite for OptimizedInitialization
Testing parallel system startup, memory management, and hardware optimizations
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import psutil
from concurrent.futures import ThreadPoolExecutor

from src.graphrag.core.optimized_initialization import (
    HardwareResourceMonitor,
    OptimizedComponentLoader,
    initialize_graphrag_system_parallel,
    initialize_graphrag_system_memory_efficient,
    get_recommended_initialization_strategy
)


class TestHardwareResourceMonitor:
    """Test hardware resource monitoring"""
    
    def test_initialization(self):
        """Test monitor initialization"""
        monitor = HardwareResourceMonitor()
        
        assert monitor.cpu_count > 0
        assert monitor.memory_total > 0
        assert monitor.memory_available > 0
    
    def test_optimal_workers_calculation(self):
        """Test optimal worker calculation"""
        monitor = HardwareResourceMonitor()
        workers = monitor.get_optimal_workers()
        
        # Should be between 2 and 8
        assert 2 <= workers <= 8
        
        # Should be reasonable fraction of CPU count
        assert workers <= monitor.cpu_count
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection"""
        monitor = HardwareResourceMonitor()
        pressure = monitor.get_memory_pressure()
        
        # Should be between 0 and 1
        assert 0 <= pressure <= 1
    
    def test_memory_availability_check(self):
        """Test memory availability check"""
        monitor = HardwareResourceMonitor()
        
        # Small amount should be available
        assert monitor.is_memory_available(0.1) == True
        
        # Huge amount should not be available
        assert monitor.is_memory_available(1000) == False
    
    def test_aggressive_caching_decision(self):
        """Test aggressive caching decision logic"""
        monitor = HardwareResourceMonitor()
        should_cache = monitor.should_use_aggressive_caching()
        
        # Should return boolean
        assert isinstance(should_cache, bool)


class TestOptimizedComponentLoader:
    """Test optimized component loading"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies"""
        with patch('src.graphrag.core.optimized_initialization.get_weaviate_connection') as mock_weaviate, \
             patch('src.graphrag.core.optimized_initialization.load_graph_data') as mock_graph, \
             patch('src.graphrag.core.optimized_initialization.get_ollama_client') as mock_ollama:
            
            # Setup mocks
            mock_weaviate_client = Mock()
            mock_weaviate_client.client = Mock()
            mock_weaviate.return_value = mock_weaviate_client
            
            mock_graph.return_value = Mock()
            mock_ollama.return_value = Mock()
            
            yield {
                'weaviate': mock_weaviate,
                'graph': mock_graph,
                'ollama': mock_ollama
            }
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        loader = OptimizedComponentLoader()
        
        assert loader.monitor is not None
        assert loader.load_times == {}
        assert hasattr(loader, 'logger')
    
    def test_weaviate_loading(self, mock_dependencies):
        """Test optimized Weaviate loading"""
        loader = OptimizedComponentLoader()
        
        result = loader.load_weaviate_optimized()
        
        # Should return optimized store
        assert result is not None
        assert 'weaviate' in loader.load_times
        assert loader.load_times['weaviate'] > 0
    
    def test_graph_loading_with_memory_management(self, mock_dependencies):
        """Test graph loading with memory management"""
        loader = OptimizedComponentLoader()
        
        result = loader.load_graph_with_memory_management()
        
        # Should return graph
        assert result is not None
        assert 'graph' in loader.load_times
        assert loader.load_times['graph'] > 0
    
    def test_ollama_loading(self, mock_dependencies):
        """Test optimized Ollama loading"""
        loader = OptimizedComponentLoader()
        
        result = loader.load_ollama_optimized()
        
        # Should return LLM client
        assert result is not None
        assert 'ollama' in loader.load_times
        assert loader.load_times['ollama'] > 0
    
    def test_load_time_tracking(self, mock_dependencies):
        """Test load time tracking"""
        loader = OptimizedComponentLoader()
        
        # Load components
        loader.load_weaviate_optimized()
        loader.load_graph_with_memory_management()
        loader.load_ollama_optimized()
        
        # Check all times are tracked
        assert len(loader.load_times) == 3
        assert all(time > 0 for time in loader.load_times.values())


class TestParallelInitialization:
    """Test parallel initialization process"""
    
    @pytest.fixture
    def mock_all_dependencies(self):
        """Mock all external dependencies for initialization"""
        with patch('src.graphrag.core.optimized_initialization.get_weaviate_connection') as mock_weaviate, \
             patch('src.graphrag.core.optimized_initialization.load_graph_data') as mock_graph, \
             patch('src.graphrag.core.optimized_initialization.get_ollama_client') as mock_ollama, \
             patch('src.graphrag.core.optimized_initialization.verify_weaviate_collections') as mock_verify, \
             patch('src.graphrag.core.optimized_initialization.GraphRAGQueryEngine') as mock_engine, \
             patch('src.graphrag.core.optimized_initialization.OptimizedWeaviateGraphStore') as mock_opt_store:
            
            # Setup mocks
            mock_weaviate_client = Mock()
            mock_weaviate_client.client = Mock()
            mock_weaviate.return_value = mock_weaviate_client
            
            mock_graph.return_value = Mock()
            mock_ollama.return_value = Mock()
            mock_verify.return_value = True
            mock_engine.return_value = Mock()
            mock_opt_store.return_value = Mock()
            
            yield {
                'weaviate': mock_weaviate,
                'graph': mock_graph,
                'ollama': mock_ollama,
                'verify': mock_verify,
                'engine': mock_engine,
                'opt_store': mock_opt_store
            }
    
    def test_parallel_initialization_success(self, mock_all_dependencies):
        """Test successful parallel initialization"""
        result = initialize_graphrag_system_parallel()
        
        # Should return tuple of 4 components
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        graph, vector_store, llm_client, engine = result
        assert graph is not None
        assert vector_store is not None
        assert llm_client is not None
        assert engine is not None
    
    def test_parallel_initialization_timing(self, mock_all_dependencies):
        """Test parallel initialization is reasonably fast"""
        start_time = time.time()
        result = initialize_graphrag_system_parallel()
        total_time = time.time() - start_time
        
        # Should complete within reasonable time (mocked, so should be very fast)
        assert total_time < 5.0  # 5 seconds max for mocked version
        assert result is not None
    
    def test_memory_efficient_initialization(self, mock_all_dependencies):
        """Test memory-efficient initialization"""
        result = initialize_graphrag_system_memory_efficient()
        
        # Should return tuple of 4 components
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        graph, vector_store, llm_client, engine = result
        assert graph is not None
        assert vector_store is not None
        assert llm_client is not None
        assert engine is not None
    
    def test_initialization_strategy_selection(self):
        """Test initialization strategy selection"""
        strategy = get_recommended_initialization_strategy()
        
        # Should return valid strategy
        assert strategy in ['parallel', 'memory_efficient']
    
    @patch('src.graphrag.core.optimized_initialization.psutil.virtual_memory')
    @patch('src.graphrag.core.optimized_initialization.psutil.cpu_count')
    def test_strategy_high_resources(self, mock_cpu, mock_memory):
        """Test strategy selection with high resources"""
        # Mock high-resource system
        mock_cpu.return_value = 16
        mock_memory.return_value.available = 8 * 1024**3  # 8GB available
        
        strategy = get_recommended_initialization_strategy()
        assert strategy == 'parallel'
    
    @patch('src.graphrag.core.optimized_initialization.psutil.virtual_memory')
    @patch('src.graphrag.core.optimized_initialization.psutil.cpu_count')
    def test_strategy_low_resources(self, mock_cpu, mock_memory):
        """Test strategy selection with low resources"""
        # Mock low-resource system
        mock_cpu.return_value = 2
        mock_memory.return_value.available = 2 * 1024**3  # 2GB available
        
        strategy = get_recommended_initialization_strategy()
        assert strategy == 'memory_efficient'


class TestInitializationErrorHandling:
    """Test error handling in initialization"""
    
    @patch('src.graphrag.core.optimized_initialization.get_weaviate_connection')
    def test_weaviate_failure_handling(self, mock_weaviate):
        """Test handling of Weaviate connection failure"""
        mock_weaviate.side_effect = Exception("Weaviate connection failed")
        
        loader = OptimizedComponentLoader()
        
        with pytest.raises(Exception):
            loader.load_weaviate_optimized()
    
    @patch('src.graphrag.core.optimized_initialization.load_graph_data')
    def test_graph_failure_handling(self, mock_graph):
        """Test handling of graph loading failure"""
        mock_graph.side_effect = Exception("Graph loading failed")
        
        loader = OptimizedComponentLoader()
        
        with pytest.raises(Exception):
            loader.load_graph_with_memory_management()
    
    @patch('src.graphrag.core.optimized_initialization.get_ollama_client')
    def test_ollama_failure_handling(self, mock_ollama):
        """Test handling of Ollama connection failure"""
        mock_ollama.side_effect = Exception("Ollama connection failed")
        
        loader = OptimizedComponentLoader()
        
        with pytest.raises(Exception):
            loader.load_ollama_optimized()


class TestInitializationIntegration:
    """Integration tests for initialization system"""
    
    def test_hardware_detection_accuracy(self):
        """Test hardware detection accuracy"""
        monitor = HardwareResourceMonitor()
        
        # Verify detected values are reasonable
        assert monitor.cpu_count >= 1
        assert monitor.memory_total > 1024**3  # At least 1GB
        assert monitor.memory_available > 0
        assert monitor.memory_available <= monitor.memory_total
    
    def test_component_loader_resource_tracking(self):
        """Test component loader tracks resources properly"""
        loader = OptimizedComponentLoader()
        
        # Verify monitor is properly initialized
        assert loader.monitor.cpu_count > 0
        assert loader.monitor.memory_total > 0
    
    def test_optimization_factory_function(self):
        """Test the factory function for optimized initialization"""
        from src.graphrag.core.optimized_initialization import initialize_graphrag_optimized
        
        # Should be callable
        assert callable(initialize_graphrag_optimized)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
