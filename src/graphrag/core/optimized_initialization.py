"""
Optimized GraphRAG Initialization for Ryzen 4800H + GTX 1650Ti + 16GB RAM
Key optimizations:
- Parallel component initialization
- Memory-aware loading strategies
- Hardware-specific resource allocation
- Intelligent caching and preloading
- Resource monitoring and adaptive loading
"""

import logging
import time
import psutil
import concurrent.futures
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import threading
import gc

from .initialization import (
    load_graph_data,
    get_weaviate_connection,
    get_ollama_client,
    verify_weaviate_collections
)
from .optimized_vector_store import OptimizedWeaviateGraphStore
from .query_engine import GraphRAGQueryEngine

logger = logging.getLogger(__name__)

class HardwareResourceMonitor:
    """Monitor system resources for optimal loading strategies"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.memory_available = psutil.virtual_memory().available
        
    def get_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on CPU cores"""
        # Use 75% of available cores, minimum 2, maximum 8
        return max(2, min(8, int(self.cpu_count * 0.75)))
    
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)"""
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    
    def is_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available"""
        memory = psutil.virtual_memory()
        required_bytes = required_gb * 1024 * 1024 * 1024
        return memory.available > required_bytes
    
    def should_use_aggressive_caching(self) -> bool:
        """Determine if aggressive caching should be used"""
        memory = psutil.virtual_memory()
        return memory.available > (4 * 1024 * 1024 * 1024)  # >4GB available

class OptimizedComponentLoader:
    """Load system components with hardware optimization"""
    
    def __init__(self):
        self.monitor = HardwareResourceMonitor()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_times = {}
    
    def load_weaviate_optimized(self) -> OptimizedWeaviateGraphStore:
        """Load Weaviate with hardware-specific optimizations"""
        start_time = time.time()
        
        try:
            # Get base connection
            base_client = get_weaviate_connection()
            
            # Create optimized wrapper
            max_workers = self.monitor.get_optimal_workers()
            cache_size = 2000 if self.monitor.should_use_aggressive_caching() else 1000
            
            optimized_store = OptimizedWeaviateGraphStore(
                client=base_client.client,
                max_workers=max_workers,
                cache_size=cache_size
            )
            
            load_time = time.time() - start_time
            self.load_times['weaviate'] = load_time
            self.logger.info(f"✅ Optimized Weaviate loaded in {load_time:.2f}s")
            
            return optimized_store
            
        except Exception as e:
            self.logger.error(f"Failed to load optimized Weaviate: {e}")
            raise
    
    def load_graph_with_memory_management(self):
        """Load graph with memory pressure monitoring"""
        start_time = time.time()
        
        try:
            # Check memory before loading
            if not self.monitor.is_memory_available(2.0):  # Need 2GB
                self.logger.warning("Low memory detected, forcing garbage collection")
                gc.collect()
            
            # Load graph with monitoring
            graph = load_graph_data()
            
            load_time = time.time() - start_time
            self.load_times['graph'] = load_time
            
            # Log memory usage
            memory_used = psutil.virtual_memory().percent
            self.logger.info(f"✅ Graph loaded in {load_time:.2f}s (Memory: {memory_used:.1f}%)")
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")
            raise
    
    def load_ollama_optimized(self):
        """Load Ollama client with hardware optimizations"""
        start_time = time.time()
        
        try:
            # Load with hardware-specific settings
            llm_client = get_ollama_client()
            
            # Apply optimizations if client supports it
            if hasattr(llm_client, 'num_threads'):
                llm_client.num_threads = self.monitor.get_optimal_workers()
            
            load_time = time.time() - start_time
            self.load_times['ollama'] = load_time
            self.logger.info(f"✅ Optimized Ollama loaded in {load_time:.2f}s")
            
            return llm_client
            
        except Exception as e:
            self.logger.error(f"Failed to load Ollama: {e}")
            raise

def initialize_graphrag_system_parallel() -> Tuple:
    """
    Parallel GraphRAG system initialization optimized for Ryzen 4800H
    
    Returns:
        Tuple[graph, vector_store, llm_client, engine]
    """
    start_time = time.time()
    loader = OptimizedComponentLoader()
    
    try:
        logger.info("🚀 Starting parallel GraphRAG initialization...")
        
        # Phase 1: Load independent components in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            logger.info("📦 Phase 1: Loading core components in parallel...")
            
            # Submit parallel loading tasks
            vector_future = executor.submit(loader.load_weaviate_optimized)
            graph_future = executor.submit(loader.load_graph_with_memory_management)
            llm_future = executor.submit(loader.load_ollama_optimized)
            
            # Wait for components with timeouts
            try:
                vector_store = vector_future.result(timeout=60)  # 1 minute for Weaviate
                graph = graph_future.result(timeout=120)        # 2 minutes for graph
                llm_client = llm_future.result(timeout=30)      # 30 seconds for Ollama
                
                logger.info("✅ All core components loaded successfully")
                
            except concurrent.futures.TimeoutError as e:
                logger.error(f"Component loading timeout: {e}")
                raise
            except Exception as e:
                logger.error(f"Component loading failed: {e}")
                raise
        
        # Phase 2: Verify system readiness
        logger.info("🔍 Phase 2: Verifying system readiness...")
        
        verification_start = time.time()
        if not verify_weaviate_collections(vector_store):
            logger.warning("Weaviate collections verification failed - some features may be limited")
        
        verification_time = time.time() - verification_start
        logger.info(f"✅ System verification completed in {verification_time:.2f}s")
        
        # Phase 3: Initialize query engine with optimizations
        logger.info("🧠 Phase 3: Initializing optimized query engine...")
        
        engine_start = time.time()
        
        # Choose engine type based on available resources
        monitor = HardwareResourceMonitor()
        if monitor.should_use_aggressive_caching():
            # Create Neo4j-optimized engine if memory allows
            try:
                engine = GraphRAGQueryEngine.create_legacy_engine(
                    graph=graph,
                    vector_store=vector_store,
                    llm_client=llm_client
                )
                logger.info("✅ Legacy engine initialized with optimizations")
            except Exception as e:
                logger.warning(f"Legacy engine initialization failed: {e}")
                # Fallback to basic engine
                engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
        else:
            # Use standard engine for lower memory systems
            engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
            logger.info("✅ Standard engine initialized")
        
        engine_time = time.time() - engine_start
        
        # Phase 4: System warmup and testing
        logger.info("🔥 Phase 4: System warmup...")
        
        warmup_start = time.time()
        try:
            # Test basic functionality
            test_result = engine._vector_entity_search("test", 1)
            
            # Preload common embeddings if memory allows
            if monitor.should_use_aggressive_caching():
                _preload_common_embeddings(vector_store)
            
            warmup_time = time.time() - warmup_start
            logger.info(f"✅ System warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"System warmup failed: {e} - continuing anyway")
        
        # Final statistics
        total_time = time.time() - start_time
        _log_initialization_summary(loader.load_times, total_time, monitor)
        
        logger.info(f"🎉 GraphRAG system initialization complete in {total_time:.2f}s")
        return graph, vector_store, llm_client, engine
        
    except Exception as e:
        logger.error(f"GraphRAG system initialization failed: {e}")
        # Cleanup on failure
        _cleanup_failed_initialization(locals())
        raise

def initialize_graphrag_system_memory_efficient() -> Tuple:
    """
    Memory-efficient initialization for constrained environments
    
    Returns:
        Tuple[graph, vector_store, llm_client, engine]
    """
    start_time = time.time()
    loader = OptimizedComponentLoader()
    
    try:
        logger.info("🔋 Starting memory-efficient GraphRAG initialization...")
        
        # Sequential loading with memory management
        logger.info("📦 Loading components sequentially for memory efficiency...")
        
        # Load Weaviate first (smallest memory footprint)
        vector_store = loader.load_weaviate_optimized()
        gc.collect()  # Force cleanup
        
        # Load Ollama next (medium memory footprint)
        llm_client = loader.load_ollama_optimized()
        gc.collect()
        
        # Load graph last (largest memory footprint)
        graph = loader.load_graph_with_memory_management()
        gc.collect()
        
        # Verify system
        if not verify_weaviate_collections(vector_store):
            logger.warning("Weaviate collections verification failed")
        
        # Initialize minimal engine
        engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Memory-efficient initialization complete in {total_time:.2f}s")
        
        return graph, vector_store, llm_client, engine
        
    except Exception as e:
        logger.error(f"Memory-efficient initialization failed: {e}")
        raise

def _preload_common_embeddings(vector_store):
    """Preload common search embeddings to cache"""
    try:
        common_terms = [
            "drug", "disease", "protein", "treatment", "therapy",
            "mechanism", "pathway", "interaction", "side effect",
            "clinical trial", "efficacy", "safety", "dosage"
        ]
        
        logger.info(f"🔥 Preloading {len(common_terms)} common embeddings...")
        
        # Preload in small batches to avoid memory spikes
        batch_size = 5
        for i in range(0, len(common_terms), batch_size):
            batch = common_terms[i:i + batch_size]
            for term in batch:
                if hasattr(vector_store, '_get_cached_embedding'):
                    vector_store._get_cached_embedding(term)
        
        logger.info("✅ Common embeddings preloaded")
        
    except Exception as e:
        logger.warning(f"Embedding preload failed: {e}")

def _log_initialization_summary(load_times: Dict[str, float], total_time: float, monitor: HardwareResourceMonitor):
    """Log detailed initialization summary"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        logger.info("📊 INITIALIZATION SUMMARY:")
        logger.info(f"   🕐 Total Time: {total_time:.2f}s")
        logger.info(f"   💾 Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
        logger.info(f"   🖥️  CPU Usage: {cpu_percent:.1f}%")
        logger.info(f"   🔧 Workers: {monitor.get_optimal_workers()}")
        
        logger.info("   ⏱️  Component Load Times:")
        for component, load_time in load_times.items():
            logger.info(f"      {component}: {load_time:.2f}s")
            
    except Exception as e:
        logger.warning(f"Summary logging failed: {e}")

def _cleanup_failed_initialization(local_vars: Dict[str, Any]):
    """Cleanup resources on failed initialization"""
    try:
        logger.info("🧹 Cleaning up failed initialization...")
        
        # Close vector store if it exists
        if 'vector_store' in local_vars and local_vars['vector_store']:
            try:
                local_vars['vector_store'].close()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ Cleanup completed")
        
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

def get_recommended_initialization_strategy() -> str:
    """
    Get recommended initialization strategy based on system resources
    
    Returns:
        'parallel' or 'memory_efficient'
    """
    monitor = HardwareResourceMonitor()
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count()
    
    if available_gb >= 4 and cpu_cores >= 4:
        return 'parallel'
    else:
        return 'memory_efficient'

# Factory function for easy access
def initialize_graphrag_optimized() -> Tuple:
    """
    Initialize GraphRAG system with optimal strategy for current hardware
    
    Returns:
        Tuple[graph, vector_store, llm_client, engine]
    """
    strategy = get_recommended_initialization_strategy()
    
    if strategy == 'parallel':
        return initialize_graphrag_system_parallel()
    else:
        return initialize_graphrag_system_memory_efficient()

# Backward compatibility
initialize_graphrag_system = initialize_graphrag_optimized
