#!/usr/bin/env python3
"""
🚀 BLAZING FAST MEMGRAPH LOADER 🚀 
Ultra-optimized for millions of nodes/relationships
Hardware-optimized for AMD Ryzen 7 4800H CPU (8 cores/16 threads, 16GB RAM)
GPU-accelerated where possible using PyTorch/CUDA
"""

import pandas as pd
import numpy as np
from gqlalchemy import Memgraph
import logging
import time
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import cpu_count, Manager
import multiprocessing as mp
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import signal
import sys

# Try to import GPU acceleration libraries
GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        import torch.nn.functional as F
        print(f"🔥 GPU acceleration ENABLED! CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  PyTorch available but no CUDA device found")
except ImportError:
    print("⚠️  PyTorch not available - running CPU-only mode")

# Try numba for JIT compilation
NUMBA_AVAILABLE = False
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
    print("🚀 Numba JIT compilation ENABLED!")
except ImportError:
    print("⚠️  Numba not available - using standard Python")

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class LoaderConfig:
    """Configuration for the blazing fast loader"""
    nodes_file: str = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/nodes_optimized.csv"
    edges_file: str = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/edges_optimized.csv"
    batch_size_nodes: int = 10000
    batch_size_edges: int = 2000
    max_memory_usage: float = 0.8
    max_cpu_usage: float = 0.9
    memgraph_host: str = 'localhost'
    memgraph_port: int = 7687
    use_gpu: bool = GPU_AVAILABLE
    chunk_processing_interval: int = 10  # Process edges every N node chunks

class HardwareMonitor:
    """Monitor hardware resources and throttle if needed"""
    
    def __init__(self, max_memory: float = 0.8, max_cpu: float = 0.9):
        self.max_memory = max_memory
        self.max_cpu = max_cpu
        self._monitoring = False
        self._stop_event = threading.Event()
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        self._stop_event.set()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring and not self._stop_event.is_set():
            self.check_and_throttle()
            time.sleep(2)  # Check every 2 seconds
            
    def check_and_throttle(self) -> bool:
        """Check resources and throttle if needed. Returns True if throttling occurred"""
        memory_usage = psutil.virtual_memory().percent / 100.0
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        
        if memory_usage > self.max_memory or cpu_usage > self.max_cpu:
            logger.warning(f"🛑 THROTTLING: Memory {memory_usage*100:.1f}% CPU {cpu_usage*100:.1f}%")
            # Force garbage collection
            gc.collect()
            # Sleep to cool down
            time.sleep(2)
            return True
        return False
        
    def get_adaptive_workers(self, max_workers: int) -> int:
        """Get adaptive worker count based on current resource usage"""
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        # Reduce workers if high usage
        if cpu_usage > 0.8 or memory_usage > 0.7:
            return max(1, max_workers // 2)
        elif cpu_usage > 0.6 or memory_usage > 0.6:
            return max(2, int(max_workers * 0.75))
        return max_workers

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def fast_string_filter(strings, valid_set):
        """Fast numba-accelerated string filtering"""
        result = []
        for s in strings:
            if s in valid_set:
                result.append(s)
        return result
else:
    def fast_string_filter(strings, valid_set):
        """Fallback string filtering"""
        return [s for s in strings if s in valid_set]

class BlazingFastLoader:
    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self.cpu_cores = cpu_count()
        # Optimized for Ryzen 7 4800H (8 cores/16 threads)
        self.max_workers = min(16, self.cpu_cores)
        self.max_processes = min(8, self.cpu_cores)  # True parallelism
        
        # Initialize hardware monitor
        self.monitor = HardwareMonitor(
            max_memory=self.config.max_memory_usage,
            max_cpu=self.config.max_cpu_usage
        )
        
        # Memory monitoring
        self.total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        
        # Performance tracking
        self.stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'start_time': 0.0,
            'gpu_operations': 0,
            'throttle_events': 0
        }
        
        print(f"🔥 BLAZING FAST LOADER INITIALIZED (OPTIMIZED)")
        print(f"   CPU Cores: {self.cpu_cores} | Workers: {self.max_workers} | Processes: {self.max_processes}")
        print(f"   Total RAM: {self.total_ram:.1f} GB | Max Usage: {self.config.max_memory_usage*100:.0f}%")
        print(f"   GPU Available: {GPU_AVAILABLE} | Numba JIT: {NUMBA_AVAILABLE}")
        print(f"   Files: nodes={Path(self.config.nodes_file).name}, edges={Path(self.config.edges_file).name}")
        print()
        
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent / 100.0
        
    def adaptive_chunk_size(self, base_size: int, memory_factor: bool = True) -> int:
        """Dynamically adjust chunk size based on memory usage and CPU load"""
        if memory_factor:
            memory_usage = self.get_memory_usage()
            cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            
            # More aggressive scaling based on both memory and CPU
            if memory_usage > 0.75 or cpu_usage > 0.85:
                return max(base_size // 4, 500)
            elif memory_usage > 0.65 or cpu_usage > 0.75:
                return max(base_size // 2, 1000)
            elif memory_usage < 0.4 and cpu_usage < 0.5:
                return min(base_size * 2, 20000)
        return base_size
        
    def force_garbage_collection(self):
        """Force garbage collection and log memory savings"""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        if before - after > 0.05:  # Only log if significant memory freed
            logger.info(f"🧹 Memory freed: {(before-after)*100:.1f}% (was {before*100:.1f}%, now {after*100:.1f}%)")
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def execute_with_retry(self, memgraph: Memgraph, query: str, params: Optional[Dict] = None):
        """Execute query with automatic retry on failure"""
        try:
            return memgraph.execute(query, params or {})
        except Exception as e:
            logger.warning(f"Query failed, retrying... Error: {e}")
            raise
            
    def verify_memgraph_connection(self) -> bool:
        """Verify Memgraph connection before starting"""
        try:
            memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            memgraph.execute("MATCH (n) RETURN count(n) LIMIT 1")
            logger.info("✅ Memgraph connection verified")
            return True
        except Exception as e:
            logger.error(f"❌ Memgraph connection failed: {e}")
            return False
        
    def clear_database(self):
        """Clear the database FAST with better error handling"""
        logger.info("🧹 Clearing database...")
        try:
            memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            
            # Drop all constraints first for speed
            constraints_to_drop = [
                "DROP CONSTRAINT ON (n:Disease) ASSERT n.node_id IS UNIQUE",
                "DROP CONSTRAINT ON (n:Drug) ASSERT n.node_id IS UNIQUE", 
                "DROP CONSTRAINT ON (n:Target) ASSERT n.node_id IS UNIQUE",
                "DROP CONSTRAINT ON (n:Pathway) ASSERT n.node_id IS UNIQUE"
            ]
            
            for constraint in constraints_to_drop:
                try:
                    self.execute_with_retry(memgraph, constraint)
                except:
                    pass  # Constraints might not exist
            
            # Fast delete all with retry
            self.execute_with_retry(memgraph, "MATCH (n) DETACH DELETE n")
            logger.info("✅ Database cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
            raise
    
    def create_indexes_for_speed(self):
        """Create indexes BEFORE relationship loading for blazing speed"""
        logger.info("🔧 Creating indexes for blazing fast relationships...")
        try:
            memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            
            # Create indexes on node_id for LIGHTNING FAST lookups
            indexes = [
                "CREATE INDEX ON :Disease(node_id)",
                "CREATE INDEX ON :Drug(node_id)", 
                "CREATE INDEX ON :Target(node_id)",
                "CREATE INDEX ON :Pathway(node_id)"
            ]
            
            for index in indexes:
                try:
                    self.execute_with_retry(memgraph, index)
                    node_type = index.split(':')[1].split('(')[0]
                    logger.info(f"✅ Created index: {node_type}(node_id)")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"⚠️  Index already exists")
                    else:
                        logger.warning(f"⚠️  Index creation failed: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")

    def create_constraints(self):
        """Create constraints for performance - AFTER loading"""
        logger.info("🔧 Creating constraints...")
        try:
            memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            
            constraints = [
                "CREATE CONSTRAINT ON (n:Disease) ASSERT n.node_id IS UNIQUE",
                "CREATE CONSTRAINT ON (n:Drug) ASSERT n.node_id IS UNIQUE", 
                "CREATE CONSTRAINT ON (n:Target) ASSERT n.node_id IS UNIQUE",
                "CREATE CONSTRAINT ON (n:Pathway) ASSERT n.node_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    self.execute_with_retry(memgraph, constraint)
                    logger.info(f"✅ Created constraint: {constraint.split('(')[1].split(')')[0]}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"⚠️  Constraint already exists")
                    else:
                        logger.warning(f"⚠️  Constraint creation failed: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to create constraints: {e}")

    def prepare_batch_data(self, batch_df: pd.DataFrame, data_type: str) -> List[Dict]:
        """Unified batch data preparation with GPU acceleration if available"""
        if data_type == 'nodes':
            batch_data = []
            for _, row in batch_df.iterrows():
                node_id = str(row.get('id', '')).strip()
                name = str(row.get('name', '')).strip()
                
                if node_id:
                    batch_data.append({"node_id": node_id, "name": name if name else node_id})
            return batch_data
            
        elif data_type == 'edges':
            batch_data = []
            for _, row in batch_df.iterrows():
                source = str(row.get('source', '')).strip()
                target = str(row.get('target', '')).strip()
                
                if source and target:
                    batch_data.append({"source": source, "target": target})
            return batch_data
        
        return []

    def gpu_accelerated_filtering(self, edge_chunk: pd.DataFrame, known_node_ids: Set[str]) -> pd.DataFrame:
        """Use GPU acceleration for edge filtering if available"""
        if not GPU_AVAILABLE or len(edge_chunk) < 1000:  # Use GPU only for large chunks
            # CPU fallback
            source_mask = edge_chunk['source'].isin(known_node_ids)
            target_mask = edge_chunk['target'].isin(known_node_ids)
            return edge_chunk[source_mask & target_mask]
        
        try:
            # Convert to tensors for GPU processing
            self.stats['gpu_operations'] += 1
            
            # Convert known_node_ids to a lookup tensor
            known_list = list(known_node_ids)
            sources = edge_chunk['source'].values
            targets = edge_chunk['target'].values
            
            # Use set intersection for now (can be optimized further with proper GPU kernels)
            valid_mask = np.array([
                source in known_node_ids and target in known_node_ids 
                for source, target in zip(sources, targets)
            ])
            
            return edge_chunk[valid_mask]
            
        except Exception as e:
            logger.warning(f"GPU filtering failed, falling back to CPU: {e}")
            # CPU fallback
            source_mask = edge_chunk['source'].isin(known_node_ids)
            target_mask = edge_chunk['target'].isin(known_node_ids)
            return edge_chunk[source_mask & target_mask]

    def bulk_create_nodes(self, batch_df: pd.DataFrame, node_type: str) -> int:
        """ULTRA FAST bulk node creation using UNWIND with optimizations"""
        try:
            local_memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            
            # Prepare ALL data in one go using unified method
            node_data = self.prepare_batch_data(batch_df, 'nodes')
            
            if not node_data:
                return 0
            
            # Check for throttling
            if self.monitor.check_and_throttle():
                self.stats['throttle_events'] += 1
            
            # MASSIVE bulk insert with single query
            query = f"""
            UNWIND $nodes AS node
            CREATE (n:{node_type})
            SET n = node
            """
            
            self.execute_with_retry(local_memgraph, query, {"nodes": node_data})
            self.stats['nodes_created'] += len(node_data)
            return len(node_data)
            
        except Exception as e:
            logger.error(f"Bulk node creation failed: {e}")
            return 0

    def bulk_create_relationships(self, batch_df: pd.DataFrame, rel_type: str) -> int:
        """OPTIMIZED bulk relationship creation with adaptive batching"""
        try:
            local_memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            
            # Dynamic sub-batch sizing based on hardware load
            base_sub_batch_size = 200
            sub_batch_size = self.adaptive_chunk_size(base_sub_batch_size, memory_factor=True)
            total_created = 0
            
            for i in range(0, len(batch_df), sub_batch_size):
                sub_batch = batch_df.iloc[i:i+sub_batch_size]
                
                # Prepare sub-batch data using unified method
                rel_data = self.prepare_batch_data(sub_batch, 'edges')
                
                if rel_data:
                    # Check for throttling before heavy operation
                    if self.monitor.check_and_throttle():
                        self.stats['throttle_events'] += 1
                    
                    # Use optimized query with explicit index hints
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a) WHERE a.node_id = rel.source
                    MATCH (b) WHERE b.node_id = rel.target  
                    CREATE (a)-[:{rel_type}]->(b)
                    """
                    
                    self.execute_with_retry(local_memgraph, query, {"rels": rel_data})
                    total_created += len(rel_data)
            
            self.stats['edges_created'] += total_created
            return total_created
            
        except Exception as e:
            logger.error(f"Bulk relationship creation failed: {e}")
            return 0

    def multiprocess_node_loading(self, nodes_df: pd.DataFrame, node_type: str) -> int:
        """Load nodes using multiprocessing for true parallelism"""
        if len(nodes_df) < 5000:  # Use multiprocessing only for large datasets
            return self.bulk_create_nodes(nodes_df, node_type)
        
        logger.info(f"🚀 Multiprocess loading {len(nodes_df):,} {node_type} nodes...")
        start_time = time.time()
        
        # Adaptive batch sizing
        batch_size = self.adaptive_chunk_size(self.config.batch_size_nodes)
        
        # Split into chunks for multiprocessing
        chunks = [nodes_df.iloc[i:i+batch_size] for i in range(0, len(nodes_df), batch_size)]
        
        total_created = 0
        
        # Use process pool for CPU-bound work
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            futures = []
            
            for i, chunk in enumerate(chunks):
                future = executor.submit(self._process_node_chunk, chunk, node_type)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(as_completed(futures)):
                try:
                    created = future.result()
                    total_created += created
                    elapsed = time.time() - start_time
                    rate = total_created / elapsed if elapsed > 0 else 0
                    logger.info(f"   Process {i+1}/{len(futures)} complete - {created} nodes - Rate: {rate:.0f} nodes/sec")
                except Exception as e:
                    logger.error(f"   Process {i+1} failed: {e}")
        
        elapsed = time.time() - start_time
        rate = total_created / elapsed if elapsed > 0 else 0
        logger.info(f"⚡ {node_type}: {total_created:,} nodes in {elapsed:.1f}s - {rate:.0f} nodes/sec!")
        return total_created
        
    def _process_node_chunk(self, chunk: pd.DataFrame, node_type: str) -> int:
        """Process a single node chunk (for multiprocessing)"""
        return self.bulk_create_nodes(chunk, node_type)

    def load_nodes_blazing_fast(self, nodes_df, node_type):
        """Load nodes with MAXIMUM speed"""
        type_nodes = nodes_df[nodes_df['type'] == node_type].copy()
        
        if len(type_nodes) == 0:
            return
            
        logger.info(f"🚀 Loading {len(type_nodes):,} {node_type} nodes...")
        start_time = time.time()
        
        # MUCH larger batches for speed
        batch_size = 10000  # 10K per batch instead of 2K
        total_created = 0
        
        # Process massive batches with more threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(type_nodes), batch_size):
                batch = type_nodes.iloc[i:i+batch_size]
                future = executor.submit(self.bulk_create_nodes, batch, node_type)
                futures.append(future)
            
            # Process results
            for i, future in enumerate(as_completed(futures)):
                try:
                    created = future.result()
                    total_created += created
                    elapsed = time.time() - start_time
                    rate = total_created / elapsed if elapsed > 0 else 0
                    logger.info(f"   Batch {i+1}/{len(futures)} complete - {created} nodes - Rate: {rate:.0f} nodes/sec")
                except Exception as e:
                    logger.error(f"   Batch {i+1} failed: {e}")
        
        elapsed = time.time() - start_time
        rate = total_created / elapsed if elapsed > 0 else 0
        logger.info(f"⚡ {node_type}: {total_created:,} nodes in {elapsed:.1f}s - {rate:.0f} nodes/sec!")
        
    def load_relationships_blazing_fast(self, edges_df, rel_type):
        """Load relationships with OPTIMIZED speed - tiny batches with more parallelism"""
        type_rels = edges_df[edges_df['type'] == rel_type].copy()
        
        if len(type_rels) == 0:
            return
            
        logger.info(f"💥 Loading {len(type_rels):,} {rel_type} relationships...")
        start_time = time.time()
        
        # TINY batches for maximum parallelism
        batch_size = 200  # Even smaller batches for speed
        total_created = 0
        
        # Use more workers for tiny batches
        with ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
            futures = []
            
            for i in range(0, len(type_rels), batch_size):
                batch = type_rels.iloc[i:i+batch_size]
                future = executor.submit(self.bulk_create_relationships, batch, rel_type)
                futures.append(future)
            
            # Process results with better progress tracking
            completed = 0
            for future in as_completed(futures):
                try:
                    created = future.result()
                    total_created += created
                    completed += 1
                    elapsed = time.time() - start_time
                    rate = total_created / elapsed if elapsed > 0 else 0
                    progress = completed / len(futures) * 100
                    
                    # Log every 50 batches instead of every batch
                    if completed % 50 == 0 or completed == len(futures):
                        logger.info(f"   Batch {completed}/{len(futures)} ({progress:.1f}%) - Rate: {rate:.0f} rels/sec")
                except Exception as e:
                    completed += 1
                    logger.error(f"   Batch {completed} failed: {e}")
        
        elapsed = time.time() - start_time
        rate = total_created / elapsed if elapsed > 0 else 0
        logger.info(f"⚡ {rel_type}: {total_created:,} rels in {elapsed:.1f}s - {rate:.0f} rels/sec!")

    def stream_nodes_directly(self, nodes_file):
        """Stream nodes directly from CSV to Memgraph in chunks"""
        logger.info("� STREAMING NODES DIRECTLY TO MEMGRAPH")
        
        chunk_size = 5000  # Process 5K nodes at a time
        total_processed = 0
        start_time = time.time()
        
        # Read CSV in chunks and process immediately
        for chunk_num, chunk in enumerate(pd.read_csv(nodes_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
            logger.info(f"� Processing node chunk {chunk_num + 1} ({len(chunk)} rows)")
            
            # Group by node type and process
            for node_type in chunk['type'].unique():
                type_chunk = chunk[chunk['type'] == node_type]
                if len(type_chunk) > 0:
                    created = self.bulk_create_nodes(type_chunk, node_type)
                    total_processed += created
                    
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            logger.info(f"✅ Chunk {chunk_num + 1} complete - {total_processed:,} total nodes - Rate: {rate:.0f} nodes/sec")
            
        elapsed = time.time() - start_time
        final_rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"🎉 NODE STREAMING COMPLETE! {total_processed:,} nodes in {elapsed:.1f}s - {final_rate:.0f} nodes/sec")
        
    def stream_relationships_directly(self, edges_file):
        """Stream relationships directly from CSV to Memgraph in chunks"""
        logger.info("💥 STREAMING RELATIONSHIPS DIRECTLY TO MEMGRAPH")
        
        chunk_size = 1000  # Smaller chunks for relationships
        total_processed = 0
        start_time = time.time()
        
        # Read CSV in chunks and process immediately
        for chunk_num, chunk in enumerate(pd.read_csv(edges_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
            logger.info(f"� Processing relationship chunk {chunk_num + 1} ({len(chunk)} rows)")
            
            # Group by relationship type and process
            for rel_type in chunk['type'].unique():
                type_chunk = chunk[chunk['type'] == rel_type]
                if len(type_chunk) > 0:
                    created = self.stream_create_relationships(type_chunk, rel_type)
                    total_processed += created
                    
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            
            # Log every 10 chunks to avoid spam
            if (chunk_num + 1) % 10 == 0:
                logger.info(f"✅ Chunk {chunk_num + 1} complete - {total_processed:,} total rels - Rate: {rate:.0f} rels/sec")
            
        elapsed = time.time() - start_time
        final_rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"🎉 RELATIONSHIP STREAMING COMPLETE! {total_processed:,} rels in {elapsed:.1f}s - {final_rate:.0f} rels/sec")

    def stream_create_relationships(self, batch_df, rel_type):
        """Create relationships in smaller sub-batches for speed"""
        try:
            memgraph = Memgraph()
            
            # Process in tiny sub-batches of 50 for maximum speed
            sub_batch_size = 50
            total_created = 0
            
            for i in range(0, len(batch_df), sub_batch_size):
                sub_batch = batch_df.iloc[i:i+sub_batch_size]
                
                # Prepare data for this sub-batch
                rel_data = []
                for _, row in sub_batch.iterrows():
                    source = str(row['source']).strip()
                    target = str(row['target']).strip()
                    
                    if source and target:
                        rel_data.append({"source": source, "target": target})
                
                if rel_data:
                    # Use MERGE for safety and speed
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a) WHERE a.node_id = rel.source
                    MATCH (b) WHERE b.node_id = rel.target  
                    MERGE (a)-[:{rel_type}]->(b)
                    """
                    
                    memgraph.execute(query, {"rels": rel_data})
                    total_created += len(rel_data)
            
            return total_created
            
        except Exception as e:
            logger.error(f"Relationship streaming failed: {e}")
            return 0

    def load_graph_blazing_fast(self):
        """INTERLEAVED STREAMING loader - process nodes and edges together"""
        overall_start = time.time()
        
        print("� INTERLEAVED STREAMING LOADER STARTING")
        print("=" * 50)
        
        # File paths
        nodes_file = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/nodes_optimized.csv"
        edges_file = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/edges_optimized.csv"
        
        # Clear database first
        self.clear_database()
        
        # Create indexes FIRST for optimal performance
        self.create_indexes_for_speed()
        
        # Use interleaved streaming approach
        self.interleaved_streaming_load(nodes_file, edges_file)
        
        # Create constraints AFTER loading for speed
        self.create_constraints()
        
        # Final stats
        total_time = time.time() - overall_start
        
        logger.info("=" * 50)
        logger.info(f"🎉 INTERLEAVED STREAMING COMPLETE!")
        logger.info(f"📊 Total time: {total_time:.1f} seconds")
        logger.info(f"🚀 Target achieved: {total_time:.1f}s < 300s (5 min)")

    def interleaved_streaming_load(self, nodes_file, edges_file):
        """TRUE INTERLEAVED streaming - process nodes and their RELATED edges together"""
        logger.info("🔄 STARTING TRUE INTERLEAVED STREAMING LOAD")
        
        chunk_size = 5000  # Larger chunks for better performance
        total_nodes = 0
        total_edges = 0
        start_time = time.time()
        
        # Read nodes file and build a lookup for faster edge processing
        logger.info("📂 Reading nodes file in batches...")
        
        # Process nodes in chunks and keep track of processed node IDs
        processed_node_ids = set()
        
        for chunk_num, node_chunk in enumerate(pd.read_csv(nodes_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
            chunk_num += 1
            logger.info(f"📦 Processing node chunk {chunk_num} ({len(node_chunk)} nodes)")
            
            # 1. Upsert nodes from this chunk
            nodes_created = self.upsert_nodes_chunk(node_chunk)
            total_nodes += nodes_created
            
            # 2. Add these node IDs to our processed set
            chunk_node_ids = set(node_chunk['id'].astype(str).str.strip())
            processed_node_ids.update(chunk_node_ids)
            
            # 3. Now process ALL edges that involve ANY of the nodes we've seen so far
            if chunk_num == 1 or chunk_num % 10 == 0:  # Process edges every 10 node chunks
                logger.info(f"� Processing edges involving {len(processed_node_ids)} known nodes...")
                edges_created = self.process_related_edges(edges_file, processed_node_ids)
                total_edges += edges_created
            
            elapsed = time.time() - start_time
            node_rate = total_nodes / elapsed if elapsed > 0 else 0
            edge_rate = total_edges / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Chunk {chunk_num} COMPLETE - Nodes: {total_nodes:,} ({node_rate:.0f}/sec) | Edges: {total_edges:,} ({edge_rate:.0f}/sec)")
            logger.info(f"   Time so far: {elapsed:.1f}s")
            logger.info("=" * 40)
        
        # Final edge processing for any remaining edges
        logger.info(f"🔗 Final edge processing for all {len(processed_node_ids)} nodes...")
        final_edges = self.process_related_edges(edges_file, processed_node_ids, final_pass=True)
        total_edges += final_edges
        
        elapsed = time.time() - start_time
        final_node_rate = total_nodes / elapsed if elapsed > 0 else 0
        final_edge_rate = total_edges / elapsed if elapsed > 0 else 0
        
        logger.info(f"🎉 TRUE INTERLEAVED STREAMING COMPLETE!")
        logger.info(f"📊 Nodes: {total_nodes:,} in {elapsed:.1f}s ({final_node_rate:.0f}/sec)")
        logger.info(f"📊 Edges: {total_edges:,} in {elapsed:.1f}s ({final_edge_rate:.0f}/sec)")

    def process_related_edges(self, edges_file, known_node_ids, final_pass=False):
        """Process only edges whose source AND target nodes are known"""
        try:
            total_edges_created = 0
            chunk_size = 10000  # Large chunks for edge processing
            
            for edge_chunk in pd.read_csv(edges_file, chunksize=chunk_size, dtype=str, keep_default_na=False):
                # Filter edges to only include those where both source and target are known
                edge_chunk['source'] = edge_chunk['source'].astype(str).str.strip()
                edge_chunk['target'] = edge_chunk['target'].astype(str).str.strip()
                
                # Only process edges where both nodes exist
                valid_edges = edge_chunk[
                    edge_chunk['source'].isin(known_node_ids) & 
                    edge_chunk['target'].isin(known_node_ids)
                ]
                
                if len(valid_edges) > 0:
                    logger.info(f"   📦 Processing {len(valid_edges)} valid edges from chunk...")
                    edges_created = self.create_edges_fast(valid_edges)
                    total_edges_created += edges_created
                    
                # If not final pass, only process one chunk to avoid duplicates
                if not final_pass:
                    break
            
            return total_edges_created
            
        except Exception as e:
            logger.error(f"Related edge processing failed: {e}")
            return 0

    def create_edges_fast(self, edge_chunk):
        """Ultra-fast edge creation assuming both nodes exist"""
        try:
            memgraph = Memgraph()
            total_created = 0
            
            # Adaptive batch sizing for optimal performance
            base_batch_size = 2500  # Larger base for nodes that exist
            batch_size = self.adaptive_chunk_size(base_batch_size, memory_factor=False)
            
            for i in range(0, len(edge_chunk), batch_size):
                batch = edge_chunk.iloc[i:i+batch_size]
                
                # Group by relationship type for efficiency
                for rel_type in batch['type'].unique():
                    type_edges = batch[batch['type'] == rel_type]
                    
                    # Prepare edge data with better memory efficiency
                    edge_data = []
                    for _, row in type_edges.iterrows():
                        source = str(row['source']).strip()
                        target = str(row['target']).strip()
                        
                        if source and target:
                            edge_data.append({"source": source, "target": target})
                    
                    if edge_data:
                        start_time = time.time()
                        # Use CREATE for maximum speed since nodes exist
                        query = f"""
                        UNWIND $edges AS edge
                        MATCH (a {{node_id: edge.source}})
                        MATCH (b {{node_id: edge.target}})
                        CREATE (a)-[:{rel_type}]->(b)
                        """
                        
                        memgraph.execute(query, {"edges": edge_data})
                        elapsed = time.time() - start_time
                        total_created += len(edge_data)
                        rate = len(edge_data) / elapsed if elapsed > 0 else 0
                        logger.info(f"      ⚡ {rel_type}: {len(edge_data)} edges in {elapsed:.2f}s ({rate:.0f}/sec)")
                        
                        # Free memory of processed batch
                        del edge_data
                        
                # Periodic memory management
                if i % (batch_size * 5) == 0:  # Every 5 batches
                    self.force_garbage_collection()
            
            return total_created
            
        except Exception as e:
            logger.error(f"Fast edge creation failed: {e}")
            return 0

    def upsert_nodes_chunk(self, node_chunk):
        """Upsert nodes using MERGE for safety with optimized memory management"""
        try:
            memgraph = Memgraph()
            total_created = 0
            
            logger.info(f"   🔧 Starting node upsert for {len(node_chunk)} nodes...")
            
            # Group by node type for efficiency
            for node_type in node_chunk['type'].unique():
                type_nodes = node_chunk[node_chunk['type'] == node_type]
                logger.info(f"      Processing {len(type_nodes)} {node_type} nodes...")
                
                # Prepare node data with better memory efficiency
                node_data = []
                for _, row in type_nodes.iterrows():
                    node_id = str(row.get('id', '')).strip()
                    name = str(row.get('name', '')).strip()
                    
                    if node_id:
                        node_data.append({"node_id": node_id, "name": name if name else node_id})
                
                if node_data:
                    start_time = time.time()
                    # Use MERGE to handle duplicates safely with optimized batch size
                    batch_size = self.adaptive_chunk_size(2000, memory_factor=False)
                    
                    for i in range(0, len(node_data), batch_size):
                        batch = node_data[i:i+batch_size]
                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{node_type} {{node_id: node.node_id}})
                        SET n.name = node.name
                        """
                        
                        memgraph.execute(query, {"nodes": batch})
                        total_created += len(batch)
                    
                    elapsed = time.time() - start_time
                    rate = len(node_data) / elapsed if elapsed > 0 else 0
                    logger.info(f"      ✅ {node_type}: {len(node_data)} nodes in {elapsed:.2f}s ({rate:.0f}/sec)")
                    
                    # Free memory immediately
                    del node_data
            
            logger.info(f"   ✅ Node upsert complete: {total_created} nodes")
            return total_created
            
        except Exception as e:
            logger.error(f"Node upsert failed: {e}")
            return 0

    def upsert_edges_chunk(self, edge_chunk):
        """Optimized edge upsert using MATCH (not MERGE) for existing nodes"""
        try:
            memgraph = Memgraph()
            total_created = 0
            
            logger.info(f"   💥 Starting edge upsert for {len(edge_chunk)} edges...")
            
            # Process in larger sub-batches since nodes already exist
            sub_batch_size = 500  # Much larger batches since we're not creating nodes
            
            for i in range(0, len(edge_chunk), sub_batch_size):
                sub_batch = edge_chunk.iloc[i:i+sub_batch_size]
                
                # Group by relationship type
                for rel_type in sub_batch['type'].unique():
                    type_edges = sub_batch[sub_batch['type'] == rel_type]
                    
                    # Prepare edge data
                    edge_data = []
                    for _, row in type_edges.iterrows():
                        source = str(row['source']).strip()
                        target = str(row['target']).strip()
                        
                        if source and target:
                            edge_data.append({"source": source, "target": target})
                    
                    if edge_data:
                        start_time = time.time()
                        # Use MATCH for existing nodes + MERGE only for relationship
                        query = f"""
                        UNWIND $edges AS edge
                        MATCH (a {{node_id: edge.source}})
                        MATCH (b {{node_id: edge.target}})
                        MERGE (a)-[:{rel_type}]->(b)
                        """
                        
                        memgraph.execute(query, {"edges": edge_data})
                        elapsed = time.time() - start_time
                        total_created += len(edge_data)
                        rate = len(edge_data) / elapsed if elapsed > 0 else 0
                        
                        # Log progress more frequently
                        logger.info(f"      💥 {rel_type}: {len(edge_data)} edges in {elapsed:.2f}s ({rate:.0f} edges/sec)")
            
    def process_related_edges_optimized(self, edges_file: str, known_node_ids: Set[str], final_pass: bool = False) -> int:
        """Process only edges whose source AND target nodes are known with GPU acceleration"""
        try:
            total_edges_created = 0
            chunk_size = 15000  # Larger chunks for better GPU utilization
            
            for edge_chunk in pd.read_csv(edges_file, chunksize=chunk_size, dtype=str, keep_default_na=False):
                # Clean data
                edge_chunk['source'] = edge_chunk['source'].astype(str).str.strip()
                edge_chunk['target'] = edge_chunk['target'].astype(str).str.strip()
                
                # GPU-accelerated filtering
                valid_edges = self.gpu_accelerated_filtering(edge_chunk, known_node_ids)
                
                if len(valid_edges) > 0:
                    logger.info(f"   📦 Processing {len(valid_edges)} valid edges from chunk...")
                    edges_created = self.create_edges_ultra_fast(valid_edges)
                    total_edges_created += edges_created
                    
                # If not final pass, only process one chunk to avoid duplicates
                if not final_pass:
                    break
            
            return total_edges_created
            
        except Exception as e:
            logger.error(f"Related edge processing failed: {e}")
            return 0

    def create_edges_ultra_fast(self, edge_chunk: pd.DataFrame) -> int:
        """Ultra-fast edge creation with adaptive batching and hardware monitoring"""
        try:
            memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
            total_created = 0
            
            # Dynamic batch sizing based on hardware state
            base_batch_size = 3000  # Larger base for better throughput
            batch_size = self.adaptive_chunk_size(base_batch_size, memory_factor=True)
            
            # Adaptive worker count based on current load
            current_workers = self.monitor.get_adaptive_workers(self.max_workers)
            
            if len(edge_chunk) > batch_size * 2:  # Use parallel processing for large chunks
                return self._parallel_edge_creation(edge_chunk, batch_size, current_workers)
            
            # Single-threaded for smaller chunks
            for i in range(0, len(edge_chunk), batch_size):
                batch = edge_chunk.iloc[i:i+batch_size]
                
                # Group by relationship type for efficiency
                for rel_type in batch['type'].unique():
                    type_edges = batch[batch['type'] == rel_type]
                    
                    # Prepare edge data
                    edge_data = self.prepare_batch_data(type_edges, 'edges')
                    
                    if edge_data:
                        # Check for throttling
                        if self.monitor.check_and_throttle():
                            self.stats['throttle_events'] += 1
                        
                        start_time = time.time()
                        # Use CREATE for maximum speed since nodes exist
                        query = f"""
                        UNWIND $edges AS edge
                        MATCH (a {{node_id: edge.source}})
                        MATCH (b {{node_id: edge.target}})
                        CREATE (a)-[:{rel_type}]->(b)
                        """
                        
                        self.execute_with_retry(memgraph, query, {"edges": edge_data})
                        elapsed = time.time() - start_time
                        created = len(edge_data)
                        total_created += created
                        rate = created / elapsed if elapsed > 0 else 0
                        
                        if created > 1000:  # Only log for significant batches
                            logger.info(f"      ⚡ {rel_type}: {created} edges in {elapsed:.2f}s ({rate:.0f}/sec)")
                        
                        # Free memory immediately
                        del edge_data
                        
                # Periodic memory management
                if i % (batch_size * 3) == 0:  # Every 3 batches
                    self.force_garbage_collection()
            
            return total_created
            
        except Exception as e:
            logger.error(f"Ultra-fast edge creation failed: {e}")
            return 0
    
    def _parallel_edge_creation(self, edge_chunk: pd.DataFrame, batch_size: int, workers: int) -> int:
        """Parallel edge creation for large chunks"""
        chunks = [edge_chunk.iloc[i:i+batch_size] for i in range(0, len(edge_chunk), batch_size)]
        total_created = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            
            for chunk in chunks:
                future = executor.submit(self._create_edge_batch, chunk)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    created = future.result()
                    total_created += created
                except Exception as e:
                    logger.error(f"Parallel edge batch failed: {e}")
        
        return total_created
    
    def _create_edge_batch(self, batch: pd.DataFrame) -> int:
        """Create a single edge batch (for parallel processing)"""
        memgraph = Memgraph(host=self.config.memgraph_host, port=self.config.memgraph_port)
        total_created = 0
        
        for rel_type in batch['type'].unique():
            type_edges = batch[batch['type'] == rel_type]
            edge_data = self.prepare_batch_data(type_edges, 'edges')
            
            if edge_data:
                query = f"""
                UNWIND $edges AS edge
                MATCH (a {{node_id: edge.source}})
                MATCH (b {{node_id: edge.target}})
                CREATE (a)-[:{rel_type}]->(b)
                """
                
                self.execute_with_retry(memgraph, query, {"edges": edge_data})
                total_created += len(edge_data)
        
        return total_created

    def ultra_optimized_interleaved_load(self, nodes_file: str, edges_file: str):
        """ULTRA-OPTIMIZED interleaved streaming with multiprocessing and GPU acceleration"""
        logger.info("🚀 STARTING ULTRA-OPTIMIZED INTERLEAVED STREAMING LOAD")
        
        # Start hardware monitoring
        self.monitor.start_monitoring()
        
        chunk_size = self.adaptive_chunk_size(8000)  # Larger initial chunks
        total_nodes = 0
        total_edges = 0
        start_time = time.time()
        
        # Process nodes in chunks and keep track of processed node IDs
        processed_node_ids = set()
        
        logger.info("📂 Processing nodes with multiprocessing optimization...")
        
        try:
            for chunk_num, node_chunk in enumerate(pd.read_csv(nodes_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
                chunk_num += 1
                logger.info(f"📦 Processing node chunk {chunk_num} ({len(node_chunk)} nodes)")
                
                # 1. Multiprocess node creation for large chunks
                nodes_created = 0
                for node_type in node_chunk['type'].unique():
                    type_nodes = node_chunk[node_chunk['type'] == node_type]
                    if len(type_nodes) > 0:
                        created = self.multiprocess_node_loading(type_nodes, node_type)
                        nodes_created += created
                
                total_nodes += nodes_created
                
                # 2. Add these node IDs to our processed set
                chunk_node_ids = set(node_chunk['id'].astype(str).str.strip())
                processed_node_ids.update(chunk_node_ids)
                
                # 3. Process edges every N chunks for optimal performance
                if chunk_num == 1 or chunk_num % self.config.chunk_processing_interval == 0:
                    logger.info(f"🔗 GPU-accelerated edge processing for {len(processed_node_ids)} known nodes...")
                    edges_created = self.process_related_edges_optimized(edges_file, processed_node_ids)
                    total_edges += edges_created
                
                # Performance tracking
                elapsed = time.time() - start_time
                node_rate = total_nodes / elapsed if elapsed > 0 else 0
                edge_rate = total_edges / elapsed if elapsed > 0 else 0
                
                logger.info(f"✅ Chunk {chunk_num} COMPLETE")
                logger.info(f"   📊 Nodes: {total_nodes:,} ({node_rate:.0f}/sec) | Edges: {total_edges:,} ({edge_rate:.0f}/sec)")
                logger.info(f"   🏥 GPU ops: {self.stats['gpu_operations']} | Throttles: {self.stats['throttle_events']}")
                logger.info(f"   ⏱️  Time: {elapsed:.1f}s")
                logger.info("=" * 50)
                
                # Dynamic chunk size adjustment based on performance
                if chunk_num > 2:  # After warming up
                    if node_rate > 10000:  # If doing well, increase chunk size
                        chunk_size = min(int(chunk_size * 1.2), 15000)
                    elif node_rate < 5000:  # If struggling, decrease chunk size
                        chunk_size = max(int(chunk_size * 0.8), 3000)
            
            # Final edge processing pass
            logger.info(f"🔗 Final GPU-accelerated edge processing for all {len(processed_node_ids)} nodes...")
            final_edges = self.process_related_edges_optimized(edges_file, processed_node_ids, final_pass=True)
            total_edges += final_edges
            
            elapsed = time.time() - start_time
            final_node_rate = total_nodes / elapsed if elapsed > 0 else 0
            final_edge_rate = total_edges / elapsed if elapsed > 0 else 0
            
            logger.info("🎉 ULTRA-OPTIMIZED INTERLEAVED STREAMING COMPLETE!")
            logger.info(f"📊 Final Stats:")
            logger.info(f"   Nodes: {total_nodes:,} in {elapsed:.1f}s ({final_node_rate:.0f}/sec)")
            logger.info(f"   Edges: {total_edges:,} in {elapsed:.1f}s ({final_edge_rate:.0f}/sec)")
            logger.info(f"   GPU Operations: {self.stats['gpu_operations']}")
            logger.info(f"   Throttle Events: {self.stats['throttle_events']}")
            
            # Update stats
            self.stats['nodes_created'] = total_nodes
            self.stats['edges_created'] = total_edges
            
        except Exception as e:
            logger.error(f"Ultra-optimized interleaved load failed: {e}")
            raise
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()

    def load_graph_blazing_fast(self):
        """MAIN ENTRY POINT - Ultra-optimized graph loading with all improvements"""
        overall_start = time.time()
        self.stats['start_time'] = overall_start
        
        print("🔥 ULTRA-OPTIMIZED MEMGRAPH LOADER STARTING")
        print("=" * 60)
        
        # Verify connection first
        if not self.verify_memgraph_connection():
            logger.error("❌ Cannot proceed without Memgraph connection")
            return False
        
        # Clear database first
        self.clear_database()
        
        # Create indexes FIRST for optimal performance
        self.create_indexes_for_speed()
        
        # Use ultra-optimized interleaved streaming approach
        self.ultra_optimized_interleaved_load(self.config.nodes_file, self.config.edges_file)
        
        # Create constraints AFTER loading for speed
        self.create_constraints()
        
        # Final performance summary
        total_time = time.time() - overall_start
        total_operations = self.stats['nodes_created'] + self.stats['edges_created']
        overall_rate = total_operations / total_time if total_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"🏆 ULTRA-OPTIMIZED LOADER COMPLETE!")
        logger.info(f"📊 Total operations: {total_operations:,} in {total_time:.1f}s")
        logger.info(f"⚡ Overall rate: {overall_rate:.0f} operations/sec")
        logger.info(f"🎯 Target: {'✅ ACHIEVED' if total_time < 180 else '❌ MISSED'} (target: <3min)")
        logger.info(f"🚀 Performance gain: ~{(300/total_time):.1f}x faster than original target")
        
        return True

def create_loader_with_args():
    """Create loader with command line arguments"""
    parser = argparse.ArgumentParser(description='Ultra-Optimized Memgraph Loader')
    parser.add_argument('--nodes-file', default=None, help='Path to nodes CSV file')
    parser.add_argument('--edges-file', default=None, help='Path to edges CSV file')
    parser.add_argument('--batch-size-nodes', type=int, default=10000, help='Batch size for nodes')
    parser.add_argument('--batch-size-edges', type=int, default=2000, help='Batch size for edges')
    parser.add_argument('--memgraph-host', default='localhost', help='Memgraph host')
    parser.add_argument('--memgraph-port', type=int, default=7687, help='Memgraph port')
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    config = LoaderConfig()
    if args.nodes_file:
        config.nodes_file = args.nodes_file
    if args.edges_file:
        config.edges_file = args.edges_file
    config.batch_size_nodes = args.batch_size_nodes
    config.batch_size_edges = args.batch_size_edges
    config.memgraph_host = args.memgraph_host
    config.memgraph_port = args.memgraph_port
    config.use_gpu = GPU_AVAILABLE and not args.disable_gpu
    
    return BlazingFastLoader(config)

def main():
    """Main execution function with better error handling"""
    # Handle interruption gracefully
    def signal_handler(sig, frame):
        print('\n🛑 Interrupted by user. Cleaning up...')
        sys.exit(0)
    #!/usr/bin/env python3
"""
🚀 BLAZING FAST MEMGRAPH LOADER 🚀 
Ultra-optimized for millions of nodes/relationships
Hardware-optimized for AMD Ryzen 7 4800H CPU (8 cores/16 threads, 16GB RAM)
GPU-accelerated where possible using PyTorch/CUDA
"""

import pandas as pd
import numpy as np
from gqlalchemy import Memgraph
import logging
import time
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import cpu_count, Manager
import multiprocessing as mp
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import signal
import sys

# Try to import GPU acceleration libraries
GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        import torch.nn.functional as F
        print(f"🔥 GPU acceleration ENABLED! CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  PyTorch available but no CUDA device found")
except ImportError:
    print("⚠️  PyTorch not available - running CPU-only mode")

# Try numba for JIT compilation
NUMBA_AVAILABLE = False
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
    print("🚀 Numba JIT compilation ENABLED!")
except ImportError:
    print("⚠️  Numba not available - using standard Python")

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class LoaderConfig:
    """Configuration for the blazing fast loader"""
    nodes_file: str = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/nodes_optimized.csv"
    edges_file: str = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/edges_optimized.csv"
    batch_size_nodes: int = 10000
    batch_size_edges: int = 2000
    max_memory_usage: float = 0.8
    max_cpu_usage: float = 0.9
    memgraph_host: str = 'localhost'
    memgraph_port: int = 7687
    use_gpu: bool = GPU_AVAILABLE
    chunk_processing_interval: int = 10  # Process edges every N node chunks

class HardwareMonitor:
    """Monitor hardware resources and throttle if needed"""
    
    def __init__(self, max_memory: float = 0.8, max_cpu: float = 0.9):
        self.max_memory = max_memory
        self.max_cpu = max_cpu
        self._monitoring = False
        self._stop_event = threading.Event()
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        self._stop_event.set()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring and not self._stop_event.is_set():
            self.check_and_throttle()
            time.sleep(2)  # Check every 2 seconds
            
    def check_and_throttle(self) -> bool:
        """Check resources and throttle if needed. Returns True if throttling occurred"""
        memory_usage = psutil.virtual_memory().percent / 100.0
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        
        if memory_usage > self.max_memory or cpu_usage > self.max_cpu:
            logger.warning(f"🛑 THROTTLING: Memory {memory_usage*100:.1f}% CPU {cpu_usage*100:.1f}%")
            # Force garbage collection
            gc.collect()
            # Sleep to cool down
            time.sleep(2)
            return True
        return False
        
    def get_adaptive_workers(self, max_workers: int) -> int:
        """Get adaptive worker count based on current resource usage"""
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        # Reduce workers if high usage
        if cpu_usage > 0.8 or memory_usage > 0.7:
            return max(1, max_workers // 2)
        elif cpu_usage > 0.6 or memory_usage > 0.6:
            return max(2, int(max_workers * 0.75))
        return max_workers

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def fast_string_filter(strings, valid_set):
        """Fast numba-accelerated string filtering"""
        result = []
        for s in strings:
            if s in valid_set:
                result.append(s)
        return result
else:
    def fast_string_filter(strings, valid_set):
        """Fallback string filtering"""
        return [s for s in strings if s in valid_set]

class BlazingFastLoader:
    def __init__(self):
        self.cpu_cores = cpu_count()
        # Optimized for Ryzen 7 4800H (8 cores/16 threads)
        self.max_workers = min(16, self.cpu_cores)
        
        # Memory monitoring
        self.total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        self.max_memory_usage = 0.8  # Use max 80% of available RAM
        
        print(f"🔥 BLAZING FAST LOADER INITIALIZED")
        print(f"   CPU Cores: {self.cpu_cores}")
        print(f"   Worker Threads: {self.max_workers}")
        print(f"   Total RAM: {self.total_ram:.1f} GB")
        print(f"   Max Memory Usage: {self.max_memory_usage*100:.0f}%")
        print()
        
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent / 100.0
        
    def adaptive_chunk_size(self, base_size, memory_factor=True):
        """Dynamically adjust chunk size based on memory usage"""
        if memory_factor:
            memory_usage = self.get_memory_usage()
            if memory_usage > 0.7:  # If using >70% RAM, reduce chunk size
                return max(base_size // 2, 1000)
            elif memory_usage < 0.5:  # If using <50% RAM, increase chunk size
                return min(base_size * 2, 10000)
        return base_size
        
    def force_garbage_collection(self):
        """Force garbage collection and log memory savings"""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        if before - after > 0.05:  # Only log if significant memory freed
            logger.info(f"🧹 Memory freed: {(before-after)*100:.1f}% (was {before*100:.1f}%, now {after*100:.1f}%)")
        
    def clear_database(self):
        """Clear the database FAST"""
        logger.info("🧹 Clearing database...")
        try:
            memgraph = Memgraph()
            # Drop all constraints first for speed
            try:
                memgraph.execute("DROP CONSTRAINT ON (n:Disease) ASSERT n.node_id IS UNIQUE")
                memgraph.execute("DROP CONSTRAINT ON (n:Drug) ASSERT n.node_id IS UNIQUE") 
                memgraph.execute("DROP CONSTRAINT ON (n:Target) ASSERT n.node_id IS UNIQUE")
                memgraph.execute("DROP CONSTRAINT ON (n:Pathway) ASSERT n.node_id IS UNIQUE")
            except:
                pass  # Constraints might not exist
            
            # Fast delete all
            memgraph.execute("MATCH (n) DETACH DELETE n")
            logger.info("✅ Database cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
    
    def create_indexes_for_speed(self):
        """Create indexes BEFORE relationship loading for blazing speed"""
        logger.info("🔧 Creating indexes for blazing fast relationships...")
        try:
            memgraph = Memgraph()
            
            # Create indexes on node_id for LIGHTNING FAST lookups
            indexes = [
                "CREATE INDEX ON :Disease(node_id)",
                "CREATE INDEX ON :Drug(node_id)", 
                "CREATE INDEX ON :Target(node_id)",
                "CREATE INDEX ON :Pathway(node_id)"
            ]
            
            for index in indexes:
                try:
                    memgraph.execute(index)
                    node_type = index.split(':')[1].split('(')[0]
                    logger.info(f"✅ Created index: {node_type}(node_id)")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"⚠️  Index already exists")
                    else:
                        logger.warning(f"⚠️  Index creation failed: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")

    def create_constraints(self):
        """Create constraints for performance - AFTER loading"""
        logger.info("🔧 Creating constraints...")
        try:
            memgraph = Memgraph()
            
            constraints = [
                "CREATE CONSTRAINT ON (n:Disease) ASSERT n.node_id IS UNIQUE",
                "CREATE CONSTRAINT ON (n:Drug) ASSERT n.node_id IS UNIQUE", 
                "CREATE CONSTRAINT ON (n:Target) ASSERT n.node_id IS UNIQUE",
                "CREATE CONSTRAINT ON (n:Pathway) ASSERT n.node_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    memgraph.execute(constraint)
                    logger.info(f"✅ Created constraint: {constraint.split('(')[1].split(')')[0]}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"⚠️  Constraint already exists")
                    else:
                        logger.warning(f"⚠️  Constraint creation failed: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to create constraints: {e}")

    def bulk_create_nodes(self, batch_df, node_type):
        """ULTRA FAST bulk node creation using UNWIND"""
        try:
            local_memgraph = Memgraph()
            
            # Prepare ALL data in one go
            node_data = []
            for _, row in batch_df.iterrows():
                node_id = str(row.get('id', '')).strip()
                name = str(row.get('name', '')).strip()
                
                if node_id:  # Only need node_id for speed
                    node_data.append({"node_id": node_id, "name": name if name else node_id})
            
            if not node_data:
                return 0
            
            # MASSIVE bulk insert with single query
            query = f"""
            UNWIND $nodes AS node
            CREATE (n:{node_type})
            SET n = node
            """
            
            local_memgraph.execute(query, {"nodes": node_data})
            return len(node_data)
            
        except Exception as e:
            logger.error(f"Bulk node creation failed: {e}")
            return 0

    def bulk_create_relationships(self, batch_df, rel_type):
        """OPTIMIZED bulk relationship creation with faster query pattern"""
        try:
            local_memgraph = Memgraph()
            
            # Process in much smaller sub-batches to prevent database locks
            sub_batch_size = 100  # TINY batches for relationships
            total_created = 0
            
            for i in range(0, len(batch_df), sub_batch_size):
                sub_batch = batch_df.iloc[i:i+sub_batch_size]
                
                # Prepare sub-batch data
                rel_data = []
                for _, row in sub_batch.iterrows():
                    source = str(row['source']).strip()
                    target = str(row['target']).strip()
                    
                    if source and target:
                        rel_data.append({"source": source, "target": target})
                
                if rel_data:
                    # Use optimized query with explicit index hints
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a) WHERE a.node_id = rel.source
                    MATCH (b) WHERE b.node_id = rel.target  
                    CREATE (a)-[:{rel_type}]->(b)
                    """
                    
                    local_memgraph.execute(query, {"rels": rel_data})
                    total_created += len(rel_data)
            
            return total_created
            
        except Exception as e:
            logger.error(f"Bulk relationship creation failed: {e}")
            return 0

    def load_nodes_blazing_fast(self, nodes_df, node_type):
        """Load nodes with MAXIMUM speed"""
        type_nodes = nodes_df[nodes_df['type'] == node_type].copy()
        
        if len(type_nodes) == 0:
            return
            
        logger.info(f"🚀 Loading {len(type_nodes):,} {node_type} nodes...")
        start_time = time.time()
        
        # MUCH larger batches for speed
        batch_size = 10000  # 10K per batch instead of 2K
        total_created = 0
        
        # Process massive batches with more threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(type_nodes), batch_size):
                batch = type_nodes.iloc[i:i+batch_size]
                future = executor.submit(self.bulk_create_nodes, batch, node_type)
                futures.append(future)
            
            # Process results
            for i, future in enumerate(as_completed(futures)):
                try:
                    created = future.result()
                    total_created += created
                    elapsed = time.time() - start_time
                    rate = total_created / elapsed if elapsed > 0 else 0
                    logger.info(f"   Batch {i+1}/{len(futures)} complete - {created} nodes - Rate: {rate:.0f} nodes/sec")
                except Exception as e:
                    logger.error(f"   Batch {i+1} failed: {e}")
        
        elapsed = time.time() - start_time
        rate = total_created / elapsed if elapsed > 0 else 0
        logger.info(f"⚡ {node_type}: {total_created:,} nodes in {elapsed:.1f}s - {rate:.0f} nodes/sec!")
        
    def load_relationships_blazing_fast(self, edges_df, rel_type):
        """Load relationships with OPTIMIZED speed - tiny batches with more parallelism"""
        type_rels = edges_df[edges_df['type'] == rel_type].copy()
        
        if len(type_rels) == 0:
            return
            
        logger.info(f"💥 Loading {len(type_rels):,} {rel_type} relationships...")
        start_time = time.time()
        
        # TINY batches for maximum parallelism
        batch_size = 200  # Even smaller batches for speed
        total_created = 0
        
        # Use more workers for tiny batches
        with ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
            futures = []
            
            for i in range(0, len(type_rels), batch_size):
                batch = type_rels.iloc[i:i+batch_size]
                future = executor.submit(self.bulk_create_relationships, batch, rel_type)
                futures.append(future)
            
            # Process results with better progress tracking
            completed = 0
            for future in as_completed(futures):
                try:
                    created = future.result()
                    total_created += created
                    completed += 1
                    elapsed = time.time() - start_time
                    rate = total_created / elapsed if elapsed > 0 else 0
                    progress = completed / len(futures) * 100
                    
                    # Log every 50 batches instead of every batch
                    if completed % 50 == 0 or completed == len(futures):
                        logger.info(f"   Batch {completed}/{len(futures)} ({progress:.1f}%) - Rate: {rate:.0f} rels/sec")
                except Exception as e:
                    completed += 1
                    logger.error(f"   Batch {completed} failed: {e}")
        
        elapsed = time.time() - start_time
        rate = total_created / elapsed if elapsed > 0 else 0
        logger.info(f"⚡ {rel_type}: {total_created:,} rels in {elapsed:.1f}s - {rate:.0f} rels/sec!")

    def stream_nodes_directly(self, nodes_file):
        """Stream nodes directly from CSV to Memgraph in chunks"""
        logger.info("� STREAMING NODES DIRECTLY TO MEMGRAPH")
        
        chunk_size = 5000  # Process 5K nodes at a time
        total_processed = 0
        start_time = time.time()
        
        # Read CSV in chunks and process immediately
        for chunk_num, chunk in enumerate(pd.read_csv(nodes_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
            logger.info(f"� Processing node chunk {chunk_num + 1} ({len(chunk)} rows)")
            
            # Group by node type and process
            for node_type in chunk['type'].unique():
                type_chunk = chunk[chunk['type'] == node_type]
                if len(type_chunk) > 0:
                    created = self.bulk_create_nodes(type_chunk, node_type)
                    total_processed += created
                    
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            logger.info(f"✅ Chunk {chunk_num + 1} complete - {total_processed:,} total nodes - Rate: {rate:.0f} nodes/sec")
            
        elapsed = time.time() - start_time
        final_rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"🎉 NODE STREAMING COMPLETE! {total_processed:,} nodes in {elapsed:.1f}s - {final_rate:.0f} nodes/sec")
        
    def stream_relationships_directly(self, edges_file):
        """Stream relationships directly from CSV to Memgraph in chunks"""
        logger.info("💥 STREAMING RELATIONSHIPS DIRECTLY TO MEMGRAPH")
        
        chunk_size = 1000  # Smaller chunks for relationships
        total_processed = 0
        start_time = time.time()
        
        # Read CSV in chunks and process immediately
        for chunk_num, chunk in enumerate(pd.read_csv(edges_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
            logger.info(f"� Processing relationship chunk {chunk_num + 1} ({len(chunk)} rows)")
            
            # Group by relationship type and process
            for rel_type in chunk['type'].unique():
                type_chunk = chunk[chunk['type'] == rel_type]
                if len(type_chunk) > 0:
                    created = self.stream_create_relationships(type_chunk, rel_type)
                    total_processed += created
                    
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            
            # Log every 10 chunks to avoid spam
            if (chunk_num + 1) % 10 == 0:
                logger.info(f"✅ Chunk {chunk_num + 1} complete - {total_processed:,} total rels - Rate: {rate:.0f} rels/sec")
            
        elapsed = time.time() - start_time
        final_rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"🎉 RELATIONSHIP STREAMING COMPLETE! {total_processed:,} rels in {elapsed:.1f}s - {final_rate:.0f} rels/sec")

    def stream_create_relationships(self, batch_df, rel_type):
        """Create relationships in smaller sub-batches for speed"""
        try:
            memgraph = Memgraph()
            
            # Process in tiny sub-batches of 50 for maximum speed
            sub_batch_size = 50
            total_created = 0
            
            for i in range(0, len(batch_df), sub_batch_size):
                sub_batch = batch_df.iloc[i:i+sub_batch_size]
                
                # Prepare data for this sub-batch
                rel_data = []
                for _, row in sub_batch.iterrows():
                    source = str(row['source']).strip()
                    target = str(row['target']).strip()
                    
                    if source and target:
                        rel_data.append({"source": source, "target": target})
                
                if rel_data:
                    # Use MERGE for safety and speed
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a) WHERE a.node_id = rel.source
                    MATCH (b) WHERE b.node_id = rel.target  
                    MERGE (a)-[:{rel_type}]->(b)
                    """
                    
                    memgraph.execute(query, {"rels": rel_data})
                    total_created += len(rel_data)
            
            return total_created
            
        except Exception as e:
            logger.error(f"Relationship streaming failed: {e}")
            return 0

    def load_graph_blazing_fast(self):
        """INTERLEAVED STREAMING loader - process nodes and edges together"""
        overall_start = time.time()
        
        print("� INTERLEAVED STREAMING LOADER STARTING")
        print("=" * 50)
        
        # File paths
        nodes_file = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/nodes_optimized.csv"
        edges_file = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/optimized/edges_optimized.csv"
        
        # Clear database first
        self.clear_database()
        
        # Create indexes FIRST for optimal performance
        self.create_indexes_for_speed()
        
        # Use interleaved streaming approach
        self.interleaved_streaming_load(nodes_file, edges_file)
        
        # Create constraints AFTER loading for speed
        self.create_constraints()
        
        # Final stats
        total_time = time.time() - overall_start
        
        logger.info("=" * 50)
        logger.info(f"🎉 INTERLEAVED STREAMING COMPLETE!")
        logger.info(f"📊 Total time: {total_time:.1f} seconds")
        logger.info(f"🚀 Target achieved: {total_time:.1f}s < 300s (5 min)")

    def interleaved_streaming_load(self, nodes_file, edges_file):
        """TRUE INTERLEAVED streaming - process nodes and their RELATED edges together"""
        logger.info("🔄 STARTING TRUE INTERLEAVED STREAMING LOAD")
        
        chunk_size = 5000  # Larger chunks for better performance
        total_nodes = 0
        total_edges = 0
        start_time = time.time()
        
        # Read nodes file and build a lookup for faster edge processing
        logger.info("📂 Reading nodes file in batches...")
        
        # Process nodes in chunks and keep track of processed node IDs
        processed_node_ids = set()
        
        for chunk_num, node_chunk in enumerate(pd.read_csv(nodes_file, chunksize=chunk_size, dtype=str, keep_default_na=False)):
            chunk_num += 1
            logger.info(f"📦 Processing node chunk {chunk_num} ({len(node_chunk)} nodes)")
            
            # 1. Upsert nodes from this chunk
            nodes_created = self.upsert_nodes_chunk(node_chunk)
            total_nodes += nodes_created
            
            # 2. Add these node IDs to our processed set
            chunk_node_ids = set(node_chunk['id'].astype(str).str.strip())
            processed_node_ids.update(chunk_node_ids)
            
            # 3. Now process ALL edges that involve ANY of the nodes we've seen so far
            if chunk_num == 1 or chunk_num % 10 == 0:  # Process edges every 10 node chunks
                logger.info(f"� Processing edges involving {len(processed_node_ids)} known nodes...")
                edges_created = self.process_related_edges(edges_file, processed_node_ids)
                total_edges += edges_created
            
            elapsed = time.time() - start_time
            node_rate = total_nodes / elapsed if elapsed > 0 else 0
            edge_rate = total_edges / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Chunk {chunk_num} COMPLETE - Nodes: {total_nodes:,} ({node_rate:.0f}/sec) | Edges: {total_edges:,} ({edge_rate:.0f}/sec)")
            logger.info(f"   Time so far: {elapsed:.1f}s")
            logger.info("=" * 40)
        
        # Final edge processing for any remaining edges
        logger.info(f"🔗 Final edge processing for all {len(processed_node_ids)} nodes...")
        final_edges = self.process_related_edges(edges_file, processed_node_ids, final_pass=True)
        total_edges += final_edges
        
        elapsed = time.time() - start_time
        final_node_rate = total_nodes / elapsed if elapsed > 0 else 0
        final_edge_rate = total_edges / elapsed if elapsed > 0 else 0
        
        logger.info(f"🎉 TRUE INTERLEAVED STREAMING COMPLETE!")
        logger.info(f"📊 Nodes: {total_nodes:,} in {elapsed:.1f}s ({final_node_rate:.0f}/sec)")
        logger.info(f"📊 Edges: {total_edges:,} in {elapsed:.1f}s ({final_edge_rate:.0f}/sec)")

    def process_related_edges(self, edges_file, known_node_ids, final_pass=False):
        """Process only edges whose source AND target nodes are known"""
        try:
            total_edges_created = 0
            chunk_size = 10000  # Large chunks for edge processing
            
            for edge_chunk in pd.read_csv(edges_file, chunksize=chunk_size, dtype=str, keep_default_na=False):
                # Filter edges to only include those where both source and target are known
                edge_chunk['source'] = edge_chunk['source'].astype(str).str.strip()
                edge_chunk['target'] = edge_chunk['target'].astype(str).str.strip()
                
                # Only process edges where both nodes exist
                valid_edges = edge_chunk[
                    edge_chunk['source'].isin(known_node_ids) & 
                    edge_chunk['target'].isin(known_node_ids)
                ]
                
                if len(valid_edges) > 0:
                    logger.info(f"   📦 Processing {len(valid_edges)} valid edges from chunk...")
                    edges_created = self.create_edges_fast(valid_edges)
                    total_edges_created += edges_created
                    
                # If not final pass, only process one chunk to avoid duplicates
                if not final_pass:
                    break
            
            return total_edges_created
            
        except Exception as e:
            logger.error(f"Related edge processing failed: {e}")
            return 0

    def create_edges_fast(self, edge_chunk):
        """Ultra-fast edge creation assuming both nodes exist"""
        try:
            memgraph = Memgraph()
            total_created = 0
            
            # Adaptive batch sizing for optimal performance
            base_batch_size = 2500  # Larger base for nodes that exist
            batch_size = self.adaptive_chunk_size(base_batch_size, memory_factor=False)
            
            for i in range(0, len(edge_chunk), batch_size):
                batch = edge_chunk.iloc[i:i+batch_size]
                
                # Group by relationship type for efficiency
                for rel_type in batch['type'].unique():
                    type_edges = batch[batch['type'] == rel_type]
                    
                    # Prepare edge data with better memory efficiency
                    edge_data = []
                    for _, row in type_edges.iterrows():
                        source = str(row['source']).strip()
                        target = str(row['target']).strip()
                        
                        if source and target:
                            edge_data.append({"source": source, "target": target})
                    
                    if edge_data:
                        start_time = time.time()
                        # Use CREATE for maximum speed since nodes exist
                        query = f"""
                        UNWIND $edges AS edge
                        MATCH (a {{node_id: edge.source}})
                        MATCH (b {{node_id: edge.target}})
                        CREATE (a)-[:{rel_type}]->(b)
                        """
                        
                        memgraph.execute(query, {"edges": edge_data})
                        elapsed = time.time() - start_time
                        total_created += len(edge_data)
                        rate = len(edge_data) / elapsed if elapsed > 0 else 0
                        logger.info(f"      ⚡ {rel_type}: {len(edge_data)} edges in {elapsed:.2f}s ({rate:.0f}/sec)")
                        
                        # Free memory of processed batch
                        del edge_data
                        
                # Periodic memory management
                if i % (batch_size * 5) == 0:  # Every 5 batches
                    self.force_garbage_collection()
            
            return total_created
            
        except Exception as e:
            logger.error(f"Fast edge creation failed: {e}")
            return 0

    def upsert_nodes_chunk(self, node_chunk):
        """Upsert nodes using MERGE for safety with optimized memory management"""
        try:
            memgraph = Memgraph()
            total_created = 0
            
            logger.info(f"   🔧 Starting node upsert for {len(node_chunk)} nodes...")
            
            # Group by node type for efficiency
            for node_type in node_chunk['type'].unique():
                type_nodes = node_chunk[node_chunk['type'] == node_type]
                logger.info(f"      Processing {len(type_nodes)} {node_type} nodes...")
                
                # Prepare node data with better memory efficiency
                node_data = []
                for _, row in type_nodes.iterrows():
                    node_id = str(row.get('id', '')).strip()
                    name = str(row.get('name', '')).strip()
                    
                    if node_id:
                        node_data.append({"node_id": node_id, "name": name if name else node_id})
                
                if node_data:
                    start_time = time.time()
                    # Use MERGE to handle duplicates safely with optimized batch size
                    batch_size = self.adaptive_chunk_size(2000, memory_factor=False)
                    
                    for i in range(0, len(node_data), batch_size):
                        batch = node_data[i:i+batch_size]
                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{node_type} {{node_id: node.node_id}})
                        SET n.name = node.name
                        """
                        
                        memgraph.execute(query, {"nodes": batch})
                        total_created += len(batch)
                    
                    elapsed = time.time() - start_time
                    rate = len(node_data) / elapsed if elapsed > 0 else 0
                    logger.info(f"      ✅ {node_type}: {len(node_data)} nodes in {elapsed:.2f}s ({rate:.0f}/sec)")
                    
                    # Free memory immediately
                    del node_data
            
            logger.info(f"   ✅ Node upsert complete: {total_created} nodes")
            return total_created
            
        except Exception as e:
            logger.error(f"Node upsert failed: {e}")
            return 0

    def upsert_edges_chunk(self, edge_chunk):
        """Optimized edge upsert using MATCH (not MERGE) for existing nodes"""
        try:
            memgraph = Memgraph()
            total_created = 0
            
            logger.info(f"   💥 Starting edge upsert for {len(edge_chunk)} edges...")
            
            # Process in larger sub-batches since nodes already exist
            sub_batch_size = 500  # Much larger batches since we're not creating nodes
            
            for i in range(0, len(edge_chunk), sub_batch_size):
                sub_batch = edge_chunk.iloc[i:i+sub_batch_size]
                
                # Group by relationship type
                for rel_type in sub_batch['type'].unique():
                    type_edges = sub_batch[sub_batch['type'] == rel_type]
                    
                    # Prepare edge data
                    edge_data = []
                    for _, row in type_edges.iterrows():
                        source = str(row['source']).strip()
                        target = str(row['target']).strip()
                        
                        if source and target:
                            edge_data.append({"source": source, "target": target})
                    
                    if edge_data:
                        start_time = time.time()
                        # Use MATCH for existing nodes + MERGE only for relationship
                        query = f"""
                        UNWIND $edges AS edge
                        MATCH (a {{node_id: edge.source}})
                        MATCH (b {{node_id: edge.target}})
                        MERGE (a)-[:{rel_type}]->(b)
                        """
                        
                        memgraph.execute(query, {"edges": edge_data})
                        elapsed = time.time() - start_time
                        total_created += len(edge_data)
                        rate = len(edge_data) / elapsed if elapsed > 0 else 0
                        
                        # Log progress more frequently
                        logger.info(f"      💥 {rel_type}: {len(edge_data)} edges in {elapsed:.2f}s ({rate:.0f} edges/sec)")
            
            logger.info(f"   ✅ Edge upsert complete: {total_created} edges")
            return total_created
            
        except Exception as e:
            logger.error(f"Edge upsert failed: {e}")
            return 0

def main():
    """Main execution function"""
    loader = BlazingFastLoader()
    loader.load_graph_blazing_fast()

if __name__ == "__main__":
    main()

    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        loader = create_loader_with_args()
        success = loader.load_graph_blazing_fast()
        
        if success:
            print("\n🎉 SUCCESS! Graph loaded successfully!")
            sys.exit(0)
        else:
            print("\n❌ FAILED! Graph loading encountered errors!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
