"""
Optimized Context Builder for Ryzen 4800H + GTX 1650Ti + 16GB RAM
Key optimizations:
- LRU caching for context fragments
- Parallel context assembly 
- Memory-aware context sizing
- Hardware-specific batch processing
- Intelligent context pruning and relevance scoring
"""

import logging
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from functools import lru_cache
import concurrent.futures
import psutil
import pickle
import hashlib

logger = logging.getLogger(__name__)

class ContextFragment:
    """Individual context fragment with metadata"""
    
    def __init__(self, content: str, source: str, relevance: float = 1.0, 
                 timestamp: Optional[float] = None):
        self.content = content
        self.source = source
        self.relevance = relevance
        self.timestamp = timestamp or time.time()
        self.hash = self._compute_hash()
        self.size = len(content.encode('utf-8'))
    
    def _compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()[:16]
    
    def __eq__(self, other):
        return isinstance(other, ContextFragment) and self.hash == other.hash
    
    def __hash__(self):
        return hash(self.hash)

class MemoryAwareContextCache:
    """LRU cache with memory pressure awareness"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, ContextFragment] = {}
        self.access_times: Dict[str, float] = {}
        self.current_memory = 0
        self.lock = threading.RLock()
        
        # Track cache performance
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[ContextFragment]:
        """Get fragment from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, fragment: ContextFragment):
        """Add fragment to cache with memory management"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self.current_memory -= self.cache[key].size
                del self.cache[key]
                del self.access_times[key]
            
            # Check memory pressure
            while (self.current_memory + fragment.size > self.max_memory_bytes or 
                   len(self.cache) >= self.max_size):
                self._evict_least_recently_used()
            
            # Add new fragment
            self.cache[key] = fragment
            self.access_times[key] = time.time()
            self.current_memory += fragment.size
    
    def _evict_least_recently_used(self):
        """Evict least recently used fragment"""
        if not self.cache:
            return
        
        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        fragment = self.cache[lru_key]
        self.current_memory -= fragment.size
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'memory_usage_mb': self.current_memory / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
        }

class OptimizedContextBuilder:
    """Hardware-optimized context builder with parallel processing"""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 1000, 
                 max_context_length: int = 8000):
        self.max_workers = max_workers
        self.max_context_length = max_context_length
        
        # Initialize caches
        self.fragment_cache = MemoryAwareContextCache(
            max_size=cache_size,
            max_memory_mb=128  # 128MB for context fragments
        )
        
        self.entity_cache = MemoryAwareContextCache(
            max_size=cache_size // 2,
            max_memory_mb=64   # 64MB for entity contexts
        )
        
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ContextBuilder"
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.build_times = []
        self.fragment_counts = []
    
    def build_context_parallel(self, entities: List[str], 
                             relationships: List[Dict[str, Any]],
                             vector_results: List[Dict[str, Any]],
                             max_length: Optional[int] = None) -> str:
        """
        Build context in parallel with caching and optimization
        
        Args:
            entities: List of entity names
            relationships: List of relationship dictionaries
            vector_results: List of vector search results
            max_length: Maximum context length (uses default if None)
            
        Returns:
            Optimized context string
        """
        start_time = time.time()
        max_length = max_length or self.max_context_length
        
        try:
            self.logger.debug(f"Building context for {len(entities)} entities, "
                            f"{len(relationships)} relationships, "
                            f"{len(vector_results)} vector results")
            
            # Phase 1: Parallel fragment generation
            fragments = self._generate_fragments_parallel(
                entities, relationships, vector_results
            )
            
            # Phase 2: Intelligent fragment selection and assembly
            optimized_context = self._assemble_context_optimized(
                fragments, max_length
            )
            
            build_time = time.time() - start_time
            self.build_times.append(build_time)
            self.fragment_counts.append(len(fragments))
            
            self.logger.debug(f"Context built in {build_time:.3f}s with "
                            f"{len(fragments)} fragments -> {len(optimized_context)} chars")
            
            return optimized_context
            
        except Exception as e:
            self.logger.error(f"Context building failed: {e}")
            raise
    
    def _generate_fragments_parallel(self, entities: List[str],
                                   relationships: List[Dict[str, Any]],
                                   vector_results: List[Dict[str, Any]]) -> List[ContextFragment]:
        """Generate context fragments in parallel"""
        
        # Submit parallel tasks
        futures = []
        
        # Entity fragments
        if entities:
            futures.append(
                self.executor.submit(self._build_entity_fragments, entities)
            )
        
        # Relationship fragments
        if relationships:
            futures.append(
                self.executor.submit(self._build_relationship_fragments, relationships)
            )
        
        # Vector result fragments  
        if vector_results:
            futures.append(
                self.executor.submit(self._build_vector_fragments, vector_results)
            )
        
        # Collect results
        all_fragments = []
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                fragments = future.result()
                all_fragments.extend(fragments)
            except Exception as e:
                self.logger.warning(f"Fragment generation failed: {e}")
        
        return all_fragments
    
    def _build_entity_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Build entity context fragments with caching"""
        fragments = []
        
        for entity in entities:
            cache_key = f"entity:{entity}"
            
            # Check cache first
            cached_fragment = self.entity_cache.get(cache_key)
            if cached_fragment:
                fragments.append(cached_fragment)
                continue
            
            # Generate new fragment
            try:
                content = self._format_entity_context(entity)
                if content:
                    fragment = ContextFragment(
                        content=content,
                        source=f"entity:{entity}",
                        relevance=0.8  # High relevance for direct entities
                    )
                    
                    # Cache for future use
                    self.entity_cache.put(cache_key, fragment)
                    fragments.append(fragment)
                    
            except Exception as e:
                self.logger.warning(f"Entity fragment generation failed for {entity}: {e}")
        
        return fragments
    
    def _build_relationship_fragments(self, relationships: List[Dict[str, Any]]) -> List[ContextFragment]:
        """Build relationship context fragments"""
        fragments = []
        
        # Group relationships by type for better context
        rel_groups = defaultdict(list)
        for rel in relationships:
            rel_type = rel.get('type', 'unknown')
            rel_groups[rel_type].append(rel)
        
        # Process each group
        for rel_type, group_rels in rel_groups.items():
            cache_key = f"rel_group:{rel_type}:{len(group_rels)}"
            
            # Check cache
            cached_fragment = self.fragment_cache.get(cache_key)
            if cached_fragment:
                fragments.append(cached_fragment)
                continue
            
            # Generate new fragment
            try:
                content = self._format_relationship_group(rel_type, group_rels)
                if content:
                    fragment = ContextFragment(
                        content=content,
                        source=f"relationships:{rel_type}",
                        relevance=0.7  # Good relevance for relationships
                    )
                    
                    self.fragment_cache.put(cache_key, fragment)
                    fragments.append(fragment)
                    
            except Exception as e:
                self.logger.warning(f"Relationship fragment generation failed: {e}")
        
        return fragments
    
    def _build_vector_fragments(self, vector_results: List[Dict[str, Any]]) -> List[ContextFragment]:
        """Build vector search result fragments"""
        fragments = []
        
        for i, result in enumerate(vector_results):
            try:
                content = self._format_vector_result(result)
                if content:
                    # Calculate relevance based on score and position
                    score = result.get('score', 0.5)
                    position_factor = 1.0 - (i * 0.1)  # Decay by position
                    relevance = min(1.0, score * position_factor)
                    
                    fragment = ContextFragment(
                        content=content,
                        source=f"vector:{i}",
                        relevance=relevance
                    )
                    fragments.append(fragment)
                    
            except Exception as e:
                self.logger.warning(f"Vector fragment generation failed: {e}")
        
        return fragments
    
    def _assemble_context_optimized(self, fragments: List[ContextFragment], 
                                  max_length: int) -> str:
        """Intelligently assemble fragments into optimized context"""
        
        if not fragments:
            return ""
        
        # Remove duplicates based on hash
        unique_fragments = list(dict.fromkeys(fragments))
        
        # Sort by relevance score (descending)
        sorted_fragments = sorted(
            unique_fragments, 
            key=lambda f: f.relevance, 
            reverse=True
        )
        
        # Greedy selection with length constraint
        selected_fragments = []
        current_length = 0
        
        # Reserve space for formatting
        format_overhead = 100
        effective_max_length = max_length - format_overhead
        
        for fragment in sorted_fragments:
            fragment_length = len(fragment.content)
            
            # Check if fragment fits
            if current_length + fragment_length <= effective_max_length:
                selected_fragments.append(fragment)
                current_length += fragment_length
            else:
                # Try to fit a truncated version if it's high relevance
                if fragment.relevance > 0.8:
                    remaining_space = effective_max_length - current_length
                    if remaining_space > 50:  # Minimum useful size
                        truncated_content = fragment.content[:remaining_space - 3] + "..."
                        truncated_fragment = ContextFragment(
                            content=truncated_content,
                            source=fragment.source,
                            relevance=fragment.relevance * 0.8  # Penalty for truncation
                        )
                        selected_fragments.append(truncated_fragment)
                        break
        
        # Assemble final context with structure
        return self._format_final_context(selected_fragments)
    
    def _format_entity_context(self, entity: str) -> str:
        """Format entity for context"""
        return f"Entity: {entity}"
    
    def _format_relationship_group(self, rel_type: str, relationships: List[Dict[str, Any]]) -> str:
        """Format relationship group for context"""
        if not relationships:
            return ""
        
        lines = [f"\n{rel_type.upper()} Relationships:"]
        
        for rel in relationships[:5]:  # Limit to top 5 per group
            source = rel.get('source', 'Unknown')
            target = rel.get('target', 'Unknown')
            description = rel.get('description', '')
            
            if description:
                lines.append(f"- {source} → {target}: {description}")
            else:
                lines.append(f"- {source} → {target}")
        
        if len(relationships) > 5:
            lines.append(f"... and {len(relationships) - 5} more {rel_type} relationships")
        
        return "\n".join(lines)
    
    def _format_vector_result(self, result: Dict[str, Any]) -> str:
        """Format vector search result for context"""
        content = result.get('content', '')
        entity = result.get('entity', '')
        
        if entity and content:
            return f"Vector Result ({entity}): {content}"
        elif content:
            return f"Vector Result: {content}"
        else:
            return ""
    
    def _format_final_context(self, fragments: List[ContextFragment]) -> str:
        """Format final context with structure and metadata"""
        if not fragments:
            return "No relevant context found."
        
        lines = ["=== CONTEXT ==="]
        
        # Group fragments by source type
        entity_fragments = [f for f in fragments if f.source.startswith('entity:')]
        rel_fragments = [f for f in fragments if f.source.startswith('relationships:')]
        vector_fragments = [f for f in fragments if f.source.startswith('vector:')]
        
        # Add entity section
        if entity_fragments:
            lines.append("\n[ENTITIES]")
            for fragment in entity_fragments:
                lines.append(fragment.content)
        
        # Add relationship section
        if rel_fragments:
            lines.append("\n[RELATIONSHIPS]")
            for fragment in rel_fragments:
                lines.append(fragment.content)
        
        # Add vector results section
        if vector_fragments:
            lines.append("\n[ADDITIONAL CONTEXT]")
            for fragment in vector_fragments:
                lines.append(fragment.content)
        
        lines.append("=== END CONTEXT ===")
        
        return "\n".join(lines)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.build_times:
            return {"status": "No builds completed yet"}
        
        avg_build_time = sum(self.build_times) / len(self.build_times)
        avg_fragments = sum(self.fragment_counts) / len(self.fragment_counts)
        
        fragment_stats = self.fragment_cache.get_stats()
        entity_stats = self.entity_cache.get_stats()
        
        return {
            "avg_build_time": avg_build_time,
            "avg_fragments": avg_fragments,
            "total_builds": len(self.build_times),
            "fragment_cache": fragment_stats,
            "entity_cache": entity_stats,
            "memory_usage": psutil.virtual_memory().percent
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.warning(f"Executor shutdown failed: {e}")

# Factory function for easy integration
def create_optimized_context_builder(max_workers: Optional[int] = None) -> OptimizedContextBuilder:
    """
    Create context builder optimized for current hardware
    
    Args:
        max_workers: Number of worker threads (auto-detected if None)
        
    Returns:
        OptimizedContextBuilder instance
    """
    if max_workers is None:
        # Use 50% of available cores, minimum 2, maximum 6
        cpu_count = psutil.cpu_count()
        max_workers = max(2, min(6, int(cpu_count * 0.5)))
    
    # Adjust cache size based on available memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cache_size = int(min(2000, memory_gb * 100))  # 100 items per GB, max 2000
    
    return OptimizedContextBuilder(
        max_workers=max_workers,
        cache_size=cache_size,
        max_context_length=8000
    )
