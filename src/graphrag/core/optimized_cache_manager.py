"""
Optimized Cache Manager for GraphRAG System
Key optimizations:
- Multi-level caching hierarchy (L1: Memory, L2: Disk)
- LRU eviction with TTL support
- Hardware-aware cache sizing
- Async cache warming and prefetching
- Cache analytics and performance monitoring
"""

import logging
import time
import threading
import pickle
import hashlib
import os
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from collections import OrderedDict
from pathlib import Path
import psutil
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def is_stale(self, max_age: float) -> bool:
        """Check if entry is stale"""
        return time.time() - self.last_access > max_age
    
    def access(self):
        """Record access to this entry"""
        self.access_count += 1
        self.last_access = time.time()

class CacheStats:
    """Cache performance statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entry_count = 0
        self.start_time = time.time()
        self.lock = threading.RLock()
    
    def record_hit(self):
        with self.lock:
            self.hits += 1
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
    
    def record_eviction(self):
        with self.lock:
            self.evictions += 1
    
    def get_hit_rate(self) -> float:
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            uptime = time.time() - self.start_time
            return {
                'hit_rate': self.get_hit_rate(),
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'entry_count': self.entry_count,
                'size_mb': self.size_bytes / (1024 * 1024),
                'uptime_seconds': uptime
            }

class OptimizedMemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 512,
                 default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Start background cleanup
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.stats.record_miss()
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats.record_miss()
                self.stats.size_bytes -= entry.size_bytes
                self.stats.entry_count -= 1
                return None
            
            # Record access and move to end (most recently used)
            entry.access()
            self.cache.move_to_end(key)
            self.stats.record_hit()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache"""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default estimate
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Ensure we have space
            while (len(self.cache) >= self.max_size or 
                   self.stats.size_bytes + size_bytes > self.max_memory_bytes):
                if not self._evict_lru():
                    # Cannot evict more
                    self.logger.warning("Cannot fit item in cache")
                    return False
            
            # Add to cache
            self.cache[key] = entry
            self.stats.size_bytes += size_bytes
            self.stats.entry_count += 1
            
            return True
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.cache:
            return False
        
        # Remove oldest item (first in OrderedDict)
        key, entry = self.cache.popitem(last=False)
        self.stats.size_bytes -= entry.size_bytes
        self.stats.entry_count -= 1
        self.stats.record_eviction()
        
        return True
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_expired():
            while True:
                try:
                    with self.lock:
                        expired_keys = []
                        for key, entry in self.cache.items():
                            if entry.is_expired():
                                expired_keys.append(key)
                        
                        for key in expired_keys:
                            entry = self.cache[key]
                            self.stats.size_bytes -= entry.size_bytes
                            self.stats.entry_count -= 1
                            del self.cache[key]
                        
                        if expired_keys:
                            self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                    
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_expired, daemon=True)
        cleanup_thread.start()
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats.size_bytes = 0
            self.stats.entry_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.get_summary()

class DiskCache:
    """Persistent disk-based cache with compression"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Load existing index
        self._load_index()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _load_index(self):
        """Load cache index from disk"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                
                # Update stats
                self.stats.entry_count = len(self.index)
                self.stats.size_bytes = sum(
                    entry.get('size_bytes', 0) for entry in self.index.values()
                )
                
                self.logger.info(f"Loaded disk cache index: {len(self.index)} entries")
        except Exception as e:
            self.logger.warning(f"Failed to load cache index: {e}")
            self.index = {}
    
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        with self.lock:
            if key not in self.index:
                self.stats.record_miss()
                return None
            
            entry_info = self.index[key]
            
            # Check expiration
            if entry_info.get('ttl') and time.time() - entry_info['timestamp'] > entry_info['ttl']:
                self._remove_entry(key)
                self.stats.record_miss()
                return None
            
            # Load from disk
            file_path = self._get_file_path(key)
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                entry_info['last_access'] = time.time()
                entry_info['access_count'] = entry_info.get('access_count', 0) + 1
                
                self.stats.record_hit()
                return value
                
            except Exception as e:
                self.logger.error(f"Failed to load from disk cache: {e}")
                self._remove_entry(key)
                self.stats.record_miss()
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in disk cache"""
        with self.lock:
            try:
                # Serialize value
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Check size limits
                if size_bytes > self.max_size_bytes:
                    self.logger.warning(f"Value too large for disk cache: {size_bytes} bytes")
                    return False
                
                # Ensure we have space
                while self.stats.size_bytes + size_bytes > self.max_size_bytes:
                    if not self._evict_lru():
                        return False
                
                # Write to disk
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    f.write(serialized)
                
                # Update index
                self.index[key] = {
                    'timestamp': time.time(),
                    'last_access': time.time(),
                    'access_count': 0,
                    'ttl': ttl,
                    'size_bytes': size_bytes,
                    'file_path': str(file_path)
                }
                
                self.stats.size_bytes += size_bytes
                self.stats.entry_count += 1
                
                self._save_index()
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save to disk cache: {e}")
                return False
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key not in self.index:
            return
        
        entry_info = self.index[key]
        file_path = Path(entry_info['file_path'])
        
        # Remove file
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to remove cache file: {e}")
        
        # Remove from index
        self.stats.size_bytes -= entry_info.get('size_bytes', 0)
        self.stats.entry_count -= 1
        del self.index[key]
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self.index:
            return False
        
        # Find LRU entry
        lru_key = min(
            self.index.keys(),
            key=lambda k: self.index[k].get('last_access', 0)
        )
        
        self._remove_entry(lru_key)
        self.stats.record_eviction()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.get_summary()

class OptimizedCacheManager:
    """Multi-level cache manager with hardware optimization"""
    
    def __init__(self,
                 memory_cache_size: int = 1000,
                 memory_cache_mb: int = 512,
                 disk_cache_mb: int = 2048,
                 cache_dir: Optional[str] = None,
                 default_ttl: Optional[float] = None):
        
        # Hardware-aware configuration
        self._configure_for_hardware()
        
        # Override with user settings
        self.memory_cache_size = memory_cache_size or self.optimal_memory_cache_size
        self.memory_cache_mb = memory_cache_mb or self.optimal_memory_cache_mb
        self.disk_cache_mb = disk_cache_mb or self.optimal_disk_cache_mb
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".graphrag_cache"
        self.cache_dir = Path(cache_dir)
        
        # Initialize caches
        self.memory_cache = OptimizedMemoryCache(
            max_size=self.memory_cache_size,
            max_memory_mb=self.memory_cache_mb,
            default_ttl=default_ttl
        )
        
        self.disk_cache = DiskCache(
            cache_dir=self.cache_dir,
            max_size_mb=self.disk_cache_mb
        )
        
        # Cache prefetching
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CachePrefetch")
        self.prefetch_queue: List[Tuple[str, Callable]] = []
        self.prefetch_lock = threading.Lock()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized cache manager - Memory: {self.memory_cache_mb}MB, Disk: {self.disk_cache_mb}MB")
    
    def _configure_for_hardware(self):
        """Configure cache sizes based on available hardware"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 16:
            # High-memory system (like Ryzen 4800H + 16GB)
            self.optimal_memory_cache_size = 2000
            self.optimal_memory_cache_mb = 1024  # 1GB
            self.optimal_disk_cache_mb = 4096    # 4GB
        elif memory_gb >= 8:
            # Medium-memory system
            self.optimal_memory_cache_size = 1000
            self.optimal_memory_cache_mb = 512   # 512MB
            self.optimal_disk_cache_mb = 2048    # 2GB
        else:
            # Low-memory system
            self.optimal_memory_cache_size = 500
            self.optimal_memory_cache_mb = 256   # 256MB
            self.optimal_disk_cache_mb = 1024    # 1GB
        
        self.logger.info(f"Hardware-optimized cache config for {memory_gb:.1f}GB RAM")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try memory cache first (L1)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache (L2)
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            disk_only: bool = False) -> bool:
        """Put value in cache with intelligent placement"""
        
        # Estimate value size
        try:
            size_estimate = len(pickle.dumps(value))
        except:
            size_estimate = 1024
        
        success = True
        
        # Always try disk cache for persistence
        if not self.disk_cache.put(key, value, ttl):
            success = False
            self.logger.warning(f"Failed to store {key} in disk cache")
        
        # Store in memory cache unless disk_only or too large
        if not disk_only and size_estimate < self.memory_cache.max_memory_bytes // 10:
            if not self.memory_cache.put(key, value, ttl):
                self.logger.warning(f"Failed to store {key} in memory cache")
        
        return success
    
    def get_or_compute(self, key: str, compute_func: Callable[[], Any], 
                      ttl: Optional[float] = None) -> Any:
        """Get value from cache or compute and store"""
        
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            return value
        
        # Compute value
        start_time = time.time()
        value = compute_func()
        compute_time = time.time() - start_time
        
        # Store in cache
        self.put(key, value, ttl)
        
        self.logger.debug(f"Computed and cached {key} in {compute_time:.3f}s")
        return value
    
    def prefetch(self, key: str, compute_func: Callable[[], Any], 
                ttl: Optional[float] = None):
        """Prefetch value asynchronously"""
        
        # Check if already cached
        if self.get(key) is not None:
            return
        
        # Add to prefetch queue
        with self.prefetch_lock:
            self.prefetch_queue.append((key, compute_func, ttl))
        
        # Submit prefetch task
        self.prefetch_executor.submit(self._execute_prefetch, key, compute_func, ttl)
    
    def _execute_prefetch(self, key: str, compute_func: Callable[[], Any], 
                         ttl: Optional[float]):
        """Execute prefetch operation"""
        try:
            # Double-check cache
            if self.get(key) is not None:
                return
            
            # Compute and cache
            value = compute_func()
            self.put(key, value, ttl)
            
            self.logger.debug(f"Prefetched {key}")
            
        except Exception as e:
            self.logger.warning(f"Prefetch failed for {key}: {e}")
    
    def warm_cache(self, key_compute_pairs: List[Tuple[str, Callable[[], Any]]]):
        """Warm cache with multiple items"""
        
        def warm_worker():
            for key, compute_func in key_compute_pairs:
                try:
                    if self.get(key) is None:
                        value = compute_func()
                        self.put(key, value)
                except Exception as e:
                    self.logger.warning(f"Cache warming failed for {key}: {e}")
        
        # Submit warming task
        self.prefetch_executor.submit(warm_worker)
        self.logger.info(f"Started cache warming for {len(key_compute_pairs)} items")
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        # Remove from memory cache
        with self.memory_cache.lock:
            if key in self.memory_cache.cache:
                entry = self.memory_cache.cache[key]
                self.memory_cache.stats.size_bytes -= entry.size_bytes
                self.memory_cache.stats.entry_count -= 1
                del self.memory_cache.cache[key]
        
        # Remove from disk cache
        with self.disk_cache.lock:
            if key in self.disk_cache.index:
                self.disk_cache._remove_entry(key)
                self.disk_cache._save_index()
    
    def clear_all(self):
        """Clear all caches"""
        self.memory_cache.clear()
        
        with self.disk_cache.lock:
            for key in list(self.disk_cache.index.keys()):
                self.disk_cache._remove_entry(key)
            self.disk_cache._save_index()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        # Calculate combined hit rate
        total_hits = memory_stats['hits'] + disk_stats['hits']
        total_requests = total_hits + memory_stats['misses'] + disk_stats['misses']
        combined_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'combined_hit_rate': combined_hit_rate,
            'memory_cache': memory_stats,
            'disk_cache': disk_stats,
            'configuration': {
                'memory_cache_size': self.memory_cache_size,
                'memory_cache_mb': self.memory_cache_mb,
                'disk_cache_mb': self.disk_cache_mb,
                'cache_dir': str(self.cache_dir)
            },
            'system_info': {
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_usage_percent': psutil.virtual_memory().percent
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.prefetch_executor.shutdown(wait=True)
        except Exception as e:
            self.logger.warning(f"Prefetch executor shutdown failed: {e}")

# Factory function
def create_optimized_cache_manager(cache_dir: Optional[str] = None) -> OptimizedCacheManager:
    """Create cache manager optimized for current hardware"""
    return OptimizedCacheManager(cache_dir=cache_dir)

# Decorators for easy caching
def cached(cache_manager: OptimizedCacheManager, ttl: Optional[float] = None):
    """Decorator to cache function results"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            return cache_manager.get_or_compute(
                key=key,
                compute_func=lambda: func(*args, **kwargs),
                ttl=ttl
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator
