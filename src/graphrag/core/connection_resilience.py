"""Robust connection management with retry mechanisms."""

import streamlit as st
import logging
import time
from typing import Callable, Any, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import psutil
import functools

logger = logging.getLogger(__name__)

class ConnectionResilience:
    """Manages robust connections with retry logic and health monitoring."""
    
    def __init__(self):
        self.health_metrics = {}
        self.connection_attempts = {}
    
    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        wait_seconds: int = 2,
        backoff_multiplier: float = 2.0,
        exception_types: tuple = (Exception,)
    ):
        """Decorator for adding retry logic to functions."""
        def decorator(func):
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=backoff_multiplier, min=wait_seconds),
                retry=retry_if_exception_type(exception_types)
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    logger.debug(f"Function {func.__name__} succeeded")
                    return result
                except Exception as e:
                    logger.warning(f"Function {func.__name__} failed: {e}")
                    raise
            return wrapper
        return decorator
    
    @staticmethod
    @with_retry(max_attempts=3, wait_seconds=1)
    def robust_weaviate_call(client, operation: str, *args, **kwargs):
        """Make robust Weaviate calls with retry logic."""
        try:
            if not client.is_ready():
                raise ConnectionError("Weaviate client not ready")
            
            if operation == 'search':
                collection_name = kwargs.get('collection_name')
                collection = client.collections.get(collection_name)
                return collection.query.bm25(*args, **kwargs)
            elif operation == 'get':
                collection_name = kwargs.get('collection_name')
                collection = client.collections.get(collection_name)
                return collection.query.get(*args, **kwargs)
            elif operation == 'aggregate':
                collection_name = kwargs.get('collection_name')
                collection = client.collections.get(collection_name)
                return collection.aggregate.over_all(*args, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Weaviate {operation} operation failed: {e}")
            raise
    
    @staticmethod
    @with_retry(max_attempts=5, wait_seconds=2)
    def robust_ollama_call(client, prompt: str, **kwargs):
        """Make robust Ollama calls with retry logic."""
        try:
            response = client.generate_with_reasoning(prompt)
            if not response or not response[1]:  # Check if final_answer exists
                raise RuntimeError("Empty response from Ollama")
            return response
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    @staticmethod
    @st.cache_data(ttl=60)
    def get_system_health() -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (basic)
            network = psutil.net_io_counters()
            
            health_data = {
                'timestamp': time.time(),
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 90 else 'critical'
                },
                'cpu': {
                    'percent': cpu_percent,
                    'status': 'healthy' if cpu_percent < 70 else 'warning' if cpu_percent < 85 else 'critical'
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100,
                    'status': 'healthy' if (disk.used / disk.total) < 0.8 else 'warning' if (disk.used / disk.total) < 0.9 else 'critical'
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
            # Overall health status
            statuses = [health_data['memory']['status'], health_data['cpu']['status'], health_data['disk']['status']]
            if 'critical' in statuses:
                health_data['overall_status'] = 'critical'
            elif 'warning' in statuses:
                health_data['overall_status'] = 'warning'
            else:
                health_data['overall_status'] = 'healthy'
            
            return health_data
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                'timestamp': time.time(),
                'overall_status': 'unknown',
                'error': str(e)
            }
    
    @staticmethod
    def adaptive_cache_ttl() -> int:
        """Calculate adaptive cache TTL based on system resources."""
        try:
            health = ConnectionResilience.get_system_health()
            memory_percent = health['memory']['percent']
            
            if memory_percent > 85:
                return 300  # 5 minutes - aggressive cleanup
            elif memory_percent > 70:
                return 600  # 10 minutes
            elif memory_percent > 50:
                return 1800  # 30 minutes
            else:
                return 3600  # 1 hour - plenty of memory
                
        except Exception as e:
            logger.warning(f"Failed to calculate adaptive TTL: {e}")
            return 1800  # Default to 30 minutes
    
    def monitor_connection_health(self, connection_name: str, health_check_func: Callable) -> bool:
        """Monitor connection health with historical tracking."""
        try:
            is_healthy = health_check_func()
            
            if connection_name not in self.health_metrics:
                self.health_metrics[connection_name] = {
                    'success_count': 0,
                    'failure_count': 0,
                    'last_success': None,
                    'last_failure': None,
                    'availability': 1.0
                }
            
            metrics = self.health_metrics[connection_name]
            
            if is_healthy:
                metrics['success_count'] += 1
                metrics['last_success'] = time.time()
            else:
                metrics['failure_count'] += 1
                metrics['last_failure'] = time.time()
            
            # Calculate availability
            total_checks = metrics['success_count'] + metrics['failure_count']
            metrics['availability'] = metrics['success_count'] / total_checks if total_checks > 0 else 0.0
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health monitoring failed for {connection_name}: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        return {
            'health_metrics': self.health_metrics,
            'connection_attempts': self.connection_attempts,
            'system_health': self.get_system_health()
        }
