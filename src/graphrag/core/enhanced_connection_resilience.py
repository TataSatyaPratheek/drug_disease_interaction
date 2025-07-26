"""
Enhanced Connection Resilience for GraphRAG System
Key optimizations:
- Circuit breaker pattern for fault tolerance
- Adaptive retry mechanisms with exponential backoff
- Connection pooling and health monitoring
- Hardware-aware timeout configurations
- Intelligent failover and recovery strategies
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable, List, Union
from enum import Enum
from dataclasses import dataclass, field
import concurrent.futures
from functools import wraps
import psutil
import random

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class ConnectionMetrics:
    """Connection performance and health metrics"""
    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.success_count / self.total_requests) * 100.0
    
    def record_success(self, response_time: float):
        """Record successful request"""
        self.success_count += 1
        self.total_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        # Update rolling average response time
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            # Simple exponential moving average
            self.avg_response_time = 0.8 * self.avg_response_time + 0.2 * response_time
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.total_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception,
                 name: str = "default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.metrics = ConnectionMetrics()
        self.last_failure_time = None
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN - service unavailable")
            
            # Attempt the call
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Record success
                self.metrics.record_success(response_time)
                
                # Reset circuit if it was half-open
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
                
                return result
                
            except self.expected_exception as e:
                # Record failure
                self.metrics.record_failure()
                self.last_failure_time = time.time()
                
                # Check if we should open the circuit
                if (self.state == CircuitState.CLOSED and 
                    self.metrics.consecutive_failures >= self.failure_threshold):
                    self.state = CircuitState.OPEN
                    self.logger.warning(f"Circuit breaker {self.name} opened due to {self.metrics.consecutive_failures} consecutive failures")
                elif self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
                    self.logger.warning(f"Circuit breaker {self.name} failed half-open test")
                
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'metrics': {
                'success_rate': self.metrics.success_rate(),
                'total_requests': self.metrics.total_requests,
                'consecutive_failures': self.metrics.consecutive_failures,
                'avg_response_time': self.metrics.avg_response_time
            }
        }

class AdaptiveRetryStrategy:
    """Adaptive retry with exponential backoff and jitter"""
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry strategy to function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    self.logger.debug(f"Retrying {func.__name__} (attempt {attempt + 1}/{self.max_retries + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed for {func.__name__}: {e}")
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)

class ConnectionPool:
    """Connection pool with health monitoring"""
    
    def __init__(self, 
                 connection_factory: Callable[[], Any],
                 pool_size: int = 5,
                 max_retries: int = 3,
                 health_check_interval: float = 30.0):
        self.connection_factory = connection_factory
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        
        self.pool: List[Any] = []
        self.pool_lock = threading.RLock()
        self.health_status: Dict[int, bool] = {}
        self.last_health_check = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pool
        self._initialize_pool()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        with self.pool_lock:
            for i in range(self.pool_size):
                try:
                    conn = self.connection_factory()
                    self.pool.append(conn)
                    self.health_status[i] = True
                    self.logger.debug(f"Created connection {i}")
                except Exception as e:
                    self.logger.error(f"Failed to create connection {i}: {e}")
                    self.pool.append(None)
                    self.health_status[i] = False
    
    def get_connection(self) -> Any:
        """Get healthy connection from pool"""
        with self.pool_lock:
            # Find healthy connection
            for i, conn in enumerate(self.pool):
                if conn is not None and self.health_status.get(i, False):
                    return conn
            
            # No healthy connections, try to create one
            self.logger.warning("No healthy connections available, creating new one")
            try:
                return self.connection_factory()
            except Exception as e:
                self.logger.error(f"Failed to create emergency connection: {e}")
                raise ConnectionError("No healthy connections available")
    
    def _check_connection_health(self, conn: Any) -> bool:
        """Check if connection is healthy"""
        try:
            # Basic health check - this should be customized per connection type
            if hasattr(conn, 'ping'):
                conn.ping()
            elif hasattr(conn, 'is_ready'):
                return conn.is_ready()
            elif hasattr(conn, '_client') and hasattr(conn._client, 'is_ready'):
                return conn._client.is_ready()
            
            return True
        except Exception:
            return False
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def health_monitor():
            while True:
                try:
                    current_time = time.time()
                    if current_time - self.last_health_check > self.health_check_interval:
                        self._perform_health_check()
                        self.last_health_check = current_time
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    self.logger.error(f"Health monitor error: {e}")
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
    
    def _perform_health_check(self):
        """Perform health check on all connections"""
        with self.pool_lock:
            for i, conn in enumerate(self.pool):
                if conn is not None:
                    is_healthy = self._check_connection_health(conn)
                    was_healthy = self.health_status.get(i, False)
                    
                    if is_healthy != was_healthy:
                        status = "healthy" if is_healthy else "unhealthy"
                        self.logger.info(f"Connection {i} is now {status}")
                    
                    self.health_status[i] = is_healthy
                    
                    # Try to recreate failed connections
                    if not is_healthy:
                        try:
                            new_conn = self.connection_factory()
                            self.pool[i] = new_conn
                            self.health_status[i] = True
                            self.logger.info(f"Recreated connection {i}")
                        except Exception as e:
                            self.logger.warning(f"Failed to recreate connection {i}: {e}")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status information"""
        with self.pool_lock:
            healthy_count = sum(1 for status in self.health_status.values() if status)
            return {
                'pool_size': len(self.pool),
                'healthy_connections': healthy_count,
                'unhealthy_connections': len(self.pool) - healthy_count,
                'health_status': dict(self.health_status)
            }

class EnhancedConnectionManager:
    """Enhanced connection manager with resilience features"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategies: Dict[str, AdaptiveRetryStrategy] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Hardware-specific configurations
        self._configure_for_hardware()
    
    def _configure_for_hardware(self):
        """Configure timeouts and limits based on hardware"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Adjust configurations based on available resources
        if memory_gb >= 16 and cpu_count >= 8:
            # High-performance configuration for Ryzen 4800H + 16GB
            self.default_timeout = 30.0
            self.max_retries = 5
            self.pool_size = 8
        elif memory_gb >= 8 and cpu_count >= 4:
            # Medium configuration
            self.default_timeout = 45.0
            self.max_retries = 3
            self.pool_size = 4
        else:
            # Conservative configuration
            self.default_timeout = 60.0
            self.max_retries = 2
            self.pool_size = 2
        
        self.logger.info(f"Configured for {memory_gb:.1f}GB RAM, {cpu_count} CPUs: "
                        f"timeout={self.default_timeout}s, retries={self.max_retries}, "
                        f"pool_size={self.pool_size}")
    
    def register_circuit_breaker(self, name: str, 
                                failure_threshold: Optional[int] = None,
                                recovery_timeout: Optional[float] = None) -> CircuitBreaker:
        """Register circuit breaker for service"""
        failure_threshold = failure_threshold or 5
        recovery_timeout = recovery_timeout or self.default_timeout * 2
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name
        )
        
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Registered circuit breaker for {name}")
        
        return circuit_breaker
    
    def register_retry_strategy(self, name: str,
                              max_retries: Optional[int] = None,
                              base_delay: Optional[float] = None) -> AdaptiveRetryStrategy:
        """Register retry strategy for service"""
        max_retries = max_retries or self.max_retries
        base_delay = base_delay or 1.0
        
        retry_strategy = AdaptiveRetryStrategy(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=self.default_timeout
        )
        
        self.retry_strategies[name] = retry_strategy
        self.logger.info(f"Registered retry strategy for {name}")
        
        return retry_strategy
    
    def register_connection_pool(self, name: str,
                               connection_factory: Callable[[], Any],
                               pool_size: Optional[int] = None) -> ConnectionPool:
        """Register connection pool for service"""
        pool_size = pool_size or self.pool_size
        
        connection_pool = ConnectionPool(
            connection_factory=connection_factory,
            pool_size=pool_size,
            max_retries=self.max_retries,
            health_check_interval=30.0
        )
        
        self.connection_pools[name] = connection_pool
        self.logger.info(f"Registered connection pool for {name} with {pool_size} connections")
        
        return connection_pool
    
    def get_resilient_executor(self, service_name: str) -> Callable:
        """Get combined resilient executor for service"""
        def resilient_wrapper(func: Callable) -> Callable:
            # Apply circuit breaker if registered
            if service_name in self.circuit_breakers:
                func = self.circuit_breakers[service_name](func)
            
            # Apply retry strategy if registered
            if service_name in self.retry_strategies:
                func = self.retry_strategies[service_name](func)
            
            return func
        
        return resilient_wrapper
    
    def get_connection(self, service_name: str) -> Any:
        """Get connection from pool if available"""
        if service_name in self.connection_pools:
            return self.connection_pools[service_name].get_connection()
        else:
            raise ValueError(f"No connection pool registered for {service_name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'hardware': {
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'memory_usage': psutil.virtual_memory().percent
            },
            'configuration': {
                'default_timeout': self.default_timeout,
                'max_retries': self.max_retries,
                'pool_size': self.pool_size
            },
            'circuit_breakers': {},
            'connection_pools': {}
        }
        
        # Circuit breaker status
        for name, cb in self.circuit_breakers.items():
            status['circuit_breakers'][name] = cb.get_state()
        
        # Connection pool status
        for name, pool in self.connection_pools.items():
            status['connection_pools'][name] = pool.get_pool_status()
        
        return status

# Global connection manager instance
connection_manager = EnhancedConnectionManager()

# Convenience functions
def with_circuit_breaker(service_name: str, **kwargs):
    """Decorator to add circuit breaker protection"""
    circuit_breaker = connection_manager.register_circuit_breaker(service_name, **kwargs)
    return circuit_breaker

def with_retry(service_name: str, **kwargs):
    """Decorator to add retry logic"""
    retry_strategy = connection_manager.register_retry_strategy(service_name, **kwargs)
    return retry_strategy

def with_resilience(service_name: str):
    """Decorator to add full resilience features"""
    return connection_manager.get_resilient_executor(service_name)
