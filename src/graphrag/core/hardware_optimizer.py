# src/graphrag/core/hardware_optimizer.py
"""
Hardware-specific optimization configuration for GraphRAG
Automatically detects and optimizes settings for different hardware configurations
"""

import os
import psutil
import subprocess
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class HardwareOptimizer:
    """Optimize GraphRAG settings for specific hardware configuration"""
    
    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_info = self._detect_gpu()
        
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                return {
                    "name": gpu_data[0],
                    "memory_mb": int(gpu_data[1]),
                    "available": True
                }
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
        
        return {"name": "None", "memory_mb": 0, "available": False}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get optimized LLM client configuration"""
        # Base configuration optimized for multi-core CPUs
        config = {
            "num_threads": min(8, max(2, self.cpu_count // 2)),
            "temperature": 0.1,
            "max_tokens": 2048,
            "context_size": 4096 if self.memory_gb >= 16 else 2048,
        }
        
        # GPU-specific settings
        if self.gpu_info["available"]:
            gpu_memory_gb = self.gpu_info["memory_mb"] / 1024
            
            if gpu_memory_gb >= 8:
                # High-end GPU
                config.update({
                    "gpu_layers": 40,
                    "batch_size": 12,
                    "use_gpu": True
                })
            elif gpu_memory_gb >= 4:
                # Mid-range GPU (like GTX 1650Ti)
                config.update({
                    "gpu_layers": 32,
                    "batch_size": 8,
                    "use_gpu": True
                })
            else:
                # Low-end GPU
                config.update({
                    "gpu_layers": 24,
                    "batch_size": 4,
                    "use_gpu": True
                })
        else:
            # CPU-only
            config.update({
                "gpu_layers": 0,
                "batch_size": 4,
                "use_gpu": False
            })
        
        return config
    
    def get_threading_config(self) -> Dict[str, Any]:
        """Get optimized threading configuration"""
        # Conservative threading for stability
        max_workers = max(2, min(6, self.cpu_count // 2))
        
        return {
            "max_workers": max_workers,
            "response_builder_workers": max_workers,
            "retriever_workers": max_workers,
            "llm_workers": min(2, max_workers),  # Keep LLM workers lower to avoid conflicts
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get optimized cache configuration based on available RAM"""
        if self.memory_gb >= 32:
            return {
                "response_cache_size": 512,
                "query_cache_size": 1024,
                "graph_cache_size": 256,
                "enable_persistent_cache": True
            }
        elif self.memory_gb >= 16:
            return {
                "response_cache_size": 256,
                "query_cache_size": 512,
                "graph_cache_size": 128,
                "enable_persistent_cache": True
            }
        elif self.memory_gb >= 8:
            return {
                "response_cache_size": 128,
                "query_cache_size": 256,
                "graph_cache_size": 64,
                "enable_persistent_cache": True
            }
        else:
            return {
                "response_cache_size": 64,
                "query_cache_size": 128,
                "graph_cache_size": 32,
                "enable_persistent_cache": False
            }
    
    def get_model_recommendations(self) -> Dict[str, Any]:
        """Get model recommendations based on hardware"""
        gpu_memory_gb = self.gpu_info["memory_mb"] / 1024 if self.gpu_info["available"] else 0
        
        if gpu_memory_gb >= 8 and self.memory_gb >= 16:
            # High-end setup
            return {
                "primary_model": "llama3.2:3b",
                "secondary_model": "qwen3:1.7b",
                "embedding_model": "nomic-embed-text",
                "recommended_models": [
                    "llama3.2:3b", "qwen3:1.7b", "phi3:mini", "gemma2:2b"
                ],
                "avoid_models": ["llama3:8b", "mixtral:8x7b", "llama3:70b"]
            }
        elif gpu_memory_gb >= 4 and self.memory_gb >= 12:
            # Mid-range setup (like GTX 1650Ti + 16GB RAM)
            return {
                "primary_model": "qwen3:1.7b",
                "secondary_model": "phi3:mini",
                "embedding_model": "nomic-embed-text",
                "recommended_models": [
                    "qwen3:1.7b", "phi3:mini", "llama3.2:1b", "gemma2:2b"
                ],
                "avoid_models": ["llama3:8b", "mixtral:8x7b", "llama3:70b"]
            }
        else:
            # Lower-end or CPU-only setup
            return {
                "primary_model": "phi3:mini",
                "secondary_model": "llama3.2:1b",
                "embedding_model": "nomic-embed-text",
                "recommended_models": ["phi3:mini", "llama3.2:1b"],
                "avoid_models": ["llama3:8b", "mixtral:8x7b", "llama3:70b", "qwen3:7b"]
            }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a comprehensive optimization summary"""
        return {
            "hardware_info": {
                "cpu_cores": self.cpu_count,
                "memory_gb": round(self.memory_gb, 1),
                "gpu": self.gpu_info
            },
            "llm_config": self.get_llm_config(),
            "threading_config": self.get_threading_config(),
            "cache_config": self.get_cache_config(),
            "model_recommendations": self.get_model_recommendations(),
            "optimization_tier": self._get_optimization_tier()
        }
    
    def _get_optimization_tier(self) -> str:
        """Determine the optimization tier based on hardware"""
        gpu_memory_gb = self.gpu_info["memory_mb"] / 1024 if self.gpu_info["available"] else 0
        
        if gpu_memory_gb >= 8 and self.memory_gb >= 24:
            return "high_performance"
        elif gpu_memory_gb >= 4 and self.memory_gb >= 12:
            return "balanced"
        elif self.memory_gb >= 8:
            return "efficient"
        else:
            return "minimal"
    
    def apply_optimizations(self) -> Dict[str, Any]:
        """Apply optimizations and return configuration"""
        config = self.get_optimization_summary()
        
        # Set environment variables for Ollama
        llm_config = config["llm_config"]
        if llm_config["use_gpu"]:
            os.environ["OLLAMA_GPU_LAYERS"] = str(llm_config["gpu_layers"])
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Set threading environment variables
        threading_config = config["threading_config"]
        os.environ["GRAPHRAG_MAX_WORKERS"] = str(threading_config["max_workers"])
        
        logger.info(f"Applied {config['optimization_tier']} optimizations")
        logger.info(f"CPU: {self.cpu_count} cores, RAM: {self.memory_gb:.1f}GB, GPU: {self.gpu_info['name']}")
        
        return config


def get_hardware_optimizer() -> HardwareOptimizer:
    """Get a singleton hardware optimizer instance"""
    if not hasattr(get_hardware_optimizer, '_instance'):
        get_hardware_optimizer._instance = HardwareOptimizer()
    return get_hardware_optimizer._instance


def auto_optimize() -> Dict[str, Any]:
    """Automatically optimize GraphRAG for current hardware"""
    optimizer = get_hardware_optimizer()
    return optimizer.apply_optimizations()


def get_recommended_models() -> list[str]:
    """Get list of recommended models for current hardware"""
    optimizer = get_hardware_optimizer()
    recommendations = optimizer.get_model_recommendations()
    return recommendations["recommended_models"]


def get_optimal_workers() -> int:
    """Get optimal number of workers for current hardware"""
    optimizer = get_hardware_optimizer()
    threading_config = optimizer.get_threading_config()
    return threading_config["max_workers"]
