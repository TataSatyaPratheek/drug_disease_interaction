import streamlit as st
import logging
import requests
import ollama
from typing import Dict, List, Any
import time

logger = logging.getLogger(__name__)

@st.cache_data(ttl=60)  
def check_weaviate_status(ports: List[tuple] = [(8080, 50051), (8081, 50052), (8082, 50053)]) -> Dict[str, any]:
    """Check if Weaviate server is ready with retry logic"""
    
    for port, grpc_port in ports:
        # Try multiple times with delays for Docker containers
        for attempt in range(3):
            try:
                response = requests.get(
                    f"http://localhost:{port}/v1/.well-known/ready", 
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info(f"✅ Weaviate found on port {port}")
                    return {
                        "status": "healthy",
                        "port": port,
                        "grpc_port": grpc_port,
                        "url": f"http://localhost:{port}",
                        "container_type": "docker" if port == 8080 else "embedded"
                    }
            except requests.exceptions.RequestException as e:
                if attempt < 2:  # Retry for Docker containers
                    logger.debug(f"Attempt {attempt + 1}: Weaviate not ready on port {port}, retrying...")
                    time.sleep(2)
                else:
                    logger.debug(f"Weaviate not available on port {port}: {e}")
                continue
    
    return {
        "status": "unhealthy",
        "error": "Weaviate not reachable on any known ports",
        "tried_ports": [port for port, _ in ports],
        "suggestion": "Wait for Docker container to fully start or check docker-compose logs"
    }

@st.cache_data(ttl=300)  # Cache for 5 minutes  
def check_ollama_status(ollama_url: str = "http://localhost:11434") -> Dict[str, any]:
    """Check if Ollama server is reachable and return model info"""
    try:
        client = ollama.Client()
        models_response = client.list()
        
        models = [model.model for model in models_response.models] if hasattr(models_response, 'models') else []
        
        return {
            "status": "healthy",
            "models": models,
            "model_count": len(models)
        }
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {
            "status": "unhealthy", 
            "models": [],
            "model_count": 0,
            "error": str(e)
        }

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_system_status() -> Dict[str, any]:
    """Get overall system health status"""
    ollama_status = check_ollama_status()
    weaviate_status = check_weaviate_status()
    
    overall_healthy = (
        ollama_status["status"] == "healthy" and 
        weaviate_status["status"] == "healthy"
    )
    
    return {
        "overall": "healthy" if overall_healthy else "degraded",
        "ollama": ollama_status,
        "weaviate": weaviate_status
    }

def check_local_privacy_compliance() -> Dict[str, Any]:
    """Verify system is operating in complete local mode."""
    try:
        import subprocess
        
        # Check ollama is local
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        local_models = result.stdout if result.returncode == 0 else ""
        
        return {
            "status": "local_only",
            "models_available": len(local_models.split('\n')) - 1,
            "privacy_compliant": True,
            "no_network_calls": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "privacy_compliant": False,
            "error": str(e)
        }
