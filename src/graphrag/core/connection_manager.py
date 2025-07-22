import streamlit as st
import logging
from typing import Dict, Any, List
import weaviate
from pathlib import Path

logger = logging.getLogger(__name__)

class WeaviateConnectionManager:
    """Manages Weaviate connections with caching and reconnection logic"""
    
    def __init__(self, persist_directory: str = "data/weaviate_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._connection_params = None
    
    @st.cache_resource
    def get_client(_self) -> weaviate.WeaviateClient:
        """Get cached Weaviate client with connection management"""
        if _self._client is None:
            _self._client = _self._establish_connection()
        
        # Verify connection is still active
        if not _self._is_client_healthy():
            logger.info("Client unhealthy, reconnecting...")
            _self._client = _self._establish_connection()
        
        return _self._client
    
    def _establish_connection(self) -> weaviate.WeaviateClient:
        """Establish connection with Docker Weaviate priority"""
        import time
        
        connection_attempts = [
            (8080, 50051, "docker_weaviate"),      # Docker first
            (8081, 50052, "existing_embedded"),    
            (8082, 50053, "new_embedded"),         
            (8083, 50054, "fallback_embedded")     
        ]
        
        for port, grpc_port, description in connection_attempts:
            try:
                # For Docker Weaviate, try multiple times with delays
                max_retries = 5 if "docker" in description else 2
                
                for attempt in range(max_retries):
                    try:
                        client = weaviate.connect_to_local(port=port, grpc_port=grpc_port)
                        client.is_ready()
                        
                        logger.info(f"✅ Connected to {description} on {port}/{grpc_port}")
                        self._connection_params = (port, grpc_port, description)
                        return client
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.debug(f"Retry {attempt + 1}/{max_retries} for {description}: {e}")
                            time.sleep(2)  # Wait before retry
                        else:
                            logger.debug(f"Failed all retries for {description}: {e}")
                            break
                            
            except Exception as e:
                logger.debug(f"Failed to connect to {description}: {e}")
                
                # Only try embedded for non-Docker attempts
                if "docker" not in description and ("new_embedded" in description or "fallback_embedded" in description):
                    try:
                        client = weaviate.connect_to_embedded(
                            port=port,
                            grpc_port=grpc_port,
                            persistence_data_path=str(self.persist_directory),
                            binary_path=str(self.persist_directory / "weaviate"),
                            version="1.25.0"
                        )
                        
                        # Wait for embedded readiness
                        for i in range(15):
                            try:
                                if client.is_ready():
                                    logger.info(f"✅ Started {description} on {port}/{grpc_port}")
                                    self._connection_params = (port, grpc_port, description)
                                    return client
                            except:
                                pass
                            time.sleep(1)
                        
                        try:
                            client.close()
                        except:
                            pass
                            
                    except Exception as e2:
                        logger.warning(f"Failed to start embedded on {port}/{grpc_port}: {e2}")
                        continue
        
        raise Exception("Failed to establish Weaviate connection on any available port")


    def _is_client_healthy(self) -> bool:
        """Check if current client connection is healthy"""
        try:
            if self._client is None:
                return False
            
            return self._client.is_ready()
            
        except Exception as e:
            logger.debug(f"Client health check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about current connection"""
        if self._connection_params:
            port, grpc_port, description = self._connection_params
            return {
                "status": "connected",
                "port": port,
                "grpc_port": grpc_port,
                "description": description,
                "healthy": self._is_client_healthy()
            }
        else:
            return {
                "status": "disconnected",
                "healthy": False
            }
    
    def close_connection(self):
        """Close current connection"""
        try:
            if self._client:
                self._client.close()
                logger.info("Weaviate connection closed")
        except Exception as e:
            logger.warning(f"Error closing Weaviate connection: {e}")
        finally:
            self._client = None
            self._connection_params = None

@st.cache_resource
def get_weaviate_manager() -> WeaviateConnectionManager:
    """Get cached Weaviate connection manager"""
    return WeaviateConnectionManager()

def verify_collections(client: weaviate.WeaviateClient, required_collections: List[str]) -> Dict[str, bool]:
    """Verify that required collections exist"""
    collection_status = {}
    
    for collection_name in required_collections:
        try:
            exists = client.collections.exists(collection_name)
            collection_status[collection_name] = exists
            
            if not exists:
                logger.warning(f"Collection '{collection_name}' does not exist")
            
        except Exception as e:
            logger.error(f"Error checking collection '{collection_name}': {e}")
            collection_status[collection_name] = False
    
    return collection_status

def get_collection_stats(client: weaviate.WeaviateClient, collection_name: str) -> Dict[str, Any]:
    """Get statistics for a specific collection"""
    try:
        if not client.collections.exists(collection_name):
            return {"exists": False, "count": 0}
        
        collection = client.collections.get(collection_name)
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count or 0
        
        return {
            "exists": True,
            "count": count,
            "name": collection_name
        }
        
    except Exception as e:
        logger.error(f"Error getting stats for collection '{collection_name}': {e}")
        return {"exists": False, "count": 0, "error": str(e)}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_all_collection_stats(client: weaviate.WeaviateClient) -> Dict[str, Any]:
    """Get statistics for all collections"""
    required_collections = ["Drug", "Disease", "Protein", "Relationship"]
    
    stats = {}
    total_count = 0
    
    for collection_name in required_collections:
        collection_stats = get_collection_stats(client, collection_name)
        stats[collection_name] = collection_stats
        total_count += collection_stats.get("count", 0)
    
    stats["total_entities"] = total_count
    stats["collections_healthy"] = all(
        stats[name].get("exists", False) for name in required_collections
    )
    
    return stats
