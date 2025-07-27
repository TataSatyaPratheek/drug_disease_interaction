# src/core/weaviate_service.py - USE WEAVIATE PYTHON CLIENT
import weaviate
from typing import List, Dict, Any
import asyncio
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class WeaviateService:
    def __init__(self, url):
        url = str(url)
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        self.client = weaviate.connect_to_local(
            host=host,
            port=port
        )
        try:
            if not self.client.is_ready():
                raise ConnectionError("Weaviate is not ready.")
            logger.info("Successfully connected to Weaviate.")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate at {url}: {e}")
            raise
    
    async def get_db_stats(self) -> Dict[str, Any]:
        stats = {}
        for col_name in ['Drug', 'Disease', 'Target', 'Pathway']:
            try:
                if self.client.collections.exists(col_name):
                    collection = self.client.collections.get(col_name)
                    stats[col_name] = {"count": len(collection)}
            except Exception:
                stats[col_name] = "error"
        return {"total_objects_by_collection": stats}
    """Weaviate service using official client - DON'T REINVENT VECTOR SEARCH"""
    
    async def hybrid_search(self, query: str, collections: List[str] = None, limit: int = 20) -> List[Dict]:
        """Async hybrid search across your populated collections"""
        if collections is None:
            collections = ['Drug', 'Disease', 'Target', 'Pathway']
        
        def _sync_search():
            results = []
            for collection_name in collections:
                try:
                    collection = self.client.collections.get(collection_name)
                    response = collection.query.hybrid(
                        query=query,
                        limit=limit // len(collections)  # Distribute across collections
                    )
                    
                    for obj in response.objects:
                        results.append({
                            'id': obj.properties.get('neo4j_id', str(obj.uuid)),
                            'name': obj.properties.get('name', ''),
                            'description': obj.properties.get('description', ''),
                            'score': obj.metadata.score if obj.metadata else 0.0,
                            'collection': collection_name,
                            'uuid': str(obj.uuid)
                        })
                except Exception as e:
                    logger.warning(f"Error searching {collection_name}: {e}")
            
            return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
        
        return await asyncio.get_event_loop().run_in_executor(None, _sync_search)
    
    def close(self):
        self.client.close()
