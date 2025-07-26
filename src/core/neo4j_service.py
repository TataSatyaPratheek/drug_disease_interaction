# src/core/neo4j_service.py - USE NEO4J PYTHON DRIVER
import neo4j
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class Neo4jService:
    """Neo4j service using official driver - DON'T REINVENT DATABASE ACCESS"""
    
    def __init__(self, uri: str, user: str, password: str, max_workers: int = 4):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def search_drug_disease_paths(self, query: str, limit: int = 20) -> List[Dict]:
        """Async drug-disease path search optimized for your dataset"""
        def _sync_search():
            with self.driver.session() as session:
                # Optimized for your 212k nodes, 1.35M relationships
                cypher = """
                MATCH path = (d:Drug)-[*1..3]-(disease:Disease) 
                WHERE d.name CONTAINS $query OR disease.name CONTAINS $query
                WITH path, 
                     [n in nodes(path) | {id: n.id, name: n.name, type: labels(n)[0]}] as node_details,
                     [r in relationships(path) | type(r)] as rel_types
                RETURN path, node_details, rel_types
                LIMIT $limit
                """
                results = session.run(cypher, query=query, limit=limit).data()
                return results
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _sync_search)
    
    async def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific entity"""
        def _sync_get():
            with self.driver.session() as session:
                cypher = """
                MATCH (n {id: $entity_id})
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, collect({relation: type(r), connected: connected.name}) as connections
                LIMIT 1
                """
                result = session.run(cypher, entity_id=entity_id).single()
                return result.data() if result else {}
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _sync_get)
    
    def close(self):
        self.driver.close()
        self.executor.shutdown()
