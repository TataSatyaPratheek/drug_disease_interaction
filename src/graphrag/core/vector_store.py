# src/graphrag/core/vector_store.py
import chromadb
from chromadb.config import Settings
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from ..generators.llm_client import EmbeddingClient

class GraphVectorStore:
    """Vector store for efficient graph entity and relationship retrieval"""
    
    def __init__(self, 
                 persist_directory: str = "data/vectorstore",
                 collection_name: str = "drug_disease_graph"):
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = collection_name
        self.collection = None
        self.embedding_client = EmbeddingClient()
        self.logger = logging.getLogger(__name__)
    
    def initialize_from_graph(self, graph: nx.MultiDiGraph, force_rebuild: bool = False):
        """Initialize vector store from NetworkX graph"""
        
        # Check if collection exists
        try:
            if force_rebuild:
                self.client.delete_collection(self.collection_name)
            
            self.collection = self.client.get_collection(self.collection_name)
            self.logger.info(f"Loaded existing collection with {self.collection.count()} items")
            return
            
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Drug-disease knowledge graph entities"}
            )
        
        self.logger.info("Building vector store from graph...")
        self._index_graph_entities(graph)
        self._index_graph_relationships(graph)
        self.logger.info(f"Vector store built with {self.collection.count()} items")
    
    def _index_graph_entities(self, graph: nx.MultiDiGraph):
        """Index all graph entities with their descriptions"""
        
        documents = []
        metadatas = []
        ids = []
        
        for node_id, data in graph.nodes(data=True):
            # Create searchable text description
            text_parts = []
            
            name = data.get('name', node_id)
            node_type = data.get('type', 'unknown')
            text_parts.append(f"{name} ({node_type})")
            
            # Add specific attributes based on node type
            if node_type == 'drug':
                if data.get('description'):
                    text_parts.append(data['description'][:500])
                if data.get('indication'):
                    text_parts.append(f"Indication: {data['indication']}")
                if data.get('mechanism_of_action'):
                    text_parts.append(f"Mechanism: {data['mechanism_of_action']}")
                if data.get('synonyms'):
                    text_parts.append(f"Also known as: {', '.join(data['synonyms'][:5])}")
            
            elif node_type == 'disease':
                if data.get('annotation'):
                    text_parts.append(data['annotation'])
                if data.get('tree_numbers'):
                    text_parts.append(f"Classification: {', '.join(data['tree_numbers'][:3])}")
            
            elif node_type in ['protein', 'polypeptide']:
                if data.get('gene_name'):
                    text_parts.append(f"Gene: {data['gene_name']}")
                if data.get('function'):
                    text_parts.append(f"Function: {data['function']}")
            
            document = " | ".join(text_parts)
            documents.append(document)
            
            metadata = {
                "node_id": node_id,
                "name": name,
                "type": node_type,
                "degree": graph.degree(node_id)
            }
            metadatas.append(metadata)
            ids.append(f"node_{node_id}")
        
        # Add to collection in batches
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
    
    def _index_graph_relationships(self, graph: nx.MultiDiGraph):
        """Index important graph relationships"""
        
        documents = []
        metadatas = []
        ids = []
        
        relationship_count = 0
        
        for source, target, key, data in graph.edges(data=True, keys=True):
            edge_type = data.get('type', 'unknown')
            
            # Create relationship description
            source_name = graph.nodes[source].get('name', source)
            target_name = graph.nodes[target].get('name', target)
            source_type = graph.nodes[source].get('type', 'unknown')
            target_type = graph.nodes[target].get('type', 'unknown')
            
            if edge_type == 'targets':
                doc = f"{source_name} (drug) targets {target_name} (protein)"
            elif edge_type == 'associated_with':
                doc = f"{target_name} (disease) is associated with {source_name} (protein)"
            else:
                doc = f"{source_name} ({source_type}) {edge_type} {target_name} ({target_type})"
            
            # Add edge attributes if available
            if data.get('actions'):
                doc += f" | Actions: {data['actions']}"
            if data.get('score'):
                doc += f" | Score: {data['score']}"
            
            documents.append(doc)
            
            metadata = {
                "source_id": source,
                "target_id": target,
                "source_name": source_name,
                "target_name": target_name,
                "source_type": source_type,
                "target_type": target_type,
                "edge_type": edge_type,
                "edge_key": str(key)
            }
            metadatas.append(metadata)
            ids.append(f"edge_{source}_{target}_{key}")
            
            relationship_count += 1
            
            # Limit relationships to avoid memory issues
            if relationship_count >= 50000:
                break
        
        # Add relationships in batches
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
    
    def search_entities(self, 
                       query: str, 
                       entity_types: List[str] = None,
                       n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for entities using vector similarity"""
        
        if not self.collection:
            return []
        
        where_clause = {}
        if entity_types:
            where_clause["type"] = {"$in": entity_types}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            entities = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                if metadata.get('node_id'):  # It's an entity, not a relationship
                    entities.append({
                        'id': metadata['node_id'],
                        'name': metadata['name'],
                        'type': metadata['type'],
                        'description': doc,
                        'similarity_score': 1 - distance,
                        'degree': metadata.get('degree', 0)
                    })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error searching entities: {e}")
            return []
    
    def search_relationships(self, 
                           query: str,
                           relationship_types: List[str] = None,
                           n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relationships using vector similarity"""
        
        if not self.collection:
            return []
        
        where_clause = {}
        if relationship_types:
            where_clause["edge_type"] = {"$in": relationship_types}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            relationships = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                if metadata.get('source_id'):  # It's a relationship
                    relationships.append({
                        'source_id': metadata['source_id'],
                        'target_id': metadata['target_id'],
                        'source_name': metadata['source_name'],
                        'target_name': metadata['target_name'],
                        'edge_type': metadata['edge_type'],
                        'description': doc,
                        'similarity_score': 1 - distance
                    })
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error searching relationships: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.collection:
            return {}
        
        return {
            "total_items": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory)
        }
