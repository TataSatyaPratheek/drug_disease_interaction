import logging
import networkx as nx
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.data import DataObject
    from weaviate.classes.query import MetadataQuery, Filter
    WEAVIATE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Weaviate not available: {e}")
    WEAVIATE_AVAILABLE = False

from .connection_manager import WeaviateConnectionManager

class WeaviateGraphStore:
    """Graph store interface for Weaviate v4 vector database."""
    
    def __init__(self, client):
        """Initialize with Weaviate v4 client."""
        self.client = client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Test connection
        try:
            if not self.client.is_ready():
                raise ConnectionError("Weaviate client is not ready")
            self.logger.info("✅ WeaviateGraphStore initialized with v4 client")
        except Exception as e:
            self.logger.error(f"Failed to initialize WeaviateGraphStore: {e}")
            raise
    
    def close(self):
        """Close Weaviate v4 client connection."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            self.logger.info("Weaviate v4 client connection closed")
        except Exception as e:
            self.logger.warning(f"Error closing Weaviate v4 client: {e}")
    
    def __del__(self):
        """Ensure connection cleanup."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
                self.logger.info("Weaviate client connection closed")
        except Exception as e:
            self.logger.warning(f"Error closing Weaviate client: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'connection_manager'):
            self.connection_manager.close_connection()
    
    def search_relationships(self, query: str, relationship_types: List[str] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relationships using vector similarity"""
        if "Relationship" not in self.collections:
            return []
        
        try:
            collection = self.collections["Relationship"]
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Build where filter if relationship types specified
            where_filter = None
            if relationship_types:
                where_filter = Filter.by_property("edge_type").contains_any(relationship_types)
            
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=n_results,
                where=where_filter,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["source_id", "target_id", "source_name", "target_name", "edge_type"]
            )
            
            relationships = []
            for obj in response.objects:
                relationships.append({
                    'source_id': obj.properties['source_id'],
                    'target_id': obj.properties['target_id'],
                    'source_name': obj.properties['source_name'],
                    'target_name': obj.properties['target_name'],
                    'edge_type': obj.properties['edge_type'],
                    'similarity_score': 1 - obj.metadata.distance
                })
            
            return relationships
        
        except Exception as e:
            self.logger.error(f"Error searching relationships: {e}")
            return []
    
    def initialize_from_graph(self, graph: nx.MultiDiGraph, force_rebuild: bool = False):
        """Initialize Weaviate store from NetworkX graph"""
        try:
            # Define collections using CORRECT v4 API
            collection_configs = {
                "Drug": {
                    "name": "Drug",
                    "properties": [
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="description", data_type=DataType.TEXT),
                        Property(name="indication", data_type=DataType.TEXT),
                        Property(name="mechanism_of_action", data_type=DataType.TEXT),
                        Property(name="synonyms", data_type=DataType.TEXT_ARRAY),
                        Property(name="degree", data_type=DataType.INT),
                        Property(name="drugbank_id", data_type=DataType.TEXT),
                    ],
                    "vectorizer_config": Configure.Vectorizer.none()
                },
                "Disease": {
                    "name": "Disease",
                    "properties": [
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="annotation", data_type=DataType.TEXT),
                        Property(name="tree_numbers", data_type=DataType.TEXT_ARRAY),
                        Property(name="degree", data_type=DataType.INT),
                        Property(name="mesh_id", data_type=DataType.TEXT),
                    ],
                    "vectorizer_config": Configure.Vectorizer.none()
                },
                "Protein": {
                    "name": "Protein",
                    "properties": [
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="gene_name", data_type=DataType.TEXT),
                        Property(name="function", data_type=DataType.TEXT),
                        Property(name="degree", data_type=DataType.INT),
                        Property(name="uniprot_id", data_type=DataType.TEXT),
                    ],
                    "vectorizer_config": Configure.Vectorizer.none()
                },
                "Relationship": {
                    "name": "Relationship",
                    "properties": [
                        Property(name="source_id", data_type=DataType.TEXT),
                        Property(name="target_id", data_type=DataType.TEXT),
                        Property(name="source_name", data_type=DataType.TEXT),
                        Property(name="target_name", data_type=DataType.TEXT),
                        Property(name="source_type", data_type=DataType.TEXT),
                        Property(name="target_type", data_type=DataType.TEXT),
                        Property(name="edge_type", data_type=DataType.TEXT),
                        Property(name="actions", data_type=DataType.TEXT),
                        Property(name="score", data_type=DataType.NUMBER),
                    ],
                    "vectorizer_config": Configure.Vectorizer.none()
                }
            }
            
            # Create or get collections using CORRECT v4 API
            for collection_name, config in collection_configs.items():
                if force_rebuild and self.client.collections.exists(collection_name):
                    self.client.collections.delete(collection_name)
                    self.logger.info(f"Deleted existing collection: {collection_name}")
                
                if not self.client.collections.exists(collection_name):
                    # CORRECT v4 collection creation
                    self.client.collections.create(
                        name=config["name"],
                        properties=config["properties"],
                        vectorizer_config=config["vectorizer_config"]
                    )
                    self.logger.info(f"Created collection: {collection_name}")
                
                # Get collection reference
                self.collections[collection_name] = self.client.collections.get(collection_name)
            
            # Check if data already exists
            if not force_rebuild:
                try:
                    drug_collection = self.collections["Drug"]
                    response = drug_collection.aggregate.over_all(total_count=True)
                    drug_count = response.total_count
                    if drug_count > 0:
                        self.logger.info(f"Found existing data: {drug_count} drugs. Use force_rebuild=True to recreate.")
                        return
                except:
                    pass
            
            self.logger.info("Building Weaviate store from graph...")
            self._index_graph_entities(graph)
            self._index_graph_relationships(graph)
            
            # Print final statistics
            stats = self.get_statistics()
            self.logger.info(f"Weaviate store built: {stats}")
        
        except Exception as e:
            self.logger.error(f"Error initializing Weaviate: {e}")
            raise
    
    def _index_graph_entities(self, graph: nx.MultiDiGraph):
        """Index all graph entities with embeddings"""
        # Group nodes by type for batch processing
        nodes_by_type = {"Drug": [], "Disease": [], "Protein": []}
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            # Classify node type
            if node_type == 'drug':
                nodes_by_type["Drug"].append((node_id, data))
            elif node_type == 'disease':
                nodes_by_type["Disease"].append((node_id, data))
            elif node_type in ['protein', 'polypeptide']:
                nodes_by_type["Protein"].append((node_id, data))
        
        # Process each type
        for collection_name, nodes in nodes_by_type.items():
            if not nodes:
                continue
            self.logger.info(f"Indexing {len(nodes)} {collection_name.lower()} entities...")
            self._batch_index_entities(nodes, collection_name, graph)
    
    def _batch_index_entities(self, nodes: List[Tuple], collection_name: str, graph: nx.MultiDiGraph):
        """Batch index entities of a specific type"""
        collection = self.collections[collection_name]
        batch_size = 100
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            # Prepare batch data using CORRECT v4 API
            objects = []
            for node_id, data in batch:
                # Create searchable text
                text_parts = self._create_entity_text(data, collection_name)
                full_text = " | ".join(text_parts)
                # Generate embedding
                embedding = self.embedding_model.encode(full_text).tolist()
                # Prepare data object
                data_object = self._prepare_entity_data(node_id, data, graph, collection_name)
                # CORRECT v4 DataObject creation
                objects.append(DataObject(
                    properties=data_object,
                    vector=embedding
                ))
            # Insert batch using CORRECT v4 API
            try:
                result = collection.data.insert_many(objects)
                if result.has_errors:
                    for error in result.errors:
                        self.logger.error(f"Batch insert error: {error}")
            except Exception as e:
                self.logger.error(f"Error inserting batch: {e}")
    
    def _create_entity_text(self, data: Dict, collection_name: str) -> List[str]:
        """Create searchable text representation of entity"""
        text_parts = []
        name = data.get('name') or ''
        text_parts.append(f"{name} ({collection_name.lower()})")
        if collection_name == "Drug":
            if data.get('description'):
                text_parts.append(data['description'][:300])
            if data.get('indication'):
                text_parts.append(f"Indication: {data['indication'][:200]}")
            if data.get('mechanism_of_action'):
                text_parts.append(f"Mechanism: {data['mechanism_of_action'][:200]}")
            if data.get('synonyms'):
                text_parts.append(f"Synonyms: {', '.join(data['synonyms'][:5])}")
        elif collection_name == "Disease":
            if data.get('annotation'):
                text_parts.append(data['annotation'][:300])
            if data.get('tree_numbers'):
                text_parts.append(f"MeSH: {', '.join(data['tree_numbers'][:3])}")
        elif collection_name == "Protein":
            if data.get('gene_name'):
                text_parts.append(f"Gene: {data['gene_name']}")
            if data.get('function'):
                text_parts.append(f"Function: {data['function'][:200]}")
        return text_parts
    
    def _prepare_entity_data(self, node_id: str, data: Dict, graph: nx.MultiDiGraph, collection_name: str) -> Dict:
        """Prepare entity data for Weaviate storage"""
        base_data = {
            "node_id": node_id,
            "name": data.get('name') or node_id,
            "degree": graph.degree(node_id)
        }
        
        if collection_name == "Drug":
            base_data.update({
                "description": data.get('description', ''),
                "indication": data.get('indication', ''),
                "mechanism_of_action": data.get('mechanism_of_action', ''),
                "synonyms": data.get('synonyms', []),
                "drugbank_id": data.get('drugbank_id', '')
            })
        elif collection_name == "Disease":
            base_data.update({
                "annotation": data.get('annotation', ''),
                "tree_numbers": data.get('tree_numbers', []),
                "mesh_id": data.get('mesh_id', '')
            })
        elif collection_name == "Protein":
            base_data.update({
                "gene_name": data.get('gene_name', ''),
                "function": data.get('function', ''),
                "uniprot_id": data.get('uniprot_id', '')
            })
        return base_data
    
    def _index_graph_relationships(self, graph: nx.MultiDiGraph):
        """Index graph relationships"""
        self.logger.info("Indexing relationships...")
        collection = self.collections["Relationship"]
        relationships = []
        for source, target, key, data in graph.edges(data=True, keys=True):
            edge_type = data.get('type', 'unknown')
            # Create relationship description
            source_name = graph.nodes[source].get('name') or source
            target_name = graph.nodes[target].get('name') or target
            source_type = graph.nodes[source].get('type', 'unknown')
            target_type = graph.nodes[target].get('type', 'unknown')
            relationship_text = f"{source_name} ({source_type}) {edge_type} {target_name} ({target_type})"
            if data.get('actions'):
                relationship_text += f" | Actions: {data['actions']}"
            relationships.append({
                'text': relationship_text,
                'data': {
                    "source_id": source,
                    "target_id": target,
                    "source_name": source_name,
                    "target_name": target_name,
                    "source_type": source_type,
                    "target_type": target_type,
                    "edge_type": edge_type,
                    "actions": data.get('actions', ''),
                    "score": float(data.get('score', 0.0))
                }
            })
            # Limit relationships to avoid memory issues
            if len(relationships) >= 50000:
                break
        
        # Batch index relationships
        batch_size = 100
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            objects = []
            for rel in batch:
                embedding = self.embedding_model.encode(rel['text']).tolist()
                objects.append(DataObject(
                    properties=rel['data'],
                    vector=embedding
                ))
            try:
                result = collection.data.insert_many(objects)
                if result.has_errors:
                    for error in result.errors:
                        self.logger.error(f"Relationship batch error: {error}")
            except Exception as e:
                self.logger.error(f"Error inserting relationship batch: {e}")
    
    def search_entities(self, query: str, entity_types: List[str] = None, n_results: int = 10) -> List[Dict]:
        """Search entities using correct schema properties."""
        try:
            results = []
            # Use only classes that exist in your schema
            search_classes = entity_types or ["Drug", "Disease", "Protein"]
            
            for class_name in search_classes:
                try:
                    # Get collection
                    collection = self.client.collections.get(class_name)
                    
                    # Define correct properties for each class based on actual schema
                    if class_name == "Drug":
                        properties = ["node_id", "name", "description", "indication", "mechanism_of_action"]
                    elif class_name == "Disease":
                        properties = ["node_id", "name", "annotation"]
                    elif class_name == "Protein":
                        properties = ["node_id", "name", "gene_name", "function"]
                    else:
                        properties = ["node_id", "name"]
                    
                    # Perform BM25 search with correct properties
                    response = collection.query.bm25(
                        query=query,
                        limit=n_results // len(search_classes) + 1,
                        return_properties=properties
                    )
                    
                    # Process results
                    for obj in response.objects:
                        # Build description from available properties
                        description_parts = []
                        if class_name == "Drug":
                            description_parts.extend([
                                obj.properties.get("description", ""),
                                obj.properties.get("indication", ""),
                                obj.properties.get("mechanism_of_action", "")
                            ])
                        elif class_name == "Disease":
                            description_parts.append(obj.properties.get("annotation", ""))
                        elif class_name == "Protein":
                            description_parts.extend([
                                obj.properties.get("gene_name", ""),
                                obj.properties.get("function", "")
                            ])
                        
                        # Clean and join description
                        description = " | ".join([part for part in description_parts if part])
                        
                        entity = {
                            "id": obj.properties.get("node_id", str(obj.uuid)),
                            "name": obj.properties.get("name", "Unknown"),
                            "type": class_name.lower(),
                            "description": description,
                            "similarity": 0.8
                        }
                        results.append(entity)
                        
                except Exception as e:
                    self.logger.warning(f"Search failed for {class_name}: {e}")
                    continue
            
            results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)[:n_results]
            self.logger.info(f"Found {len(results)} entities for query: '{query}'")
            return results
            
        except Exception as e:
            self.logger.error(f"Weaviate search failed: {e}")
            return []

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the vector store using direct approach"""
        try:
            collections = ["Drug", "Disease", "Protein", "Relationship"]
            stats = {"total_entities": 0}
            
            for collection_name in collections:
                try:
                    if self.client.collections.exists(collection_name):
                        collection = self.client.collections.get(collection_name)
                        response = collection.aggregate.over_all(total_count=True)
                        count = response.total_count or 0
                        
                        key = collection_name.lower() + "s"
                        stats[key] = count
                        stats["total_entities"] += count
                    else:
                        key = collection_name.lower() + "s"
                        stats[key] = 0
                except Exception as e:
                    self.logger.error(f"Error counting {collection_name}: {e}")
                    key = collection_name.lower() + "s"
                    stats[key] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {"total_entities": 0, "drugs": 0, "diseases": 0, "proteins": 0, "relationships": 0}


# Alias for backward compatibility
GraphVectorStore = WeaviateGraphStore
