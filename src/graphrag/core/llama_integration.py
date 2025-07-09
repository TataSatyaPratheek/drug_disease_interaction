"""LlamaIndex integration for GraphRAG with NetworkX backend."""

import logging
from typing import Dict, List, Any, Optional, Union
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response import Response
from llama_index.core.base.llms.base import BaseLLM
import networkx as nx

from ..retrievers.path_retriever import PathRetriever
from ..retrievers.community_retriever import CommunityRetriever
from ..retrievers.subgraph_retriever import SubgraphRetriever
from .vector_store import WeaviateGraphStore

logger = logging.getLogger(__name__)

class NetworkXGraphStore:
    """LlamaIndex-compatible wrapper around NetworkX graph."""
    
    def __init__(self, nx_graph: nx.MultiDiGraph):
        self.graph = nx_graph
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_rel_map(self, subjs: Optional[List[str]] = None, depth: int = 2) -> Dict[str, List[List[str]]]:
        """Get relationship map for given subjects."""
        if subjs is None:
            subjs = list(self.graph.nodes())[:100]  # Limit for performance
        
        rel_map = {}
        for subj in subjs:
            if subj not in self.graph:
                continue
                
            paths = []
            visited = set()
            
            def dfs_paths(node, path, current_depth):
                if current_depth >= depth or node in visited:
                    return
                
                visited.add(node)
                
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if edge_data:
                        edge_type = list(edge_data.values())[0].get('type', 'connected_to')
                        new_path = path + [edge_type, neighbor]
                        paths.append(new_path)
                        dfs_paths(neighbor, new_path, current_depth + 1)
            
            dfs_paths(subj, [subj], 0)
            rel_map[subj] = paths
        
        return rel_map

class GraphRAGRetriever(BaseRetriever):
    """Custom retriever that combines vector search with graph traversal."""
    
    def __init__(
        self,
        vector_store: WeaviateGraphStore,
        graph_store: NetworkXGraphStore,
        path_retriever: PathRetriever,
        community_retriever: CommunityRetriever,
        subgraph_retriever: SubgraphRetriever,
        top_k: int = 10
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.path_retriever = path_retriever
        self.community_retriever = community_retriever
        self.subgraph_retriever = subgraph_retriever
        self.top_k = top_k
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using hybrid vector + graph approach."""
        query_str = query_bundle.query_str
        
        # 1. Vector search for seed entities
        vector_results = self.vector_store.search_entities(
            query_str, 
            entity_types=["Drug", "Disease", "Protein", "Polypeptide"],
            n_results=5 # Get a smaller, high-confidence set of seed nodes
        )
        
        # 2. Graph expansion
        nodes_with_scores = []
        seed_entities = [res for res in vector_results if res.get('id')]
        seed_ids = [e['id'] for e in seed_entities]
        
        for entity in seed_entities:
            node_content = self._create_node_content(entity)
            node = NodeWithScore(
                node=node_content,
                score=entity.get('similarity', 0.0)
            )
            nodes_with_scores.append(node)

        # 3. Path Finding Context (if multiple entities are found)
        if len(seed_ids) >= 2:
            # Attempt to find paths between the top 2 seed entities
            paths = self.path_retriever.find_drug_disease_paths(seed_ids[0], seed_ids[1], max_paths=2)
            if paths:
                path_context = self._create_path_content(paths)
                path_node = NodeWithScore(node=path_context, score=0.85) # High score for direct paths
                nodes_with_scores.append(path_node)
        
        # 4. Neighborhood Context
        if seed_ids:
            neighborhood_subgraph = self.subgraph_retriever.get_entity_subgraph(seed_ids, hops=1)
            if neighborhood_subgraph.number_of_nodes() > len(seed_ids):
                neighborhood_context = self._create_subgraph_content(neighborhood_subgraph)
                neighborhood_node = NodeWithScore(node=neighborhood_context, score=0.7)
                nodes_with_scores.append(neighborhood_node)
        
        # 5. Community context
        try:
            if seed_ids:
                communities = self.community_retriever.get_communities(
                    seed_ids
                )
                
                if communities:
                    community_content = self._create_community_content(communities)
                    community_node = NodeWithScore(
                        node=community_content,
                        score=0.6
                    )
                    nodes_with_scores.append(community_node)
        except Exception as e:
            logger.warning(f"Community retrieval failed: {e}")
        
        # Sort by score and return all enriched context nodes
        nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
        return nodes_with_scores
    def _create_path_content(self, paths: List[Dict]) -> Any:
        """Create a LlamaIndex node from retrieved paths."""
        from llama_index.core.schema import TextNode

        path_text = "Found Explanatory Paths:\n"
        for i, path_data in enumerate(paths):
            path_str = " -> ".join(path_data.get('path_names', []))
            path_text += f"  Path {i+1}: {path_str}\n"

        return TextNode(
            text=path_text,
            metadata={'content_type': 'path_context'}
        )

    def _create_subgraph_content(self, subgraph: nx.MultiDiGraph) -> Any:
        """Create a LlamaIndex node from a neighborhood subgraph."""
        from llama_index.core.schema import TextNode

        subgraph_text = (
            f"Neighborhood Context: A subgraph of {subgraph.number_of_nodes()} entities and "
            f"{subgraph.number_of_edges()} relationships was found around the core entities."
        )
        return TextNode(
            text=subgraph_text, metadata={'content_type': 'subgraph_context'}
        )
    
    def _create_node_content(self, entity: Dict[str, Any]) -> Any:
        """Create LlamaIndex node content from entity data."""
        from llama_index.core.schema import TextNode
        
        name = entity.get('name', entity.get('id', 'Unknown'))
        entity_type = entity.get('type', 'unknown')
        description = entity.get('description', '')
        
        text_content = f"{name} ({entity_type})"
        if description:
            text_content += f": {description}"
        
        return TextNode(
            text=text_content,
            metadata={
                'entity_id': entity.get('id'),
                'entity_type': entity_type,
                'name': name
            }
        )
    
    def _create_community_content(self, communities: List[Dict]) -> Any:
        """Create community context node."""
        from llama_index.core.schema import TextNode
        
        community_text = "Related communities:\n"
        for community in communities[:3]:
            community_text += f"- Community {community.get('id', 'Unknown')}: "
            community_text += f"{community.get('size', 0)} entities\n"
        
        return TextNode(
            text=community_text,
            metadata={'content_type': 'community_context'}
        )

class LlamaGraphRAGEngine:
    """Main LlamaIndex-powered GraphRAG engine."""
    
    def __init__(
        self,
        nx_graph: nx.MultiDiGraph,
        vector_store: WeaviateGraphStore,
        llm_client,
        path_retriever: PathRetriever,
        community_retriever: CommunityRetriever,
        subgraph_retriever: SubgraphRetriever
    ):
        self.nx_graph = nx_graph
        self.vector_store = vector_store
        self.llm_client = llm_client
        
        # Create graph store
        self.graph_store = NetworkXGraphStore(nx_graph)
        
        # Create custom retriever
        self.retriever = GraphRAGRetriever(
            vector_store=vector_store,
            graph_store=self.graph_store,
            path_retriever=path_retriever,
            community_retriever=community_retriever,
            subgraph_retriever=subgraph_retriever
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def query(self, query_str: str, **kwargs) -> Dict[str, Any]:
        """Process query using LlamaIndex + custom graph logic."""
        try:
            # Use custom retriever
            query_bundle = QueryBundle(query_str=query_str)
            retrieved_nodes = self.retriever.retrieve(query_bundle)
            
            # Build context from retrieved nodes
            context_parts = []
            entity_data = {'drugs': [], 'diseases': [], 'proteins': []}
            
            for node_with_score in retrieved_nodes:
                node = node_with_score.node
                
                # Add to context
                context_parts.append(f"[Score: {node_with_score.score:.3f}] {node.text}")
                
                # Extract entity data for response
                metadata = node.metadata
                entity_id = metadata.get('entity_id')
                entity_type = metadata.get('entity_type')
                
                if entity_id and entity_type:
                    entity_info = {
                        'id': entity_id,
                        'name': metadata.get('name', entity_id),
                        'type': entity_type,
                        'similarity': node_with_score.score
                    }
                    
                    if entity_type == 'drug':
                        entity_data['drugs'].append(entity_info)
                    elif entity_type == 'disease':
                        entity_data['diseases'].append(entity_info)
                    elif entity_type == 'protein':
                        entity_data['proteins'].append(entity_info)
            
            # Generate response using existing LLM client
            context = "\n".join(context_parts)
            reasoning, final_answer = self.llm_client.generate_with_reasoning(
                f"Based on the following graph context, answer the query:\n\n"
                f"Context:\n{context}\n\n"
                f"Query: {query_str}\n\n"
                f"Provide a comprehensive answer with clear reasoning."
            )
            
            # Build response in expected format
            response = {
                'response': final_answer,
                'reasoning': reasoning,
                'retrieved_data': entity_data,
                'citations': self._build_citations(retrieved_nodes),
                'suggested_followups': self._generate_followups(query_str, entity_data),
                'query_type': 'llama_graphrag',
                'confidence_score': 0.8
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"LlamaIndex query failed: {e}")
            raise
    
    def _build_citations(self, retrieved_nodes: List[NodeWithScore]) -> List[Dict]:
        """Build citation list from retrieved nodes."""
        citations = []
        
        for i, node_with_score in enumerate(retrieved_nodes, 1):
            node = node_with_score.node
            metadata = node.metadata
            
            citation = {
                'id': i,
                'name': metadata.get('name', 'Unknown'),
                'type': metadata.get('entity_type', 'unknown'),
                'entity_id': metadata.get('entity_id', ''),
                'score': node_with_score.score
            }
            citations.append(citation)
        
        return citations
    
    def _generate_followups(self, query: str, entity_data: Dict) -> List[str]:
        """Generate follow-up questions based on retrieved entities."""
        followups = []
        
        if entity_data.get('drugs'):
            followups.append("What are the mechanisms of action for these drugs?")
        
        if entity_data.get('diseases'):
            followups.append("What other treatments are available for these conditions?")
        
        if entity_data.get('proteins'):
            followups.append("What other drugs target these proteins?")
        
        if len(entity_data.get('drugs', [])) > 1:
            followups.append("How do these drugs compare in terms of efficacy and safety?")
        
        return followups[:3]
