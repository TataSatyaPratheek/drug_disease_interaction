# src/graphrag/frontend/streamlit_app.py
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
import pickle
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.graphrag.core.query_engine import GraphRAGQueryEngine
from src.graphrag.core.vector_store import GraphVectorStore
from src.graphrag.generators.llm_client import OllamaClient
from src.ddi.visualization.graph_viz import GraphVisualizer

st.set_page_config(
    page_title="Local Drug-Disease GraphRAG",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_graph_and_engine():
    """Load the knowledge graph and initialize local GraphRAG engine"""
    
    # Load your graph
    graph_path = project_root / "data/graph/full_mapped/ddi_knowledge_graph.pickle"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    
    # Initialize local LLM client
    llm_client = OllamaClient(model_name="llama3.1")  # You can change this model
    
    # Initialize vector store
    vector_store = GraphVectorStore()
    
    with st.spinner("🔄 Building vector index (first time only)..."):
        vector_store.initialize_from_graph(graph)
    
    # Initialize GraphRAG engine
    engine = GraphRAGQueryEngine(graph, llm_client, vector_store)
    
    return graph, engine, vector_store

def main():
    st.title("🧬 Local Drug-Disease Knowledge Graph RAG")
    st.markdown("Ask questions about drug-disease interactions using **local AI models** (no API keys required!)")
    
    # Check Ollama status
    try:
        from ollama import Client
        client = Client()
        models = client.list()
        st.success(f"✅ Ollama connected - {len(models['models'])} models available")
    except Exception as e:
        st.error(f"❌ Ollama not running: {e}")
        st.markdown("Please start Ollama: `ollama serve`")
        return
    
    # Load resources
    try:
        graph, engine, vector_store = load_graph_and_engine()
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return
    
    # Sidebar with graph statistics
    with st.sidebar:
        st.header("📊 Knowledge Graph Stats")
        st.metric("Total Nodes", f"{graph.number_of_nodes():,}")
        st.metric("Total Edges", f"{graph.number_of_edges():,}")
        
        # Vector store stats
        vector_stats = vector_store.get_statistics()
        st.metric("Vector Store Items", f"{vector_stats.get('total_items', 0):,}")
        
        # Node type breakdown
        node_types = {}
        for _, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        st.subheader("Node Types")
        for node_type, count in node_types.items():
            st.metric(node_type.title(), f"{count:,}")
        
        # Model selection
        st.subheader("🤖 AI Model")
        try:
            client = Client()
            available_models = [m['name'] for m in client.list()['models']]
            selected_model = st.selectbox("Choose Model:", available_models)
            
            if selected_model != engine.llm_client.model_name:
                engine.llm_client.model_name = selected_model
                st.success(f"Switched to {selected_model}")
        except:
            st.warning("Could not fetch available models")
    
    # Main query interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Example queries
        example_queries = [
            "What drugs could be repurposed for Alzheimer's disease?",
            "How does aspirin work for cardiovascular disease?",
            "Compare the mechanisms of metformin and insulin",
            "What proteins are targeted by cancer drugs?",
            "Find drugs similar to imatinib",
            "Test the hypothesis that EGFR inhibitors work for lung cancer",
            "What biomarkers predict response to immunotherapy?"
        ]
        
        selected_example = st.selectbox("Try an example query:", [""] + example_queries)
        
        user_query = st.text_area(
            "Your question:",
            value=selected_example,
            height=100,
            placeholder="Ask about drug mechanisms, repurposing opportunities, hypothesis testing, or drug comparisons..."
        )
        
        query_button = st.button("🔍 Search Knowledge Graph", type="primary")
    
    with col2:
        st.header("⚙️ Query Options")
        query_type = st.selectbox(
            "Query Type:",
            ["auto", "drug_repurposing", "mechanism_explanation", "drug_comparison", "target_discovery", "hypothesis_testing", "general"]
        )
        
        max_results = st.slider("Max Results", 5, 50, 10)
        include_visualization = st.checkbox("Include Network Visualization", True)
        temperature = st.slider("AI Creativity", 0.0, 1.0, 0.1, help="Lower = more factual, Higher = more creative")
    
    # Process query
    if query_button and user_query.strip():
        with st.spinner("🧠 Thinking... Retrieving from local knowledge graph and generating response..."):
            try:
                # Update temperature
                engine.llm_client.temperature = temperature
                
                result = engine.query(user_query, query_type, max_results=max_results)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Display response
                    st.header("🤖 Local AI Response")
                    st.markdown(result["response"])
                    
                    # Display confidence and citations
                    col1, col2 = st.columns(2)
                    with col1:
                        if "confidence_score" in result:
                            st.metric("Confidence", f"{result['confidence_score']:.2f}")
                    with col2:
                        if "citations" in result:
                            st.metric("Sources", len(result['citations']))
                    
                    # Display retrieved data
                    st.header("📊 Retrieved Knowledge")
                    
                    # Show related entities
                    if "related_entities" in result and result['related_entities']:
                        st.subheader("🎯 Related Entities")
                        entities_df = pd.DataFrame(result['related_entities'])
                        st.dataframe(entities_df, use_container_width=True)
                    
                    # Suggested follow-ups
                    if "suggested_followups" in result:
                        st.subheader("💡 Suggested Follow-up Questions")
                        for i, followup in enumerate(result['suggested_followups'][:3]):
                            if st.button(followup, key=f"followup_{i}"):
                                st.rerun()
                    
                    with st.expander("View Raw Retrieved Data", expanded=False):
                        st.json(result["retrieved_data"])
                    
                    # Visualization
                    if include_visualization and "paths" in result.get("retrieved_data", {}):
                        st.header("🕸️ Network Visualization")
                        visualize_paths(result["retrieved_data"]["paths"], graph)
                    
                    # Context used
                    with st.expander("View Graph Context Sent to AI", expanded=False):
                        st.text(result["subgraph_context"])
            
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                with st.expander("Debug Info"):
                    st.exception(e)

def visualize_paths(paths, graph):
    """Visualize drug-disease paths using plotly"""
    if not paths:
        st.info("No paths found to visualize")
        return
    
    # Create a subgraph from all paths
    subgraph_nodes = set()
    for path_data in paths[:3]:  # Top 3 paths
        if 'path' in path_data:
            subgraph_nodes.update(path_data['path'])
    
    if not subgraph_nodes:
        st.info("No valid paths to visualize")
        return
    
    subgraph = graph.subgraph(subgraph_nodes)
    
    # Create plotly visualization
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    # Color map for node types
    color_map = {
        'drug': '#FF6B6B',
        'protein': '#4ECDC4', 
        'disease': '#45B7D1',
        'unknown': '#96CEB4'
    }
    
    # Nodes
    node_colors = []
    node_texts = []
    node_hovers = []
    
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get('type', 'unknown')
        node_name = node_data.get('name', node)[:20]
        
        node_colors.append(color_map.get(node_type, '#96CEB4'))
        node_texts.append(node_name)
        node_hovers.append(f"<b>{node_name}</b><br>Type: {node_type}<br>ID: {node}")
    
    node_trace = go.Scatter(
        x=[pos[node][0] for node in subgraph.nodes()],
        y=[pos[node][1] for node in subgraph.nodes()],
        mode='markers+text',
        text=node_texts,
        textposition="middle center",
        marker=dict(
            size=25,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=node_hovers
    )
    
    # Edges
    edge_trace = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='gray'),
            hoverinfo='none'
        ))
    
    # Create figure
    fig = go.Figure(data=[node_trace] + edge_trace)
    fig.update_layout(
        title="Drug-Disease Pathway Network (Local GraphRAG)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Hover over nodes for details",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
