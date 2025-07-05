# src/graphrag/frontend/graphrag_app.py
import streamlit as st
import pickle
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
print(f"Adding project root to sys.path: {project_root}")
sys.path.append(str(project_root))

from src.graphrag.core.query_engine import GraphRAGQueryEngine
from src.graphrag.core.vector_store import WeaviateGraphStore
from src.graphrag.generators.llm_client import OllamaClient

st.set_page_config(
    page_title="Drug-Disease GraphRAG",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def initialize_graphrag_system():
    """Initialize the complete GraphRAG system and ensure cleanup."""
    
    # Initialize Weaviate vector store
    vector_store = WeaviateGraphStore()

    try:
        # Load graph
        graph_path = project_root / "data/graph/full_mapped/ddi_knowledge_graph.pickle"
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        # After initializing vector_store
        stats = vector_store.get_statistics()
        if stats['total_entities'] == 0:
            raise ValueError("Weaviate database is empty! Run migration script.")

        # Initialize Ollama client
        llm_client = OllamaClient(model_name="qwen3:1.7b")

        # Initialize GraphRAG engine
        engine = GraphRAGQueryEngine(graph, llm_client, vector_store)

        yield graph, vector_store, llm_client, engine

    finally:
        st.info("Shutting down Weaviate connection...")
        vector_store.close()

def display_reasoning_and_response(result):
    """Display qwen3's reasoning process and final answer"""
    
    # Show reasoning process
    if "reasoning" in result and result["reasoning"]:
        st.header("🧠 AI Reasoning Process")
        with st.expander("👀 See How Qwen3 Thinks Through This Problem", expanded=True):
            st.markdown(result["reasoning"])
    
    # Show final answer
    st.header("🎯 Final Answer")
    st.markdown(result["response"])


def main():
    st.title("🧬 Drug-Disease GraphRAG System")
    st.markdown("**Powered by Weaviate Vector Database + Local Ollama + NetworkX Graph Analysis**")
    
    # Check Ollama status with correct Model object handling
    try:
        import ollama
        client = ollama.Client()
        models_response = client.list()
        
        # Extract model names from Ollama Model objects
        available_models = []
        if hasattr(models_response, 'models'):
            models_list = models_response.models
        elif isinstance(models_response, dict) and 'models' in models_response:
            models_list = models_response['models']
        else:
            models_list = models_response
        
        for model in models_list:
            if hasattr(model, 'model'):
                # Ollama Model object with .model attribute
                available_models.append(model.model)
            elif isinstance(model, dict):
                model_name = model.get('name') or model.get('model') or str(model)
                available_models.append(model_name)
            else:
                available_models.append(str(model))
        
        if available_models:
            st.success(f"✅ Ollama running with {len(available_models)} models: {', '.join(available_models[:3])}")
        else:
            st.warning("⚠️ Ollama is running but no models found.")
            
    except ConnectionError:
        st.error("❌ Cannot connect to Ollama. Is it running?")
        st.info("Start Ollama: `ollama serve`")
        return
    except Exception as e:
        st.error(f"❌ Ollama error: {e}")
        st.info("Try: `ollama serve` in another terminal")
        return
    
    # Initialize system
    try:
        with st.spinner("🚀 Initializing GraphRAG system..."):
            graph, vector_store, llm_client, engine = initialize_graphrag_system()
        
        st.success(f"✅ System ready! Graph: {graph.number_of_nodes():,} nodes, Vector DB: {vector_store.get_statistics()['total_entities']:,} entities")
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        st.exception(e)  # Show full traceback for debugging
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        selected_model = st.selectbox("Ollama Model:", available_models)
        if selected_model != engine.llm_client.model_name:
            engine.llm_client.model_name = selected_model
            st.success(f"Switched to {selected_model}")
        
        # Query parameters
        max_results = st.slider("Max Vector Results", 5, 50, 15)
        temperature = st.slider("AI Temperature", 0.0, 1.0, 0.1)
        engine.llm_client.temperature = temperature
        
        # System stats
        st.subheader("📊 System Stats")
        stats = vector_store.get_statistics()
        for key, value in stats.items():
            if isinstance(value, int):
                st.metric(key.replace('_', ' ').title(), f"{value:,}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Pre-defined example queries
        example_queries = [
            "What drugs could be repurposed for Alzheimer's disease?",
            "How does aspirin work for cardiovascular disease?",
            "Test the hypothesis that EGFR inhibitors are effective for lung cancer",
            "What are the mechanisms of metformin in diabetes?",
            "Find drugs similar to imatinib for cancer treatment",
            "What proteins are involved in Parkinson's disease pathways?",
            "Which biomarkers predict response to immunotherapy?",
            "Compare the side effects of ACE inhibitors vs ARBs"
        ]
        
        selected_example = st.selectbox("Try an example:", [""] + example_queries)
        
        user_query = st.text_area(
            "Your research question:",
            value=selected_example,
            height=120,
            placeholder="Ask about drug repurposing, mechanisms, hypothesis testing, or drug discovery..."
        )
        
        # Query type selection
        query_types = ["auto", "drug_repurposing", "mechanism_explanation", "hypothesis_testing", "drug_comparison", "target_discovery", "general"]
        query_type = st.selectbox("Query Type:", query_types)
        
        # Advanced options
        with st.expander("🔬 Advanced Options"):
            include_visualization = st.checkbox("Include Network Visualization", True)
            include_confidence = st.checkbox("Show Confidence Scores", True)
            include_citations = st.checkbox("Show Citations", True)
        
        search_button = st.button("🔍 Search Knowledge Graph", type="primary", use_container_width=True)
    
    with col2:
        st.header("🎯 Quick Actions")
        
        if st.button("🔬 Run Hypothesis H1", use_container_width=True):
            st.info("Running H1: Indication Distance Analysis...")
            # You could integrate your existing hypothesis testing here
        
        if st.button("📊 Run Hypothesis H3", use_container_width=True):
            st.info("Running H3: Trial Enrichment Analysis...")
        
        if st.button("🎲 Random Drug Discovery", use_container_width=True):
            random_query = "Find potential new indications for a random FDA-approved drug"
            st.session_state.random_query = random_query
    
    # Process query
    if search_button and user_query.strip():
        process_graphrag_query(user_query, query_type, max_results, engine, include_visualization, include_confidence, include_citations, graph)

def process_graphrag_query(query, query_type, max_results, engine, include_viz, include_conf, include_cit, graph):
    """Process GraphRAG query and display results"""
    
    with st.spinner("🧠 GraphRAG Processing: Vector search → Graph traversal → AI generation..."):
        try:
            # Execute GraphRAG query
            result = engine.query(query, query_type, max_results)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                return
            
            # Display reasoning and response
            display_reasoning_and_response(result)

            
            # Display confidence and metadata
            if include_conf:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "confidence_score" in result:
                        st.metric("Confidence", f"{result['confidence_score']:.2f}")
                with col2:
                    st.metric("Query Type", result.get('query_type', 'unknown').title())
                with col3:
                    entity_count = sum(len(entities) for entities in result.get('retrieved_data', {}).values())
                    st.metric("Entities Found", entity_count)
            
            # Display retrieved entities
            st.header("📊 Retrieved Knowledge")
            
            entities_data = result.get('retrieved_data', {})
            if entities_data:
                for entity_type, entities in entities_data.items():
                    if entities:
                        st.subheader(f"🎯 {entity_type.title()}")
                        entities_df = pd.DataFrame(entities)
                        st.dataframe(entities_df, use_container_width=True)
            
            # Display citations
            if include_cit and "citations" in result:
                st.subheader("📚 Citations")
                for citation in result['citations']:
                    st.write(f"- {citation.get('name', 'Unknown')} ({citation.get('type', 'Unknown')})")
            
            # Show suggested follow-ups
            if "suggested_followups" in result:
                st.subheader("💡 Suggested Follow-up Questions")
                for i, followup in enumerate(result['suggested_followups'][:3]):
                    if st.button(followup, key=f"followup_{i}"):
                        st.rerun()
            
            # Network visualization
            if include_viz and entities_data:
                st.header("🕸️ Knowledge Graph Visualization")
                visualize_graphrag_results(entities_data, graph)
            
            # Raw context (expandable)
            with st.expander("🔍 View Raw Graph Context", expanded=False):
                st.text(result.get("subgraph_context", "No context available"))
        
        except Exception as e:
            st.error(f"❌ GraphRAG error: {str(e)}")
            with st.expander("🐛 Debug Info"):
                st.exception(e)

def visualize_graphrag_results(entities_data, graph):
    """Create interactive visualization of GraphRAG results"""
    
    # Collect all entity IDs
    entity_ids = set()
    for entity_list in entities_data.values():
        for entity in entity_list:
            entity_ids.add(entity['id'])
    
    if not entity_ids:
        st.info("No entities to visualize")
        return
    
    # Create subgraph
    subgraph = graph.subgraph(entity_ids)
    
    if subgraph.number_of_nodes() == 0:
        st.info("No connected subgraph found")
        return
    
    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50)
    
    # Color mapping
    color_map = {
        'drug': '#FF6B6B',      # Red
        'disease': '#4ECDC4',   # Teal  
        'protein': '#45B7D1',   # Blue
        'unknown': '#96CEB4'    # Green
    }
    
    # Create traces
    node_trace = create_node_trace(subgraph, pos, color_map)
    edge_traces = create_edge_traces(subgraph, pos)
    
    # Create figure
    fig = go.Figure(data=[node_trace] + edge_traces)
    fig.update_layout(
        title="GraphRAG Results Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[dict(
            text="GraphRAG: Vector search + Graph traversal results",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_node_trace(subgraph, pos, color_map):
    """Create Plotly node trace"""
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = subgraph.nodes[node]
        name = node_data.get('name', node)[:25]
        node_type = node_data.get('type', 'unknown')
        degree = subgraph.degree(node)
        
        node_text.append(f"{name}<br>Type: {node_type}<br>Connections: {degree}")
        node_color.append(color_map.get(node_type, '#96CEB4'))
        node_size.append(min(30, 10 + degree * 2))
    
    return go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[t.split('<br>')[0] for t in node_text],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )

def create_edge_traces(subgraph, pos):
    """Create Plotly edge traces"""
    edge_traces = []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none'
        ))
    
    return edge_traces

if __name__ == "__main__":
    main()
