# src/graphrag/frontend/simple_app.py
import streamlit as st
import pickle
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

st.set_page_config(
    page_title="Drug-Disease GraphRAG",
    page_icon="🧬",
    layout="wide"
)

def main():
    st.title("🧬 Drug-Disease Knowledge Graph Explorer")
    st.markdown("**Simple version - No LLM dependencies required**")
    
    # Try to load the graph
    try:
        graph_path = project_root / "data/graph/full_mapped/ddi_knowledge_graph.pickle"
        
        if not graph_path.exists():
            st.error(f"Graph file not found at {graph_path}")
            st.info("Make sure your graph file exists at the correct path")
            return
        
        with st.spinner("Loading knowledge graph..."):
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)
        
        st.success(f"✅ Loaded graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", f"{graph.number_of_nodes():,}")
        with col2:
            st.metric("Total Edges", f"{graph.number_of_edges():,}")
        with col3:
            st.metric("Avg Degree", f"{sum(dict(graph.degree()).values()) / graph.number_of_nodes():.1f}")
        
        # Node type breakdown
        st.subheader("📊 Node Types")
        node_types = {}
        for _, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Display as DataFrame
        type_df = pd.DataFrame([
            {"Type": node_type.title(), "Count": count, "Percentage": f"{100*count/graph.number_of_nodes():.1f}%"}
            for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(type_df, use_container_width=True)
        
        # Simple entity search
        st.subheader("🔍 Entity Search")
        search_term = st.text_input("Search for drugs, diseases, or proteins:", placeholder="e.g., aspirin, cancer, EGFR")
        
        if search_term:
            with st.spinner("Searching..."):
                results = []
                search_lower = search_term.lower()
                
                for node_id, data in graph.nodes(data=True):
                    # FIX: Handle None values properly
                    name = (data.get('name') or '').lower()  # This prevents the AttributeError
                    if search_lower in name:
                        results.append({
                            'ID': node_id,
                            'Name': data.get('name') or node_id,  # Also fix display name
                            'Type': data.get('type', 'unknown'),
                            'Degree': graph.degree(node_id)
                        })
                
                if results:
                    st.write(f"Found {len(results)} matches:")
                    results_df = pd.DataFrame(results[:20])  # Limit to 20 results
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Show details for first result
                    if results:
                        selected_entity = st.selectbox("Select entity for details:", 
                                                     [r['Name'] for r in results[:10]])
                        
                        if selected_entity:
                            entity_id = next(r['ID'] for r in results if r['Name'] == selected_entity)
                            show_entity_details(graph, entity_id)
                else:
                    st.info("No matches found")
        
        # Random entity explorer
        st.subheader("🎲 Random Entity Explorer")
        if st.button("Show Random Drug"):
            drugs = [n for n, d in graph.nodes(data=True) if d.get('type') == 'drug']
            if drugs:
                import random
                random_drug = random.choice(drugs)
                show_entity_details(graph, random_drug)
        
        if st.button("Show Random Disease"):
            diseases = [n for n, d in graph.nodes(data=True) if d.get('type') == 'disease']
            if diseases:
                import random
                random_disease = random.choice(diseases)
                show_entity_details(graph, random_disease)
    
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        st.exception(e)

def show_entity_details(graph, entity_id):
    """Show detailed information about an entity"""
    if entity_id not in graph:
        st.error("Entity not found")
        return
    
    data = graph.nodes[entity_id]
    
    # FIX: Handle None names in display too
    entity_name = data.get('name') or entity_id
    st.subheader(f"📝 Details: {entity_name}")
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Type:** {data.get('type', 'unknown')}")
        st.write(f"**Degree:** {graph.degree(entity_id)}")
    with col2:
        st.write(f"**In-degree:** {graph.in_degree(entity_id)}")
        st.write(f"**Out-degree:** {graph.out_degree(entity_id)}")
    
    # Additional attributes
    if len(data) > 2:  # More than just name and type
        st.write("**Additional Attributes:**")
        for key, value in data.items():
            if key not in ['name', 'type'] and value:
                if isinstance(value, str) and len(value) > 100:
                    st.write(f"**{key}:** {value[:100]}...")
                else:
                    st.write(f"**{key}:** {value}")
    
    # Connections
    st.write("**Connected Entities:**")
    connections = []
    
    # Outgoing edges
    for _, target, edge_data in graph.out_edges(entity_id, data=True):
        # FIX: Handle None names in connections too
        target_name = graph.nodes[target].get('name') or target
        target_type = graph.nodes[target].get('type', 'unknown')
        edge_type = edge_data.get('type', 'unknown')
        connections.append({
            'Direction': 'Outgoing',
            'Edge Type': edge_type,
            'Connected To': target_name,
            'Target Type': target_type
        })
    
    # Incoming edges
    for source, _, edge_data in graph.in_edges(entity_id, data=True):
        # FIX: Handle None names in connections too
        source_name = graph.nodes[source].get('name') or source
        source_type = graph.nodes[source].get('type', 'unknown')
        edge_type = edge_data.get('type', 'unknown')
        connections.append({
            'Direction': 'Incoming',
            'Edge Type': edge_type,
            'Connected To': source_name,
            'Target Type': source_type
        })
    
    if connections:
        conn_df = pd.DataFrame(connections[:20])  # Limit to 20 connections
        st.dataframe(conn_df, use_container_width=True)
    else:
        st.info("No connections found")

if __name__ == "__main__":
    main()
