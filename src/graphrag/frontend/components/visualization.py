# src/graphrag/frontend/components/visualization.py
"""Graph visualization with caching and layout optimization."""
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Tuple
import logging
from .. import config

logger = logging.getLogger(__name__)

@st.cache_data(hash_funcs={nx.Graph: lambda g: nx.weisfeiler_lehman_graph_hash(g)})
def compute_graph_layout(subgraph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    """Compute and cache graph layout positions."""
    try:
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        logger.info(f"Computed layout for {len(subgraph.nodes)} nodes.")
        return pos
    except Exception as e:
        logger.error(f"Layout computation failed: {e}")
        return {}

def render_graph_visualization(subgraph: nx.Graph, title: str = "Knowledge Graph"):
    """Render interactive graph visualization, capped for performance."""
    if not subgraph or len(subgraph.nodes) == 0:
        st.info("No entities found to visualize.")
        return

    # Performance cap
    if len(subgraph.nodes) > config.MAX_NODES_VISUALIZATION:
        st.warning(f"Displaying top {config.MAX_NODES_VISUALIZATION} nodes for performance.")
        top_nodes = sorted(subgraph.degree, key=lambda item: item[1], reverse=True)[:config.MAX_NODES_VISUALIZATION]
        subgraph = subgraph.subgraph([n for n, d in top_nodes]).copy()
    
    pos = compute_graph_layout(subgraph)
    if not pos:
        st.error("Failed to compute graph layout.")
        return

    # Create traces
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'), mode='lines')
    
    node_x, node_y, node_text, node_color = [], [], [], []
    type_colors = {'drug': '#FF6B6B', 'disease': '#4ECDC4', 'protein': '#45B7D1'}
    for node in subgraph.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = subgraph.nodes[node]
            node_text.append(f"{node_data.get('name', node)} (Type: {node_data.get('type', 'unknown')})")
            node_color.append(type_colors.get(node_data.get('type'), '#96CEB4'))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', textfont=dict(size=10),
        hovertext=node_text, hoverinfo='text',
        marker=dict(size=20, color=node_color, line_width=2)
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=title, showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    st.plotly_chart(fig, use_container_width=True)
