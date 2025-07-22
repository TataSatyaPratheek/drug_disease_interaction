"""Interactive graph component using streamlit-agraph."""

import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class InteractiveGraphRenderer:
    """Renders interactive graphs with click-to-expand functionality."""
    
    def __init__(self):
        self.node_colors = {
            'drug': '#FF6B6B',
            'disease': '#4ECDC4', 
            'protein': '#45B7D1',
            'unknown': '#95A5A6'
        }
        
        self.edge_colors = {
            'targets': '#E74C3C',
            'treats': '#27AE60',
            'associated_with': '#3498DB',
            'has_category': '#F39C12',
            'default': '#BDC3C7'
        }
    
    def render_interactive_graph(
        self,
        subgraph: nx.MultiDiGraph,
        title: str = "Knowledge Graph",
        height: int = 600,
        physics: bool = True,
        selected_node: Optional[str] = None
    ) -> Optional[str]:
        """Render interactive graph and return clicked node ID."""
        
        if subgraph.number_of_nodes() == 0:
            st.info("No graph data to display.")
            return None
        
        # Performance check
        if subgraph.number_of_nodes() > 100:
            st.warning(f"Graph has {subgraph.number_of_nodes()} nodes. Limiting to 100 for performance.")
            # Get top 100 nodes by degree
            top_nodes = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[:100]
            node_ids = [node for node, degree in top_nodes]
            subgraph = subgraph.subgraph(node_ids).copy()
        
        # Create nodes
        nodes = []
        for node_id, data in subgraph.nodes(data=True):
            name = data.get('name', node_id)
            node_type = data.get('type', 'unknown')
            
            # Highlight selected node
            color = self.node_colors.get(node_type, self.node_colors['unknown'])
            size = 25
            
            if selected_node and node_id == selected_node:
                color = '#FFD700'  # Gold for selected
                size = 35
            
            node = Node(
                id=node_id,
                label=name,
                size=size,
                color=color,
                title=f"{name} ({node_type})"  # Hover text
            )
            nodes.append(node)
        
        # Create edges
        edges = []
        for source, target, data in subgraph.edges(data=True):
            edge_type = data.get('type', 'default')
            color = self.edge_colors.get(edge_type, self.edge_colors['default'])
            
            edge = Edge(
                source=source,
                target=target,
                color=color,
                title=edge_type
            )
            edges.append(edge)
        
        # Configure graph
        config = Config(
            width=800,
            height=height,
            directed=True,
            physics=physics,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7DC6F",
            collapsible=False
        )
        
        # Render graph
        st.subheader(title)
        
        # Add legend
        with st.expander("Legend"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Node Types:**")
                for node_type, color in self.node_colors.items():
                    st.markdown(f"🔴 {node_type.title()}", unsafe_allow_html=True)
            
            with col2:
                st.write("**Edge Types:**")
                for edge_type, color in self.edge_colors.items():
                    if edge_type != 'default':
                        st.markdown(f"→ {edge_type.replace('_', ' ').title()}")
        
        # Render the graph
        return_value = agraph(nodes=nodes, edges=edges, config=config)
        
        # Handle click events
        clicked_node = None
        if return_value:
            clicked_node = return_value
            if clicked_node:
                st.info(f"Clicked on: {clicked_node}")
        
        return clicked_node
    
    def render_path_visualization(
        self,
        path_data: Dict,
        title: str = "Path Visualization"
    ) -> None:
        """Render a specific path as an interactive linear graph."""
        
        if not path_data or 'path' not in path_data:
            st.warning("No path data to visualize.")
            return
        
        path_nodes = path_data['path']
        path_names = path_data.get('path_names', path_nodes)
        path_types = path_data.get('path_types', ['unknown'] * len(path_nodes))
        
        # Create nodes for the path
        nodes = []
        for i, (node_id, name, node_type) in enumerate(zip(path_nodes, path_names, path_types)):
            color = self.node_colors.get(node_type, self.node_colors['unknown'])
            
            node = Node(
                id=node_id,
                label=name,
                size=30,
                color=color,
                x=i * 150,  # Linear layout
                y=0,
                fixed=True
            )
            nodes.append(node)
        
        # Create edges for the path
        edges = []
        for i in range(len(path_nodes) - 1):
            edge = Edge(
                source=path_nodes[i],
                target=path_nodes[i + 1],
                color=self.edge_colors['default']
            )
            edges.append(edge)
        
        # Configure for path display
        config = Config(
            width=800,
            height=200,
            directed=True,
            physics=False,  # Fixed positions
            hierarchical=False
        )
        
        st.subheader(title)
        agraph(nodes=nodes, edges=edges, config=config)
    
    def render_community_overview(
        self,
        communities: List[Dict],
        title: str = "Community Overview"
    ) -> None:
        """Render community structure as an interactive overview."""
        
        if not communities:
            st.warning("No community data to visualize.")
            return
        
        # Create nodes for communities
        nodes = []
        for i, community in enumerate(communities[:10]):  # Limit to 10 communities
            community_id = f"community_{i}"
            size = min(community.get('size', 1) * 2, 50)  # Scale by size
            
            node = Node(
                id=community_id,
                label=f"Community {i+1}\n({community.get('size', 0)} nodes)",
                size=size,
                color='#8E44AD',
                title=f"Community {i+1}: {community.get('size', 0)} nodes"
            )
            nodes.append(node)
        
        # No edges between communities for now
        edges = []
        
        config = Config(
            width=800,
            height=400,
            directed=False,
            physics=True,
            hierarchical=False
        )
        
        st.subheader(title)
        agraph(nodes=nodes, edges=edges, config=config)

def render_graph_tabs(
    subgraph: nx.MultiDiGraph,
    path_data: Optional[Dict] = None,
    communities: Optional[List[Dict]] = None,
    selected_node: Optional[str] = None
) -> Optional[str]:
    """Render tabbed interface for different graph views."""
    
    renderer = InteractiveGraphRenderer()
    
    tab1, tab2, tab3 = st.tabs(["Neighborhood", "Paths", "Communities"])
    
    clicked_node = None
    
    with tab1:
        clicked_node = renderer.render_interactive_graph(
            subgraph, 
            title="Entity Neighborhood",
            selected_node=selected_node
        )
        # Add click-to-expand functionality
        if clicked_node:
            st.session_state.selected_node = clicked_node
            st.session_state.expand_node = True
            st.rerun()

    with tab2:
        if path_data:
            renderer.render_path_visualization(path_data, "Key Pathways")
        else:
            st.info("💡 **Path Analysis**: Run queries with multiple entities to discover molecular pathways and drug-disease connections.")

    with tab3:
        if communities:
            renderer.render_community_overview(communities, "Community Structure")
        else:
            st.info("💡 **Community Analysis**: Functional groups and therapeutic categories will be displayed here based on your query results.")

    return clicked_node
