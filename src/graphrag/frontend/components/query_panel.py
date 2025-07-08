# src/graphrag/frontend/components/query_panel.py
"""Renders the query input panel."""
import streamlit as st
from .. import config, state

def render_query_panel():
    """Render main query input section and return the user's query."""
    st.markdown("## 💬 Ask Your Question")

    # Handle follow-up questions first
    if 'selected_followup' in st.session_state:
        query_text = st.session_state.pop('selected_followup')
    else:
        query_text = ""

    # Example queries
    with st.expander("Try an example question:"):
        for i, example in enumerate(config.EXAMPLE_QUERIES):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                query_text = example

    # Main query input
    query = st.text_area(
        "Your research question:",
        value=query_text,
        height=100,
        placeholder="Enter your drug-disease research question here...",
        label_visibility="collapsed"
    )

    query_type = st.selectbox(
        "Query Type:",
        options=list(config.QUERY_TYPES.keys()),
        format_func=lambda x: config.QUERY_TYPES[x],
        index=0
    )

    if st.button("Process Query", type="primary", use_container_width=True, disabled=state.get_state('busy')):
        if query.strip():
            state.set_state('busy', True)
            return query.strip(), query_type
    
    return None, None
