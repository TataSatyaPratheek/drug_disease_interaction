import streamlit as st
from graphrag.frontend import config, state
import logging
logger = logging.getLogger(__name__)

def render_query_panel() -> tuple[str | None, str | None, bool]:
    """Render main query input section and return the user's query."""
    st.markdown("## 💬 Ask Your Question")

    # pre-fill if a suggested follow-up was clicked
    query_text = st.session_state.pop("selected_followup", "")

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

    submitted = st.button("Process Query", type="primary", use_container_width=True, disabled=state.get_state('busy'))
    
    if submitted:
        if query.strip():
            logger.info(f"Query submitted: {query.strip()}")
            return query.strip(), query_type, True
    
    return None, None, False