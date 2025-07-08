import streamlit as st
from graphrag.frontend import config, state

@st.fragment   # ✅ only this block reruns on every keystroke
def render_query_panel() -> tuple[str | None, str | None]:
    """Return (query_text, query_type) *only* when user presses Submit."""
    with st.form(key="query_form", clear_on_submit=False):
        st.markdown("## 💬 Ask Your Question")

        # pre-fill if a suggested follow-up was clicked
        default_txt = st.session_state.pop("selected_followup", "")
        query_text = st.text_area(
            "question",
            value=default_txt,
            height=120,
            placeholder="Enter your drug-disease research question here…",
            label_visibility="collapsed",
        )

        query_type = st.selectbox(
            "Query Type",
            options=list(config.QUERY_TYPES.keys()),
            format_func=lambda k: config.QUERY_TYPES[k],
            index=0,
        )

        submitted = st.form_submit_button("▶ Process Query", use_container_width=True)

    if submitted and query_text.strip():
        # prevent double-clicks while the back-end is busy
        state.set_state("busy", True)
        return query_text.strip(), query_type

    return None, None
