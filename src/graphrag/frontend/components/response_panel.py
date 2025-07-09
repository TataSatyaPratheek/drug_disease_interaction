# src/graphrag/frontend/components/response_panel.py
"""Renders the response, citations, and follow-up questions."""
import streamlit as st
from .. import state

def render_response():
    """Render enhanced response with professional layout."""
    response_data = state.get_state('last_response')
    if not response_data:
        st.info("💡 Submit a query above to see results here.")
        return

    if not isinstance(response_data, dict):
        st.error("Invalid response format received")
        return

    # Get response components
    final_answer = response_data.get('response', '').strip()
    reasoning = response_data.get('reasoning', '').strip()
    
    # Main Response with improved formatting
    if final_answer and final_answer != "Failed to generate response. Please check Ollama setup.":
        st.markdown("## 🎯 **Research Summary**")
        st.markdown(f"**{final_answer}**")
        st.markdown("---")
    else:
        st.warning("⚠️ No response generated.")
        return

    # Detailed Analysis (Collapsible)
    with st.expander("🧠 **Detailed Analysis & Reasoning**", expanded=True):
        if reasoning and not reasoning.startswith("Error in reasoning"):
            st.markdown(reasoning)
        else:
            st.warning("⚠️ No detailed analysis available.")

    # Citations
    citations = response_data.get('citations')
    if citations:
        with st.expander("📚 **Evidence Sources**", expanded=False):
            for i, citation in enumerate(citations[:10], 1):
                st.markdown(f"**{i}. {citation.get('name', 'Unknown')}** ({citation.get('type', 'unknown')})")

    # Follow-up Questions
    followups = response_data.get('suggested_followups')
    if followups:
        st.markdown("## 💡 Suggested Follow-up Questions")
        for i, question in enumerate(followups):
            if st.button(question, key=f"followup_{i}_{hash(question)}", use_container_width=True):
                _handle_followup_question(question)

    # Debug Information (Collapsible)
    with st.expander("🔧 **Debug Information**", expanded=False):
        st.write("**Retrieved entities:**", {k: len(v) for k, v in response_data.get('retrieved_data', {}).items()})
        st.write("**Confidence score:**", response_data.get('confidence_score', 0))
        st.write("**Query type:**", response_data.get('query_type', 'unknown'))

@st.fragment
def _handle_followup_question(question: str):
    """Handle follow-up question with simplified state management."""
    # Simply set the question as the next query
    st.session_state.pending_followup = question
    st.rerun()
