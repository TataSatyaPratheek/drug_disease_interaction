# src/graphrag/frontend/components/response_panel.py
"""Renders the response, citations, and follow-up questions."""
import streamlit as st
from .. import state

def render_response():
    """Render the complete response section from session state."""
    response_data = state.get_state('last_response')
    if not response_data:
        # Show helpful message if no response yet
        st.info("💡 Submit a query above to see results here.")
        return

    # Debug info (remove in production)
    st.write("Debug: Response data keys:", list(response_data.keys()) if isinstance(response_data, dict) else "Not a dict")

    # Main Response
    st.markdown("## 🎯 Final Answer")
    st.markdown(response_data.get('response', 'No response generated.'))

    # Reasoning
    if response_data.get('reasoning'):
        with st.expander("🧠 AI Reasoning Process"):
            st.markdown(response_data['reasoning'])

    # Citations
    citations = response_data.get('citations')
    if citations:
        st.markdown("## 📚 Citations")
        for i, citation in enumerate(citations[:10], 1):
            with st.expander(f"Citation {i}: {citation.get('name', 'Unknown')}"):
                st.write(f"**Type:** {citation.get('type', 'Unknown')}")
                st.write(f"**ID:** {citation.get('entity_id', 'N/A')}")

    # Follow-up Questions
    followups = response_data.get('suggested_followups')
    if followups:
        st.markdown("## 💡 Suggested Follow-up Questions")
        for i, question in enumerate(followups):
            if st.button(question, key=f"followup_{i}"):
                st.session_state.selected_followup = question
                st.rerun()
