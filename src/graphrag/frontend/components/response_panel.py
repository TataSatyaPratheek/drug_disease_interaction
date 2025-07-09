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
    
    # Main Response Container
    with st.container():
        if final_answer and final_answer != "Failed to generate response. Please check Ollama setup.":
            st.markdown("## 🎯 Executive Summary")
            st.markdown(f"**{final_answer}**")
            st.markdown("---")
        else:
            st.warning("⚠️ No response generated.")
            return

    # Detailed Analysis Container (Collapsible)
    with st.expander("🧠 **Detailed Scientific Analysis**", expanded=True):
        if reasoning and not reasoning.startswith("Error in reasoning"):
            st.markdown(reasoning)
        else:
            st.warning("⚠️ No detailed analysis available.")

    # Citations Container
    citations = response_data.get('citations', [])
    if citations:
        with st.expander("📚 **Evidence Sources**", expanded=False):
            for citation in citations[:10]:
                st.markdown(f"**{citation.get('name', 'Unknown')}** ({citation.get('type', 'unknown')})")

    # Follow-up Questions Container (Fixed)
    followups = response_data.get('suggested_followups', [])
    if followups:
        st.markdown("### 💡 Related Questions")
        for i, question in enumerate(followups):
            if st.button(question, key=f"followup_{i}_{hash(question)}", use_container_width=True):
                # Store in conversation and trigger new query
                _handle_followup_question(question)

    # Debug Information (Collapsible)
    with st.expander("🔧 **Debug Information**", expanded=False):
        st.write("**Retrieved entities:**", {k: len(v) for k, v in response_data.get('retrieved_data', {}).items()})
        st.write("**Confidence score:**", response_data.get('confidence_score', 0))
        st.write("**Query type:**", response_data.get('query_type', 'unknown'))

@st.fragment
def _handle_followup_question(question: str):
    """Handle follow-up question with proper state management."""
    from .conversation_manager import ConversationManager
    
    # Store the follow-up as a conversation turn
    ConversationManager.store_conversation_turn(
        question, 
        {"response": "Follow-up question selected", "retrieved_data": {}},
        "follow_up"
    )
    
    # Set the question as the next query
    st.session_state.pending_followup = question
    
    # Clear previous results to avoid confusion
    st.session_state.last_response = None
    st.session_state.last_query = None
    
    # Trigger rerun
    st.rerun()
