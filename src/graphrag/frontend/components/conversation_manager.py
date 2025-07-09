"""Conversation memory management with 10-turn limit."""

import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime

class ConversationManager:
    """Manages conversation history and context."""
    
    MAX_TURNS = 10
    
    @staticmethod
    def initialize_conversation():
        """Initialize conversation storage."""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = {}
    
    @staticmethod
    def store_conversation_turn(query: str, response: dict, turn_type: str = "user_query"):
        """Store a conversation turn with context."""
        ConversationManager.initialize_conversation()
        
        turn = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'turn_type': turn_type,
            'entities': response.get('retrieved_data', {}),
            'confidence': response.get('confidence_score', 0)
        }
        
        # Add to history
        st.session_state.conversation_history.append(turn)
        
        # Maintain 10-turn limit
        if len(st.session_state.conversation_history) > ConversationManager.MAX_TURNS:
            st.session_state.conversation_history.pop(0)
        
        # Update context
        ConversationManager._update_context()
    
    @staticmethod
    def _update_context():
        """Update conversation context for continuity."""
        history = st.session_state.conversation_history
        if not history:
            return
        
        # Extract recent entities and topics
        recent_entities = {}
        recent_topics = []
        
        for turn in history[-5:]:  # Last 5 turns
            entities = turn.get('entities', {})
            for entity_type, entity_list in entities.items():
                if entity_type not in recent_entities:
                    recent_entities[entity_type] = []
                recent_entities[entity_type].extend(entity_list)
        
        st.session_state.conversation_context = {
            'recent_entities': recent_entities,
            'turn_count': len(history),
            'last_query': history[-1]['query'] if history else None
        }
    
    @staticmethod
    def get_conversation_context() -> str:
        """Get formatted conversation context for prompts."""
        if not st.session_state.get('conversation_history'):
            return ""
        
        context_parts = ["CONVERSATION CONTEXT:"]
        
        # Add recent queries
        recent_queries = [turn['query'] for turn in st.session_state.conversation_history[-3:]]
        if recent_queries:
            context_parts.append(f"Recent queries: {'; '.join(recent_queries)}")
        
        # Add persistent entities
        context = st.session_state.get('conversation_context', {})
        recent_entities = context.get('recent_entities', {})
        if recent_entities:
            entity_summary = []
            for entity_type, entities in recent_entities.items():
                unique_entities = list(set(e['name'] for e in entities if e.get('name')))
                if unique_entities:
                    entity_summary.append(f"{entity_type}: {', '.join(unique_entities[:3])}")
            
            if entity_summary:
                context_parts.append(f"Relevant entities: {'; '.join(entity_summary)}")
        
        return "\n".join(context_parts) + "\n\n"
    
    @staticmethod
    def render_conversation_history():
        """Render conversation history in sidebar."""
        if not st.session_state.get('conversation_history'):
            return
        
        st.sidebar.markdown("### 💬 Conversation History")
        
        for i, turn in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.sidebar.expander(f"Q{len(st.session_state.conversation_history)-i}: {turn['query'][:50]}..."):
                st.write(f"**Query:** {turn['query']}")
                st.write(f"**Response:** {turn['response'].get('response', 'No response')[:100]}...")
                st.write(f"**Entities:** {sum(len(v) for v in turn.get('entities', {}).values())}")
