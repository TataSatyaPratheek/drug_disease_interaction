# src/graphrag/frontend/router.py
"""Handles multi-page navigation."""
from .pages import statistics, browse_entities

def render_page(page_name: str):
    if page_name == "statistics":
        statistics.render()
    elif page_name == "browse_entities":
        browse_entities.render()
    else:
        # Default to main content, which is handled in app.py
        pass 
