from gqlalchemy import Memgraph
import logging
import os
import sys
from rich.console import Console
from rich.panel import Panel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rich console
console = Console()

# Memgraph connection
memgraph = Memgraph(host='localhost', port=7687)

def execute_cypher_file(memgraph, cypher_file_path):
    """Execute Cypher queries from a file."""
    if not os.path.exists(cypher_file_path):
        logger.error(f"Cypher file not found: {cypher_file_path}")
        return
    
    with open(cypher_file_path, 'r') as f:
        content = f.read()
    
    # Remove comment lines but preserve structure
    lines = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('//'):
            lines.append(line)  # Keep original line with whitespace
        elif not stripped:
            lines.append('')  # Keep empty lines for structure
    
    # Rejoin with newlines to preserve multi-line structure
    cypher_content = '\n'.join(lines)
    
    # Split queries - much simpler approach
    # Split on semicolon followed by newline and whitespace, then look ahead for keywords or end
    import re
    # Split on semicolon at end of line, followed by optional whitespace/newlines, then a keyword
    queries = re.split(r';\s*\n+\s*(?=(?:CREATE|MATCH|LOAD|SHOW|CALL|WITH|RETURN|\s*$))', cypher_content)
    
    # Clean up queries
    cleaned_queries = []
    for query in queries:
        query = query.strip()
        if query and not query.startswith('//'):
            cleaned_queries.append(query)
    
    for query in cleaned_queries:
        if not query:
            continue
        try:
            # Add semicolon back if not present
            if not query.endswith(';'):
                query += ';'
            memgraph.execute(query)
            # Show more context for successful queries
            preview = query.replace('\n', ' ')[:150] + "..." if len(query) > 150 else query.replace('\n', ' ')
            logger.info(f"Executed: {preview}")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            # For debugging, show the first part of the failed query
            preview = query.replace('\n', ' ')[:200] + "..." if len(query) > 200 else query.replace('\n', ' ')
            logger.error(f"Failed query preview: {preview}")

def load_cypher_fallback():
    """Fallback cypher loader"""
    cypher_file = "/home/vi/Documents/drug_disease_interaction/src/scripts/memgraph_setup.cypher"
    if os.path.exists(cypher_file):
        execute_cypher_file(memgraph, cypher_file)
    else:
        console.print("❌ No fallback cypher file found")

def main():
    """Main execution with Blazing Fast loader - CLEAN VERSION"""
    console.print(Panel("🚀 BLAZING FAST MEMGRAPH LOADER (OPTIMIZED)", border_style="red"))
    
    # Try blazing fast loader first (OPTIMIZED VERSION)
    try:
        print("🚀 Attempting BLAZING FAST loader (OPTIMIZED)...")
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from blazing_fast_loader import BlazingFastLoader
        loader = BlazingFastLoader()
        loader.load_graph_blazing_fast()
        print("✅ BLAZING FAST loader completed successfully!")
        return
    except Exception as e:
        print(f"❌ BLAZING FAST loader failed: {e}")
        print("📦 Falling back to cypher loader...")
        import traceback
        traceback.print_exc()
    
    # Fallback to original Cypher approach
    console.print("📜 Falling back to CYPHER LOADER...")
    load_cypher_fallback()

if __name__ == "__main__":
    main()
