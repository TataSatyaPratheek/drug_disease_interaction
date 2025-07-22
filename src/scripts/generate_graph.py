#!/usr/bin/env python3
"""
Orchestrator script to generate comprehensive drug-disease interaction knowledge graph
in two phases: nodes first, then edges.

This approach maximizes memory efficiency and allows for comprehensive edge generation.
"""

import subprocess
import sys
import time
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/vi/Documents/drug_disease_interaction/logs/graph_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path, script_name):
    """Run a Python script and handle errors"""
    logger.info(f"="*60)
    logger.info(f"STARTING {script_name.upper()}")
    logger.info(f"="*60)
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        # Log stdout if available
        if result.stdout:
            logger.info(f"{script_name} output:\n{result.stdout}")
        
        duration = time.time() - start_time
        logger.info(f"✓ {script_name} completed successfully in {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"✗ {script_name} failed after {duration:.2f} seconds")
        logger.error(f"Error code: {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Unexpected error in {script_name} after {duration:.2f} seconds: {e}")
        return False

def main():
    """Main orchestrator function"""
    logger.info("="*80)
    logger.info("COMPREHENSIVE DRUG-DISEASE INTERACTION KNOWLEDGE GRAPH GENERATION")
    logger.info("="*80)
    logger.info("Phase 1: Nodes generation (memory-optimized)")
    logger.info("Phase 2: Edges generation (comprehensive, no limits)")
    logger.info("="*80)
    
    total_start = time.time()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Phase 1: Generate nodes
    nodes_script = os.path.join(script_dir, "generate_nodes.py")
    if not run_script(nodes_script, "Nodes Generation"):
        logger.error("Nodes generation failed. Stopping.")
        return False
    
    logger.info("\n" + "="*40)
    logger.info("PHASE 1 COMPLETE - STARTING PHASE 2")
    logger.info("="*40)
    
    # Phase 2: Generate edges
    edges_script = os.path.join(script_dir, "generate_edges.py")
    if not run_script(edges_script, "Edges Generation"):
        logger.error("Edges generation failed.")
        return False
    
    # Final summary
    total_duration = time.time() - total_start
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE KNOWLEDGE GRAPH GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    logger.info("Output files:")
    logger.info("  - nodes.csv: Complete node definitions")
    logger.info("  - edges.csv: Comprehensive edge relationships")
    logger.info("="*80)
    
    print(f"\n🎉 Comprehensive knowledge graph generated successfully!")
    print(f"⏱️  Total time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"📁 Output: /home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
