# src/scripts/build_graph.py
import argparse
import logging
import os
import sys
import pickle
import json
from pathlib import Path
import networkx as nx
import torch
from tqdm import tqdm
from typing import Any, Optional, Dict, List, Set, Tuple, Union

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi.graph.builder import KnowledgeGraphBuilder
from ddi.utils.config import config
from ddi.utils.logging import setup_logging

# --- Add this helper function (if not already present) ---
def load_data(file_path: str) -> Any:
    """Load data from file with error handling

    Args:
        file_path: Path to data file

    Returns:
        Loaded data
    """
    logger = logging.getLogger("load_data") # Use a specific logger
    if not file_path:
        logger.error("No file path provided to load_data.")
        return None
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        if file_path.endswith('.pickle') or file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling file {file_path}: {e}. File might be corrupted or not a pickle file.", exc_info=True)
        return None
    except EOFError as e:
         logger.error(f"EOFError loading pickle file {file_path}: {e}. File might be empty or truncated.", exc_info=True)
         return None
    except json.JSONDecodeError as e:
         logger.error(f"Error decoding JSON from file: {file_path}. Error: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading file {file_path}: {e}", exc_info=True)
        return None


# --- Replace the existing main function with this one ---
def main():
    parser = argparse.ArgumentParser(description="Build integrated knowledge graph")
    parser.add_argument("--drugbank", required=True, help="Path to processed DrugBank data")
    # --- Corrected Arguments ---
    parser.add_argument("--mesh", required=True, help="Path to processed MeSH disease data (e.g., mesh_data_2025.pickle)")
    parser.add_argument("--opentargets_td_assoc", required=True, help="Path to processed OpenTargets target-disease associations")
    parser.add_argument("--disease_mapping", required=True, help="Path to the precomputed disease mapping JSON file (EFO/MONDO -> MeSH)")
    # --- End Corrected Arguments ---
    # Optional arguments for other associations if needed later
    # parser.add_argument("--indications", help="Path to drug-disease indications file (optional)")
    # parser.add_argument("--ot_dt_assoc", help="Path to OpenTargets drug-target associations (optional)")
    parser.add_argument("--output", required=True, help="Output directory for graph")
    parser.add_argument("--log_file", help="Path to log file")
    parser.add_argument("--formats", default="graphml,pickle", help="Comma-separated list of output formats (pyg optional)")
    args = parser.parse_args()

    # Set up logging using the utility function if available, otherwise basic config
    try:
        # Assuming setup_logging is correctly imported or defined elsewhere
        # If not, use basicConfig as fallback
        from ddi.utils.logging import setup_logging
        setup_logging(args.log_file, level=logging.INFO)
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        if args.log_file:
             # Ensure log directory exists
             log_dir = os.path.dirname(args.log_file)
             if log_dir: os.makedirs(log_dir, exist_ok=True)
             # Add file handler
             file_handler = logging.FileHandler(args.log_file)
             file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
             logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger("build_graph")

    logger.info("Starting knowledge graph construction with ID mapping")

    # Initialize graph builder WITH mapping
    # Assuming KnowledgeGraphBuilder is correctly imported or defined elsewhere
    # from ddi.graph.builder import KnowledgeGraphBuilder # Make sure this import is present at the top
    graph_builder = KnowledgeGraphBuilder(output_dir=args.output, disease_mapping_path=args.disease_mapping)

    # Load and add DrugBank data
    logger.info(f"Loading DrugBank data from {args.drugbank}")
    drugbank_data = load_data(args.drugbank)
    if drugbank_data:
        graph_builder.build_graph_from_drugbank(drugbank_data) # Adds drugs, proteins, MeSH categories
    else:
        logger.error("Failed to load DrugBank data. Cannot proceed.")
        return # Exit if critical data is missing

    # Load and add MeSH disease data (ensures MeSH nodes exist before OT processing)
    logger.info(f"Loading MeSH disease data from {args.mesh}")
    mesh_data = load_data(args.mesh)
    if mesh_data:
        graph_builder.add_disease_data(mesh_data) # Adds MeSH disease nodes
    else:
        logger.error("Failed to load MeSH data. Cannot proceed.")
        return # Exit if critical data is missing

    # Load and add OpenTargets target-disease associations (using mapping)
    logger.info(f"Loading OpenTargets target-disease associations from {args.opentargets_td_assoc}")
    ot_td_associations = load_data(args.opentargets_td_assoc)
    if ot_td_associations is not None: # Check for None, allow empty list
        graph_builder.add_target_disease_associations(ot_td_associations) # This method now uses the mapping
    else:
        logger.warning("Failed to load OpenTargets associations. Continuing without them.")


    # --- Add other association types if files are provided ---
    # Example:
    # indications_path = getattr(args, 'indications', None) # Check if arg exists
    # if indications_path:
    #     logger.info(f"Loading indications from {indications_path}")
    #     indications = load_data(indications_path)
    #     if indications:
    #         graph_builder.add_drug_disease_associations(indications) # Assumes MeSH IDs

    # Get graph statistics
    stats = graph_builder.get_statistics()
    logger.info("Graph statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items(): logger.info(f"    {k}: {v}")
        else: logger.info(f"  {key}: {value}")

    # Save graph
    formats = args.formats.split(',')
    logger.info(f"Saving graph in formats: {formats}")
    output_files = graph_builder.save_graph(formats=formats)
    for fmt, file_path in output_files.items(): logger.info(f"Graph saved in {fmt} format: {file_path}")

    logger.info("Knowledge graph construction complete")

# Ensure necessary imports are at the top of the file
# import argparse, logging, os, sys, pickle, json
# from pathlib import Path
# from ddi.graph.builder import KnowledgeGraphBuilder
# from ddi.utils.logging import setup_logging # Or use basicConfig

if __name__ == "__main__":
    # Ensure the load_data function is defined before calling main
    main()
