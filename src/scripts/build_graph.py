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

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi.graph.builder import KnowledgeGraphBuilder
from ddi.utils.config import config
from ddi.utils.logging import setup_logging

def load_data(file_path: str) -> dict:
    """Load data from file
    
    Args:
        file_path: Path to data file
        
    Returns:
        Loaded data
    """
    if file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Build integrated knowledge graph")
    parser.add_argument("--drugbank", required=True, help="Path to processed DrugBank data")
    parser.add_argument("--disease", help="Path to processed disease taxonomy data")
    parser.add_argument("--associations", help="Path to drug-disease associations")
    parser.add_argument("--opentargets", help="Path to OpenTargets associations")
    parser.add_argument("--output", required=True, help="Output directory for graph")
    parser.add_argument("--log_file", help="Path to log file")
    parser.add_argument("--formats", default="graphml,pickle,dgl", help="Comma-separated list of output formats")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_file, level=logging.INFO)
    logger = logging.getLogger("build_graph")
    
    # Log start
    logger.info("Starting knowledge graph construction")
    
    # Initialize graph builder
    graph_builder = KnowledgeGraphBuilder(output_dir=args.output)
    
    # Load and add DrugBank data
    logger.info(f"Loading DrugBank data from {args.drugbank}")
    drugbank_data = load_data(args.drugbank)
    logger.info(f"Building graph from DrugBank data with {len(drugbank_data.get('drugs', []))} drugs")
    graph_builder.build_graph_from_drugbank(drugbank_data)
    
    # Load and add disease data if provided
    if args.disease:
        logger.info(f"Loading disease taxonomy data from {args.disease}")
        disease_data = load_data(args.disease)
        logger.info(f"Adding {len(disease_data)} diseases to graph")
        graph_builder.add_disease_data(disease_data)
    
    # Load and add drug-disease associations if provided
    if args.associations:
        logger.info(f"Loading drug-disease associations from {args.associations}")
        associations = load_data(args.associations)
        logger.info(f"Adding {len(associations)} drug-disease associations to graph")
        graph_builder.add_drug_disease_associations(associations)
    
    # Load and add OpenTargets data if provided
    if args.opentargets:
        logger.info(f"Loading OpenTargets associations from {args.opentargets}")
        opentargets_data = load_data(args.opentargets)
        
        # Add drug-target associations
        drug_target_assocs = opentargets_data.get('drug_target_associations', [])
        logger.info(f"Adding {len(drug_target_assocs)} drug-target associations from OpenTargets")
        graph_builder.add_drug_target_associations(drug_target_assocs)
        
        # Add target-disease associations
        target_disease_assocs = opentargets_data.get('target_disease_associations', [])
        logger.info(f"Adding {len(target_disease_assocs)} target-disease associations from OpenTargets")
        graph_builder.add_target_disease_associations(target_disease_assocs)
    
    # Get graph statistics
    stats = graph_builder.get_statistics()
    logger.info("Graph statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save graph
    formats = args.formats.split(',')
    logger.info(f"Saving graph in formats: {formats}")
    output_files = graph_builder.save_graph(formats=formats)
    
    for fmt, file_path in output_files.items():
        logger.info(f"Graph saved in {fmt} format: {file_path}")
    
    logger.info("Knowledge graph construction complete")

if __name__ == "__main__":
    main()