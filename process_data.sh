#!/bin/bash
# process_data.sh - Script to process MeSH and OpenTargets data

# Set up logging
echo "Setting up data processing..."
mkdir -p logs
mkdir -p data/processed/diseases/mesh
mkdir -p data/processed/associations/opentargets
mkdir -p data/processed/drugs
mkdir -p data/graph/full

# Step 1: Fix MeSH Parser and Process MeSH data
echo "Processing MeSH data..."
python -m src.ddi.data.sources.mesh.parser \
  --mesh_dir data/raw/mesh \
  --output_dir data/processed/diseases/mesh \
  --format pickle

# Step 2: Process OpenTargets data
echo "Processing OpenTargets data..."
python -m src.ddi.data.sources.opentargets.parser \
  --data_dir data/raw/open_targets \
  --output_dir data/processed/associations/opentargets \
  --format pickle

# Step 3: Check if there's already a processed DrugBank file
if [ -f "data/raw/full_database/drugbank_parsed.pickle" ]; then
  echo "Using existing parsed DrugBank data"
  cp data/raw/full_database/drugbank_parsed.pickle data/processed/drugs/drugbank_parsed.pickle
else
  echo "Integrating DrugBank vocabulary with XML data..."
  python -m src.ddi.data.sources.drugbank.integration \
    --xml data/raw/full_database/full_database.xml \
    --vocab data/raw/open_data/drugbank_all_drugbank_vocabulary.csv \
    --output data/processed/drugs \
    --format pickle
fi

# Step 4: Build unified knowledge graph without PyG
echo "Building knowledge graph..."
python -c "
import sys
import pickle
import os
import networkx as nx
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('build_graph')

# Output directory
output_dir = 'data/graph/full'
os.makedirs(output_dir, exist_ok=True)

# Initialize graph
logger.info('Initializing knowledge graph')
graph = nx.MultiDiGraph()
node_types = {}
edge_types = {}

# Load DrugBank data
try:
    logger.info('Loading DrugBank data')
    with open('data/processed/drugs/drugbank_parsed.pickle', 'rb') as f:
        drugbank_data = pickle.load(f)
    
    # Add drugs to graph
    drug_count = 0
    for drug in drugbank_data.get('drugs', []):
        drug_id = drug.get('drugbank_id')
        if not drug_id:
            continue
        
        # Add drug node
        graph.add_node(
            drug_id,
            type='drug',
            name=drug.get('name', drug_id)
        )
        
        # Count node type
        if 'drug' not in node_types:
            node_types['drug'] = 0
        node_types['drug'] += 1
        
        drug_count += 1
    
    logger.info(f'Added {drug_count} drugs to graph')
except Exception as e:
    logger.error(f'Error loading DrugBank data: {e}')

# Save graph
logger.info('Saving graph')
# Save as GraphML
try:
    nx.write_graphml(graph, os.path.join(output_dir, 'knowledge_graph.graphml'))
    logger.info(f'Saved graph in GraphML format')
except Exception as e:
    logger.error(f'Error saving GraphML: {e}')

# Save as pickle
try:
    with open(os.path.join(output_dir, 'knowledge_graph.pickle'), 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f'Saved graph in pickle format')
except Exception as e:
    logger.error(f'Error saving pickle: {e}')

# Save mappings
mappings = {
    'node_types': node_types,
    'edge_types': edge_types,
    'statistics': {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges()
    }
}

with open(os.path.join(output_dir, 'graph_mappings.json'), 'w') as f:
    json.dump(mappings, f, indent=2)

logger.info(f'Knowledge graph construction complete with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges')
"

echo "Data processing complete!"