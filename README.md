# README.md
"""
# Drug-Disease Interaction Prediction

A graph-based AI system for predicting drug-disease interactions with applications in drug repurposing and R&D prioritization.

## Project Overview

This project builds a knowledge graph from biomedical data sources and applies graph neural networks to predict potential therapeutic relationships between drugs and diseases. The system leverages DrugBank, DisGeNET, and other biomedical databases to create a comprehensive network of drugs, proteins, pathways, and diseases.

## Features

- Comprehensive knowledge graph construction from multiple biomedical sources
- Drug and disease feature extraction 
- Graph neural network models for interaction prediction
- Explainable AI mechanisms to identify evidence paths
- API for predictions with confidence scores

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- DGL (Deep Graph Library) 0.8+
- RDKit 2022.3.5+
- NetworkX 2.6+

### Setup

```bash
# Clone the repository
git clone https://github.com/username/drug-disease-interaction.git
cd drug-disease-interaction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e .
```

## Data Preparation

The project uses DrugBank data as its primary source, along with disease information from DisGeNET.

### DrugBank Data

To download DrugBank data, you need to register for a free account at [DrugBank.ca](https://go.drugbank.com/).

```bash
# Parse DrugBank XML
python src/scripts/parse_drugbank.py --input data/raw/full_database/full_database.xml --output data/processed/drugs

# Process vocabulary
python src/scripts/parse_vocabulary.py --input data/raw/open_data/drugbank_all_drugbank_vocabulary.csv
```

### Disease Data

```bash
# Download and process disease data
python src/scripts/download_additional_data.py --output data/external
```

## Building the Knowledge Graph

```bash
# Build the knowledge graph
python src/scripts/build_graph.py --drugbank data/processed/drugs/drugbank_parsed.pickle --disease data/external/diseases/disgenet.pickle --output data/graph/full
```

## Training Models

```bash
# Train GNN model
python src/scripts/train_model.py --graph data/graph/full/knowledge_graph.dgl --output models/gnn
```

## API Usage

```bash
# Start API service
uvicorn src.ddi.api.main:app --reload
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""

