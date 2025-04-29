# Drug-Disease Interaction Prediction

![version](https://img.shields.io/badge/version-0.1.0-blue)
![python](https://img.shields.io/badge/python-3.8%2B-green)
![license](https://img.shields.io/badge/license-MIT-orange)

A graph-based AI system for predicting drug-disease interactions with applications in drug repurposing and R&D prioritization.

## 📋 Project Overview

This project builds a comprehensive knowledge graph from biomedical data sources and applies graph neural networks to predict potential therapeutic relationships between drugs and diseases. By leveraging established biomedical databases and state-of-the-art graph deep learning techniques, we aim to identify novel drug repurposing opportunities and assist in R&D prioritization.

The system integrates data from:
- DrugBank for comprehensive drug information
- Disease ontologies (MeSH) for disease classification
- OpenTargets Platform for evidence-based target-disease associations

## 🔍 Key Features

- **Comprehensive Knowledge Graph Construction**
  - Integrates multiple biomedical data sources
  - Resolves entities across databases
  - Builds a unified representation of drugs, proteins, pathways, and diseases

- **Advanced Feature Engineering**
  - Drug feature extraction from molecular properties
  - Disease feature extraction from ontology structures
  - Protein embedding generation

- **Graph Neural Network Models**
  - PyTorch Geometric (PyG) based implementation
  - Relation-aware graph attention mechanisms
  - Multi-task prediction capabilities

- **Explainable AI Components**
  - Path-based reasoning for evidence tracking
  - Subgraph highlighting of important interactions
  - Confidence scoring system

- **Production-Ready API**
  - FastAPI-based prediction service
  - Batch processing capabilities
  - Versioned model management

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.3.0+
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

## 📊 Data Preparation

The project uses several biomedical databases that require registration and download.

### DrugBank Data

To download DrugBank data, you need to register for a free account at [DrugBank.ca](https://go.drugbank.com/).

```bash
# Parse DrugBank XML
python src/scripts/parse_drugbank.py --input data/raw/full_database/full_database.xml --output data/processed/drugs

# Process vocabulary
python src/scripts/parse_vocabulary.py --input data/raw/open_data/drugbank_all_drugbank_vocabulary.csv
```

### Disease Data from MeSH

```bash
# Download and process MeSH data
python src/scripts/download_mesh.py --output data/raw/mesh
python src/scripts/process_mesh.py --input data/raw/mesh --output data/processed/diseases/mesh
```

### OpenTargets Platform Data

```bash
# Download and process OpenTargets data
python src/scripts/download_opentargets.py --output data/raw/open_targets
python src/scripts/process_opentargets.py --input data/raw/open_targets --output data/processed/associations/opentargets
```

## 🏗️ Building the Knowledge Graph

```bash
# Build the unified knowledge graph
python src/scripts/build_graph.py \
  --drugbank data/processed/drugs/drugbank_parsed.pickle \
  --disease data/processed/diseases/mesh/disease_taxonomy.pickle \
  --associations data/processed/associations/opentargets/drug_disease_indications.pickle \
  --output data/graph/full
```

## 🧠 Training Models

```bash
# Train GNN model
python src/scripts/train_model.py --graph data/graph/full/knowledge_graph.pyg
```

## 🌐 API Usage

```bash
# Start API service
uvicorn src.ddi.api.main:app --reload
```

## 📊 Example Notebook

We provide example notebooks demonstrating data exploration and model usage:

```bash
# Run Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🔬 Research Applications

This system can be applied to:

1. **Drug Repurposing**
   - Identify existing drugs for new disease indications
   - Prioritize candidates based on confidence scores

2. **R&D Prioritization**
   - Evaluate promising targets for specific diseases
   - Identify potential mechanisms of action

3. **Side Effect Prediction**
   - Identify potential adverse effects early in development
   - Understand the mechanism of observed side effects

## 🚀 Future Work

- Integration of patient genomic data
- Temporal analysis of clinical trial outcomes
- Multimodal data integration with imaging and clinical notes
- Federated learning across institutional data silos

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citations

If you use this system in your research, please cite the following resources:

### DrugBank
```
Knox C, Wilson M, Klinger CM, et al. DrugBank 6.0: the DrugBank Knowledgebase for 2024. 
Nucleic Acids Res. 2024 Jan 5;52(D1):D1265-D1275. doi: 10.1093/nar/gkad976.
```

### OpenTargets Platform
```
Ochoa D, Karim M, Ghoussaini M, et al. Human genetics evidence supports two-thirds of the 
2021 FDA-approved drugs. Nat Rev Drug Discov. 2022 Aug;21(8):551. doi: 10.1038/d41573-022-00114-1.
```

### MeSH (Medical Subject Headings)
```
Nelson SJ, Schopen M, Savage AG, Schulman JL, Arluk N. The MeSH translation maintenance system: 
structure, interface design, and implementation. Stud Health Technol Inform. 2004;107(Pt 1):67-9.
```

### PyTorch Geometric
```
Fey M, Lenssen JE. Fast Graph Representation Learning with PyTorch Geometric. 
ICLR Workshop on Representation Learning on Graphs and Manifolds. 2019.
```