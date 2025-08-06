# 🧬 DDI-AI: The Biomedical Research Copilot

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

An AI-powered research copilot designed to accelerate drug discovery and analysis. This system now leverages a state-of-the-art **Hybrid Retrieval-Augmented Generation (RAG)** architecture, integrating a biomedical Knowledge Graph, vector search, and Large Language Models to provide accurate, context-aware, and verifiable answers to complex biomedical questions.

## ✨ Key Features

- **Hybrid RAG Architecture**: Combines the structured, relational power of a **Neo4j Knowledge Graph** with the semantic search capabilities of a **Weaviate vector database**.
- **Advanced Reranking**: Uses a `CrossEncoder` model to rerank and merge results from both data sources, ensuring the most relevant information is used to generate answers.
- **Interactive Frontend**: Responsive user interface built with **Streamlit** for real-time interaction and data visualization.
- **High-Performance Backend**: Fully asynchronous API built with **FastAPI**, optimized for speed and concurrent requests, with Redis caching for repeated queries.
- **Local LLM Integration**: Powered by **Ollama** and orchestrated with **LlamaIndex**, enabling private and cost-effective language model inference on local hardware.
- **Fully Containerized**: The entire stack (frontend, backend, databases, LLM) is containerized using **Docker** and managed with a single `docker-compose` file for easy deployment.


## 🏗️ System Architecture

The system is designed with a modern, decoupled microservices architecture, ensuring scalability and maintainability.

**High-level Architecture Overview:**

```text
User (Researcher)
   │
   ▼
Streamlit Frontend (UI)
   │  REST API
   ▼
FastAPI Backend (API Layer)
   │  Natural Language Query
   ▼
Hybrid RAG Engine (Core)
   ├── Cypher Query ───────────────► Neo4j Graph DB
   ├── Vector Search ─────────────► Weaviate Vector DB
   │
   └─► CrossEncoder Reranker
           │
           ▼
        Ollama LLM Service
           │
           ▼
   <─── Generated Answer ──────────
   (Back to Backend, then to Frontend)
```


**Description:**

- The user interacts with the Streamlit frontend, which communicates with the FastAPI backend via REST API.
- The backend passes queries to the Hybrid RAG Engine, which retrieves information from both Neo4j (graph) and Weaviate (vector) databases.
- Results are reranked using a CrossEncoder model, and the best context is sent to the Ollama LLM service for answer generation.
- The generated answer is returned to the frontend for display.

## 🛠️ Tech Stack

| Component         | Technology                                                                                             |
| :---------------- | :---------------------------------------------------------------------------------------------------- |
| **Frontend**      | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)            |
| **Backend API**   | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)                  |
| **Graph DB**      | ![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)                        |
| **Vector DB**     | ![Weaviate](https://img.shields.io/badge/Weaviate-00A98F?logo=weaviate&logoColor=white)               |
| **Caching**       | ![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)                        |
| **LLM Service**   | ![Ollama](https://img.shields.io/badge/Ollama-2395FF?logo=ollama&logoColor=white)                     |
| **Orchestration** | ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-4B0082?logo=llama&logoColor=white)              |
| **Deployment**    | ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)                     |

## 🚀 Getting Started: One-Click Local Launch

This project is designed for a simple, one-click local launch. The entire stack will be orchestrated by Docker.

### Prerequisites

1. **Docker & Docker Compose**: Ensure Docker Desktop or Docker Engine with the Compose plugin is installed and running.
2. **Git**: For cloning the repository.
3. **A Capable Machine**: Recommended **16GB+ RAM** and an NVIDIA GPU (like your 1650 Ti) for the best experience. The system will run on CPU but will be slower.
4. **`sudo` access**: The startup script needs `sudo` to set correct file permissions for the database volumes.

### Installation & Launch

From your terminal, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/TataSatyaPratheek/drug_disease_interaction.git
cd drug_disease_interaction

# 2. Make the startup script executable
chmod +x start_local_poc.sh

# 3. Run the one-click startup script
./start_local_poc.sh
```

The script will handle everything. After a few minutes (the first build is longest), your full application stack will be running.

### How to Use the Application

Once the startup script completes, access the system via your web browser:

- **Main Application (Frontend)**: `http://localhost:8501`
- **Backend API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Neo4j Database Browser**: `http://localhost:7474`
- **Weaviate Health Status**: `http://localhost:8080/v1/.well-known/ready`

## 📁 Project Structure

The project is organized into a clean, modular structure:

```
.
├── docker/                 # Dockerfile for the API and docker-compose.yml
├── config/                 # Hardware and application configuration files
├── data/                   # Raw, processed, and database persistence files
├── scripts/                # Data parsing, model downloading, and utility scripts
├── src/
│   ├── api/                # FastAPI backend: routes, models, dependencies
│   ├── core/               # Core logic: Hybrid RAG engine, DB services
│   ├── frontend/           # Streamlit frontend application and components
│   ├── parser/             # Parsers for DrugBank, MeSH, OpenTargets
│   ├── tests/              # Unit and integration tests (pytest)
│   └── utils/              # Utility modules (config, logging, etc.)
├── .env                    # Local environment variables (generated by script)
├── pyproject.toml          # Project dependencies and metadata
└── start_local_poc.sh      # The main startup script
```



## 📊 Data Preparation

This project requires several biomedical datasets. You must manually obtain these files from their official sources and place them in the correct locations under the `data/` directory before running any processing scripts.

### Required Raw Data Files

- **DrugBank**
  - Register for a free account at [DrugBank](https://go.drugbank.com/).
  - Download the full DrugBank XML file (e.g., `full_database.xml`).
  - Place it at: `data/raw/full_database/full_database.xml`

- **MeSH (Medical Subject Headings)**
  - Download the latest MeSH descriptor data from the [NIH MeSH Download Page](https://www.nlm.nih.gov/databases/download/mesh.html).
  - Place the descriptor file (e.g., `desc2025.xml` or similar) at: `data/raw/mesh/desc2025.xml`

- **OpenTargets**
  - Download the latest OpenTargets Platform data from the [OpenTargets Downloads](https://platform.opentargets.org/downloads-data).
  - Place the relevant JSON or parquet files at: `data/raw/open_targets/`

Refer to the script docstrings and comments in `scripts/` for the exact filenames and formats expected by each processing step.

**Note:** This project does not provide the raw data files. You are responsible for obtaining them and placing them in the correct locations as described above.

## 🗺️ Roadmap & Future Work

This proof of concept establishes a powerful foundation. Future development will focus on:

- [ ] **Advanced Investigation Modes**: Implementing the backend logic for the "Hypothesis Testing," "Drug Repurposing," and "Drug Discovery" modes selectable in the UI.
- [ ] **Interactive Graph Visualizations**: Displaying retrieved Neo4j subgraphs directly in the frontend to provide visual evidence for answers.
- [ ] **Source-Cited Streaming**: Enhancing the streaming response to include real-time citations pointing back to the specific entities or documents used for generation.
- [ ] **User Authentication & History**: Adding user accounts to save and manage research sessions.
- [ ] **Fine-Tuning LLMs**: Fine-tuning smaller, specialized language models on biomedical corpora for improved accuracy and reduced computational cost.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 Citations

If you use this system in your research, please cite the following resources:


### DrugBank

```text
Knox C, Wilson M, Klinger CM, et al. DrugBank 6.0: the DrugBank Knowledgebase for 2024.
Nucleic Acids Res. 2024 Jan 5;52(D1):D1265-D1275. doi: 10.1093/nar/gkad976.
```


### OpenTargets Platform

```text
Ochoa D, Karim M, Ghoussaini M, et al. Human genetics evidence supports two-thirds of the
2021 FDA-approved drugs. Nat Rev Drug Discov. 2022 Aug;21(8):551. doi: 10.1038/d41573-022-00114-1.
```


### MeSH (Medical Subject Headings)

```text
Nelson SJ, Schopen M, Savage AG, Schulman JL, Arluk N. The MeSH translation maintenance system:
structure, interface design, and implementation. Stud Health Technol Inform. 2004;107(Pt 1):67-9.
```


### PyTorch Geometric

```text
Fey M, Lenssen JE. Fast Graph Representation Learning with PyTorch Geometric.
ICLR Workshop on Representation Learning on Graphs and Manifolds.
```
