# Drug-Disease Interaction Knowledge Graph

## Overview
This project creates a comprehensive knowledge graph representing drug-disease interactions through biological pathways, combining data from Open Targets, DrugBank, MeSH, and Reactome.

## Graph Schema

### Node Types
- **Drug** (18,081 nodes): Pharmaceutical compounds from ChEMBL/DrugBank
- **Target** (78,726 nodes): Protein targets with UniProt harmonization  
- **Pathway** (2,769 nodes): Biological pathways from Reactome
- **Disease** (38,959 nodes): Diseases with MeSH harmonization

### Relationship Types
- **Drug -[:TARGETS]-> Target**: Direct drug-target interactions (253,442 edges)
- **Target -[:INVOLVED_IN]-> Pathway**: Target participation in pathways (46,977 edges)
- **Pathway -[:ASSOCIATED_WITH]-> Disease**: Pathway-disease associations (17,267,754 edges)

### Total Graph Size
- **Nodes**: 138,535
- **Relationships**: 17,568,173
- **Generation Time**: ~336 seconds

## Setup Instructions

### 1. Prerequisites
```bash
# Install Memgraph
sudo apt install memgraph

# Or use Docker
docker run -p 7687:7687 -p 7444:7444 memgraph/memgraph
```

### 2. Generate Graph Data
```bash
cd /home/vi/Documents/drug_disease_interaction
python src/scripts/path.py
```

### 3. Load into Memgraph
```bash
# Copy Cypher scripts to Memgraph accessible location
sudo cp memgraph_setup.cypher /var/lib/memgraph/
sudo cp memgraph_validation.cypher /var/lib/memgraph/

# Connect to Memgraph
mgconsole

# Execute setup
:source /var/lib/memgraph/memgraph_setup.cypher

# Validate data
:source /var/lib/memgraph/memgraph_validation.cypher
```

### 4. Alternative: CSV Import via mgconsole
```cypher
-- If file paths need adjustment, update the setup script
-- Default path: file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/
```

## Key Features

### 1. Biological Pathway Integration
- Reactome pathway data provides mechanistic insights
- Multi-hop queries: Drug → Target → Pathway → Disease
- Supports pathway-centric drug discovery

### 2. ID Harmonization
- **Targets**: Ensembl IDs mapped to UniProt 
- **Diseases**: Open Targets IDs mapped to MeSH
- **Drugs**: ChEMBL IDs maintained for consistency

### 3. Inferred Relationships
- Pathway-Disease associations inferred via shared targets
- Evidence tracking through `inferred_via_target` property
- 17M+ high-confidence inferred edges

### 4. GraphRAG Ready
- Clean schema for LLM queries
- Semantic relationship types
- Rich metadata for context

## Sample Queries

### Find Drug-Disease Paths via Pathways
```cypher
MATCH path = (d:Drug)-[:TARGETS]->(t:Target)-[:INVOLVED_IN]->(p:Pathway)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE disease.name CONTAINS 'cancer'
RETURN d.name, p.name, disease.name
LIMIT 10;
```

### Pathway Hub Analysis
```cypher
MATCH (p:Pathway)
OPTIONAL MATCH (t:Target)-[:INVOLVED_IN]->(p)
OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(d:Disease)
RETURN p.name, count(DISTINCT t) AS targets, count(DISTINCT d) AS diseases
ORDER BY targets DESC
LIMIT 20;
```

### Drug Repurposing Candidates
```cypher
MATCH (d:Drug)-[:TARGETS]->(t:Target)-[:INVOLVED_IN]->(p:Pathway)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE disease.name CONTAINS 'Alzheimer'
AND NOT (d)-[:TREATS]->(disease)  // Assuming direct treatment relationships exist
RETURN d.name AS drug, count(DISTINCT p) AS pathway_count
ORDER BY pathway_count DESC;
```

## Performance Optimizations

### Memory Management
- Chunked processing (5,000 rows per chunk)
- Garbage collection every 10 chunks
- cuDF → pandas conversion for iteration compatibility

### Database Optimization
- Unique constraints on all node IDs
- Indexes on name properties
- Batch transactions (10K nodes, 50K edges)

### Query Optimization
- Use constraints for faster lookups
- Index on frequently queried properties
- Consider relationship direction in queries

## Troubleshooting

### Common Issues
1. **File Path Errors**: Ensure CSV files are accessible to Memgraph user
2. **Memory Issues**: Increase Memgraph memory limits for large datasets
3. **Timeout Errors**: Use smaller transaction batches

### Validation Checks
```cypher
// Check data completeness
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count;

// Verify relationships
MATCH ()-[r]->() RETURN type(r), count(*) AS count;

// Find orphan nodes
MATCH (n) WHERE NOT (n)-[]-() RETURN labels(n)[0], count(*);
```

## Data Sources
- **Open Targets Platform**: Drug-target-disease associations
- **DrugBank**: Comprehensive drug information
- **MeSH**: Medical subject headings
- **Reactome**: Pathway data
- **UniProt**: Protein harmonization

## Next Steps
1. Add direct drug-disease treatment relationships
2. Integrate clinical trial data
3. Add temporal/versioning support
4. Implement GraphRAG query interface
5. Create visualization dashboards

## File Structure
```
├── src/scripts/path.py                    # Main graph generation script
├── memgraph_setup.cypher                  # Database setup and loading
├── memgraph_validation.cypher             # Data quality validation
├── data/processed/graph_csv/
│   ├── nodes.csv                          # All nodes with types
│   └── edges.csv                          # All relationships
└── logs/path_generation.log               # Processing logs
```
