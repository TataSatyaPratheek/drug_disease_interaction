// =============================================================================
// MEMGRAPH DATA VALIDATION & QUALITY CHECKS
// Verify consistency, completeness, and quality of loaded graph
// =============================================================================

// 1. BASIC STATISTICS
CALL {
  MATCH (n) RETURN 'Total Nodes' AS metric, count(*) AS value
  UNION
  MATCH ()-[r]->() RETURN 'Total Relationships' AS metric, count(*) AS value
  UNION
  MATCH (n:Disease) RETURN 'Disease Nodes' AS metric, count(*) AS value
  UNION
  MATCH (n:Target) RETURN 'Target Nodes' AS metric, count(*) AS value
  UNION
  MATCH (n:Drug) RETURN 'Drug Nodes' AS metric, count(*) AS value
  UNION
  MATCH (n:Pathway) RETURN 'Pathway Nodes' AS metric, count(*) AS value
}
RETURN metric, value ORDER BY value DESC;

// 2. ENHANCED RELATIONSHIP TYPE ANALYSIS
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC;

// Relationship distribution by data source
MATCH ()-[r]->()
WHERE r.data_source IS NOT NULL
RETURN r.data_source AS data_source, type(r) AS relationship_type, count(*) AS count
ORDER BY data_source, count DESC;

// 3. NODE DEGREE DISTRIBUTION
// Find highly connected nodes
MATCH (n)
OPTIONAL MATCH (n)-[r]-()
WITH n, count(r) AS degree, labels(n)[0] AS label
WHERE degree > 0
RETURN label, 
       min(degree) AS min_degree,
       max(degree) AS max_degree,
       avg(degree) AS avg_degree,
       count(*) AS node_count
ORDER BY max_degree DESC;

// 4. PATHWAY CONNECTIVITY ANALYSIS
// Check pathway hub status
MATCH (p:Pathway)
OPTIONAL MATCH (t:Target)-[:INVOLVED_IN]->(p)
OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(d:Disease)
WITH p, count(DISTINCT t) AS targets, count(DISTINCT d) AS diseases
RETURN 
  CASE 
    WHEN targets = 0 THEN 'Isolated Pathways'
    WHEN targets < 5 THEN 'Low Connected (1-4 targets)'
    WHEN targets < 20 THEN 'Medium Connected (5-19 targets)'
    ELSE 'Highly Connected (20+ targets)'
  END AS connectivity_level,
  count(*) AS pathway_count,
  avg(targets) AS avg_targets,
  avg(diseases) AS avg_diseases
ORDER BY connectivity_level;

// 5. ORPHAN NODE DETECTION
// Find nodes with no connections
MATCH (n)
WHERE NOT EXISTS((n)-[]-())
RETURN labels(n)[0] AS node_type, count(*) AS orphan_count;

// 6. ENHANCED DATA QUALITY CHECKS

// Check for missing essential properties
MATCH (d:Disease)
WHERE d.name IS NULL OR d.name = ''
RETURN 'Disease nodes missing names' AS issue, count(*) AS count
UNION
MATCH (t:Target)
WHERE t.symbol IS NULL OR t.symbol = ''
RETURN 'Target nodes missing symbols' AS issue, count(*) AS count
UNION
MATCH (dr:Drug)
WHERE dr.name IS NULL OR dr.name = ''
RETURN 'Drug nodes missing names' AS issue, count(*) AS count
UNION
MATCH (p:Pathway)
WHERE p.name IS NULL OR p.name = ''
RETURN 'Pathway nodes missing names' AS issue, count(*) AS count;

// Chemical structure data quality
MATCH (d:Drug)
WHERE d.has_structure_data = true
RETURN 'Drugs with structure data' AS metric, count(*) AS count
UNION
MATCH (d:Drug)
WHERE d.canonical_smiles IS NOT NULL AND d.canonical_smiles <> ''
RETURN 'Drugs with SMILES' AS metric, count(*) AS count
UNION
MATCH (d:Drug)
WHERE d.molecular_weight IS NOT NULL
RETURN 'Drugs with molecular weight' AS metric, count(*) AS count
UNION
MATCH (d:Drug)
WHERE d.lipinski_compliant = true
RETURN 'Lipinski compliant drugs' AS metric, count(*) AS count
UNION
MATCH (d:Drug)
WHERE d.isApproved = true
RETURN 'Approved drugs' AS metric, count(*) AS count;

// 7. ENHANCED PATH COMPLETENESS CHECK
// Verify complete drug→target→pathway→disease paths exist
MATCH path = (d:Drug)-[:TARGETS]->(t:Target)-[:INVOLVED_IN]->(p:Pathway)-[:ASSOCIATED_WITH]->(dis:Disease)
RETURN 'Complete 4-hop paths' AS metric, count(*) AS count
UNION
MATCH (d:Drug)-[:TARGETS]->(t:Target)
WHERE NOT EXISTS((t)-[:INVOLVED_IN]->(:Pathway))
RETURN 'Drug-Target pairs without pathway info' AS metric, count(*) AS count
UNION
MATCH (d:Drug)-[:INDICATED_FOR]->(dis:Disease)
RETURN 'Direct drug-disease indications' AS metric, count(*) AS count
UNION
MATCH (d:Drug)-[:HAS_MECHANISM]->(t:Target)
RETURN 'Drug-target mechanisms' AS metric, count(*) AS count;

// 8. ENHANCED TOP ENTITIES BY CONNECTIVITY

// Most targeted pathways
MATCH (t:Target)-[:INVOLVED_IN]->(p:Pathway)
RETURN p.name AS pathway, count(t) AS target_count
ORDER BY target_count DESC
LIMIT 10;

// Most connected diseases
MATCH (p:Pathway)-[:ASSOCIATED_WITH]->(d:Disease)
RETURN d.name AS disease, count(p) AS pathway_count
ORDER BY pathway_count DESC
LIMIT 10;

// Most active drugs (by target count) with chemical data
MATCH (d:Drug)-[:TARGETS]->(t:Target)
WHERE d.has_structure_data = true
RETURN d.name AS drug, 
       count(t) AS target_count,
       d.molecular_weight AS weight,
       d.lipinski_compliant AS lipinski
ORDER BY target_count DESC
LIMIT 10;

// Top drugs by clinical phase
MATCH (d:Drug)-[:INDICATED_FOR]->(dis:Disease)
WHERE d.maxPhase IS NOT NULL
RETURN d.name AS drug,
       d.maxPhase AS max_phase,
       count(dis) AS indication_count,
       d.isApproved AS approved
ORDER BY d.maxPhase DESC, indication_count DESC
LIMIT 10;

// 9. MEMORY AND PERFORMANCE
SHOW STORAGE INFO;

// 10. ENHANCED SAMPLE MULTI-HOP QUERIES FOR TESTING

// Test GraphRAG-style query: "What approved drugs with good chemical properties might treat Alzheimer's disease?"
MATCH path = (d:Drug)-[:TARGETS]->(t:Target)-[:INVOLVED_IN]->(p:Pathway)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS 'Alzheimer' OR dis.name CONTAINS 'dementia'
  AND d.isApproved = true
  AND d.lipinski_compliant = true
  AND d.molecular_weight < 500
RETURN d.name AS drug, 
       d.molecular_weight AS weight,
       d.maxPhase AS phase,
       t.symbol AS target, 
       p.name AS pathway, 
       dis.name AS disease,
       size(relationships(path)) AS path_length
LIMIT 5;

// Test drug indication analysis: "What drugs are indicated for cancer with high clinical phases?"
MATCH (d:Drug)-[ind:INDICATED_FOR]->(dis:Disease)
WHERE dis.name CONTAINS 'cancer' OR dis.name CONTAINS 'carcinoma'
  AND ind.max_phase >= 2
RETURN d.name AS drug,
       dis.name AS disease,
       ind.max_phase AS phase,
       d.molecular_formula AS formula,
       d.mechanism_of_action AS mechanism
ORDER BY ind.max_phase DESC
LIMIT 10;

// Test pathway analysis: "What are the key pathways in neurological diseases with drug targets?"
MATCH (d:Drug)-[:TARGETS]->(t:Target)-[:INVOLVED_IN]->(p:Pathway)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS 'neuro' OR dis.name CONTAINS 'brain' OR dis.name CONTAINS 'nervous'
RETURN p.name AS pathway, 
       count(DISTINCT d) AS drug_count,
       count(DISTINCT t) AS target_count,
       count(DISTINCT dis) AS disease_count,
       collect(DISTINCT d.name)[0..3] AS sample_drugs
ORDER BY drug_count DESC
LIMIT 10;

// Test chemical structure filtering: "Find small molecule drugs targeting kinases"
MATCH (d:Drug)-[:TARGETS]->(t:Target)
WHERE t.symbol CONTAINS 'kinase' OR t.symbol ENDS WITH 'K'
  AND d.molecular_weight < 600
  AND d.has_structure_data = true
  AND d.canonical_smiles IS NOT NULL
RETURN d.name AS drug,
       t.symbol AS target,
       d.molecular_weight AS weight,
       d.rotatable_bonds AS flexibility,
       d.lipinski_compliant AS lipinski
ORDER BY d.molecular_weight ASC
LIMIT 10;
