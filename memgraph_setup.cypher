// =============================================================================
// MEMGRAPH KNOWLEDGE GRAPH SETUP
// Drug-Disease Interaction Graph with Pathway Nodes
// =============================================================================

// 1. DROP EXISTING DATA (if reloading)
// CAUTION: This will delete all data in the database
// MATCH (n) DETACH DELETE n;

// 2. CREATE CONSTRAINTS FOR DATA INTEGRITY
// These ensure uniqueness and improve query performance

// Node constraints
CREATE CONSTRAINT ON (d:Disease) ASSERT d.id IS UNIQUE;
CREATE CONSTRAINT ON (t:Target) ASSERT t.id IS UNIQUE;
CREATE CONSTRAINT ON (p:Pathway) ASSERT p.id IS UNIQUE;
CREATE CONSTRAINT ON (dr:Drug) ASSERT dr.id IS UNIQUE;

// 3. CREATE INDEXES FOR PERFORMANCE
// These speed up lookups and traversals

CREATE INDEX ON :Disease(name);
CREATE INDEX ON :Target(symbol);
CREATE INDEX ON :Pathway(name);
CREATE INDEX ON :Drug(name);

// 4. LOAD NODES FROM CSV
// Enhanced schema with chemical structures and drug enrichment data

// Load Disease nodes
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/nodes.csv" WITH HEADER AS row
WHERE row.type = 'Disease'
CREATE (:Disease {
  id: row.id,
  name: row.name,
  mesh_terms: CASE WHEN row.mesh_terms <> '' AND row.mesh_terms IS NOT NULL 
               THEN split(replace(replace(replace(row.mesh_terms, '[', ''), ']', ''), '"', ''), ',') 
               ELSE [] END,
  mesh_description: row.mesh_description,
  tree_numbers: CASE WHEN row.tree_numbers <> '' AND row.tree_numbers IS NOT NULL
                 THEN split(replace(replace(replace(row.tree_numbers, '[', ''), ']', ''), '"', ''), ',')
                 ELSE [] END
});

// Load Target nodes
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/nodes.csv" WITH HEADER AS row
WHERE row.type = 'Target'
CREATE (:Target {
  id: row.id,
  symbol: row.symbol,
  proteinIds: CASE WHEN row.proteinIds <> '' AND row.proteinIds IS NOT NULL
               THEN split(replace(replace(replace(row.proteinIds, '[', ''), ']', ''), '"', ''), ',') 
               ELSE [] END
});

// Load Drug nodes
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/nodes.csv" WITH HEADER AS row
WHERE row.type = 'Drug'
CREATE (:Drug {
  id: row.id,
  name: row.name,
  description: row.description,
  state: row.state,
  groups: row.groups,
  indication: row.indication,
  mechanism_of_action: row.mechanism_of_action,
  pharmacodynamics: row.pharmacodynamics,
  toxicity: row.toxicity,
  synonyms: row.synonyms,
  
  // Chemical structure properties (enhanced)
  has_structure_data: row.has_structure_data = 'True',
  molecular_formula: row.molecular_formula,
  molecular_weight: CASE WHEN row.molecular_weight <> '' AND row.molecular_weight IS NOT NULL THEN toFloat(row.molecular_weight) ELSE null END,
  canonical_smiles: row.canonical_smiles,
  inchi: row.inchi,
  inchikey: row.inchikey,
  logp: CASE WHEN row.logp <> '' AND row.logp IS NOT NULL THEN toFloat(row.logp) ELSE null END,
  tpsa: CASE WHEN row.tpsa <> '' AND row.tpsa IS NOT NULL THEN toFloat(row.tpsa) ELSE null END,
  rotatable_bonds: CASE WHEN row.rotatable_bonds <> '' AND row.rotatable_bonds IS NOT NULL THEN toInteger(row.rotatable_bonds) ELSE null END,
  h_bond_donors: CASE WHEN row.h_bond_donors <> '' AND row.h_bond_donors IS NOT NULL THEN toInteger(row.h_bond_donors) ELSE null END,
  h_bond_acceptors: CASE WHEN row.h_bond_acceptors <> '' AND row.h_bond_acceptors IS NOT NULL THEN toInteger(row.h_bond_acceptors) ELSE null END,
  aromatic_rings: CASE WHEN row.aromatic_rings <> '' AND row.aromatic_rings IS NOT NULL THEN toInteger(row.aromatic_rings) ELSE null END,
  heavy_atoms: CASE WHEN row.heavy_atoms <> '' AND row.heavy_atoms IS NOT NULL THEN toInteger(row.heavy_atoms) ELSE null END,
  lipinski_compliant: CASE WHEN row.lipinski_compliant = 'True' THEN true WHEN row.lipinski_compliant = 'False' THEN false ELSE null END,
  lipinski_violations: CASE WHEN row.lipinski_violations <> '' AND row.lipinski_violations IS NOT NULL THEN toInteger(row.lipinski_violations) ELSE null END,
  
  // Target interaction counts
  target_count: CASE WHEN row.target_count <> '' AND row.target_count IS NOT NULL THEN toInteger(row.target_count) ELSE 0 END,
  enzyme_count: CASE WHEN row.enzyme_count <> '' AND row.enzyme_count IS NOT NULL THEN toInteger(row.enzyme_count) ELSE 0 END,
  transporter_count: CASE WHEN row.transporter_count <> '' AND row.transporter_count IS NOT NULL THEN toInteger(row.transporter_count) ELSE 0 END,
  carrier_count: CASE WHEN row.carrier_count <> '' AND row.carrier_count IS NOT NULL THEN toInteger(row.carrier_count) ELSE 0 END,
  
  // Classification
  atc_codes: row.atc_codes,
  categories: row.categories,
  pathway_count: CASE WHEN row.pathway_count <> '' AND row.pathway_count IS NOT NULL THEN toInteger(row.pathway_count) ELSE 0 END,
  
  // ChEMBL specific properties (if available)
  drugType: row.drugType,
  maxPhase: CASE WHEN row.maxPhase <> '' AND row.maxPhase IS NOT NULL THEN toInteger(row.maxPhase) ELSE null END,
  isApproved: CASE WHEN row.isApproved = 'True' THEN true WHEN row.isApproved = 'False' THEN false ELSE null END,
  hasBeenWithdrawn: CASE WHEN row.hasBeenWithdrawn = 'True' THEN true WHEN row.hasBeenWithdrawn = 'False' THEN false ELSE null END,
  blackBoxWarning: CASE WHEN row.blackBoxWarning = 'True' THEN true WHEN row.blackBoxWarning = 'False' THEN false ELSE null END,
  yearOfFirstApproval: CASE WHEN row.yearOfFirstApproval <> '' AND row.yearOfFirstApproval IS NOT NULL THEN toInteger(row.yearOfFirstApproval) ELSE null END,
  tradeNames: row.tradeNames,
  linkedDiseases: CASE WHEN row.linkedDiseases <> '' AND row.linkedDiseases IS NOT NULL THEN toInteger(row.linkedDiseases) ELSE 0 END,
  linkedTargets: CASE WHEN row.linkedTargets <> '' AND row.linkedTargets IS NOT NULL THEN toInteger(row.linkedTargets) ELSE 0 END,
  
  // Data source
  data_source: row.data_source
});

// Load Pathway nodes
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/nodes.csv" WITH HEADER AS row
WHERE row.type = 'Pathway'
CREATE (:Pathway {
  id: row.id,
  name: row.name,
  description: row.description,
  source: row.source
});

// 5. LOAD EDGES FROM CSV
// Enhanced relationships with drug indications, mechanisms, and inferred pathway connections

// Load TARGETS relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'TARGETS'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:TARGETS {
  evidence: row.evidence,
  target_name: row.target_name,
  actions: row.actions,
  organism: row.organism,
  data_source: row.data_source
}]->(dst);

// Load METABOLIZED_BY relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'METABOLIZED_BY'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:METABOLIZED_BY {
  target_name: row.target_name,
  actions: row.actions,
  organism: row.organism,
  data_source: row.data_source
}]->(dst);

// Load TRANSPORTED_BY relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'TRANSPORTED_BY'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:TRANSPORTED_BY {
  target_name: row.target_name,
  actions: row.actions,
  organism: row.organism,
  data_source: row.data_source
}]->(dst);

// Load CARRIED_BY relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'CARRIED_BY'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:CARRIED_BY {
  target_name: row.target_name,
  actions: row.actions,
  organism: row.organism,
  data_source: row.data_source
}]->(dst);

// Load INDICATED_FOR relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'INDICATED_FOR'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:INDICATED_FOR {
  max_phase: CASE WHEN row.max_phase <> '' AND row.max_phase IS NOT NULL THEN toInteger(row.max_phase) ELSE null END,
  efo_name: row.efo_name,
  data_source: row.data_source
}]->(dst);

// Load HAS_MECHANISM relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'HAS_MECHANISM'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:HAS_MECHANISM {
  action_type: row.action_type,
  mechanism: row.mechanism,
  target_name: row.target_name,
  target_type_detailed: row.target_type_detailed,
  data_source: row.data_source
}]->(dst);

// Load INVOLVED_IN relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'INVOLVED_IN'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:INVOLVED_IN {
  pathwayName: row.pathwayName
}]->(dst);

// Load ASSOCIATED_WITH relationships
LOAD CSV FROM "file:///home/vi/Documents/drug_disease_interaction/data/processed/graph_csv/edges.csv" WITH HEADER AS row
WHERE row.type = 'ASSOCIATED_WITH'
MATCH (src {id: row.source})
MATCH (dst {id: row.target})
CREATE (src)-[:ASSOCIATED_WITH {
  via_target: row.via_target,
  data_source: row.data_source
}]->(dst);

// 6. VERIFY DATA LOAD
// Quick sanity check
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count;

// Relationship counts
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count;

// 7. ENHANCED SAMPLE QUERIES TO TEST FUNCTIONALITY

// Find drugs with chemical structure data that target cancer pathways
MATCH (d:Drug)-[:TARGETS]->(t:Target)-[:INVOLVED_IN]->(p:Pathway)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS 'cancer' OR dis.name CONTAINS 'carcinoma'
  AND d.has_structure_data = true
  AND d.canonical_smiles IS NOT NULL
RETURN d.name AS drug, 
       d.molecular_formula AS formula,
       d.molecular_weight AS weight,
       d.lipinski_compliant AS lipinski,
       p.name AS pathway, 
       dis.name AS disease
LIMIT 10;

// Find Lipinski-compliant drugs with high clinical phases
MATCH (d:Drug)
WHERE d.lipinski_compliant = true 
  AND d.maxPhase >= 3
  AND d.molecular_weight < 500
RETURN d.name AS drug,
       d.molecular_weight AS weight,
       d.maxPhase AS phase,
       d.logp AS logp,
       d.tpsa AS tpsa,
       d.isApproved AS approved
ORDER BY d.maxPhase DESC, d.molecular_weight ASC
LIMIT 15;

// Find drug indications with high clinical trial phases
MATCH (d:Drug)-[ind:INDICATED_FOR]->(dis:Disease)
WHERE ind.max_phase >= 2
RETURN d.name AS drug,
       dis.name AS disease,
       ind.max_phase AS phase,
       ind.efo_name AS indication_name,
       d.data_source AS source
ORDER BY ind.max_phase DESC
LIMIT 20;

// Find the most connected pathways (hub analysis with enhanced data)
MATCH (p:Pathway)
OPTIONAL MATCH (t:Target)-[:INVOLVED_IN]->(p)
OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(d:Disease)
OPTIONAL MATCH (dr:Drug)-[:TARGETS]->(t)
RETURN p.name AS pathway, 
       count(DISTINCT t) AS target_count,
       count(DISTINCT d) AS disease_count,
       count(DISTINCT dr) AS drug_count,
       count(DISTINCT t) + count(DISTINCT d) + count(DISTINCT dr) AS total_connections
ORDER BY total_connections DESC
LIMIT 20;

// Analyze drug mechanism diversity
MATCH (d:Drug)-[mech:HAS_MECHANISM]->(t:Target)
RETURN mech.action_type AS mechanism_type,
       count(DISTINCT d) AS drug_count,
       count(DISTINCT t) AS target_count,
       collect(DISTINCT d.name)[0..3] AS sample_drugs
ORDER BY drug_count DESC
LIMIT 10;

// Memory usage and performance info
SHOW STORAGE INFO;
