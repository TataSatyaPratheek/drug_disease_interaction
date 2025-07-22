import cudf  # GPU-accelerated DataFrame
import pandas as pd
import os
import json
import pickle
import logging
import time
import gc  # Garbage collection
from collections import defaultdict
from datetime import datetime

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Setup logging with memory optimization
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/vi/Documents/drug_disease_interaction/logs/path_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_status(stage: str) -> None:
    """Log current memory usage status with fallback when psutil is unavailable."""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            logger.info(f"{stage}: RAM usage {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
            
            # Warning for high memory usage
            if memory.percent > 85:
                logger.warning(f"High memory usage detected: {memory.percent:.1f}%")
                gc.collect()  # Force cleanup
                
        except Exception as e:
            logger.warning(f"{stage}: Memory monitoring failed: {e}")
    else:
        logger.info(f"{stage}: Memory monitoring unavailable (install psutil: pip install psutil)")

def log_progress(message, start_time=None):
    """Log progress with timestamp and optional duration"""
    current_time = time.time()
    if start_time:
        duration = current_time - start_time
        logger.info(f"{message} (Duration: {duration:.2f}s)")
    else:
        logger.info(message)
    return current_time

# Paths (update as needed)
logger.info("="*60)
logger.info("STARTING MEMORY-OPTIMIZED DRUG-DISEASE INTERACTION PATH GENERATION")
logger.info("="*60)

start_total = time.time()
log_memory_status("Initial memory state")

OPEN_TARGETS_DIR = "/home/vi/Documents/drug_disease_interaction/data/processed/open_targets_merged"
MESH_DIR = "/home/vi/Documents/drug_disease_interaction/data/processed/mesh"
DRUGBANK_PICKLE = "/home/vi/Documents/drug_disease_interaction/data/processed/drugs/drugbank_parsed.pickle"
MAPPINGS_DIR = "/home/vi/Documents/drug_disease_interaction/data/processed/mappings"
UNIPROT_JSON = os.path.join(MAPPINGS_DIR, "ensembl_to_uniprot_full.json")
DISEASE_MAPPING_CSV = os.path.join(MAPPINGS_DIR, "disease_mapping.csv")
TARGET_MAPPING_CSV = os.path.join(MAPPINGS_DIR, "target_mapping.csv")
OUTPUT_DIR = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv"

# MeSH years to load (2020-2025)
MESH_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

logger.info(f"Output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("Checking required files...")
mesh_pickle_files = [os.path.join(MESH_DIR, f"mesh_data_{year}.pickle") for year in MESH_YEARS]
required_files = mesh_pickle_files + [DRUGBANK_PICKLE, UNIPROT_JSON, DISEASE_MAPPING_CSV, TARGET_MAPPING_CSV]
for file_path in required_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        logger.info(f"✓ Found: {os.path.basename(file_path)} ({size:.1f} MB)")
    else:
        logger.error(f"✗ Missing: {file_path}")

# Load mappings
logger.info("\n" + "="*40)
logger.info("LOADING MAPPING FILES")
logger.info("="*40)

start_mappings = time.time()
logger.info("Loading UniProt mappings...")
log_memory_status("Before loading UniProt mappings")

with open(UNIPROT_JSON, 'r') as f:
    ens_to_uni = json.load(f)
logger.info(f"✓ Loaded {len(ens_to_uni)} Ensembl→UniProt mappings")
log_memory_status("After loading UniProt mappings")
gc.collect()  # Clean up after large JSON load

logger.info("Loading disease mappings...")
disease_mapping = pd.read_csv(DISEASE_MAPPING_CSV)
logger.info(f"✓ Loaded disease mapping: {disease_mapping.shape}")

logger.info("Loading target mappings...")
target_mapping = pd.read_csv(TARGET_MAPPING_CSV)
logger.info(f"✓ Loaded target mapping: {target_mapping.shape}")

disease_map = dict(zip(disease_mapping['index'], disease_mapping['0']))  # Assuming columns 'index' (original) and '0' (mapped)
target_map = dict(zip(target_mapping['index'], target_mapping['0']))
logger.info(f"✓ Created disease map: {len(disease_map)} entries")
logger.info(f"✓ Created target map: {len(target_map)} entries")
log_progress("Mappings loaded", start_mappings)
log_memory_status("After loading all mappings")
gc.collect()  # Clean up after mappings

# Load and process data with cuDF
logger.info("\n" + "="*40)
logger.info("LOADING PARQUET FILES")
logger.info("="*40)

def load_parquet_gpu(file_path):
    """Load parquet file with logging and memory management"""
    filename = os.path.basename(file_path)
    logger.info(f"Loading {filename}...")
    start_load = time.time()
    
    if not os.path.exists(file_path):
        logger.error(f"✗ File not found: {file_path}")
        return None
    
    try:
        df = cudf.read_parquet(file_path)
        duration = time.time() - start_load
        logger.info(f"✓ Loaded {filename}: {df.shape} (Duration: {duration:.2f}s)")
        return df
    except Exception as e:
        logger.error(f"✗ Error loading {filename}: {e}")
        return None

def process_dataframe_in_chunks(df, chunk_size=2000, process_func=None, desc="Processing"):
    """Memory-optimized processing of large dataframes in smaller chunks"""
    if df is None:
        return []
    
    results = []
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    logger.info(f"{desc}: {total_rows} rows in {num_chunks} chunks of {chunk_size}")
    
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        
        try:
            chunk_results = process_func(chunk) if process_func else []
            if chunk_results:
                results.extend(chunk_results)
        except Exception as e:
            logger.warning(f"Error processing chunk {i//chunk_size + 1}: {e}")
            continue
        
        # Memory management
        chunk_num = (i // chunk_size) + 1
        
        # Log progress every 10th chunk or last chunk
        if chunk_num % 10 == 0 or chunk_num == num_chunks:
            logger.info(f"  Processed chunk {chunk_num}/{num_chunks} ({len(results)} items so far)")
            
            # Memory monitoring
            if PSUTIL_AVAILABLE and chunk_num % 20 == 0:
                try:
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85:
                        logger.warning(f"High memory usage ({memory_percent:.1f}%), forcing cleanup")
                        gc.collect()
                        # Emergency stop if memory critical
                        if memory_percent > 90:
                            logger.error(f"Critical memory usage ({memory_percent:.1f}%), stopping chunk processing")
                            break
                except:
                    pass
        
        # Force garbage collection every 20 chunks
        if chunk_num % 20 == 0:
            gc.collect()
        
        # Clear chunk reference
        del chunk
    
    return results

start_parquet = time.time()
ot_target = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "target.parquet"))
ot_disease = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "disease.parquet"))
ot_known_drug = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "known_drug.parquet"))
ot_pathways = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "reactome.parquet"))  # Use reactome for pathway data
ot_molecule = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "drug_molecule.parquet"))  # ChEMBL drug data
ot_associations = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "association_overall_direct.parquet"))  # For pathway-disease inference
log_progress("All parquet files loaded", start_parquet)

logger.info("\n" + "="*40)
logger.info("LOADING PICKLE FILES")
logger.info("="*40)

start_pickle = time.time()
logger.info("Loading multi-year MeSH data...")

def load_mesh_multi_year(mesh_dir, years):
    """Load and combine MeSH data from multiple years"""
    combined_descriptors = {}
    combined_term_to_id = {}
    all_mesh_lookup = {}
    years_loaded = []
    
    for year in years:
        mesh_pickle_path = os.path.join(mesh_dir, f"mesh_data_{year}.pickle")
        if os.path.exists(mesh_pickle_path):
            logger.info(f"Loading MeSH {year} data...")
            try:
                with open(mesh_pickle_path, 'rb') as f:
                    year_data = pickle.load(f)
                
                if isinstance(year_data, dict):
                    year_descriptors = year_data.get('descriptors', {})
                    year_term_to_id = year_data.get('term_to_id', {})
                    
                    # Combine descriptors (latest year takes precedence for duplicates)
                    combined_descriptors.update(year_descriptors)
                    combined_term_to_id.update(year_term_to_id)
                    
                    logger.info(f"✓ Loaded {len(year_descriptors)} descriptors from {year}")
                    years_loaded.append(year)
                else:
                    logger.warning(f"✗ Unexpected data format in {year} file")
                    
            except Exception as e:
                logger.warning(f"✗ Error loading {year} data: {e}")
        else:
            logger.warning(f"✗ Missing MeSH file for {year}")
    
    # Create combined disease lookup
    for desc_id, desc_data in combined_descriptors.items():
        name = desc_data.get('name', '')
        if name:
            all_mesh_lookup[desc_id] = {
                'name': name,
                'synonyms': desc_data.get('synonyms', []),
                'description': desc_data.get('description', ''),
                'tree_numbers': desc_data.get('tree_numbers', [])
            }
    
    logger.info(f"✓ Combined MeSH data from years: {years_loaded}")
    logger.info(f"✓ Total descriptors: {len(combined_descriptors)}")
    logger.info(f"✓ Total term mappings: {len(combined_term_to_id)}")
    logger.info(f"✓ Disease lookup entries: {len(all_mesh_lookup)}")
    
    return {
        'descriptors': combined_descriptors,
        'term_to_id': combined_term_to_id,
        'disease_lookup': all_mesh_lookup,
        'years_processed': years_loaded
    }

# Load multi-year MeSH data
mesh_combined = load_mesh_multi_year(MESH_DIR, MESH_YEARS)
mesh_descriptors = mesh_combined['descriptors']
mesh_term_to_id = mesh_combined['term_to_id']
mesh_disease_lookup = mesh_combined['disease_lookup']

logger.info("Loading DrugBank data...")
with open(DRUGBANK_PICKLE, 'rb') as f:
    drugbank_data = pickle.load(f)  # Load DrugBank
logger.info(f"✓ Loaded DrugBank data: {len(drugbank_data)} entries")
log_progress("Pickle files loaded", start_pickle)

# Create nodes CSV
logger.info("\n" + "="*40)
logger.info("CREATING NODES")
logger.info("="*40)

nodes = []
start_nodes = time.time()

# Disease nodes from Open Targets + MeSH
logger.info("Processing disease nodes...")

def process_disease_chunk(chunk_df):
    chunk_nodes = []
    for idx, row in chunk_df.iterrows():
        mesh_id = disease_map.get(row['id'], row['id'])  # Harmonized MeSH
        
        # Get MeSH disease info if available
        mesh_info = mesh_disease_lookup.get(mesh_id, {})
        mesh_terms = mesh_info.get('synonyms', [])
        mesh_description = mesh_info.get('description', '')
        
        chunk_nodes.append({
            'id': mesh_id,
            'type': 'Disease',
            'name': row.get('approvedName', row.get('name', '')),
            'mesh_terms': mesh_terms,  # From MeSH descriptors
            'mesh_description': mesh_description,
            'tree_numbers': mesh_info.get('tree_numbers', [])
        })
    return chunk_nodes

disease_nodes = []
if ot_disease is not None:
    # Convert to pandas and process in chunks
    disease_df = ot_disease.to_pandas()
    disease_nodes = process_dataframe_in_chunks(
        disease_df, 
        chunk_size=5000, 
        process_func=process_disease_chunk,
        desc="Processing disease nodes"
    )
    nodes.extend(disease_nodes)
    logger.info(f"✓ Created {len(disease_nodes)} disease nodes")
    
    # Clean up memory
    del disease_df
    gc.collect()
else:
    logger.warning("✗ No disease data available - skipping disease nodes")

# Target nodes from Open Targets
logger.info("Processing target nodes...")

def process_target_chunk(chunk_df):
    chunk_nodes = []
    for idx, row in chunk_df.iterrows():
        uni_id = target_map.get(row['id'], ens_to_uni.get(row['id'], [row['id']])[0])  # Harmonized UniProt
        chunk_nodes.append({
            'id': uni_id,
            'type': 'Target',
            'symbol': row.get('approvedSymbol', ''),
            'proteinIds': row.get('proteinIds', [])
        })
    return chunk_nodes

if ot_target is not None:
    # Convert to pandas and process in chunks
    target_df = ot_target.to_pandas()
    target_nodes = process_dataframe_in_chunks(
        target_df, 
        chunk_size=5000, 
        process_func=process_target_chunk,
        desc="Processing target nodes"
    )
    nodes.extend(target_nodes)
    logger.info(f"✓ Created {len(target_nodes)} target nodes")
    
    # Keep target_df for pathway processing later, but clean up cudf version
    del ot_target
    gc.collect()
else:
    logger.warning("✗ No target data available - skipping target nodes")
    target_df = None

# Pathway nodes from Open Targets reactome.parquet
logger.info("Processing pathway nodes...")
pathway_count = 0
if ot_pathways is not None:
    pathway_df = ot_pathways.to_pandas()
    logger.info(f"Found {len(pathway_df)} pathways in Reactome data")
    
    def process_pathway_nodes(chunk_df):
        chunk_nodes = []
        for idx, row in chunk_df.iterrows():
            chunk_nodes.append({
                'id': row['id'],  # Reactome pathway ID
                'type': 'Pathway',
                'name': row.get('name', ''),
                'description': row.get('description', ''),
                'source': 'Reactome'
            })
        return chunk_nodes
    
    pathway_nodes = process_dataframe_in_chunks(
        pathway_df,
        chunk_size=5000,
        process_func=process_pathway_nodes,
        desc="Processing pathway nodes"
    )
    nodes.extend(pathway_nodes)
    pathway_count = len(pathway_nodes)
    logger.info(f"✓ Created {pathway_count} pathway nodes from reactome.parquet")
    
    # Clean up memory
    del pathway_df, ot_pathways
    gc.collect()
else:
    logger.warning("✗ reactome.parquet not found - skipping pathway nodes")

# Drug nodes from enhanced DrugBank + Open Targets drug_molecule (ChEMBL IDs)
logger.info("Processing enhanced drug nodes with chemical structures...")
drug_count = 0

# Load additional drug enrichment files
drug_indication_df = None
drug_mechanism_df = None
drug_warning_df = None
drug_adverse_df = None

# Load drug enrichment parquet files
try:
    drug_indication_path = os.path.join(OPEN_TARGETS_DIR, "drug_indication.parquet")
    drug_mechanism_path = os.path.join(OPEN_TARGETS_DIR, "drug_mechanism_of_action.parquet")
    drug_warning_path = os.path.join(OPEN_TARGETS_DIR, "drug_warning.parquet")
    drug_adverse_path = os.path.join(OPEN_TARGETS_DIR, "openfda_significant_adverse_drug_reactions.parquet")
    
    if os.path.exists(drug_indication_path):
        drug_indication_df = load_parquet_gpu(drug_indication_path)
        if drug_indication_df is not None:
            drug_indication_df = drug_indication_df.to_pandas()
            logger.info(f"✓ Loaded drug indications: {len(drug_indication_df)} entries")
    
    if os.path.exists(drug_mechanism_path):
        drug_mechanism_df = load_parquet_gpu(drug_mechanism_path)
        if drug_mechanism_df is not None:
            drug_mechanism_df = drug_mechanism_df.to_pandas()
            logger.info(f"✓ Loaded drug mechanisms: {len(drug_mechanism_df)} entries")
    
    if os.path.exists(drug_warning_path):
        drug_warning_df = load_parquet_gpu(drug_warning_path)
        if drug_warning_df is not None:
            drug_warning_df = drug_warning_df.to_pandas()
            logger.info(f"✓ Loaded drug warnings: {len(drug_warning_df)} entries")
    
    if os.path.exists(drug_adverse_path):
        drug_adverse_df = load_parquet_gpu(drug_adverse_path)
        if drug_adverse_df is not None:
            drug_adverse_df = drug_adverse_df.to_pandas()
            logger.info(f"✓ Loaded adverse drug reactions: {len(drug_adverse_df)} entries")
        
except Exception as e:
    logger.warning(f"Error loading drug enrichment files: {e}")

# Create enhanced drug nodes from DrugBank data with chemical structures
logger.info("Creating enhanced drug nodes from DrugBank data...")
drugbank_nodes = []
drugbank_chembl_map = {}  # Map DrugBank IDs to ChEMBL IDs for linking

for drug in drugbank_data:
    db_id = drug.get('drugbank_id')
    if not db_id:
        continue
    
    # Enhanced drug node with chemical properties
    drug_node = {
        'id': db_id,
        'type': 'Drug',
        'name': drug.get('name', ''),
        'description': drug.get('description', ''),
        'state': drug.get('state', ''),
        'groups': ','.join([group for group in drug.get('groups', []) if group is not None]),
        'indication': drug.get('indication', ''),
        'mechanism_of_action': drug.get('mechanism_of_action', ''),
        'pharmacodynamics': drug.get('pharmacodynamics', ''),
        'toxicity': drug.get('toxicity', ''),
        'synonyms': ','.join([syn for syn in drug.get('synonyms', []) if syn is not None]),
        
        # Chemical structure data (enhanced from SDF/RDKit)
        'has_structure_data': drug.get('has_structure_data', False),
        'molecular_formula': drug.get('molecular_formula', ''),
        'molecular_weight': drug.get('molecular_weight_rdkit', drug.get('molecular_weight', '')),
        'canonical_smiles': drug.get('canonical_smiles', drug.get('smiles', '')),
        'inchi': drug.get('inchi', ''),
        'inchikey': drug.get('inchikey_rdkit', drug.get('inchikey', '')),
        'logp': drug.get('logp', ''),
        'tpsa': drug.get('tpsa', ''),
        'rotatable_bonds': drug.get('rotatable_bonds', ''),
        'h_bond_donors': drug.get('h_bond_donors', ''),
        'h_bond_acceptors': drug.get('h_bond_acceptors', ''),
        'aromatic_rings': drug.get('aromatic_rings', ''),
        'heavy_atoms': drug.get('heavy_atoms', ''),
        'lipinski_compliant': drug.get('lipinski_compliant', ''),
        'lipinski_violations': drug.get('lipinski_violations', ''),
        
        # Target information
        'target_count': len(drug.get('targets', [])),
        'enzyme_count': len(drug.get('enzymes', [])),
        'transporter_count': len(drug.get('transporters', [])),
        'carrier_count': len(drug.get('carriers', [])),
        
        # ATC codes and categories (with null handling)
        'atc_codes': ','.join([code for code in drug.get('atc_codes', []) if code is not None]),
        'categories': ','.join([cat for cat in drug.get('categories', []) if cat is not None]),
        
        # Pathways
        'pathway_count': len(drug.get('pathways', [])),
        
        # Data source
        'data_source': 'DrugBank_Enhanced'
    }
    
    drugbank_nodes.append(drug_node)

logger.info(f"✓ Created {len(drugbank_nodes)} enhanced DrugBank nodes with chemical structures")

# Merge with Open Targets molecule data if available
if ot_molecule is not None:
    molecule_df = ot_molecule.to_pandas()
    logger.info(f"Found {len(molecule_df)} drug molecules in Open Targets")
    
    def process_drug_chunk(chunk_df):
        chunk_nodes = []
        for idx, row in chunk_df.iterrows():
            chembl_id = row['id']
            
            # Create ChEMBL drug node
            drug_node = {
                'id': chembl_id,
                'type': 'Drug',
                'name': row.get('name', ''),
                'drugType': row.get('drugType', ''),
                'maxPhase': row.get('maximumClinicalTrialPhase', 0),
                'isApproved': row.get('isApproved', False),
                'hasBeenWithdrawn': row.get('hasBeenWithdrawn', False),
                'blackBoxWarning': row.get('blackBoxWarning', False),
                'yearOfFirstApproval': row.get('yearOfFirstApproval', ''),
                'canonicalSmiles': row.get('canonicalSmiles', ''),
                'inchiKey': row.get('inchiKey', ''),
                'synonyms': ','.join(row.get('synonyms', [])) if row.get('synonyms') else '',
                'tradeNames': ','.join(row.get('tradeNames', [])) if row.get('tradeNames') else '',
                'linkedDiseases': row.get('linkedDiseases', {}).get('count', 0) if row.get('linkedDiseases') else 0,
                'linkedTargets': row.get('linkedTargets', {}).get('count', 0) if row.get('linkedTargets') else 0,
                'data_source': 'ChEMBL_OpenTargets'
            }
            chunk_nodes.append(drug_node)
        return chunk_nodes
    
    chembl_drug_nodes = process_dataframe_in_chunks(
        molecule_df,
        chunk_size=5000,
        process_func=process_drug_chunk,
        desc="Processing ChEMBL drug nodes"
    )
    
    # Combine DrugBank and ChEMBL nodes
    all_drug_nodes = drugbank_nodes + chembl_drug_nodes
    drug_count = len(all_drug_nodes)
    
    logger.info(f"✓ Combined {len(drugbank_nodes)} DrugBank + {len(chembl_drug_nodes)} ChEMBL = {drug_count} total drug nodes")
    
    # Clean up memory
    del molecule_df, ot_molecule
    gc.collect()
    
else:
    # Use only enhanced DrugBank nodes
    all_drug_nodes = drugbank_nodes
    drug_count = len(all_drug_nodes)
    logger.info(f"✓ Using {drug_count} enhanced DrugBank drug nodes only")

# Add drug nodes to main nodes list
nodes.extend(all_drug_nodes)

logger.info(f"Total nodes created: {len(nodes)}")
nodes_df = pd.DataFrame(nodes)
nodes_output_path = os.path.join(OUTPUT_DIR, "nodes.csv")
nodes_df.to_csv(nodes_output_path, index=False)
logger.info(f"✓ Saved nodes to: {nodes_output_path}")
log_progress("Nodes creation completed", start_nodes)

# Create edges CSV (drug→target→pathway→disease)
logger.info("\n" + "="*40)
logger.info("CREATING EDGES")
logger.info("="*40)

edges = []
start_edges = time.time()

# Drug → Target from Open Targets known_drug
logger.info("Processing drug→target edges...")
if ot_known_drug is not None:
    # Convert to pandas and process in chunks
    known_drug_df = ot_known_drug.to_pandas()
    
    def process_drug_target_chunk(chunk_df):
        chunk_edges = []
        for idx, row in chunk_df.iterrows():
            chunk_edges.append({
                'source': row['drugId'],
                'target': target_map.get(row['targetId'], row['targetId']),
                'type': 'TARGETS',
                'evidence': row.get('evidence', {}),
                'source_type': 'Drug',
                'target_type': 'Target'
            })
        return chunk_edges
    
    drug_target_edges = process_dataframe_in_chunks(
        known_drug_df,
        chunk_size=5000,
        process_func=process_drug_target_chunk,
        desc="Processing drug→target edges"
    )
    edges.extend(drug_target_edges)
    logger.info(f"✓ Created {len(drug_target_edges)} drug→target edges")
    
    # Clean up memory
    del known_drug_df, ot_known_drug
    gc.collect()
else:
    logger.warning("✗ No drug-target data available")

# Enhanced Drug → Target from DrugBank data
logger.info("Processing enhanced drug→target edges from DrugBank...")
drugbank_target_edges = []
for drug in drugbank_data:
    db_id = drug.get('drugbank_id')
    if not db_id:
        continue
    
    # Process targets
    for target in drug.get('targets', []):
        if target.get('uniprot_id'):
            drugbank_target_edges.append({
                'source': db_id,
                'target': target['uniprot_id'],
                'type': 'TARGETS',
                'target_name': target.get('name', ''),
                'actions': ','.join(target.get('actions', [])),
                'organism': target.get('organism', ''),
                'source_type': 'Drug',
                'target_type': 'Target',
                'data_source': 'DrugBank'
            })
    
    # Process enzymes
    for enzyme in drug.get('enzymes', []):
        if enzyme.get('uniprot_id'):
            drugbank_target_edges.append({
                'source': db_id,
                'target': enzyme['uniprot_id'],
                'type': 'METABOLIZED_BY',
                'target_name': enzyme.get('name', ''),
                'actions': ','.join(enzyme.get('actions', [])),
                'organism': enzyme.get('organism', ''),
                'source_type': 'Drug',
                'target_type': 'Target',
                'data_source': 'DrugBank'
            })
    
    # Process transporters
    for transporter in drug.get('transporters', []):
        if transporter.get('uniprot_id'):
            drugbank_target_edges.append({
                'source': db_id,
                'target': transporter['uniprot_id'],
                'type': 'TRANSPORTED_BY',
                'target_name': transporter.get('name', ''),
                'actions': ','.join(transporter.get('actions', [])),
                'organism': transporter.get('organism', ''),
                'source_type': 'Drug',
                'target_type': 'Target',
                'data_source': 'DrugBank'
            })
    
    # Process carriers
    for carrier in drug.get('carriers', []):
        if carrier.get('uniprot_id'):
            drugbank_target_edges.append({
                'source': db_id,
                'target': carrier['uniprot_id'],
                'type': 'CARRIED_BY',
                'target_name': carrier.get('name', ''),
                'actions': ','.join(carrier.get('actions', [])),
                'organism': carrier.get('organism', ''),
                'source_type': 'Drug',
                'target_type': 'Target',
                'data_source': 'DrugBank'
            })

edges.extend(drugbank_target_edges)
logger.info(f"✓ Created {len(drugbank_target_edges)} enhanced drug→target edges from DrugBank")

# Drug → Disease (Indications) from drug_indication.parquet
logger.info("Processing drug→disease indication edges...")
if drug_indication_df is not None:
    def process_drug_indication_chunk(chunk_df):
        chunk_edges = []
        for idx, row in chunk_df.iterrows():
            drug_id = row['id']
            indications = row.get('indications', [])
            
            if indications and isinstance(indications, list):
                for indication in indications:
                    if indication and isinstance(indication, dict):
                        disease_id = indication.get('disease')
                        max_phase = indication.get('maxPhaseForIndication', 0)
                        
                        if disease_id:
                            harmonized_disease = disease_map.get(disease_id, disease_id)
                            chunk_edges.append({
                                'source': drug_id,
                                'target': harmonized_disease,
                                'type': 'INDICATED_FOR',
                                'max_phase': max_phase,
                                'efo_name': indication.get('efoName', ''),
                                'source_type': 'Drug',
                                'target_type': 'Disease',
                                'data_source': 'OpenTargets_Indications'
                            })
        return chunk_edges
    
    indication_edges = process_dataframe_in_chunks(
        drug_indication_df,
        chunk_size=5000,
        process_func=process_drug_indication_chunk,
        desc="Processing drug→disease indication edges"
    )
    edges.extend(indication_edges)
    logger.info(f"✓ Created {len(indication_edges)} drug→disease indication edges")
else:
    logger.warning("✗ No drug indication data available")

# Drug → Target (Mechanism of Action) from drug_mechanism_of_action.parquet
logger.info("Processing drug mechanism of action edges...")
if drug_mechanism_df is not None:
    def process_drug_mechanism_chunk(chunk_df):
        chunk_edges = []
        for idx, row in chunk_df.iterrows():
            chembl_ids = row.get('chemblIds', [])
            targets = row.get('targets', [])
            
            if chembl_ids and targets:
                for chembl_id in chembl_ids:
                    for target_id in targets:
                        harmonized_target = target_map.get(target_id, target_id)
                        chunk_edges.append({
                            'source': chembl_id,
                            'target': harmonized_target,
                            'type': 'HAS_MECHANISM',
                            'action_type': row.get('actionType', ''),
                            'mechanism': row.get('mechanismOfAction', ''),
                            'target_name': row.get('targetName', ''),
                            'target_type_detailed': row.get('targetType', ''),
                            'source_type': 'Drug',
                            'target_type': 'Target',
                            'data_source': 'OpenTargets_Mechanism'
                        })
        return chunk_edges
    
    mechanism_edges = process_dataframe_in_chunks(
        drug_mechanism_df,
        chunk_size=5000,
        process_func=process_drug_mechanism_chunk,
        desc="Processing drug mechanism edges"
    )
    edges.extend(mechanism_edges)
    logger.info(f"✓ Created {len(mechanism_edges)} drug mechanism edges")
else:
    logger.warning("✗ No drug mechanism data available")

# Target → Pathway from Open Targets target.parquet (pathways column)
logger.info("Processing target→pathway edges...")
if target_df is not None:
    try:
        # Filter for targets that have pathway information
        targets_with_pathways = target_df[target_df['pathways'].notna()]
        logger.info(f"Found {len(targets_with_pathways)} targets with pathway data")
        
        def process_pathway_chunk(chunk_df):
            chunk_edges = []
            for idx, row in chunk_df.iterrows():
                target_id = target_map.get(row['id'], row['id'])  # Use harmonized ID
                
                # The 'pathways' column contains a list of pathway dictionaries
                pathways = row.get('pathways', [])
                if pathways and isinstance(pathways, list):
                    for pathway in pathways:
                        # Use correct key names from debug output
                        if pathway and isinstance(pathway, dict) and pathway.get('pathwayId'):
                            chunk_edges.append({
                                'source': target_id,
                                'target': pathway['pathwayId'],  # Use pathwayId not id
                                'type': 'INVOLVED_IN',
                                'pathwayName': pathway.get('pathway', ''),  # Use pathway not name
                                'source_type': 'Target',
                                'target_type': 'Pathway'
                            })
            return chunk_edges
        
        pathway_edges = process_dataframe_in_chunks(
            targets_with_pathways,
            chunk_size=5000,
            process_func=process_pathway_chunk,
            desc="Processing target→pathway edges"
        )
        edges.extend(pathway_edges)
        logger.info(f"✓ Created {len(pathway_edges)} target→pathway edges from target.parquet")
        
    except Exception as e:
        logger.warning(f"Could not process pathway data from target.parquet: {e}")
        logger.info("Continuing without pathway edges...")
else:
    logger.warning("✗ No target data available for pathways")

# Pathway → Disease (inferred via shared targets) - MEMORY OPTIMIZED FOR 16GB RAM
# Pathway → Disease (inferred via shared targets) - MEMORY OPTIMIZED FOR 16GB RAM
logger.info("Processing pathway→disease edges (memory optimized for 16GB RAM)...")

try:
    # Check if required data is available
    if 'ot_associations' not in locals() or ot_associations is None:
        logger.warning("✗ Missing association data - cannot infer pathway→disease edges")
    elif 'target_df' not in locals() or target_df is None:
        logger.warning("✗ Missing target data - cannot infer pathway→disease edges")
    else:
        log_memory_status("Before pathway-disease inference")
        
        # STEP 1: Build target-to-disease mapping with memory limits
        logger.info("Building target-disease mapping (memory optimized)...")
        assoc_df = ot_associations.to_pandas()
        target_to_diseases = defaultdict(set)
        
        # Process associations in smaller chunks to prevent memory overflow
        assoc_chunk_size = 50000  # Reduced chunk size
        total_assoc_rows = len(assoc_df)
        
        for i in range(0, total_assoc_rows, assoc_chunk_size):
            chunk = assoc_df.iloc[i:i + assoc_chunk_size]
            
            for idx, row in chunk.iterrows():
                target_id = row['targetId']
                disease_id = disease_map.get(row['diseaseId'], row['diseaseId'])
                
                target_to_diseases[target_id].add(disease_id)
            
            # Memory management every 20 chunks
            chunk_num = (i // assoc_chunk_size) + 1
            if chunk_num % 20 == 0:
                logger.info(f"  Processed {chunk_num} association chunks ({len(target_to_diseases)} targets so far)")
                gc.collect()
                
                # Emergency memory check
                if PSUTIL_AVAILABLE:
                    try:
                        import psutil
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 85:
                            logger.warning(f"High memory usage ({memory_percent:.1f}%), limiting processing")
                            break
                    except:
                        pass
            
            del chunk
        
        logger.info(f"Built target-to-disease mapping for {len(target_to_diseases)} targets")
        
        # Clean up associations memory immediately
        del assoc_df
        if 'ot_associations' in locals():
            del ot_associations
        gc.collect()
        log_memory_status("After building target-disease mapping")
        
        # STEP 2: Build target-to-pathway mapping with memory limits
        logger.info("Building target-pathway mapping (memory optimized)...")
        target_to_pathways = defaultdict(set)
        
        # Process target pathways in smaller chunks
        target_chunk_size = 2000  # Reduced chunk size
        total_target_rows = len(target_df)
        
        for i in range(0, total_target_rows, target_chunk_size):
            chunk = target_df.iloc[i:i + target_chunk_size]
            
            for idx, row in chunk.iterrows():
                target_id = row['id']
                pathways = row.get('pathways', [])
                
                if pathways and isinstance(pathways, list):
                    for pathway in pathways:
                        if isinstance(pathway, dict) and 'pathwayId' in pathway:
                            target_to_pathways[target_id].add(pathway['pathwayId'])
            
            # Memory management
            chunk_num = (i // target_chunk_size) + 1
            if chunk_num % 10 == 0:
                logger.info(f"  Processed {chunk_num} target chunks ({len(target_to_pathways)} targets with pathways so far)")
                gc.collect()
            
            del chunk
        
        logger.info(f"Built target-to-pathway mapping for {len(target_to_pathways)} targets")
        
        # Clean up target dataframe
        if 'target_df' in locals():
            del target_df
        gc.collect()
        log_memory_status("After building target-pathway mapping")
        
        # STEP 3: Generate pathway-disease edges with STRICT memory limits
        shared_targets = set(target_to_diseases.keys()) & set(target_to_pathways.keys())
        logger.info(f"Found {len(shared_targets)} targets with both disease and pathway associations")
        
        # CRITICAL: Limit edge generation to prevent memory overflow
        max_edges_limit = 5000000  # Increased limit for comprehensive coverage
        pathway_disease_edges = []
        edge_count = 0
        
        logger.info(f"Generating pathway→disease edges (limit: {max_edges_limit:,} edges)...")
        
        for i, target_id in enumerate(shared_targets):
            if edge_count >= max_edges_limit:
                logger.warning(f"Reached edge limit ({max_edges_limit:,}), stopping to prevent memory overflow")
                break
            
            target_diseases = target_to_diseases[target_id]
            target_pathways = target_to_pathways[target_id]
            
            # Skip targets with too many combinations to prevent memory explosion
            edge_combinations = len(target_diseases) * len(target_pathways)
            if edge_combinations > 10000:  # Skip targets that would create too many edges
                continue
            
            # Generate edges for this target (with limit)
            for pathway_id in target_pathways:
                for disease_id in target_diseases:
                    if edge_count >= max_edges_limit:
                        break
                    
                    pathway_disease_edges.append({
                        'source': pathway_id,
                        'target': disease_id,
                        'type': 'ASSOCIATED_WITH',
                        'via_target': target_id,
                        'source_type': 'Pathway',
                        'target_type': 'Disease',
                        'data_source': 'OpenTargets_Inferred'
                    })
                    edge_count += 1
                
                if edge_count >= max_edges_limit:
                    break
            
            # Progress logging every 1000 targets
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i+1:,}/{len(shared_targets):,} targets, {edge_count:,} edges generated")
                
                # Memory check every 2000 targets
                if PSUTIL_AVAILABLE and (i + 1) % 2000 == 0:
                    try:
                        import psutil
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 80:
                            logger.warning(f"Memory usage at {memory_percent:.1f}%, may need to stop early")
                            if memory_percent > 85:
                                logger.error(f"Critical memory usage ({memory_percent:.1f}%), stopping edge generation")
                                break
                    except:
                        pass
                
                # Periodic garbage collection for large edge lists
                if (i + 1) % 5000 == 0:
                    gc.collect()
        
        edges.extend(pathway_disease_edges)
        logger.info(f"✓ Created {len(pathway_disease_edges):,} pathway→disease edges (memory limited)")
        
        # Clean up memory
        del target_to_diseases, target_to_pathways, shared_targets, pathway_disease_edges
        gc.collect()
        log_memory_status("After pathway-disease edge generation")
        
except Exception as e:
    logger.error(f"Error in pathway→disease edge generation: {e}")
    logger.info("Continuing without pathway→disease edges...")
    gc.collect()

logger.info(f"Total edges created: {len(edges)}")
edges_df = pd.DataFrame(edges)
edges_output_path = os.path.join(OUTPUT_DIR, "edges.csv")
edges_df.to_csv(edges_output_path, index=False)
logger.info(f"✓ Saved edges to: {edges_output_path}")
log_progress("Edges creation completed", start_edges)

# Final summary
logger.info("\n" + "="*60)
logger.info("GENERATION COMPLETE")
logger.info("="*60)
total_duration = time.time() - start_total
logger.info(f"Total execution time: {total_duration:.2f} seconds")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Files generated:")
logger.info(f"  - nodes.csv: {len(nodes)} nodes")
logger.info(f"  - edges.csv: {len(edges)} edges")
print(f"\n🎉 CSVs generated successfully in {OUTPUT_DIR}")
print(f"📊 Summary: {len(nodes)} nodes, {len(edges)} edges")
print(f"⏱️  Total time: {total_duration:.2f} seconds")
