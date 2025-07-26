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

logger.info(f"Total nodes created: {len(nodes)} (before deduplication)")

# Deduplicate nodes - merge entries with same ID but different information
logger.info("Deduplicating nodes and merging entries with same ID...")
start_dedup = time.time()

# Group nodes by ID to handle duplicates
nodes_by_id = defaultdict(list)
for node in nodes:
    nodes_by_id[node['id']].append(node)

# Merge duplicate entries for the same ID
deduplicated_nodes = []
duplicates_found = 0
entries_merged = 0

for node_id, node_list in nodes_by_id.items():
    if len(node_list) == 1:
        # No duplicates, keep as is
        deduplicated_nodes.append(node_list[0])
    else:
        # Multiple entries for same ID - merge them
        duplicates_found += 1
        entries_merged += len(node_list)
        
        # Start with the first entry as base
        merged_node = node_list[0].copy()
        
        # Merge information from other entries, preferring non-None/non-empty values
        for other_node in node_list[1:]:
            for key, value in other_node.items():
                if key == 'id' or key == 'type':
                    continue  # Keep original ID and type
                
                current_value = merged_node.get(key)
                
                # Use the more informative value (non-None, non-empty, longer text)
                if value and str(value).strip() and str(value) != 'None':
                    if not current_value or str(current_value).strip() == '' or str(current_value) == 'None':
                        merged_node[key] = value
                    elif isinstance(value, str) and isinstance(current_value, str):
                        # For string fields, keep the longer/more detailed one
                        if len(str(value).strip()) > len(str(current_value).strip()):
                            merged_node[key] = value
        
        deduplicated_nodes.append(merged_node)

# Replace original nodes list with deduplicated version
nodes = deduplicated_nodes

logger.info(f"✓ Deduplication completed:")
logger.info(f"  - Found {duplicates_found:,} IDs with duplicates")
logger.info(f"  - Merged {entries_merged:,} entries into {duplicates_found:,} unique nodes")
logger.info(f"  - Final node count: {len(nodes):,}")
log_progress("Node deduplication completed", start_dedup)

# Build ID to group mapping for Neo4j admin import
logger.info("Building ID-to-group mapping for Neo4j admin import...")
id_to_group = {}
for node in nodes:
    id_to_group[node['id']] = node['type']
logger.info(f"✓ Built ID-to-group mapping with {len(id_to_group)} entries")

# Create nodes CSVs with proper Neo4j admin import formatting
logger.info("Creating Neo4j admin import formatted node files...")
nodes_by_type = defaultdict(list)

# Group all nodes by type
for node in nodes:
    node_type = node['type']
    nodes_by_type[node_type].append(node)

# Save separate CSV for each type with import headers
for node_type, type_nodes in nodes_by_type.items():
    df = pd.DataFrame(type_nodes)
    
    # Convert unhashable types (lists, dicts) to strings before deduplication
    logger.info(f"Processing {node_type} nodes for deduplication...")
    
    # Create a copy for deduplication with all columns as strings
    df_for_dedup = df.copy()
    for col in df_for_dedup.columns:
        def safe_str_convert(x):
            try:
                if x is None:
                    return ''
                elif pd.isna(x):
                    return ''
                elif isinstance(x, (list, tuple)):
                    return ','.join(map(str, x))
                elif hasattr(x, '__iter__') and not isinstance(x, str):
                    # Handle numpy arrays and other iterables
                    try:
                        return ','.join(map(str, x))
                    except:
                        return str(x)
                else:
                    return str(x)
            except:
                return str(x)
        
        df_for_dedup[col] = df_for_dedup[col].apply(safe_str_convert)
    
    # Deduplicate exactly identical rows within each type
    initial_count = len(df_for_dedup)
    dedup_indices = df_for_dedup.drop_duplicates().index
    df = df.loc[dedup_indices]
    final_count = len(df)
    
    if initial_count > final_count:
        logger.info(f"  Removed {initial_count - final_count:,} exactly identical {node_type} rows")
    
    # Clean multiline fields (replace newlines with spaces)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Handle lists by converting to comma-separated strings
            df[col] = df[col].apply(lambda x: 
                ','.join(map(str, x)) if isinstance(x, list) else str(x)
            ).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).fillna('')
        else:
            df[col] = df[col].fillna('')
    
    # Add group suffix to node IDs for Neo4j admin import consistency
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str) + f"({node_type})"
    
    # Add required import headers - rename existing columns
    rename_dict = {'id': 'id:ID', 'name': 'name:string'}
    
    # Only rename columns that exist
    if 'description' in df.columns:
        rename_dict['description'] = 'description:string'
    if 'data_source' in df.columns:
        rename_dict['data_source'] = 'data_source:string'
    
    df = df.rename(columns=rename_dict)
    
    # Add label column
    df[':LABEL'] = node_type
    
    # Ensure required columns exist
    if 'name:string' not in df.columns:
        df['name:string'] = ''
    if 'description:string' not in df.columns:
        df['description:string'] = ''
    if 'data_source:string' not in df.columns:
        df['data_source:string'] = ''
    
    # Select relevant columns for each node type
    base_columns = ['id:ID', ':LABEL', 'name:string', 'data_source:string']
    
    # Add type-specific columns
    if node_type == 'Drug':
        extra_columns = ['indication:string', 'mechanism_of_action:string', 'synonyms:string', 
                        'canonical_smiles:string', 'molecular_weight:string', 'state:string']
    elif node_type == 'Disease':
        extra_columns = ['mesh_description:string', 'mesh_terms:string']
    elif node_type == 'Target':
        extra_columns = ['symbol:string', 'proteinIds:string']
    elif node_type == 'Pathway':
        extra_columns = ['description:string', 'source:string']
    else:
        extra_columns = []
    
    # Ensure all extra columns exist and rename appropriately
    for col in extra_columns:
        col_name = col.replace(':string', '')
        if col_name in df.columns:
            df = df.rename(columns={col_name: col})
        else:
            df[col] = ''
    
    # Final column selection - only include columns that actually exist
    final_columns = []
    for col in base_columns + extra_columns:
        if col in df.columns:
            final_columns.append(col)
    
    output_path = os.path.join(OUTPUT_DIR, f"nodes_{node_type.lower()}.csv")
    df[final_columns].to_csv(output_path, index=False)
    
    logger.info(f"✓ Saved {len(df):,} {node_type} nodes to {output_path}")

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

logger.info(f"Total edges created: {len(edges)} (before deduplication)")

# Deduplicate edges - remove exactly identical edges
logger.info("Deduplicating edges...")
start_edge_dedup = time.time()

edges_df = pd.DataFrame(edges)

# Count before deduplication
initial_edge_count = len(edges_df)

# Convert any unhashable types to strings for deduplication
for col in ['evidence']:  # Known columns that might have dicts/lists
    if col in edges_df.columns:
        edges_df[col] = edges_df[col].apply(lambda x: str(x) if not pd.isna(x) else '')

# Remove exactly duplicate edges based on source, target, and type
edges_df = edges_df.drop_duplicates(subset=['source', 'target', 'type'])

final_edge_count = len(edges_df)
if initial_edge_count > final_edge_count:
    logger.info(f"✓ Removed {initial_edge_count - final_edge_count:,} duplicate edges")
    logger.info(f"✓ Final edge count: {final_edge_count:,}")

# Remove orphaned edges - STREAMING VERSION to prevent memory overload
logger.info("Removing orphaned edges (edges referencing missing nodes) - STREAMING...")
start_orphan_removal = time.time()

# Get all valid node IDs from our deduplicated nodes
valid_node_ids = set(id_to_group.keys())
logger.info(f"Valid node IDs: {len(valid_node_ids):,}")

# STREAMING APPROACH: Process edges in chunks to prevent memory issues
chunk_size = 50000  # Process 50k edges at a time
total_edges = len(edges_df)
valid_edges = []
orphaned_count = 0

logger.info(f"Processing {total_edges:,} edges in chunks of {chunk_size:,}...")

for i in range(0, total_edges, chunk_size):
    end_idx = min(i + chunk_size, total_edges)
    chunk = edges_df.iloc[i:end_idx].copy()
    
    # Filter valid edges in this chunk
    valid_mask = (chunk['source'].isin(valid_node_ids)) & (chunk['target'].isin(valid_node_ids))
    valid_chunk = chunk[valid_mask]
    orphaned_in_chunk = len(chunk) - len(valid_chunk)
    
    valid_edges.append(valid_chunk)
    orphaned_count += orphaned_in_chunk
    
    # Progress logging
    chunk_num = (i // chunk_size) + 1
    total_chunks = (total_edges + chunk_size - 1) // chunk_size
    if chunk_num % 10 == 0 or chunk_num == total_chunks:
        logger.info(f"  Processed chunk {chunk_num}/{total_chunks} (removed {orphaned_count:,} orphaned edges so far)")
    
    # Memory cleanup
    del chunk
    if chunk_num % 20 == 0:
        gc.collect()

# Combine all valid edges
logger.info("Combining valid edge chunks...")
edges_df = pd.concat(valid_edges, ignore_index=True)
del valid_edges
gc.collect()

logger.info(f"✓ Removed {orphaned_count:,} orphaned edges referencing missing nodes")
logger.info(f"✓ Final edge count after orphan removal: {len(edges_df):,}")
log_progress("Orphaned edge removal completed", start_orphan_removal)

# Create edges CSV with proper Neo4j admin import formatting
logger.info("Creating Neo4j admin import formatted edges file...")
# edges_df already created above

# Clean multiline fields
for col in edges_df.columns:
    if edges_df[col].dtype == 'object':
        edges_df[col] = edges_df[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).fillna('')
    else:
        edges_df[col] = edges_df[col].fillna('')

# Add group suffixes to IDs using the mapping we built - STREAMING VERSION
logger.info("Adding group suffixes to edge IDs (streaming)...")
start_suffix = time.time()

missing_source = 0
missing_target = 0
total_edges = len(edges_df)
chunk_size = 50000  # Process 50k edges at a time

logger.info(f"Processing {total_edges:,} edges for group suffixes in chunks of {chunk_size:,}...")

for i in range(0, total_edges, chunk_size):
    end_idx = min(i + chunk_size, total_edges)
    
    # Process chunk
    for idx in range(i, end_idx):
        source_id = edges_df.iloc[idx]['source']
        target_id = edges_df.iloc[idx]['target']
        
        source_group = id_to_group.get(source_id)
        if source_group:
            edges_df.iloc[idx, edges_df.columns.get_loc('source')] = f"{source_id}({source_group})"
        else:
            missing_source += 1
            
        target_group = id_to_group.get(target_id)
        if target_group:
            edges_df.iloc[idx, edges_df.columns.get_loc('target')] = f"{target_id}({target_group})"
        else:
            missing_target += 1
    
    # Progress logging
    chunk_num = (i // chunk_size) + 1
    total_chunks = (total_edges + chunk_size - 1) // chunk_size
    if chunk_num % 10 == 0 or chunk_num == total_chunks:
        logger.info(f"  Processed chunk {chunk_num}/{total_chunks}")
    
    # Memory cleanup every 20 chunks
    if chunk_num % 20 == 0:
        gc.collect()

logger.info(f"✓ Added group suffixes (missing sources: {missing_source}, missing targets: {missing_target})")
log_progress("Group suffix addition completed", start_suffix)

# Remove edges with missing group information (orphaned edges)
initial_count = len(edges_df)
edges_df = edges_df[edges_df['source'].str.contains(r'\(', regex=True, na=False)]
edges_df = edges_df[edges_df['target'].str.contains(r'\(', regex=True, na=False)]
final_count = len(edges_df)

if initial_count > final_count:
    logger.info(f"Removed {initial_count - final_count:,} orphaned edges")

# Rename for Neo4j admin import format
edges_df = edges_df.rename(columns={
    'source': ':START_ID',
    'target': ':END_ID',
    'type': ':TYPE',
    'data_source': 'data_source:string',
    'evidence': 'evidence:string',
    'target_name': 'target_name:string',
    'actions': 'actions:string',
    'organism': 'organism:string'
})

# Select relevant columns for edges
columns = [':START_ID', ':END_ID', ':TYPE', 'data_source:string', 'evidence:string']

# Add optional columns if they exist
optional_columns = ['target_name:string', 'actions:string', 'organism:string', 
                   'max_phase:string', 'efo_name:string', 'pathwayName:string', 'via_target:string']
for col in optional_columns:
    if col in edges_df.columns:
        columns.append(col)

edges_output_path = os.path.join(OUTPUT_DIR, "edges.csv")
edges_df[columns].to_csv(edges_output_path, index=False)

logger.info(f"✓ Saved {len(edges_df):,} edges to {edges_output_path}")
log_progress("Edges creation completed", start_edges)

# Final summary
logger.info("\n" + "="*60)
logger.info("GENERATION COMPLETE - NEO4J ADMIN IMPORT FORMAT")
logger.info("="*60)
total_duration = time.time() - start_total
logger.info(f"Total execution time: {total_duration:.2f} seconds")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Files generated for Neo4j admin import:")

# Count nodes by type
node_counts = {}
for node_type in nodes_by_type.keys():
    node_counts[node_type] = len(nodes_by_type[node_type])
    logger.info(f"  - nodes_{node_type.lower()}.csv: {node_counts[node_type]:,} nodes")

logger.info(f"  - edges.csv: {len(edges_df):,} edges")
logger.info(f"Total: {sum(node_counts.values()):,} nodes, {len(edges_df):,} edges")

print(f"\n🎉 Neo4j admin import CSVs generated successfully!")
print(f"📊 Summary: {sum(node_counts.values()):,} nodes, {len(edges_df):,} edges")
print(f"⏱️  Total time: {total_duration:.2f} seconds")
print(f"\n🚀 To import into Neo4j:")
print(f"sudo systemctl stop neo4j")
print(f"sudo rm -rf /var/lib/neo4j/data/databases/neo4j/*")
print(f"sudo cp {OUTPUT_DIR}/* /var/lib/neo4j/import/")
print(f"sudo chown neo4j:neo4j /var/lib/neo4j/import/*")

# Generate the import command
import_cmd = "sudo neo4j-admin database import full neo4j \\\n"
for node_type in sorted(node_counts.keys()):
    import_cmd += f"  --nodes={node_type}=/var/lib/neo4j/import/nodes_{node_type.lower()}.csv \\\n"
import_cmd += f"  --relationships=/var/lib/neo4j/import/edges.csv \\\n"
import_cmd += f"  --overwrite-destination=true"

print(import_cmd)
print(f"sudo systemctl start neo4j")
print(f"sudo neo4j-admin database import full neo4j   --nodes=Disease=/var/lib/neo4j/import/nodes_disease.csv   --nodes=Drug=/var/lib/neo4j/import/nodes_drug.csv   --nodes=Pathway=/var/lib/neo4j/import/nodes_pathway.csv   --nodes=Target=/var/lib/neo4j/import/nodes_target.csv   --relationships=/var/lib/neo4j/import/edges.csv   --overwrite-destination=true   --verbose")