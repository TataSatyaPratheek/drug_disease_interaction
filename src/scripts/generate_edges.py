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
        logging.FileHandler('/home/vi/Documents/drug_disease_interaction/logs/edges_generation.log'),
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
logger.info("STARTING COMPREHENSIVE DRUG-DISEASE INTERACTION EDGES GENERATION")
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

logger.info(f"Output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load mappings (lighter version - only what's needed for edges)
logger.info("\n" + "="*40)
logger.info("LOADING ESSENTIAL MAPPING FILES")
logger.info("="*40)

start_mappings = time.time()
logger.info("Loading disease mappings...")
disease_mapping = pd.read_csv(DISEASE_MAPPING_CSV)
logger.info(f"✓ Loaded disease mapping: {disease_mapping.shape}")

logger.info("Loading target mappings...")
target_mapping = pd.read_csv(TARGET_MAPPING_CSV)
logger.info(f"✓ Loaded target mapping: {target_mapping.shape}")

disease_map = dict(zip(disease_mapping['index'], disease_mapping['0']))
target_map = dict(zip(target_mapping['index'], target_mapping['0']))
logger.info(f"✓ Created disease map: {len(disease_map)} entries")
logger.info(f"✓ Created target map: {len(target_map)} entries")
log_progress("Mappings loaded", start_mappings)
log_memory_status("After loading mappings")
gc.collect()

# Load and process data with cuDF
logger.info("\n" + "="*40)
logger.info("LOADING PARQUET FILES FOR EDGES")
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
                        # For edges, continue processing even at high memory
                        if memory_percent > 95:
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
ot_known_drug = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "known_drug.parquet"))
ot_associations = load_parquet_gpu(os.path.join(OPEN_TARGETS_DIR, "association_overall_direct.parquet"))  # For pathway-disease inference

# Load drug enrichment files
drug_indication_df = None
drug_mechanism_df = None

try:
    drug_indication_path = os.path.join(OPEN_TARGETS_DIR, "drug_indication.parquet")
    drug_mechanism_path = os.path.join(OPEN_TARGETS_DIR, "drug_mechanism_of_action.parquet")
    
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
        
except Exception as e:
    logger.warning(f"Error loading drug enrichment files: {e}")

log_progress("All parquet files loaded", start_parquet)

# Load DrugBank data
logger.info("\n" + "="*40)
logger.info("LOADING DRUGBANK DATA")
logger.info("="*40)

start_pickle = time.time()
logger.info("Loading DrugBank data...")
with open(DRUGBANK_PICKLE, 'rb') as f:
    drugbank_data = pickle.load(f)
logger.info(f"✓ Loaded DrugBank data: {len(drugbank_data)} entries")
log_progress("DrugBank data loaded", start_pickle)

# Create edges CSV (drug→target→pathway→disease)
logger.info("\n" + "="*40)
logger.info("CREATING COMPREHENSIVE EDGES")
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
                'target_type': 'Target',
                'data_source': 'OpenTargets'
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
if ot_target is not None:
    try:
        target_df = ot_target.to_pandas()
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
                                'target_type': 'Pathway',
                                'data_source': 'OpenTargets'
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
        
        # Keep target_df for pathway-disease inference
        log_memory_status("After target→pathway processing")
        
    except Exception as e:
        logger.warning(f"Could not process pathway data from target.parquet: {e}")
        logger.info("Continuing without pathway edges...")
        target_df = None
else:
    logger.warning("✗ No target data available for pathways")
    target_df = None

# Pathway → Disease (inferred via shared targets) - NO MEMORY LIMITS (COMPREHENSIVE)
logger.info("Processing pathway→disease edges (COMPREHENSIVE - no memory limits)...")

try:
    # Check if required data is available
    if ot_associations is None:
        logger.warning("✗ Missing association data - cannot infer pathway→disease edges")
    elif target_df is None:
        logger.warning("✗ Missing target data - cannot infer pathway→disease edges")
    else:
        log_memory_status("Before pathway-disease inference")
        
        # STEP 1: Build target-to-disease mapping
        logger.info("Building comprehensive target-disease mapping...")
        assoc_df = ot_associations.to_pandas()
        target_to_diseases = defaultdict(set)
        
        # Process ALL associations (no memory limits)
        logger.info(f"Processing all {len(assoc_df)} association records...")
        for idx, row in assoc_df.iterrows():
            target_id = row['targetId']
            disease_id = disease_map.get(row['diseaseId'], row['diseaseId'])
            target_to_diseases[target_id].add(disease_id)
            
            # Progress logging every 500K records
            if (idx + 1) % 500000 == 0:
                logger.info(f"  Processed {idx+1:,} associations, {len(target_to_diseases)} targets so far")
                gc.collect()  # Periodic cleanup
        
        logger.info(f"Built comprehensive target-to-disease mapping for {len(target_to_diseases)} targets")
        
        # Clean up associations memory
        del assoc_df, ot_associations
        gc.collect()
        log_memory_status("After building target-disease mapping")
        
        # STEP 2: Build target-to-pathway mapping
        logger.info("Building comprehensive target-pathway mapping...")
        target_to_pathways = defaultdict(set)
        
        logger.info(f"Processing all {len(target_df)} target records...")
        for idx, row in target_df.iterrows():
            target_id = row['id']
            pathways = row.get('pathways', [])
            
            if pathways and isinstance(pathways, list):
                for pathway in pathways:
                    if isinstance(pathway, dict) and 'pathwayId' in pathway:
                        target_to_pathways[target_id].add(pathway['pathwayId'])
            
            # Progress logging every 10K records
            if (idx + 1) % 10000 == 0:
                logger.info(f"  Processed {idx+1:,} targets, {len(target_to_pathways)} with pathways so far")
        
        logger.info(f"Built comprehensive target-to-pathway mapping for {len(target_to_pathways)} targets")
        
        # Clean up target dataframe
        del target_df, ot_target
        gc.collect()
        log_memory_status("After building target-pathway mapping")
        
        # STEP 3: Generate ALL pathway-disease edges (comprehensive)
        shared_targets = set(target_to_diseases.keys()) & set(target_to_pathways.keys())
        logger.info(f"Found {len(shared_targets)} targets with both disease and pathway associations")
        
        pathway_disease_edges = []
        edge_count = 0
        skipped_targets = 0
        
        logger.info(f"Generating ALL possible pathway→disease edges (comprehensive analysis)...")
        
        for i, target_id in enumerate(shared_targets):
            target_diseases = target_to_diseases[target_id]
            target_pathways = target_to_pathways[target_id]
            
            # Calculate potential edges for this target
            potential_edges = len(target_diseases) * len(target_pathways)
            
            # Skip targets that would create excessive edges (>50K per target)
            if potential_edges > 50000:
                skipped_targets += 1
                if skipped_targets <= 10:  # Log first 10 skips
                    logger.info(f"  Skipping target {target_id}: would create {potential_edges:,} edges")
                continue
            
            # Generate all edges for this target
            for pathway_id in target_pathways:
                for disease_id in target_diseases:
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
            
            # Progress logging every 1000 targets
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i+1:,}/{len(shared_targets):,} targets, {edge_count:,} edges generated")
                
                # Memory monitoring every 2000 targets
                if PSUTIL_AVAILABLE and (i + 1) % 2000 == 0:
                    try:
                        memory_percent = psutil.virtual_memory().percent
                        logger.info(f"    Memory usage: {memory_percent:.1f}%")
                        if memory_percent > 90:
                            logger.warning(f"High memory usage ({memory_percent:.1f}%), but continuing comprehensive processing")
                    except:
                        pass
                
                # Garbage collection every 5000 targets
                if (i + 1) % 5000 == 0:
                    gc.collect()
        
        if skipped_targets > 10:
            logger.info(f"  Skipped {skipped_targets} total targets with >50K potential edges each")
        
        edges.extend(pathway_disease_edges)
        logger.info(f"✓ Created {len(pathway_disease_edges):,} comprehensive pathway→disease edges")
        
        # Clean up memory
        del target_to_diseases, target_to_pathways, shared_targets, pathway_disease_edges
        gc.collect()
        log_memory_status("After comprehensive pathway-disease edge generation")
        
except Exception as e:
    logger.error(f"Error in pathway→disease edge generation: {e}")
    logger.info("Continuing without pathway→disease edges...")
    gc.collect()

logger.info(f"Total edges created: {len(edges):,}")
log_memory_status("Before saving edges to CSV")

# Save edges to CSV
edges_df = pd.DataFrame(edges)
edges_output_path = os.path.join(OUTPUT_DIR, "edges.csv")
edges_df.to_csv(edges_output_path, index=False)
logger.info(f"✓ Saved edges to: {edges_output_path}")
log_progress("Edges creation completed", start_edges)

# Final summary
logger.info("\n" + "="*60)
logger.info("EDGES GENERATION COMPLETE")
logger.info("="*60)
total_duration = time.time() - start_total
logger.info(f"Total execution time: {total_duration:.2f} seconds")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Files generated:")
logger.info(f"  - edges.csv: {len(edges):,} edges")
print(f"\n🎉 Edges CSV generated successfully in {OUTPUT_DIR}")
print(f"📊 Summary: {len(edges):,} edges")
print(f"⏱️  Total time: {total_duration:.2f} seconds")
