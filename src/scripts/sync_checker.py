import pandas as pd
import pickle
import os
import json
import logging
from collections import defaultdict
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, Set, List, Tuple
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_checker.log'),
        logging.StreamHandler()
    ]
)

# Configuration for chunking and memory management
CHUNK_SIZE = 10000  # Process data in chunks
MAX_WORKERS = min(4, os.cpu_count())  # Limit concurrent threads
MEMORY_THRESHOLD = 0.8  # Memory usage threshold (80%)

# Global status tracking
class GlobalStatusTracker:
    def __init__(self):
        self.current_operation = "Initializing"
        self.start_time = time.time()
        self.last_update = time.time()
        self.status_thread = None
        self.running = True
        
    def set_operation(self, operation):
        self.current_operation = operation
        self.last_update = time.time()
        logging.info(f"🔄 {operation}")
        
    def start_monitoring(self):
        def monitor():
            while self.running:
                time.sleep(15)  # Update every 15 seconds
                if self.running:
                    elapsed = time.time() - self.start_time
                    current_elapsed = time.time() - self.last_update
                    memory_usage = psutil.virtual_memory().percent
                    logging.info(f"⏱️  Status: {self.current_operation} | "
                               f"Total time: {timedelta(seconds=int(elapsed))} | "
                               f"Current operation: {timedelta(seconds=int(current_elapsed))} | "
                               f"Memory: {memory_usage:.1f}%")
        
        self.status_thread = threading.Thread(target=monitor, daemon=True)
        self.status_thread.start()
        
    def stop(self):
        self.running = False

# Initialize global status tracker
status_tracker = GlobalStatusTracker()
status_tracker.start_monitoring()

class ProgressTracker:
    """Track progress and provide ETA estimates"""
    
    def __init__(self, total_tasks: int, task_name: str = "Processing"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_name = task_name
        self.last_update = time.time()
        
    def update(self, increment: int = 1):
        """Update progress and log ETA"""
        self.completed_tasks += increment
        current_time = time.time()
        
        # Only log every 5 seconds or when completed
        if current_time - self.last_update >= 5 or self.completed_tasks >= self.total_tasks:
            self.last_update = current_time
            self._log_progress()
    
    def _log_progress(self):
        """Log current progress with ETA"""
        if self.completed_tasks == 0:
            return
            
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.completed_tasks / self.total_tasks) * 100
        
        if self.completed_tasks < self.total_tasks:
            estimated_total_time = elapsed_time * (self.total_tasks / self.completed_tasks)
            remaining_time = estimated_total_time - elapsed_time
            eta = datetime.now() + timedelta(seconds=remaining_time)
            
            logging.info(f"{self.task_name}: {self.completed_tasks}/{self.total_tasks} "
                        f"({progress_percent:.1f}%) - "
                        f"Elapsed: {timedelta(seconds=int(elapsed_time))} - "
                        f"ETA: {eta.strftime('%H:%M:%S')} "
                        f"(~{timedelta(seconds=int(remaining_time))} remaining)")
        else:
            logging.info(f"{self.task_name}: Complete! "
                        f"Total time: {timedelta(seconds=int(elapsed_time))}")

def check_memory_usage():
    """Check current memory usage and log warning if high"""
    memory_percent = psutil.virtual_memory().percent / 100
    if memory_percent > MEMORY_THRESHOLD:
        logging.warning(f"High memory usage: {memory_percent:.1%}")
        gc.collect()  # Force garbage collection
    return memory_percent

# Paths from your tree output (update if needed)
MAPPINGS_DIR = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings"
OPEN_TARGETS_DIR = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/open_targets_merged"
MESH_PICKLE = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/diseases/mesh_data_2025.pickle"
DRUGBANK_PICKLE = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/drugs/drugbank_parsed.pickle"

# Load mappings
def load_mapping(file_name):
    file_path = os.path.join(MAPPINGS_DIR, file_name)
    logging.info(f"Loading mapping file: {file_name}")
    return pd.read_csv(file_path)

def load_unmapped_file(file_name):
    """Load unmapped file with proper error handling for empty files"""
    file_path = os.path.join(MAPPINGS_DIR, file_name)
    
    if not os.path.exists(file_path):
        logging.warning(f"Unmapped file {file_name} does not exist")
        return []
    
    if os.path.getsize(file_path) == 0:
        logging.info(f"Unmapped file {file_name} is empty - no unmapped entities found")
        return []
    
    try:
        # Use chunked reading for large files
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB threshold
            logging.info(f"Large file detected ({file_size / (1024*1024):.1f}MB), using chunked reading")
            unmapped_list = []
            
            # Count total lines first for progress tracking
            with open(file_path, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            chunk_reader = pd.read_csv(file_path, header=None, chunksize=CHUNK_SIZE)
            progress_tracker = ProgressTracker(total_lines, f"Loading {file_name}")
            
            for chunk in tqdm(chunk_reader, desc=f"Loading {file_name}", unit="chunks"):
                unmapped_list.extend(chunk[0].tolist())
                progress_tracker.update(len(chunk))
                check_memory_usage()
        else:
            unmapped_list = pd.read_csv(file_path, header=None)[0].tolist()
        
        logging.info(f"Loaded {len(unmapped_list)} unmapped entities from {file_name}")
        return unmapped_list
    except Exception as e:
        logging.error(f"Error loading unmapped file {file_name}: {e}")
        return []

def load_json_streaming(file_path: str, max_size_mb: int = 100) -> Dict:
    """Load JSON file with streaming for large files"""
    start_time = time.time()
    try:
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        filename = os.path.basename(file_path)
        
        if file_size_mb > max_size_mb:
            logging.info(f"Large JSON file detected ({file_size_mb:.1f}MB), using memory-efficient loading")
            check_memory_usage()
        
        logging.info(f"📁 Starting to load: {filename} ({file_size_mb:.1f}MB)")
        
        # For very large files (>500MB), show periodic updates
        if file_size_mb > 500:
            logging.info(f"⚠️  This is a very large file ({file_size_mb:.1f}MB). Loading may take several minutes...")
            logging.info("🔄 Loading in progress... (this may appear to hang but it's working)")
            
            # Start a progress reporting thread for large files
            import threading
            
            progress_stop_flag = {'stop': False}
            
            def report_progress():
                while not progress_stop_flag['stop']:
                    time.sleep(10)  # Report every 10 seconds
                    if not progress_stop_flag['stop']:
                        elapsed = time.time() - start_time
                        logging.info(f"⏱️  Still loading {filename}... {elapsed:.0f}s elapsed")
            
            progress_thread = threading.Thread(target=report_progress, daemon=True)
            progress_thread.start()
        else:
            progress_stop_flag = None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Stop progress reporting for large files
        if file_size_mb > 500 and progress_stop_flag:
            progress_stop_flag['stop'] = True
        
        load_time = time.time() - start_time
        logging.info(f"✅ Loaded {filename}: {len(data)} entries ({file_size_mb:.1f}MB) in {load_time:.1f}s")
        
        # Log memory usage after loading large files
        if file_size_mb > 100:
            memory_usage = check_memory_usage()
            logging.info(f"💾 Memory usage after loading {filename}: {memory_usage:.1%}")
        
        return data
    except Exception as e:
        load_time = time.time() - start_time
        logging.error(f"❌ Error loading JSON from {file_path} after {load_time:.1f}s: {e}")
        return {}

logging.info("🚀 Starting sync checker script...")
script_start_time = time.time()
status_tracker.set_operation("Loading initial mappings")

disease_mapping = load_mapping("disease_mapping.csv")
target_mapping = load_mapping("target_mapping.csv")
unmapped_disease = load_unmapped_file("unmapped_disease.txt")
unmapped_target = load_unmapped_file("unmapped_target.txt")

# Load all UniProt mapping files
logging.info("📁 Loading UniProt mapping files...")
status_tracker.set_operation("Loading UniProt mapping files")
mapping_files = {
    'ensembl_to_uniprot_full': 'ensembl_to_uniprot_full.json',
    'ensembl_protein_to_uniprot': 'ensembl_protein_to_uniprot.json',
    'gene_name_to_uniprot': 'gene_name_to_uniprot.json',
    'refseq_to_uniprot': 'refseq_to_uniprot.json',
    'comprehensive_id_mappings': 'comprehensive_id_mappings.json'
}

def load_mapping_file(mapping_name: str, file_name: str) -> Tuple[str, Dict]:
    """Load a single mapping file (for parallel processing)"""
    file_path = os.path.join(MAPPINGS_DIR, file_name)
    if os.path.exists(file_path):
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logging.info(f"🔄 Starting {mapping_name} ({file_size_mb:.1f}MB)")
            
            mapping_data = load_json_streaming(file_path)
            logging.info(f"✅ Completed {mapping_name}: {len(mapping_data)} mappings")
            return mapping_name, mapping_data
        except Exception as e:
            logging.error(f"❌ Error loading {mapping_name}: {e}")
            return mapping_name, {}
    else:
        logging.warning(f"⚠️  Mapping file not found: {file_name}")
        return mapping_name, {}

# Load mapping files in parallel
uniprot_mappings = {}
loading_start_time = time.time()

# Sort files by size (smallest first) for better progress visibility
mapping_files_with_sizes = []
for mapping_name, file_name in mapping_files.items():
    file_path = os.path.join(MAPPINGS_DIR, file_name)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        mapping_files_with_sizes.append((mapping_name, file_name, file_size))
    else:
        mapping_files_with_sizes.append((mapping_name, file_name, 0))

# Sort by file size (smallest first)
mapping_files_with_sizes.sort(key=lambda x: x[2])

logging.info("📋 Loading order (smallest to largest):")
for mapping_name, file_name, file_size in mapping_files_with_sizes:
    size_mb = file_size / (1024 * 1024) if file_size > 0 else 0
    logging.info(f"   {mapping_name}: {size_mb:.1f}MB")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_mapping = {
        executor.submit(load_mapping_file, mapping_name, file_name): mapping_name 
        for mapping_name, file_name, _ in mapping_files_with_sizes
    }
    
    # Create progress bar for mapping file loading
    total_files = len(mapping_files_with_sizes)
    progress_bar = tqdm(total=total_files, desc="Loading mapping files", unit="files")
    completed_files = 0
    
    for future in as_completed(future_to_mapping):
        mapping_name, mapping_data = future.result()
        uniprot_mappings[mapping_name] = mapping_data
        completed_files += 1
        
        elapsed_time = time.time() - loading_start_time
        logging.info(f"📊 Progress: {completed_files}/{total_files} files loaded in {elapsed_time:.1f}s")
        
        progress_bar.update(1)
        check_memory_usage()
    
    progress_bar.close()

loading_time = time.time() - loading_start_time
logging.info(f"🎉 All mapping files loaded in {loading_time:.1f}s")

# Load Open Targets data (example for disease and target)
status_tracker.set_operation("Loading Open Targets data")
logging.info("📊 Loading Open Targets data...")
try:
    # Load parquet files with chunked reading if they're large
    disease_file_path = os.path.join(OPEN_TARGETS_DIR, "disease.parquet")
    target_file_path = os.path.join(OPEN_TARGETS_DIR, "target.parquet")
    
    disease_size = os.path.getsize(disease_file_path) / (1024 * 1024)
    target_size = os.path.getsize(target_file_path) / (1024 * 1024)
    
    logging.info(f"Disease file size: {disease_size:.1f}MB, Target file size: {target_size:.1f}MB")
    
    if disease_size > 100:  # 100MB threshold
        logging.info("Using chunked reading for disease data")
        ot_disease = pd.read_parquet(disease_file_path, engine='pyarrow')
    else:
        ot_disease = pd.read_parquet(disease_file_path)
    
    if target_size > 100:  # 100MB threshold
        logging.info("Using chunked reading for target data")
        ot_target = pd.read_parquet(target_file_path, engine='pyarrow')
    else:
        ot_target = pd.read_parquet(target_file_path)
    
    check_memory_usage()
    
except Exception as e:
    logging.error(f"Error loading Open Targets data: {e}")
    raise

# Load MeSH and DrugBank pickled data
status_tracker.set_operation("Loading MeSH and DrugBank data")
logging.info("🧬 Loading MeSH data...")
with open(MESH_PICKLE, 'rb') as f:
    mesh_data = pickle.load(f)

logging.info("💊 Loading DrugBank data...")
with open(DRUGBANK_PICKLE, 'rb') as f:
    drugbank_data = pickle.load(f)

logging.info(f"✅ Loaded {len(drugbank_data)} drugs from DrugBank")

# Function to check sync for diseases (Open Targets vs MeSH)
def check_disease_sync():
    logging.info("Checking disease sync between Open Targets and MeSH...")
    
    ot_diseases = set(ot_disease['id'])
    mesh_diseases = set(mesh_data.get('descriptors', {}).keys())  # Adjust based on your pickle structure
    
    mapped_ot = set(disease_mapping['index'])  # Assuming 'index' is original ID
    sync_ot_mesh = mapped_ot.intersection(mesh_diseases)
    
    coverage = len(sync_ot_mesh) / len(ot_diseases) * 100 if ot_diseases else 0
    
    logging.info(f"Disease Sync Results:")
    logging.info(f"  - Total OT diseases: {len(ot_diseases)}")
    logging.info(f"  - Total MeSH diseases: {len(mesh_diseases)}")
    logging.info(f"  - Mapped OT diseases: {len(mapped_ot)}")
    logging.info(f"  - Synchronized diseases: {len(sync_ot_mesh)}")
    logging.info(f"  - Coverage: {coverage:.2f}%")
    logging.info(f"  - Unmapped diseases: {len(unmapped_disease)}")
    
    print(f"Disease Sync: {len(sync_ot_mesh)} OT diseases mapped to MeSH ({coverage:.2f}%)")
    return coverage, len(unmapped_disease)

# Function to check sync for targets (Open Targets vs DrugBank vs UniProt JSON)
def check_target_sync():
    sync_start_time = time.time()
    logging.info("Checking target sync between Open Targets, DrugBank, and UniProt...")
    
    ot_targets = set(ot_target['id'])
    drugbank_targets = set()  # Extract from drugbank_data, e.g., targets in drug entries
    
    # Handle potential issues with drugbank_data structure
    try:
        logging.info("Processing DrugBank targets...")
        drug_count = 0
        total_drugs = len(drugbank_data)
        
        # Progress bar for DrugBank processing
        with tqdm(total=total_drugs, desc="Processing DrugBank drugs", unit="drugs") as pbar:
            for drug in drugbank_data:
                drug_count += 1
                pbar.update(1)
                
                if drug_count % 1000 == 0:
                    check_memory_usage()
                
                for target in drug.get('targets', []):
                    uniprot_id = target.get('uniprot_id')
                    if uniprot_id:
                        drugbank_targets.add(uniprot_id)
        
        drugbank_time = time.time() - sync_start_time
        logging.info(f"DrugBank processing completed in {drugbank_time:.1f}s")
        
    except Exception as e:
        logging.error(f"Error processing DrugBank targets: {e}")
        drugbank_targets = set()
    
    mapped_ot = set(target_mapping['index'])
    sync_ot_drugbank = mapped_ot.intersection(drugbank_targets)
    
    # Check coverage across all UniProt mapping files using parallel processing
    def check_mapping_coverage(mapping_item: Tuple[str, Dict]) -> Tuple[str, int, Set]:
        mapping_name, mapping_data = mapping_item
        mapped_in_this_file = set(mapping_data.keys()) & ot_targets
        return mapping_name, len(mapped_in_this_file), mapped_in_this_file
    
    mapping_coverage = {}
    total_mapped_targets = set()
    
    mapping_start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(check_mapping_coverage, (name, data))
            for name, data in uniprot_mappings.items()
        ]
        
        # Progress bar for mapping coverage check
        with tqdm(total=len(futures), desc="Checking mapping coverage", unit="mappings") as pbar:
            for future in as_completed(futures):
                mapping_name, count, mapped_targets = future.result()
                mapping_coverage[mapping_name] = count
                total_mapped_targets.update(mapped_targets)
                logging.info(f"  - {mapping_name}: {count} targets covered")
                pbar.update(1)
                check_memory_usage()
    
    mapping_time = time.time() - mapping_start_time
    logging.info(f"Mapping coverage check completed in {mapping_time:.1f}s")
    
    total_json_covered = len(total_mapped_targets)
    
    coverage = len(sync_ot_drugbank) / len(ot_targets) * 100 if ot_targets else 0
    json_coverage = total_json_covered / len(ot_targets) * 100 if ot_targets else 0
    
    total_sync_time = time.time() - sync_start_time
    
    logging.info(f"Target Sync Results (completed in {total_sync_time:.1f}s):")
    logging.info(f"  - Total OT targets: {len(ot_targets)}")
    logging.info(f"  - Total DrugBank targets: {len(drugbank_targets)}")
    logging.info(f"  - Mapped OT targets: {len(mapped_ot)}")
    logging.info(f"  - Synchronized targets: {len(sync_ot_drugbank)}")
    logging.info(f"  - Coverage: {coverage:.2f}%")
    logging.info(f"  - Total UniProt JSON covered: {total_json_covered} ({json_coverage:.2f}%)")
    logging.info(f"  - Unmapped targets: {len(unmapped_target)}")
    
    # Log detailed mapping coverage
    for mapping_name, count in mapping_coverage.items():
        if count > 0:
            logging.info(f"    * {mapping_name}: {count} targets")
    
    print(f"Target Sync: {len(sync_ot_drugbank)} OT targets mapped to DrugBank ({coverage:.2f}%)")
    print(f"UniProt Coverage: {total_json_covered} targets covered across all mapping files ({json_coverage:.2f}%)")
    return coverage, len(unmapped_target)

# Run checks and generate report
try:
    status_tracker.set_operation("Running disease sync check")
    logging.info("🔍 Running disease sync check...")
    disease_coverage, unmapped_d = check_disease_sync()
    
    status_tracker.set_operation("Running target sync check")
    logging.info("🎯 Running target sync check...")
    target_coverage, unmapped_t = check_target_sync()
    
    # Generate comprehensive report
    status_tracker.set_operation("Generating comprehensive reports")
    logging.info("📈 Generating comprehensive sync report...")
    
    # Basic report
    report = pd.DataFrame({
        'Entity': ['Disease', 'Target'],
        'Coverage (%)': [disease_coverage, target_coverage],
        'Unmapped Count': [unmapped_d, unmapped_t]
    })
    
    # Detailed mapping coverage report
    ot_targets = set(ot_target['id'])
    mapping_details = []
    
    def process_mapping_detail(mapping_item: Tuple[str, Dict]) -> Dict:
        mapping_name, mapping_data = mapping_item
        mapped_count = len(set(mapping_data.keys()) & ot_targets)
        coverage_pct = mapped_count / len(ot_targets) * 100 if ot_targets else 0
        return {
            'Mapping_File': mapping_name,
            'Total_Mappings': len(mapping_data),
            'OT_Targets_Covered': mapped_count,
            'Coverage_Percentage': round(coverage_pct, 2)
        }
    
    # Process mapping details in parallel
    report_start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_mapping_detail, (name, data))
            for name, data in uniprot_mappings.items()
        ]
        
        # Progress bar for report generation
        with tqdm(total=len(futures), desc="Generating detailed report", unit="mappings") as pbar:
            for future in as_completed(futures):
                mapping_details.append(future.result())
                pbar.update(1)
                check_memory_usage()
    
    report_time = time.time() - report_start_time
    logging.info(f"Detailed report generated in {report_time:.1f}s")
    
    mapping_report = pd.DataFrame(mapping_details)
    
    # Save reports
    report_path = os.path.join(MAPPINGS_DIR, "sync_report.csv")
    mapping_report_path = os.path.join(MAPPINGS_DIR, "mapping_coverage_report.csv")
    
    report.to_csv(report_path, index=False)
    mapping_report.to_csv(mapping_report_path, index=False)
    
    logging.info(f"Basic sync report saved to: {report_path}")
    logging.info(f"Detailed mapping coverage report saved to: {mapping_report_path}")
    
    print("\nBasic Sync Report:")
    print(report)
    print("\nDetailed Mapping Coverage Report:")
    print(mapping_report)
    
    # Clean up memory
    status_tracker.set_operation("Cleaning up and finalizing")
    logging.info("🧹 Cleaning up memory...")
    del uniprot_mappings
    gc.collect()
    
    final_memory = check_memory_usage()
    total_script_time = time.time() - script_start_time
    
    # Stop the status tracker
    status_tracker.stop()
    
    logging.info(f"💾 Final memory usage: {final_memory:.1%}")
    logging.info(f"⏱️  Total script execution time: {timedelta(seconds=int(total_script_time))}")
    logging.info("🎉 Sync checker script completed successfully!")
    
    print(f"\n🎉 Script completed in {timedelta(seconds=int(total_script_time))}")
    
except Exception as e:
    total_script_time = time.time() - script_start_time
    status_tracker.stop()
    logging.error(f"❌ Error during sync check execution after {timedelta(seconds=int(total_script_time))}: {e}")
    print(f"Script failed with error after {timedelta(seconds=int(total_script_time))}: {e}")
    # Clean up on error
    gc.collect()
    raise
