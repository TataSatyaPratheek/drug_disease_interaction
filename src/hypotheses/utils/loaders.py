# src/hypotheses/utils/loaders.py
import pandas as pd
from pathlib import Path
import gzip
from rich.console import Console

# --- Configuration (Using absolute paths to avoid any path issues) ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Go up to project root
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "additional"
PROCESSED_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "hypotheses_cache"
PROCESSED_CACHE_PATH.mkdir(parents=True, exist_ok=True)
CONSOLE = Console()

def load_aact_data() -> pd.DataFrame:
    """
    Loads and processes the AACT database dump to get a clean
    list of (drug_name, condition_name) pairs from relevant clinical trials.
    """
    cache_file = PROCESSED_CACHE_PATH / "aact_drug_disease_pairs.pickle"
    
    CONSOLE.print(f"[blue]Debug: Checking cache at {cache_file.absolute()}[/blue]")
    
    if cache_file.exists():
        CONSOLE.print(f"Loading cached AACT data from {cache_file}...")
        try:
            df = pd.read_pickle(cache_file)
            CONSOLE.print(f"[green]Successfully loaded {len(df):,} cached trial pairs.[/green]")
            return df
        except Exception as e:
            CONSOLE.print(f"[red]Error loading cache file: {e}. Regenerating...[/red]")
    
    CONSOLE.print("Processing AACT database dump (first time only, can take a few minutes)...")
    aact_dir = RAW_DATA_PATH / "acct-database"
    
    if not aact_dir.exists():
        CONSOLE.print(f"[bold red]Error: AACT directory not found at {aact_dir}.[/bold red]")
        return pd.DataFrame()

    try:
        CONSOLE.print("Loading AACT tables...")
        studies = pd.read_csv(aact_dir / "studies.txt", sep="|", 
                            usecols=['nct_id', 'phase', 'study_type'], 
                            on_bad_lines='warn', low_memory=False)
        CONSOLE.print(f"Loaded {len(studies):,} studies")
        
        interventions = pd.read_csv(aact_dir / "interventions.txt", sep="|", 
                                  usecols=['nct_id', 'name', 'intervention_type'], 
                                  on_bad_lines='warn', low_memory=False)
        CONSOLE.print(f"Loaded {len(interventions):,} interventions")
        
        conditions = pd.read_csv(aact_dir / "conditions.txt", sep="|", 
                               usecols=['nct_id', 'name'], 
                               on_bad_lines='warn', low_memory=False)
        CONSOLE.print(f"Loaded {len(conditions):,} conditions")
        
    except Exception as e:
        CONSOLE.print(f"[bold red]Error reading AACT files: {e}[/bold red]")
        return pd.DataFrame()

    # Join tables first, then filter
    CONSOLE.print("Joining tables...")
    merged = pd.merge(studies, interventions, on='nct_id', how='inner')
    CONSOLE.print(f"After studies-interventions join: {len(merged):,} rows")
    
    merged = pd.merge(merged, conditions, on='nct_id', how='inner')
    CONSOLE.print(f"After adding conditions: {len(merged):,} rows")

    # Apply filters with correct case-sensitive values
    CONSOLE.print("Applying filters...")
    
    # Fix 1: Use correct case for study_type
    interventional = merged[merged['study_type'] == 'INTERVENTIONAL']
    CONSOLE.print(f"Interventional studies: {len(interventional):,}")
    
    with_phase = interventional.dropna(subset=['phase'])
    CONSOLE.print(f"With phase info: {len(with_phase):,}")
    
    # Fix 2: Use correct phase format (PHASE2, PHASE3, PHASE4)
    late_phase = with_phase[with_phase['phase'].str.contains('PHASE2|PHASE3|PHASE4', regex=True, na=False)]
    CONSOLE.print(f"Phase 2+ studies: {len(late_phase):,}")
    
    # Fix 3: Use correct case for intervention_type
    drug_trials = late_phase[late_phase['intervention_type'].isin(['DRUG', 'BIOLOGICAL'])]
    CONSOLE.print(f"Drug/Biological trials: {len(drug_trials):,}")
    
    # Extract drug-condition pairs
    final_pairs = drug_trials[['name_x', 'name_y']].rename(columns={'name_x': 'drug_name', 'name_y': 'condition_name'})
    final_pairs = final_pairs.dropna().drop_duplicates().reset_index(drop=True)

    CONSOLE.print(f"Final unique drug-condition pairs: {len(final_pairs):,}")
    
    # Save to cache
    final_pairs.to_pickle(cache_file)
    CONSOLE.print(f"✅ Processed and cached {len(final_pairs):,} trial pairs to {cache_file}")
    return final_pairs

def load_sider_data() -> pd.DataFrame:
    """Loads SIDER side effect data."""
    cache_file = PROCESSED_CACHE_PATH / "sider_side_effects.pickle"
    
    CONSOLE.print(f"[blue]Debug: Checking SIDER cache at {cache_file.absolute()}[/blue]")
    CONSOLE.print(f"[blue]Debug: SIDER cache exists? {cache_file.exists()}[/blue]")
    
    if cache_file.exists():
        CONSOLE.print(f"Loading cached SIDER data from {cache_file}...")
        return pd.read_pickle(cache_file)
    
    CONSOLE.print("Processing SIDER side effect data...")
    sider_path = RAW_DATA_PATH / "meddra_freq.tsv.gz"
    
    CONSOLE.print(f"[blue]Debug: Looking for SIDER at {sider_path.absolute()}[/blue]")
    CONSOLE.print(f"[blue]Debug: SIDER file exists? {sider_path.exists()}[/blue]")
    
    if not sider_path.exists():
        CONSOLE.print(f"[bold red]Error: SIDER file not found at {sider_path}[/bold red]")
        return pd.DataFrame()
    
    try:
        with gzip.open(sider_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', header=None, 
                           names=["stitch_id_flat", "stitch_id_stereo", "umls_cui", 
                                 "meddra_concept_type", "umls_cui_meddra", "side_effect_name"])
        
        df.to_pickle(cache_file)
        CONSOLE.print(f"✅ Processed and cached {len(df):,} SIDER entries to {cache_file}")
        return df
    except Exception as e:
        CONSOLE.print(f"[bold red]Error processing SIDER file: {e}[/bold red]")
        return pd.DataFrame()

def get_stitch_to_drugbank_mapping() -> dict:
    """
    Creates a simple mock mapping for now since the UniChem API approach 
    was too complex. This is a placeholder for demonstration.
    """
    cache_file = PROCESSED_CACHE_PATH / "stitch_drugbank_map.pickle"
    
    CONSOLE.print(f"[blue]Debug: Checking mapping cache at {cache_file.absolute()}[/blue]")
    CONSOLE.print(f"[blue]Debug: Mapping cache exists? {cache_file.exists()}[/blue]")
    
    if cache_file.exists():
        CONSOLE.print("Loading cached STITCH-DrugBank mapping...")
        mapping = pd.read_pickle(cache_file)
        CONSOLE.print(f"[green]Loaded {len(mapping):,} STITCH-DrugBank mappings from cache.[/green]")
        return mapping
    
    CONSOLE.print("Creating simple STITCH-DrugBank mapping (demo version)...")
    # For now, create a simple demonstration mapping
    # In a real implementation, this would use UniChem or another service
    mapping = {}
    sider_df = load_sider_data()
    
    if not sider_df.empty:
        # Create some demo mappings (this is not real data, just for testing)
        unique_stitch_ids = sider_df['stitch_id_flat'].unique()[:100]  # Take first 100
        for i, stitch_id in enumerate(unique_stitch_ids):
            mapping[stitch_id] = f"DB{i+10000:05d}"  # Generate fake DrugBank IDs
    
    pd.to_pickle(mapping, cache_file)
    CONSOLE.print(f"✅ Created demo mapping with {len(mapping):,} entries (saved to {cache_file})")
    return mapping
