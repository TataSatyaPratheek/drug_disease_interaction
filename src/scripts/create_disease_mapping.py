# /Users/vi/Documents/drug_disease_interaction/src/scripts/create_disease_mapping.py
import os
import pickle
import json
import requests
import logging
import time
from tqdm import tqdm
from typing import Any, Dict, List, Set, Optional
from pathlib import Path
import argparse
import sys

# Add src directory to path to import modules if needed (e.g., for logging setup)
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from ddi.utils.logging import setup_logging # Optional: use your setup

# --- Configuration ---
# Default paths (can be overridden by command-line args)
DEFAULT_MESH_PATH = "/Users/vi/Documents/drug_disease_interaction/data/processed/mesh/mesh_data_2025.pickle"
DEFAULT_OT_PATH = "/Users/vi/Documents/drug_disease_interaction/data/processed/associations/open_targets/opentargets_target_disease_associations.pickle"
DEFAULT_OUTPUT_DIR = "/Users/vi/Documents/drug_disease_interaction/data/processed/mappings"
DEFAULT_OUTPUT_FILE = "disease_mapping_oxo.json"

OXO_BASE_URL = "https://www.ebi.ac.uk/spot/oxo/api/search"
BATCH_SIZE = 150 # Number of IDs per OxO API call
SLEEP_TIME = 0.1 # Seconds to wait between batches

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("create_disease_mapping")

# --- Helper Functions ---
def load_pickle(file_path: str) -> Any:
    """Load data from a pickle file with error handling."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
        return None

def query_oxo_batch(ids: List[str], target_ontologies: List[str] = ["EFO", "MONDO"]) -> List[Dict[str, str]]:
    """Queries the OxO API for a batch of IDs."""
    mappings = []
    if not ids:
        return mappings

    params = {
        "ids": ",".join(ids),
        "mappingTarget": ",".join(target_ontologies),
        "distance": 1 # Direct mappings only
    }
    try:
        response = requests.get(OXO_BASE_URL, params=params, timeout=60) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "_embedded" in data and "mappings" in data["_embedded"]:
            for mapping in data["_embedded"]["mappings"]:
                try:
                    # Ensure required fields exist
                    from_curie = mapping.get("fromTerm", {}).get("curie")
                    to_curie = mapping.get("toTerm", {}).get("curie")
                    if from_curie and to_curie:
                        mappings.append({
                            "from": from_curie,
                            "to": to_curie
                        })
                except Exception as parse_e:
                    logger.warning(f"Could not parse individual mapping: {mapping}. Error: {parse_e}")
        else:
            logger.debug(f"No mappings found in OxO response for batch starting with {ids[0]}")

    except requests.exceptions.RequestException as e:
        logger.error(f"OxO API request failed: {e}")
    except json.JSONDecodeError as e:
         logger.error(f"Failed to decode JSON response from OxO: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during OxO query: {e}")

    return mappings

# --- Main Script Logic ---
def main(mesh_path: str, ot_path: str, output_path: str):
    """Extracts IDs, queries OxO, and saves mappings."""

    # 1. Load Data and Extract IDs
    logger.info("--- Step 1: Loading data and extracting IDs ---")
    mesh_data = load_pickle(mesh_path)
    ot_data = load_pickle(ot_path)

    if mesh_data is None or ot_data is None:
        logger.error("Failed to load input data. Exiting.")
        return

    # Extract MeSH IDs (ensure prefix 'MESH:')
    mesh_ids: Set[str] = set()
    if isinstance(mesh_data, dict) and 'descriptors' in mesh_data:
        for mesh_id in mesh_data['descriptors'].keys():
            if isinstance(mesh_id, str) and mesh_id.startswith("D"):
                mesh_ids.add(f"MESH:{mesh_id}")
    logger.info(f"Extracted {len(mesh_ids)} unique MeSH IDs.")

    # Extract OpenTargets IDs (EFO/MONDO - assume already prefixed)
    ot_ids: Set[str] = set()
    if isinstance(ot_data, list):
        for assoc in ot_data:
            if isinstance(assoc, dict):
                disease_id = assoc.get('disease_id')
                if isinstance(disease_id, str) and (disease_id.startswith("EFO_") or disease_id.startswith("MONDO:")):
                    ot_ids.add(disease_id)
    logger.info(f"Extracted {len(ot_ids)} unique OpenTargets (EFO/MONDO) IDs.")

    if not mesh_ids or not ot_ids:
        logger.warning("No MeSH or OpenTargets IDs extracted. Cannot create mapping.")
        return

    # 2. Query OxO API in Batches
    logger.info("--- Step 2: Querying OxO API for mappings ---")
    all_oxo_mappings: List[Dict[str, str]] = []

    # Query MeSH -> EFO/MONDO
    logger.info(f"Querying MeSH -> EFO/MONDO ({len(mesh_ids)} IDs)...")
    mesh_id_list = sorted(list(mesh_ids))
    for i in tqdm(range(0, len(mesh_id_list), BATCH_SIZE), desc="MeSH->EFO/MONDO"):
        batch_ids = mesh_id_list[i : i + BATCH_SIZE]
        batch_mappings = query_oxo_batch(batch_ids, target_ontologies=["EFO", "MONDO"])
        all_oxo_mappings.extend(batch_mappings)
        time.sleep(SLEEP_TIME) # Rate limiting

    # Query EFO/MONDO -> MeSH
    logger.info(f"Querying EFO/MONDO -> MeSH ({len(ot_ids)} IDs)...")
    ot_id_list = sorted(list(ot_ids))
    for i in tqdm(range(0, len(ot_id_list), BATCH_SIZE), desc="EFO/MONDO->MeSH"):
        batch_ids = ot_id_list[i : i + BATCH_SIZE]
        batch_mappings = query_oxo_batch(batch_ids, target_ontologies=["MESH"])
        all_oxo_mappings.extend(batch_mappings)
        time.sleep(SLEEP_TIME)

    logger.info(f"Retrieved {len(all_oxo_mappings)} potential mappings from OxO.")

    # 3. Process Mappings (Create EFO/MONDO -> MeSH dictionary)
    logger.info("--- Step 3: Processing mappings ---")
    # Target structure: {'EFO_xxxx': 'MESH:Dyyyy', 'MONDO:zzzz': 'MESH:Dwwww'}
    final_mapping: Dict[str, str] = {}

    for mapping in all_oxo_mappings:
        id1 = mapping['from']
        id2 = mapping['to']

        mesh_id: Optional[str] = None
        other_id: Optional[str] = None

        if id1.startswith("MESH:") and (id2.startswith("EFO_") or id2.startswith("MONDO:")):
            mesh_id = id1
            other_id = id2
        elif id2.startswith("MESH:") and (id1.startswith("EFO_") or id1.startswith("MONDO:")):
            mesh_id = id2
            other_id = id1

        if mesh_id and other_id:
            if other_id in final_mapping:
                # Handle conflict: EFO/MONDO ID already maps to a MeSH ID
                existing_mesh_id = final_mapping[other_id]
                if existing_mesh_id != mesh_id:
                    logger.warning(f"Conflict: {other_id} maps to both {existing_mesh_id} and {mesh_id}. Keeping first mapping ({existing_mesh_id}).")
            else:
                # Add the mapping
                final_mapping[other_id] = mesh_id

    logger.info(f"Created final mapping dictionary with {len(final_mapping)} entries (EFO/MONDO -> MeSH).")

    # 4. Save Mappings
    logger.info("--- Step 4: Saving mappings ---")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_mapping, f, indent=2, sort_keys=True)
        logger.info(f"Successfully saved disease mappings to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save mapping file: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create disease ID mappings using OxO API.")
    parser.add_argument("--mesh_data", default=DEFAULT_MESH_PATH, help="Path to processed MeSH data pickle file.")
    parser.add_argument("--ot_data", default=DEFAULT_OT_PATH, help="Path to processed OpenTargets target-disease associations pickle file.")
    parser.add_argument("--output", default=os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE), help="Path to save the output JSON mapping file.")
    args = parser.parse_args()

    main(args.mesh_data, args.ot_data, args.output)
