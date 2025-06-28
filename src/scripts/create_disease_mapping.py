# src/scripts/create_disease_mapping.py

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

# --- Configuration ---
# Use relative paths from the project root
DEFAULT_MESH_PATH = "data/processed/diseases/mesh_data_2025.pickle"
DEFAULT_OT_PATH = "data/processed/associations/open_targets/opentargets_target_disease_associations.pickle"
DEFAULT_OUTPUT_DIR = "data/processed/mappings"
DEFAULT_OUTPUT_FILE = "disease_mapping_oxo.json"

OXO_BASE_URL = "https://www.ebi.ac.uk/spot/oxo/api/search"
BATCH_SIZE = 150  # Number of IDs per OxO API call
SLEEP_TIME = 0.1  # Seconds to wait between batches

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

    # Standardize ID format from 'EFO_123' to 'EFO:123' for the API call
    standardized_ids = [id_str.replace('_', ':', 1) for id_str in ids]

    params = {
        "ids": ",".join(standardized_ids),
        "mappingTarget": ",".join(target_ontologies),
        "distance": "1"  # Direct mappings only
    }
    try:
        response = requests.get(OXO_BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "_embedded" in data and "mappings" in data["_embedded"]:
            for mapping in data["_embedded"]["mappings"]:
                try:
                    from_curie = mapping.get("fromTerm", {}).get("curie")
                    to_curie = mapping.get("toTerm", {}).get("curie")
                    if from_curie and to_curie:
                        mappings.append({"from": from_curie, "to": to_curie})
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
    logger.info("--- Step 1: Loading data and extracting IDs ---")
    mesh_data = load_pickle(mesh_path)
    ot_data = load_pickle(ot_path)

    if mesh_data is None or ot_data is None:
        logger.error("Failed to load input data. Exiting.")
        return

    # Extract MeSH IDs (ensure prefix 'MESH:')
    mesh_ids: Set[str] = {f"MESH:{mesh_id}" for mesh_id in mesh_data.get('descriptors', {})}
    logger.info(f"Extracted {len(mesh_ids)} unique MeSH IDs.")

    # Extract OpenTargets IDs (EFO/MONDO)
    ot_ids: Set[str] = set(ot_data['diseaseId'].unique())
    logger.info(f"Extracted {len(ot_ids)} unique OpenTargets (EFO/MONDO) IDs.")

    if not mesh_ids or not ot_ids:
        logger.warning("No MeSH or OpenTargets IDs extracted. Cannot create mapping.")
        return

    # 2. Query OxO API in Batches
    logger.info("--- Step 2: Querying OxO API for mappings ---")
    all_oxo_mappings: List[Dict[str, str]] = []
    
    # Query EFO/MONDO -> MeSH, as this is the direction we need for the final map
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
    final_mapping: Dict[str, str] = {}
    for mapping in all_oxo_mappings:
        from_curie = mapping['from'] # e.g., 'EFO:0000408'
        to_curie = mapping['to']   # e.g., 'MESH:D009369'

        # The key for our mapping needs to match the original OpenTargets ID format (with an underscore).
        original_key = from_curie.replace(':', '_', 1)

        # We queried EFO/MONDO -> MeSH, so from_curie is EFO/MONDO and to_curie is MeSH
        if to_curie.startswith("MESH:") and (from_curie.startswith("EFO:") or from_curie.startswith("MONDO:")):
            if original_key in final_mapping and final_mapping[original_key] != to_curie:
                logger.warning(f"Conflict: {original_key} maps to both {final_mapping[original_key]} and {to_curie}. Keeping first mapping.")
            else:
                final_mapping[original_key] = to_curie
    
    logger.info(f"Created final mapping dictionary with {len(final_mapping)} entries (EFO/MONDO -> MeSH).")

    # 4. Save Mappings
    logger.info("--- Step 4: Saving mappings ---")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_mapping, f, indent=2, sort_keys=True)
        logger.info(f"Successfully saved disease mappings to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save mapping file: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create disease ID mappings using OxO API.")
    parser.add_argument("--mesh_data", default=DEFAULT_MESH_PATH, help="Path to processed MeSH data pickle file.")
    parser.add_argument("--ot_data", default=DEFAULT_OT_PATH, help="Path to processed OpenTargets associations pickle file.")
    parser.add_argument("--output", default=os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE), help="Path to save the output JSON mapping file.")
    args = parser.parse_args()
    
    main(args.mesh_data, args.ot_data, args.output)
