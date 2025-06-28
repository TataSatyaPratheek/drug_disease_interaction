# src/scripts/create_disease_mapping.py

import os
import pickle
import json
import requests
import logging
import time
from tqdm import tqdm
from typing import Any, Dict, List, Set, Optional
import argparse

# --- Configuration ---
DEFAULT_MESH_PATH = "data/processed/diseases/mesh_data_2025.pickle"
DEFAULT_OT_PATH = "data/processed/associations/open_targets/opentargets_target_disease_associations.pickle"
DEFAULT_OUTPUT_DIR = "data/processed/mappings"
DEFAULT_OUTPUT_FILE = "disease_mapping_oxo.json"

OXO_BASE_URL = "https://www.ebi.ac.uk/spot/oxo/api/search"
BATCH_SIZE = 350 # Reduced batch size for more stability
SLEEP_TIME = 0.1 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("create_disease_mapping")

# --- Helper Functions ---
def load_pickle(file_path: str) -> Any:
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
        return None

def query_oxo_batch(ids: List[str], target_ontologies: List[str]) -> List[Dict[str, str]]:
    """Queries the OxO API using the correct POST method."""
    mappings = []
    if not ids:
        return mappings

    standardized_ids = [id_str.replace('_', ':', 1) for id_str in ids]
    payload = {
        "ids": standardized_ids,
        "mappingTarget": target_ontologies,
        "distance": "2"
    }

    try:
        response = requests.post(OXO_BASE_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "_embedded" in data and "mappings" in data["_embedded"]:
            for mapping in data["_embedded"]["mappings"]:
                from_curie = mapping.get("fromTerm", {}).get("curie")
                to_curie = mapping.get("toTerm", {}).get("curie")
                if from_curie and to_curie:
                    mappings.append({"from": from_curie, "to": to_curie})
    except requests.exceptions.RequestException as e:
        logger.error(f"OxO API request failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error during OxO query: {e}")
    
    return mappings

def debug_one_batch(ot_ids: List[str]):
    """A special function to debug a single small batch and print everything."""
    logger.info("--- RUNNING IN DEBUG MODE ---")
    # Using a small, specific batch of common diseases for a reliable test
    debug_batch_ids = [
        'EFO_0000685', 'MONDO_0005148', 'EFO_0000270', 'EFO_0000729', 
        'EFO_0005842', 'EFO_0000692', 'EFO_0000571', 'MONDO_0005393',
        'EFO_0000311', 'EFO_0000222'
    ]
    logger.info(f"Debug IDs (original): {debug_batch_ids}")
    
    standardized_ids = [id_str.replace('_', ':', 1) for id_str in debug_batch_ids]
    logger.info(f"Debug IDs (standardized for API): {standardized_ids}")
    
    payload = {"ids": standardized_ids, "mappingTarget": ["MeSH"], "distance": "1"}
    logger.info(f"Sending POST request to {OXO_BASE_URL} with payload:\n{json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(OXO_BASE_URL, json=payload, timeout=60)
        logger.info(f"API Response Status Code: {response.status_code}")
        response.raise_for_status()
        logger.info(f"API Response Body:\n{json.dumps(response.json(), indent=2)}")
    except Exception as e:
        logger.error(f"Debug request failed: {e}")
    logger.info("--- DEBUG MODE FINISHED ---")


# --- Main Script Logic ---
def main(mesh_path: str, ot_path: str, output_path: str, debug: bool):
    """Extracts IDs, queries OxO, and saves mappings."""
    logger.info("--- Step 1: Loading data and extracting IDs ---")
    ot_data = load_pickle(ot_path)
    if ot_data is None: return

    ot_ids_list = sorted(list(set(ot_data['diseaseId'].unique())))
    logger.info(f"Extracted {len(ot_ids_list)} unique OpenTargets (EFO/MONDO) IDs.")

    if debug:
        debug_one_batch(ot_ids_list)
        return

    # --- Step 2: Query OxO API in Batches ---
    logger.info(f"Querying EFO/MONDO -> MeSH ({len(ot_ids_list)} IDs)...")
    all_oxo_mappings: List[Dict[str, str]] = []
    for i in tqdm(range(0, len(ot_ids_list), BATCH_SIZE), desc="EFO/MONDO->MeSH"):
        batch_ids = ot_ids_list[i : i + BATCH_SIZE]
        batch_mappings = query_oxo_batch(batch_ids, target_ontologies=["MESH"])
        all_oxo_mappings.extend(batch_mappings)
        time.sleep(SLEEP_TIME)

    logger.info(f"Retrieved {len(all_oxo_mappings)} potential mappings from OxO.")

    # --- Step 3: Process and Save Mappings ---
    final_mapping: Dict[str, str] = {}
    for mapping in all_oxo_mappings:
        from_curie = mapping['from']
        to_curie = mapping['to']
        original_key = from_curie.replace(':', '_', 1)
        if to_curie.startswith("MESH:"):
            final_mapping[original_key] = to_curie
            
    logger.info(f"Created final mapping dictionary with {len(final_mapping)} entries.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_mapping, f, indent=2, sort_keys=True)
    logger.info(f"Successfully saved disease mappings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create disease ID mappings using OxO API.")
    parser.add_argument("--mesh_data", default=DEFAULT_MESH_PATH, help="Path to processed MeSH data pickle file.")
    parser.add_argument("--ot_data", default=DEFAULT_OT_PATH, help="Path to processed OpenTargets associations pickle file.")
    parser.add_argument("--output", default=os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE), help="Path to save the output JSON mapping file.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode on a small batch and print API response.")
    args = parser.parse_args()
    
    # We don't need the mesh_data for the main logic, only for the log message.
    main(args.mesh_data, args.ot_data, args.output, args.debug)
