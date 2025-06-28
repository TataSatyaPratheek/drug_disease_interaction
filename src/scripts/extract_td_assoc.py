# /Users/vi/Documents/drug_disease_interaction/src/scripts/extract_td_assoc.py
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
input_dir = "/Users/vi/Documents/drug_disease_interaction/data/processed/associations/open_targets"
old_assoc_file = "opentargets_associations.pickle" # File created by the FIRST parser run
new_assoc_file = "opentargets_target_disease_associations.pickle" # File expected by the LEAN analysis script
# -------------------

input_path = os.path.join(input_dir, old_assoc_file)
output_path = os.path.join(input_dir, new_assoc_file)

if not os.path.exists(input_path):
    logging.error(f"Input file not found: {input_path}")
    exit(1)

logging.info(f"Loading data from {input_path}...")
try:
    with open(input_path, 'rb') as f:
        # The original parser saved a dictionary like:
        # {'drug_target_associations': [], 'target_disease_associations': [...], 'drug_disease_associations': []}
        data = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load pickle file: {e}")
    exit(1)

if not isinstance(data, dict):
    logging.error(f"Expected a dictionary in {input_path}, but found {type(data)}")
    exit(1)

# Extract the target-disease associations
td_associations = data.get('target_disease_associations')

if td_associations is None:
    logging.error("Key 'target_disease_associations' not found in the loaded data.")
    exit(1)

if not isinstance(td_associations, list):
     logging.error(f"Expected a list for 'target_disease_associations', but found {type(td_associations)}")
     exit(1)

logging.info(f"Extracted {len(td_associations)} target-disease associations.")

if not td_associations:
    logging.warning("The extracted list is empty. Saving empty file.")

logging.info(f"Saving extracted list to {output_path}...")
try:
    with open(output_path, 'wb') as f:
        pickle.dump(td_associations, f)
    logging.info("Successfully saved.")
except Exception as e:
    logging.error(f"Failed to save extracted data: {e}")
    exit(1)

