# src/ddi/data/preprocessing/subset_chembl.py

import os
import pickle
import json
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from typing import Set, Dict, Any, Optional, List, Tuple # Added Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("subset_chembl")

# --- Helper Functions ---

def load_pickle(file_path: str) -> Optional[Any]:
    """Loads data from a pickle file."""
    logger.info(f"Loading pickle file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
        return None

def load_json(file_path: str) -> Optional[Any]:
    """Loads data from a JSON file."""
    logger.info(f"Loading JSON file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}", exc_info=True)
        return None

def extract_drugbank_ids(drugbank_data: Dict[str, Any]) -> Tuple[Set[str], Set[str]]: # Use Tuple
    """Extracts ChEMBL compound IDs and UniProt target IDs from parsed DrugBank data."""
    chembl_ids: Set[str] = set()
    uniprot_ids: Set[str] = set()

    if not drugbank_data or "drugs" not in drugbank_data:
        logger.warning("DrugBank data is empty or missing 'drugs' key.")
        return chembl_ids, uniprot_ids

    logger.info("Extracting relevant IDs from DrugBank data...")
    drug_list = drugbank_data.get("drugs", []) if isinstance(drugbank_data, dict) else drugbank_data

    for drug in tqdm(drug_list, desc="Processing DrugBank drugs"):
        # Extract ChEMBL Compound ID
        for identifier in drug.get("external_identifiers", []):
            if identifier.get("resource") == "ChEMBL":
                chembl_id = identifier.get("identifier")
                if chembl_id:
                    chembl_ids.add(chembl_id)

        # Extract UniProt IDs from Targets, Enzymes, Carriers, Transporters
        for protein_type in ["targets", "enzymes", "carriers", "transporters"]:
            for protein in drug.get(protein_type, []):
                for polypeptide in protein.get("polypeptides", []):
                    for identifier in polypeptide.get("external_identifiers", []):
                        if identifier.get("resource") == "UniProtKB":
                            uniprot_id = identifier.get("identifier")
                            if uniprot_id:
                                uniprot_ids.add(uniprot_id)
                    if polypeptide.get("source") == "UniProtKB" and polypeptide.get("id"):
                         uniprot_ids.add(polypeptide["id"])

    logger.info(f"Extracted {len(chembl_ids)} unique ChEMBL compound IDs from DrugBank.")
    logger.info(f"Extracted {len(uniprot_ids)} unique UniProt target IDs from DrugBank.")
    return chembl_ids, uniprot_ids

def extract_opentargets_ids(opentargets_data: List[Dict[str, Any]]) -> Set[str]:
    """Extracts Target IDs (UniProt, Ensembl) from parsed OpenTargets association data."""
    target_ids_from_ot: Set[str] = set()

    if not opentargets_data:
        logger.warning("OpenTargets data is empty.")
        return target_ids_from_ot

    logger.info("Extracting relevant Target IDs from OpenTargets data...")
    for assoc in tqdm(opentargets_data, desc="Processing OpenTargets associations"):
        target_id = assoc.get('target_id')
        if target_id:
            target_ids_from_ot.add(target_id)

    logger.info(f"Extracted {len(target_ids_from_ot)} unique target IDs from OpenTargets (will be mapped via ChEMBL mapping file).")
    return target_ids_from_ot


def filter_uniprot_mapping(mapping_file: str, relevant_source_ids: Set[str]) -> Tuple[pd.DataFrame, Set[str]]: # Use Tuple
    """
    Reads ChEMBL UniProt mapping and filters based on relevant source IDs (UniProt, Ensembl, etc.).
    Returns the filtered mapping dataframe and the set of corresponding ChEMBL Target IDs.
    """
    logger.info(f"Filtering UniProt mapping file: {mapping_file}")
    relevant_target_chembl_ids: Set[str] = set()
    filtered_mapping = pd.DataFrame()

    if not os.path.exists(mapping_file):
        logger.error(f"UniProt mapping file not found: {mapping_file}")
        return filtered_mapping, relevant_target_chembl_ids
    if not relevant_source_ids:
        logger.warning("No relevant source target IDs provided for filtering UniProt mapping. Returning empty DataFrame.")
        return filtered_mapping, relevant_target_chembl_ids

    try:
        # --- MODIFICATION START ---
        # Define expected column names based on typical ChEMBL mapping file structure
        # Adjust these if your file has a different order or different columns
        expected_names = ['uniprot_accession', 'chembl_target_id', 'target_name', 'target_type']

        # Read the CSV specifying no header and providing names
        mapping_df = pd.read_csv(
            mapping_file,
            sep='\t',
            comment='#',       # Keep skipping comment lines
            header=None,       # Explicitly state there is NO header row in the data itself
            names=expected_names, # Assign the column names
            low_memory=False
        )
        # --- MODIFICATION END ---

        logger.info(f"Assigned mapping file columns: {mapping_df.columns.tolist()}") # Log assigned names

        # --- Adjust column identification logic to use assigned names ---
        # Source ID columns to check (primarily the one we assigned)
        # We still check others in case the file structure is unexpected, but prioritize the assigned name.
        possible_source_id_cols = ['uniprot_accession', 'accession', 'UniProt Accession', 'uniprot_id', 'gene_symbol', 'ensembl_id', 'protein_accession']
        source_id_cols_found = [col for col in possible_source_id_cols if col in mapping_df.columns]

        if not source_id_cols_found:
             # This error is less likely now but kept as a safeguard
             logger.error(f"Could not find any suitable source ID column using assigned/possible names in {mapping_file}. Tried: {possible_source_id_cols}")
             return filtered_mapping, relevant_target_chembl_ids
        logger.info(f"Using potential source ID columns for matching: {source_id_cols_found}")

        # ChEMBL Target ID column (use the assigned name)
        chembl_target_col = 'chembl_target_id'
        if chembl_target_col not in mapping_df.columns:
             logger.error(f"Assigned ChEMBL Target ID column '{chembl_target_col}' not found in {mapping_file}. Cannot proceed.")
             return filtered_mapping, relevant_target_chembl_ids
        logger.info(f"Using ChEMBL Target ID column: '{chembl_target_col}'")
        # --- End adjustment ---


        mask = pd.Series(False, index=mapping_df.index)
        for col in source_id_cols_found:
             # Check if column exists before using it (redundant check, but safe)
             if col in mapping_df.columns:
                 # Ensure comparison is robust (handle NaN, convert types if needed)
                 mask |= mapping_df[col].dropna().astype(str).isin(relevant_source_ids)

        filtered_mapping = mapping_df[mask].copy()

        if not filtered_mapping.empty:
            # Ensure target column exists before accessing
            if chembl_target_col in filtered_mapping.columns:
                 # Drop rows where the target ID might be NaN after filtering
                 filtered_mapping.dropna(subset=[chembl_target_col], inplace=True)
                 relevant_target_chembl_ids = set(filtered_mapping[chembl_target_col].unique())
                 logger.info(f"Found {len(filtered_mapping)} mappings for relevant source IDs.")
                 logger.info(f"Identified {len(relevant_target_chembl_ids)} unique relevant ChEMBL Target IDs.")
            else:
                 # This case should be caught earlier now
                 logger.error(f"ChEMBL Target ID column '{chembl_target_col}' not found in the filtered mapping results. Cannot extract target IDs.")
                 relevant_target_chembl_ids = set()
        else:
            logger.warning(f"No mappings found for the provided {len(relevant_source_ids)} source IDs.")

        return filtered_mapping, relevant_target_chembl_ids

    except Exception as e:
        logger.error(f"Error processing UniProt mapping file {mapping_file}: {e}", exc_info=True)
        return pd.DataFrame(), set()


def filter_chemreps(chemreps_file: str, relevant_compound_chembl_ids: Set[str]) -> pd.DataFrame:
    """Reads ChEMBL chem reps file (potentially large) and filters for relevant compounds."""
    logger.info(f"Filtering chemical representations file: {chemreps_file}")
    if not os.path.exists(chemreps_file):
        logger.error(f"Chem reps file not found: {chemreps_file}")
        return pd.DataFrame()
    if not relevant_compound_chembl_ids:
        logger.warning("No relevant compound ChEMBL IDs provided for filtering chem reps. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        chunk_size = 100000
        filtered_chunks = []
        header_df = pd.read_csv(chemreps_file, sep='\t', nrows=1)
        logger.info(f"Chem reps file columns: {header_df.columns.tolist()}")

        chembl_compound_col = None
        possible_chembl_compound_cols = ['chembl_id', 'molecule_chembl_id']
        for col in possible_chembl_compound_cols:
             if col in header_df.columns:
                 chembl_compound_col = col
                 logger.info(f"Using ChEMBL Compound ID column: '{chembl_compound_col}'")
                 break
        if not chembl_compound_col:
             logger.error(f"Could not find a suitable ChEMBL Compound ID column in {chemreps_file}. Tried: {possible_chembl_compound_cols}")
             return pd.DataFrame()

        pbar_desc = "Reading chem reps"
        iterator = pd.read_csv(chemreps_file, sep='\t', chunksize=chunk_size, low_memory=False)
        pbar = tqdm(iterator, desc=pbar_desc)

        for chunk in pbar:
            if chembl_compound_col not in chunk.columns:
                 logger.error(f"Column '{chembl_compound_col}' not found in a chunk of {chemreps_file}. Skipping chunk.")
                 continue
            chunk[chembl_compound_col] = chunk[chembl_compound_col].astype(str)
            filtered_chunk = chunk[chunk[chembl_compound_col].isin(relevant_compound_chembl_ids)]
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)

        pbar.close()

        if not filtered_chunks:
            logger.warning("No relevant chemical representations found after filtering.")
            return pd.DataFrame()

        final_df = pd.concat(filtered_chunks, ignore_index=True)
        logger.info(f"Found {len(final_df)} chemical representations for relevant compounds.")
        return final_df

    except Exception as e:
        logger.error(f"Error processing chem reps file {chemreps_file}: {e}", exc_info=True)
        return pd.DataFrame()

# --- Main Execution ---

def main():
    # Updated description to reflect inclusion of mesh_data
    parser = argparse.ArgumentParser(description="Subset ChEMBL data (chemreps, mapping) based on DrugBank/OpenTargets IDs and include processed MeSH data (from mesh_data*.pickle).")

    # Input Processed Files
    parser.add_argument("--drugbank_processed", required=True, help="Path to processed DrugBank data (pickle or json)")
    parser.add_argument("--opentargets_processed", required=True, help="Path to processed OpenTargets associations (pickle or json)")
    # Changed argument to load the main mesh_data file
    parser.add_argument("--mesh_data_processed", required=True, help="Path to processed MeSH data (e.g., mesh_data_2025.pickle)")

    # Input ChEMBL Files
    parser.add_argument("--chembl_dir", required=True, help="Directory containing ChEMBL files (chemreps.txt, uniprot_mapping.txt)")
    parser.add_argument("--chembl_version", default="35", help="ChEMBL version string (e.g., 35) used for filename construction")

    # Output File
    parser.add_argument("--output_pickle", required=True, help="Path to save the final combined data (subsetted ChEMBL + MeSH data) (pickle)")

    args = parser.parse_args()

    # --- 1. Load Processed Data ---
    logger.info("--- Loading Processed Source Data ---")
    db_ext = os.path.splitext(args.drugbank_processed)[1].lower()
    ot_ext = os.path.splitext(args.opentargets_processed)[1].lower()
    # Use the new argument for MeSH data file extension
    mesh_ext = os.path.splitext(args.mesh_data_processed)[1].lower()

    drugbank_data = load_pickle(args.drugbank_processed) if db_ext == '.pickle' else load_json(args.drugbank_processed)
    opentargets_data = load_pickle(args.opentargets_processed) if ot_ext == '.pickle' else load_json(args.opentargets_processed)
    # Load the main mesh_data file using the new argument
    mesh_data = load_pickle(args.mesh_data_processed) if mesh_ext == '.pickle' else load_json(args.mesh_data_processed)

    if drugbank_data is None or opentargets_data is None:
        logger.critical("Failed to load essential processed data (DrugBank or OpenTargets). Exiting.")
        return
    # Check if mesh_data loaded successfully
    if mesh_data is None:
        logger.warning("Failed to load processed MeSH data. Output will not include MeSH info.")
        # Continue without MeSH data

    # --- 2. Extract Relevant IDs ---
    logger.info("--- Extracting Relevant IDs ---")
    db_chembl_ids, db_uniprot_ids = extract_drugbank_ids(drugbank_data)
    ot_target_ids = extract_opentargets_ids(opentargets_data)

    all_relevant_source_target_ids = db_uniprot_ids.union(ot_target_ids)
    all_relevant_compound_chembl_ids = db_chembl_ids

    logger.info(f"Total unique relevant ChEMBL Compound IDs from DrugBank: {len(all_relevant_compound_chembl_ids)}")
    logger.info(f"Total unique source Target IDs (UniProt, Ensembl, etc.) from DrugBank/OpenTargets: {len(all_relevant_source_target_ids)}")

    if not all_relevant_compound_chembl_ids and not all_relevant_source_target_ids:
        logger.warning("No relevant compound or target IDs found from source data. ChEMBL subsetting might yield empty results.")

    # --- 3. Construct ChEMBL File Paths ---
    logger.info("--- Locating ChEMBL Files ---")
    chembl_rep_file = os.path.join(args.chembl_dir, f"chembl_{args.chembl_version}_chemreps.txt")
    chembl_map_file = os.path.join(args.chembl_dir, "chembl_uniprot_mapping.txt")

    if not os.path.exists(chembl_rep_file):
         logger.error(f"Required ChEMBL chem reps file not found: {chembl_rep_file}. Exiting.")
         return
    if not os.path.exists(chembl_map_file):
         logger.error(f"Required ChEMBL mapping file not found: {chembl_map_file}. Exiting.")
         return

    # --- 4. Filter ChEMBL Data ---
    logger.info("--- Filtering ChEMBL Data ---")
    subset_data = {}

    # Filter UniProt Mapping -> Get relevant ChEMBL Target IDs
    filtered_mapping_df, relevant_target_chembl_ids = filter_uniprot_mapping(
        chembl_map_file, all_relevant_source_target_ids
    )
    subset_data['uniprot_mapping'] = filtered_mapping_df
    logger.info(f"Derived {len(relevant_target_chembl_ids)} relevant ChEMBL Target IDs from mapping.")

    # Filter Chem Reps
    filtered_chemreps_df = filter_chemreps(
        chembl_rep_file, all_relevant_compound_chembl_ids
    )
    subset_data['chemreps'] = filtered_chemreps_df

    # --- H5 Filtering Skipped ---
    logger.info("--- Skipping ChEMBL H5 File Processing (Assumed Unavailable) ---")

    # --- 5. Add MeSH Data ---
    # Include the loaded mesh_data (which contains descriptors, qualifiers, hierarchy etc.)
    if mesh_data is not None:
        logger.info("--- Including loaded MeSH Data (from mesh_data*.pickle) ---")
        # Use the key 'mesh_data' to store the entire loaded dictionary
        subset_data['mesh_data'] = mesh_data
    else:
        logger.warning("--- MeSH Data not included (failed to load) ---")


    # --- 6. Save Combined Data ---
    logger.info("--- Saving Combined Data ---")
    output_dir = os.path.dirname(args.output_pickle)
    # Ensure the output directory exists *before* trying to open the file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Handle case where output path is just a filename in the current directory
        logger.warning(f"Output path '{args.output_pickle}' does not specify a directory. Saving to current directory.")


    try:
        # Check if the path is actually a directory *before* trying to open
        if os.path.isdir(args.output_pickle):
             logger.error(f"Output path '{args.output_pickle}' is a directory, not a file. Please provide a full filename.")
             # Optionally, construct a default filename within the directory
             # default_filename = os.path.join(args.output_pickle, "chembl_mesh_subset.pickle")
             # logger.warning(f"Attempting to save to default filename: {default_filename}")
             # args.output_pickle = default_filename # Uncomment to try saving with a default name
             # raise IsADirectoryError(f"[Errno 21] Is a directory: '{args.output_pickle}' - Corrected path needed.") # Re-raise or handle
             return # Exit if it's a directory and we don't auto-correct

        with open(args.output_pickle, "wb") as f:
            pickle.dump(subset_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Update log message
        logger.info(f"Successfully saved combined data (subsetted ChEMBL, MeSH data) to: {args.output_pickle}")
    except IsADirectoryError:
         # This catch might be redundant if the check above works, but good as a fallback
         logger.error(f"Failed to save output pickle file: '{args.output_pickle}' is a directory. Please provide a full filename.", exc_info=False) # Don't need full traceback again
    except Exception as e:
        logger.error(f"Failed to save output pickle file {args.output_pickle}: {e}", exc_info=True)

    # Update final log message
    logger.info("ChEMBL subsetting and MeSH data combination process finished.")


if __name__ == "__main__":
    main()
