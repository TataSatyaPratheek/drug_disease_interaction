# src/scripts/build_protein_mapping.py

import pandas as pd
import pickle
import os
from UniProtMapper import ProtMapper

def build_protein_id_mapping():
    """
    Fetches protein IDs from processed DrugBank and OpenTargets data,
    maps Ensembl IDs to UniProt IDs, and saves the result.
    """
    print("Starting protein ID mapping process...")
    
    # --- Step 1: Define paths based on your project structure ---
    drugbank_pickle_path = 'data/processed/drugs/drugbank_parsed.pickle'
    opentargets_pickle_path = 'data/processed/associations/open_targets/opentargets_target_disease_associations.pickle'
    output_dir = 'data/processed/mappings'
    output_csv_path = os.path.join(output_dir, 'protein_ensembl_to_uniprot_mapping.csv')
    failed_ids_path = os.path.join(output_dir, 'failed_protein_mappings.txt')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Step 2: Collect all protein IDs from your datasets ---
    print(f"Loading DrugBank data from {drugbank_pickle_path}...")
    with open(drugbank_pickle_path, 'rb') as f:
        drugbank_data = pickle.load(f)
    
    print(f"Loading OpenTargets data from {opentargets_pickle_path}...")
    opentargets_raw = pd.read_pickle(opentargets_pickle_path)
    
    # Handle different data types and inspect structure
    if isinstance(opentargets_raw, list):
        print("Data is a list, converting to DataFrame...")
        print(f"List length: {len(opentargets_raw)}")
        
        if len(opentargets_raw) > 0:
            print(f"First item type: {type(opentargets_raw[0])}")
            print(f"First item sample: {opentargets_raw[0] if isinstance(opentargets_raw[0], dict) else str(opentargets_raw[0])[:200]}")
        
        opentargets_data = pd.DataFrame(opentargets_raw)
    else:
        print("Data is already a DataFrame...")
        opentargets_data = opentargets_raw
    
    # Debug: Check the structure of the DataFrame
    print(f"DataFrame shape: {opentargets_data.shape}")
    print(f"DataFrame columns: {list(opentargets_data.columns)}")
    
    # Find the correct column for target IDs
    target_id_column = None
    possible_target_columns = ['target_id', 'targetId', 'target', 'ensembl_id', 'gene_id', 'targetFromSourceId']
    
    for col in possible_target_columns:
        if col in opentargets_data.columns:
            target_id_column = col
            print(f"Found target ID column: {col}")
            break
    
    if target_id_column is None:
        print("Available columns:")
        for i, col in enumerate(opentargets_data.columns):
            print(f"  {i}: {col}")
        
        # Try to find any column that might contain Ensembl IDs
        for col in opentargets_data.columns:
            sample_values = opentargets_data[col].dropna().head(5).tolist()
            print(f"Sample values for '{col}': {sample_values}")
            # Check if values look like Ensembl IDs (typically start with ENSG)
            if any(str(val).startswith('ENSG') for val in sample_values if pd.notna(val)):
                target_id_column = col
                print(f"Detected Ensembl-like IDs in column: {col}")
                break
    
    if target_id_column is None:
        print("ERROR: Could not find a suitable target ID column.")
        print("Please check your data structure and update the script accordingly.")
        return
    
    # Extract protein IDs from OpenTargets (Ensembl Gene IDs)
    print(f"Extracting Ensembl IDs from column: {target_id_column}")
    opentargets_ensembl_ids = set(opentargets_data[target_id_column].dropna().unique())
    
    # Filter out any non-Ensembl IDs if needed
    ensembl_ids_filtered = set()
    for target_id in opentargets_ensembl_ids:
        target_id_str = str(target_id)
        # Keep IDs that look like Ensembl Gene IDs
        if target_id_str.startswith('ENSG') or target_id_str.startswith('ENS'):
            ensembl_ids_filtered.add(target_id_str)
    
    opentargets_ensembl_ids = ensembl_ids_filtered
    
    print(f"Found {len(opentargets_ensembl_ids)} unique Ensembl IDs")
    
    if not opentargets_ensembl_ids:
        print("No Ensembl IDs found. Exiting.")
        return
    
    # Show sample IDs for verification
    sample_ids = list(opentargets_ensembl_ids)[:5]
    print(f"Sample Ensembl IDs: {sample_ids}")
    
    # --- Step 3: Use UniProtMapper to map IDs ---
    print("Initializing UniProtMapper...")
    mapper = ProtMapper()
    
    # Check available database mappings
    print("Checking supported database mappings...")
    try:
        # Print some supported databases for debugging
        print("Mapper object created successfully")
    except Exception as e:
        print(f"Error creating mapper: {e}")
        return
    
    ensembl_id_list = list(opentargets_ensembl_ids)
    
    # Try different parameter combinations for the mapping
    mapping_attempts = [
        {"from_db": "Ensembl", "to_db": "UniProtKB"},
        {"from_db": "Ensembl", "to_db": "UniProtKB-ID"},
        {"from_db": "Ensembl_Gene", "to_db": "UniProtKB"},
        {"from_db": "Ensembl_Gene", "to_db": "UniProtKB-ID"},
        {"from_db": "Ensembl", "to_db": "UniProt"},
    ]
    
    mapping_df = None
    failed_ids = []
    
    # Try with a small subset first to test the parameters
    test_ids = ensembl_id_list[:10]  # Test with first 10 IDs
    
    for attempt in mapping_attempts:
        print(f"\nTrying mapping with from_db='{attempt['from_db']}' to_db='{attempt['to_db']}'...")
        
        try:
            test_mapping_df, test_failed_ids = mapper.get(
                ids=test_ids,
                from_db=attempt['from_db'],
                to_db=attempt['to_db']
            )
            
            if not test_mapping_df.empty:
                print(f"✓ Success! Found {len(test_mapping_df)} mappings out of {len(test_ids)} test IDs")
                print(f"Sample mapping: {test_mapping_df.head(2).to_dict('records')}")
                
                # Now run the full mapping with the working parameters
                print(f"\nRunning full mapping with {len(ensembl_id_list)} IDs...")
                
                # Process in batches to avoid API limits
                batch_size = 500  # Adjust if needed
                all_mappings = []
                all_failed = []
                
                for i in range(0, len(ensembl_id_list), batch_size):
                    batch = ensembl_id_list[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(ensembl_id_list)-1)//batch_size + 1} ({len(batch)} IDs)...")
                    
                    try:
                        batch_mapping_df, batch_failed_ids = mapper.get(
                            ids=batch,
                            from_db=attempt['from_db'],
                            to_db=attempt['to_db']
                        )
                        
                        if not batch_mapping_df.empty:
                            all_mappings.append(batch_mapping_df)
                        if batch_failed_ids:
                            all_failed.extend(batch_failed_ids)
                            
                    except Exception as batch_e:
                        print(f"Error in batch {i//batch_size + 1}: {batch_e}")
                        all_failed.extend(batch)
                
                # Combine all successful mappings
                if all_mappings:
                    mapping_df = pd.concat(all_mappings, ignore_index=True)
                    failed_ids = all_failed
                    print("✓ Full mapping completed!")
                    break
                else:
                    print("No successful mappings in full run")
                    
            else:
                print("✗ No mappings found with these parameters")
                
        except Exception as e:
            print(f"✗ Error with from_db='{attempt['from_db']}', to_db='{attempt['to_db']}': {e}")
            continue
    
    # --- Step 4: Process and Save the Mapping File ---
    if mapping_df is not None and not mapping_df.empty:
        print(f"\n✓ Successfully mapped {len(mapping_df)} IDs out of {len(ensembl_id_list)} total IDs.")
        
        # Standardize column names
        if 'From' in mapping_df.columns and 'To' in mapping_df.columns:
            mapping_df.rename(columns={'From': 'Ensembl_ID', 'To': 'UniProt_ID'}, inplace=True)
        
        mapping_df.to_csv(output_csv_path, index=False)
        print(f"✓ Protein ID mapping saved to {output_csv_path}")
        
        # Show some statistics
        print("✓ Mapping statistics:")
        print(f"  - Total Ensembl IDs: {len(ensembl_id_list)}")
        print(f"  - Successfully mapped: {len(mapping_df)}")
        print(f"  - Failed to map: {len(failed_ids)}")
        print(f"  - Success rate: {len(mapping_df)/len(ensembl_id_list)*100:.1f}%")
        
    else:
        print("\n✗ All mapping attempts failed.")
        print("This could be due to:")
        print("  - Incorrect database identifiers")
        print("  - API connectivity issues") 
        print("  - UniProt service unavailable")
        
    if failed_ids:
        print(f"\n⚠ Saving {len(failed_ids)} failed IDs to {failed_ids_path}")
        with open(failed_ids_path, 'w') as f:
            for item in failed_ids:
                f.write(f"{item}\n")
    
    print("\nProtein ID mapping process finished.")

if __name__ == '__main__':
    # Before running, make sure you have installed the library:
    # pip install uniprot-id-mapper
    build_protein_id_mapping()
