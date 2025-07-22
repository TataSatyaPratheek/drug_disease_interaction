#!/usr/bin/env python3
"""
Generates the EFO/MONDO -> MeSH mapping from the OxO bulk dump.
This script is robust to malformed CSVs by using Python's built-in csv module.
It reads only ols_mappings.csv, as datasources.csv is broken and redundant.
"""

from pathlib import Path
import pandas as pd
import json
import csv
from tqdm import tqdm

# --- Configuration ---
ROOT = Path("data/raw/oxo/oxo-mappings-2020-02-04")
MAPPING_CSV_PATH = ROOT / "ols_mappings.csv"
OUTPUT_JSON_PATH = Path("data/processed/mappings/disease_mapping_oxo.json")

def create_mapping_from_dump():
    """Main function to parse the dump and create the mapping file."""
    print(f"Starting disease mapping generation from: {MAPPING_CSV_PATH}")
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    mappings = []
    try:
        with open(MAPPING_CSV_PATH, 'r', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            
            # Read header and find column indices dynamically for robustness
            header = next(reader)
            print(f"Detected header: {header}")
            try:
                from_idx = header.index("fromCurie")
                to_idx = header.index("toCurie")
                prefix_idx = header.index("datasourcePrefix")
            except ValueError as e:
                print(f"FATAL: Required column not found in CSV header: {e}")
                return

            print("Processing mappings... (This may take a minute)")
            # Use tqdm to show progress on the large file
            for row in tqdm(reader, desc="Parsing ols_mappings.csv"):
                # Basic safety check for row length
                if len(row) > max(from_idx, to_idx, prefix_idx):
                    prefix = row[prefix_idx]
                    to_curie = row[to_idx]

                    if (prefix in ["EFO", "MONDO"]) and to_curie.startswith("MeSH:"):
                        from_curie = row[from_idx]
                        mappings.append({'from': from_curie, 'to': to_curie})

    except FileNotFoundError:
        print(f"FATAL: Mapping file not found at {MAPPING_CSV_PATH}")
        return
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return

    if not mappings:
        print("WARNING: No EFO/MONDO -> MeSH mappings were found in the file.")
        return

    print(f"Found {len(mappings)} potential mappings. Deduplicating...")

    # Use pandas for its efficient deduplication
    df = pd.DataFrame(mappings)
    df.drop_duplicates(inplace=True)

    # Final conversion to the required dictionary format {EFO_ID: MeSH_ID}
    final_mapping = {row['from'].replace(":", "_", 1): row['to']
                     for _, row in df.iterrows()}

    with open(OUTPUT_JSON_PATH, "w") as fh:
        json.dump(final_mapping, fh, indent=2, sort_keys=True)

    print(f"✅ Successfully wrote {len(final_mapping):,} EFO/MONDO -> MeSH mappings to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    create_mapping_from_dump()
