#!/usr/bin/env python3

import cudf
import pandas as pd
import os

# Quick debug script to understand pathway data format
OPEN_TARGETS_DIR = "/home/vi/Documents/drug_disease_interaction/data/processed/open_targets_merged"

print("=== DEBUGGING PATHWAY DATA FORMAT ===")

# Load target data
target_file = os.path.join(OPEN_TARGETS_DIR, "target.parquet")
print(f"Loading {target_file}...")

ot_target = cudf.read_parquet(target_file)
target_df = ot_target.to_pandas()

print(f"Target data shape: {target_df.shape}")
print(f"Columns: {list(target_df.columns)}")

# Check pathways column
pathways_col = target_df['pathways']
print(f"\nPathways column info:")
print(f"Non-null count: {pathways_col.notna().sum()}")
print(f"Data type: {pathways_col.dtype}")

# Look at first few non-null pathway entries
non_null_pathways = target_df[target_df['pathways'].notna()]
print(f"\nFirst 3 non-null pathway entries:")

for i in range(min(3, len(non_null_pathways))):
    row = non_null_pathways.iloc[i]
    pathways = row['pathways']
    print(f"\nEntry {i+1}:")
    print(f"  Target ID: {row['id']}")
    print(f"  Pathways type: {type(pathways)}")
    print(f"  Pathways value: {repr(pathways)}")
    
    if isinstance(pathways, str):
        print(f"  Length: {len(pathways)}")
        print(f"  First 100 chars: {pathways[:100]}")

print("\n=== END DEBUG ===")
