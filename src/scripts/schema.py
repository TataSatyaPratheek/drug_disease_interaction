import os
import pickle
import pandas as pd
import pyarrow.parquet as pq
import json
import gc
import sys
from collections import Counter
from typing import Iterator, Any, Dict, List

# --- Utilities ---

def safe_pickle_iterator(pickle_path: str, sample_size: int = 100) -> Iterator[Any]:
    """Memory-safe iterator for pickle files that may contain large lists."""
    try:
        with open(pickle_path, 'rb') as f:
            obj = pickle.load(f)
            
        if isinstance(obj, list):
            # Sample from the list without loading all into memory at once
            import random
            if len(obj) <= sample_size:
                for item in obj:
                    yield item
            else:
                # Random sampling for large lists
                indices = random.sample(range(len(obj)), min(sample_size, len(obj)))
                for i in indices:
                    yield obj[i]
            del obj  # Explicit cleanup
            
        elif isinstance(obj, dict):
            # For dicts, yield key-value pairs or sample values
            count = 0
            for key, value in obj.items():
                if count >= sample_size:
                    break
                yield (key, value)
                count += 1
            del obj  # Explicit cleanup
        else:
            yield obj
            del obj  # Explicit cleanup
            
        gc.collect()  # Force garbage collection
        
    except Exception as e:
        print(f"Error reading {pickle_path}: {e}")

def analyze_dict_list_memory_safe(records_iter: Iterator[Any], max_keys: int = 30, sample_size: int = 100) -> List[str]:
    """Memory-safe analysis of dict lists using iterator."""
    key_counter = Counter()
    count = 0
    
    for rec in records_iter:
        if isinstance(rec, dict):
            key_counter.update(rec.keys())
        count += 1
        if count >= sample_size:
            break
            
        # Periodic garbage collection for large iterations
        if count % 50 == 0:
            gc.collect()
    
    top_keys = [k for k, _ in key_counter.most_common(max_keys)]
    return top_keys

def print_schema_preview_dict_list_memory_safe(records_iter: Iterator[Any], name: str = "", sample_size: int = 100):
    """Memory-safe schema preview for dict lists using iterator."""
    print(f"\n=== {name or 'DICT LIST'} | sampling first {sample_size} records ===")
    
    # Collect a small sample for analysis
    sample_records = []
    count = 0
    for rec in records_iter:
        if count >= sample_size:
            break
        sample_records.append(rec)
        count += 1
    
    if not sample_records:
        print("No records found")
        return
    
    # Analyze keys from sample
    key_counter = Counter()
    for rec in sample_records:
        if isinstance(rec, dict):
            key_counter.update(rec.keys())
    
    top_keys = [k for k, _ in key_counter.most_common(30)]
    
    for k in top_keys:
        # Get examples from sample
        vals = [rec.get(k, None) for rec in sample_records[:10] if isinstance(rec, dict)]
        non_none_vals = [v for v in vals if v is not None]
        
        if not non_none_vals:
            print(f"- {k} : NoneType\texamples: []")
            continue
        
        # Handle unhashable types (like lists, dicts)
        try:
            unique_vals = list(set(non_none_vals))
        except TypeError:
            # If we can't hash the values, just take unique by converting to string and back
            seen_strs = set()
            unique_vals = []
            for v in non_none_vals:
                v_str = str(v)
                if v_str not in seen_strs:
                    seen_strs.add(v_str)
                    unique_vals.append(v)
                if len(unique_vals) >= 2:  # Limit to 2 examples
                    break
        
        val_type = type(unique_vals[0]).__name__ if unique_vals else "NoneType"
        
        # For lists/dicts, show length info and truncated examples
        if unique_vals and isinstance(unique_vals[0], (list, dict)):
            if isinstance(unique_vals[0], list):
                examples = [f"list[{len(v)}]" for v in unique_vals[:2]]
            else:
                examples = [f"dict[{len(v)} keys]" for v in unique_vals[:2]]
            print(f"- {k} : {val_type}\texamples: {examples}")
        else:
            # Truncate long string examples
            truncated_examples = []
            for v in unique_vals[:2]:
                if isinstance(v, str) and len(v) > 100:
                    truncated_examples.append(f'"{v[:100]}..."')
                else:
                    truncated_examples.append(repr(v))
            print(f"- {k} : {val_type}\texamples: {truncated_examples}")
    
    # Cleanup
    del sample_records
    gc.collect()

def print_schema_preview_dict_list(records, name=""):
    """Legacy function for backward compatibility - converts to iterator."""
    def records_iter():
        for rec in records[:100]:  # Limit to first 100 for memory safety
            yield rec
    print_schema_preview_dict_list_memory_safe(records_iter(), name, 100)

def print_schema_preview_mapping(mapping, name=""):
    print(f"\n=== {name or 'DICT'} | top-level keys: {list(mapping.keys())[:10]} ===")
    for k in list(mapping.keys())[:5]:
        v = mapping[k]
        print(f"- {k} : {type(v).__name__}")

def print_schema_preview_df(df, name=""):
    print(f"\n=== {name or 'DATAFRAME'} | n={len(df)} ===")
    for col in df.columns:
        ex = df[col].dropna().head(2)
        vtype = ex.apply(type).astype(str).tolist()
        print(f"- {col} : {str(df[col].dtype)}\texamples: {ex.tolist()}, types: {vtype}")

def analyze_pickle_file(pickle_path, name=""):
    """Memory-safe pickle file analysis using iterators."""
    print(f"\n#### {pickle_path} ####")
    
    # First, peek at the file to understand its structure without loading everything
    try:
        with open(pickle_path, 'rb') as f:
            obj = pickle.load(f)
        
        if isinstance(obj, list):
            print(f"Detected list with {len(obj)} items")
            # Use memory-safe iterator
            del obj  # Free memory immediately
            gc.collect()
            
            iterator = safe_pickle_iterator(pickle_path, sample_size=200)
            print_schema_preview_dict_list_memory_safe(iterator, name, 200)
            
        elif isinstance(obj, dict):
            print(f"Detected dict with {len(obj)} keys")
            sample_val = next(iter(obj.values()))
            
            if isinstance(sample_val, list):
                print(f"Dict values are lists. Sampling first value list with {len(sample_val)} items")
                del obj
                gc.collect()
                
                # Create iterator for the first list value
                def dict_list_iter():
                    with open(pickle_path, 'rb') as f:
                        obj = pickle.load(f)
                    first_key = next(iter(obj.keys()))
                    first_list = obj[first_key]
                    for i, item in enumerate(first_list[:200]):  # Sample first 200 items
                        yield item
                    del obj
                    gc.collect()
                
                print_schema_preview_dict_list_memory_safe(dict_list_iter(), name=f"{name}/sublist", sample_size=200)
                
            elif isinstance(sample_val, dict):
                print("Top-level dict where values are dicts; printing subfield keys of a sample value:")
                print_schema_preview_dict_list_memory_safe(iter([sample_val]), name=f"{name}/subdict", sample_size=1)
                del obj
                gc.collect()
            else:
                print_schema_preview_mapping(obj, name)
                del obj
                gc.collect()
        else:
            print(f"Unknown type: {type(obj)}")
            del obj
            gc.collect()
            
    except Exception as e:
        print(f"Error analyzing {pickle_path}: {e}")
        gc.collect()

def analyze_parquet_file(parquet_path, name=""):
    print(f"\n#### {parquet_path} ####")
    # Print column names and a sample row
    try:
        # Print schema via pyarrow (for nested types)
        sch = pq.read_schema(parquet_path)
        print(f"PyArrow schema:\n{sch}")
    except Exception as e:
        print(f"Could not load schema for {parquet_path}: {e}")
    try:
        # Use pyarrow to read only a limited number of rows to avoid memory issues
        parquet_file = pq.ParquetFile(parquet_path)
        # Read first batch (limited rows)
        batch = parquet_file.read_row_group(0, columns=None)
        df = batch.to_pandas()
        
        # If still too large, sample it further
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)
        
        print_schema_preview_df(df, name=name)
        del df, batch  # Explicit cleanup
        gc.collect()
    except Exception as e:
        # Fallback: try to read full file but limit columns for memory safety
        try:
            df = pd.read_parquet(parquet_path)
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)
            print_schema_preview_df(df, name=name)
            del df
            gc.collect()
        except Exception as e2:
            print(f"Couldn't load df with either method: {e}, {e2}")

# --- Main analysis ---

# Redirect output to file
output_file = "/home/vi/Documents/drug_disease_interaction/src/scripts/schema_analysis_output.txt"
with open(output_file, 'w') as f:
    # Redirect stdout to file
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=== SCHEMA ANALYSIS REPORT ===")
    print(f"Generated on: {pd.Timestamp.now()}")
    print("="*50)

    # 1. DrugBank (Pickle)
    drugbank_file = "/home/vi/Documents/drug_disease_interaction/data/processed/drugs/drugbank_parsed.pickle"
    if os.path.exists(drugbank_file):
        analyze_pickle_file(drugbank_file, "DrugBank")
    else:
        print(f"{drugbank_file} not found.")

    # 2. MeSH (Pickle)
    mesh_file = "/home/vi/Documents/drug_disease_interaction/data/processed/mesh/mesh_data_2025.pickle"
    if os.path.exists(mesh_file):
        analyze_pickle_file(mesh_file, "MeSH")
    else:
        print(f"{mesh_file} not found.")

    # 3. All Open Targets Parquet files
    parquet_dir = "/home/vi/Documents/drug_disease_interaction/data/processed/open_targets_merged/"
    if os.path.exists(parquet_dir):
        for fname in sorted(os.listdir(parquet_dir)):
            if fname.lower().endswith(".parquet"):
                full_path = os.path.join(parquet_dir, fname)
                analyze_parquet_file(full_path, name=fname)
    else:
        print(f"{parquet_dir} not found.")
    
    print("\n=== ANALYSIS COMPLETE ===")
    
    # Restore stdout
    sys.stdout = original_stdout

print(f"Schema analysis complete! Output saved to: {output_file}")
print("You can view the results with: cat schema_analysis_output.txt")
