#!/usr/bin/env python3
"""
Simple CSV optimization script for Memgraph compatibility
Fixes the key issues identified in the analysis
"""
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path

def optimize_nodes_for_memgraph(df):
    """Optimize nodes CSV for Memgraph import"""
    print(f"🔧 Optimizing {len(df):,} nodes...")
    
    # Create optimized copy
    opt_df = df.copy()
    
    # Handle missing names - use ID as fallback
    if 'name' in opt_df.columns:
        missing_names = opt_df['name'].isna()
        opt_df.loc[missing_names, 'name'] = opt_df.loc[missing_names, 'id']
        print(f"  ✅ Fixed {missing_names.sum():,} missing names")
    
    # Convert boolean-like strings to actual booleans
    bool_columns = ['has_structure_data', 'lipinski_compliant', 'isApproved', 'hasBeenWithdrawn']
    for col in bool_columns:
        if col in opt_df.columns:
            # Handle string booleans
            opt_df[col] = opt_df[col].astype(str).str.lower()
            opt_df[col] = opt_df[col].map({
                'true': True, 
                'false': False, 
                'nan': None,
                'none': None
            }).fillna(opt_df[col])
            print(f"  ✅ Standardized boolean column: {col}")
    
    # Simplify proteinIds JSON arrays
    if 'proteinIds' in opt_df.columns:
        def simplify_protein_ids(protein_ids_str):
            if pd.isna(protein_ids_str) or protein_ids_str == '[]':
                return '[]'
            try:
                protein_data = json.loads(protein_ids_str)
                if isinstance(protein_data, list) and protein_data:
                    # Extract just the IDs, removing complex structure
                    ids = [item.get('id', '') for item in protein_data if isinstance(item, dict)]
                    return json.dumps(ids)
                return '[]'
            except:
                return '[]'
        
        opt_df['proteinIds'] = opt_df['proteinIds'].apply(simplify_protein_ids)
        print(f"  ✅ Simplified proteinIds arrays")
    
    # Handle array columns (mesh_terms, tree_numbers)
    array_columns = ['mesh_terms', 'tree_numbers']
    for col in array_columns:
        if col in opt_df.columns:
            # Ensure all array fields are proper JSON
            opt_df[col] = opt_df[col].fillna('[]')
            opt_df[col] = opt_df[col].apply(lambda x: '[]' if x == '[]' else x)
            print(f"  ✅ Standardized array column: {col}")
    
    # Fill remaining NaN values with appropriate defaults
    for col in opt_df.columns:
        if opt_df[col].dtype == 'object':
            opt_df[col] = opt_df[col].fillna('')
        else:
            opt_df[col] = opt_df[col].fillna(0)
    
    print(f"  ✅ Filled remaining NaN values")
    return opt_df

def optimize_edges_for_memgraph(df):
    """Optimize edges CSV for Memgraph import"""
    print(f"🔧 Optimizing {len(df):,} edges...")
    
    # Create optimized copy
    opt_df = df.copy()
    
    # Convert numeric strings to proper numbers
    numeric_columns = ['max_phase']
    for col in numeric_columns:
        if col in opt_df.columns:
            opt_df[col] = pd.to_numeric(opt_df[col], errors='coerce')
            print(f"  ✅ Converted {col} to numeric")
    
    # Standardize action arrays
    if 'actions' in opt_df.columns:
        def standardize_actions(actions_str):
            if pd.isna(actions_str) or actions_str == '[]':
                return '[]'
            try:
                # If it's already a list string, keep it
                if actions_str.startswith('[') and actions_str.endswith(']'):
                    return actions_str
                # Otherwise, make it a single-item array
                return json.dumps([actions_str])
            except:
                return '[]'
        
        opt_df['actions'] = opt_df['actions'].apply(standardize_actions)
        print(f"  ✅ Standardized actions arrays")
    
    # Fill remaining NaN values
    for col in opt_df.columns:
        if opt_df[col].dtype == 'object':
            opt_df[col] = opt_df[col].fillna('')
        else:
            opt_df[col] = opt_df[col].fillna(0)
    
    print(f"  ✅ Filled remaining NaN values")
    return opt_df

def main():
    """Main optimization process"""
    print("🚀 CREATING OPTIMIZED CSVS FOR MEMGRAPH")
    print("=" * 50)
    
    # Paths
    base_dir = Path("/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv")
    nodes_path = base_dir / "nodes.csv"
    edges_path = base_dir / "edges.csv"
    output_dir = base_dir / "optimized"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load and optimize nodes
    print(f"\n📊 Loading nodes from: {nodes_path}")
    nodes_df = pd.read_csv(nodes_path)
    print(f"  Loaded {len(nodes_df):,} nodes with {len(nodes_df.columns)} columns")
    
    optimized_nodes = optimize_nodes_for_memgraph(nodes_df)
    
    # Save optimized nodes
    nodes_output = output_dir / "nodes_optimized.csv"
    optimized_nodes.to_csv(nodes_output, index=False)
    print(f"  💾 Saved optimized nodes: {nodes_output}")
    print(f"     Size: {os.path.getsize(nodes_output):,} bytes")
    
    # Load and optimize edges
    print(f"\n📊 Loading edges from: {edges_path}")
    edges_df = pd.read_csv(edges_path)
    print(f"  Loaded {len(edges_df):,} edges with {len(edges_df.columns)} columns")
    
    optimized_edges = optimize_edges_for_memgraph(edges_df)
    
    # Save optimized edges
    edges_output = output_dir / "edges_optimized.csv"
    optimized_edges.to_csv(edges_output, index=False)
    print(f"  💾 Saved optimized edges: {edges_output}")
    print(f"     Size: {os.path.getsize(edges_output):,} bytes")
    
    # Create sample verification
    print(f"\n🔍 VERIFICATION SAMPLES:")
    
    # Show node samples
    print(f"\n📋 NODE SAMPLES:")
    for node_type in ['Drug', 'Target', 'Disease']:
        sample = optimized_nodes[optimized_nodes['type'] == node_type].iloc[0]
        print(f"  {node_type} ({sample['id']}):")
        print(f"    Name: {sample['name']}")
        if node_type == 'Drug':
            print(f"    Has structure: {sample.get('has_structure_data', 'N/A')}")
            print(f"    Lipinski: {sample.get('lipinski_compliant', 'N/A')}")
        elif node_type == 'Target':
            print(f"    Symbol: {sample.get('symbol', 'N/A')}")
            protein_ids = sample.get('proteinIds', '[]')
            if len(protein_ids) > 50:
                protein_ids = protein_ids[:50] + "..."
            print(f"    Protein IDs: {protein_ids}")
    
    # Show edge samples
    print(f"\n🔗 EDGE SAMPLES:")
    for edge_type in ['TARGETS', 'INDICATED_FOR', 'ASSOCIATED_WITH']:
        if edge_type in optimized_edges['type'].values:
            sample = optimized_edges[optimized_edges['type'] == edge_type].iloc[0]
            print(f"  {edge_type}:")
            print(f"    {sample['source']} -> {sample['target']}")
            print(f"    Data source: {sample.get('data_source', 'N/A')}")
    
    print(f"\n✅ OPTIMIZATION COMPLETE!")
    print(f"   📁 Optimized files ready in: {output_dir}")
    print(f"   🎯 Ready for Memgraph import!")

if __name__ == "__main__":
    main()
