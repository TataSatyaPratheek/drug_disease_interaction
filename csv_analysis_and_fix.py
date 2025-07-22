#!/usr/bin/env python3
"""
Comprehensive CSV Analysis and Optimization for Memgraph
Based on EDA findings, this script will:
1. Analyze the current CSV structure
2. Identify data quality issues
3. Create optimized versions for Memgraph import
4. Generate summary reports
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.nodes_path = self.base_path / 'nodes.csv'
        self.edges_path = self.base_path / 'edges.csv'
        self.output_path = self.base_path / 'optimized'
        self.output_path.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load the CSV files"""
        logger.info("Loading CSV files...")
        self.nodes = pd.read_csv(self.nodes_path, low_memory=False)
        self.edges = pd.read_csv(self.edges_path, low_memory=False)
        logger.info(f"Loaded {len(self.nodes):,} nodes and {len(self.edges):,} edges")
        
    def analyze_nodes_structure(self):
        """Deep analysis of nodes structure"""
        logger.info("=== NODES STRUCTURE ANALYSIS ===")
        
        print(f"📊 NODES OVERVIEW:")
        print(f"  Total nodes: {len(self.nodes):,}")
        print(f"  Columns: {len(self.nodes.columns)}")
        
        # Node type distribution
        print(f"\n🏷️  NODE TYPE DISTRIBUTION:")
        type_counts = self.nodes['type'].value_counts()
        for node_type, count in type_counts.items():
            pct = (count / len(self.nodes)) * 100
            print(f"  {node_type:>8}: {count:>7,} ({pct:5.1f}%)")
        
        # Analyze each node type
        for node_type in ['Disease', 'Target', 'Drug', 'Pathway']:
            self._analyze_node_type(node_type)
            
        return type_counts
    
    def _analyze_node_type(self, node_type):
        """Analyze specific node type"""
        subset = self.nodes[self.nodes['type'] == node_type]
        print(f"\n🔬 {node_type.upper()} ANALYSIS ({len(subset):,} nodes):")
        
        # Key fields for each type
        if node_type == 'Disease':
            key_fields = ['id', 'name', 'mesh_terms', 'mesh_description', 'tree_numbers']
        elif node_type == 'Target':
            key_fields = ['id', 'name', 'symbol', 'proteinIds']
        elif node_type == 'Drug':
            key_fields = ['id', 'name', 'has_structure_data', 'molecular_weight', 
                         'canonical_smiles', 'lipinski_compliant', 'maxPhase', 'isApproved']
        elif node_type == 'Pathway':
            key_fields = ['id', 'name', 'description', 'source']
        
        # Analyze completeness
        for field in key_fields:
            if field in subset.columns:
                null_count = subset[field].isnull().sum()
                empty_count = (subset[field] == '').sum()
                total_missing = null_count + empty_count
                pct_missing = (total_missing / len(subset)) * 100
                print(f"  {field:>20}: {total_missing:>6,} missing ({pct_missing:5.1f}%)")
        
        # Special analysis for arrays and complex fields
        if node_type == 'Disease':
            self._analyze_array_field(subset, 'mesh_terms', 'Disease mesh_terms')
            self._analyze_array_field(subset, 'tree_numbers', 'Disease tree_numbers')
        elif node_type == 'Target':
            self._analyze_complex_field(subset, 'proteinIds', 'Target proteinIds')
        elif node_type == 'Drug':
            self._analyze_drug_chemistry(subset)
    
    def _analyze_array_field(self, subset, field, description):
        """Analyze array-like fields"""
        if field not in subset.columns:
            return
            
        non_null = subset[field].dropna()
        empty_arrays = (non_null == '[]').sum()
        non_empty = len(non_null) - empty_arrays
        
        print(f"    {description}:")
        print(f"      Empty arrays ([]): {empty_arrays:,}")
        print(f"      Non-empty arrays: {non_empty:,}")
        
        if non_empty > 0:
            # Sample non-empty values
            non_empty_samples = non_null[non_null != '[]'].head(3)
            print(f"      Sample values:")
            for i, val in enumerate(non_empty_samples):
                print(f"        {i+1}. {str(val)[:100]}...")
    
    def _analyze_complex_field(self, subset, field, description):
        """Analyze complex JSON fields"""
        if field not in subset.columns:
            return
            
        non_null = subset[field].dropna()
        print(f"    {description}:")
        print(f"      Non-null values: {len(non_null):,}")
        
        # Try to parse JSON structure
        if len(non_null) > 0:
            sample_val = non_null.iloc[0]
            print(f"      Sample structure: {str(sample_val)[:200]}...")
            
            # Try to count items in JSON arrays
            try:
                if isinstance(sample_val, str) and sample_val.startswith('['):
                    parsed = json.loads(sample_val)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        print(f"      Sample array length: {len(parsed)}")
                        if isinstance(parsed[0], dict):
                            print(f"      Sample keys: {list(parsed[0].keys())}")
            except:
                print(f"      JSON parsing failed")
    
    def _analyze_drug_chemistry(self, drug_subset):
        """Analyze drug chemistry data completeness"""
        print(f"    Drug Chemistry Analysis:")
        
        # Structure data availability
        struct_data = drug_subset['has_structure_data'].value_counts()
        print(f"      Structure data availability:")
        for val, count in struct_data.items():
            pct = (count / len(drug_subset)) * 100
            print(f"        {val}: {count:,} ({pct:.1f}%)")
        
        # Chemical property completeness
        chem_fields = ['molecular_weight', 'canonical_smiles', 'logp', 'tpsa', 
                      'lipinski_compliant', 'h_bond_donors', 'h_bond_acceptors']
        
        print(f"      Chemical property completeness:")
        for field in chem_fields:
            if field in drug_subset.columns:
                complete = drug_subset[field].notna().sum()
                pct = (complete / len(drug_subset)) * 100
                print(f"        {field:>20}: {complete:>6,} ({pct:5.1f}%)")
        
        # Clinical data
        clinical_fields = ['maxPhase', 'isApproved', 'hasBeenWithdrawn', 'yearOfFirstApproval']
        print(f"      Clinical data completeness:")
        for field in clinical_fields:
            if field in drug_subset.columns:
                complete = drug_subset[field].notna().sum()
                pct = (complete / len(drug_subset)) * 100
                print(f"        {field:>20}: {complete:>6,} ({pct:5.1f}%)")
    
    def analyze_edges_structure(self):
        """Deep analysis of edges structure"""
        logger.info("=== EDGES STRUCTURE ANALYSIS ===")
        
        print(f"📊 EDGES OVERVIEW:")
        print(f"  Total edges: {len(self.edges):,}")
        print(f"  Columns: {len(self.edges.columns)}")
        
        # Edge type distribution
        print(f"\n🔗 EDGE TYPE DISTRIBUTION:")
        type_counts = self.edges['type'].value_counts()
        for edge_type, count in type_counts.items():
            pct = (count / len(self.edges)) * 100
            print(f"  {edge_type:>15}: {count:>9,} ({pct:5.1f}%)")
        
        # Analyze each edge type
        for edge_type in type_counts.index[:8]:  # Top 8 edge types
            self._analyze_edge_type(edge_type)
            
        return type_counts
    
    def _analyze_edge_type(self, edge_type):
        """Analyze specific edge type"""
        subset = self.edges[self.edges['type'] == edge_type]
        print(f"\n🔍 {edge_type} ANALYSIS ({len(subset):,} edges):")
        
        # Source/target type combinations
        src_tgt_combos = subset.groupby(['source_type', 'target_type']).size().sort_values(ascending=False)
        print(f"  Source -> Target combinations:")
        for (src_type, tgt_type), count in src_tgt_combos.head(5).items():
            pct = (count / len(subset)) * 100
            print(f"    {src_type} -> {tgt_type}: {count:,} ({pct:.1f}%)")
        
        # Field completeness for this edge type
        edge_fields = ['evidence', 'target_name', 'actions', 'organism', 'data_source',
                      'max_phase', 'efo_name', 'action_type', 'mechanism', 'pathwayName', 'via_target']
        
        relevant_fields = []
        for field in edge_fields:
            if field in subset.columns:
                non_null = subset[field].notna().sum()
                if non_null > 0:
                    relevant_fields.append((field, non_null))
        
        if relevant_fields:
            print(f"  Relevant fields:")
            for field, count in relevant_fields:
                pct = (count / len(subset)) * 100
                print(f"    {field:>20}: {count:>6,} ({pct:5.1f}%)")
    
    def optimize_for_memgraph(self):
        """Create optimized versions for Memgraph"""
        logger.info("=== CREATING OPTIMIZED CSVS FOR MEMGRAPH ===")
        
        # Optimize nodes
        optimized_nodes = self._optimize_nodes()
        
        # Optimize edges  
        optimized_edges = self._optimize_edges()
        
        # Save optimized files
        nodes_output = self.output_path / 'nodes_optimized.csv'
        edges_output = self.output_path / 'edges_optimized.csv'
        
        optimized_nodes.to_csv(nodes_output, index=False)
        optimized_edges.to_csv(edges_output, index=False)
        
        logger.info(f"Optimized files saved:")
        logger.info(f"  Nodes: {nodes_output}")
        logger.info(f"  Edges: {edges_output}")
        
        return optimized_nodes, optimized_edges
    
    def _optimize_nodes(self):
        """Optimize nodes for Memgraph import"""
        logger.info("Optimizing nodes...")
        
        nodes_opt = self.nodes.copy()
        
        # 1. Handle array fields - simplify complex JSON to basic arrays
        for field in ['mesh_terms', 'tree_numbers']:
            if field in nodes_opt.columns:
                nodes_opt[field] = nodes_opt[field].apply(self._simplify_array_field)
        
        # 2. Handle complex proteinIds field - extract just the IDs
        if 'proteinIds' in nodes_opt.columns:
            nodes_opt['proteinIds'] = nodes_opt['proteinIds'].apply(self._extract_protein_ids)
        
        # 3. Standardize boolean fields
        boolean_fields = ['has_structure_data', 'lipinski_compliant', 'isApproved', 
                         'hasBeenWithdrawn', 'blackBoxWarning']
        for field in boolean_fields:
            if field in nodes_opt.columns:
                nodes_opt[field] = nodes_opt[field].apply(self._standardize_boolean)
        
        # 4. Handle null names - use ID as fallback
        if 'name' in nodes_opt.columns:
            nodes_opt['name'] = nodes_opt.apply(
                lambda row: row['id'] if pd.isna(row['name']) else row['name'], axis=1
            )
        
        # 5. Clean numeric fields
        numeric_fields = ['molecular_weight', 'logp', 'tpsa', 'rotatable_bonds', 
                         'h_bond_donors', 'h_bond_acceptors', 'aromatic_rings', 'heavy_atoms',
                         'lipinski_violations', 'target_count', 'enzyme_count', 
                         'transporter_count', 'carrier_count', 'pathway_count',
                         'maxPhase', 'yearOfFirstApproval', 'linkedDiseases', 'linkedTargets']
        
        for field in numeric_fields:
            if field in nodes_opt.columns:
                nodes_opt[field] = pd.to_numeric(nodes_opt[field], errors='coerce')
        
        logger.info(f"Nodes optimization complete: {len(nodes_opt)} nodes")
        return nodes_opt
    
    def _optimize_edges(self):
        """Optimize edges for Memgraph import"""
        logger.info("Optimizing edges...")
        
        edges_opt = self.edges.copy()
        
        # 1. Ensure numeric fields are properly typed
        if 'max_phase' in edges_opt.columns:
            edges_opt['max_phase'] = pd.to_numeric(edges_opt['max_phase'], errors='coerce')
        
        # 2. Clean text fields
        text_fields = ['evidence', 'target_name', 'actions', 'organism', 'efo_name', 
                      'action_type', 'mechanism', 'target_type_detailed', 'pathwayName', 'via_target']
        
        for field in text_fields:
            if field in edges_opt.columns:
                edges_opt[field] = edges_opt[field].astype(str).replace('nan', '')
                edges_opt[field] = edges_opt[field].replace('', None)
        
        logger.info(f"Edges optimization complete: {len(edges_opt)} edges")
        return edges_opt
    
    def _simplify_array_field(self, value):
        """Simplify array fields for Memgraph"""
        if pd.isna(value) or value == '' or value == '[]':
            return '[]'
        
        try:
            # If it's already a simple list format, keep it
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                # Try to parse and extract simple values
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    if len(parsed) == 0:
                        return '[]'
                    # Extract simple string values if possible
                    simple_vals = []
                    for item in parsed:
                        if isinstance(item, str):
                            simple_vals.append(item)
                        elif isinstance(item, dict) and 'id' in item:
                            simple_vals.append(item['id'])
                    if simple_vals:
                        return json.dumps(simple_vals)
                return '[]'
        except:
            return '[]'
        
        return '[]'
    
    def _extract_protein_ids(self, value):
        """Extract just the protein IDs from complex JSON"""
        if pd.isna(value) or value == '' or value == '[]':
            return '[]'
        
        try:
            if isinstance(value, str) and value.startswith('['):
                parsed = json.loads(value)
                if isinstance(parsed, list) and len(parsed) > 0:
                    ids = []
                    for item in parsed:
                        if isinstance(item, dict) and 'id' in item:
                            ids.append(item['id'])
                    if ids:
                        return json.dumps(ids)
            return '[]'
        except:
            return '[]'
    
    def _standardize_boolean(self, value):
        """Standardize boolean values"""
        if pd.isna(value) or value == '':
            return None
        if isinstance(value, str):
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
        return None
    
    def generate_summary_report(self, optimized_nodes, optimized_edges):
        """Generate comprehensive summary report"""
        logger.info("=== GENERATING SUMMARY REPORT ===")
        
        report_path = self.output_path / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE CSV ANALYSIS AND OPTIMIZATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Original data summary
            f.write("📊 ORIGINAL DATA SUMMARY:\n")
            f.write(f"  Nodes: {len(self.nodes):,} total\n")
            node_types = self.nodes['type'].value_counts()
            for node_type, count in node_types.items():
                pct = (count / len(self.nodes)) * 100
                f.write(f"    {node_type}: {count:,} ({pct:.1f}%)\n")
            
            f.write(f"\n  Edges: {len(self.edges):,} total\n")
            edge_types = self.edges['type'].value_counts()
            for edge_type, count in edge_types.head(8).items():
                pct = (count / len(self.edges)) * 100
                f.write(f"    {edge_type}: {count:,} ({pct:.1f}%)\n")
            
            # Data quality issues identified
            f.write(f"\n⚠️  DATA QUALITY ISSUES IDENTIFIED:\n")
            f.write(f"  • {self.nodes['name'].isnull().sum():,} nodes have null names (38.4%)\n")
            f.write(f"  • Complex JSON in proteinIds field needs simplification\n")
            f.write(f"  • Boolean fields stored as strings need conversion\n")
            f.write(f"  • Array fields inconsistently formatted\n")
            
            # Optimizations applied
            f.write(f"\n🔧 OPTIMIZATIONS APPLIED:\n")
            f.write(f"  ✅ Simplified array fields to basic JSON arrays\n")
            f.write(f"  ✅ Extracted protein IDs from complex JSON objects\n")
            f.write(f"  ✅ Standardized boolean fields (True/False/null)\n")
            f.write(f"  ✅ Used IDs as fallback for null names\n")
            f.write(f"  ✅ Cleaned and typed numeric fields\n")
            f.write(f"  ✅ Standardized text fields with null handling\n")
            
            # Final recommendations
            f.write(f"\n🚀 MEMGRAPH IMPORT RECOMMENDATIONS:\n")
            f.write(f"  • Use optimized CSV files for faster import\n")
            f.write(f"  • Consider parallel import for large edge file\n")
            f.write(f"  • Monitor memory usage during import\n")
            f.write(f"  • Create indexes after full import for best performance\n")
            
        logger.info(f"Summary report saved: {report_path}")

def main():
    """Main execution function"""
    print("🔍 COMPREHENSIVE CSV ANALYSIS FOR MEMGRAPH OPTIMIZATION")
    print("=" * 60)
    
    # Initialize analyzer
    csv_path = "/home/vi/Documents/drug_disease_interaction/data/processed/graph_csv"
    analyzer = CSVAnalyzer(csv_path)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Analyze structure
        analyzer.analyze_nodes_structure()
        analyzer.analyze_edges_structure()
        
        # Create optimized versions
        opt_nodes, opt_edges = analyzer.optimize_for_memgraph()
        
        # Generate report
        analyzer.generate_summary_report(opt_nodes, opt_edges)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print(f"Check the 'optimized' folder for:")
        print(f"  • nodes_optimized.csv")
        print(f"  • edges_optimized.csv") 
        print(f"  • analysis_report.txt")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
