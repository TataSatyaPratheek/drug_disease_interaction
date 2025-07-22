#!/bin/bash

# Parallel sync checker wrapper using GNU parallel
# This script can be used for extremely large datasets where even more parallelization is needed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPINGS_DIR="/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings"

echo "Starting parallel sync checker..."

# Function to process a single mapping file
process_mapping_file() {
    local mapping_file="$1"
    local temp_dir="/tmp/sync_checker_$$"
    mkdir -p "$temp_dir"
    
    echo "Processing $mapping_file in parallel..."
    
    # Extract just the filename without path
    filename=$(basename "$mapping_file")
    
    # Create a temporary Python script for this specific mapping
    cat > "$temp_dir/process_${filename}.py" << EOF
import json
import sys
import os

mapping_file = "$mapping_file"
try:
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    print(f"${filename}: {len(data)} mappings loaded")
    
    # You can add specific processing logic here
    # For now, just report the count
    
except Exception as e:
    print(f"Error processing ${filename}: {e}")
    sys.exit(1)
EOF
    
    # Run the processing script
    python "$temp_dir/process_${filename}.py"
    
    # Clean up
    rm -rf "$temp_dir"
}

# Export the function so parallel can use it
export -f process_mapping_file

# Find all JSON mapping files
JSON_FILES=(
    "$MAPPINGS_DIR/ensembl_to_uniprot_full.json"
    "$MAPPINGS_DIR/ensembl_protein_to_uniprot.json"
    "$MAPPINGS_DIR/gene_name_to_uniprot.json"
    "$MAPPINGS_DIR/refseq_to_uniprot.json"
    "$MAPPINGS_DIR/comprehensive_id_mappings.json"
)

# Check if parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for maximum performance..."
    
    # Process files in parallel (limit to 4 jobs to avoid overwhelming the system)
    printf '%s\n' "${JSON_FILES[@]}" | parallel -j4 process_mapping_file {}
    
    echo "Parallel processing complete. Now running main sync checker..."
else
    echo "GNU parallel not available. Using standard Python threading..."
fi

# Run the main sync checker script
echo "Running enhanced sync checker with streaming and chunking..."
cd "$SCRIPT_DIR"
python src/scripts/sync_checker.py

echo "Sync checking complete!"
