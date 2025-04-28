# src/ddi/data/sources/mesh/parser.py
import os
import logging
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from tqdm import tqdm
import pickle
import json

class MeSHParser:
    """Parser for MeSH disease taxonomy and hierarchical information"""
    
    def __init__(self, mesh_dir: str, output_dir: str = None):
        """Initialize MeSH parser
        
        Args:
            mesh_dir: Directory containing MeSH XML files
            output_dir: Directory to save processed data
        """
        self.mesh_dir = mesh_dir
        self.output_dir = output_dir or "data/processed/diseases/mesh"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data containers
        self.descriptors = {}  # MeSH ID -> descriptor data
        self.qualifiers = {}   # Qualifier ID -> qualifier data
        self.disease_hierarchy = {}  # Tree number -> list of child tree numbers
        self.term_to_id = {}   # Term -> MeSH ID
        
        # Disease category prefixes
        self.disease_categories = {"C"}  # MeSH tree numbers starting with C are diseases
    
    def parse_descriptor_file(self, file_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Parse MeSH descriptor XML file
        
        Args:
            file_path: Path to descriptor XML file
            limit: Limit number of descriptors to parse (for testing)
            
        Returns:
            Dictionary of descriptor data
        """
        self.logger.info(f"Parsing descriptor file: {file_path}")
        
        # Parse XML file
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            self.logger.error(f"Error parsing XML file {file_path}: {str(e)}")
            return {}
        
        # Extract descriptor records
        descriptor_records = root.findall(".//DescriptorRecord")
        
        if limit:
            descriptor_records = descriptor_records[:limit]
            
        self.logger.info(f"Found {len(descriptor_records)} descriptor records")
        
        # Process descriptor records
        for record in tqdm(descriptor_records, desc="Processing descriptor records"):
            try:
                # Extract descriptor UI
                descriptor_ui = record.find("./DescriptorUI")
                if descriptor_ui is None:
                    continue
                    
                descriptor_id = descriptor_ui.text
                
                # Extract descriptor name
                descriptor_name_elem = record.find("./DescriptorName/String")
                if descriptor_name_elem is None:
                    continue
                    
                descriptor_name = descriptor_name_elem.text
                
                # Extract tree numbers
                tree_numbers = []
                tree_number_elems = record.findall("./TreeNumberList/TreeNumber")
                for tree_elem in tree_number_elems:
                    if tree_elem.text:
                        tree_numbers.append(tree_elem.text)
                
                # Check if this is a disease category
                is_disease = False
                for tree_number in tree_numbers:
                    category = tree_number.split('.')[0] if '.' in tree_number else tree_number
                    if category in self.disease_categories:
                        is_disease = True
                        break
                
                # Skip if not a disease and not a main category
                if not is_disease and all(tn not in self.disease_categories for tn in tree_numbers):
                    continue
                
                # Extract scope note (description)
                scope_note = None
                scope_note_elem = record.find("./ScopeNote")
                if scope_note_elem is not None:
                    scope_note = scope_note_elem.text
                
                # Extract synonyms (terms)
                synonyms = []
                concept_elems = record.findall("./ConceptList/Concept")
                for concept in concept_elems:
                    term_elems = concept.findall("./TermList/Term")
                    for term in term_elems:
                        term_string = term.find("./String")
                        if term_string is not None and term_string.text:
                            synonyms.append(term_string.text)
                
                # Extract allowed qualifiers
                allowed_qualifiers = []
                qualifier_elems = record.findall("./AllowableQualifierList/AllowableQualifier/QualifierReferredTo/QualifierUI")
                for qualifier in qualifier_elems:
                    if qualifier.text:
                        allowed_qualifiers.append(qualifier.text)
                
                # Store descriptor data
                self.descriptors[descriptor_id] = {
                    "id": descriptor_id,
                    "name": descriptor_name,
                    "tree_numbers": tree_numbers,
                    "description": scope_note,
                    "synonyms": list(set(synonyms)),  # Remove duplicates
                    "allowed_qualifiers": allowed_qualifiers,
                    "is_disease": is_disease
                }
                
                # Update term to ID mapping
                self.term_to_id[descriptor_name.lower()] = descriptor_id
                for synonym in synonyms:
                    if synonym.lower() not in self.term_to_id:
                        self.term_to_id[synonym.lower()] = descriptor_id
                
                # Update disease hierarchy
                for tree_number in tree_numbers:
                    parts = tree_number.split('.')
                    for i in range(1, len(parts)):
                        parent = '.'.join(parts[:i])
                        child = '.'.join(parts[:i+1])
                        
                        if parent not in self.disease_hierarchy:
                            self.disease_hierarchy[parent] = []
                            
                        if child not in self.disease_hierarchy[parent]:
                            self.disease_hierarchy[parent].append(child)
                
            except Exception as e:
                self.logger.warning(f"Error processing descriptor record: {str(e)}")
                continue
        
        self.logger.info(f"Processed {len(self.descriptors)} descriptor records")
        return self.descriptors
    
    def parse_qualifier_file(self, file_path: str) -> Dict[str, Any]:
        """Parse MeSH qualifier XML file
        
        Args:
            file_path: Path to qualifier XML file
            
        Returns:
            Dictionary of qualifier data
        """
        self.logger.info(f"Parsing qualifier file: {file_path}")
        
        # Parse XML file
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            self.logger.error(f"Error parsing XML file {file_path}: {str(e)}")
            return {}
        
        # Extract qualifier records
        qualifier_records = root.findall(".//QualifierRecord")
        self.logger.info(f"Found {len(qualifier_records)} qualifier records")
        
        # Process qualifier records
        for record in tqdm(qualifier_records, desc="Processing qualifier records"):
            try:
                # Extract qualifier UI
                qualifier_ui = record.find("./QualifierUI")
                if qualifier_ui is None:
                    continue
                    
                qualifier_id = qualifier_ui.text
                
                # Extract qualifier name
                qualifier_name_elem = record.find("./QualifierName/String")
                if qualifier_name_elem is None:
                    continue
                    
                qualifier_name = qualifier_name_elem.text
                
                # Extract tree numbers
                tree_numbers = []
                tree_number_elems = record.findall("./TreeNumberList/TreeNumber")
                for tree_elem in tree_number_elems:
                    if tree_elem.text:
                        tree_numbers.append(tree_elem.text)
                
                # Extract scope note (description)
                scope_note = None
                scope_note_elem = record.find("./ScopeNote")
                if scope_note_elem is not None:
                    scope_note = scope_note_elem.text
                
                # Store qualifier data
                self.qualifiers[qualifier_id] = {
                    "id": qualifier_id,
                    "name": qualifier_name,
                    "tree_numbers": tree_numbers,
                    "description": scope_note
                }
                
            except Exception as e:
                self.logger.warning(f"Error processing qualifier record: {str(e)}")
                continue
        
        self.logger.info(f"Processed {len(self.qualifiers)} qualifier records")
        return self.qualifiers
    
    def parse_latest_mesh(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Parse the latest MeSH data
        
        Args:
            limit: Limit number of descriptors to parse (for testing)
            
        Returns:
            Dictionary containing processed MeSH data
        """
        # Find the latest desc and qual files
        desc_files = sorted([f for f in os.listdir(self.mesh_dir) if f.startswith("desc") and f.endswith(".xml")], reverse=True)
        qual_files = sorted([f for f in os.listdir(self.mesh_dir) if f.startswith("qual") and f.endswith(".xml")], reverse=True)
        
        if not desc_files or not qual_files:
            self.logger.error(f"No MeSH files found in {self.mesh_dir}")
            return {}
        
        latest_desc = desc_files[0]
        latest_qual = qual_files[0]
        
        self.logger.info(f"Using latest descriptor file: {latest_desc}")
        self.logger.info(f"Using latest qualifier file: {latest_qual}")
        
        # Parse descriptor file
        desc_path = os.path.join(self.mesh_dir, latest_desc)
        self.parse_descriptor_file(desc_path, limit=limit)
        
        # Parse qualifier file
        qual_path = os.path.join(self.mesh_dir, latest_qual)
        self.parse_qualifier_file(qual_path)
        
        # Process disease hierarchy
        self._process_disease_hierarchy()
        
        # Prepare result
        result = {
            "descriptors": self.descriptors,
            "qualifiers": self.qualifiers,
            "disease_hierarchy": self.disease_hierarchy,
            "term_to_id": self.term_to_id,
            "version": latest_desc.replace("desc", "").replace(".xml", "")
        }
        
        return result
    
    def _process_disease_hierarchy(self) -> None:
        """Process disease hierarchy to add additional metadata"""
        # Add descriptor information to hierarchy
        hierarchy_with_info = {}
        
        for tree_number, children in self.disease_hierarchy.items():
            # Find descriptor for this tree number
            descriptor_id = None
            for d_id, descriptor in self.descriptors.items():
                if tree_number in descriptor.get("tree_numbers", []):
                    descriptor_id = d_id
                    break
            
            # If no descriptor found, use first part of tree number
            if descriptor_id is None:
                # Skip non-category tree numbers without descriptors
                if '.' in tree_number:
                    continue
                
                # For category tree numbers, use generic name
                category_name = f"Category {tree_number}"
                hierarchy_with_info[tree_number] = {
                    "tree_number": tree_number,
                    "name": category_name,
                    "descriptor_id": None,
                    "children": children
                }
            else:
                descriptor = self.descriptors[descriptor_id]
                hierarchy_with_info[tree_number] = {
                    "tree_number": tree_number,
                    "name": descriptor["name"],
                    "descriptor_id": descriptor_id,
                    "children": children
                }
        
        self.disease_hierarchy = hierarchy_with_info
    
    def extract_disease_taxonomy(self) -> Dict[str, Any]:
        """Extract disease taxonomy from processed MeSH data
        
        Returns:
            Dictionary of disease taxonomy
        """
        self.logger.info("Extracting disease taxonomy")
        
        taxonomy = {}
        
        # Get all disease descriptors
        disease_descriptors = {d_id: desc for d_id, desc in self.descriptors.items() if desc.get("is_disease", False)}
        self.logger.info(f"Found {len(disease_descriptors)} disease descriptors")
        
        # Extract taxonomy
        for d_id, descriptor in tqdm(disease_descriptors.items(), desc="Extracting disease taxonomy"):
            tree_numbers = descriptor.get("tree_numbers", [])
            
            # Filter for disease tree numbers (starting with C)
            disease_tree_numbers = [tn for tn in tree_numbers if tn.startswith(tuple(self.disease_categories))]
            
            if not disease_tree_numbers:
                continue
            
            # Get parents for each tree number
            parents = []
            for tree_number in disease_tree_numbers:
                parts = tree_number.split('.')
                if len(parts) > 1:
                    parent = '.'.join(parts[:-1])
                    
                    # Find descriptor for parent
                    parent_id = None
                    for pd_id, pdesc in self.descriptors.items():
                        if parent in pdesc.get("tree_numbers", []):
                            parent_id = pd_id
                            parents.append({
                                "id": parent_id,
                                "name": pdesc["name"],
                                "tree_number": parent
                            })
                            break
            
            # Create taxonomy entry
            taxonomy[d_id] = {
                "id": d_id,
                "name": descriptor["name"],
                "description": descriptor.get("description", ""),
                "tree_numbers": disease_tree_numbers,
                "synonyms": descriptor.get("synonyms", []),
                "parents": parents,
                "is_top_level": all(tn.count('.') == 0 for tn in disease_tree_numbers)
            }
        
        self.logger.info(f"Extracted taxonomy for {len(taxonomy)} diseases")
        return taxonomy
    
    def save_mesh_data(self, format: str = "pickle") -> str:
        """Save processed MeSH data
        
        Args:
            format: Output format (pickle or json)
            
        Returns:
            Path to saved file
        """
        # Prepare data
        result = {
            "descriptors": self.descriptors,
            "qualifiers": self.qualifiers,
            "disease_hierarchy": self.disease_hierarchy,
            "term_to_id": self.term_to_id,
            "version": "latest"
        }
        
        # Save data
        output_path = os.path.join(self.output_dir, f"mesh_data.{format}")
        
        if format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
        else:
            self.logger.error(f"Unsupported output format: {format}")
            return None
            
        self.logger.info(f"Saved MeSH data to {output_path}")
        return output_path
    
    def save_disease_taxonomy(self, format: str = "pickle") -> str:
        """Save disease taxonomy
        
        Args:
            format: Output format (pickle or json)
            
        Returns:
            Path to saved file
        """
        # Extract taxonomy
        taxonomy = self.extract_disease_taxonomy()
        
        # Save taxonomy
        output_path = os.path.join(self.output_dir, f"disease_taxonomy.{format}")
        
        if format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(taxonomy, f)
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(taxonomy, f, indent=2)
        else:
            self.logger.error(f"Unsupported output format: {format}")
            return None
            
        self.logger.info(f"Saved disease taxonomy to {output_path}")
        return output_path


# Example usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse MeSH descriptor and qualifier files")
    parser.add_argument("--mesh_dir", required=True, help="Directory containing MeSH XML files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output format")
    parser.add_argument("--limit", type=int, help="Limit number of descriptors to parse (for testing)")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize parser
    parser = MeSHParser(args.mesh_dir, args.output_dir)
    
    # Parse MeSH data
    parser.parse_latest_mesh(limit=args.limit)
    
    # Save data
    parser.save_mesh_data(format=args.format)
    parser.save_disease_taxonomy(format=args.format)
    
    logging.info("MeSH parsing complete")

if __name__ == "__main__":
    main()