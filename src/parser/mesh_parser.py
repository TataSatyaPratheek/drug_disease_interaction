# src/ddi/data/sources/mesh/parser.py
import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
from tqdm import tqdm
import pickle
import json

class MeSHParser:
    """
    Parser for MeSH disease taxonomy and hierarchical information, supporting multiple years
    (2020-2025) including descriptors, qualifiers, and supplementary records.
    """

    def __init__(self, mesh_dir: str, output_dir: str = None):
        """Initialize MeSH parser

        Args:
            mesh_dir: Directory containing MeSH XML files (supports desc/qual/supp YYYY.xml format)
            output_dir: Directory to save processed data
        """
        self.mesh_dir = mesh_dir
        self.output_dir = output_dir or "data/processed/mesh"
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Data containers - Initialized/reset in parse_mesh
        self.descriptors = {}
        self.qualifiers = {}
        self.supplementary = {}
        self.disease_hierarchy = {}
        self.term_to_id = {}

        # Disease category prefixes (expanded for comprehensive coverage)
        self.disease_categories = {"C"}  # MeSH tree numbers starting with C are diseases
        
        # Available years (auto-detected from files)
        self.available_years = self._detect_available_years()

    def _detect_available_years(self) -> list:
        """Detect available MeSH years from files in the directory"""
        years = set()
        if os.path.exists(self.mesh_dir):
            for filename in os.listdir(self.mesh_dir):
                if filename.endswith('.xml'):
                    # Extract year from filename (e.g., desc2025.xml -> 2025)
                    parts = filename.replace('.xml', '')
                    if len(parts) >= 4 and parts[-4:].isdigit():
                        years.add(int(parts[-4:]))
        
        available_years = sorted(list(years))
        self.logger.info(f"Detected MeSH years: {available_years}")
        return available_years

    def _parse_xml_robustly(self, file_path: str) -> Optional[ET.Element]:
        """Attempts to parse an XML file, trying standard ET and then lxml if available."""
        try:
            tree = ET.parse(file_path)
            return tree.getroot()
        except ET.ParseError as e_et:
            self.logger.warning(f"Standard XML parsing failed for {file_path}: {e_et}. Trying lxml...")
            try:
                from lxml import etree
                # Use recover=True to handle potential errors more gracefully
                parser = etree.XMLParser(recover=True, encoding='utf-8')
                tree = etree.parse(file_path, parser)
                # Convert back to standard ET root element if needed downstream,
                # or adjust downstream code to handle lxml elements.
                # For simplicity here, we'll assume downstream uses standard ET methods.
                # If lxml is used, downstream findall might need namespace handling if present.
                # Let's return the lxml root for now, assuming downstream can handle it or we adapt.
                # Re-parsing with standard ET after recovery might be safer if strict ET compatibility is needed.
                # For now, let's just return the lxml root.
                # return tree.getroot()
                # --- Safer approach: re-parse the string representation ---
                xml_bytes = etree.tostring(tree.getroot())
                return ET.fromstring(xml_bytes)

            except ImportError:
                self.logger.error(f"lxml not available. Cannot recover from parsing error in {file_path}.")
                return None
            except Exception as e_lxml:
                self.logger.error(f"lxml parsing also failed for {file_path}: {e_lxml}")
                return None
        except Exception as e_other:
             self.logger.error(f"An unexpected error occurred during parsing of {file_path}: {e_other}")
             return None


    def parse_descriptor_file(self, file_path: str, limit: Optional[int] = None) -> None:
        """Parse MeSH descriptor XML file (desc2025.xml) and update internal state.

        Args:
            file_path: Path to descriptor XML file.
            limit: Limit number of descriptors to parse (for testing).
        """
        self.logger.info(f"Parsing descriptor file: {file_path}")
        root = self._parse_xml_robustly(file_path)
        if root is None:
            self.logger.error(f"Failed to parse descriptor file: {file_path}")
            return # Cannot proceed without descriptors

        # Extract descriptor records
        # Use .// to find anywhere in the tree, robust to slight structure variations
        descriptor_records = root.findall(".//DescriptorRecord")

        if limit:
            descriptor_records = descriptor_records[:limit]

        self.logger.info(f"Found {len(descriptor_records)} descriptor records to process.")

        # Process descriptor records
        processed_count = 0
        skipped_count = 0
        for record in tqdm(descriptor_records, desc="Processing descriptor records"):
            try:
                # Extract descriptor UI
                descriptor_ui_elem = record.find("./DescriptorUI")
                if descriptor_ui_elem is None or not descriptor_ui_elem.text:
                    skipped_count += 1
                    continue
                descriptor_id = descriptor_ui_elem.text

                # Extract descriptor name
                descriptor_name_elem = record.find("./DescriptorName/String")
                if descriptor_name_elem is None or not descriptor_name_elem.text:
                    skipped_count += 1
                    continue
                descriptor_name = descriptor_name_elem.text

                # Extract tree numbers
                tree_numbers = []
                tree_number_list = record.find("./TreeNumberList")
                if tree_number_list is not None:
                    for tree_elem in tree_number_list.findall("./TreeNumber"):
                        if tree_elem.text:
                            tree_numbers.append(tree_elem.text.strip()) # Ensure no leading/trailing whitespace

                # Check if this is a disease category (starts with 'C')
                is_relevant_category = any(tn.startswith(tuple(self.disease_categories)) for tn in tree_numbers)

                # --- Filter: Only process disease-related descriptors ---
                if not is_relevant_category:
                    skipped_count += 1
                    continue
                # --------------------------------------------------------

                # Determine if it's specifically a disease (for now, any 'C' branch node)
                is_disease_node = is_relevant_category

                # Extract scope note (description)
                scope_note = None
                scope_note_elem = record.find("./ScopeNote")
                if scope_note_elem is not None and scope_note_elem.text:
                    scope_note = scope_note_elem.text.strip()

                # Extract synonyms (terms)
                synonyms = set() # Use a set to avoid duplicates initially
                concept_list = record.find("./ConceptList")
                if concept_list is not None:
                    for concept in concept_list.findall("./Concept"):
                        term_list = concept.find("./TermList")
                        if term_list is not None:
                            for term in term_list.findall("./Term"):
                                term_string_elem = term.find("./String")
                                if term_string_elem is not None and term_string_elem.text:
                                    synonyms.add(term_string_elem.text.strip())

                # Extract allowed qualifiers
                allowed_qualifiers = []
                qualifier_list = record.find("./AllowableQualifierList")
                if qualifier_list is not None:
                    for qualifier in qualifier_list.findall("./AllowableQualifier"):
                        qualifier_ref = qualifier.find("./QualifierReferredTo/QualifierUI")
                        if qualifier_ref is not None and qualifier_ref.text:
                            allowed_qualifiers.append(qualifier_ref.text.strip())

                # Store descriptor data
                self.descriptors[descriptor_id] = {
                    "id": descriptor_id,
                    "name": descriptor_name,
                    "tree_numbers": tree_numbers,
                    "description": scope_note,
                    "synonyms": sorted(list(synonyms)),  # Store as sorted list
                    "allowed_qualifiers": allowed_qualifiers,
                    "is_disease": is_disease_node # Store the flag
                }

                # Update term to ID mapping (only for relevant descriptors)
                self.term_to_id[descriptor_name.lower()] = descriptor_id
                for synonym in synonyms:
                    if isinstance(synonym, str) and synonym.lower() not in self.term_to_id:
                        self.term_to_id[synonym.lower()] = descriptor_id

                # Update disease hierarchy (only for relevant tree numbers)
                for tree_number in tree_numbers:
                    if tree_number.startswith(tuple(self.disease_categories)):
                        parts = tree_number.split('.')
                        for i in range(1, len(parts)):
                            parent = '.'.join(parts[:i])
                            child = '.'.join(parts[:i+1])

                            if parent not in self.disease_hierarchy:
                                self.disease_hierarchy[parent] = {"children": []} # Store as dict for later info

                            # Avoid adding duplicates if a node appears multiple times
                            if child not in [c["tree_number"] for c in self.disease_hierarchy[parent]["children"]]:
                                self.disease_hierarchy[parent]["children"].append({"tree_number": child}) # Store child TN

                processed_count += 1

            except Exception as e:
                self.logger.warning(f"Error processing descriptor record near UI {descriptor_id if 'descriptor_id' in locals() else 'unknown'}: {e}", exc_info=True)
                skipped_count += 1
                continue

        self.logger.info(f"Finished processing descriptor file. Processed: {processed_count}, Skipped/Filtered: {skipped_count}")


    def parse_qualifier_file(self, file_path: str) -> None:
        """Parse MeSH qualifier XML file (qual2025.xml) and update internal state.

        Args:
            file_path: Path to qualifier XML file.
        """
        self.logger.info(f"Parsing qualifier file: {file_path}")
        root = self._parse_xml_robustly(file_path)
        if root is None:
            self.logger.error(f"Failed to parse qualifier file: {file_path}")
            return # Qualifiers might be optional, but log error

        # Extract qualifier records
        qualifier_records = root.findall(".//QualifierRecord")
        self.logger.info(f"Found {len(qualifier_records)} qualifier records to process.")

        # Process qualifier records
        processed_count = 0
        skipped_count = 0
        for record in tqdm(qualifier_records, desc="Processing qualifier records"):
            try:
                # Extract qualifier UI
                qualifier_ui_elem = record.find("./QualifierUI")
                if qualifier_ui_elem is None or not qualifier_ui_elem.text:
                    skipped_count += 1
                    continue
                qualifier_id = qualifier_ui_elem.text

                # Extract qualifier name
                qualifier_name_elem = record.find("./QualifierName/String")
                if qualifier_name_elem is None or not qualifier_name_elem.text:
                    skipped_count += 1
                    continue
                qualifier_name = qualifier_name_elem.text

                # Extract tree numbers
                tree_numbers = []
                tree_number_list = record.find("./TreeNumberList")
                if tree_number_list is not None:
                    for tree_elem in tree_number_list.findall("./TreeNumber"):
                        if tree_elem.text:
                            tree_numbers.append(tree_elem.text.strip())

                # Extract scope note (description)
                scope_note = None
                scope_note_elem = record.find("./ScopeNote")
                if scope_note_elem is not None and scope_note_elem.text:
                    scope_note = scope_note_elem.text.strip()

                # Store qualifier data
                self.qualifiers[qualifier_id] = {
                    "id": qualifier_id,
                    "name": qualifier_name,
                    "tree_numbers": tree_numbers,
                    "description": scope_note
                }
                processed_count += 1

            except Exception as e:
                 self.logger.warning(f"Error processing qualifier record near UI {qualifier_id if 'qualifier_id' in locals() else 'unknown'}: {e}", exc_info=True)
                 skipped_count += 1
                 continue

        self.logger.info(f"Finished processing qualifier file. Processed: {processed_count}, Skipped: {skipped_count}")

    def parse_supplementary_file(self, file_path: str) -> None:
        """Parse MeSH supplementary XML file (suppYYYY.xml) and update internal state.

        Args:
            file_path: Path to supplementary XML file.
        """
        self.logger.info(f"Parsing supplementary file: {file_path}")
        root = self._parse_xml_robustly(file_path)
        if root is None:
            self.logger.error(f"Failed to parse supplementary file: {file_path}")
            return

        # Extract supplementary records
        supp_records = root.findall(".//SupplementalRecord")
        self.logger.info(f"Found {len(supp_records)} supplementary records to process.")

        processed_count = 0
        skipped_count = 0
        for record in tqdm(supp_records, desc="Processing supplementary records"):
            try:
                # Extract supplementary UI
                supp_ui_elem = record.find("./SupplementalRecordUI")
                if supp_ui_elem is None or not supp_ui_elem.text:
                    skipped_count += 1
                    continue
                supp_id = supp_ui_elem.text

                # Extract supplementary name
                supp_name_elem = record.find("./SupplementalRecordName/String")
                if supp_name_elem is None or not supp_name_elem.text:
                    skipped_count += 1
                    continue
                supp_name = supp_name_elem.text

                # Extract mapped descriptors (diseases that this supplementary maps to)
                mapped_descriptors = []
                heading_mapped_to_list = record.find("./HeadingMappedToList")
                if heading_mapped_to_list is not None:
                    for mapping in heading_mapped_to_list.findall("./HeadingMappedTo"):
                        desc_ref = mapping.find("./DescriptorReferredTo/DescriptorUI")
                        if desc_ref is not None and desc_ref.text:
                            mapped_descriptors.append(desc_ref.text.strip())

                # Extract pharmacological actions (for drug supplements)
                pharm_actions = []
                pharm_action_list = record.find("./PharmacologicalActionList")
                if pharm_action_list is not None:
                    for action in pharm_action_list.findall("./PharmacologicalAction"):
                        action_ref = action.find("./DescriptorReferredTo/DescriptorUI")
                        if action_ref is not None and action_ref.text:
                            pharm_actions.append(action_ref.text.strip())

                # Store supplementary data
                self.supplementary[supp_id] = {
                    "id": supp_id,
                    "name": supp_name,
                    "mapped_descriptors": mapped_descriptors,
                    "pharmacological_actions": pharm_actions
                }
                processed_count += 1

            except Exception as e:
                self.logger.warning(f"Error processing supplementary record near UI {supp_id if 'supp_id' in locals() else 'unknown'}: {e}", exc_info=True)
                skipped_count += 1
                continue

        self.logger.info(f"Finished processing supplementary file. Processed: {processed_count}, Skipped: {skipped_count}")

    def parse_mesh(self, year: int = 2025, include_supplementary: bool = True, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Parse MeSH data for a specific year including descriptors, qualifiers, and optionally supplementary records.

        Args:
            year: MeSH year to parse (default: 2025)
            include_supplementary: Whether to include supplementary records
            limit: Limit number of descriptors to parse (for testing)

        Returns:
            Dictionary containing processed MeSH data, or None if parsing fails critically.
        """
        if year not in self.available_years:
            self.logger.error(f"Year {year} not available. Available years: {self.available_years}")
            return None

        # Define filenames for the specified year
        desc_file_name = f"desc{year}.xml"
        qual_file_name = f"qual{year}.xml"
        supp_file_name = f"supp{year}.xml"
        
        desc_path = os.path.join(self.mesh_dir, desc_file_name)
        qual_path = os.path.join(self.mesh_dir, qual_file_name)
        supp_path = os.path.join(self.mesh_dir, supp_file_name)

        # Reset internal state for a fresh parse
        self.descriptors = {}
        self.qualifiers = {}
        self.supplementary = {}
        self.disease_hierarchy = {}
        self.term_to_id = {}

        self.logger.info(f"Starting MeSH {year} parsing...")

        # --- Parse Descriptor File (Mandatory) ---
        if os.path.exists(desc_path):
            self.logger.info(f"Using descriptor file: {desc_path}")
            self.parse_descriptor_file(desc_path, limit=limit)
            if not self.descriptors:
                self.logger.warning(f"No disease-related descriptors found or parsed from {desc_path}. Result might be empty.")
        else:
            self.logger.error(f"Required descriptor file not found: {desc_path}. Cannot proceed.")
            return None

        # --- Parse Qualifier File (Optional) ---
        if os.path.exists(qual_path):
            self.logger.info(f"Using qualifier file: {qual_path}")
            self.parse_qualifier_file(qual_path)
        else:
            self.logger.warning(f"Qualifier file not found: {qual_path}. Proceeding without qualifier data.")

        # --- Parse Supplementary File (Optional) ---
        if include_supplementary and os.path.exists(supp_path):
            self.logger.info(f"Using supplementary file: {supp_path}")
            self.parse_supplementary_file(supp_path)
        elif include_supplementary:
            self.logger.warning(f"Supplementary file not found: {supp_path}. Proceeding without supplementary data.")

        # --- Process Disease Hierarchy ---
        self._process_disease_hierarchy()

        # --- Prepare Result ---
        result = {
            "descriptors": self.descriptors,
            "qualifiers": self.qualifiers,
            "supplementary": self.supplementary,
            "disease_hierarchy": self.disease_hierarchy,
            "term_to_id": self.term_to_id,
            "version": str(year),
            "available_years": self.available_years
        }

        self.logger.info(f"MeSH {year} parsing complete. Found {len(self.descriptors)} disease descriptors, {len(self.qualifiers)} qualifiers, {len(self.supplementary)} supplementary records.")
        return result

    def parse_multiple_years(self, years: Optional[list] = None, include_supplementary: bool = True) -> Dict[str, Any]:
        """
        Parse MeSH data for multiple years and combine the results.
        
        Args:
            years: List of years to parse (default: all available years)
            include_supplementary: Whether to include supplementary records
            
        Returns:
            Combined dictionary with historical MeSH data
        """
        if years is None:
            years = self.available_years
            
        combined_result = {
            "descriptors_by_year": {},
            "qualifiers_by_year": {},
            "supplementary_by_year": {},
            "term_to_id_by_year": {},
            "latest_descriptors": {},
            "latest_hierarchy": {},
            "years_processed": []
        }
        
        for year in sorted(years):
            self.logger.info(f"Processing MeSH year {year}...")
            year_result = self.parse_mesh(year, include_supplementary=include_supplementary)
            
            if year_result:
                combined_result["descriptors_by_year"][year] = year_result["descriptors"]
                combined_result["qualifiers_by_year"][year] = year_result["qualifiers"]
                combined_result["supplementary_by_year"][year] = year_result["supplementary"]
                combined_result["term_to_id_by_year"][year] = year_result["term_to_id"]
                combined_result["years_processed"].append(year)
                
                # Keep the latest year as the primary reference
                if year == max(years):
                    combined_result["latest_descriptors"] = year_result["descriptors"]
                    combined_result["latest_hierarchy"] = year_result["disease_hierarchy"]
        
        self.logger.info(f"Multi-year parsing complete. Processed years: {combined_result['years_processed']}")
        return combined_result

    def _process_disease_hierarchy(self) -> None:
        """
        Processes the raw disease hierarchy (parent_tn -> {children: [{tn},...]})
        to add descriptor names and IDs to each node where available.
        Operates on self.disease_hierarchy and self.descriptors.
        """
        self.logger.info("Processing disease hierarchy to add node details...")
        processed_hierarchy = {}
        nodes_to_process = list(self.disease_hierarchy.keys())

        # Add children nodes to the processing list if they aren't already keys
        all_child_nodes = set()
        for parent_data in self.disease_hierarchy.values():
            for child_info in parent_data.get("children", []):
                 all_child_nodes.add(child_info["tree_number"])

        nodes_to_process.extend(list(all_child_nodes - set(nodes_to_process))) # Add nodes that are only children

        # Create a reverse map from tree number to descriptor ID for faster lookup
        tn_to_descriptor_id = {}
        for d_id, desc_data in self.descriptors.items():
            for tn in desc_data.get("tree_numbers", []):
                # Handle potential multiple descriptors mapping to the same TN (less common)
                # Prioritize shorter descriptor IDs? Or just take the first one? Taking first for now.
                if tn not in tn_to_descriptor_id:
                    tn_to_descriptor_id[tn] = d_id

        processed_node_count = 0
        for tree_number in tqdm(nodes_to_process, desc="Populating hierarchy nodes"):
            descriptor_id = tn_to_descriptor_id.get(tree_number)
            node_data = {
                "tree_number": tree_number,
                "name": f"Unknown Node ({tree_number})", # Default name
                "descriptor_id": None,
                # Get children from original hierarchy if this node was a parent
                "children": [c["tree_number"] for c in self.disease_hierarchy.get(tree_number, {}).get("children", [])]
            }

            if descriptor_id and descriptor_id in self.descriptors:
                descriptor = self.descriptors[descriptor_id]
                node_data["name"] = descriptor.get("name", node_data["name"])
                node_data["descriptor_id"] = descriptor_id
            elif tree_number.isalpha() and len(tree_number) == 1: # Handle top-level category nodes (e.g., "C")
                 node_data["name"] = f"Category {tree_number}"


            processed_hierarchy[tree_number] = node_data
            processed_node_count += 1

        self.disease_hierarchy = processed_hierarchy # Replace raw hierarchy with processed one
        self.logger.info(f"Finished processing hierarchy. Populated {processed_node_count} nodes.")


    def extract_disease_taxonomy(self) -> Dict[str, Any]:
        """
        Extract disease taxonomy information based on the parsed descriptors (self.descriptors).
        Focuses on descriptors marked 'is_disease'.

        Returns:
            Dictionary mapping disease descriptor ID to taxonomy info (parents, etc.).
        """
        self.logger.info("Extracting disease taxonomy from parsed descriptors...")

        taxonomy = {}
        if not self.descriptors:
             self.logger.warning("No descriptors available to extract taxonomy from.")
             return taxonomy

        # Use the processed hierarchy for parent lookup if available and reliable
        # Create a map from child tree number to parent tree number for faster lookup
        child_to_parent_tn = {}
        if self.disease_hierarchy:
             for parent_tn, node_data in self.disease_hierarchy.items():
                 for child_tn in node_data.get("children", []):
                     # A child might have multiple parents listed if hierarchy is complex, store list
                     if child_tn not in child_to_parent_tn:
                          child_to_parent_tn[child_tn] = []
                     if parent_tn not in child_to_parent_tn[child_tn]: # Avoid duplicates
                        child_to_parent_tn[child_tn].append(parent_tn)


        # Get all disease descriptors marked during parsing
        disease_descriptors = {d_id: desc for d_id, desc in self.descriptors.items() if desc.get("is_disease", False)}
        self.logger.info(f"Found {len(disease_descriptors)} disease descriptors for taxonomy extraction.")

        # Extract taxonomy details
        extracted_count = 0
        for d_id, descriptor in tqdm(disease_descriptors.items(), desc="Extracting disease taxonomy"):
            tree_numbers = descriptor.get("tree_numbers", [])

            # Filter for disease tree numbers (should be redundant if filtering worked, but safe)
            disease_tree_numbers = [tn for tn in tree_numbers if tn.startswith(tuple(self.disease_categories))]

            if not disease_tree_numbers:
                continue # Skip if somehow a non-disease descriptor is processed

            # Get parents using the precomputed hierarchy map
            parents_info = []
            parent_ids_found = set() # Track parent IDs to avoid duplicates if multiple paths lead to same parent descriptor
            for tree_number in disease_tree_numbers:
                 parent_tns = child_to_parent_tn.get(tree_number, [])
                 for parent_tn in parent_tns:
                     # Find the descriptor associated with the parent tree number
                     parent_node_info = self.disease_hierarchy.get(parent_tn)
                     if parent_node_info:
                         parent_id = parent_node_info.get("descriptor_id")
                         parent_name = parent_node_info.get("name")
                         if parent_id and parent_id not in parent_ids_found:
                             parents_info.append({
                                 "id": parent_id,
                                 "name": parent_name,
                                 "tree_number": parent_tn
                             })
                             parent_ids_found.add(parent_id)
                         elif not parent_id and parent_name: # Handle category parents without specific descriptor ID
                              # Check if we already added this category parent
                              if not any(p.get("tree_number") == parent_tn for p in parents_info):
                                   parents_info.append({
                                        "id": None, # No specific descriptor ID
                                        "name": parent_name,
                                        "tree_number": parent_tn
                                   })


            # Create taxonomy entry
            taxonomy[d_id] = {
                "id": d_id,
                "name": descriptor["name"],
                "description": descriptor.get("description", ""),
                "tree_numbers": disease_tree_numbers,
                "synonyms": descriptor.get("synonyms", []),
                "parents": parents_info, # List of parent dicts
                "is_top_level": all(tn.count('.') == 0 for tn in disease_tree_numbers) # Check if any TN is top level
            }
            extracted_count += 1

        self.logger.info(f"Extracted taxonomy for {extracted_count} diseases.")
        return taxonomy

    def save_mesh_data(self, parsed_data: Dict[str, Any], format: str = "pickle") -> Optional[str]:
        """Save the main processed MeSH data (descriptors, qualifiers, hierarchy, version).

        Args:
            parsed_data: The dictionary returned by parse_2025_mesh.
            format: Output format ('pickle' or 'json').

        Returns:
            Path to the saved file, or None on failure.
        """
        if not parsed_data:
            self.logger.error("No parsed data provided to save.")
            return None

        version = parsed_data.get("version", "unknown")
        output_filename = f"mesh_data_{version}.{format}"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            if format == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(parsed_data, f)
            elif format == "json":
                # Convert sets to lists for JSON compatibility if any exist (shouldn't be sets here)
                def convert_sets(obj):
                    if isinstance(obj, set): return sorted(list(obj))
                    if isinstance(obj, dict): return {k: convert_sets(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [convert_sets(i) for i in obj]
                    return obj
                serializable_data = convert_sets(parsed_data)
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            else:
                self.logger.error(f"Unsupported output format: {format}")
                return None

            self.logger.info(f"Saved MeSH data ({version}) to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to save MeSH data to {output_path}: {e}", exc_info=True)
            return None

    def save_disease_taxonomy(self, taxonomy_data: Dict[str, Any], version: str, format: str = "pickle") -> Optional[str]:
        """Save the extracted disease taxonomy.

        Args:
            taxonomy_data: The dictionary returned by extract_disease_taxonomy.
            version: The MeSH version string (e.g., "2025").
            format: Output format ('pickle' or 'json').

        Returns:
            Path to the saved file, or None on failure.
        """
        if not taxonomy_data:
            self.logger.warning("No taxonomy data provided to save.")
            return None

        output_filename = f"disease_taxonomy_{version}.{format}"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            if format == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(taxonomy_data, f)
            elif format == "json":
                 # Convert sets to lists for JSON compatibility if any exist (e.g., synonyms)
                def convert_sets(obj):
                    if isinstance(obj, set): return sorted(list(obj))
                    if isinstance(obj, dict): return {k: convert_sets(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [convert_sets(i) for i in obj]
                    return obj
                serializable_data = convert_sets(taxonomy_data)
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            else:
                self.logger.error(f"Unsupported output format: {format}")
                return None

            self.logger.info(f"Saved disease taxonomy ({version}) to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to save disease taxonomy to {output_path}: {e}", exc_info=True)
            return None


# Example usage (remains largely the same, but calls the specific 2025 parser)
def main():
    import argparse

    parser_arg = argparse.ArgumentParser(description="Parse MeSH descriptor, qualifier, and supplementary files for multiple years (2020-2025).")
    parser_arg.add_argument("--mesh_dir", required=True, help="Directory containing MeSH XML files (descYYYY.xml, qualYYYY.xml, suppYYYY.xml)")
    parser_arg.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser_arg.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output format for saved files")
    parser_arg.add_argument("--year", type=int, default=2025, help="MeSH year to parse (default: 2025)")
    parser_arg.add_argument("--multi_year", action="store_true", help="Parse all available years")
    parser_arg.add_argument("--limit", type=int, help="Limit number of descriptors to parse (for testing)")
    args = parser_arg.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", # Added levelname
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Initialize parser
    mesh_parser = MeSHParser(args.mesh_dir, args.output_dir)

    if args.multi_year:
        # Parse all available years
        logging.info("Multi-year parsing requested...")
        parsed_mesh_data = mesh_parser.parse_multiple_years()
        
        if parsed_mesh_data and parsed_mesh_data["years_processed"]:
            # Save combined data
            combined_filename = f"mesh_data_combined_{min(parsed_mesh_data['years_processed'])}-{max(parsed_mesh_data['years_processed'])}"
            parsed_mesh_data["version"] = combined_filename
            saved_mesh_path = mesh_parser.save_mesh_data(parsed_mesh_data, format=args.format)
            logging.info(f"Multi-year MeSH processing finished. Years: {parsed_mesh_data['years_processed']}")
        else:
            logging.error("Multi-year MeSH parsing failed.")
    else:
        # Parse specific year
        parsed_mesh_data = mesh_parser.parse_mesh(year=args.year, include_supplementary=True, limit=args.limit)

        if parsed_mesh_data:
            # Save main parsed data
            saved_mesh_path = mesh_parser.save_mesh_data(parsed_mesh_data, format=args.format)

            if saved_mesh_path:
                # Extract and save disease taxonomy
                taxonomy = mesh_parser.extract_disease_taxonomy()
                mesh_parser.save_disease_taxonomy(taxonomy, parsed_mesh_data['version'], format=args.format)
            else:
                 logging.error("Failed to save main MeSH data, skipping taxonomy saving.")

            logging.info(f"MeSH {args.year} processing finished successfully.")
        else:
            logging.error(f"MeSH {args.year} parsing failed. No data saved.")

if __name__ == "__main__":
    main()
