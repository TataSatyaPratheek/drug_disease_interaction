# src/ddi/data/sources/drugbank/xml_parser.py
import os
# Use lxml instead of the standard library's ElementTree for memory efficiency
# import xml.etree.ElementTree as ET
from lxml import etree
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import json
import pickle

class DrugBankXMLParser:
    """Parser for DrugBank full_database.xml file using lxml for large files"""

    def __init__(self, xml_file_path: str):
        """Initialize parser

        Args:
            xml_file_path: Path to full_database.xml
        """
        self.xml_file_path = xml_file_path
        self.logger = logging.getLogger(self.__class__.__name__)
        # Namespace will be determined during parsing
        self.ns = {}
        # DrugBank namespace URI (common value, adjust if needed)
        self.db_ns_uri = "http://www.drugbank.ca"
        self.db_tag_prefix = f"{{{self.db_ns_uri}}}"

    def _get_namespace(self, element):
        """Extract namespace URI from an element's tag (e.g., {uri}tag)."""
        tag = element.tag
        # Check if the tag is namespaced (contains '}')
        if '}' in tag and tag.startswith('{'):
            # Find the index of the closing brace
            end_brace_index = tag.find('}')
            # Extract the URI part between the braces
            return tag[1:end_brace_index]
        # Return None if no namespace URI is found in the tag format {uri}tag
        return None


    def parse(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Parse the DrugBank XML file iteratively using lxml.iterparse

        Args:
            limit: Optional limit on number of drugs to parse (for debugging)

        Returns:
            Dictionary containing parsed drug data
        """
        self.logger.info(f"Parsing DrugBank XML file iteratively: {self.xml_file_path}")

        drugs = []
        version = "unknown"
        context = None
        pbar = None # Initialize pbar to None

        try:
            # Use iterparse for memory efficiency - DO NOT filter by tag here
            context = etree.iterparse(self.xml_file_path, events=("start", "end"), recover=True) # Added recover=True for robustness

            drug_count = 0
            root_processed = False

            # Use tqdm for progress bar - Initialize later when context is confirmed
            # pbar = tqdm(desc="Parsing drugs", unit=" drugs") # Moved initialization

            for event, elem in context:
                # Determine namespace and version from the root element ('start' event)
                if not root_processed and event == 'start':
                    # Check if it's the drugbank root element (heuristic: ends with 'drugbank')
                    if '}' in elem.tag and elem.tag.endswith('}drugbank'):
                        ns_uri = self._get_namespace(elem)
                        if ns_uri:
                            self.db_ns_uri = ns_uri
                            self.db_tag_prefix = f"{{{self.db_ns_uri}}}"
                            self.ns = {"db": self.db_ns_uri} # Set self.ns HERE
                            self.logger.info(f"Detected XML namespace: {self.db_ns_uri}")
                        else:
                             self.logger.warning("Could not determine namespace from root element. Parsing might fail.")
                             # Attempt to guess common namespace
                             self.db_ns_uri = "http://www.drugbank.ca"
                             self.db_tag_prefix = f"{{{self.db_ns_uri}}}"
                             self.ns = {"db": self.db_ns_uri}
                             self.logger.warning(f"Assuming default namespace: {self.db_ns_uri}")


                        # Find version within the root element using the determined/assumed namespace
                        version_elem = elem.find(f"{self.db_tag_prefix}version")
                        if version_elem is not None and version_elem.text is not None:
                            version = version_elem.text.strip()
                            self.logger.info(f"DrugBank version: {version}")
                        root_processed = True

                        # Initialize tqdm now that we are likely processing a valid file
                        pbar = tqdm(desc="Parsing drugs", unit=" drugs")


                # Process each 'drug' element at its 'end' event
                # Use the now-defined self.db_tag_prefix
                if event == 'end' and elem.tag == f"{self.db_tag_prefix}drug":
                    if not self.ns:
                        # This should ideally not happen if root was processed, but as a fallback:
                        ns_uri = self._get_namespace(elem)
                        if ns_uri:
                            self.db_ns_uri = ns_uri
                            self.db_tag_prefix = f"{{{self.db_ns_uri}}}"
                            self.ns = {"db": self.db_ns_uri}
                            self.logger.warning(f"Namespace detected late from drug element: {self.db_ns_uri}")
                        else:
                            self.logger.error("Namespace not found, cannot parse drug details correctly. Skipping drug.")
                            # Clear element and continue
                            elem.clear()
                            # Also eliminate now-empty references from the root node to elem
                            while elem.getprevious() is not None:
                                del elem.getparent()[0]
                            continue

                    try:
                        drug = self._parse_drug(elem)
                        if drug:
                            drugs.append(drug)
                            drug_count += 1
                            if pbar is not None: pbar.update(1)
                    except Exception as e:
                        # Attempt to get drug ID for error logging
                        drug_id_elem = elem.find(f"{self.db_tag_prefix}drugbank-id[@primary='true']", self.ns)
                        if drug_id_elem is None:
                            drug_id_elem = elem.find(f"{self.db_tag_prefix}drugbank-id", self.ns)
                        drug_id = drug_id_elem.text.strip() if drug_id_elem is not None and drug_id_elem.text else "unknown_id"
                        self.logger.error(f"Error parsing drug {drug_id}: {str(e)}", exc_info=True)

                    # Crucial for memory management with iterparse: clear the element
                    elem.clear()
                    # Also eliminate now-empty references from the root node to elem
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

                    # Apply limit if specified
                    if limit is not None and drug_count >= limit:
                        break

            if pbar is not None: pbar.close() # Close pbar if it was initialized

        except FileNotFoundError:
             self.logger.error(f"XML file not found: {self.xml_file_path}")
             return {"version": version, "drugs": drugs}
        except etree.XMLSyntaxError as e: # Catch XML syntax errors specifically
            self.logger.error(f"XML Syntax Error in {self.xml_file_path}: {str(e)}", exc_info=True)
            if pbar is not None:
                try: pbar.close()
                except: pass
            return {"version": version, "drugs": drugs}
        except Exception as e: # Catch other potential parsing errors
            self.logger.error(f"Error during XML parsing setup or iteration for {self.xml_file_path}: {str(e)}", exc_info=True)
            if pbar is not None:
                 try: pbar.close()
                 except: pass
            return {"version": version, "drugs": drugs}
        finally: # Ensure context is cleaned up
             if context is not None:
                 try:
                     # Clear the context/iterator
                     # For lxml's iterparse, clearing elements in the loop is the main thing.
                     # Explicitly clearing the context object itself might not be necessary or available.
                     pass
                 except Exception as cleanup_e:
                     self.logger.warning(f"Error during XML parser context cleanup: {cleanup_e}")

        self.logger.info(f"Successfully parsed {len(drugs)} drugs")
        return {"version": version, "drugs": drugs}

    # --- All helper methods (_get_text, _parse_drug, etc.) should now work ---
    # --- Make sure they consistently use self.ns ---

    def _get_text(self, element: etree._Element, xpath: str) -> Optional[str]:
        """Get text content from an XML element using lxml and namespaces."""
        if not self.ns: # Add check if namespace is missing
             # self.logger.warning(f"Namespace not set when trying to find {xpath}. Returning None.")
             return None
        target = element.find(xpath, self.ns)
        return target.text.strip() if target is not None and target.text is not None else None

    def _parse_list_elements(self, parent: etree._Element, xpath: str) -> List[str]:
        """Parse list of text elements using lxml and namespaces."""
        if not self.ns: return [] # Add check
        elements = []
        for element in parent.findall(xpath, self.ns):
            if element.text is not None:
                 text = element.text.strip()
                 if text:
                     elements.append(text)
        return elements

    def _parse_drug(self, drug_element: etree._Element) -> Dict[str, Any]:
        """Parse a single drug element (using lxml element)"""
        if not self.ns:
            self.logger.error("Cannot parse drug, namespace not set.")
            return {}

        # Extract primary drugbank-id first
        primary_id_elem = drug_element.find("db:drugbank-id[@primary='true']", self.ns)
        if primary_id_elem is not None and primary_id_elem.text:
            drug_id = primary_id_elem.text.strip()
        else:
            # Fallback to the first drugbank-id if primary is not found
            first_id_elem = drug_element.find("db:drugbank-id", self.ns)
            drug_id = first_id_elem.text.strip() if first_id_elem is not None and first_id_elem.text else None

        if not drug_id:
             self.logger.warning(f"Drug element found without a DrugBank ID. Skipping.")
             # Or raise an error, depending on desired behavior
             # raise ValueError("Drug element is missing a DrugBank ID")
             return {} # Return empty dict or handle as appropriate

        drug = {
            "drugbank_id": drug_id,
            "type": drug_element.get("type"),
            "name": self._get_text(drug_element, "db:name"),
            "description": self._get_text(drug_element, "db:description"),
            "cas_number": self._get_text(drug_element, "db:cas-number"),
            "unii": self._get_text(drug_element, "db:unii"), # Added UNII
            "state": self._get_text(drug_element, "db:state"), # Added state
            "groups": self._parse_groups(drug_element),
            "synthesis_reference": self._get_text(drug_element, "db:synthesis-reference"), # Added synthesis-reference
            "indication": self._get_text(drug_element, "db:indication"), # Changed from indications
            "pharmacodynamics": self._get_text(drug_element, "db:pharmacodynamics"),
            "mechanism_of_action": self._get_text(drug_element, "db:mechanism-of-action"),
            "toxicity": self._get_text(drug_element, "db:toxicity"),
            "metabolism": self._get_text(drug_element, "db:metabolism"),
            "absorption": self._get_text(drug_element, "db:absorption"),
            "half_life": self._get_text(drug_element, "db:half-life"),
            "protein_binding": self._get_text(drug_element, "db:protein-binding"),
            "route_of_elimination": self._get_text(drug_element, "db:route-of-elimination"),
            "volume_of_distribution": self._get_text(drug_element, "db:volume-of-distribution"),
            "clearance": self._get_text(drug_element, "db:clearance"),
            "classification": self._parse_classification(drug_element),
            "salts": self._parse_salts(drug_element),
            "synonyms": self._parse_synonyms(drug_element),
            "products": self._parse_products(drug_element),
            "international_brands": self._parse_international_brands(drug_element),
            "mixtures": self._parse_mixtures(drug_element),
            "packagers": self._parse_packagers(drug_element),
            "manufacturers": self._parse_manufacturers(drug_element),
            "prices": self._parse_prices(drug_element),
            "categories": self._parse_categories(drug_element), # Moved categories here
            "affected_organisms": self._parse_affected_organisms(drug_element),
            "dosages": self._parse_dosages(drug_element),
            "atc_codes": self._parse_atc_codes(drug_element), # Moved ATC codes here
            "ahfs_codes": self._parse_list_elements(drug_element, "db:ahfs-codes/db:ahfs-code"), # Added AHFS codes
            "pdb_entries": self._parse_list_elements(drug_element, "db:pdb-entries/db:pdb-entry"), # Added PDB entries
            "fda_label": self._get_text(drug_element, "db:fda-label"), # Added FDA label link
            "msds": self._get_text(drug_element, "db:msds"), # Added MSDS link
            "patents": self._parse_patents(drug_element),
            "food_interactions": self._parse_food_interactions(drug_element),
            "drug_interactions": self._parse_drug_interactions(drug_element),
            "sequences": self._parse_sequences(drug_element), # Added sequences
            "calculated_properties": self._parse_calculated_properties(drug_element),
            "experimental_properties": self._parse_experimental_properties(drug_element),
            "external_identifiers": self._parse_external_identifiers(drug_element),
            "external_links": self._parse_external_links(drug_element),
            "pathways": self._parse_pathways(drug_element),
            "reactions": self._parse_reactions(drug_element),
            "snp_effects": self._parse_snp_effects(drug_element),
            "snp_adverse_drug_reactions": self._parse_snp_adverse_drug_reactions(drug_element),
            "targets": self._parse_targets(drug_element),
            "enzymes": self._parse_enzymes(drug_element),
            "carriers": self._parse_carriers(drug_element),
            "transporters": self._parse_transporters(drug_element)
        }

        # Remove keys with None or empty list values if desired for cleaner output
        # drug = {k: v for k, v in drug.items() if v is not None and v != []}

        return drug

    def _parse_groups(self, drug_element: etree._Element) -> List[str]:
        """Parse drug groups"""
        return self._parse_list_elements(drug_element, "db:groups/db:group")

    def _parse_atc_codes(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse ATC codes"""
        atc_codes = []
        for atc_code_elem in drug_element.findall("db:atc-codes/db:atc-code", self.ns):
            code = {
                "code": atc_code_elem.get("code"),
                "levels": []
            }
            for level in atc_code_elem.findall("db:level", self.ns):
                level_data = {
                    "code": level.get("code"),
                    "name": level.text.strip() if level.text else None
                }
                if level_data["name"]: # Only add if name exists
                    code["levels"].append(level_data)
            if code["code"]: # Only add if ATC code exists
                atc_codes.append(code)
        return atc_codes

    def _parse_categories(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse drug categories"""
        categories = []
        for category_elem in drug_element.findall("db:categories/db:category", self.ns):
            cat = {
                "category": self._get_text(category_elem, "db:category"),
                "mesh_id": self._get_text(category_elem, "db:mesh-id")
            }
            # Ensure at least one field is present
            if cat["category"] or cat["mesh_id"]:
                categories.append(cat)
        return categories

    def _parse_classification(self, drug_element: etree._Element) -> Optional[Dict[str, Any]]:
        """Parse drug classification"""
        classification_elem = drug_element.find("db:classification", self.ns)
        if classification_elem is None:
            return None

        # Helper to parse multiple elements into a list
        def get_list(parent, tag):
            return [elem.text.strip() for elem in parent.findall(tag, self.ns) if elem.text and elem.text.strip()]

        return {
            "description": self._get_text(classification_elem, "db:description"),
            "direct_parent": self._get_text(classification_elem, "db:direct-parent"),
            "kingdom": self._get_text(classification_elem, "db:kingdom"),
            "superclass": self._get_text(classification_elem, "db:superclass"),
            "class": self._get_text(classification_elem, "db:class"),
            "subclass": self._get_text(classification_elem, "db:subclass"),
            # Handle potentially multiple alternative parents and substituents
            "alternative_parents": get_list(classification_elem, "db:alternative-parent"),
            "substituents": get_list(classification_elem, "db:substituent")
        }

    def _parse_salts(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse drug salts"""
        salts = []
        for salt_elem in drug_element.findall("db:salts/db:salt", self.ns):
            salt_data = {
                "drugbank_id": self._get_text(salt_elem, "db:drugbank-id"),
                "name": self._get_text(salt_elem, "db:name"),
                "unii": self._get_text(salt_elem, "db:unii"),
                "cas_number": self._get_text(salt_elem, "db:cas-number"),
                "inchikey": self._get_text(salt_elem, "db:inchikey"),
                "average_mass": self._get_text(salt_elem, "db:average-mass"),
                "monoisotopic_mass": self._get_text(salt_elem, "db:monoisotopic-mass")
            }
            # Add only if at least drugbank_id or name is present
            if salt_data["drugbank_id"] or salt_data["name"]:
                salts.append(salt_data)
        return salts

    def _parse_synonyms(self, drug_element: etree._Element) -> List[str]:
        """Parse drug synonyms"""
        synonyms = []
        # Synonyms structure might vary slightly, check common paths
        for syn_elem in drug_element.findall("db:synonyms/db:synonym", self.ns):
             if syn_elem.text:
                 synonyms.append(syn_elem.text.strip())
        return synonyms

    def _parse_products(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse drug products"""
        products = []
        for product_elem in drug_element.findall("db:products/db:product", self.ns):
            product_data = {
                "name": self._get_text(product_elem, "db:name"),
                "labeller": self._get_text(product_elem, "db:labeller"),
                "ndc_id": self._get_text(product_elem, "db:ndc-id"),
                "ndc_product_code": self._get_text(product_elem, "db:ndc-product-code"),
                "dpd_id": self._get_text(product_elem, "db:dpd-id"),
                "ema_product_code": self._get_text(product_elem, "db:ema-product-code"), # Added EMA code
                "ema_ma_number": self._get_text(product_elem, "db:ema-ma-number"), # Added EMA MA number
                "started_marketing_on": self._get_text(product_elem, "db:started-marketing-on"),
                "ended_marketing_on": self._get_text(product_elem, "db:ended-marketing-on"),
                "dosage_form": self._get_text(product_elem, "db:dosage-form"),
                "strength": self._get_text(product_elem, "db:strength"),
                "route": self._get_text(product_elem, "db:route"),
                "fda_application_number": self._get_text(product_elem, "db:fda-application-number"),
                "generic": product_elem.get("generic") == "true",
                "over_the_counter": product_elem.get("over-the-counter") == "true",
                "approved": product_elem.get("approved") == "true",
                "country": self._get_text(product_elem, "db:country"), # Changed to get text
                "source": self._get_text(product_elem, "db:source") # Changed to get text
            }
            if product_data["name"]: # Add only if name exists
                products.append(product_data)
        return products

    def _parse_international_brands(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse international brands"""
        brands = []
        for brand_elem in drug_element.findall("db:international-brands/db:international-brand", self.ns):
            brand_data = {
                "name": self._get_text(brand_elem, "db:name"),
                "company": self._get_text(brand_elem, "db:company")
            }
            if brand_data["name"]: # Add only if name exists
                brands.append(brand_data)
        return brands

    def _parse_mixtures(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse mixtures"""
        mixtures = []
        for mixture_elem in drug_element.findall("db:mixtures/db:mixture", self.ns):
            mixture_data = {
                "name": self._get_text(mixture_elem, "db:name"),
                "ingredients": self._get_text(mixture_elem, "db:ingredients")
            }
            if mixture_data["name"]: # Add only if name exists
                mixtures.append(mixture_data)
        return mixtures

    def _parse_packagers(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse packagers"""
        packagers = []
        for packager_elem in drug_element.findall("db:packagers/db:packager", self.ns):
            packager_data = {
                "name": self._get_text(packager_elem, "db:name"),
                "url": self._get_text(packager_elem, "db:url")
            }
            if packager_data["name"]: # Add only if name exists
                packagers.append(packager_data)
        return packagers

    def _parse_manufacturers(self, drug_element: etree._Element) -> List[str]:
        """Parse manufacturers"""
        return self._parse_list_elements(drug_element, "db:manufacturers/db:manufacturer")

    def _parse_prices(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse prices"""
        prices = []
        for price_elem in drug_element.findall("db:prices/db:price", self.ns):
            price_data = {
                "description": self._get_text(price_elem, "db:description"),
                "cost": self._get_text(price_elem, "db:cost"),
                "currency": self._get_text(price_elem, "db:currency"),
                "unit": self._get_text(price_elem, "db:unit") # Added unit
            }
            # Add only if cost exists
            if price_data["cost"]:
                prices.append(price_data)
        return prices

    def _parse_affected_organisms(self, drug_element: etree._Element) -> List[str]:
        """Parse affected organisms"""
        return self._parse_list_elements(drug_element, "db:affected-organisms/db:affected-organism")

    def _parse_dosages(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse dosages"""
        dosages = []
        for dosage_elem in drug_element.findall("db:dosages/db:dosage", self.ns):
            dosage_data = {
                "form": self._get_text(dosage_elem, "db:form"),
                "route": self._get_text(dosage_elem, "db:route"),
                "strength": self._get_text(dosage_elem, "db:strength")
            }
            # Add only if form or strength exists
            if dosage_data["form"] or dosage_data["strength"]:
                dosages.append(dosage_data)
        return dosages

    def _parse_patents(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse patents"""
        patents = []
        for patent_elem in drug_element.findall("db:patents/db:patent", self.ns):
            patent_data = {
                "number": self._get_text(patent_elem, "db:number"),
                "country": self._get_text(patent_elem, "db:country"),
                "approved": self._get_text(patent_elem, "db:approved"),
                "expires": self._get_text(patent_elem, "db:expires"),
                "pediatric_extension": patent_elem.get("pediatric-extension") == "true"
            }
            if patent_data["number"]: # Add only if number exists
                patents.append(patent_data)
        return patents

    def _parse_food_interactions(self, drug_element: etree._Element) -> List[str]:
        """Parse food interactions"""
        return self._parse_list_elements(drug_element, "db:food-interactions/db:food-interaction")

    def _parse_drug_interactions(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse drug interactions"""
        interactions = []
        for interaction_elem in drug_element.findall("db:drug-interactions/db:drug-interaction", self.ns):
            interaction_data = {
                "drugbank_id": self._get_text(interaction_elem, "db:drugbank-id"),
                "name": self._get_text(interaction_elem, "db:name"),
                "description": self._get_text(interaction_elem, "db:description")
            }
            # Add only if interacting drugbank_id exists
            if interaction_data["drugbank_id"]:
                interactions.append(interaction_data)
        return interactions

    def _parse_sequences(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse sequences (handling multiple possible sequence tags)"""
        sequences = []
        for seq_elem in drug_element.findall("db:sequences/db:sequence", self.ns):
            if seq_elem.text:
                seq_data = {
                    "header": seq_elem.get("header"), # Get header attribute if exists
                    "format": seq_elem.get("format"), # Get format attribute if exists
                    "sequence": seq_elem.text.strip()
                }
                sequences.append(seq_data)
        return sequences

    def _parse_calculated_properties(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse calculated properties"""
        properties = []
        for prop_elem in drug_element.findall("db:calculated-properties/db:property", self.ns):
            property_data = {
                "kind": self._get_text(prop_elem, "db:kind"),
                "value": self._get_text(prop_elem, "db:value"),
                "source": self._get_text(prop_elem, "db:source")
            }
            if property_data["kind"] and property_data["value"]: # Add only if kind and value exist
                properties.append(property_data)
        return properties

    def _parse_experimental_properties(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse experimental properties"""
        properties = []
        for prop_elem in drug_element.findall("db:experimental-properties/db:property", self.ns):
            property_data = {
                "kind": self._get_text(prop_elem, "db:kind"),
                "value": self._get_text(prop_elem, "db:value"),
                "source": self._get_text(prop_elem, "db:source")
            }
            if property_data["kind"] and property_data["value"]: # Add only if kind and value exist
                properties.append(property_data)
        return properties

    def _parse_external_identifiers(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse external identifiers"""
        identifiers = []
        for identifier_elem in drug_element.findall("db:external-identifiers/db:external-identifier", self.ns):
            identifier_data = {
                "resource": self._get_text(identifier_elem, "db:resource"),
                "identifier": self._get_text(identifier_elem, "db:identifier")
            }
            if identifier_data["resource"] and identifier_data["identifier"]: # Add only if both exist
                identifiers.append(identifier_data)
        return identifiers

    def _parse_external_links(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse external links"""
        links = []
        for link_elem in drug_element.findall("db:external-links/db:external-link", self.ns):
            link_data = {
                "resource": self._get_text(link_elem, "db:resource"),
                "url": self._get_text(link_elem, "db:url")
            }
            if link_data["resource"] and link_data["url"]: # Add only if both exist
                links.append(link_data)
        return links

    def _parse_pathways(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse pathways"""
        pathways = []
        for pathway_elem in drug_element.findall("db:pathways/db:pathway", self.ns):
            pathway_data = {
                "smpdb_id": self._get_text(pathway_elem, "db:smpdb-id"),
                "name": self._get_text(pathway_elem, "db:name"),
                "category": self._get_text(pathway_elem, "db:category"),
                "drugs": [], # Added drugs list
                "enzymes": []
            }
            for drug_elem_path in pathway_elem.findall("db:drugs/db:drug", self.ns):
                 drug_ref = {
                     "drugbank_id": self._get_text(drug_elem_path, "db:drugbank-id"),
                     "name": self._get_text(drug_elem_path, "db:name")
                 }
                 if drug_ref["drugbank_id"]:
                     pathway_data["drugs"].append(drug_ref)

            for enzyme_elem in pathway_elem.findall("db:enzymes/db:uniprot-id", self.ns):
                if enzyme_elem.text:
                    pathway_data["enzymes"].append(enzyme_elem.text.strip())

            if pathway_data["smpdb_id"]: # Add only if smpdb_id exists
                pathways.append(pathway_data)
        return pathways

    def _parse_reactions(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse reactions"""
        reactions = []
        for reaction_elem in drug_element.findall("db:reactions/db:reaction", self.ns):
            reaction_data = {
                "sequence": self._get_text(reaction_elem, "db:sequence"),
                "left_element": self._parse_reaction_element(reaction_elem.find("db:left-element", self.ns)),
                "right_element": self._parse_reaction_element(reaction_elem.find("db:right-element", self.ns)),
                "enzymes": []
            }

            for enzyme_elem in reaction_elem.findall("db:enzymes/db:enzyme", self.ns):
                enzyme_data = {
                    "drugbank_id": self._get_text(enzyme_elem, "db:drugbank-id"),
                    "name": self._get_text(enzyme_elem, "db:name"),
                    "uniprot_id": self._get_text(enzyme_elem, "db:uniprot-id")
                }
                # Add only if at least one identifier exists
                if enzyme_data["drugbank_id"] or enzyme_data["uniprot_id"]:
                    reaction_data["enzymes"].append(enzyme_data)

            # Add reaction only if it has elements or enzymes
            if reaction_data["left_element"] or reaction_data["right_element"] or reaction_data["enzymes"]:
                 reactions.append(reaction_data)
        return reactions

    def _parse_reaction_element(self, element: Optional[etree._Element]) -> Optional[Dict[str, str]]:
        """Parse reaction element"""
        if element is None:
            return None
        data = {
            "drugbank_id": self._get_text(element, "db:drugbank-id"),
            "name": self._get_text(element, "db:name")
        }
        # Return None if both fields are empty, otherwise return the dict
        return data if data.get("drugbank_id") or data.get("name") else None

    def _parse_snp_effects(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse SNP effects"""
        effects = []
        for effect_elem in drug_element.findall("db:snp-effects/db:effect", self.ns):
            effect_data = {
                "protein_name": self._get_text(effect_elem, "db:protein-name"),
                "gene_symbol": self._get_text(effect_elem, "db:gene-symbol"),
                "uniprot_id": self._get_text(effect_elem, "db:uniprot-id"),
                "rs_id": self._get_text(effect_elem, "db:rs-id"),
                "allele": self._get_text(effect_elem, "db:allele"),
                "defining_change": self._get_text(effect_elem, "db:defining-change"),
                "description": self._get_text(effect_elem, "db:description"),
                "pubmed_id": self._get_text(effect_elem, "db:pubmed-id")
            }
            # Add only if rs_id exists
            if effect_data["rs_id"]:
                effects.append(effect_data)
        return effects

    def _parse_snp_adverse_drug_reactions(self, drug_element: etree._Element) -> List[Dict[str, str]]:
        """Parse SNP adverse drug reactions"""
        reactions = []
        for reaction_elem in drug_element.findall("db:snp-adverse-drug-reactions/db:reaction", self.ns):
            reaction_data = {
                "protein_name": self._get_text(reaction_elem, "db:protein-name"),
                "gene_symbol": self._get_text(reaction_elem, "db:gene-symbol"),
                "uniprot_id": self._get_text(reaction_elem, "db:uniprot-id"),
                "rs_id": self._get_text(reaction_elem, "db:rs-id"),
                "allele": self._get_text(reaction_elem, "db:allele"),
                "adverse_reaction": self._get_text(reaction_elem, "db:adverse-reaction"),
                "description": self._get_text(reaction_elem, "db:description"),
                "pubmed_id": self._get_text(reaction_elem, "db:pubmed-id")
            }
            # Add only if rs_id exists
            if reaction_data["rs_id"]:
                reactions.append(reaction_data)
        return reactions

    def _parse_protein_elements(self, drug_element: etree._Element, xpath: str) -> List[Dict[str, Any]]:
        """Parse protein elements (targets, enzymes, carriers, transporters)"""
        proteins = []
        for protein_elem in drug_element.findall(xpath, self.ns):
            protein_data = {
                "id": self._get_text(protein_elem, "db:id"), # Often the DrugBank protein ID
                "name": self._get_text(protein_elem, "db:name"),
                "organism": self._get_text(protein_elem, "db:organism"),
                "actions": self._parse_list_elements(protein_elem, "db:actions/db:action"),
                "references": self._parse_references(protein_elem), # Use dedicated parser
                "known_action": self._get_text(protein_elem, "db:known-action"),
                "polypeptides": self._parse_polypeptides(protein_elem),
                "position": protein_elem.get("position"), # Get position attribute if exists
            }
            # Add only if id or name exists
            if protein_data["id"] or protein_data["name"]:
                proteins.append(protein_data)
        return proteins

    def _parse_references(self, element: etree._Element) -> List[Dict[str, str]]:
        """Parse references which can be complex (articles, textbooks, links)."""
        refs = []
        # Articles
        for article in element.findall("db:references/db:articles/db:article", self.ns):
            ref_data = {
                "type": "article",
                "pubmed_id": self._get_text(article, "db:pubmed-id"),
                "citation": self._get_text(article, "db:citation"),
                "ref_id": self._get_text(article, "db:ref-id") # Added ref-id
            }
            if ref_data["pubmed_id"] or ref_data["citation"]:
                refs.append(ref_data)
        # Textbooks
        for textbook in element.findall("db:references/db:textbooks/db:textbook", self.ns):
            ref_data = {
                "type": "textbook",
                "isbn": self._get_text(textbook, "db:isbn"),
                "citation": self._get_text(textbook, "db:citation"),
                "ref_id": self._get_text(textbook, "db:ref-id") # Added ref-id
            }
            if ref_data["isbn"] or ref_data["citation"]:
                refs.append(ref_data)
        # Links
        for link in element.findall("db:references/db:links/db:link", self.ns):
            ref_data = {
                "type": "link",
                "title": self._get_text(link, "db:title"),
                "url": self._get_text(link, "db:url"),
                "ref_id": self._get_text(link, "db:ref-id") # Added ref-id
            }
            if ref_data["title"] or ref_data["url"]:
                refs.append(ref_data)
        # Attachments (less common, but possible)
        for attachment in element.findall("db:references/db:attachments/db:attachment", self.ns):
            ref_data = {
                "type": "attachment",
                "title": self._get_text(attachment, "db:title"),
                "url": self._get_text(attachment, "db:url"),
                "ref_id": self._get_text(attachment, "db:ref-id") # Added ref-id
            }
            if ref_data["title"] or ref_data["url"]:
                refs.append(ref_data)

        return refs

    def _parse_polypeptides(self, protein_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse polypeptides"""
        polypeptides = []
        for polypeptide_elem in protein_element.findall("db:polypeptide", self.ns):
            polypeptide_data = {
                "id": polypeptide_elem.get("id"), # DrugBank polypeptide ID
                "source": polypeptide_elem.get("source"), # e.g., UniProtKB
                "name": self._get_text(polypeptide_elem, "db:name"),
                "general_function": self._get_text(polypeptide_elem, "db:general-function"),
                "specific_function": self._get_text(polypeptide_elem, "db:specific-function"),
                "gene_name": self._get_text(polypeptide_elem, "db:gene-name"),
                "locus": self._get_text(polypeptide_elem, "db:locus"),
                "cellular_location": self._get_text(polypeptide_elem, "db:cellular-location"),
                "transmembrane_regions": self._get_text(polypeptide_elem, "db:transmembrane-regions"),
                "signal_regions": self._get_text(polypeptide_elem, "db:signal-regions"),
                "theoretical_pi": self._get_text(polypeptide_elem, "db:theoretical-pi"),
                "molecular_weight": self._get_text(polypeptide_elem, "db:molecular-weight"),
                "chromosome_location": self._get_text(polypeptide_elem, "db:chromosome-location"),
                "organism_ncbi_taxonomy_id": self._get_text(polypeptide_elem, "db:organism[@ncbi-taxonomy-id]"), # Get organism text
                "organism": polypeptide_elem.findtext("db:organism", namespaces=self.ns), # Get organism text
                "external_identifiers": self._parse_external_identifiers_element(polypeptide_elem),
                "synonyms": self._parse_list_elements(polypeptide_elem, "db:synonyms/db:synonym"),
                "amino_acid_sequence": self._get_text(polypeptide_elem, "db:amino-acid-sequence"), # Simplified sequence parsing
                "gene_sequence": self._get_text(polypeptide_elem, "db:gene-sequence"), # Simplified sequence parsing
                "pfams": self._parse_pfams(polypeptide_elem),
                "go_classifiers": self._parse_go_classifiers(polypeptide_elem)
            }
            # Add only if id exists
            if polypeptide_data["id"]:
                polypeptides.append(polypeptide_data)
        return polypeptides

    def _parse_external_identifiers_element(self, element: etree._Element) -> List[Dict[str, str]]:
        """Parse external identifiers from an element"""
        identifiers = []
        for identifier_elem in element.findall("db:external-identifiers/db:external-identifier", self.ns):
            identifier_data = {
                "resource": self._get_text(identifier_elem, "db:resource"),
                "identifier": self._get_text(identifier_elem, "db:identifier")
            }
            if identifier_data["resource"] and identifier_data["identifier"]:
                identifiers.append(identifier_data)
        return identifiers

    # Removed _parse_sequence as sequences are now simplified in _parse_polypeptides

    def _parse_pfams(self, polypeptide: etree._Element) -> List[Dict[str, str]]:
        """Parse Pfam data"""
        pfams = []
        for pfam_elem in polypeptide.findall("db:pfams/db:pfam", self.ns):
            pfam_data = {
                "identifier": self._get_text(pfam_elem, "db:identifier"),
                "name": self._get_text(pfam_elem, "db:name")
            }
            if pfam_data["identifier"]: # Add only if identifier exists
                pfams.append(pfam_data)
        return pfams

    def _parse_go_classifiers(self, polypeptide: etree._Element) -> List[Dict[str, str]]:
        """Parse Gene Ontology classifiers"""
        classifiers = []
        for classifier_elem in polypeptide.findall("db:go-classifiers/db:go-classifier", self.ns):
            classifier_data = {
                "category": self._get_text(classifier_elem, "db:category"),
                "description": self._get_text(classifier_elem, "db:description"),
                "go_id": self._get_text(classifier_elem, "db:go-id"), # Added GO ID
                "source": self._get_text(classifier_elem, "db:source") # Added source
            }
            if classifier_data["go_id"]: # Add only if go_id exists
                classifiers.append(classifier_data)
        return classifiers

    # --- Add parsers for targets, enzymes, carriers, transporters ---
    def _parse_targets(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse targets"""
        return self._parse_protein_elements(drug_element, "db:targets/db:target")

    def _parse_enzymes(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse enzymes"""
        return self._parse_protein_elements(drug_element, "db:enzymes/db:enzyme")

    def _parse_carriers(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse carriers"""
        return self._parse_protein_elements(drug_element, "db:carriers/db:carrier")

    def _parse_transporters(self, drug_element: etree._Element) -> List[Dict[str, Any]]:
        """Parse transporters"""
        return self._parse_protein_elements(drug_element, "db:transporters/db:transporter")


# --- Main execution block remains the same ---
def main():
    parser = argparse.ArgumentParser(description="Parse DrugBank XML database")
    parser.add_argument("--input", required=True, help="Path to full_database.xml")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--format", choices=["json", "pickle"], default="pickle", help="Output format")
    parser.add_argument("--limit", type=int, help="Limit number of drugs to parse (for testing)")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Parse DrugBank XML
    db_parser = DrugBankXMLParser(args.input) # Renamed parser variable
    drug_data = db_parser.parse(limit=args.limit)

    # Save to output file
    output_file = os.path.join(args.output, f"drugbank_parsed.{args.format}")

    if args.format == "json":
        logging.info(f"Saving to JSON: {output_file}")
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(drug_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error writing JSON file: {e}")
            # Optionally, try saving drug by drug if the whole structure fails
            # logging.info("Attempting to save drug by drug...")
            # with open(output_file, "w", encoding='utf-8') as f:
            #     f.write('{\n"version": ' + json.dumps(drug_data.get("version", "unknown")) + ',\n"drugs": [\n')
            #     for i, drug in enumerate(drug_data.get("drugs", [])):
            #         try:
            #             f.write(json.dumps(drug, indent=2, ensure_ascii=False))
            #             if i < len(drug_data["drugs"]) - 1:
            #                 f.write(',\n')
            #             else:
            #                 f.write('\n')
            #         except Exception as drug_e:
            #             logging.error(f"Error writing drug {drug.get('drugbank_id', 'unknown')} to JSON: {drug_e}")
            #     f.write(']\n}')

    else: # pickle format
        logging.info(f"Saving to pickle: {output_file}")
        try:
            with open(output_file, "wb") as f:
                pickle.dump(drug_data, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol for efficiency
        except Exception as e:
            logging.error(f"Error writing pickle file: {e}")

    logging.info(f"Parsed {len(drug_data['drugs'])} drugs and attempted to save to {output_file}")

if __name__ == "__main__":
    main()
