# src/ddi/parser/drugbank_parser.py

import os
import logging
import pickle
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from lxml import etree
from tqdm import tqdm

class DrugBankParser:
    """
    A simplified, all-in-one parser for DrugBank data.
    It parses the main XML database and enriches it with data from the
    drug vocabulary CSV file, producing a single, clean output.
    """
    def __init__(self, xml_path: str, vocabulary_path: str, output_dir: str):
        self.xml_path = xml_path  # <<< The correct attribute is named 'xml_path'
        self.vocabulary_path = vocabulary_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ns = {}  # XML namespace, to be detected automatically
        self.vocabulary_df = None
        self.integrated_data = None

    def parse_and_integrate(self, limit: Optional[int] = None) -> None:
        """
        Main orchestration method to perform all parsing and integration steps.
        
        Args:
            limit: Optional limit on the number of drugs to process for testing.
        """
        # Step 1: Load the vocabulary CSV for enrichment
        self._load_vocabulary()

        # Step 2: Parse the main XML file
        version, parsed_drugs = self._parse_xml(limit=limit)

        # Step 3: Integrate vocabulary data into the parsed XML data
        self._integrate_data(version, parsed_drugs)

        # Step 4: Validate the final integrated data
        self._validate_data()
        
        self.logger.info("DrugBank parsing and integration process complete.")

    def _load_vocabulary(self) -> None:
        """Loads and prepares the DrugBank vocabulary CSV."""
        self.logger.info(f"Loading DrugBank vocabulary from {self.vocabulary_path}")
        try:
            self.vocabulary_df = pd.read_csv(self.vocabulary_path)
            # Rename columns for consistency
            column_mapping = {
                "DrugBank ID": "drugbank_id", "Accession Numbers": "accession_numbers",
                "Common name": "name", "CAS": "cas_number", "UNII": "unii",
                "Synonyms": "synonyms", "Standard InChI Key": "inchikey"
            }
            self.vocabulary_df.rename(columns=column_mapping, inplace=True)
            # Set drugbank_id as the index for fast lookups
            self.vocabulary_df.set_index('drugbank_id', inplace=True)
            self.logger.info(f"Loaded {len(self.vocabulary_df)} entries from vocabulary.")
        except FileNotFoundError:
            self.logger.error(f"Vocabulary file not found: {self.vocabulary_path}. Proceeding without enrichment.")
            self.vocabulary_df = None
        except Exception as e:
            self.logger.error(f"Error loading vocabulary CSV: {e}. Proceeding without enrichment.")
            self.vocabulary_df = None

    def _parse_xml(self, limit: Optional[int]) -> Tuple[str, List[Dict]]:
        """Parses the DrugBank XML file iteratively using lxml."""
        self.logger.info(f"Parsing DrugBank XML file: {self.xml_path}")
        drugs = []
        version = "unknown"
        try:
            # <<< THE FIX IS HERE: Changed 'self.xml_file_path' to 'self.xml_path'
            context = etree.iterparse(self.xml_path, events=("start", "end"), recover=True)
            drug_count = 0
            
            for event, elem in context:
                # Detect namespace from the root element
                if event == 'start' and 'drugbank' in elem.tag and not self.ns:
                    if '}' in elem.tag and elem.tag.startswith('{'):
                        self.ns = {"db": elem.tag.split('}')[0][1:]}
                        version_elem = elem.find("db:version", self.ns)
                        if version_elem is not None:
                            version = version_elem.text.strip()
                        self.logger.info(f"Detected DrugBank Version: {version}, Namespace: {self.ns['db']}")

                # Process each 'drug' element at its end
                if event == 'end' and elem.tag == f"{{{self.ns.get('db')}}}drug":
                    try:
                        drug_dict = self._parse_drug_element(elem)
                        if drug_dict:
                            drugs.append(drug_dict)
                            drug_count += 1
                    except Exception as e:
                        self.logger.error(f"Error parsing a drug element: {e}", exc_info=True)
                    
                    # Clear element for memory efficiency
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                
                if limit and drug_count >= limit:
                    break

        except etree.XMLSyntaxError as e:
            self.logger.error(f"XML Syntax Error in {self.xml_path}: {e}")
        except FileNotFoundError:
            self.logger.error(f"XML file not found: {self.xml_path}")
        
        self.logger.info(f"Successfully parsed {len(drugs)} drugs from XML.")
        return version, drugs

    def _integrate_data(self, version: str, drugs: List[Dict]):
        """Enriches parsed drug data with information from the vocabulary."""
        if self.vocabulary_df is None:
            self.logger.warning("Vocabulary not loaded. Skipping integration.")
            self.integrated_data = drugs
            return
        
        self.logger.info("Integrating XML data with vocabulary...")
        enriched_drugs = []
        for drug in tqdm(drugs, desc="Enriching drug data"):
            drug_id = drug.get("drugbank_id")
            if drug_id in self.vocabulary_df.index:
                vocab_series = self.vocabulary_df.loc[drug_id]
                # Enrich fields that are missing or empty in the XML data
                if not drug.get("inchikey") and pd.notna(vocab_series.get("inchikey")):
                    drug["inchikey"] = vocab_series["inchikey"]
                if not drug.get("unii") and pd.notna(vocab_series.get("unii")):
                    drug["unii"] = vocab_series["unii"]
                # Combine synonyms
                xml_syns = set(drug.get("synonyms", []))
                vocab_syns_raw = vocab_series.get("synonyms")
                if pd.notna(vocab_syns_raw):
                    vocab_syns = set(vocab_syns_raw.split('|'))
                    drug["synonyms"] = sorted(list(xml_syns.union(vocab_syns)))
            enriched_drugs.append(drug)
        
        # The final processed data is just the list of drugs
        self.integrated_data = enriched_drugs
        self.logger.info(f"Finished integration. Total drugs: {len(self.integrated_data)}")

    def _validate_data(self) -> None:
        """Performs a simple validation on the final integrated data."""
        if not self.integrated_data:
            self.logger.warning("No integrated data to validate.")
            return

        num_drugs = len(self.integrated_data)
        missing_fields = {"drugbank_id": 0, "name": 0, "inchikey": 0}
        
        for drug in self.integrated_data:
            for field in missing_fields:
                if not drug.get(field):
                    missing_fields[field] += 1
        
        self.logger.info("Data validation results:")
        for field, count in missing_fields.items():
            percentage = (count / num_drugs) * 100
            self.logger.info(f"  - Drugs missing '{field}': {count} ({percentage:.2f}%)")

    def save(self, format: str = "pickle") -> Optional[str]:
        """Saves the final integrated data to a file."""
        if not self.integrated_data:
            self.logger.warning("No integrated data to save.")
            return None
        
        # We save just the list of drugs, not the wrapper dictionary
        output_path = os.path.join(self.output_dir, f"drugbank_parsed.{format}")
        self.logger.info(f"Saving {len(self.integrated_data)} drugs to {output_path}...")

        try:
            if format == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(self.integrated_data, f)
            elif format == "json":
                with open(output_path, "w") as f:
                    json.dump(self.integrated_data, f, indent=2)
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None
            
            self.logger.info(f"Successfully saved data to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return None

    # --- Helper methods for parsing specific XML elements ---
    # These are consolidated from the original xml_parser.py
    
    def _get_text(self, element: etree._Element, xpath: str) -> Optional[str]:
        """Safely gets text from a sub-element."""
        target = element.find(xpath, self.ns)
        return target.text.strip() if target is not None and target.text else None

    def _parse_list(self, element: etree._Element, xpath: str) -> List[str]:
        """Parses a list of simple text elements."""
        items = []
        for item in element.findall(xpath, self.ns):
            if item.text:
                items.append(item.text.strip())
        return items

    def _parse_drug_element(self, drug_elem: etree._Element) -> Dict[str, Any]:
        """Parses a single <drug> element into a dictionary."""
        # Get the primary ID first
        primary_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", self.ns)
        drug_id = primary_id_elem.text.strip() if primary_id_elem is not None else self._get_text(drug_elem, "db:drugbank-id")
        
        if not drug_id:
            return None

        # Extract calculated properties for InChIKey
        inchikey = None
        calc_props = drug_elem.find("db:calculated-properties", self.ns)
        if calc_props is not None:
            for prop in calc_props.findall("db:property", self.ns):
                kind = self._get_text(prop, "db:kind")
                if kind == "InChIKey":
                    inchikey = self._get_text(prop, "db:value")
                    break

        return {
            "drugbank_id": drug_id,
            "name": self._get_text(drug_elem, "db:name"),
            "description": self._get_text(drug_elem, "db:description"),
            "cas_number": self._get_text(drug_elem, "db:cas-number"),
            "unii": self._get_text(drug_elem, "db:unii"),
            "inchikey": inchikey,
            "state": self._get_text(drug_elem, "db:state"),
            "groups": self._parse_list(drug_elem, "db:groups/db:group"),
            "indication": self._get_text(drug_elem, "db:indication"),
            "pharmacodynamics": self._get_text(drug_elem, "db:pharmacodynamics"),
            "mechanism_of_action": self._get_text(drug_elem, "db:mechanism-of-action"),
            "synonyms": self._parse_list(drug_elem, "db:synonyms/db:synonym"),
            "targets": self._parse_protein_relations(drug_elem, "db:targets/db:target"),
            "enzymes": self._parse_protein_relations(drug_elem, "db:enzymes/db:enzyme"),
            "transporters": self._parse_protein_relations(drug_elem, "db:transporters/db:transporter"),
            "carriers": self._parse_protein_relations(drug_elem, "db:carriers/db:carrier"),
        }

    def _parse_protein_relations(self, drug_elem: etree._Element, xpath: str) -> List[Dict]:
        """Parses protein relationships like targets, enzymes, etc."""
        relations = []
        for protein_elem in drug_elem.findall(xpath, self.ns):
            polypeptide = protein_elem.find("db:polypeptide", self.ns)
            uniprot_id = None
            if polypeptide is not None:
                ext_ids = polypeptide.find("db:external-identifiers", self.ns)
                if ext_ids is not None:
                    for ext_id in ext_ids.findall("db:external-identifier", self.ns):
                        if self._get_text(ext_id, "db:resource") == "UniProtKB":
                            uniprot_id = self._get_text(ext_id, "db:identifier")
                            break
            
            relation = {
                "id": self._get_text(protein_elem, "db:id"),
                "name": self._get_text(protein_elem, "db:name"),
                "organism": self._get_text(protein_elem, "db:organism"),
                "actions": self._parse_list(protein_elem, "db:actions/db:action"),
                "uniprot_id": uniprot_id
            }
            relations.append(relation)
        return relations


# --- Main execution block to make the script runnable ---
def main():
    parser = argparse.ArgumentParser(description="Parse and integrate DrugBank data from XML and CSV vocabulary.")
    parser.add_argument("--xml", required=True, help="Path to the DrugBank full_database.xml file.")
    parser.add_argument("--vocab", required=True, help="Path to the drugbank_vocabulary.csv file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the processed output file.")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output file format.")
    parser.add_argument("--limit", type=int, help="Optional: limit the number of drugs to parse for testing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize and run the parser
    db_parser = DrugBankParser(
        xml_path=args.xml,
        vocabulary_path=args.vocab,
        output_dir=args.output_dir
    )
    db_parser.parse_and_integrate(limit=args.limit)
    db_parser.save(format=args.format)

if __name__ == "__main__":
    main()
