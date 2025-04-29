# src/ddi/data/sources/drugbank/vocabulary.py
import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class DrugBankVocabulary:
    """Parser and utility for DrugBank vocabulary CSV"""
    
    def __init__(self, csv_path: str):
        """Initialize the vocabulary parser
        
        Args:
            csv_path: Path to the drugbank_all_drugbank_vocabulary.csv file
        """
        self.csv_path = csv_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = None
        self.id_mapping = {}
        self.cas_to_id = {}
        self.name_to_id = {}
        self.inchi_to_id = {}
        self.unii_to_id = {}
        
    def load(self) -> pd.DataFrame:
        """Load the vocabulary CSV

        Returns:
            Pandas DataFrame containing the vocabulary data
        """
        self.logger.info(f"Loading DrugBank vocabulary from {self.csv_path}")
        expected_cols = ["drugbank_id", "accession_numbers", "name",
                         "cas_number", "unii", "synonyms", "inchikey"] # Define expected final columns

        try:
            # Read CSV, let pandas infer types initially, but handle potential errors
            self.data = pd.read_csv(self.csv_path)

            # Rename columns to standardized names if necessary
            column_mapping = {
                "DrugBank ID": "drugbank_id",
                "Accession Numbers": "accession_numbers",
                "Common name": "name",
                "CAS": "cas_number",
                "UNII": "unii",
                "Synonyms": "synonyms",
                "Standard InChI Key": "inchikey"
            }
            rename_dict = {k: v for k, v in column_mapping.items() if k in self.data.columns}
            if rename_dict:
                self.data = self.data.rename(columns=rename_dict)

            # --- Convert relevant columns to string AFTER loading & renaming ---
            for col in expected_cols:
                if col in self.data.columns:
                    self.data[col] = self.data[col].fillna('').astype(str)
                else:
                    self.logger.warning(f"Column '{col}' not found in vocabulary CSV. Adding empty column.")
                    self.data[col] = '' # Adds the column, likely at the end

            # --- FIX: Ensure columns are in the expected order ---
            self.data = self.data[expected_cols]
            # -----------------------------------------------------

            # Build mappings using the cleaned string data
            self._build_mappings()

            self.logger.info(f"Loaded {len(self.data)} entries from vocabulary")
            return self.data

        except Exception as e:
            self.logger.error(f"Error loading or processing vocabulary CSV: {str(e)}")
            self.data = pd.DataFrame(columns=expected_cols)
            self._build_mappings()
            return self.data

    def _build_mappings(self) -> None:
        """Build mapping dictionaries for efficient lookups"""
        # Reset mappings
        self.id_mapping = {}
        self.cas_to_id = {}
        self.name_to_id = {}
        self.inchi_to_id = {}
        self.unii_to_id = {}

        if self.data is None or self.data.empty:
            self.logger.warning("Cannot build mappings: data not loaded or empty")
            return

        # --- Revert to iterrows, keeping str() conversion inside loop ---
        for idx, row in self.data.iterrows():
            try:
                # Map DrugBank ID to row index
                # Ensure row.drugbank_id is treated as string before stripping
                drug_id = str(row['drugbank_id']).strip()
                if drug_id:
                    self.id_mapping[drug_id] = idx

                # Map CAS number to DrugBank ID
                cas = str(row['cas_number']).strip()
                if cas and drug_id: # drug_id already checked
                    self.cas_to_id[cas] = drug_id

                # Map name to DrugBank ID (case-insensitive)
                name = str(row['name']).strip()
                if name and drug_id:
                    name_lower = name.lower()
                    # Add primary name (don't check if exists, primary should overwrite if duplicate)
                    self.name_to_id[name_lower] = drug_id

                # Map InChI Key to DrugBank ID
                inchi = str(row['inchikey']).strip()
                if inchi and drug_id:
                    self.inchi_to_id[inchi] = drug_id

                # Map UNII to DrugBank ID
                unii = str(row['unii']).strip()
                if unii and drug_id:
                    self.unii_to_id[unii] = drug_id

                # Parse and map synonyms (add to name_to_id)
                syns = str(row['synonyms']).strip()
                if syns and drug_id:
                    for synonym in syns.split('|'):
                        synonym_clean = synonym.strip()
                        if synonym_clean:
                            synonym_lower = synonym_clean.lower()
                            # Use lowercase for case-insensitive mapping
                            # Add synonym only if the name isn't already mapped (primary name takes precedence)
                            if synonym_lower not in self.name_to_id:
                                self.name_to_id[synonym_lower] = drug_id
            except Exception as e:
                 # Log error for the specific row but continue processing others
                 current_id = row.get('drugbank_id', f'index {idx}') # Get ID if possible
                 self.logger.error(f"Error processing mappings for row {current_id}: {e}")
                 continue 

    def get_drug_by_id(self, drugbank_id: str) -> Optional[Dict[str, Any]]:
        """Get drug information by DrugBank ID"""
        # Strip input ID just in case
        drugbank_id_clean = drugbank_id.strip()
        if drugbank_id_clean not in self.id_mapping:
            return None

        idx = self.id_mapping[drugbank_id_clean]
        # Use .loc for potentially safer access than iloc if index isn't guaranteed sequential
        try:
             row = self.data.loc[idx]
        except KeyError:
             self.logger.warning(f"Index {idx} not found for drug {drugbank_id_clean}. Data inconsistency?")
             return None


        try:
            # Access columns using the cleaned names, handle potential missing columns gracefully
            return {
                "drugbank_id": row.get('drugbank_id', drugbank_id_clean), # Use cleaned ID as fallback
                "accession_numbers": row.get('accession_numbers', None), # Assuming this column might exist
                "name": row.get('name', None),
                "cas_number": row.get('cas_number', None),
                "unii": row.get('unii', None),
                # Split synonyms only if it's a non-empty string
                "synonyms": row.get('synonyms', '').split('|') if row.get('synonyms') else [],
                "inchikey": row.get('inchikey', None)
            }
        except Exception as e:
            self.logger.warning(f"Error retrieving data for drug {drugbank_id_clean} at index {idx}: {str(e)}")
            return {
                "drugbank_id": drugbank_id_clean,
                "name": drugbank_id_clean,
                "synonyms": []
            }

    def get_id_by_name(self, name: str) -> Optional[str]:
        """Get DrugBank ID by drug name (case-insensitive)"""
        if not isinstance(name, str): # Handle non-string input
             name = str(name)
        # Use cleaned name for lookup
        return self.name_to_id.get(name.strip().lower())

    # ... (get_id_by_cas, get_id_by_inchikey, get_id_by_unii should also strip input) ...
    def get_id_by_cas(self, cas: str) -> Optional[str]:
        if not isinstance(cas, str): cas = str(cas)
        return self.cas_to_id.get(cas.strip())

    def get_id_by_inchikey(self, inchikey: str) -> Optional[str]:
        if not isinstance(inchikey, str): inchikey = str(inchikey)
        return self.inchi_to_id.get(inchikey.strip())

    def get_id_by_unii(self, unii: str) -> Optional[str]:
        if not isinstance(unii, str): unii = str(unii)
        return self.unii_to_id.get(unii.strip())

    def validate_drug_id(self, drugbank_id: str) -> bool:
        """Validate if a DrugBank ID exists in the vocabulary"""
        if not isinstance(drugbank_id, str): return False
        return drugbank_id.strip() in self.id_mapping


    
    def enrich_drug_data(self, drug_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich drug data with information from vocabulary

        Args:
            drug_data: Drug data dictionary with drugbank_id

        Returns:
            Enriched drug data
        """
        if "drugbank_id" not in drug_data:
            self.logger.warning("Cannot enrich drug data: drugbank_id missing")
            return drug_data

        drugbank_id_clean = str(drug_data["drugbank_id"]).strip() # Clean ID for lookup
        voc_data = self.get_drug_by_id(drugbank_id_clean) # Use cleaned ID
        if not voc_data:
            # Log if enrichment fails because vocab data wasn't found
            self.logger.debug(f"No vocabulary data found for {drugbank_id_clean} during enrichment.")
            return drug_data

        # Start with a copy of the input data
        enriched = drug_data.copy()

        # --- Explicitly check and add/overwrite fields from vocabulary ---
        # Use voc_data.get() for safety, although get_drug_by_id should handle missing keys

        # Add accession numbers if missing in enriched
        if "accession_numbers" not in enriched or not enriched["accession_numbers"]:
            acc_nums = voc_data.get("accession_numbers")
            if acc_nums: # Only add if not None/empty
                enriched["accession_numbers"] = acc_nums

        # Add UNII if missing in enriched
        if "unii" not in enriched or not enriched["unii"]:
             unii_val = voc_data.get("unii")
             if unii_val: # Only add if not None/empty
                 enriched["unii"] = unii_val # This is the key line

        # Add InChIKey if missing in enriched
        if "inchikey" not in enriched or not enriched["inchikey"]:
            inchikey_val = voc_data.get("inchikey")
            if inchikey_val: # Only add if not None/empty
                enriched["inchikey"] = inchikey_val

        # Add CAS number if missing in enriched (less likely, but for completeness)
        if "cas_number" not in enriched or not enriched["cas_number"]:
             cas_val = voc_data.get("cas_number")
             if cas_val: # Only add if not None/empty
                 enriched["cas_number"] = cas_val

        # Merge synonyms (ensure lists are handled correctly)
        existing_synonyms = set(enriched.get("synonyms", []) or []) # Handle None from XML
        voc_synonyms_list = voc_data.get("synonyms", []) or [] # Handle None/empty list from vocab
        voc_synonyms = set(s for s in voc_synonyms_list if s) # Ensure only non-empty strings in set
        enriched["synonyms"] = sorted(list(existing_synonyms.union(voc_synonyms))) # Combine and sort for consistency

        # Overwrite name if it's missing/empty in enriched (unlikely but possible)
        if "name" not in enriched or not enriched["name"]:
             name_val = voc_data.get("name")
             if name_val:
                 enriched["name"] = name_val

        return enriched


# Example usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process DrugBank vocabulary CSV")
    parser.add_argument("--input", required=True, help="Path to drugbank_all_drugbank_vocabulary.csv")
    parser.add_argument("--query", help="DrugBank ID to look up")
    parser.add_argument("--output", help="Path to save processed vocabulary (optional)")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load vocabulary
    vocabulary = DrugBankVocabulary(args.input)
    data = vocabulary.load()
    
    logging.info(f"Loaded {len(data)} entries from vocabulary")
    
    # Show column information
    logging.info("Column info:")
    for col in data.columns:
        dtype = data[col].dtype
        non_null = data[col].count()
        logging.info(f"  - {col}: {dtype}, {non_null}/{len(data)} non-null values")
    
    # Query specific drug if requested
    if args.query:
        drug = vocabulary.get_drug_by_id(args.query)
        if drug:
            for key, value in drug.items():
                logging.info(f"{key}: {value}")
        else:
            logging.warning(f"Drug {args.query} not found in vocabulary")
    
    # Save processed vocabulary if output path provided
    if args.output:
        import pickle
        with open(args.output, 'wb') as f:
            pickle.dump({
                'data': data,
                'id_mapping': vocabulary.id_mapping,
                'name_to_id': vocabulary.name_to_id,
                'cas_to_id': vocabulary.cas_to_id,
                'inchi_to_id': vocabulary.inchi_to_id
            }, f)
        logging.info(f"Saved processed vocabulary to {args.output}")

if __name__ == "__main__":
    main()