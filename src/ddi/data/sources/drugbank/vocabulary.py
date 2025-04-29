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
        
        # Read CSV with string type enforced for text columns
        try:
            self.data = pd.read_csv(
                self.csv_path, 
                dtype={
                    'DrugBank ID': str,
                    'Common name': str,
                    'CAS': str,
                    'UNII': str,
                    'Synonyms': str,
                    'Standard InChI Key': str
                }
            )
            
            # Rename columns to standardized names if necessary
            if "DrugBank ID" in self.data.columns:
                self.data = self.data.rename(columns={
                    "DrugBank ID": "drugbank_id",
                    "Accession Numbers": "accession_numbers",
                    "Common name": "name",
                    "CAS": "cas_number",
                    "UNII": "unii",
                    "Synonyms": "synonyms",
                    "Standard InChI Key": "inchikey"
                })
            
            # Build mappings for lookups
            self._build_mappings()
            
            self.logger.info(f"Loaded {len(self.data)} entries from vocabulary")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading vocabulary CSV: {str(e)}")
            # Initialize empty DataFrame with expected columns to prevent further errors
            self.data = pd.DataFrame(columns=[
                "drugbank_id", "accession_numbers", "name", 
                "cas_number", "unii", "synonyms", "inchikey"
            ])
            return self.data
    
    def _build_mappings(self) -> None:
        """Build mapping dictionaries for efficient lookups"""
        if self.data is None:
            self.logger.warning("Cannot build mappings: data not loaded")
            return
        
        # Map DrugBank ID to row index
        self.id_mapping = {}
        for idx, row in self.data.iterrows():
            if pd.notna(row.drugbank_id):
                self.id_mapping[row.drugbank_id] = idx
        
        # Map CAS number to DrugBank ID
        self.cas_to_id = {}
        for idx, row in self.data.iterrows():
            if pd.notna(row.cas_number) and row.cas_number:
                self.cas_to_id[row.cas_number] = row.drugbank_id
                
        # Map name to DrugBank ID (case-insensitive)
        self.name_to_id = {}
        for idx, row in self.data.iterrows():
            if pd.notna(row.name) and row.name:
                try:
                    # Ensure name is a string before calling lower()
                    if isinstance(row.name, str):
                        self.name_to_id[row.name.lower()] = row.drugbank_id
                    else:
                        # Convert to string if it's not already
                        name_str = str(row.name).lower()
                        self.name_to_id[name_str] = row.drugbank_id
                        self.logger.warning(f"Name '{row.name}' was not a string, converted to '{name_str}'")
                except Exception as e:
                    self.logger.warning(f"Error processing name '{row.name}': {str(e)}")
                
        # Map InChI Key to DrugBank ID
        self.inchi_to_id = {}
        for idx, row in self.data.iterrows():
            if pd.notna(row.inchikey) and row.inchikey:
                self.inchi_to_id[row.inchikey] = row.drugbank_id
                
        # Map UNII to DrugBank ID
        self.unii_to_id = {}
        for idx, row in self.data.iterrows():
            if pd.notna(row.unii) and row.unii:
                self.unii_to_id[row.unii] = row.drugbank_id
                
        # Parse and map synonyms
        for idx, row in self.data.iterrows():
            if pd.notna(row.synonyms) and row.synonyms:
                try:
                    # Split synonyms by pipe character
                    if isinstance(row.synonyms, str):
                        for synonym in row.synonyms.split('|'):
                            # Use lowercase for case-insensitive mapping
                            # Don't overwrite existing primary names
                            if synonym.lower() not in self.name_to_id:
                                self.name_to_id[synonym.lower()] = row.drugbank_id
                except Exception as e:
                    self.logger.warning(f"Error processing synonyms for drug {row.drugbank_id}: {str(e)}")
    
    def get_drug_by_id(self, drugbank_id: str) -> Optional[Dict[str, Any]]:
        """Get drug information by DrugBank ID
        
        Args:
            drugbank_id: DrugBank ID
            
        Returns:
            Dictionary with drug information or None if not found
        """
        if drugbank_id not in self.id_mapping:
            return None
            
        idx = self.id_mapping[drugbank_id]
        row = self.data.iloc[idx]
        
        try:
            return {
                "drugbank_id": row.drugbank_id,
                "accession_numbers": row.accession_numbers if pd.notna(row.accession_numbers) else None,
                "name": row.name if pd.notna(row.name) else None,
                "cas_number": row.cas_number if pd.notna(row.cas_number) else None,
                "unii": row.unii if pd.notna(row.unii) else None,
                "synonyms": row.synonyms.split('|') if pd.notna(row.synonyms) else [],
                "inchikey": row.inchikey if pd.notna(row.inchikey) else None
            }
        except Exception as e:
            self.logger.warning(f"Error retrieving drug {drugbank_id}: {str(e)}")
            return {
                "drugbank_id": drugbank_id,
                "name": drugbank_id  # Fallback to using ID as name
            }
    
    def get_id_by_name(self, name: str) -> Optional[str]:
        """Get DrugBank ID by drug name (case-insensitive)
        
        Args:
            name: Drug name
            
        Returns:
            DrugBank ID or None if not found
        """
        try:
            return self.name_to_id.get(name.lower())
        except:
            # Handle case where name is not a string
            try:
                return self.name_to_id.get(str(name).lower())
            except:
                return None
    
    def get_id_by_cas(self, cas: str) -> Optional[str]:
        """Get DrugBank ID by CAS number
        
        Args:
            cas: CAS number
            
        Returns:
            DrugBank ID or None if not found
        """
        return self.cas_to_id.get(cas)
    
    def get_id_by_inchikey(self, inchikey: str) -> Optional[str]:
        """Get DrugBank ID by InChIKey
        
        Args:
            inchikey: InChIKey
            
        Returns:
            DrugBank ID or None if not found
        """
        return self.inchi_to_id.get(inchikey)
    
    def get_id_by_unii(self, unii: str) -> Optional[str]:
        """Get DrugBank ID by UNII
        
        Args:
            unii: UNII
            
        Returns:
            DrugBank ID or None if not found
        """
        return self.unii_to_id.get(unii)
    
    def validate_drug_id(self, drugbank_id: str) -> bool:
        """Validate if a DrugBank ID exists in the vocabulary
        
        Args:
            drugbank_id: DrugBank ID to validate
            
        Returns:
            True if ID exists, False otherwise
        """
        return drugbank_id in self.id_mapping
    
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
            
        voc_data = self.get_drug_by_id(drug_data["drugbank_id"])
        if not voc_data:
            return drug_data
            
        # Add missing fields from vocabulary
        enriched = drug_data.copy()
        
        # Add accession numbers if missing
        if "accession_numbers" not in enriched and "accession_numbers" in voc_data:
            enriched["accession_numbers"] = voc_data["accession_numbers"]
            
        # Add UNII if missing
        if "unii" not in enriched and "unii" in voc_data:
            enriched["unii"] = voc_data["unii"]
            
        # Add InChIKey if missing
        if "inchikey" not in enriched and "inchikey" in voc_data:
            enriched["inchikey"] = voc_data["inchikey"]
            
        # Merge synonyms
        existing_synonyms = set(enriched.get("synonyms", []))
        voc_synonyms = set(voc_data.get("synonyms", []))
        enriched["synonyms"] = list(existing_synonyms.union(voc_synonyms))
        
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