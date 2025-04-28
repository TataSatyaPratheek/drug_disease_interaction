# src/ddi/data/sources/drugbank/integration.py
import os
import logging
import pickle
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm

from ..drugbank.xml_parser import DrugBankXMLParser
from ..drugbank.vocabulary import DrugBankVocabulary

class DrugBankIntegrator:
    """Integrates DrugBank XML and vocabulary data"""
    
    def __init__(self, xml_path: str, vocabulary_path: str, output_dir: str = None):
        """Initialize integrator
        
        Args:
            xml_path: Path to full_database.xml
            vocabulary_path: Path to drugbank_all_drugbank_vocabulary.csv
            output_dir: Output directory for processed data
        """
        self.xml_path = xml_path
        self.vocabulary_path = vocabulary_path
        self.output_dir = output_dir or "data/processed/drugs"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize parsers
        self.xml_parser = DrugBankXMLParser(xml_path)
        self.vocabulary = DrugBankVocabulary(vocabulary_path)
        
        # Data containers
        self.xml_data = None
        self.integrated_data = None
        
    def process(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Process and integrate DrugBank data
        
        Args:
            limit: Optional limit on number of drugs to parse
            
        Returns:
            Dictionary with integrated data
        """
        # Load vocabulary
        self.logger.info("Loading DrugBank vocabulary")
        self.vocabulary.load()
        
        # Parse XML
        self.logger.info("Parsing DrugBank XML")
        self.xml_data = self.xml_parser.parse(limit=limit)
        
        # Integrate data
        self.logger.info("Integrating XML and vocabulary data")
        self._integrate_data()
        
        # Validate data
        self._validate_data()
        
        return self.integrated_data
    
    def _integrate_data(self) -> None:
        """Integrate XML and vocabulary data"""
        integrated_drugs = []
        
        for drug in tqdm(self.xml_data["drugs"], desc="Integrating drug data"):
            # Get DrugBank ID
            drugbank_id = drug["drugbank_id"]
            
            # Check if drug exists in vocabulary
            if self.vocabulary.validate_drug_id(drugbank_id):
                # Enrich drug data with vocabulary information
                integrated_drug = self.vocabulary.enrich_drug_data(drug)
                integrated_drugs.append(integrated_drug)
            else:
                self.logger.warning(f"Drug {drugbank_id} not found in vocabulary, using XML data only")
                integrated_drugs.append(drug)
        
        self.integrated_data = {
            "version": self.xml_data["version"],
            "drugs": integrated_drugs
        }
        
        self.logger.info(f"Integrated {len(integrated_drugs)} drugs")
    
    def _validate_data(self) -> None:
        """Validate integrated data"""
        if not self.integrated_data:
            self.logger.warning("No integrated data to validate")
            return
            
        drugs = self.integrated_data["drugs"]
        
        # Check essential fields
        missing_fields = {
            "drugbank_id": 0,
            "name": 0,
            "cas_number": 0,
            "inchikey": 0
        }
        
        for drug in drugs:
            for field in missing_fields.keys():
                if field not in drug or not drug[field]:
                    missing_fields[field] += 1
        
        self.logger.info("Data validation results:")
        for field, count in missing_fields.items():
            percentage = (count / len(drugs)) * 100
            self.logger.info(f"  - {field}: {count} drugs missing ({percentage:.2f}%)")
    
    def save(self, format: str = "pickle") -> str:
        """Save integrated data
        
        Args:
            format: Output format (pickle or json)
            
        Returns:
            Path to saved file
        """
        if not self.integrated_data:
            self.logger.warning("No integrated data to save")
            return None
            
        output_path = os.path.join(self.output_dir, f"drugbank_integrated.{format}")
        
        if format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(self.integrated_data, f)
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(self.integrated_data, f, indent=2)
        else:
            self.logger.error(f"Unsupported output format: {format}")
            return None
            
        self.logger.info(f"Saved integrated data to {output_path}")
        return output_path


# Update the graph builder to use the vocabulary data
def update_graph_builder_for_vocabulary():
    """Code snippet to update graph builder with vocabulary integration"""
    
    # This is pseudocode to show how the graph builder should be modified
    # to use vocabulary information
    
    """
    # In KnowledgeGraphBuilder._add_drugs method, update to use vocabulary:
    
    # Load the vocabulary for cross-reference
    vocabulary = DrugBankVocabulary(vocab_path)
    vocabulary.load()
    
    for drug in drugs:
        drug_id = drug["drugbank_id"]
        
        # Validate ID against vocabulary
        if not vocabulary.validate_drug_id(drug_id):
            self.logger.warning(f"Drug {drug_id} not in vocabulary, skipping")
            continue
        
        # Get additional info from vocabulary if needed
        voc_data = vocabulary.get_drug_by_id(drug_id)
        
        # Use InChIKey as a property if available
        inchikey = voc_data.get("inchikey") if voc_data else drug.get("inchikey")
        
        # Add all synonyms from both XML and vocabulary
        synonyms = set(drug.get("synonyms", []))
        if voc_data and "synonyms" in voc_data:
            synonyms.update(voc_data["synonyms"])
        
        # Add node to graph with enhanced properties
        self._add_node(
            node_id=drug_id,
            node_type="drug",
            name=drug["name"],
            properties={
                # ... other properties ...
                "inchikey": inchikey,
                "synonyms": list(synonyms),
                # ... more properties ...
            }
        )
    """
    
    return "Updated KnowledgeGraphBuilder to use vocabulary for ID validation and enrichment"


# Example script to merge DrugBank data sources
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate DrugBank XML and vocabulary data")
    parser.add_argument("--xml", required=True, help="Path to full_database.xml")
    parser.add_argument("--vocab", required=True, help="Path to drugbank_all_drugbank_vocabulary.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output format")
    parser.add_argument("--limit", type=int, help="Limit number of drugs to parse (for testing)")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize integrator
    integrator = DrugBankIntegrator(args.xml, args.vocab, args.output)
    
    # Process data
    integrated_data = integrator.process(limit=args.limit)
    
    # Save data
    output_path = integrator.save(format=args.format)
    
    logging.info(f"Integrated data saved to {output_path}")
    
    # Provide sample code for graph builder
    update_graph_builder_for_vocabulary()

if __name__ == "__main__":
    main()