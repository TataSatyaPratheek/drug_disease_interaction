# src/ddi/data/sources/opentargets/parser.py
import os
import logging
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import pickle
import json

class OpenTargetsParser:
    """Parser for OpenTargets drug-target-disease association data"""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """Initialize OpenTargets parser
        
        Args:
            data_dir: Directory containing OpenTargets parquet files
            output_dir: Directory to save processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or "data/processed/associations/opentargets"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data containers
        self.drug_target_associations = []
        self.target_disease_associations = []
        self.drug_disease_associations = []
        self.targets = {}
        self.diseases = {}
        self.drugs = {}
    
    def parse_parquet_files(self, limit_files: Optional[int] = None, limit_rows: Optional[int] = None) -> Dict[str, Any]:
        """Parse OpenTargets parquet files
        
        Args:
            limit_files: Limit number of parquet files to parse (for testing)
            limit_rows: Limit number of rows to parse per file (for testing)
            
        Returns:
            Dictionary containing processed OpenTargets data
        """
        self.logger.info(f"Parsing OpenTargets parquet files from {self.data_dir}")
        
        # Find all parquet files
        parquet_files = [f for f in os.listdir(self.data_dir) if f.endswith(".parquet")]
        
        if limit_files:
            parquet_files = parquet_files[:limit_files]
            
        self.logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Process parquet files
        for file_idx, file_name in enumerate(tqdm(parquet_files, desc="Processing parquet files")):
            file_path = os.path.join(self.data_dir, file_name)
            
            try:
                # Read parquet file schema first
                parquet_file = pq.ParquetFile(file_path)
                schema = parquet_file.schema.to_arrow_schema()
                
                # Determine what type of data this file contains
                column_names = [field.name for field in schema]
                
                # Initialize lists for different types of associations
                drug_target_batch = []
                target_disease_batch = []
                drug_disease_batch = []
                
                # Read parquet file
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                if limit_rows:
                    df = df.head(limit_rows)
                
                # Identify and process different types of data based on columns
                if 'targetId' in column_names and 'diseaseId' in column_names and 'drugId' not in column_names:
                    # Target-Disease associations
                    self._process_target_disease_associations(df)
                    
                elif 'targetId' in column_names and 'drugId' in column_names and 'diseaseId' not in column_names:
                    # Drug-Target associations
                    self._process_drug_target_associations(df)
                    
                elif 'drugId' in column_names and 'diseaseId' in column_names and 'targetId' not in column_names:
                    # Drug-Disease associations
                    self._process_drug_disease_associations(df)
                    
                elif 'targetId' in column_names and 'drugId' in column_names and 'diseaseId' in column_names:
                    # Combined associations (extract all types)
                    self._process_drug_target_associations(df[['targetId', 'drugId', 'score', 'confidence']].dropna())
                    self._process_target_disease_associations(df[['targetId', 'diseaseId', 'score', 'confidence']].dropna())
                    self._process_drug_disease_associations(df[['drugId', 'diseaseId', 'score', 'confidence']].dropna())
                
                # Extract entity information
                if 'targetId' in column_names and 'targetName' in column_names:
                    self._extract_target_information(df)
                    
                if 'diseaseId' in column_names and 'diseaseName' in column_names:
                    self._extract_disease_information(df)
                    
                if 'drugId' in column_names and 'drugName' in column_names:
                    self._extract_drug_information(df)
                    
            except Exception as e:
                self.logger.error(f"Error processing parquet file {file_name}: {str(e)}")
                continue
        
        self.logger.info(f"Processed {len(self.drug_target_associations)} drug-target associations")
        self.logger.info(f"Processed {len(self.target_disease_associations)} target-disease associations")
        self.logger.info(f"Processed {len(self.drug_disease_associations)} drug-disease associations")
        self.logger.info(f"Extracted information for {len(self.targets)} targets, {len(self.diseases)} diseases, and {len(self.drugs)} drugs")
        
        # Prepare result
        result = {
            "drug_target_associations": self.drug_target_associations,
            "target_disease_associations": self.target_disease_associations,
            "drug_disease_associations": self.drug_disease_associations,
            "targets": self.targets,
            "diseases": self.diseases,
            "drugs": self.drugs
        }
        
        return result
    
    def _process_drug_target_associations(self, df: pd.DataFrame) -> None:
        """Process drug-target associations from dataframe
        
        Args:
            df: DataFrame containing drug-target associations
        """
        # Ensure required columns are present
        required_columns = {'targetId', 'drugId'}
        if not required_columns.issubset(df.columns):
            self.logger.warning(f"Missing required columns for drug-target associations: {required_columns - set(df.columns)}")
            return
        
        # Extract drug-target associations
        for _, row in df.iterrows():
            try:
                target_id = row['targetId']
                drug_id = row['drugId']
                
                # Skip if target or drug ID is missing
                if pd.isna(target_id) or pd.isna(drug_id):
                    continue
                
                # Create association record
                association = {
                    "target_id": target_id,
                    "drug_id": drug_id,
                }
                
                # Add optional fields if available
                if 'score' in df.columns and not pd.isna(row['score']):
                    association["score"] = float(row['score'])
                    
                if 'confidence' in df.columns and not pd.isna(row['confidence']):
                    association["confidence"] = float(row['confidence'])
                
                if 'mechanism' in df.columns and not pd.isna(row['mechanism']):
                    association["mechanism"] = row['mechanism']
                    
                if 'action' in df.columns and not pd.isna(row['action']):
                    association["action"] = row['action']
                
                # Add association to list
                self.drug_target_associations.append(association)
                
            except Exception as e:
                self.logger.warning(f"Error processing drug-target association: {str(e)}")
                continue
    
    def _process_target_disease_associations(self, df: pd.DataFrame) -> None:
        """Process target-disease associations from dataframe
        
        Args:
            df: DataFrame containing target-disease associations
        """
        # Ensure required columns are present
        required_columns = {'targetId', 'diseaseId'}
        if not required_columns.issubset(df.columns):
            self.logger.warning(f"Missing required columns for target-disease associations: {required_columns - set(df.columns)}")
            return
        
        # Extract target-disease associations
        for _, row in df.iterrows():
            try:
                target_id = row['targetId']
                disease_id = row['diseaseId']
                
                # Skip if target or disease ID is missing
                if pd.isna(target_id) or pd.isna(disease_id):
                    continue
                
                # Create association record
                association = {
                    "target_id": target_id,
                    "disease_id": disease_id,
                }
                
                # Add optional fields if available
                if 'score' in df.columns and not pd.isna(row['score']):
                    association["score"] = float(row['score'])
                    
                if 'confidence' in df.columns and not pd.isna(row['confidence']):
                    association["confidence"] = float(row['confidence'])
                
                if 'evidence' in df.columns and not pd.isna(row['evidence']):
                    association["evidence"] = row['evidence']
                    
                # Add association to list
                self.target_disease_associations.append(association)
                
            except Exception as e:
                self.logger.warning(f"Error processing target-disease association: {str(e)}")
                continue
    
    def _process_drug_disease_associations(self, df: pd.DataFrame) -> None:
        """Process drug-disease associations from dataframe
        
        Args:
            df: DataFrame containing drug-disease associations
        """
        # Ensure required columns are present
        required_columns = {'drugId', 'diseaseId'}
        if not required_columns.issubset(df.columns):
            self.logger.warning(f"Missing required columns for drug-disease associations: {required_columns - set(df.columns)}")
            return
        
        # Extract drug-disease associations
        for _, row in df.iterrows():
            try:
                drug_id = row['drugId']
                disease_id = row['diseaseId']
                
                # Skip if drug or disease ID is missing
                if pd.isna(drug_id) or pd.isna(disease_id):
                    continue
                
                # Create association record
                association = {
                    "drug_id": drug_id,
                    "disease_id": disease_id,
                }
                
                # Add optional fields if available
                if 'score' in df.columns and not pd.isna(row['score']):
                    association["score"] = float(row['score'])
                    
                if 'confidence' in df.columns and not pd.isna(row['confidence']):
                    association["confidence"] = float(row['confidence'])
                
                if 'clinicalPhase' in df.columns and not pd.isna(row['clinicalPhase']):
                    association["clinical_phase"] = row['clinicalPhase']
                    
                if 'clinicalStatus' in df.columns and not pd.isna(row['clinicalStatus']):
                    association["clinical_status"] = row['clinicalStatus']
                
                # Add association to list
                self.drug_disease_associations.append(association)
                
            except Exception as e:
                self.logger.warning(f"Error processing drug-disease association: {str(e)}")
                continue
    
    def _extract_target_information(self, df: pd.DataFrame) -> None:
        """Extract target information from dataframe
        
        Args:
            df: DataFrame containing target information
        """
        # Ensure required columns are present
        required_columns = {'targetId', 'targetName'}
        if not required_columns.issubset(df.columns):
            return
        
        # Extract target information
        target_df = df[['targetId', 'targetName']].drop_duplicates(subset=['targetId']).dropna()
        
        for _, row in target_df.iterrows():
            target_id = row['targetId']
            target_name = row['targetName']
            
            # Skip if already processed
            if target_id in self.targets:
                continue
                
            # Create target record
            target = {
                "id": target_id,
                "name": target_name,
            }
            
            # Add optional fields if available
            optional_fields = [
                ('targetSymbol', 'symbol'),
                ('targetBiotype', 'biotype'),
                ('targetClass', 'target_class'),
                ('targetDescription', 'description')
            ]
            
            for df_field, target_field in optional_fields:
                if df_field in df.columns and not pd.isna(row.get(df_field, None)):
                    target[target_field] = row[df_field]
            
            # Add target to dictionary
            self.targets[target_id] = target
    
    def _extract_disease_information(self, df: pd.DataFrame) -> None:
        """Extract disease information from dataframe
        
        Args:
            df: DataFrame containing disease information
        """
        # Ensure required columns are present
        required_columns = {'diseaseId', 'diseaseName'}
        if not required_columns.issubset(df.columns):
            return
        
        # Extract disease information
        disease_df = df[['diseaseId', 'diseaseName']].drop_duplicates(subset=['diseaseId']).dropna()
        
        for _, row in disease_df.iterrows():
            disease_id = row['diseaseId']
            disease_name = row['diseaseName']
            
            # Skip if already processed
            if disease_id in self.diseases:
                continue
                
            # Create disease record
            disease = {
                "id": disease_id,
                "name": disease_name,
            }
            
            # Add optional fields if available
            optional_fields = [
                ('diseaseDescription', 'description'),
                ('diseaseClass', 'disease_class'),
                ('therapeuticArea', 'therapeutic_area')
            ]
            
            for df_field, disease_field in optional_fields:
                if df_field in df.columns and not pd.isna(row.get(df_field, None)):
                    disease[disease_field] = row[df_field]
            
            # Add disease to dictionary
            self.diseases[disease_id] = disease
    
    def _extract_drug_information(self, df: pd.DataFrame) -> None:
        """Extract drug information from dataframe
        
        Args:
            df: DataFrame containing drug information
        """
        # Ensure required columns are present
        required_columns = {'drugId', 'drugName'}
        if not required_columns.issubset(df.columns):
            return
        
        # Extract drug information
        drug_df = df[['drugId', 'drugName']].drop_duplicates(subset=['drugId']).dropna()
        
        for _, row in drug_df.iterrows():
            drug_id = row['drugId']
            drug_name = row['drugName']
            
            # Skip if already processed
            if drug_id in self.drugs:
                continue
                
            # Create drug record
            drug = {
                "id": drug_id,
                "name": drug_name,
            }
            
            # Add optional fields if available
            optional_fields = [
                ('drugType', 'type'),
                ('drugDescription', 'description'),
                ('drugYearOfFirstApproval', 'year_first_approval'),
                ('drugMaxPhase', 'max_phase'),
                ('mechanismOfAction', 'mechanism_of_action')
            ]
            
            for df_field, drug_field in optional_fields:
                if df_field in df.columns and not pd.isna(row.get(df_field, None)):
                    drug[drug_field] = row[df_field]
            
            # Add drug to dictionary
            self.drugs[drug_id] = drug
    
    def save_opentargets_data(self, format: str = "pickle") -> Dict[str, str]:
        """Save processed OpenTargets data
        
        Args:
            format: Output format (pickle or json)
            
        Returns:
            Dictionary of output file paths
        """
        # Prepare data
        associations = {
            "drug_target_associations": self.drug_target_associations,
            "target_disease_associations": self.target_disease_associations,
            "drug_disease_associations": self.drug_disease_associations
        }
        
        entities = {
            "targets": self.targets,
            "diseases": self.diseases,
            "drugs": self.drugs
        }
        
        # Save associations
        assoc_path = os.path.join(self.output_dir, f"opentargets_associations.{format}")
        
        if format == "pickle":
            with open(assoc_path, "wb") as f:
                pickle.dump(associations, f)
        elif format == "json":
            with open(assoc_path, "w") as f:
                json.dump(associations, f, indent=2)
        
        # Save entities
        entities_path = os.path.join(self.output_dir, f"opentargets_entities.{format}")
        
        if format == "pickle":
            with open(entities_path, "wb") as f:
                pickle.dump(entities, f)
        elif format == "json":
            with open(entities_path, "w") as f:
                json.dump(entities, f, indent=2)
                
        self.logger.info(f"Saved OpenTargets associations to {assoc_path}")
        self.logger.info(f"Saved OpenTargets entities to {entities_path}")
        
        return {
            "associations": assoc_path,
            "entities": entities_path
        }
        
    def extract_drug_disease_indications(self) -> List[Dict[str, Any]]:
        """Extract drug-disease indications from processed data
        
        Returns:
            List of drug-disease indication dictionaries
        """
        indications = []
        
        # Process drug-disease associations
        for assoc in self.drug_disease_associations:
            try:
                drug_id = assoc["drug_id"]
                disease_id = assoc["disease_id"]
                
                # Only include associations with a high score
                if "score" in assoc and assoc["score"] < 0.5:
                    continue
                    
                # Get drug and disease info
                drug = self.drugs.get(drug_id, {"id": drug_id, "name": drug_id})
                disease = self.diseases.get(disease_id, {"id": disease_id, "name": disease_id})
                
                # Create indication record
                indication = {
                    "drug_id": drug_id,
                    "disease_id": disease_id,
                    "drug_name": drug["name"],
                    "disease_name": disease["name"],
                    "score": assoc.get("score", None),
                    "confidence": assoc.get("confidence", None),
                    "clinical_phase": assoc.get("clinical_phase", None),
                    "clinical_status": assoc.get("clinical_status", None),
                    "source": "opentargets"
                }
                
                indications.append(indication)
                
            except Exception as e:
                self.logger.warning(f"Error extracting indication: {str(e)}")
                continue
                
        self.logger.info(f"Extracted {len(indications)} drug-disease indications")
        return indications
    
    def save_indications(self, format: str = "pickle") -> str:
        """Save drug-disease indications
        
        Args:
            format: Output format (pickle or json)
            
        Returns:
            Path to saved file
        """
        # Extract indications
        indications = self.extract_drug_disease_indications()
        
        # Save indications
        output_path = os.path.join(self.output_dir, f"drug_disease_indications.{format}")
        
        if format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(indications, f)
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(indications, f, indent=2)
        else:
            self.logger.error(f"Unsupported output format: {format}")
            return None
            
        self.logger.info(f"Saved drug-disease indications to {output_path}")
        return output_path


# Example usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse OpenTargets parquet files")
    parser.add_argument("--data_dir", required=True, help="Directory containing OpenTargets parquet files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output format")
    parser.add_argument("--limit_files", type=int, help="Limit number of parquet files to parse (for testing)")
    parser.add_argument("--limit_rows", type=int, help="Limit number of rows to parse per file (for testing)")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize parser
    parser = OpenTargetsParser(args.data_dir, args.output_dir)
    
    # Parse OpenTargets data
    parser.parse_parquet_files(limit_files=args.limit_files, limit_rows=args.limit_rows)
    
    # Save data
    parser.save_opentargets_data(format=args.format)
    parser.save_indications(format=args.format)
    
    logging.info("OpenTargets parsing complete")

if __name__ == "__main__":
    main()