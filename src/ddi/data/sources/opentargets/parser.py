# src/ddi/data/sources/opentargets/parser.py
import os
import logging
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import pickle
import json
from typing import List, Dict, Any, Optional

class OpenTargetsParser:
    """
    Parses OpenTargets parquet files, focusing *only* on target-disease associations
    (type 'associationByDatasourceDirect').
    """
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        # Simplified: Only store target-disease associations
        self.target_disease_associations: List[Dict[str, Any]] = []

    def parse_parquet_files(self) -> None:
        """
        Iterates through parquet files in the data directory, reads them,
        and extracts only target-disease associations.
        """
        self.logger.info(f"Parsing OpenTargets parquet files from {self.data_dir}")
        try:
            all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
            self.logger.info(f"Found {len(all_files)} parquet files")
        except FileNotFoundError:
            self.logger.error(f"Data directory not found: {self.data_dir}")
            return
        except Exception as e:
            self.logger.error(f"Error listing files in {self.data_dir}: {e}")
            return

        # Reset list before parsing
        self.target_disease_associations = []
        processed_td_count = 0

        for file_path in tqdm(all_files, desc="Processing parquet files"):
            try:
                table = pq.read_table(file_path)
                df = table.to_pandas()

                # Process rows, looking *only* for target-disease associations
                for row in df.itertuples(index=False):
                    row_dict = row._asdict() # Convert namedtuple to dict
                    row_type = row_dict.get('type')

                    if row_type == 'associationByDatasourceDirect':
                        # Extract relevant fields for target-disease association
                        assoc = {
                            'target_id': row_dict.get('targetId'),
                            'disease_id': row_dict.get('diseaseId'),
                            'score': row_dict.get('score'),
                            'datasource': row_dict.get('datasourceId'),
                            # Add other relevant fields if needed from this type
                        }
                        # Basic validation
                        if assoc['target_id'] and assoc['disease_id'] and assoc['score'] is not None:
                            self.target_disease_associations.append(assoc)
                            processed_td_count += 1
                        else:
                             self.logger.debug(f"Skipping incomplete target-disease association: {assoc}")

                    # Ignore all other row types ('target', 'disease', 'drug', 'knownDrug', 'indication')

            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}", exc_info=True)

        self.logger.info(f"Processed {processed_td_count} target-disease associations")
        # Log if other types were expected but not found (based on previous runs)
        self.logger.warning("Parser is now configured to ONLY process 'associationByDatasourceDirect'. Other types are ignored.")


    def save_data(self, format: str = "pickle") -> None:
        """
        Saves the extracted target-disease associations to a file.
        Removes saving for entities and indications.

        Args:
            format (str): Output format ('pickle' or 'json').
        """
        if not self.target_disease_associations:
            self.logger.warning("No target-disease associations were processed. Saving an empty file.")
            # Still save an empty file to avoid downstream "File not found" errors if expected
            # Or could choose to not save anything

        # --- Save Associations (Only Target-Disease) ---
        assoc_filename = f"opentargets_target_disease_associations.{format}" # Renamed file
        assoc_path = os.path.join(self.output_dir, assoc_filename)
        try:
            if format == "pickle":
                with open(assoc_path, "wb") as f:
                    pickle.dump(self.target_disease_associations, f)
            elif format == "json":
                with open(assoc_path, "w", encoding='utf-8') as f:
                    json.dump(self.target_disease_associations, f, indent=2, ensure_ascii=False)
            else:
                self.logger.error(f"Unsupported format: {format}")
                return
            self.logger.info(f"Saved OpenTargets target-disease associations to {assoc_path}")
        except Exception as e:
            self.logger.error(f"Failed to save associations to {assoc_path}: {e}", exc_info=True)

        # --- Remove Entity Saving ---
        # self.logger.info("Entity saving is removed as entities are no longer extracted.")

        # --- Remove Indication Saving ---
        # self.logger.info("Drug-disease indication saving is removed.")


# Main execution block
def main():
    import argparse
    parser_arg = argparse.ArgumentParser(description="Parse OpenTargets parquet files focusing on target-disease associations.")
    parser_arg.add_argument("--data_dir", required=True, help="Directory containing raw OpenTargets parquet files.")
    parser_arg.add_argument("--output_dir", required=True, help="Directory to save processed data.")
    parser_arg.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output format for saved files.")
    args = parser_arg.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize and run parser
    parser = OpenTargetsParser(args.data_dir, args.output_dir)
    parser.parse_parquet_files()
    parser.save_data(format=args.format)

    logging.info("OpenTargets target-disease parsing complete.")

if __name__ == "__main__":
    main()
