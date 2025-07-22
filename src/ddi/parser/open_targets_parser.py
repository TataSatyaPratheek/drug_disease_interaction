# src/ddi/parser/open_targets_parser.py

import os
import logging
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import List, Dict, Any

class OpenTargetsParser:
    """
    Parses OpenTargets parquet files for target-disease associations.
    The downloaded directory itself defines the data type (e.g., associationByDatasourceDirect),
    so we process all rows within the files.
    """
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.target_disease_associations: List[Dict[str, Any]] = []

    def parse_parquet_files(self) -> None:
        """
        Iterates through parquet files and extracts target-disease associations from all rows.
        """
        self.logger.info(f"Parsing OpenTargets parquet files from {self.data_dir}")
        try:
            all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
            self.logger.info(f"Found {len(all_files)} parquet files")
        except FileNotFoundError:
            self.logger.error(f"Data directory not found: {self.data_dir}")
            return
        
        self.target_disease_associations = []
        for file_path in tqdm(all_files, desc="Processing parquet files"):
            try:
                table = pq.read_table(file_path)
                df = table.to_pandas()
                for row in df.itertuples(index=False):
                    row_dict = row._asdict()
                    
                    # --- THE FIX IS HERE ---
                    # We no longer filter by an internal 'type' column.
                    # We assume all rows in this dataset are the associations we want.
                    assoc = {
                        'targetId': row_dict.get('targetId'),
                        'diseaseId': row_dict.get('diseaseId'),
                        'score': row_dict.get('score'),
                        'datasourceId': row_dict.get('datasourceId'),
                    }
                    if assoc['targetId'] and assoc['diseaseId'] and assoc['score'] is not None:
                        self.target_disease_associations.append(assoc)

            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}", exc_info=True)

        self.logger.info(f"Processed {len(self.target_disease_associations)} total target-disease associations")

    def save_data(self) -> None:
        """
        Converts associations to a DataFrame and saves to a pickle file.
        """
        if not self.target_disease_associations:
            self.logger.warning("No target-disease associations were processed. Nothing to save.")
            return

        associations_df = pd.DataFrame(self.target_disease_associations)
        self.logger.info(f"Created DataFrame with {len(associations_df)} associations.")
        
        assoc_filename = "opentargets_target_disease_associations.pickle"
        assoc_path = os.path.join(self.output_dir, assoc_filename)

        try:
            associations_df.to_pickle(assoc_path)
            self.logger.info(f"Saved OpenTargets associations DataFrame to {assoc_path}")
        except Exception as e:
            self.logger.error(f"Failed to save associations DataFrame to {assoc_path}: {e}", exc_info=True)

def main():
    import argparse
    parser_arg = argparse.ArgumentParser(description="Parse OpenTargets parquet files into a DataFrame.")
    parser_arg.add_argument("--data_dir", required=True, help="Directory containing raw OpenTargets parquet files.")
    parser_arg.add_argument("--output_dir", required=True, help="Directory to save processed data.")
    args = parser_arg.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = OpenTargetsParser(args.data_dir, args.output_dir)
    parser.parse_parquet_files()
    parser.save_data()
    logging.info("OpenTargets target-disease parsing complete.")

if __name__ == "__main__":
    main()
