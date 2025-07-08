# src/scripts/analyze_raw_data.py
import argparse
import logging
import os
import sys
import gc # Import garbage collection module
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET # Use standard ET for basic structure checks
try:
    from lxml import etree # Use lxml for more robust/faster iteration if available
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    logging.warning("lxml not available. Using standard xml.etree.ElementTree for iteration (might be slower/less robust).")

try:
    import pyarrow.parquet as pq
    import pandas as pd
    from tqdm import tqdm # Import tqdm
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("pyarrow, pandas or tqdm not available. Parquet file analysis will be limited.")


# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi.utils.logging import setup_logging

# --- Helper Functions ---

def get_file_size(file_path: str) -> str:
    """Get human-readable file size."""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.2f} MB"
        else:
            return f"{size_bytes/1024**3:.2f} GB"
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {e}"

def log_element_structure(element: ET.Element, logger: logging.Logger, indent: str = "  ", max_depth=2, current_depth=0):
    """Recursively log the tag structure of an XML element."""
    if current_depth > max_depth:
        logger.info(f"{indent}...")
        return
    # Use element.tag directly if lxml, otherwise split namespace if present
    tag_name = element.tag
    if '}' in tag_name:
        tag_name = tag_name.split('}', 1)[1]
    logger.info(f"{indent}<{tag_name}{' ...' if element.text and element.text.strip() else ''}> (Attribs: {list(element.attrib.keys())})")
    for child in list(element)[:5]: # Limit children shown per level
        log_element_structure(child, logger, indent + "  ", max_depth, current_depth + 1)
    if len(list(element)) > 5:
         logger.info(f"{indent}  ...")

# --- Raw Data Analysis Functions ---

def analyze_raw_drugbank(xml_path: str):
    """Performs EDA on the raw DrugBank XML file without full parsing."""
    logger = logging.getLogger("analyze_raw_drugbank")
    logger.info("\n" + "="*30 + " Analyzing Raw DrugBank XML " + "="*30)
    logger.info(f"File: {xml_path}")
    logger.info(f"Size: {get_file_size(xml_path)}")

    if not os.path.exists(xml_path):
        logger.error("File not found.")
        return

    drug_count = 0
    version = "unknown"
    namespace = "unknown"
    first_drug_element = None
    root_tag = "unknown"
    events = ("start", "end")
    context = None

    try:
        # Use lxml if available for speed and robustness
        if LXML_AVAILABLE:
            # Attempt to get root info efficiently first
            try:
                root_context = etree.iterparse(xml_path, events=("start",), recover=True, tag='{*}drugbank')
                _, root = next(root_context)
                root_tag = root.tag
                if '}' in root_tag:
                    namespace = root_tag.split('}', 1)[0][1:] # Extract namespace from root tag
                else:
                    # Handle case where there might not be a namespace
                    namespace = "" # Or set a default if expected
                    logger.warning("Could not determine namespace from root element tag.")

                # Construct tag with namespace for finding version
                version_tag = f"{{{namespace}}}version" if namespace else "version"
                version_elem = root.find(version_tag)
                if version_elem is not None and version_elem.text:
                    version = version_elem.text.strip()
                del root_context # Clean up iterator
                del root
            except Exception as root_e:
                 logger.warning(f"Could not efficiently get root info with lxml: {root_e}")
                 # Attempt to guess namespace if root parsing failed but we know it's DrugBank
                 if namespace == "unknown":
                     namespace = "http://www.drugbank.ca"
                     logger.info(f"Assuming DrugBank namespace: {namespace}")

            # Now iterate for drug elements
            drug_tag_with_ns = f"{{{namespace}}}drug" if namespace else "drug"
            context = etree.iterparse(xml_path, events=events, recover=True, tag=drug_tag_with_ns)
            logger.info("Using lxml.etree.iterparse")

        else: # Fallback to standard ET
            context = ET.iterparse(xml_path, events=events)
            logger.info("Using xml.etree.ElementTree.iterparse")

        logger.info("Iterating through XML elements...")
        for event, elem in context:
            # Get root info with standard ET (less efficient)
            if not LXML_AVAILABLE and event == 'start' and 'drugbank' in elem.tag and drug_count == 0:
                 root_tag = elem.tag
                 if '}' in root_tag:
                     namespace = root_tag.split('}', 1)[0][1:]
                 else:
                     namespace = ""
                     logger.warning("Could not determine namespace from root element tag (standard ET).")

                 version_tag = f"{{{namespace}}}version" if namespace else "version"
                 version_elem = elem.find(version_tag) # Find version within root
                 if version_elem is not None and version_elem.text:
                     version = version_elem.text.strip()

            # Process drug elements at their end event
            # Determine the expected tag name (with or without namespace)
            expected_drug_tag = f"{{{namespace}}}drug" if namespace else "drug"

            # Check if the element tag matches the expected drug tag
            is_drug_tag = elem.tag == expected_drug_tag

            # Standard ET might sometimes return local name if namespace is handled differently
            # or if the file has no namespace. Add a fallback check.
            if not is_drug_tag and not LXML_AVAILABLE:
                 is_drug_tag = elem.tag == 'drug'

            if event == 'end' and is_drug_tag:
                drug_count += 1
                if first_drug_element is None and not LXML_AVAILABLE: # Capture first element structure with standard ET
                    try:
                        first_drug_element = elem # Keep the reference
                        logger.info("Captured first <drug> element structure (using standard ET - limited detail):")
                        log_element_structure(elem, logger, max_depth=2) # Use helper to log structure
                    except Exception as parse_e:
                        logger.warning(f"Could not log first drug element structure (standard ET): {parse_e}")

                # Clear element to save memory (crucial for iterparse)
                # lxml does this more automatically, but explicit clear is vital for standard ET
                elem.clear()
                # Also clear siblings if needed (complex with standard ET, often requires parent tracking)
                # For simplicity, we rely on elem.clear() and Python's GC

        # If using lxml, parse the first drug element separately after counting (more robust)
        if LXML_AVAILABLE and drug_count > 0:
             try:
                 drug_tag_with_ns = f"{{{namespace}}}drug" if namespace else "drug"
                 first_drug_context = etree.iterparse(xml_path, events=("end",), recover=True, tag=drug_tag_with_ns)
                 _, first_drug_element_lxml = next(first_drug_context)
                 logger.info("Sample <drug> element structure (using lxml):")
                 log_element_structure(first_drug_element_lxml, logger, max_depth=2)
                 del first_drug_context # Clean up iterator
                 del first_drug_element_lxml
             except StopIteration:
                 logger.warning("Could not find any drug elements using lxml iterparse to show structure.")
             except Exception as first_e:
                 logger.warning(f"Could not parse first drug element structure with lxml: {first_e}")


    except FileNotFoundError:
        logger.error(f"File not found: {xml_path}")
    except (ET.ParseError, etree.XMLSyntaxError) as e: # Catch both parser errors
        logger.error(f"XML Parse Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred during XML analysis: {e}", exc_info=True)
    finally:
        # Explicitly delete context iterator reference if it exists
        if context is not None:
            del context
        # Suggest garbage collection after processing the large file
        logger.info("Suggesting garbage collection after DrugBank analysis.")
        gc.collect()


    logger.info(f"Detected Root Element: {root_tag}")
    logger.info(f"Detected Namespace: {namespace}")
    logger.info(f"Detected Version: {version}")
    logger.info(f"Estimated Total <drug> elements: {drug_count}")


def analyze_raw_mesh(mesh_dir: str):
    """Performs EDA on the raw MeSH XML files (specifically desc2025.xml and qual2025.xml)."""
    logger = logging.getLogger("analyze_raw_mesh")
    logger.info("\n" + "="*30 + " Analyzing Raw MeSH XML Files (2025 only) " + "="*30)
    logger.info(f"Directory: {mesh_dir}")

    if not os.path.isdir(mesh_dir):
        logger.error("Directory not found.")
        return

    # --- Specific File Filtering for 2025 ---
    desc_file_name = "desc2025.xml"
    qual_file_name = "qual2025.xml"
    desc_file_path = os.path.join(mesh_dir, desc_file_name)
    qual_file_path = os.path.join(mesh_dir, qual_file_name)

    desc_files_found = []
    qual_files_found = []

    if os.path.exists(desc_file_path):
        desc_files_found.append(desc_file_name)
        logger.info(f"Found descriptor file: {desc_file_name}")
    else:
        logger.warning(f"Descriptor file not found: {desc_file_path}")

    if os.path.exists(qual_file_path):
        qual_files_found.append(qual_file_name)
        logger.info(f"Found qualifier file: {qual_file_name}")
    else:
        logger.warning(f"Qualifier file not found: {qual_file_path}")

    total_desc_records = 0
    total_qual_records = 0
    first_desc_element_logged = False
    first_qual_element_logged = False
    # MeSH XML typically doesn't use namespaces, but handle defensively
    desc_tag_to_find = "DescriptorRecord"
    qual_tag_to_find = "QualifierRecord"

    # Analyze Descriptor File (2025)
    if desc_files_found:
        logger.info(f"\n--- Analyzing {desc_file_name} ---")
        fpath = desc_file_path
        logger.info(f"File: {desc_file_name}, Size: {get_file_size(fpath)}")
        count = 0
        context = None
        try:
            if LXML_AVAILABLE:
                context = etree.iterparse(fpath, events=("end",), tag=desc_tag_to_find, recover=True)
            else:
                context = ET.iterparse(fpath, events=("end",))

            for event, elem in context:
                 # Check tag name robustly (lxml provides local name with tag filter)
                 is_record_tag = elem.tag == desc_tag_to_find or elem.tag.endswith('}' + desc_tag_to_find)

                 if event == 'end' and is_record_tag:
                    count += 1
                    if not first_desc_element_logged:
                        # Capture structure of the first record
                        try:
                            logger.info(f"Sample <{desc_tag_to_find}> structure (from {desc_file_name}):")
                            log_element_structure(elem, logger, max_depth=3)
                            first_desc_element_logged = True # Mark as logged
                        except Exception as struct_e:
                            logger.warning(f"Could not log first {desc_tag_to_find} structure: {struct_e}")

                    if not LXML_AVAILABLE:
                        elem.clear() # Memory management for standard ET
                    # No need to clear for lxml when using tag filter usually

            total_desc_records += count
            logger.info(f"  Found {count} <{desc_tag_to_find}> elements in {desc_file_name}")

        except (ET.ParseError, etree.XMLSyntaxError) as e:
            logger.error(f"  XML Parse Error in {desc_file_name}: {e}")
        except Exception as e:
            logger.error(f"  Error analyzing {desc_file_name}: {e}", exc_info=True)
        finally:
            if context is not None:
                del context # Ensure iterator reference is removed

    # Analyze Qualifier File (2025)
    if qual_files_found:
        logger.info(f"\n--- Analyzing {qual_file_name} ---")
        fpath = qual_file_path
        logger.info(f"File: {qual_file_name}, Size: {get_file_size(fpath)}")
        count = 0
        context = None
        try:
            if LXML_AVAILABLE:
                context = etree.iterparse(fpath, events=("end",), tag=qual_tag_to_find, recover=True)
            else:
                context = ET.iterparse(fpath, events=("end",))

            for event, elem in context:
                is_record_tag = elem.tag == qual_tag_to_find or elem.tag.endswith('}' + qual_tag_to_find)

                if event == 'end' and is_record_tag:
                    count += 1
                    if not first_qual_element_logged:
                         try:
                            logger.info(f"Sample <{qual_tag_to_find}> structure (from {qual_file_name}):")
                            log_element_structure(elem, logger, max_depth=3)
                            first_qual_element_logged = True
                         except Exception as struct_e:
                            logger.warning(f"Could not log first {qual_tag_to_find} structure: {struct_e}")

                    if not LXML_AVAILABLE:
                        elem.clear()

            total_qual_records += count
            logger.info(f"  Found {count} <{qual_tag_to_find}> elements in {qual_file_name}")

        except (ET.ParseError, etree.XMLSyntaxError) as e:
            logger.error(f"  XML Parse Error in {qual_file_name}: {e}")
        except Exception as e:
            logger.error(f"  Error analyzing {qual_file_name}: {e}", exc_info=True)
        finally:
            if context is not None:
                del context

    logger.info("\n--- MeSH 2025 Summary ---")
    logger.info(f"Total <{desc_tag_to_find}> elements found: {total_desc_records}")
    logger.info(f"Total <{qual_tag_to_find}> elements found: {total_qual_records}")
    # Suggest garbage collection
    logger.info("Suggesting garbage collection after MeSH analysis.")
    gc.collect()


def analyze_raw_opentargets(ot_dir: str):
    """Performs EDA on the raw OpenTargets Parquet files."""
    logger = logging.getLogger("analyze_raw_opentargets")
    logger.info("\n" + "="*30 + " Analyzing Raw OpenTargets Parquet Files " + "="*30)
    logger.info(f"Directory: {ot_dir}")

    if not PYARROW_AVAILABLE:
        logger.error("Cannot analyze Parquet files: pyarrow, pandas or tqdm not installed.")
        return

    if not os.path.isdir(ot_dir):
        logger.error("Directory not found.")
        return

    try:
        parquet_files = [f for f in os.listdir(ot_dir) if f.endswith(".parquet")]
    except Exception as e:
        logger.error(f"Error listing files in directory {ot_dir}: {e}")
        return

    logger.info(f"Found {len(parquet_files)} Parquet files.")

    total_rows = 0
    total_size_bytes = 0
    schemas = defaultdict(list) # Store list of files for each unique schema string
    sample_rows_per_schema = {} # Store sample rows for each unique schema string
    rows_per_file = {}

    # Wrap the loop with tqdm for progress bar
    for fname in tqdm(parquet_files, desc="Analyzing Parquet files", unit="file"):
        fpath = os.path.join(ot_dir, fname)
        parquet_file = None # Initialize to ensure it exists for finally block
        try:
            file_size = os.path.getsize(fpath)
            total_size_bytes += file_size

            # Read metadata using ParquetFile context manager if possible, or manually manage
            parquet_file = pq.ParquetFile(fpath)
            metadata = parquet_file.metadata
            num_rows = metadata.num_rows
            total_rows += num_rows
            rows_per_file[fname] = num_rows

            # Get schema
            schema = parquet_file.schema.to_arrow_schema()
            schema_key = tuple(sorted([f.name for f in schema]))
            schemas[schema_key].append(fname)

            # Get sample rows if this is the first file with this schema
            if schema_key not in sample_rows_per_schema:
                try:
                    if parquet_file.num_row_groups > 0:
                        table_sample = parquet_file.read_row_group(0, columns=schema.names)
                        df_sample = table_sample.to_pandas(self_destruct=False).head(5)

                        # Check the 'type' column in the sample if it exists
                        if 'type' in df_sample.columns:
                            sample_types = df_sample['type'].unique()
                            logger.info(f"  Sample types found in {fname}: {list(sample_types)}")
                            if not all(t == 'associationByDatasourceDirect' for t in sample_types if t):
                                logger.warning(f"  File {fname} contains types other than 'associationByDatasourceDirect' in sample: {list(sample_types)}")
                        else:
                            logger.warning(f"  File {fname} schema does not contain a 'type' column.")

                        sample_rows_per_schema[schema_key] = df_sample.to_string()
                        del table_sample
                        del df_sample
                    else:
                         sample_rows_per_schema[schema_key] = "File has no row groups."


                except Exception as sample_e:
                    logger.warning(f"Could not read sample rows from {fname}: {sample_e}")
                    sample_rows_per_schema[schema_key] = "Error reading sample."

        except FileNotFoundError:
            logger.error(f"File not found during analysis: {fpath}")
        except Exception as e:
            logger.error(f"Error analyzing {fname}: {e}", exc_info=True)
        # No explicit finally needed for ParquetFile if not using 'with', GC handles it.
        # If resource issues occur, consider `with pq.ParquetFile(fpath) as pf:` structure.

    # Log Summary
    total_size_readable = get_file_size(total_size_bytes) # Use helper for size
    logger.info("\n--- Summary ---")
    logger.info(f"Total Parquet Files Analyzed: {len(parquet_files)}")
    logger.info(f"Total Size: {total_size_readable}")
    logger.info(f"Estimated Total Rows: {total_rows:,}") # Add comma formatting
    logger.info(f"Number of Distinct Schemas Found: {len(schemas)}")

    logger.info("\n--- Schema Details & Samples ---")
    for i, (schema_key, file_list) in enumerate(schemas.items()):
        logger.info(f"\nSchema #{i+1} (Found in {len(file_list)} files):")
        logger.info(f"  Columns: {list(schema_key)}")
        # Add a check based on common columns for target-disease associations
        expected_cols = {'targetId', 'diseaseId', 'score', 'datasourceId', 'type'}
        if not expected_cols.issubset(set(schema_key)):
             logger.warning(f"  Schema does not contain all expected target-disease columns ({expected_cols - set(schema_key)} missing).")

    # Suggest garbage collection
    logger.info("Suggesting garbage collection after OpenTargets analysis.")
    gc.collect()


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Perform EDA on RAW biomedical data source files.")
    parser.add_argument("--drugbank_xml", help="Path to the raw DrugBank XML file (full_database.xml)")
    parser.add_argument("--mesh_dir", help="Path to the directory containing raw MeSH XML files (expecting desc2025.xml, qual2025.xml)")
    parser.add_argument("--opentargets_dir", help="Path to the directory containing raw OpenTargets Parquet files")
    parser.add_argument("--log_file", help="Path to log file (optional)")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.INFO
    setup_logging(args.log_file, level=log_level)
    logger = logging.getLogger("main_raw_eda")

    # Add handler to root logger to catch warnings from dependencies if not already configured
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    logger.info("Starting Raw Data EDA...")

    # Analyze DrugBank
    if args.drugbank_xml:
        analyze_raw_drugbank(args.drugbank_xml)
    else:
        logger.info("Skipping Raw DrugBank XML analysis (no path provided).")

    # Analyze MeSH (2025 files)
    if args.mesh_dir:
        analyze_raw_mesh(args.mesh_dir)
    else:
        logger.info("Skipping Raw MeSH XML analysis (no directory provided).")

    # Analyze OpenTargets
    if args.opentargets_dir:
        analyze_raw_opentargets(args.opentargets_dir)
    else:
        logger.info("Skipping Raw OpenTargets Parquet analysis (no directory provided).")

    logger.info("\nRaw Data EDA complete.")
    logger.info("Compare these results with the output of analyze_processed_data.py to identify parsing discrepancies.")

if __name__ == "__main__":
    main()
