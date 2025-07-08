# /Users/vi/Documents/drug_disease_interaction/src/scripts/analyze_processed_data.py
import argparse
import logging
import os
import sys
import pickle # Ensure pickle is imported
import json
from pathlib import Path
from collections import Counter # Keep defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Optional, Union

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi.utils.logging import setup_logging

# --- Helper Functions ---

# --- ADDED load_pickle function ---
def load_pickle(file_path: str) -> Any:
    """Load data from a pickle file with error handling."""
    logger = logging.getLogger("load_pickle") # Use a specific logger name
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling file {file_path}: {e}. File might be corrupted or not a pickle file.", exc_info=True)
        return None
    except EOFError as e:
         logger.error(f"EOFError loading pickle file {file_path}: {e}. File might be empty or truncated.", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading pickle file {file_path}: {e}", exc_info=True)
        return None
# --- END ADDED load_pickle function ---

# --- ADDED print_list_summary function ---
def print_list_summary(data: list, name: str, top_n_items: int = 5, top_n_keys: int = 5):
    """Prints a summary of a list, assuming it often contains dictionaries."""
    logger = logging.getLogger("list_summary") # Use a specific logger name
    logger.info(f"\n--- {name} Summary ---")
    if not isinstance(data, list):
        logger.warning(f"{name} is not a list (type: {type(data)}). Skipping summary.")
        return
    logger.info(f"Total items: {len(data)}")

    if len(data) > 0:
        # Sample first few items
        logger.info(f"Sample first {min(top_n_items, len(data))} items:")
        for i, item in enumerate(data[:top_n_items]):
            if isinstance(item, dict):
                # If item is a dict, show some keys/values
                keys_sample = list(item.keys())[:top_n_keys]
                values_sample = {k: item[k] for k in keys_sample}
                # Truncate long values in sample
                values_str = ", ".join([f"{k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" for k, v in values_sample.items()])
                logger.info(f"  Item {i} (dict): {{ {values_str}{'...' if len(item) > top_n_keys else ''} }}")
                # Log keys only once for the first dict item
                if i == 0:
                     logger.info(f"    Keys in first item: {list(item.keys())}")
            else:
                # If item is not a dict, show its type and truncated value
                try:
                    item_str = str(item)
                except Exception:
                    item_str = "[Unstringable Item]"
                logger.info(f"  Item {i} ({type(item).__name__}): {item_str[:100]}{'...' if len(item_str) > 100 else ''}")
    else:
        logger.info("List is empty.")
# --- END ADDED print_list_summary function ---

# --- ADDED print_dict_summary function (already present but ensure it's here) ---
def print_dict_summary(data: dict, name: str, top_n: int = 5):
    """Prints a summary of a dictionary."""
    logger = logging.getLogger("dict_summary")
    logger.info(f"\n--- {name} Summary ---")
    if not isinstance(data, dict):
        logger.warning(f"{name} is not a dictionary (type: {type(data)}). Skipping summary.")
        return
    logger.info(f"Total keys: {len(data)}")
    if len(data) > 0:
        logger.info("Sample keys:")
        keys_list = list(data.keys())
        for i, key in enumerate(keys_list[:top_n]):
            logger.info(f"  - {key}")
        # Check value types
        value_types = Counter(type(v).__name__ for v in data.values())
        logger.info(f"Value types: {dict(value_types)}")
        # Sample value structure (if dict)
        try:
            first_value = data[keys_list[0]] # Use the first key we already got
            if isinstance(first_value, dict):
                logger.info(f"Structure of first value (dict): {list(first_value.keys())}")
            elif isinstance(first_value, list) and first_value:
                 first_list_item = first_value[0]
                 if isinstance(first_list_item, dict):
                     logger.info(f"Structure of first item in first value (list of dicts): {list(first_list_item.keys())}")
                 else:
                      logger.info(f"Structure of first item in first value (list): Type={type(first_list_item).__name__}")
            elif isinstance(first_value, list) and not first_value:
                 logger.info("First value is an empty list.")
            else:
                 logger.info(f"Type of first value: {type(first_value).__name__}")

        except Exception as e:
             logger.warning(f"Could not determine structure of first value: {e}")
    else:
         logger.info("Dictionary is empty.")
# --- END print_dict_summary function ---


def plot_histogram(data: list, title: str, xlabel: str, output_path: Optional[str] = None):
    """Plots a histogram for numerical data in a list."""
    logger = logging.getLogger("plot_histogram")
    numeric_data = [x for x in data if isinstance(x, (int, float)) and not np.isnan(x)]
    if not numeric_data:
        logger.warning(f"No valid numeric data provided for histogram: {title}")
        return
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(numeric_data, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved histogram to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting histogram '{title}': {e}")

def plot_bar_chart(data: dict, title: str, xlabel: str, ylabel: str, output_path: Optional[str] = None, top_n: int = 20):
    """Plots a bar chart for categorical counts."""
    logger = logging.getLogger("plot_bar_chart")
    if not data:
        logger.warning(f"No data provided for bar chart: {title}")
        return
    try:
        string_keyed_data = {str(k): v for k, v in data.items()}
        sorted_data = dict(sorted(string_keyed_data.items(), key=lambda item: item[1], reverse=True)[:top_n])
        keys = list(sorted_data.keys())
        values = list(sorted_data.values())

        plt.figure(figsize=(12, max(6, len(keys) * 0.3)))
        # Updated sns.barplot call (assuming you applied the fix for FutureWarning)
        sns.barplot(x=values, y=keys, hue=keys, palette="viridis", orient='h', legend=False)
        plt.title(f"{title} (Top {top_n})")
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved bar chart to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting bar chart '{title}': {e}")

# --- Analysis Functions ---
# ... (analyze_drugbank, analyze_mesh, analyze_opentargets, analyze_overlaps remain the same as the 'lean' version) ...
def analyze_drugbank(data: dict, output_dir: Optional[str] = None):
    """Performs EDA on the processed DrugBank data."""
    logger = logging.getLogger("analyze_drugbank")
    logger.info("\n" + "="*30 + " Analyzing Processed DrugBank Data " + "="*30)
    if not data or 'drugs' not in data:
        logger.warning("DrugBank data is empty or missing 'drugs' key.")
        return {}

    drugs = data.get('drugs', [])
    logger.info(f"DrugBank Version found in processed data: {data.get('version', 'Unknown')}")
    logger.info(f"Total drugs parsed: {len(drugs)}")
    if not drugs:
        return {}

    print_list_summary(drugs, "DrugBank Drugs Sample") # Uses the added function

    # --- Property Analysis ---
    prop_counts = Counter()
    list_prop_counts = Counter()
    list_prop_lengths = Counter()
    drug_groups = Counter()
    drug_types = Counter()
    has_target = 0
    has_enzyme = 0
    has_transporter = 0
    has_carrier = 0
    has_interaction = 0
    has_category = 0
    has_ext_id = 0
    target_ids = set()
    enzyme_ids = set()
    category_mesh_ids = set()
    ext_id_resources = Counter()

    for drug in drugs:
        if not isinstance(drug, dict):
            logger.warning(f"Skipping non-dict item in drugs list: {type(drug)}")
            continue
        drug_types[drug.get('type', 'unknown')] += 1
        groups_val = drug.get('groups')
        if isinstance(groups_val, list):
            for group in groups_val:
                drug_groups[group] += 1
        elif isinstance(groups_val, str):
             drug_groups[groups_val] += 1

        for key, value in drug.items():
            if value is not None and value != []:
                prop_counts[key] += 1
                if isinstance(value, list) and value:
                    list_prop_counts[key] += 1
                    list_prop_lengths[key] += len(value)

        targets_val = drug.get('targets')
        if targets_val and isinstance(targets_val, list):
            has_target += 1
            for t in targets_val:
                if isinstance(t, dict): target_ids.add(t.get('id'))
        enzymes_val = drug.get('enzymes')
        if enzymes_val and isinstance(enzymes_val, list):
            has_enzyme += 1
            for e in enzymes_val:
                if isinstance(e, dict): enzyme_ids.add(e.get('id'))
        if drug.get('transporters'): has_transporter += 1
        if drug.get('carriers'): has_carrier += 1
        if drug.get('drug_interactions'): has_interaction += 1
        categories_val = drug.get('categories')
        if categories_val and isinstance(categories_val, list):
            has_category += 1
            for cat in categories_val:
                if isinstance(cat, dict): category_mesh_ids.add(cat.get('mesh_id'))
        ext_ids_val = drug.get('external_identifiers')
        if ext_ids_val and isinstance(ext_ids_val, list):
            has_ext_id += 1
            for ext_id in ext_ids_val:
                if isinstance(ext_id, dict):
                    ext_id_resources[ext_id.get('resource')] += 1

    # --- Logging Summaries ---
    logger.info("\n--- Drug Property Coverage ---")
    for key, count in prop_counts.most_common():
        logger.info(f"  - {key}: {count} ({count/len(drugs)*100:.1f}%)")
    logger.info("\n--- List Property Statistics ---")
    for key, count in list_prop_counts.items():
        avg_len = list_prop_lengths[key] / count if count > 0 else 0
        logger.info(f"  - {key}: Present in {count} drugs, Avg length: {avg_len:.2f}")
    logger.info("\n--- Drug Types & Groups ---")
    logger.info(f"Drug Types: {dict(drug_types)}")
    logger.info(f"Drug Groups: {dict(drug_groups)}")
    logger.info("\n--- Drug Relationships ---")
    logger.info(f"Drugs with Targets: {has_target} ({has_target/len(drugs)*100:.1f}%) - Unique Target IDs: {len(target_ids - {None})}")
    logger.info(f"Drugs with Enzymes: {has_enzyme} ({has_enzyme/len(drugs)*100:.1f}%) - Unique Enzyme IDs: {len(enzyme_ids - {None})}")
    logger.info(f"Drugs with Transporters: {has_transporter} ({has_transporter/len(drugs)*100:.1f}%)")
    logger.info(f"Drugs with Carriers: {has_carrier} ({has_carrier/len(drugs)*100:.1f}%)")
    logger.info(f"Drugs with Interactions: {has_interaction} ({has_interaction/len(drugs)*100:.1f}%)")
    logger.info(f"Drugs with Categories: {has_category} ({has_category/len(drugs)*100:.1f}%) - Unique Category MeSH IDs: {len(category_mesh_ids - {None})}")
    logger.info(f"Drugs with External IDs: {has_ext_id} ({has_ext_id/len(drugs)*100:.1f}%)")
    logger.info(f"External ID Resources: {dict(ext_id_resources.most_common(10))}")

    # --- Visualization ---
    if output_dir:
        if drug_groups:
            plot_bar_chart(drug_groups, "DrugBank Drug Groups", "Group", "Count",
                           os.path.join(output_dir, "drugbank_groups_dist.png"))
        if ext_id_resources:
            plot_bar_chart(ext_id_resources, "DrugBank External ID Resources", "Resource", "Count",
                           os.path.join(output_dir, "drugbank_ext_ids_dist.png"))

    # --- Return Value ---
    return {
        "total_drugs": len(drugs),
        "drug_ids": {d.get("drugbank_id") for d in drugs if isinstance(d, dict) and d.get("drugbank_id")},
        "target_ids": target_ids - {None},
        "enzyme_ids": enzyme_ids - {None},
        "category_mesh_ids": category_mesh_ids - {None},
        "external_id_resources": set(ext_id_resources.keys()) - {None},
        "drugs_with_targets": has_target,
        "drugs_with_enzymes": has_enzyme,
        "drugs_with_categories": has_category,
    }

def analyze_mesh(data: dict, output_dir: Optional[str] = None):
    """Performs EDA on the processed MeSH data (expected to be from 2025 files)."""
    logger = logging.getLogger("analyze_mesh")
    logger.info("\n" + "="*30 + " Analyzing Processed MeSH Data (Expected: 2025) " + "="*30)
    if not data:
        logger.warning("MeSH data is empty.")
        return {}

    descriptors = data.get('descriptors', {})
    qualifiers = data.get('qualifiers', {})
    hierarchy = data.get('disease_hierarchy', {})
    term_to_id = data.get('term_to_id', {})
    processed_version = data.get('version', 'Unknown')

    logger.info(f"MeSH Version found in processed data: {processed_version}")
    if processed_version != 'Unknown' and '2025' not in str(processed_version):
        logger.warning(f"Processed MeSH data version '{processed_version}' does not explicitly contain '2025'. Analysis assumes it's based on 2025 files.")
    elif processed_version == 'Unknown':
        logger.warning("Could not determine MeSH version from processed data. Analysis assumes it's based on 2025 files.")

    print_dict_summary(descriptors, "MeSH Descriptors") # Uses the added function
    print_dict_summary(qualifiers, "MeSH Qualifiers") # Uses the added function
    print_dict_summary(hierarchy, "MeSH Disease Hierarchy (Processed)") # Uses the added function
    logger.info(f"Total terms mapped to IDs: {len(term_to_id)}")

    if not descriptors:
        logger.warning("No descriptors found in processed MeSH data.")
        return {}

    disease_descriptors = {k: v for k, v in descriptors.items() if isinstance(v, dict) and v.get('is_disease')}
    logger.info(f"\nTotal 'disease' descriptors found in processed data: {len(disease_descriptors)}")

    if not disease_descriptors:
         logger.warning("No descriptors marked as 'is_disease' found in the processed data.")
         return {"total_descriptors": len(descriptors)}

    desc_prop_counts = Counter()
    tree_num_counts = Counter()
    tree_depths = []
    top_level_categories = Counter()
    synonym_counts = []

    for desc_id, desc in disease_descriptors.items():
        if not isinstance(desc, dict):
            logger.warning(f"Skipping non-dict disease descriptor: {desc_id}")
            continue
        synonyms = desc.get('synonyms', [])
        if isinstance(synonyms, list):
            synonym_counts.append(len(synonyms))

        for key, value in desc.items():
            if value is not None and value != []:
                desc_prop_counts[key] += 1

        tree_numbers = desc.get('tree_numbers', [])
        if isinstance(tree_numbers, list) and tree_numbers:
            tree_num_counts[desc_id] = len(tree_numbers)
            for tn in tree_numbers:
                if isinstance(tn, str):
                    try:
                        depth = tn.count('.')
                        tree_depths.append(depth)
                        top_level_categories[tn.split('.')[0]] += 1
                    except AttributeError:
                        logger.warning(f"Skipping invalid tree number format for analysis: {tn} in descriptor {desc_id}")
                else:
                     logger.warning(f"Skipping non-string tree number: {tn} in descriptor {desc_id}")

    # --- Logging Summaries ---
    logger.info("\n--- Disease Descriptor Property Coverage (based on processed 2025 data) ---")
    for key, count in desc_prop_counts.most_common():
        logger.info(f"  - {key}: {count} ({count/len(disease_descriptors)*100:.1f}%)")
    if tree_num_counts:
        logger.info("\n--- Tree Number Statistics (Diseases, based on processed 2025 data) ---")
        avg_tree_nums = np.mean(list(tree_num_counts.values())) if tree_num_counts else 0
        logger.info(f"Avg Tree Numbers per Descriptor: {avg_tree_nums:.2f}")
        if tree_depths:
            logger.info(f"Avg Tree Depth: {np.mean(tree_depths):.2f} (Min: {min(tree_depths)}, Max: {max(tree_depths)})")
            logger.info(f"Top-Level Category Distribution: {dict(top_level_categories.most_common())}")
        else:
            logger.warning("No valid tree depths found for statistics.")
    else:
        logger.warning("No tree number counts found for disease descriptors.")
    if synonym_counts:
        logger.info("\n--- Synonym Statistics (Diseases, based on processed 2025 data) ---")
        avg_synonyms = np.mean(synonym_counts) if synonym_counts else 0
        logger.info(f"Avg Synonyms per Descriptor: {avg_synonyms:.2f}")
    else:
        logger.warning("No synonym counts found for disease descriptors.")

    # --- Hierarchy Analysis ---
    if hierarchy:
        hierarchy_depths = []
        children_counts = []
        valid_hierarchy_nodes = 0
        for node_tn, node_data in hierarchy.items():
             if isinstance(node_tn, str) and isinstance(node_data, dict):
                 valid_hierarchy_nodes += 1
                 hierarchy_depths.append(node_tn.count('.'))
                 children = node_data.get('children', [])
                 if isinstance(children, list):
                     children_counts.append(len(children))
                 else:
                      children_counts.append(0)
             else:
                 logger.warning(f"Skipping invalid hierarchy node format for analysis: Key={node_tn}, ValueType={type(node_data)}")

        if valid_hierarchy_nodes > 0:
            logger.info("\n--- Disease Hierarchy Statistics (based on processed 2025 data) ---")
            logger.info(f"Total valid nodes in hierarchy: {valid_hierarchy_nodes}")
            if hierarchy_depths:
                 avg_hier_depth = np.mean(hierarchy_depths) if hierarchy_depths else 0
                 logger.info(f"Avg Hierarchy Depth: {avg_hier_depth:.2f}")
            if children_counts:
                 avg_children = np.mean(children_counts) if children_counts else 0
                 logger.info(f"Avg Children per Node: {avg_children:.2f}")
        else:
            logger.warning("No valid nodes found in the processed disease hierarchy.")
    else:
        logger.info("\n--- Disease Hierarchy Statistics ---")
        logger.info("No processed disease hierarchy data found.")

    # --- Visualization ---
    if output_dir and disease_descriptors:
        if tree_depths:
            plot_histogram(tree_depths, "MeSH Disease Tree Depths (Processed 2025 Data)", "Depth",
                           os.path.join(output_dir, "mesh_disease_depth_dist_2025.png"))
        if top_level_categories:
            plot_bar_chart(top_level_categories, "MeSH Disease Top-Level Categories (Processed 2025 Data)", "Category", "Count",
                           os.path.join(output_dir, "mesh_disease_categories_dist_2025.png"))

    # --- Return Value ---
    return {
        "total_descriptors": len(descriptors),
        "disease_descriptor_ids": set(disease_descriptors.keys()),
        "total_qualifiers": len(qualifiers),
        "hierarchy_nodes": len(hierarchy),
        "top_level_categories": set(top_level_categories.keys()) - {None}
    }

def analyze_opentargets(td_associations: list, output_dir: Optional[str] = None):
    """Performs EDA on the processed OpenTargets target-disease associations."""
    logger = logging.getLogger("analyze_opentargets")
    logger.info("\n" + "="*30 + " Analyzing Processed OpenTargets Target-Disease Data " + "="*30)

    if not td_associations:
        logger.warning("OpenTargets target-disease association data is missing or empty.")
        return {}

    if not isinstance(td_associations, list):
        logger.error(f"Expected a list for OpenTargets data, but got {type(td_associations)}. Cannot analyze.")
        return {}

    logger.info(f"Total target-disease associations: {len(td_associations)}")
    print_list_summary(td_associations, "Target-Disease Associations Sample") # Uses the added function

    # --- Association Analysis (Only Target-Disease) ---
    td_scores = [a['score'] for a in td_associations if isinstance(a, dict) and 'score' in a and isinstance(a['score'], (int, float)) and not np.isnan(a['score'])]

    logger.info("\n--- Association Score Statistics ---")
    if td_scores:
        logger.info(f"Target-Disease Scores: Count={len(td_scores)}, Mean={np.mean(td_scores):.3f}, Median={np.median(td_scores):.3f}, Min={min(td_scores):.3f}, Max={max(td_scores):.3f}")
    else:
        logger.info("No valid scores found in target-disease associations.")

    td_datasources = Counter(a['datasource'] for a in td_associations if isinstance(a, dict) and 'datasource' in a and a['datasource'])
    logger.info("\n--- Target-Disease Datasources ---")
    logger.info(f"Datasources: {dict(td_datasources.most_common(10))}")

    # --- Visualization ---
    if output_dir:
        if td_scores:
            plot_histogram(td_scores, "OpenTargets Target-Disease Scores", "Score", os.path.join(output_dir, "ot_td_scores_dist.png"))
        if td_datasources:
             plot_bar_chart(td_datasources, "OpenTargets Target-Disease Datasources", "Datasource", "Count", os.path.join(output_dir, "ot_td_datasources_dist.png"))

    # --- Return Value ---
    return {
        "total_td_assocs": len(td_associations),
        "td_associations": td_associations,
    }

def analyze_overlaps(db_analysis: dict, mesh_analysis: dict, ot_analysis: dict):
    """Analyzes overlaps between the datasets based on IDs (adapted for OT TD associations only)."""
    logger = logging.getLogger("analyze_overlaps")
    logger.info("\n" + "="*30 + " Analyzing Dataset Overlaps " + "="*30)

    if not db_analysis:
        logger.warning("DrugBank analysis results missing. Some overlaps cannot be calculated.")
    if not mesh_analysis:
        logger.warning("MeSH analysis results missing. Some overlaps cannot be calculated.")
    if not ot_analysis or 'td_associations' not in ot_analysis:
        logger.warning("OpenTargets target-disease association analysis results missing. Skipping OT overlaps.")
        ot_analysis = {}

    # Safely get ID sets
    db_drug_ids = db_analysis.get("drug_ids", set())
    db_protein_ids = db_analysis.get("target_ids", set()).union(db_analysis.get("enzyme_ids", set()))
    db_category_mesh_ids = db_analysis.get("category_mesh_ids", set())
    mesh_disease_ids = mesh_analysis.get("disease_descriptor_ids", set())

    # Extract Target/Disease IDs from OpenTargets Associations
    ot_td_associations = ot_analysis.get('td_associations', [])
    ot_target_ids_from_assoc = {a['target_id'] for a in ot_td_associations if isinstance(a, dict) and 'target_id' in a and a['target_id']}
    ot_disease_ids_from_assoc = {a['disease_id'] for a in ot_td_associations if isinstance(a, dict) and 'disease_id' in a and a['disease_id']}

    # --- Drug Overlap ---
    logger.info(f"DrugBank Drug IDs: {len(db_drug_ids)}")
    logger.info("-> OpenTargets drug data not processed in this lean version.")

    # --- Disease Overlap ---
    logger.info(f"\nMeSH Disease Descriptor IDs (2025): {len(mesh_disease_ids)}")
    logger.info(f"OpenTargets Disease IDs (from TD Assocs): {len(ot_disease_ids_from_assoc)}")
    direct_disease_overlap = mesh_disease_ids.intersection(ot_disease_ids_from_assoc)
    logger.info(f"Direct Disease ID Overlap (MeSH == EFO/MONDO): {len(direct_disease_overlap)}")
    if db_category_mesh_ids and mesh_disease_ids:
        mesh_cat_overlap = db_category_mesh_ids.intersection(mesh_disease_ids)
        logger.info(f"Overlap between DrugBank Category MeSH IDs and MeSH Disease IDs (2025): {len(mesh_cat_overlap)}")
    logger.info("-> Note: Full Disease ID overlap requires mapping (MeSH <-> EFO/MONDO).")

    # --- Target/Protein Overlap ---
    logger.info(f"\nDrugBank Protein IDs (Targets/Enzymes): {len(db_protein_ids)}")
    logger.info(f"OpenTargets Target IDs (from TD Assocs): {len(ot_target_ids_from_assoc)}")
    direct_target_overlap = db_protein_ids.intersection(ot_target_ids_from_assoc)
    logger.info(f"Direct Target/Protein ID Overlap: {len(direct_target_overlap)}")
    logger.info("-> Note: Full Target/Protein overlap relies on consistent identifiers (e.g., UniProt) across sources.")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Perform EDA on processed biomedical data sources.")
    parser.add_argument("--drugbank", required=True, help="Path to processed DrugBank data (pickle)")
    parser.add_argument("--mesh", required=True, help="Path to processed MeSH data (pickle - expected 2025)")
    parser.add_argument("--opentargets_td_assoc", required=True, help="Path to processed OpenTargets target-disease associations (pickle)")
    parser.add_argument("--output_dir", help="Directory to save plots and analysis results (optional)")
    parser.add_argument("--log_file", help="Path to log file (optional)")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_file, level=logging.INFO)
    logger = logging.getLogger("main_eda")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {args.output_dir}")

    # Load data (uses the added load_pickle function)
    logger.info("Loading datasets...")
    drugbank_data = load_pickle(args.drugbank)
    mesh_data = load_pickle(args.mesh)
    ot_td_associations = load_pickle(args.opentargets_td_assoc)

    # Perform analysis
    db_results = analyze_drugbank(drugbank_data, args.output_dir)
    mesh_results = analyze_mesh(mesh_data, args.output_dir)
    ot_results = analyze_opentargets(ot_td_associations, args.output_dir)

    # Analyze overlaps
    analyze_overlaps(db_results, mesh_results, ot_results)

    logger.info("\nEDA complete.")
    # Optionally save summary results
    if args.output_dir:
        summary = {
            "drugbank": db_results,
            "mesh": mesh_results,
            "opentargets": ot_results
        }
        summary_path = os.path.join(args.output_dir, "eda_summary.json")
        try:
            # Keep the robust JSON serialization helper
            def convert_to_serializable(obj: Any) -> Union[dict, list, str, int, float, bool, None]:
                if isinstance(obj, set):
                    return sorted([convert_to_serializable(item) for item in obj if item is not None])
                if isinstance(obj, dict):
                    return {str(k): convert_to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                if isinstance(obj, (int, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    if isinstance(obj, bool): return bool(obj)
                    return int(obj)
                elif isinstance(obj, (float, np.float16, np.float32, np.float64)):
                    if np.isnan(obj): return None
                    if np.isinf(obj): return None
                    return float(obj)
                elif isinstance(obj, (bool, np.bool)):
                     return bool(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return convert_to_serializable(obj.tolist())
                elif isinstance(obj, (np.void)):
                    return None
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                logger.warning(f"Converting unrecognized type {type(obj)} to string for JSON serialization.")
                try:
                    return str(obj)
                except Exception:
                    return f"[Unserializable Type: {type(obj).__name__}]"

            serializable_summary = convert_to_serializable(summary)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved EDA summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save EDA summary: {e}", exc_info=True)

if __name__ == "__main__":
    main()
