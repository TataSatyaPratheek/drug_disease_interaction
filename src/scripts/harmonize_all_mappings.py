import pandas as pd
from biothings_client import client as biothings_client
import os

import json
import pyarrow.parquet as pq
from collections import defaultdict
from glob import glob
from typing import Dict, Set
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import time
import logging
import concurrent.futures
import subprocess
import shlex

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLS_OXO_MAP_CSV = "/Users/vi/Documents/not_work/drug_disease_interaction/data/raw/oxo/oxo-mappings-2020-02-04/ols_mappings.csv"
OLS_OXO_MAP_JSON = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings/ols_mapping.json"

BIOMAPPINGS_SSSOM = "/Users/vi/Documents/not_work/drug_disease_interaction/data/raw/biomappings/positive.sssom.tsv"

DATA_DIR = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/open_targets_merged"
logger.debug(f"Data directory set to: {DATA_DIR}")

SCHEMA_PATH = "/Users/vi/Documents/not_work/drug_disease_interaction/open_targets_schema.json"
DISEASE_REF = os.path.join(DATA_DIR, "disease.parquet")
TARGET_REF = os.path.join(DATA_DIR, "target.parquet")
CHUNK_SIZE = 200_000

console = Console()
logger.debug("Console initialized for rich output")

def get_association_files_from_schema(schema_path: str, data_dir: str) -> Dict[str, str]:
    with open(schema_path) as f:
        schema = json.load(f)
    association_types = []
    for k, v in schema.items():
        fields = v if isinstance(v, dict) else {}
        if any(xx in fields for xx in ("diseaseId", "targetId", "drugId")):
            if k not in {"disease", "target"}:
                association_types.append(k)
    found = {}
    for typ in association_types:
        cand = os.path.join(data_dir, f"{typ}.parquet")
        if os.path.exists(cand): 
            found[typ] = cand
    logger.info(f"Found {len(found)} association tables from schema")
    console.print(f"[yellow]Association tables: {list(found.keys())}[/yellow]")
    return found

def read_ids_from_table_streaming(filepath: str, id_col: str, progress=None, schema_name=None) -> Set[str]:
    """Streaming approach to read IDs from parquet files with better memory efficiency"""
    ids = set()
    try:
        # Get total rows for progress tracking
        nrows = pq.read_metadata(filepath).num_rows
        
        # Use smaller chunk size for better memory management
        stream_chunk_size = min(50000, CHUNK_SIZE)  # Smaller chunks for streaming
        num_steps = (nrows // stream_chunk_size) + 1
        
        with Progress(
            SpinnerColumn(), TextColumn(f"[progress.description]{schema_name}"), 
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
        ) as inner_progress:
            t = inner_progress.add_task(f"[teal]Stream {id_col}", total=num_steps)
            
            processed_rows = 0
            # Stream through parquet file in smaller chunks
            for batch in pd.read_parquet(filepath, columns=[id_col], chunksize=stream_chunk_size):
                # Process batch immediately and clear memory
                unique_batch_ids = batch[id_col].dropna().unique()
                ids.update(unique_batch_ids)
                
                # Clear batch from memory
                del batch
                del unique_batch_ids
                
                processed_rows += stream_chunk_size
                inner_progress.update(t, advance=1)
                
                # Progress update every 10 chunks
                if processed_rows % (stream_chunk_size * 10) == 0:
                    logger.debug(f"Streamed {processed_rows:,} rows from {filepath}, found {len(ids):,} unique {id_col}s so far")
            
            inner_progress.update(t, completed=num_steps)
            
    except Exception as e:
        logger.warning(f"Streaming failed for {filepath}:{id_col}, trying single batch: {e}")
        try:
            # Fallback to single batch read
            batch = pd.read_parquet(filepath, columns=[id_col])
            ids.update(batch[id_col].dropna().unique())
            del batch  # Clear memory
        except Exception as e2:
            console.print(f"[red]Couldn't read {id_col} from {filepath}: {e2}[/red]")
    
    logger.debug(f"Streaming complete: extracted {len(ids)} unique IDs from {filepath} for column {id_col}")
    if len(ids) == 0:
        logger.warning(f"No IDs found in {filepath} for {id_col} - possible schema issue")
    return ids

def collect_graph_entity_ids(assoc_files: Dict[str, str]):
    all_disease_ids, all_target_ids, all_drug_ids = set(), set(), set()
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        task = progress.add_task("[blue]Scan all edge tables (outputs inner bars per schema)", total=len(assoc_files))
        for typ, file in assoc_files.items():
            min_bytes = 256
            logger.debug(f"Checking file {file} for size >= {min_bytes}")
            if not os.path.exists(file) or os.path.getsize(file) < min_bytes:
                console.print(f"[yellow]Skipping {typ} ({file}): appears empty or incomplete.[/yellow]")
                progress.advance(task)
                continue
            try:
                cols = pq.read_schema(file).names
            except Exception as e:
                console.print(f"[red]Skipping {typ} ({file}): could not read schema ({e})[/red]")
                progress.advance(task)
                continue
            if "diseaseId" in cols:
                logger.debug(f"Extracting diseaseIds from {typ}")
                all_disease_ids.update(read_ids_from_table_streaming(file, "diseaseId", progress, typ))
            if "targetId" in cols:
                all_target_ids.update(read_ids_from_table_streaming(file, "targetId", progress, typ))
            if "drugId" in cols:
                all_drug_ids.update(read_ids_from_table_streaming(file, "drugId", progress, typ))
            progress.advance(task)
    logger.info(f"Entity statistics: {len(all_disease_ids)} diseases, {len(all_target_ids)} targets, {len(all_drug_ids)} drugs")
    console.print(f"[green]Entity stats: Diseases: {len(all_disease_ids)}, Targets: {len(all_target_ids)}, Drugs: {len(all_drug_ids)}[/green]")
    if len(all_target_ids) == 0:
        logger.warning("No targets found - check schema for 'targetId' column in association tables")
        console.print("[red]Warning: No targets extracted. Schema may need update for targetId columns.[/red]")
    return all_disease_ids, all_target_ids, all_drug_ids

def build_disease_mapping_streaming(all_disease_ids, ref_path):
    """Streaming approach for disease mapping with memory efficiency"""
    console.print(f"[yellow]📊 Loading disease reference table in streaming mode...[/yellow]")
    
    # Load disease reference in chunks for memory efficiency
    crossmaps = defaultdict(dict)
    total_rows = 0
    
    try:
        # Get total rows for progress tracking
        metadata = pq.read_metadata(ref_path)
        total_rows = metadata.num_rows
        console.print(f"[cyan]Disease reference has {total_rows:,} total rows[/cyan]")
        
        # Stream through disease reference in chunks
        stream_chunk_size = 50000  # Smaller chunks for better memory management
        processed_rows = 0
        
        with Progress(
            SpinnerColumn(), TextColumn("[yellow]Streaming disease reference..."), 
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
        ) as progress:
            ref_task = progress.add_task("Loading disease ref", total=total_rows)
            
            for chunk in pd.read_parquet(ref_path, chunksize=stream_chunk_size):
                # Process chunk immediately
                for _, row in chunk.iterrows():
                    row_id = row.get("id")
                    if row_id:
                        crossmaps[row_id] = row
                        
                        # Add dbXRefs mapping
                        db_xrefs = row.get("dbXRefs")
                        if isinstance(db_xrefs, list):
                            for alt in db_xrefs:
                                if alt:  # Skip None/empty values
                                    crossmaps[alt] = row
                
                processed_rows += len(chunk)
                progress.update(ref_task, completed=processed_rows)
                
                # Clear chunk from memory
                del chunk
                
                # Progress logging every 200k rows
                if processed_rows % 200000 == 0:
                    logger.debug(f"Processed {processed_rows:,}/{total_rows:,} disease reference rows, {len(crossmaps):,} mappings built")
            
            progress.update(ref_task, completed=total_rows)
    
    except Exception as e:
        logger.warning(f"Streaming failed for disease reference, trying single load: {e}")
        # Fallback to single load
        with Progress(SpinnerColumn(), TextColumn("[yellow]Loading disease reference (fallback)..."), transient=True, console=console) as p:
            p.add_task("load", total=None)
            df = pd.read_parquet(ref_path)
            
        for _, row in df.iterrows():
            row_id = row.get("id")
            if row_id:
                crossmaps[row_id] = row
                if isinstance(row.get("dbXRefs"), list):
                    for alt in row["dbXRefs"]:
                        if alt:
                            crossmaps[alt] = row
        del df  # Clear from memory

    logger.info(f"Built disease crossmaps with {len(crossmaps):,} total mappings")
    console.print(f"[green]Disease crossmaps built: {len(crossmaps):,} total mappings[/green]")
    
    # Stream through disease ID mapping
    mapped, missing = {}, []
    batch_size = 10000  # Process disease IDs in batches
    disease_id_list = list(all_disease_ids)
    total_diseases = len(disease_id_list)
    
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        map_task = progress.add_task("[green]Stream disease harmonization", total=total_diseases)
        
        for start_idx in range(0, total_diseases, batch_size):
            end_idx = min(start_idx + batch_size, total_diseases)
            batch_diseases = disease_id_list[start_idx:end_idx]
            
            # Process batch
            batch_mapped = 0
            batch_missing = 0
            
            for did in batch_diseases:
                node = crossmaps.get(did)
                mesh = None
                
                if node is not None and isinstance(node.get("dbXRefs", None), list):
                    meshlist = [k for k in node["dbXRefs"] if str(k).startswith("MESH:")]
                    mesh = meshlist[0] if meshlist else None
                elif node is not None and isinstance(node.get("id", None), str) and str(node.get("id", "")).startswith("MESH:"):
                    mesh = node["id"]
                
                if mesh:
                    mapped[did] = mesh
                    batch_mapped += 1
                elif node is not None and "id" in node:
                    mapped[did] = node["id"]
                    batch_mapped += 1
                else:
                    missing.append(did)
                    batch_missing += 1
            
            # Update progress
            progress.update(map_task, completed=end_idx)
            
            # Log progress every few batches
            if (start_idx // batch_size) % 10 == 0:
                current_mapped = len(mapped)
                current_missing = len(missing)
                map_rate = (current_mapped / end_idx) * 100 if end_idx > 0 else 0
                logger.info(f"Disease progress: {end_idx:,}/{total_diseases:,} processed, {current_mapped:,} mapped, {current_missing:,} missing, rate: {map_rate:.2f}%")
        
        progress.update(map_task, completed=total_diseases)
    
    final_map_rate = (len(mapped) / total_diseases) * 100 if total_diseases > 0 else 0
    console.print(f"[bold green]Disease mapping complete: {len(mapped):,}/{total_diseases:,} mapped ({final_map_rate:.2f}%)[/bold green]")
    logger.info(f"Disease mapping final rate: {final_map_rate:.2f}%")
    
    return mapped, missing

def load_uniprot_mapping_json(json_path):
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    logger.info(f"Loaded {len(mapping)} Ensembl to UniProt mappings from JSON")
    return mapping

def process_target_df_streaming(df, uniprot_mapping, gene_name_mapping, protein_mapping, refseq_mapping, chunk_size=10000):
    """Streaming approach for target processing with comprehensive mappings"""
    ens_to_up = {}
    total_rows = len(df)
    processed = 0
    found_mappings = 0
    
    # Mapping source counters for detailed reporting
    source_mapping_counts = {
        'direct_uniprot_mapping': 0,
        'gene_name_mapping': 0, 
        'ensembl_protein_mapping': 0,
        'refseq_mapping': 0,
        'uniprot_swissprot': 0,
        'uniprot_trembl': 0,
        'uniprot_obsolete': 0,
        'ensembl_PRO': 0,
        'PDB': 0,
        'HGNC': 0,
        'InterPro': 0,
        'Reactome': 0,
        'ChEMBL': 0,
        'DrugBank': 0,
        'ProjectScore': 0,
        'signalP': 0
    }
    
    # UniProt source priority mapping (higher priority = lower number)
    uniprot_priority = {
        'uniprot_swissprot': 1, 'uniprot_trembl': 2, 'uniprot_obsolete': 3,
        'ensembl_PRO': 10, 'PDB': 15, 'HGNC': 20, 'InterPro': 25,
        'Reactome': 30, 'ChEMBL': 35, 'DrugBank': 40, 'ProjectScore': 45, 'signalP': 50
    }
    
    console.print(f"[yellow]🔄 Processing {total_rows:,} target entries with comprehensive mappings...[/yellow]")
    
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        t = progress.add_task("[blue]Stream target processing", total=total_rows)
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]
            
            chunk_mapped = 0
            
            for i, row in chunk.iterrows():
                ens = row.get("id")
                uniprot = None
                current_source = None
                
                # Strategy 1: Direct Ensembl gene to UniProt mapping (highest priority)
                if ens and ens in uniprot_mapping:
                    uniprot_candidates = uniprot_mapping[ens]
                    if isinstance(uniprot_candidates, list) and uniprot_candidates:
                        uniprot = uniprot_candidates[0]  # Take first match
                        current_source = 'direct_uniprot_mapping'
                    elif isinstance(uniprot_candidates, str):
                        uniprot = uniprot_candidates
                        current_source = 'direct_uniprot_mapping'
                
                # Strategy 2: Gene name mapping (using comprehensive gene mappings)
                if not uniprot and ens:
                    # Try to extract gene symbol from target data
                    gene_symbol = row.get("approvedSymbol") or row.get("symbol")
                    if gene_symbol and gene_symbol in gene_name_mapping:
                        uniprot_candidates = gene_name_mapping[gene_symbol]
                        if isinstance(uniprot_candidates, list) and uniprot_candidates:
                            uniprot = uniprot_candidates[0]
                            current_source = 'gene_name_mapping'
                        elif isinstance(uniprot_candidates, str):
                            uniprot = uniprot_candidates
                            current_source = 'gene_name_mapping'
                    
                    # Also try synonyms for broader coverage
                    if not uniprot:
                        synonyms = row.get("synonyms", [])
                        if hasattr(synonyms, '__iter__') and len(synonyms) > 0:
                            if hasattr(synonyms, 'tolist'):
                                synonyms = synonyms.tolist()
                            for syn in synonyms:
                                if isinstance(syn, str) and syn in gene_name_mapping:
                                    uniprot_candidates = gene_name_mapping[syn]
                                    if isinstance(uniprot_candidates, list) and uniprot_candidates:
                                        uniprot = uniprot_candidates[0]
                                        current_source = 'gene_name_mapping'
                                        break
                                    elif isinstance(uniprot_candidates, str):
                                        uniprot = uniprot_candidates
                                        current_source = 'gene_name_mapping'
                                        break
                
                # Strategy 3: Ensembl protein mapping (using comprehensive protein mappings)
                if not uniprot and ens:
                    # Try protein ID patterns (ENSP conversion)
                    if ens.startswith("ENSG"):
                        ensp_pattern = ens.replace("ENSG", "ENSP")
                        if ensp_pattern in protein_mapping:
                            uniprot_candidates = protein_mapping[ensp_pattern]
                            if isinstance(uniprot_candidates, list) and uniprot_candidates:
                                uniprot = uniprot_candidates[0]
                                current_source = 'ensembl_protein_mapping'
                            elif isinstance(uniprot_candidates, str):
                                uniprot = uniprot_candidates
                                current_source = 'ensembl_protein_mapping'
                    
                    # Also try without version suffix for broader coverage
                    if not uniprot and "." in ens:
                        base_ens = ens.split(".")[0]
                        ensp_pattern = base_ens.replace("ENSG", "ENSP")
                        if ensp_pattern in protein_mapping:
                            uniprot_candidates = protein_mapping[ensp_pattern]
                            if isinstance(uniprot_candidates, list) and uniprot_candidates:
                                uniprot = uniprot_candidates[0]
                                current_source = 'ensembl_protein_mapping'
                            elif isinstance(uniprot_candidates, str):
                                uniprot = uniprot_candidates
                                current_source = 'ensembl_protein_mapping'
                
                # Strategy 4: RefSeq mapping (from target data using comprehensive RefSeq mappings)
                if not uniprot:
                    # Check for RefSeq in dbXrefs or proteinIds
                    db_xrefs = row.get("dbXrefs", [])
                    if db_xrefs is not None and hasattr(db_xrefs, '__iter__'):
                        # Convert numpy array to list if needed
                        if hasattr(db_xrefs, 'tolist'):
                            db_xrefs = db_xrefs.tolist()
                        
                        # Check if we have any elements
                        if len(db_xrefs) > 0:
                            for xref in db_xrefs:
                                # Check both structured and string formats
                                refseq_id = None
                                if isinstance(xref, dict):
                                    if xref.get("source") == "RefSeq":
                                        refseq_id = xref.get("id")
                                elif isinstance(xref, str):
                                    if (xref.startswith("NP_") or xref.startswith("XP_") or 
                                        xref.startswith("NM_") or xref.startswith("XM_") or
                                        xref.startswith("RefSeq:")):
                                        refseq_id = xref.replace("RefSeq:", "") if xref.startswith("RefSeq:") else xref
                                
                                if refseq_id and refseq_id in refseq_mapping:
                                    uniprot_candidates = refseq_mapping[refseq_id]
                                    if isinstance(uniprot_candidates, list) and uniprot_candidates:
                                        uniprot = uniprot_candidates[0]
                                        current_source = 'refseq_mapping'
                                        break
                                    elif isinstance(uniprot_candidates, str):
                                        uniprot = uniprot_candidates
                                        current_source = 'refseq_mapping'
                                        break
                
                # Strategy 5: Parse proteinIds field for structured data
                if not uniprot:
                    protein_ids_field = row.get("proteinIds")
                    if protein_ids_field is not None and hasattr(protein_ids_field, '__iter__') and len(protein_ids_field) > 0:
                        if hasattr(protein_ids_field, 'tolist'):
                            protein_ids_list = protein_ids_field.tolist()
                        else:
                            protein_ids_list = list(protein_ids_field)
                        
                        best_priority = 999
                        best_uniprot = None
                        best_source = None
                        
                        for item in protein_ids_list:
                            if isinstance(item, dict):
                                source = item.get("source", "")
                                protein_id = item.get("id", "")
                                
                                if source in uniprot_priority and protein_id:
                                    priority = uniprot_priority[source]
                                    if priority < best_priority:
                                        best_priority = priority
                                        best_uniprot = protein_id
                                        best_source = source
                        
                        if best_uniprot:
                            uniprot = best_uniprot
                            current_source = best_source
                
                # Strategy 6: Version-less Ensembl ID mapping (broader coverage)
                if not uniprot and ens and "." in ens:
                    base_ens = ens.split(".")[0]
                    if base_ens in uniprot_mapping:
                        uniprot_candidates = uniprot_mapping[base_ens]
                        if isinstance(uniprot_candidates, list) and uniprot_candidates:
                            uniprot = uniprot_candidates[0]
                            current_source = 'direct_uniprot_mapping'
                        elif isinstance(uniprot_candidates, str):
                            uniprot = uniprot_candidates
                            current_source = 'direct_uniprot_mapping'
                
                # Store mapping result
                if ens and uniprot:
                    ens_to_up[ens] = uniprot
                    chunk_mapped += 1
                    found_mappings += 1
                    
                    # Update source counters
                    if current_source in source_mapping_counts:
                        source_mapping_counts[current_source] += 1
            
            processed += len(chunk)
            progress.update(t, completed=processed)
            
            # Clear chunk from memory
            del chunk
            
            # Progress logging every 50k rows
            if processed % 50000 == 0:
                current_rate = (found_mappings / processed) * 100 if processed > 0 else 0
                logger.debug(f"Target streaming: {processed:,}/{total_rows:,} processed, {found_mappings:,} mapped ({current_rate:.2f}%)")
        
        progress.update(t, completed=total_rows)
    
    # Report mapping statistics
    final_rate = (found_mappings / total_rows) * 100 if total_rows > 0 else 0
    console.print(f"[yellow]Target streaming complete: {found_mappings:,}/{total_rows:,} mapped ({final_rate:.2f}%)[/yellow]")
    
    console.print(f"[bold cyan]Mapping breakdown by source:[/bold cyan]")
    total_source_mapped = 0
    for source, count in sorted(source_mapping_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            color = "green" if "uniprot" in source or "mapping" in source else "white"
            console.print(f"  [{color}]{source}[/{color}]: {count:,} mappings")
            total_source_mapped += count
    
    console.print(f"[bold green]Total unique proteins mapped: {total_source_mapped:,}[/bold green]")
    logger.info(f"Target processing complete: {final_rate:.2f}% mapping rate")
    
    return ens_to_up
    
    # Debug: Check first few rows to understand data structure and collect all sources
    all_sources = set()
    source_counts = {}
    
    if len(df) > 0:
        sample_row = df.iloc[0]
        console.print(f"[cyan]Debug - Sample row columns: {list(sample_row.index)}[/cyan]")
        console.print(f"[cyan]Debug - Sample proteinIds: {sample_row.get('proteinIds', 'NOT FOUND')}[/cyan]")
        console.print(f"[cyan]Debug - Sample dbXrefs: {sample_row.get('dbXrefs', 'NOT FOUND')}[/cyan]")
        
        # Scan first 100000 rows to collect all unique sources
        console.print(f"[yellow]Scanning first 100000 rows to identify all protein ID sources...[/yellow]")
        scan_limit = min(100000, len(df))
        
        for idx in range(scan_limit):
            row = df.iloc[idx]
            
            # Check proteinIds
            protein_ids_field = row.get("proteinIds")
            if protein_ids_field is not None and hasattr(protein_ids_field, '__iter__') and len(protein_ids_field) > 0:
                # Convert numpy array to list if needed
                if hasattr(protein_ids_field, 'tolist'):
                    protein_ids_list = protein_ids_field.tolist()
                else:
                    protein_ids_list = list(protein_ids_field)
                
                for item in protein_ids_list:
                    if isinstance(item, dict):
                        source = item.get("source", "")
                        if source:
                            all_sources.add(source)
                            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Check dbXrefs too
            dbx = row.get("dbXrefs")
            if dbx is not None and hasattr(dbx, '__iter__') and len(dbx) > 0:
                # Convert numpy array to list if needed
                if hasattr(dbx, 'tolist'):
                    dbx_list = dbx.tolist()
                else:
                    dbx_list = list(dbx)
                
                for item in dbx_list:
                    if isinstance(item, dict):
                        source = item.get("source", "")
                        if source:
                            all_sources.add(source)
                            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Display all found sources
        console.print(f"[bold yellow]Found {len(all_sources)} unique protein ID sources in first {scan_limit} rows:[/bold yellow]")
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        for source, count in sorted_sources:
            is_uniprot = "uniprot" in source.lower()
            color = "green" if is_uniprot else "white"
            console.print(f"  [{color}]{source}[/{color}]: {count} occurrences")
        
        # Test parsing on the first row with current logic
        first_protein_ids = sample_row.get("proteinIds")
        if first_protein_ids is not None and hasattr(first_protein_ids, '__iter__') and len(first_protein_ids) > 0:
            # Convert numpy array to list if needed
            if hasattr(first_protein_ids, 'tolist'):
                first_protein_ids_list = first_protein_ids.tolist()
            else:
                first_protein_ids_list = list(first_protein_ids)
            
            console.print(f"[cyan]Debug - First row proteinIds analysis:[/cyan]")
            for i, item in enumerate(first_protein_ids_list[:3]):  # Show first 3
                if isinstance(item, dict):
                    source = item.get("source", "")
                    uid = item.get("id", "")
                    console.print(f"  [{i}] source: '{source}', id: '{uid}', uniprot_match: {source in ['uniprot_swissprot', 'uniprot_trembl']}")
        
        # Identify all protein ID sources and create prioritized mapping
        protein_sources = [source for source in all_sources if source in uniprot_priority]
        console.print(f"[bold cyan]Protein ID sources found (will be mapped): {protein_sources}[/bold cyan]")
        other_sources = [source for source in all_sources if source not in uniprot_priority]
        console.print(f"[bold white]Other sources found (not mapped): {other_sources[:10]}{'...' if len(other_sources) > 10 else ''}[/bold white]")
        
        available_protein_sources = [(src, uniprot_priority.get(src, 999)) for src in protein_sources]
        available_protein_sources.sort(key=lambda x: x[1])  # Sort by priority
        console.print(f"[bold yellow]Protein source priority order: {[src for src, _ in available_protein_sources]}[/bold yellow]")
        
        # Debug: Count how many unique proteins have each mappable protein source type
        debug_counts = {}
        for idx in range(scan_limit):
            row = df.iloc[idx]
            ens_id = row.get("id")
            
            # Check proteinIds
            protein_ids_field = row.get("proteinIds")
            if protein_ids_field is not None and hasattr(protein_ids_field, '__iter__') and len(protein_ids_field) > 0:
                if hasattr(protein_ids_field, 'tolist'):
                    protein_ids_list = protein_ids_field.tolist()
                else:
                    protein_ids_list = list(protein_ids_field)
                
                has_protein_ids = {}
                for item in protein_ids_list:
                    if isinstance(item, dict):
                        source = item.get("source", "")
                        if source in uniprot_priority:
                            has_protein_ids[source] = True
                
                for protein_src in has_protein_ids:
                    if protein_src not in debug_counts:
                        debug_counts[protein_src] = 0
                    debug_counts[protein_src] += 1
            
            # Check dbXrefs too
            dbx = row.get("dbXrefs")
            if dbx is not None and hasattr(dbx, '__iter__') and len(dbx) > 0:
                if hasattr(dbx, 'tolist'):
                    dbx_list = dbx.tolist()
                else:
                    dbx_list = list(dbx)
                
                has_protein_ids = {}
                for item in dbx_list:
                    if isinstance(item, dict):
                        source = item.get("source", "")
                        if source in uniprot_priority:
                            has_protein_ids[source] = True
                
                for protein_src in has_protein_ids:
                    if protein_src not in debug_counts:
                        debug_counts[protein_src] = 0
                    debug_counts[protein_src] += 1
        
        console.print(f"[bold magenta]Debug - Unique proteins with each mappable source:[/bold magenta]")
        for src in sorted(debug_counts.keys(), key=lambda x: uniprot_priority.get(x, 999)):
            count = debug_counts.get(src, 0)
            console.print(f"  {src}: {count} unique proteins")
    
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        t = progress.add_task("[blue]Processing target data in chunks", total=total_rows)
        
        # Add counters for each protein ID source type
        source_mapping_counts = {
            # UniProt sources
            'uniprot_swissprot': 0, 'uniprot_trembl': 0, 'uniprot_obsolete': 0,
            # Other protein ID sources
            'ensembl_PRO': 0, 'PDB': 0, 'HGNC': 0, 'InterPro': 0, 
            'Reactome': 0, 'ChEMBL': 0, 'DrugBank': 0, 'ProjectScore': 0, 'signalP': 0
        }
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]
            
            for i, row in chunk.iterrows():
                ens = row.get("id")
                uniprot = None
                current_source = None  # Track the source of current mapping
                
                # Debug first few rows more extensively
                is_debug_row = processed < 3
                
                # Main check: proteinIds as list of structs
                protein_ids_field = row.get("proteinIds")
                if is_debug_row:
                    pass
                    #console.print(f"[cyan]Debug row {processed}: {ens} has proteinIds type: {type(protein_ids_field)}, length: {len(protein_ids_field) if hasattr(protein_ids_field, '__len__') and protein_ids_field is not None else 'N/A'}[/cyan]")
                
                # Handle both list and numpy array types
                if protein_ids_field is not None and hasattr(protein_ids_field, '__iter__') and len(protein_ids_field) > 0:
                    # Convert numpy array to list if needed
                    if hasattr(protein_ids_field, 'tolist'):
                        protein_ids_list = protein_ids_field.tolist()
                    else:
                        protein_ids_list = list(protein_ids_field)
                    
                    # Debug logging for first few rows
                    if is_debug_row:
                        pass
                        #logger.info(f"Debug row {processed}: {ens} has proteinIds: {protein_ids_list}")
                    
                    # Look for UniProt mappings with priority handling
                    for item in protein_ids_list:
                        if isinstance(item, dict):
                            source = item.get("source", "")
                            uid = item.get("id", "")
                            
                            # Debug for first few
                            if is_debug_row:
                                pass
                                #logger.info(f"  Checking item: source='{source}', id='{uid}'")
                            
                            # Multi-source protein ID mapping with comprehensive priority system
                            if uid:
                                candidate_mapping = None
                                mapped_source = None
                                source_priority = 999
                                
                                # UniProt sources (highest priority)
                                if 'uniprot' in source.lower():
                                    candidate_mapping = f"UNIPROT:{uid}"
                                    mapped_source = source
                                    source_priority = uniprot_priority.get(source, 999)
                                
                                # Ensembl Protein IDs (medium-high priority)
                                elif source == "ensembl_PRO":
                                    candidate_mapping = f"ENSEMBL_PRO:{uid}"
                                    mapped_source = source
                                    source_priority = 10
                                
                                # PDB structures (medium priority - good for structural analysis)
                                elif source == "PDB":
                                    candidate_mapping = f"PDB:{uid}"
                                    mapped_source = source
                                    source_priority = 15
                                
                                # HGNC gene symbols (medium priority - standardized gene names)
                                elif source == "HGNC":
                                    candidate_mapping = f"HGNC:{uid}"
                                    mapped_source = source
                                    source_priority = 20
                                
                                # InterPro protein families (medium-low priority)
                                elif source == "InterPro":
                                    candidate_mapping = f"INTERPRO:{uid}"
                                    mapped_source = source
                                    source_priority = 25
                                
                                # Reactome pathways (useful for pathway analysis)
                                elif source == "Reactome":
                                    candidate_mapping = f"REACTOME:{uid}"
                                    mapped_source = source
                                    source_priority = 30
                                
                                # ChEMBL targets (drug-target interactions)
                                elif source == "ChEMBL":
                                    candidate_mapping = f"CHEMBL:{uid}"
                                    mapped_source = source
                                    source_priority = 35
                                
                                # DrugBank targets
                                elif source == "DrugBank":
                                    candidate_mapping = f"DRUGBANK:{uid}"
                                    mapped_source = source
                                    source_priority = 40
                                
                                # ProjectScore (CRISPR screening data)
                                elif source == "ProjectScore":
                                    candidate_mapping = f"PROJECT_SCORE:{uid}"
                                    mapped_source = source
                                    source_priority = 45
                                
                                # signalP (signal peptide prediction)
                                elif source == "signalP":
                                    candidate_mapping = f"SIGNALP:{uid}"
                                    mapped_source = source
                                    source_priority = 50
                                
                                # If we have a valid mapping candidate
                                if candidate_mapping:
                                    # If we don't have a mapping yet, or this is higher priority
                                    current_priority = uniprot_priority.get(current_source, 999) if current_source else 999
                                    if not uniprot or source_priority < current_priority:
                                        # Only count as new mapping if we didn't have one before
                                        was_new_mapping = not uniprot
                                        if was_new_mapping:
                                            found_mappings += 1
                                        
                                        uniprot = candidate_mapping
                                        current_source = mapped_source
                                        
                                        # Count by source type (expand the counter)
                                        if was_new_mapping:
                                            if mapped_source not in source_mapping_counts:
                                                source_mapping_counts[mapped_source] = 0
                                            source_mapping_counts[mapped_source] += 1
                                        
                                        if found_mappings <= 20 or is_debug_row:  # Show more examples
                                            pass
                                            #logger.info(f"Found protein mapping: {ens} -> {uniprot} (source: {mapped_source}, priority: {source_priority})")
                                        
                                        # If we found highest priority UniProt, break immediately
                                        if mapped_source == "uniprot_swissprot":
                                            break
                        elif isinstance(item, str) and item.startswith("UNIPROT:"):
                            if not uniprot:
                                uniprot = item
                                found_mappings += 1
                                if found_mappings <= 10 or is_debug_row:
                                    logger.info(f"Found UniProt string: {uniprot}")
                
                # NEW: Direct Ensembl to UniProt mapping fallback if no protein IDs found
                if not uniprot and ens:
                    # Try direct lookup in comprehensive mapping
                    direct_uniprot = None
                    if hasattr(process_target_df_streaming, 'uniprot_mapping'):
                        mapping = process_target_df_streaming.uniprot_mapping
                        if ens in mapping:
                            uniprot_ids = mapping[ens]
                            if isinstance(uniprot_ids, list) and uniprot_ids:
                                # Prefer shorter IDs (typically SwissProt)
                                best_id = min(uniprot_ids, key=len)
                                direct_uniprot = f"UNIPROT:{best_id}"
                            elif isinstance(uniprot_ids, str):
                                direct_uniprot = f"UNIPROT:{uniprot_ids}"
                        
                        # Try without version suffix
                        if not direct_uniprot and "." in ens:
                            base_ens = ens.split(".")[0]
                            if base_ens in mapping:
                                uniprot_ids = mapping[base_ens]
                                if isinstance(uniprot_ids, list) and uniprot_ids:
                                    best_id = min(uniprot_ids, key=len)
                                    direct_uniprot = f"UNIPROT:{best_id}"
                                elif isinstance(uniprot_ids, str):
                                    direct_uniprot = f"UNIPROT:{uniprot_ids}"
                    
                    if direct_uniprot:
                        uniprot = direct_uniprot
                        found_mappings += 1
                        if 'direct_uniprot_mapping' not in source_mapping_counts:
                            source_mapping_counts['direct_uniprot_mapping'] = 0
                        source_mapping_counts['direct_uniprot_mapping'] += 1
                        if found_mappings <= 20 or is_debug_row:
                            pass
                            #logger.info(f"Found direct Ensembl->UniProt mapping: {ens} -> {uniprot}")
                
                # NEW: Gene symbol fallback mapping
                if not uniprot and ens:
                    # Try gene symbol mappings
                    gene_symbols = []
                    
                    # Extract from approvedSymbol
                    approved_symbol = row.get("approvedSymbol")
                    if approved_symbol and isinstance(approved_symbol, str):
                        gene_symbols.append(approved_symbol.strip())
                    
                    # Extract from synonyms
                    synonyms = row.get("synonyms")
                    if synonyms and hasattr(synonyms, '__iter__'):
                        if hasattr(synonyms, 'tolist'):
                            syns = synonyms.tolist()
                        else:
                            syns = list(synonyms)
                        for syn in syns:
                            if isinstance(syn, str):
                                gene_symbols.append(syn.strip())
                    
                    # Extract from symbolSynonyms
                    symbol_synonyms = row.get("symbolSynonyms")
                    if symbol_synonyms and hasattr(symbol_synonyms, '__iter__'):
                        if hasattr(symbol_synonyms, 'tolist'):
                            sym_syns = symbol_synonyms.tolist()
                        else:
                            sym_syns = list(symbol_synonyms)
                        for sym_syn in sym_syns:
                            if isinstance(sym_syn, str):
                                gene_symbols.append(sym_syn.strip())
                    
                    # Try to map using gene symbols
                    if gene_symbols and hasattr(process_target_df_streaming, 'gene_name_mapping'):
                        gene_mapping = process_target_df_streaming.gene_name_mapping
                        for symbol in gene_symbols:
                            if symbol in gene_mapping:
                                uniprot_ids = gene_mapping[symbol]
                                if isinstance(uniprot_ids, list) and uniprot_ids:
                                    best_id = min(uniprot_ids, key=len)
                                    uniprot = f"UNIPROT:{best_id}"
                                    found_mappings += 1
                                    if 'gene_symbol_mapping' not in source_mapping_counts:
                                        source_mapping_counts['gene_symbol_mapping'] = 0
                                    source_mapping_counts['gene_symbol_mapping'] += 1
                                    if found_mappings <= 20 or is_debug_row:
                                        pass
                                        #logger.info(f"Found gene symbol mapping: {ens} -> {uniprot} (symbol: {symbol})")
                                    break
                                elif isinstance(uniprot_ids, str):
                                    uniprot = f"UNIPROT:{uniprot_ids}"
                                    found_mappings += 1
                                    if 'gene_symbol_mapping' not in source_mapping_counts:
                                        source_mapping_counts['gene_symbol_mapping'] = 0
                                    source_mapping_counts['gene_symbol_mapping'] += 1
                                    break
                
                if is_debug_row:
                    pass
                    #console.print(f"[cyan]Debug row {processed}: Final uniprot for {ens}: {uniprot} (source: {current_source})[/cyan]")
                
                # Fallback: dbXrefs (structured data like proteinIds)
                dbx = row.get("dbXrefs")
                if uniprot is None and dbx is not None and hasattr(dbx, '__iter__') and len(dbx) > 0:
                    # Convert numpy array to list if needed
                    if hasattr(dbx, 'tolist'):
                        dbx_list = dbx.tolist()
                    else:
                        dbx_list = list(dbx)
                    
                    for item in dbx_list:
                        if isinstance(item, dict):
                            source = item.get("source", "")
                            uid = item.get("id")
                            if uid and not uniprot:
                                candidate_mapping = None
                                source_priority = 999
                                
                                # Apply same comprehensive mapping as above
                                if 'uniprot' in source.lower():
                                    candidate_mapping = f"UNIPROT:{uid}"
                                    source_priority = uniprot_priority.get(source, 999)
                                elif source == "ensembl_PRO":
                                    candidate_mapping = f"ENSEMBL_PRO:{uid}"
                                    source_priority = 10
                                elif source == "PDB":
                                    candidate_mapping = f"PDB:{uid}"
                                    source_priority = 15
                                elif source == "HGNC":
                                    candidate_mapping = f"HGNC:{uid}"
                                    source_priority = 20
                                elif source == "InterPro":
                                    candidate_mapping = f"INTERPRO:{uid}"
                                    source_priority = 25
                                elif source == "Reactome":
                                    candidate_mapping = f"REACTOME:{uid}"
                                    source_priority = 30
                                elif source == "ChEMBL":
                                    candidate_mapping = f"CHEMBL:{uid}"
                                    source_priority = 35
                                elif source == "DrugBank":
                                    candidate_mapping = f"DRUGBANK:{uid}"
                                    source_priority = 40
                                elif source == "ProjectScore":
                                    candidate_mapping = f"PROJECT_SCORE:{uid}"
                                    source_priority = 45
                                elif source == "signalP":
                                    candidate_mapping = f"SIGNALP:{uid}"
                                    source_priority = 50
                                
                                if candidate_mapping:
                                    uniprot = candidate_mapping
                                    found_mappings += 1
                                    if source not in source_mapping_counts:
                                        source_mapping_counts[source] = 0
                                    source_mapping_counts[source] += 1
                                    if found_mappings <= 20:
                                        logger.info(f"Found protein ID in dbXrefs: {ens} -> {uniprot} (source: {source}, priority: {source_priority})")
                        elif isinstance(item, str) and any(item.startswith(prefix) for prefix in ["UNIPROT:", "ENSEMBL_PRO:", "PDB:", "HGNC:", "INTERPRO:", "REACTOME:", "CHEMBL:", "DRUGBANK:"]):
                            if not uniprot:
                                uniprot = item
                                found_mappings += 1
                                if found_mappings <= 20:
                                    logger.info(f"Found protein ID string in dbXrefs: {uniprot}")
                
                if ens:
                    ens_to_up[ens] = uniprot
            
            processed += len(chunk)
            progress.update(t, completed=processed)
            logger.debug(f"Processed {processed} / {total_rows} rows, found {found_mappings} mappings so far")
    
    console.print(f"[yellow]Chunked processing complete: {found_mappings} protein ID mappings found[/yellow]")
    console.print(f"[bold cyan]Mapping breakdown by source:[/bold cyan]")
    total_mapped = 0
    for source, count in sorted(source_mapping_counts.items(), key=lambda x: uniprot_priority.get(x[0], 999)):
        if count > 0:
            console.print(f"  {source}: {count} mappings")
            total_mapped += count
    console.print(f"[bold green]Total unique proteins mapped: {total_mapped}[/bold green]")
    
    return ens_to_up

def build_target_mapping(all_target_ids, ref_path, do_mapper=True, batch_size=2000):
    # Load comprehensive UniProt mapping JSONs
    UNIPROT_MAPPING_JSON = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings/ensembl_to_uniprot_full.json"
    COMPREHENSIVE_MAPPING_JSON = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings/comprehensive_id_mappings.json"
    
    uniprot_mapping = {}
    gene_name_mapping = {}
    protein_mapping = {}
    refseq_mapping = {}
    
    # Load Ensembl to UniProt mapping
    if os.path.exists(UNIPROT_MAPPING_JSON):
        uniprot_mapping = load_uniprot_mapping_json(UNIPROT_MAPPING_JSON)
        logger.info(f"Loaded {len(uniprot_mapping)} Ensembl to UniProt mappings")
    else:
        logger.warning("Primary UniProt mapping JSON not found - mappings will be limited")
    
    # Load comprehensive mapping with memory-efficient streaming for large files
    if os.path.exists(COMPREHENSIVE_MAPPING_JSON):
        console.print(f"[yellow]🔄 Loading comprehensive mappings (streaming for memory efficiency)...[/yellow]")
        
        # Check file size and use streaming for large files (>500MB)
        file_size = os.path.getsize(COMPREHENSIVE_MAPPING_JSON)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > 500:
            console.print(f"[cyan]Large mapping file detected ({file_size_mb:.1f}MB) - using streaming approach[/cyan]")
            
            # Load individual mapping files instead of the large comprehensive one
            gene_mapping_path = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings/gene_name_to_uniprot.json"
            protein_mapping_path = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings/ensembl_protein_to_uniprot.json"
            refseq_mapping_path = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/mappings/refseq_to_uniprot.json"
            
            # Load smaller individual files
            if os.path.exists(gene_mapping_path):
                with open(gene_mapping_path, 'r') as f:
                    gene_name_mapping = json.load(f)
                console.print(f"[green]✓ Gene name mappings: {len(gene_name_mapping):,}[/green]")
            
            if os.path.exists(protein_mapping_path):
                with open(protein_mapping_path, 'r') as f:
                    protein_mapping = json.load(f)
                console.print(f"[green]✓ Protein mappings: {len(protein_mapping):,}[/green]")
            
            if os.path.exists(refseq_mapping_path):
                with open(refseq_mapping_path, 'r') as f:
                    refseq_mapping = json.load(f)
                console.print(f"[green]✓ RefSeq mappings: {len(refseq_mapping):,}[/green]")
                
            logger.info(f"Loaded comprehensive mappings (streaming): {len(gene_name_mapping)} gene names, {len(protein_mapping)} proteins, {len(refseq_mapping)} RefSeq")
        else:
            # For smaller files, load normally
            with open(COMPREHENSIVE_MAPPING_JSON, 'r') as f:
                comprehensive = json.load(f)
                gene_name_mapping = comprehensive.get("gene_name", {})
                protein_mapping = comprehensive.get("ensembl_protein", {})
                refseq_mapping = comprehensive.get("refseq", {})
                logger.info(f"Loaded comprehensive mappings: {len(gene_name_mapping)} gene names, {len(protein_mapping)} proteins, {len(refseq_mapping)} RefSeq")
    else:
        logger.warning("Comprehensive mapping JSON not found - using basic mappings only")

    with Progress(SpinnerColumn(), TextColumn("[yellow]Loading target reference table..."), transient=True, console=console) as p:
        p.add_task("load", total=None)
        df = pd.read_parquet(ref_path)
    logger.debug(f"Loaded target reference with {len(df)} rows")
    console.print(f"[yellow]Target ref loaded: {len(df)} entries[/yellow]")

    # Load and merge target_essentiality
    df_ess = pd.read_parquet(os.path.join(DATA_DIR, "target_essentiality.parquet"))
    logger.debug(f"Loaded target_essentiality with {len(df_ess)} rows")

    # Load target_prioritisation
    df_prior = pd.read_parquet(os.path.join(DATA_DIR, "target_prioritisation.parquet"))
    logger.debug(f"Loaded target_prioritisation with {len(df_prior)} rows")

    # Merge all target data on 'id' or 'targetId'
    df = df.merge(df_ess, left_on='id', right_on='id', how='left', suffixes=('', '_ess'))
    df = df.merge(df_prior, left_on='id', right_on='targetId', how='left', suffixes=('', '_prior'))
    logger.info(f"Merged target data: total rows {len(df)}")
    if len(df) == 0:
        logger.warning("Merged target data is empty - check schema for 'id'/'targetId' consistency")
        console.print("[red]Warning: No merged target data. Schema may need id alignment.[/red]")

    # Process target dataframe with streaming approach and comprehensive mappings
    ens_to_up = process_target_df_streaming(df, uniprot_mapping, gene_name_mapping, protein_mapping, refseq_mapping, chunk_size=10000)

    # Sample output for debugging
    sample = list(ens_to_up.items())[:5]
    missed_sample = [tid for tid in all_target_ids if ens_to_up.get(tid) is None][:5]
    console.print(f"Sample mapping ens_to_up: {sample}")
    console.print(f"Sample missed target IDs: {missed_sample}")

    # Enhanced parsing from additional fields (also chunked for consistency)
    console.print("[cyan]Processing alternative genes and additional fields...")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        t = progress.add_task("[blue]Enhanced parsing", total=len(df))
        processed = 0
        chunk_size = 10000
        
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]
            
            for i, row in chunk.iterrows():
                ens = row.get("id")
                if ens in ens_to_up and ens_to_up[ens] is not None:
                    continue  # Already mapped
                # Parse alternativeGenes (list<string>) for more Ensembl IDs
                alt_genes = row.get("alternativeGenes", [])
                if alt_genes is not None and len(alt_genes) > 0:
                    for alt in alt_genes:
                        alt_up = ens_to_up.get(alt)
                        if alt_up:
                            ens_to_up[ens] = alt_up
                            break
                # Parse from target_prioritisation fields if merged (e.g., if any UniProt-like in prioritisation data)
                # (Add similar for essentiality if it has relevant columns)
            
            processed += len(chunk)
            progress.update(t, completed=processed)

    mapped, missing = {}, []
    for tid in all_target_ids:
        up = ens_to_up.get(tid)
        if up:
            mapped[tid] = up
            logger.debug(f"Mapped {tid} to {up}")
        else:
            missing.append(tid)
    console.print(f"[yellow]First pass: mapped {len(mapped)} targets, {len(missing)} still unmapped after ref table.")
    map_rate = (len(mapped) / len(all_target_ids)) * 100 if all_target_ids else 0
    logger.info(f"First pass mapping rate: {map_rate:.2f}%")
    if map_rate < 50:
        console.print("[yellow]Low first-pass rate - schema may need more UniProt sources (e.g., add dbXRefs parsing).[/yellow]")

    # 2nd pass - use enhanced mapper for unmapped
    if do_mapper and missing:
        console.print("[cyan]Running enhanced UniProt mapping on unmapped Ensembl IDs (batching for memory)")
        batch_mapped = {}

        # Define a comprehensive fallback function for mapping Ensembl to UniProt
        def map_ensembl_to_uniprot_multi_strategy(ensembl_id):
            # Strategy 1: Direct lookup in our comprehensive Ensembl->UniProt mapping
            if uniprot_mapping and ensembl_id in uniprot_mapping:
                uniprot_ids = uniprot_mapping[ensembl_id]
                if isinstance(uniprot_ids, list) and uniprot_ids:
                    # Prefer SwissProt over TrEMBL if multiple options
                    # SwissProt IDs are typically shorter and manually reviewed
                    best_id = min(uniprot_ids, key=len) if uniprot_ids else uniprot_ids[0]
                    return f"UNIPROT:{best_id}"
                elif isinstance(uniprot_ids, str):
                    return f"UNIPROT:{uniprot_ids}"
            
            # Strategy 2: Try removing version suffix (e.g., ENSG00000000003.15 -> ENSG00000000003)
            if "." in ensembl_id:
                base_id = ensembl_id.split(".")[0]
                if uniprot_mapping and base_id in uniprot_mapping:
                    uniprot_ids = uniprot_mapping[base_id]
                    if isinstance(uniprot_ids, list) and uniprot_ids:
                        best_id = min(uniprot_ids, key=len) if uniprot_ids else uniprot_ids[0]
                        return f"UNIPROT:{best_id}"
                    elif isinstance(uniprot_ids, str):
                        return f"UNIPROT:{uniprot_ids}"
            
            # Strategy 3: Try with ENSP prefix conversion for protein IDs
            if ensembl_id.startswith("ENSG"):
                # Some mappings might be under ENSP (protein) instead of ENSG (gene)
                ensp_pattern = ensembl_id.replace("ENSG", "ENSP")
                if protein_mapping and ensp_pattern in protein_mapping:
                    uniprot_ids = protein_mapping[ensp_pattern]
                    if isinstance(uniprot_ids, list) and uniprot_ids:
                        best_id = min(uniprot_ids, key=len) if uniprot_ids else uniprot_ids[0]
                        return f"UNIPROT:{best_id}"
                    elif isinstance(uniprot_ids, str):
                        return f"UNIPROT:{uniprot_ids}"
            
            # Strategy 4: Try transcript ID patterns (ENST)
            if ensembl_id.startswith("ENSG"):
                enst_pattern = ensembl_id.replace("ENSG", "ENST")
                if uniprot_mapping and enst_pattern in uniprot_mapping:
                    uniprot_ids = uniprot_mapping[enst_pattern]
                    if isinstance(uniprot_ids, list) and uniprot_ids:
                        best_id = min(uniprot_ids, key=len) if uniprot_ids else uniprot_ids[0]
                        return f"UNIPROT:{best_id}"
                    elif isinstance(uniprot_ids, str):
                        return f"UNIPROT:{uniprot_ids}"
            
            # Strategy 5: Gene symbol fallback (if target data has gene symbols)
            # This will be useful for targets that have gene names in alternativeGenes or approvedSymbol
            # Note: This strategy will be expanded when we have gene symbol extraction
            
            return None

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
        ) as progress:
            batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
            t = progress.add_task("[blue]Mapping with enhanced strategies", total=len(batches))
            for batch in batches:
                for ensembl_id in batch:
                    uniprot_id = map_ensembl_to_uniprot_multi_strategy(ensembl_id)
                    if uniprot_id:
                        # Handle case where uniprot_id might be a list
                        if isinstance(uniprot_id, list):
                            uniprot_id = uniprot_id[0] if uniprot_id else None
                        if uniprot_id and isinstance(uniprot_id, str):
                            batch_mapped[ensembl_id] = f"UNIPROT:{uniprot_id}" if not uniprot_id.startswith("UNIPROT:") else uniprot_id
                progress.advance(t)
        # Add these to mapped
        for k, v in batch_mapped.items():
            if k not in mapped:
                mapped[k] = v
        still_missing = [x for x in missing if x not in batch_mapped]
        console.print(f"[magenta]After UniProtMapper: mapped {len(batch_mapped)} more; {len(still_missing)} unmapped.")
        final_rate = ((len(mapped) + len(batch_mapped)) / len(all_target_ids)) * 100 if all_target_ids else 0
        logger.info(f"Final mapping rate: {final_rate:.2f}%")
        if final_rate < 70:
            console.print("[yellow]Overall low mapping rate - consider adding TrEMBL fallback or external API for unverified IDs.[/yellow]")
        return mapped, still_missing
    else:
        return mapped, missing

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Schema-driven, low-memory, live-logged harmonization for Open Targets graph.")
    parser.add_argument('--data_dir', default=DATA_DIR, help="Dir with all merged OT Parquet files")
    parser.add_argument('--schema_json', default=SCHEMA_PATH, help="Open Targets schema JSON")
    parser.add_argument('--output_dir', required=True, help="Where to write mappings/logs")
    parser.add_argument('--disease_ref', default=DISEASE_REF)
    parser.add_argument('--target_ref', default=TARGET_REF)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    console.print(f"[bold yellow]Step 1: Locating association tables from schema.")
    assoc_files = get_association_files_from_schema(args.schema_json, args.data_dir)
    console.print(f"[bold yellow]Using {len(assoc_files)} association tables: {list(assoc_files.keys())}")
    logger.info(f"Association tables loaded: {len(assoc_files)}")
    if len(assoc_files) < 5:
        console.print("[yellow]Few association tables found - verify schema JSON for completeness.[/yellow]")

    console.print("[bold yellow]Step 2: Identifying unique graph entities (live progress)")
    all_disease_ids, all_target_ids, all_drug_ids = collect_graph_entity_ids(assoc_files)
    console.print(f"[green]Found {len(all_disease_ids):,} disease nodes, {len(all_target_ids):,} protein/target nodes, {len(all_drug_ids):,} drugs in edges[/green]")

    console.print("[bold yellow]Step 3: Harmonizing diseases with streaming (live progress)")
    disease_map, unmapped_dis = build_disease_mapping_streaming(all_disease_ids, args.disease_ref)
    console.print("[bold yellow]Step 4: Harmonizing proteins/targets (live progress)")
    target_map, unmapped_tar = build_target_mapping(all_target_ids, args.target_ref, do_mapper=True)

    console.print(f"[cyan]Mapped {len(disease_map)} / {len(all_disease_ids)} diseases, {len(unmapped_dis)} unmapped[/cyan]")
    console.print(f"[cyan]Mapped {len(target_map)} / {len(all_target_ids)} targets, {len(unmapped_tar)} unmapped[/cyan]")
    logger.info(f"Mapping complete. Disease unmapped: {len(unmapped_dis)}, Target unmapped: {len(unmapped_tar)}")

    pd.DataFrame.from_dict(disease_map, orient="index").reset_index().to_csv(os.path.join(args.output_dir,"disease_mapping.csv"), index=False)
    pd.DataFrame.from_dict(target_map, orient="index").reset_index().to_csv(os.path.join(args.output_dir,"target_mapping.csv"), index=False)
    pd.Series(unmapped_dis).to_csv(os.path.join(args.output_dir,"unmapped_disease.txt"), index=False, header=None)
    pd.Series(unmapped_tar).to_csv(os.path.join(args.output_dir,"unmapped_target.txt"), index=False, header=None)
    console.print(f"[bold green]All outputs written to {args.output_dir}[/bold green]")

if __name__ == "__main__":
    main()
