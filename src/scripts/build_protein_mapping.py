# src/scripts/build_protein_mapping.py

import os
import pickle
import argparse
import pandas as pd
from UniProtMapper import ProtMapper
from typing import List

# --- Import Rich for a better CLI experience ---
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

# --- Configuration ---
# We will manually control the batch size to update the progress bar
BATCH_SIZE = 2600

# Initialize the rich console
console = Console()

def build_protein_id_mapping(opentargets_path: str, output_dir: str):
    """
    Fetches protein IDs, maps them using UniProtMapper, and displays rich progress.
    """
    console.rule("[bold green]Reliable UniProt ID Mapping with Rich Progress[/bold green]")

    # --- Step 1: Define paths and ensure output directory exists ---
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, 'protein_ensembl_to_uniprot_mapping.csv')
    failed_ids_path = os.path.join(output_dir, 'failed_protein_mappings.txt')

    # --- Step 2: Load OpenTargets data and extract Ensembl IDs ---
    with console.status("[bold cyan]Loading and preparing data...", spinner="dots"):
        try:
            opentargets_data = pd.read_pickle(opentargets_path)
        except FileNotFoundError:
            console.print(f"[bold red]FATAL: Input file not found at {opentargets_path}[/bold red]")
            return
        except Exception as e:
            console.print(f"[bold red]FATAL: Error loading pickle file: {e}[/bold red]")
            return

        target_id_column = 'targetId'
        if target_id_column not in opentargets_data.columns:
            console.print(f"[bold red]FATAL: Expected column '{target_id_column}' not found.[/bold red]")
            return
            
        all_ensembl_ids = opentargets_data[target_id_column].dropna().unique()
        ensembl_id_list = sorted([str(eid) for eid in all_ensembl_ids if str(eid).startswith('ENSG')])
        total_unique_ids = len(ensembl_id_list)
    
    console.print(f"✅ Found [bold yellow]{total_unique_ids:,}[/bold yellow] unique Ensembl Gene IDs to map.")

    if not ensembl_id_list:
        console.print("[yellow]No valid Ensembl Gene IDs found. Exiting.[/yellow]")
        return

    # --- Step 3: Use UniProtMapper in a manual loop for progress updates ---
    console.print("[cyan]Initializing UniProtMapper...[/cyan]")
    mapper = ProtMapper()

    all_results: List[pd.DataFrame] = []
    all_failed_ids: List[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("[blue]Mapping Ensembl IDs", total=total_unique_ids)
        
        # Create batches to process
        batches = [ensembl_id_list[i:i + BATCH_SIZE] for i in range(0, total_unique_ids, BATCH_SIZE)]
        
        for i, batch in enumerate(batches):
            progress.update(task_id, description=f"[blue]Mapping batch {i+1}/{len(batches)}")
            
            # The library handles retries for this single batch internally
            mapping_df_batch, failed_ids_batch = mapper.get(
                ids=batch,
                from_db="Ensembl",
                to_db="UniProtKB" # Get the full UniProt entry
            )
            
            if mapping_df_batch is not None and not mapping_df_batch.empty:
                all_results.append(mapping_df_batch)
            if failed_ids_batch:
                all_failed_ids.extend(failed_ids_batch)
            
            progress.advance(task_id, len(batch))

    # --- Step 4: Process and Save Combined Results ---
    console.rule("[bold green]Processing and Saving Results[/bold green]")

    if not all_results:
        console.print(Panel("[bold red]Mapping failed critically. No results were returned from the library.[/bold red]", title="Error", border_style="red"))
        return

    # Combine results from all batches
    final_df = pd.concat(all_results, ignore_index=True)

    # <<< THE FIX IS HERE: Rename the 'Entry' column, not the 'To' column >>>
    final_df.rename(columns={'From': 'Ensembl_ID', 'Entry': 'UniProt_ID'}, inplace=True)
    
    # Now select just the columns you need
    final_df = final_df[['Ensembl_ID', 'UniProt_ID']].drop_duplicates()
    
    final_df.to_csv(output_csv_path, index=False)
    console.print(f"✅ Successfully saved {len(final_df):,} mappings to [cyan]'{output_csv_path}'[/cyan]")

    if all_failed_ids:
        unique_failed = sorted(list(set(all_failed_ids)))
        with open(failed_ids_path, 'w') as f:
            for item in unique_failed:
                f.write(f"{item}\n")
        console.print(f"✅ List of {len(unique_failed):,} failed IDs saved to [cyan]'{failed_ids_path}'[/cyan]")

    # --- Step 5: Display Final Statistics Table ---
    stats_table = Table(title="📊 Final Mapping Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="yellow")

    stats_table.add_row("Total Ensembl IDs to Map", f"{total_unique_ids:,}")
    stats_table.add_row("Ensembl IDs Successfully Mapped", f"{final_df['Ensembl_ID'].nunique():,}")
    stats_table.add_row("Total UniProt Mappings Found", f"{len(final_df):,}")
    stats_table.add_row("Ensembl IDs Failed to Map", f"{len(set(all_failed_ids)):,}")
    
    console.print(stats_table)
    console.rule("[bold green]Pipeline Finished[/bold green]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map Ensembl IDs to UniProt IDs using uniprot-id-mapper.")
    parser.add_argument("--opentargets", default='data/processed/associations/open_targets/opentargets_target_disease_associations.pickle', help="Path to the processed OpenTargets pickle file.")
    parser.add_argument("--output_dir", default='data/processed/mappings', help="Directory to save the output files.")
    args = parser.parse_args()

    # Before running, make sure you have installed the rich library:
    # pip install rich
    build_protein_id_mapping(args.opentargets, args.output_dir)
