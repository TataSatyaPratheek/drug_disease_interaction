# src/hypotheses/h4_side_effect_overlap/compute.py
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
from scipy.stats import spearmanr
from ..core.base import Hypothesis
from ..utils.loaders import load_sider_data, get_stitch_to_drugbank_mapping
from rich.table import Table
from tqdm import tqdm

class HypothesisH4(Hypothesis):
    name = "H4 Side-Effect Overlap"
    description = "Tests if drugs with overlapping targets have similar side-effect profiles using real data."

    def sparse_jaccard_similarity(self, X, Y=None):
        """
        Compute Jaccard similarity for sparse matrices.
        Based on the formula: J(A,B) = |A ∩ B| / |A ∪ B|
        """
        if Y is None:
            Y = X
        
        X = X.astype(bool).astype(int)
        Y = Y.astype(bool).astype(int)
        
        # Compute intersection: X * Y.T
        intersect = X.dot(Y.T)
        
        # Compute row sums (number of 1s in each row)
        x_sum = np.array(X.sum(axis=1)).flatten()
        y_sum = np.array(Y.sum(axis=1)).flatten()
        
        # Create meshgrid for union calculation
        xx, yy = np.meshgrid(x_sum, y_sum, indexing='ij')
        union = xx + yy - intersect.toarray()
        
        # Avoid division by zero
        union[union == 0] = 1
        
        # Compute Jaccard similarity
        similarity = intersect.toarray() / union
        
        return similarity

    def run(self, sample_size=100):  # Reduced sample size for testing
        # 1. Load SIDER data and the STITCH-DrugBank mapping
        sider_df = load_sider_data()
        stitch_map = get_stitch_to_drugbank_mapping()
        if sider_df.empty or not stitch_map:
            self.console.print("[bold red]Missing SIDER data or STITCH-DrugBank mapping. Aborting H4.[/bold red]")
            return

        # 2. Integrate data: Map STITCH IDs to DrugBank IDs
        sider_df['drugbank_id'] = sider_df['stitch_id_flat'].map(stitch_map)
        sider_df.dropna(subset=['drugbank_id'], inplace=True)
        
        graph_drugs = {n for n, d in self.graph.nodes(data=True) if d.get('type') == 'drug'}
        sider_df = sider_df[sider_df['drugbank_id'].isin(graph_drugs)]
        
        if sider_df.empty:
            self.console.print("[bold red]No overlap found between SIDER drugs and graph drugs after mapping.[/bold red]")
            return

        self.console.print(f"Found {sider_df['drugbank_id'].nunique():,} common drugs between graph and SIDER.")

        # 3. Create Drug-Target Matrix
        self.console.print("Creating Drug-Target matrix...")
        proteins = sorted([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'protein'])
        protein_map = {name: i for i, name in enumerate(proteins)}
        
        drug_list = sorted(sider_df['drugbank_id'].unique())
        drug_map = {name: i for i, name in enumerate(drug_list)}
        
        drug_target_matrix = lil_matrix((len(drug_list), len(proteins)), dtype=np.int8)
        for drug_id in tqdm(drug_list, desc="Building Target Matrix"):
            row_idx = drug_map[drug_id]
            targets = {n for n in self.graph.neighbors(drug_id) if self.graph.nodes[n].get('type') == 'protein'}
            for target in targets:
                if target in protein_map:
                    drug_target_matrix[row_idx, protein_map[target]] = 1

        # 4. Create Drug-Side-Effect Matrix
        self.console.print("Creating Drug-Side-Effect matrix...")
        side_effects = sorted(sider_df['side_effect_name'].unique())
        se_map = {name: i for i, name in enumerate(side_effects)}
        
        drug_se_matrix = lil_matrix((len(drug_list), len(side_effects)), dtype=np.int8)
        for _, row in tqdm(sider_df.iterrows(), total=len(sider_df), desc="Building SE Matrix"):
            if row['drugbank_id'] in drug_map:
                row_idx = drug_map[row['drugbank_id']]
                col_idx = se_map[row['side_effect_name']]
                drug_se_matrix[row_idx, col_idx] = 1

        # 5. Calculate similarities and correlate
        actual_sample_size = min(sample_size, len(drug_list))
        self.console.print(f"Calculating pairwise similarities for {actual_sample_size} drugs...")
        
        if actual_sample_size < 2:
            self.console.print("[bold red]Need at least 2 drugs for correlation analysis.[/bold red]")
            return
        
        sampled_indices = np.random.choice(range(len(drug_list)), size=actual_sample_size, replace=False)
        
        target_csr = drug_target_matrix[sampled_indices, :].tocsr()
        se_csr = drug_se_matrix[sampled_indices, :].tocsr()

        # Use our custom sparse Jaccard similarity function
        target_similarity = self.sparse_jaccard_similarity(target_csr)
        se_similarity = self.sparse_jaccard_similarity(se_csr)
        
        # Extract upper triangle to avoid self-correlation
        indices = np.triu_indices_from(target_similarity, k=1)
        target_vals = target_similarity[indices]
        se_vals = se_similarity[indices]
        
        # Only correlate if we have enough data points
        if len(target_vals) < 3:
            self.console.print("[bold red]Not enough pairs for correlation analysis.[/bold red]")
            return
        
        corr, p_value = spearmanr(target_vals, se_vals)

        table = Table(title="H4: Target Overlap vs. Side-Effect Similarity (Real Data)")
        table.add_column("Metric", style="cyan"); table.add_column("Value", style="yellow")
        table.add_row("Common Drugs (Graph & SIDER)", f"{len(drug_list):,}")
        table.add_row("Sampled for Correlation", f"{actual_sample_size:,}")
        table.add_row("Protein Targets", f"{len(proteins):,}")
        table.add_row("Side Effects", f"{len(side_effects):,}")
        table.add_row("Spearman Correlation", f"{corr:.3f}")
        table.add_row("P-value", f"{p_value:.2e}")

        self.console.print(table)
        
        if p_value < 0.05:
            self.console.print(f"[bold green]✓ Significant correlation found! Target similarity explains {corr**2:.1%} of side-effect similarity variance.[/bold green]")
        else:
            self.console.print("[bold yellow]✗ No significant correlation detected.[/bold yellow]")
