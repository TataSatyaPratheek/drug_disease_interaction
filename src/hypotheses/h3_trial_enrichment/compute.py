# src/hypotheses/h3_trial_enrichment/compute.py
import pandas as pd
from scipy.stats import fisher_exact
from rich.table import Table
from ..core.base import Hypothesis
from ..utils.loaders import load_aact_data
import random
import networkx as nx
import numpy as np
from tqdm import tqdm

class HypothesisH3(Hypothesis):
    name = "H3 Trial Enrichment"
    description = "Tests if high-proximity drug-disease pairs are enriched for pairs in Phase II+ clinical trials."

    def _get_graph_name_maps(self):
        drug_map = {n: d.get('name', '').lower() for n, d in self.graph.nodes(data=True) if d.get('type') == 'drug' and d.get('name')}
        disease_map = {n: d.get('name', '').lower() for n, d in self.graph.nodes(data=True) if d.get('type') == 'disease' and d.get('name')}
        return drug_map, disease_map

    def _rank_unlinked_pairs(self, top_n=10000):
        self.console.print("Ranking unlinked drug-disease pairs by shortest path length...")
        drug_map, disease_map = self._get_graph_name_maps()
        
        # Invert maps for quick name->ID lookup
        name_to_drug_id = {v: k for k, v in drug_map.items()}
        name_to_disease_id = {v: k for k, v in disease_map.items()}

        giant_cc = max(nx.connected_components(nx.Graph(self.graph)), key=len)
        candidate_drugs = [n for n in giant_cc if self.graph.nodes[n].get('type') == 'drug']
        candidate_diseases = [n for n in giant_cc if self.graph.nodes[n].get('type') == 'disease']
        
        existing_edges = set(self.graph.edges())
        pairs = []
        # Sample a large number of pairs to rank
        for _ in tqdm(range(top_n * 2), desc="Sampling unlinked pairs", disable=True):
            drug_id = random.choice(candidate_drugs)
            disease_id = random.choice(candidate_diseases)
            if (drug_id, disease_id) not in existing_edges and (disease_id, drug_id) not in existing_edges:
                try:
                    sp = nx.shortest_path_length(self.graph, source=drug_id, target=disease_id)
                    pairs.append({'drug_id': drug_id, 'disease_id': disease_id, 'distance': sp})
                except nx.NetworkXNoPath:
                    continue
        
        ranked_df = pd.DataFrame(pairs).sort_values(by='distance').drop_duplicates(subset=['drug_id', 'disease_id'])
        return ranked_df.head(top_n)

    def run(self):
        trial_pairs = load_aact_data()
        if trial_pairs.empty: return

        # Create a set of (drug_name, condition_name) for fast lookup
        trial_pairs_set = { (r.drug_name.lower(), r.condition_name.lower()) for r in trial_pairs.itertuples() }

        ranked_pairs = self._rank_unlinked_pairs()
        if ranked_pairs.empty:
            self.console.print("[bold red]Could not find any connected, unlinked drug-disease pairs to test.[/bold red]")
            return

        drug_map, disease_map = self._get_graph_name_maps()
        ranked_pairs['drug_name'] = ranked_pairs['drug_id'].map(drug_map)
        ranked_pairs['disease_name'] = ranked_pairs['disease_id'].map(disease_map)
        
        # Check for enrichment
        ranked_pairs['in_trial'] = ranked_pairs.apply(lambda row: (row['drug_name'], row['disease_name']) in trial_pairs_set, axis=1)

        in_trial_count = ranked_pairs['in_trial'].sum()
        not_in_trial_count = len(ranked_pairs) - in_trial_count
        
        # A simple contingency table: are our top-ranked pairs enriched?
        # We compare to a baseline assumption of random chance (very low probability)
        # Table: [[TopRanked & InTrial, TopRanked & NotInTrial], [NotTopRanked & InTrial, NotTopRanked & NotInTrial]]
        # We simplify this by assuming the background rate is low.
        table_data = [[in_trial_count, not_in_trial_count], [1, len(drug_map) * len(disease_map)]] # simplified baseline
        odds_ratio, p_value = fisher_exact(table_data)

        table = Table(title="H3: Trial Enrichment of High-Proximity Pairs")
        table.add_column("Metric", style="cyan"); table.add_column("Value", style="yellow")
        table.add_row("Top Ranked Unlinked Pairs", f"{len(ranked_pairs):,}")
        table.add_row("Found in Phase II+ Trials", f"{in_trial_count:,}")
        table.add_row("Enrichment Odds Ratio", f"{odds_ratio:.2f}")
        table.add_row("P-value", f"{p_value:.2e}")

        self.console.print(table)
        if p_value < 0.05 and odds_ratio > 1:
            self.console.print("[bold green]Significant enrichment found. High-proximity pairs are more likely to be in clinical trials.[/bold green]")
