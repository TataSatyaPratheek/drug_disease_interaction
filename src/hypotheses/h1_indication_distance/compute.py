import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from rich.table import Table
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from ..core import Hypothesis


class HypothesisH1(Hypothesis):
    name = "H1 Indication Distance"
    description = (
        "Approved drug–disease indications have shorter network path "
        "lengths than random pairs."
    )

    def _positive_pairs(self, max_pairs: int) -> List[Tuple[str, str]]:
        self.console.print(f"Finding up to {max_pairs:,} known indications…")
        pairs = set()
        protein2drugs = defaultdict(set)
        for d, p, _, attr in self.graph.edges(data=True, keys=True):
            if attr.get("type") in {"targets", "enzymes", "carriers", "transporters"}:
                if (
                    self.graph.nodes[d].get("type") == "drug"
                    and self.graph.nodes[p].get("type") == "protein"
                ):
                    protein2drugs[p].add(d)

        for p, dis, attr in self.graph.out_edges(list(protein2drugs), data=True):
            if (
                attr.get("type") == "associated_with"
                and self.graph.nodes[dis].get("type") == "disease"
            ):
                for dr in protein2drugs[p]:
                    pairs.add((dr, dis))
                    if len(pairs) >= max_pairs:
                        return list(pairs)
        return list(pairs)

    def _negative_pairs(
        self,
        n_pairs: int,
        positive: set,
        candidate_drugs: List[str] = None,
        candidate_dis: List[str] = None,
    ) -> List[Tuple[str, str]]:
        self.console.print(f"Sampling {n_pairs:,} random (negative) pairs…")
        drugs = candidate_drugs or [n for n, d in self.graph.nodes(data=True) if d["type"] == "drug"]
        diseases = candidate_dis or [n for n, d in self.graph.nodes(data=True) if d["type"] == "disease"]
        neg = set()
        while len(neg) < n_pairs:
            cand = (random.choice(drugs), random.choice(diseases))
            if cand not in positive:
                neg.add(cand)
        return list(neg)

    def _compute_sp_lengths(
        self, pairs: List[Tuple[str, str]], desc: str
    ) -> List[float]:
        self.console.print(f"Computing shortest-path lengths for {desc}…")
        G = nx.Graph(self.graph)  # work on simple, undirected graph
        lengths = []
        
        # Process each pair individually to get source-to-target distance
        for s, t in tqdm(pairs, desc=desc):
            try:
                length = nx.shortest_path_length(G, source=s, target=t)
                lengths.append(length)
            except nx.NetworkXNoPath:
                lengths.append(np.inf)  # no path exists
        
        return lengths

    def run(self, sample_size: int = 1_000) -> None:
        pos_pairs = self._positive_pairs(sample_size)
        if not pos_pairs:
            self.console.print("[red]No positive pairs found – aborting.[/]")
            return
        # restrict negatives to nodes in the same connected component
        giant_cc = max(nx.connected_components(nx.Graph(self.graph)), key=len)
        neg_pairs = self._negative_pairs(
            len(pos_pairs),
            set(pos_pairs),
            candidate_drugs=[n for n in giant_cc if self.graph.nodes[n]["type"] == "drug"],
            candidate_dis=[n for n in giant_cc if self.graph.nodes[n]["type"] == "disease"],
        )

        pos_sp = self._compute_sp_lengths(pos_pairs, "positive pairs")
        neg_sp = self._compute_sp_lengths(neg_pairs, "negative pairs")

        finite_pos = [d for d in pos_sp if np.isfinite(d)]
        finite_neg = [d for d in neg_sp if np.isfinite(d)]
        stat, p = mannwhitneyu(finite_pos, finite_neg, alternative="less")

        tbl = Table(title="H1 – Drug–disease path-length test")
        for k, v in [
            ("#Positives", f"{len(pos_sp):,}"),
            ("#Negatives", f"{len(neg_sp):,}"),
            ("Reachable pos", f"{len(finite_pos):,}"),
            ("Reachable neg", f"{len(finite_neg):,}"),
            ("Median SP (pos)", f"{np.median(finite_pos):.2f}"),
            ("Median SP (neg)", f"{np.median(finite_neg):.2f}"),
            ("Q1–Q3 pos", f"{np.percentile(finite_pos,[25,75])}"),
            ("Q1–Q3 neg", f"{np.percentile(finite_neg,[25,75])}"),
            ("U-statistic", f"{stat:.1f}"),
            ("one-sided p", f"{p:.2e}"),
        ]:
            tbl.add_row(k, v)

        self.console.print(tbl)
        if p < 0.05 and finite_pos and finite_neg:
            self.console.print("[bold green]✓ Significant.[/]")
        else:
            self.console.print("[bold yellow]✗ Not significant.[/]")
