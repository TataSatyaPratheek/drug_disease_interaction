from collections import Counter, defaultdict
from typing import Dict, List

import networkx as nx
import pandas as pd
from rich.panel import Panel
from rich.table import Table
from scipy.stats import fisher_exact
from tqdm import tqdm

from ..core import Hypothesis

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


class HypothesisH2(Hypothesis):
    name = "H2 Module PageRank"
    description = (
        "Within disease-dense Louvain modules, proteins with highest PageRank "
        "are enriched for known drug targets."
    )

    @staticmethod
    def _module_name(subg: nx.Graph, disease_nodes: List[str]) -> str:
        names = [subg.nodes[n].get("name", "") for n in disease_nodes]
        non_empty_names = [d for d in names if d and d.strip()]
        
        if non_empty_names:
            most_common = Counter(non_empty_names).most_common(1)
            return most_common[0][0] if most_common else "Unknown"
        else:
            return "Unknown"


    def run(
        self,
        min_comm_size: int = 100,
        disease_fraction: float = 0.5,
        top_k_proteins: int = 10,
    ) -> None:
        if not community_louvain:
            self.console.print("[red]Install: pip install python-louvain")
            return

        self.console.print("Detecting Louvain communities…")
        partition: Dict[str, int] = community_louvain.best_partition(self.graph.to_undirected())
        comm2nodes: Dict[int, List[str]] = defaultdict(list)
        for node, cid in partition.items():
            comm2nodes[cid].append(node)

        results = []
        for cid, nodes in tqdm(comm2nodes.items(), desc="Scanning communities"):
            if len(nodes) < min_comm_size:
                continue

            subg = self.graph.subgraph(nodes)
            diseases = [n for n in subg if subg.nodes[n].get("type") == "disease"]
            if len(diseases) / len(subg) < disease_fraction:
                continue

            proteins = [n for n in subg if subg.nodes[n].get("type") == "protein"]
            if not proteins:
                continue

            pr = nx.pagerank(subg, alpha=0.85, max_iter=100)
            top_proteins = sorted(proteins, key=lambda p: pr.get(p, 0), reverse=True)[:top_k_proteins]
            known_targets = {p for p in proteins if subg.in_degree(p) > 0}

            a = len(set(top_proteins) & known_targets)
            b = len(top_proteins) - a
            c = len(known_targets - set(top_proteins))
            d = len(proteins) - len(set(top_proteins) | known_targets)

            odds, p_val = fisher_exact([[a, b], [c, d]], alternative="greater")
            results.append(
                dict(
                    cid=cid, size=len(subg), dis_ratio=len(diseases)/len(subg),
                    name=self._module_name(subg, diseases),
                    odds=odds, p=p_val, top_proteins=top_proteins
                )
            )

        results.sort(key=lambda r: r["odds"], reverse=True)
        sig = [r for r in results if r["p"] < 0.05 and r["odds"] > 1]

        leaderboard = results[:10]
        for r in leaderboard:
            tbl = Table(title=f"H2 – Module {r['cid']}: {r['name']}")
            tbl.add_column("Size", str(r["size"]), style="magenta")
            tbl.add_column("%Disease", f"{r['dis_ratio']:.1%}", style="yellow")
            tbl.add_column("Odds Ratio", f"{r['odds']:.2f}", style="blue")
            tbl.add_column("p-value", f"{r['p']:.1e}", style="green")
            self.console.print(tbl)
            self.console.print(Panel(
                f"[bold]Top {top_k_proteins} proteins:[/]\n" +
                ", ".join(self.graph.nodes[p].get("name", p) for p in r["top_proteins"]),
                border_style="cyan"
            ))

        # save csv of all modules
        pd.DataFrame(results).to_csv(self.output_dir / "h2_module_stats.csv", index=False)
        self.console.print(f"[green]Module table saved → {self.output_dir/'h2_module_stats.csv'}[/]")
