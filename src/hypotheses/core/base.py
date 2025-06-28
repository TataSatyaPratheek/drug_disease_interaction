from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import networkx as nx
from rich.console import Console


class Hypothesis(ABC):
    """
    Abstract base-class every hypothesis module must extend.
    """

    name: str = "Unnamed Hypothesis"
    description: str = "No description provided."

    def __init__(self, graph_path: Path, output_dir: Path) -> None:
        self.console = Console()
        self.graph_path = Path(graph_path).resolve()

        # each hypothesis writes in its own sub-folder:  reports/hypotheses/h1/
        self.output_dir = (Path(output_dir) / self.name.split()[0].lower()).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with self.console.status(f"[bold cyan]Loading graph from {self.graph_path}…"):
            with open(self.graph_path, "rb") as fh:
                self.graph: nx.MultiDiGraph = pickle.load(fh)

        self.console.print(
            f"✅ Graph loaded: {self.graph.number_of_nodes():,} nodes – "
            f"{self.graph.number_of_edges():,} edges."
        )

    @abstractmethod
    def run(self) -> None:
        """Execute analysis and write artefacts."""
        raise NotImplementedError
