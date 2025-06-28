# scripts/generate_hypotheses.py

import typer
import importlib
import pkgutil
from pathlib import Path
from rich.console import Console
from rich.table import Table

# --- Configuration ---
# Based on your tree output, this is the final, fully connected graph
DEFAULT_GRAPH_PATH = Path("data/graph/full_mapped/ddi_knowledge_graph.pickle")
DEFAULT_OUTPUT_DIR = Path("reports/hypotheses")
HYPOTHESES_PACKAGE = "src.hypotheses"

app = typer.Typer(help="A CLI for running hypothesis tests on the DDI knowledge graph.")
console = Console()

def get_available_hypotheses():
    """Dynamically finds all available hypothesis modules."""
    package = importlib.import_module(HYPOTHESES_PACKAGE.replace("src.", ""))
    return [name for _, name, _ in pkgutil.iter_modules(package.__path__) if name.startswith('h')]

@app.command()
def list():
    """Lists all available hypothesis tests."""
    console.print("[bold green]Available Hypothesis Tests:[/bold green]")
    table = Table("Name", "Description")
    for name in get_available_hypotheses():
        try:
            mod = importlib.import_module(f"{HYPOTHESES_PACKAGE}.{name}.compute")
            HypoCls = next(c for c in mod.__dict__.values() if isinstance(c, type) and issubclass(c, mod.Hypothesis) and c is not mod.Hypothesis)
            table.add_row(f"[cyan]{name}[/cyan]", HypoCls.description)
        except (ImportError, StopIteration) as e:
            table.add_row(f"[red]{name}[/red]", f"Error loading: {e}")
    console.print(table)

@app.command()
def run(
    name: str = typer.Argument(..., help="The name of the hypothesis to run (e.g., h1_network_proximity)."),
    graph: Path = typer.Option(DEFAULT_GRAPH_PATH, "--graph", "-g", help="Path to the knowledge graph pickle file."),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output", "-o", help="Directory to save report files.")
):
    """Runs a specific hypothesis test."""
    if name not in get_available_hypotheses():
        console.print(f"[bold red]Error: Hypothesis '{name}' not found.[/bold red]")
        console.print("Available tests are:", get_available_hypotheses())
        raise typer.Exit(code=1)

    if not graph.exists():
        console.print(f"[bold red]Error: Graph file not found at '{graph}'[/bold red]")
        raise typer.Exit(code=1)

    try:
        mod = importlib.import_module(f"{HYPOTHESES_PACKAGE}.{name}.compute")
        HypoCls = next(c for c in mod.__dict__.values() if isinstance(c, type) and issubclass(c, mod.Hypothesis) and c is not mod.Hypothesis)
        
        console.rule(f"[bold blue]Running: {HypoCls.name}[/bold blue]")
        instance = HypoCls(graph_path=graph, output_dir=output_dir)
        instance.run()
        console.rule(f"[bold green]Finished: {name}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]An error occurred while running '{name}':[/bold red]")
        console.print_exception()
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
