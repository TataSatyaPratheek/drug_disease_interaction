import pickle
from gqlalchemy import Memgraph
import networkx as nx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.logging import RichHandler
from rich.table import Table
import logging

# --- Setup Rich logging ---
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)]
)
log = logging.getLogger("memgraph_import")

GRAPH_PICKLE_PATH = "/Users/vi/Documents/not_work/drug_disease_interaction/data/graph/full_mapped/ddi_knowledge_graph.pickle"

with console.status("[bold cyan]Connecting to Memgraph...[/bold cyan]", spinner="dots") as status:
    memgraph = Memgraph("127.0.0.1", 7687)
    log.info("Connected to Memgraph on 127.0.0.1:7687")

with console.status("[orange_red1]Clearing Memgraph database (CAREFUL!)...") as status:
    memgraph.execute("MATCH (n) DETACH DELETE n")
    log.info("Database cleared.")

with console.status("[green]Loading NetworkX graph from pickle...") as status:
    with open(GRAPH_PICKLE_PATH, "rb") as f:
        graph = pickle.load(f)
    log.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# --- Import nodes ---
nodes = list(graph.nodes(data=True))
edges = list(graph.edges(keys=True, data=True))

added_nodes, added_edges, error_nodes, error_edges = 0, 0, 0, 0

# Node import with rich progress bar
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=True
) as progress:
    task = progress.add_task("[cyan]Importing nodes", total=len(nodes))
    for node_id, attr in nodes:
        try:
            node_type = attr.get("type", "Node")
            safe_properties = {str(k): str(v) if v is not None else "" for k, v in attr.items() if k != 'type'}
            cypher = f"CREATE (n:{node_type} {{id: $id"
            for k in safe_properties:
                cypher += f", `{k}`: ${k}"
            cypher += "})"
            params = {"id": str(node_id), **safe_properties}
            memgraph.execute(cypher, params)
            added_nodes += 1
        except Exception as ex:
            log.error(f"[Node Import Error] Node {node_id}: {ex}")
            error_nodes += 1
        progress.update(task, advance=1)
log.info(f"Node import done: {added_nodes} successful, {error_nodes} errors.")

# Edge import with rich progress bar
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=True
) as progress:
    task = progress.add_task("[magenta]Importing edges", total=len(edges))
    for u, v, key, attr in edges:
        try:
            edge_type = attr.get("type", key)
            from_id, to_id = str(u), str(v)
            safe_properties = {str(k): str(v) if v is not None else "" for k, v in attr.items() if k != 'type'}
            cypher = (f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) "
                      f"CREATE (a)-[r:{edge_type} {{"
                      f"id: $eid" +
                      "".join([f", `{k}`: ${k}" for k in safe_properties]) +
                      "}]->(b)")
            params = {"from_id": from_id, "to_id": to_id, "eid": str(key), **safe_properties}
            memgraph.execute(cypher, params)
            added_edges += 1
        except Exception as ex:
            log.error(f"[Edge Import Error] {u}->{v} ({edge_type}): {ex}")
            error_edges += 1
        progress.update(task, advance=1)
log.info(f"Edge import done: {added_edges} successful, {error_edges} errors.")

# Final summary panel
summary = Table(title="Memgraph Import Summary", show_header=False)
summary.add_row("Nodes Imported", str(added_nodes))
summary.add_row("Node import errors", str(error_nodes))
summary.add_row("Edges Imported", str(added_edges))
summary.add_row("Edge import errors", str(error_edges))

console.print()
console.print("[bold green]✔ NetworkX graph imported to Memgraph successfully.[/bold green]")
console.print(summary)
