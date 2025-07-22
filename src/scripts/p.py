# scripts/build_graph_snapshot.py
import pickle, json, networkx as nx
from pathlib import Path

GRAPH_PKL = Path("data/graph/full_mapped/ddi_knowledge_graph.pickle")
SNAPSHOT  = GRAPH_PKL.with_suffix(".stats.json")

def main():
    g: nx.MultiDiGraph = pickle.loads(Path(GRAPH_PKL).read_bytes())
    snap = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "node_types": {},
        "edge_types": {},
    }
    for _, d in g.nodes(data=True):
        snap["node_types"][d.get("type", "unknown")] = \
            snap["node_types"].get(d.get("type", "unknown"), 0) + 1
    for _, _, d in g.edges(data=True):
        snap["edge_types"][d.get("type", "unknown")] = \
            snap["edge_types"].get(d.get("type", "unknown"), 0) + 1
    SNAPSHOT.write_text(json.dumps(snap))
    print("Snapshot written →", SNAPSHOT)

if __name__ == "__main__":
    main()
