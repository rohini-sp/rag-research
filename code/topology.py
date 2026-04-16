"""
Compute graph-topology metrics for any extracted-triples CSV.

Usage:
    python code/topology.py path/to/extracted_triples.csv [--llm <name>]

Builds a directed multigraph from the triples and computes:
    n_nodes, n_edges, mean_degree, max_degree,
    n_components (weakly connected),
    largest_component_fraction,
    density,
    n_self_loops.

Outputs:
    - Prints to stdout
    - Appends one row to results/topology.csv
    - Saves the graph to graphs/<llm>.graphml for later inspection
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd

from config import GRAPHS, RESULTS
from metrics import normalise


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for r in df.itertuples():
        s, p, o = normalise(r.subject), str(r.predicate), normalise(r.object)
        if not s or not o:
            continue
        s_type = getattr(r, "subject_type", "") if "subject_type" in df.columns else ""
        o_type = getattr(r, "object_type",  "") if "object_type"  in df.columns else ""
        g.add_node(s, type=s_type)
        g.add_node(o, type=o_type)
        g.add_edge(s, o, predicate=p)
    return g


def topology_metrics(g: nx.MultiDiGraph) -> dict:
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    degrees = dict(g.degree())
    mean_degree = (sum(degrees.values()) / n_nodes) if n_nodes else 0.0
    max_degree  = max(degrees.values()) if degrees else 0

    # Weakly connected components on the underlying undirected graph
    undirected = g.to_undirected(as_view=False)
    components = list(nx.connected_components(undirected))
    n_components = len(components)
    largest = max((len(c) for c in components), default=0)
    largest_frac = (largest / n_nodes) if n_nodes else 0.0
    self_loops = nx.number_of_selfloops(g)

    # Density: edges / (n*(n-1)) for directed graph
    density = nx.density(g) if n_nodes > 1 else 0.0

    return {
        "n_nodes":                    n_nodes,
        "n_edges":                    n_edges,
        "mean_degree":                round(mean_degree, 3),
        "max_degree":                 max_degree,
        "n_components":               n_components,
        "largest_component_fraction": round(largest_frac, 3),
        "density":                    round(density, 5),
        "n_self_loops":               self_loops,
    }


def append_to_results(llm: str, row: dict) -> Path:
    out = RESULTS / "topology.csv"
    full = {"llm": llm,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **row}
    df_new = pd.DataFrame([full])
    if out.exists():
        df = pd.read_csv(out)
        df = df[df["llm"] != llm]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("extracted_csv", type=Path)
    ap.add_argument("--llm", default=None)
    args = ap.parse_args()

    if not args.extracted_csv.exists():
        print(f"Not found: {args.extracted_csv}")
        return 1
    llm = args.llm or args.extracted_csv.stem

    df = pd.read_csv(args.extracted_csv)
    g = build_graph(df)
    metrics = topology_metrics(g)

    print(f"Topology for: {llm}")
    for k, v in metrics.items():
        print(f"  {k:<32} {v}")

    graph_path = GRAPHS / f"{llm}.graphml"
    nx.write_graphml(g, graph_path)
    print(f"\nGraph saved to {graph_path}")

    out = append_to_results(llm, metrics)
    print(f"Appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
