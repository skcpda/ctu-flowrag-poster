from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

__all__ = ["build_discourse_graph"]


_EDGE_FORWARD = "next"
_EDGE_BACK = "prev"


def _add_edges(g: nx.DiGraph, num_nodes: int) -> None:
    for idx in range(num_nodes):
        if idx < num_nodes - 1:
            g.add_edge(idx, idx + 1, type=_EDGE_FORWARD, weight=1.0)
        if idx > 0:
            g.add_edge(idx, idx - 1, type=_EDGE_BACK, weight=0.5)


def build_discourse_graph(ctus: List[Dict]) -> Dict:
    """Build a NetworkX DiGraph and compute salience (PageRank).

    Returns a dict with keys expected by higher-level modules but now also
    mutates the *ctus* list by injecting the `salience` field.
    """
    num_nodes = len(ctus)
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    _add_edges(g, num_nodes)

    # Centrality measures for salience
    pr = nx.pagerank(g, alpha=0.9, weight="weight")
    bc = nx.betweenness_centrality(g, normalized=True, weight="weight")

    # Attach salience back to CTUs (mean of PR & BC)
    for idx, ctu in enumerate(ctus):
        sal = (pr.get(idx, 0.0) + bc.get(idx, 0.0)) / 2.0
        ctu["salience"] = float(sal)
        ctu["pagerank"] = float(pr.get(idx, 0.0))
        ctu["betweenness"] = float(bc.get(idx, 0.0))

    # Prepare adjacency matrix (dense list for JSON easiness)
    adjacency: List[List[float]] = [[0.0] * num_nodes for _ in range(num_nodes)]
    edge_weights: List[Tuple[int, int, float]] = []
    for u, v, data in g.edges(data=True):
        w = float(data.get("weight", 1.0))
        adjacency[u][v] = w
        edge_weights.append((u, v, w))

    return {
        "num_nodes": num_nodes,
        "num_edges": g.number_of_edges(),
        "adjacency_matrix": adjacency,
        "edge_weights": edge_weights,
        "parameters": {"version": "pagerank+betweenness"},
    } 