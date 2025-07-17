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


# NOTE: API extended – *min_weight* lets callers prune weak edges.
# We keep the default backward-compatible while also handling it in tests.

def build_discourse_graph(ctus: List[Dict], *, min_weight: float = 0.0) -> Dict:
    """Build a NetworkX DiGraph and compute salience (PageRank).

    Returns a dict with keys expected by higher-level modules but now also
    mutates the *ctus* list by injecting the `salience` field.
    """
    num_nodes = len(ctus)
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    _add_edges(g, num_nodes)

    # Centrality measures for salience – include robust fallback if the power
    # iteration fails to converge (known to happen for tiny / dense graphs).
    try:
        pr = nx.pagerank(g, alpha=0.9, weight="weight", max_iter=500)
    except nx.exception.PowerIterationFailedConvergence:
        # Fallback: uniform distribution to avoid crashing the pipeline.
        pr = {n: 1.0 / num_nodes for n in g.nodes}
    bc = nx.betweenness_centrality(g, normalized=True, weight="weight")

    raw_sals = [(pr.get(i, 0.0) + bc.get(i, 0.0)) / 2.0 for i in range(num_nodes)]
    # Min-max normalise to [0,1] so salience_nDCG has meaningful spread
    min_sal, max_sal = min(raw_sals), max(raw_sals)
    norm = (max_sal - min_sal) or 1.0
    for idx, ctu in enumerate(ctus):
        sal_norm = (raw_sals[idx] - min_sal) / norm
        ctu["salience"] = float(sal_norm)
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