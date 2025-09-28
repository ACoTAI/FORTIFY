from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
import math
import networkx as nx

Hyperedge = Tuple[str, str, str]  # (i, j, k) for edges (i->j, j->k)

@dataclass
class Hypergraph:
    nodes: List[str]
    hyperedges: List[Hyperedge]
    weights: Dict[Hyperedge, float]

def _edge_pairs_from_paths(paths: Iterable[list[str]]) -> Dict[Hyperedge, int]:
    """
    Count co-occurring consecutive edge pairs (i->j, j->k) along provided paths.
    Each path is a list of node ids. We count for every triple that appears consecutively.
    """
    counts: Dict[Hyperedge, int] = {}
    for path in paths or []:
        for a, b, c in zip(path, path[1:], path[2:]):
            he = (a, b, c)
            counts[he] = counts.get(he, 0) + 1
    return counts

def _structural_edge_pairs(SCG: nx.DiGraph) -> List[Hyperedge]:
    """Structural criterion: share middle node j if both (i->j) and (j->k) exist in SCG."""
    pairs: List[Hyperedge] = []
    for j in SCG.nodes:
        preds = list(SCG.predecessors(j))
        succs = list(SCG.successors(j))
        for i in preds:
            for k in succs:
                if i != k:
                    pairs.append((i, j, k))
    return pairs

def build_hypergraph(SCG: nx.DiGraph, paths: Optional[Iterable[list[str]]] = None) -> Hypergraph:
    """
    Create hyperedges ε_ijk when (i->j) and (j->k) are in SCG. If `paths` are provided,
    we mark semantic co-occurrence by counting the number of times ε appears in paths.
    Weights w(ε) = log(1+|Π_ε|)/log(1+max_ε' |Π_ε'|), defaulting |Π_ε|=1 if semantic data missing,
    which results in uniform weights.
    """
    structural = _structural_edge_pairs(SCG)
    counts = _edge_pairs_from_paths(paths) if paths is not None else {}
    if not counts:
        # fallback: assign all structural pairs a count of 1
        counts = {he: 1 for he in structural}
    else:
        # ensure structural condition holds; drop invalid triples
        counts = {he: c for he, c in counts.items() if SCG.has_edge(he[0], he[1]) and SCG.has_edge(he[1], he[2])}

    max_count = max(counts.values()) if counts else 1
    weights: Dict[Hyperedge, float] = {}
    for he, c in counts.items():
        w = math.log(1 + c) / math.log(1 + max_count)
        weights[he] = max(0.0, min(1.0, w))

    nodes = list(SCG.nodes)
    hyperedges = list(weights.keys())
    return Hypergraph(nodes=nodes, hyperedges=hyperedges, weights=weights)

def adaptive_prune(H: Hypergraph, centrality: Dict[str, float], thresh_scale: float = 0.1) -> Hypergraph:
    """
    Eq.(5): keep ε with w(ε) >= 0.1 * (sum_n C(n))/|N_scg| .
    """
    if not H.nodes:
        return H
    avg_c = sum(centrality.get(n, 0.0) for n in H.nodes) / max(1, len(H.nodes))
    tau = thresh_scale * avg_c
    kept = [he for he in H.hyperedges if H.weights.get(he, 0.0) >= tau]
    kept_w = {he: H.weights[he] for he in kept}
    return Hypergraph(nodes=H.nodes, hyperedges=kept, weights=kept_w)

def nodes_on_hyperedges(H: Hypergraph) -> Set[str]:
    s: Set[str] = set()
    for i,j,k in H.hyperedges:
        s.add(i); s.add(j); s.add(k)
    return s
