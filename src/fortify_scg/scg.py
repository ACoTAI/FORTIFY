from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set, Iterable, Tuple, Optional
import networkx as nx

class EdgeType(str, Enum):
    CASCADING = "cascading"
    LOCAL_DEG = "local_degree"
    GLOBAL_DEG = "global_degree"
    STRUCTURAL = "structural"

def composite_centrality(G: nx.DiGraph, w_d: float = 0.5, w_c: float = 0.3, w_b: float = 0.2) -> Dict[str, float]:
    Gu = G.to_undirected()
    Cd = nx.degree_centrality(Gu)
    Cc = nx.closeness_centrality(Gu)
    Cb = nx.betweenness_centrality(Gu, normalized=True)
    return {n: w_d*Cd.get(n, 0.0) + w_c*Cc.get(n, 0.0) + w_b*Cb.get(n, 0.0) for n in G.nodes}

def select_sensitive_nodes(G: nx.DiGraph, C: Dict[str, float], tau_top: float = 0.10, api_set: Optional[Set[str]] = None) -> Set[str]:
    api_set = set(a.lower() for a in (api_set or {"strcpy","memcpy","strcat","gets","sprintf"}))
    scores = sorted(((n, C[n]) for n in G.nodes), key=lambda x: x[1], reverse=True)
    k = max(1, int(len(scores) * tau_top))
    top = {n for n, _ in scores[:k]}
    api_hits = {n for n, data in G.nodes(data=True) if str(data.get("api","")).lower() in api_set}
    return top | api_hits

def backward_slice(PDG: nx.DiGraph, seeds: Iterable[str]) -> nx.DiGraph:
    slice_nodes: Set[str] = set()
    frontier = list(seeds)
    while frontier:
        v = frontier.pop()
        if v in slice_nodes:
            continue
        slice_nodes.add(v)
        for u in PDG.predecessors(v):
            if PDG.nodes[u].get("is_input_boundary", False):
                slice_nodes.add(u)
            else:
                frontier.append(u)
    return PDG.subgraph(slice_nodes).copy()

@dataclass
class AugInputs:
    cfg: Optional[nx.DiGraph] = None
    ast: Optional[nx.DiGraph] = None
    def_use: Optional[Dict[str, Set[str]]] = None
    interproc_pairs: Optional[Set[Tuple[str, str]]] = None

def augment_edges(sliceG: nx.DiGraph, aux: AugInputs) -> nx.DiGraph:
    SCG = nx.DiGraph()
    SCG.add_nodes_from(sliceG.nodes(data=True))

    if aux.cfg is not None:
        for u, v in aux.cfg.edges():
            if u in SCG and v in SCG:
                SCG.add_edge(u, v, etype=EdgeType.CASCADING)

    if aux.def_use is not None:
        for u, neigh in aux.def_use.items():
            if u not in SCG:
                continue
            for v in neigh:
                if v in SCG and sliceG.nodes[u].get("bb") == sliceG.nodes[v].get("bb"):
                    SCG.add_edge(u, v, etype=EdgeType.LOCAL_DEG)

    if aux.interproc_pairs is not None:
        for u, v in aux.interproc_pairs:
            if u in SCG and v in SCG:
                SCG.add_edge(u, v, etype=EdgeType.GLOBAL_DEG)

    if aux.ast is not None:
        # naive: connect ast parents to node if both exist
        for n in list(SCG.nodes):
            if n in aux.ast:
                for p in aux.ast.predecessors(n):
                    if p in SCG:
                        SCG.add_edge(p, n, etype=EdgeType.STRUCTURAL)

    return SCG
