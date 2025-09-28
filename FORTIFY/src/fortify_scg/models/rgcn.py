"""
Relation-aware Graph Convolution with risk-sensitive attention and degree normalization.
Assumes: 
- SCG is a networkx.DiGraph whose edges carry 'etype' (relation).
- h is a dict[node_id] -> np.ndarray of shape (d_in,).
- central data for high-risk factor is provided via an API set; edge is high-risk if its endpoints include an API node.
Inputs needed at call-sites:
- relations: iterable of relation strings present in SCG (e.g., {"cascading","local_degree","global_degree","structural"}).
- api_nodes: set of node_ids that are API/sensitive.
"""
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Set
import numpy as np
import networkx as nx

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    s = e.sum() + 1e-12
    return e / s

class RGCNLayer:
    def __init__(self, d_in: int, d_out: int, relations: Iterable[str], seed: int = 0):
        self.d_in, self.d_out = d_in, d_out
        self.relations = list(relations)
        rng = np.random.default_rng(seed)
        self.W_r = {r: rng.standard_normal((d_out, d_in)) / np.sqrt(d_in) for r in self.relations}
        # shared projections for attention
        self.WQ = rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)
        self.WK = rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)
        self.q_r = {r: rng.standard_normal((2*d_in,)) / np.sqrt(2*d_in) for r in self.relations}

    def forward(self, G: nx.DiGraph, h: Dict[str, np.ndarray], api_nodes: Set[str], gamma: float = 1.5) -> Dict[str, np.ndarray]:
        d_in = self.d_in
        out: Dict[str, np.ndarray] = {i: np.zeros((self.d_out,), dtype=np.float64) for i in G.nodes}
        for r in self.relations:
            # build neighbor lists by relation
            for i in G.nodes:
                # neighbors under type r
                nbrs = [j for j in G.predecessors(i) if G.get_edge_data(j, i).get("etype") == r] + \
                       [j for j in G.successors(i)   if G.get_edge_data(i, j).get("etype") == r]
                if not nbrs:
                    continue
                # attention scores per neighbor
                scores = []
                for j in nbrs:
                    hi = h[i]; hj = h[j]
                    q = self.q_r[r]
                    qi = self.WQ.T @ hi
                    kj = self.WK.T @ hj
                    cat = np.concatenate([qi, kj], axis=0)
                    s = (q * cat).sum() / np.sqrt(2*d_in)
                    beta = 1.0 + gamma * (1 if (i in api_nodes or j in api_nodes) else 0)
                    scores.append(s + np.log(beta))  # multiply inside softmax by adding log beta
                alphas = _softmax(np.array(scores))
                deg_norm = 1.0 / np.sqrt(len(nbrs) + 1.0)
                acc = np.zeros((self.d_out,), dtype=np.float64)
                for a, j in zip(alphas, nbrs):
                    acc += a * (self.W_r[r] @ h[j])
                out[i] += deg_norm * acc
        # activation (ReLU)
        for i in out:
            out[i] = np.maximum(out[i], 0.0)
        return out
