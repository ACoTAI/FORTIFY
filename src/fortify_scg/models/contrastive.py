"""
Contrastive learning utilities: graph augmentations, InfoNCE loss, cosine similarity, and lambda schedule.
Assumes:
- A "graph sample" contains (SCG: nx.DiGraph, h0: Dict[node_id]->np.ndarray, label: int).
- Negative samples are graphs with different CWE labels or non-vulnerable graphs.
- For augmentation, we apply edge dropout (p=0.3) and node feature masking (p=0.15).
"""
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
import random

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float((a @ b) / (na * nb))

def mean_pool(h: Dict[str, np.ndarray]) -> np.ndarray:
    if not h:
        return np.zeros((1,), dtype=np.float32)
    M = np.stack(list(h.values()), axis=0)
    return M.mean(axis=0)

def edge_dropout(G: nx.DiGraph, p: float, rnd: random.Random) -> nx.DiGraph:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u,v,d in G.edges(data=True):
        if rnd.random() > p:
            H.add_edge(u,v, **d)
    return H

def node_mask(h: Dict[str, np.ndarray], p: float, rnd: random.Random) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in h.items():
        if rnd.random() < p:
            out[k] = np.zeros_like(v)
        else:
            out[k] = v.copy()
    return out

def augment(G: nx.DiGraph, h: Dict[str, np.ndarray], seed: int = 0) -> Tuple[nx.DiGraph, Dict[str, np.ndarray]]:
    rnd = random.Random(seed)
    G2 = edge_dropout(G, p=0.3, rnd=rnd)
    h2 = node_mask(h, p=0.15, rnd=rnd)
    return G2, h2

def info_nce_loss(z_i: np.ndarray, z_j: np.ndarray, negatives: List[np.ndarray], tau: float) -> Tuple[float, float, float]:
    s_pos = cosine(z_i, z_j)
    logits = [s_pos]
    for z_k in negatives:
        logits.append(cosine(z_i, z_k))
    logits = np.array(logits, dtype=np.float64) / max(1e-12, tau)
    # numerically stable softmax cross-entropy with positive as index 0
    m = logits.max()
    exps = np.exp(logits - m)
    denom = exps.sum()
    loss = -np.log(exps[0] / denom + 1e-12)
    return float(loss), float(s_pos), float(len(negatives))

def lambda_schedule(t: int, lam0: float = 0.7) -> float:
    return float(lam0 * np.exp(-0.02 * t))
