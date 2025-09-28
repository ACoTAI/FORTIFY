"""
End-to-end training loop for Section 4.3:
- Builds L-layer RGCN with risk-sensitive attention.
- Produces graph embeddings via mean pooling and computes:
  * Cross-entropy loss (task head) + lambda(t) * InfoNCE contrastive loss.
Data expected in ctx at runtime:
- ctx["SCG"]: nx.DiGraph with 'etype' on edges.
- ctx["h0"]: dict[node_id] -> list/ndarray of shape (d_in,). (from 4.2)
- ctx["LABEL"]: int in [0..C-1] for current graph.
- ctx["NEG_GRAPHS"]: list of dicts with keys {"SCG","h0","LABEL"} used for negatives (labels different from ctx["LABEL"]).
- ctx["NUM_CLASSES"]: int (#classes for classification head).
- ctx["API_NODES"]: set of node_ids considered sensitive (for risk factor β_ij).
Hyperparameters (optional):
- ctx["HP"] = {"L":2, "hidden":[256,256], "tau":0.5, "K":16, "epochs":3, "seed":0, "gamma":1.5}
"""
from __future__ import annotations
from typing import Dict, List, Set
import numpy as np
import networkx as nx

from fortify_scg.models.rgcn import RGCNLayer
from fortify_scg.models.contrastive import augment, info_nce_loss, mean_pool, lambda_schedule

def _to_nd(hlist):
    return {k: (np.array(v, dtype=np.float64) if not isinstance(v, np.ndarray) else v) for k,v in hlist.items()}

def _relations(G: nx.DiGraph):
    return sorted({str(d.get("etype")) for _,_,d in G.edges(data=True)})

def _forward_rgcn(G: nx.DiGraph, h: Dict[str,np.ndarray], layers: List[RGCNLayer], api_nodes: Set[str], gamma: float):
    h_cur = h
    for layer in layers:
        h_cur = layer.forward(G, h_cur, api_nodes=api_nodes, gamma=gamma)
    return h_cur

def _task_head(W: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    logits = W @ z + b  # shape (C,)
    # softmax
    m = logits.max()
    exps = np.exp(logits - m)
    return exps / (exps.sum() + 1e-12)

def _xent(probs: np.ndarray, y: int) -> float:
    p = probs[y] + 1e-12
    return float(-np.log(p))

def train_rgcn_contrastive(ctx: Dict) -> Dict:
    G = ctx["SCG"]
    h0 = _to_nd(ctx["h0"])
    y = int(ctx["LABEL"])
    num_classes = int(ctx["NUM_CLASSES"])
    api_nodes = set(ctx.get("API_NODES", []))

    hp = {"L":2, "hidden":[256,256], "tau":0.5, "K":16, "epochs":3, "seed":0, "gamma":1.5}
    hp.update(ctx.get("HP", {}))

    d_in = len(next(iter(h0.values())))
    dims = [d_in] + hp["hidden"]
    rels = _relations(G)
    layers = []
    for l in range(hp["L"]):
        layers.append(RGCNLayer(d_in=dims[l], d_out=dims[l+1], relations=rels, seed=hp["seed"]+l))

    # task head
    rng = np.random.default_rng(hp["seed"] + 999)
    W = rng.standard_normal((num_classes, dims[-1])) / np.sqrt(dims[-1])
    b = np.zeros((num_classes,), dtype=np.float64)

    logs = []
    for t in range(hp["epochs"]):
        # Augmented positive pair
        G1, h1 = augment(G, h0, seed=hp["seed"] + 10*t + 1)
        G2, h2 = augment(G, h0, seed=hp["seed"] + 10*t + 2)

        H1 = _forward_rgcn(G1, h1, layers, api_nodes=api_nodes, gamma=hp["gamma"])
        H2 = _forward_rgcn(G2, h2, layers, api_nodes=api_nodes, gamma=hp["gamma"])

        z1 = mean_pool(H1); z2 = mean_pool(H2)
        # task probability on view1
        probs = _task_head(W, b, z1)
        loss_ce = _xent(probs, y)

        # Negatives: take first K from provided negative graphs with label != y
        negs = []
        for gsample in ctx["NEG_GRAPHS"]:
            if int(gsample["LABEL"]) != y:
                Gi = gsample["SCG"]; h0i = _to_nd(gsample["h0"])
                Gi1, hi1 = augment(Gi, h0i, seed=hp["seed"] + 33*t + len(negs))
                Hi = _forward_rgcn(Gi1, hi1, layers, api_nodes=api_nodes, gamma=hp["gamma"])
                zi = mean_pool(Hi)
                negs.append(zi)
                if len(negs) >= hp["K"]:
                    break

        loss_con, s_pos, k_used = info_nce_loss(z1, z2, negs, tau=hp["tau"])
        lam = lambda_schedule(t, lam0=0.7)

        loss_total = loss_ce + lam * loss_con

        # Simple gradient-free "update": moving average to illustrate training (for pure numpy demo without autograd)
        # In practice, replace with PyTorch and backprop. Here we just log values.
        logs.append({"epoch": t, "loss_ce": loss_ce, "loss_con": loss_con, "lambda": lam, "loss_total": loss_total, "s_pos": s_pos, "k": k_used})

    ctx["train_log"] = logs
    # final embeddings and prediction
    H_final = _forward_rgcn(G, h0, layers, api_nodes=api_nodes, gamma=hp["gamma"])
    z = mean_pool(H_final)
    probs = _task_head(W, b, z)
    ctx["pred"] = {"probs": probs.tolist(), "label_hat": int(np.argmax(probs))}
    ctx["model_summary"] = {"layers": len(layers), "relations": rels, "dim_in": d_in, "dim_out": dims[-1]}
    return ctx
