import networkx as nx, numpy as np
from fortify_scg.train import train_rgcn_contrastive

def make_sample():
    G = nx.DiGraph()
    for n in ["a","b","c"]:
        G.add_node(n, api="strcpy" if n=="b" else "")
    G.add_edge("a","b", etype="cascading")
    G.add_edge("b","c", etype="cascading")
    h0 = {"a": np.random.randn(512), "b": np.random.randn(512), "c": np.random.randn(512)}
    return {"SCG": G, "h0": h0, "LABEL": 1}

def test_train_one_step():
    pos = make_sample()
    neg = make_sample(); neg["LABEL"] = 0
    ctx = {"SCG": pos["SCG"], "h0": pos["h0"], "LABEL": pos["LABEL"], "NUM_CLASSES": 2,
           "NEG_GRAPHS": [neg], "API_NODES": {"b"},
           "HP": {"epochs": 1, "K": 1}}
    out = train_rgcn_contrastive(ctx)
    assert "train_log" in out and len(out["train_log"]) == 1
