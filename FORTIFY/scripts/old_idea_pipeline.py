#!/usr/bin/env python
from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
import networkx as nx

from fortify_scg.pipeline import Pipeline
from fortify_scg.scg import composite_centrality, select_sensitive_nodes, backward_slice, augment_edges, AugInputs
from fortify_scg.keywords import compile_patterns, is_sensitive
pipeline = Pipeline()

@pipeline.register("load_data", "Load PDG/CFG/AST from JSON")
def load_data(ctx):
    cfg_path = Path(ctx["_cfg"]["inputs"]["pdg_json"])
    data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    # Build PDG/CFG/AST
    PDG = nx.DiGraph()
    for n in data["nodes"]:
        PDG.add_node(n["id"], **n)
    for u,v in data["pdg_edges"]:
        PDG.add_edge(u,v)
    CFG = nx.DiGraph()
    CFG.add_edges_from(data.get("cfg_edges", []))
    AST = nx.DiGraph()
    AST.add_edges_from(data.get("ast_edges", []))
    DEF_USE = {k:set(v) for k,v in data.get("def_use", {}).items()}
    INTERPROC = set(tuple(x) for x in data.get("interproc_pairs", []))

    ctx["PDG"] = PDG
    ctx["CFG"] = CFG
    ctx["AST"] = AST
    ctx["DEF_USE"] = DEF_USE
    ctx["INTERPROC"] = INTERPROC
    return ctx

@pipeline.register("preprocess", "Centrality + seed selection")
def preprocess(ctx):
    C = composite_centrality(ctx["PDG"], 0.5, 0.3, 0.2)
    # top-10% by centrality
    import math
    scores = sorted(((n, C[n]) for n in ctx["PDG"].nodes), key=lambda x: x[1], reverse=True)
    pct = float(ctx["_cfg"].get("tau_top_percent", 10))/100.0
    k = max(1, int(len(scores)*pct))
    top = {n for n,_ in scores[:k]}
    # keyword-sensitive hits
    api_hits = {n for n,d in ctx["PDG"].nodes(data=True) if is_sensitive(d.get("api") or d.get("name"), ctx["KEYWORDS_COMPILED"])}
    seeds = top | api_hits
    ctx["centrality"] = C
    ctx["seeds"] = seeds
    ctx["api_hits"] = list(api_hits)
    return ctx

@pipeline.register("build_scg", "Slice + multi-type edge augmentation")
def build_scg(ctx):
    sliceG = backward_slice(ctx["PDG"], ctx["seeds"])
    aux = AugInputs(cfg=ctx["CFG"], ast=ctx["AST"], def_use=ctx["DEF_USE"], interproc_pairs=ctx["INTERPROC"])
    SCG = augment_edges(sliceG, aux)
    ctx["SCG"] = SCG
    etypes = {}
    for _,_,d in SCG.edges(data=True):
        etypes[str(d.get("etype"))] = etypes.get(str(d.get("etype")), 0) + 1
    ctx["scg_stats"] = {"nodes": SCG.number_of_nodes(), "edges": SCG.number_of_edges(), "etype_hist": etypes}
    return ctx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=str, default="load_data,preprocess,build_scg")
    parser.add_argument("--export", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    order = [s.strip() for s in args.run.split(",") if s.strip()]

    ctx = {"_cfg": cfg}
    # load sensitive keywords
    try:
        import json, os
        kw_path = cfg.get("sensitive_keywords_path", "configs/sensitive_keywords.json")
        if os.path.exists(kw_path):
            ctx["KEYWORDS"] = json.loads(open(kw_path, "r", encoding="utf-8").read())
        else:
            ctx["KEYWORDS"] = cfg.get("api_set", [])
    except Exception:
        ctx["KEYWORDS"] = cfg.get("api_set", [])
    ctx["KEYWORDS_COMPILED"] = compile_patterns(ctx["KEYWORDS"])
    ctx = pipeline.run(order, ctx)

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        # Convert non-serializable parts
        serial = dict(ctx)
        serial.pop("PDG", None); serial.pop("CFG", None); serial.pop("AST", None)
        serial.pop("SCG", None)  # could be exported separately as edgelist
        (args.export).write_text(json.dumps(serial, indent=2), encoding="utf-8")
        # Export SCG edgelist
        edgelist = args.export.with_suffix(".scg.edgelist")
        import networkx as nx
        nx.write_edgelist(ctx["SCG"], edgelist, data=["etype"])
    print("Done.")

if __name__ == "__main__":
    main()


from fortify_scg.hypergraph import build_hypergraph, adaptive_prune, nodes_on_hyperedges
from fortify_scg.embed import build_node_features
from fortify_scg.keywords import compile_patterns, is_sensitive

@pipeline.register("build_hypergraph", "Hyperedge construction + adaptive pruning")
def build_hg(ctx):
    # Optionally provide PATHS (list of node-id paths) for semantic co-occurrence
    paths = ctx.get("PATHS", None)
    H = build_hypergraph(ctx["SCG"], paths=paths)
    Hp = adaptive_prune(H, centrality=ctx["centrality"], thresh_scale=0.1)
    ctx["H_raw"] = H
    ctx["H_pruned"] = Hp
    kw = ctx["KEYWORDS_COMPILED"]
    api_nodes = {n for n,d in ctx["PDG"].nodes(data=True) if is_sensitive(d.get("api") or d.get("name"), kw)}
    ctx["sensitive_nodes_4_2"] = nodes_on_hyperedges(Hp) | api_nodes
    ctx["API_NODES"] = api_nodes
    return ctx

@pipeline.register("embed_nodes", "Context-aware + lightweight embeddings; build h0")
def embed_nodes(ctx):
    sens = ctx.get("sensitive_nodes_4_2", set())
    feats = build_node_features(ctx["SCG"].nodes(), sens)
    ctx["h0"] = {k: v.tolist() for k, v in feats.items()}  # JSON-friendly
    ctx["feature_dim"] = 512
    return ctx


from fortify_scg.train import train_rgcn_contrastive

@pipeline.register("build_model", "Prepare configs for RGCN+Contrastive")
def build_model(ctx):
    # Expect LABEL, NUM_CLASSES, NEG_GRAPHS, API_NODES provided by user
    # h0 must be produced by 'embed_nodes'
    ctx.setdefault("HP", {"L": 2, "hidden": [256,256], "tau": 0.5, "K": 8, "epochs": 3, "seed": 0, "gamma": 1.5})
    return ctx

@pipeline.register("train", "Run RGCN + risk-attention + contrastive learning")
def train(ctx):
    ctx = train_rgcn_contrastive(ctx)
    return ctx

@pipeline.register("evaluate", "Simple eval placeholder (reports pred probs)")
def evaluate(ctx):
    # For a proper evaluation, extend to dataset-level metrics (F1/Precision/Recall).
    return ctx


from fortify_scg.eval import precision_recall_f1_acc, save_metrics_table, plot_metrics_bar

@pipeline.register("evaluate_dataset", "Compute dataset-level metrics and export CSV/plots")
def evaluate_dataset(ctx):
    # Expect ctx["DATASET"]: list of samples, each sample must already include fields:
    #  {"SCG","h0","LABEL","NUM_CLASSES","NEG_GRAPHS","API_NODES"}
    # This step will call 'train' per-sample implicitly via train_rgcn_contrastive-like loop
    # for demonstration. For speed, epochs may be small and negatives limited.
    dataset = ctx["DATASET"]
    out_dir = ctx["_cfg"].get("out_dir", "out")
    preds = []
    for sample in dataset:
        # reuse current ctx HP but override data fields
        subctx = dict(ctx)
        subctx.update({k: sample[k] for k in ["SCG","h0","LABEL","NUM_CLASSES","NEG_GRAPHS"] if k in sample})
        subctx["API_NODES"] = set(sample.get("API_NODES", []))
        # light training
        subctx.setdefault("HP", {"epochs": 1, "K": 2})
        subctx = train_rgcn_contrastive(subctx)
        preds.append({"y_true": int(sample["LABEL"]), "y_hat": int(subctx["pred"]["label_hat"])})
    y_true = [r["y_true"] for r in preds]
    y_hat = [r["y_hat"] for r in preds]
    num_classes = int(dataset[0]["NUM_CLASSES"])
    metrics = precision_recall_f1_acc(y_true, y_hat, num_classes)
    rows = preds + [dict(metric=k, value=v) for k,v in metrics.items()]
    table_path = save_metrics_table(out_dir, rows)
    fig_path = plot_metrics_bar(out_dir, metrics)
    ctx["dataset_metrics"] = metrics
    ctx["dataset_metrics_paths"] = {"csv": table_path, "png": fig_path}
    return ctx

import json
from pathlib import Path
import networkx as nx
from fortify_scg.embed import build_node_features
from fortify_scg.keywords import compile_patterns, is_sensitive_with_switch

@pipeline.register("load_dataset_json", "Load a tiny dataset and build per-graph h0 features")
def load_dataset_json(ctx):
    # Expect ctx["_cfg"]["dataset_json"] path, and embedding switches in config
    kw = ctx.get("KEYWORDS_COMPILED", [])
    dpath = Path(ctx["_cfg"].get("dataset_json", "examples/dataset_min.json"))
    data = json.loads(dpath.read_text(encoding="utf-8"))
    num_classes = int(data["num_classes"])
    use_real = bool(ctx["_cfg"].get("use_real_embeddings", False))
    codebert_model = ctx["_cfg"].get("codebert_model", "microsoft/codebert-base")
    hf_cache_dir = ctx["_cfg"].get("hf_cache_dir", ".cache/huggingface")
    doc2vec_model_path = ctx["_cfg"].get("doc2vec_model_path", ".cache/doc2vec/doc2vec.model")

    samples = []
    # First, build all SCGs and h0
    tmp = []
    for g in data["graphs"]:
        G = nx.DiGraph()
        for n in g["nodes"]:
            G.add_node(n["id"], **n)
        for u,v,etype in g["edges"]:
            G.add_edge(u, v, etype=etype)
        api_nodes = set(g.get("api_nodes", [])) or {n["id"] for n in g["nodes"] if is_sensitive(n.get("api") or n.get("id"), kw)}
        node_texts = g.get("node_texts", {})
        # Build features (uses placeholder if real backends or texts are missing)
        feats = build_node_features_with_switch(
            G.nodes(), api_nodes,
            use_real=use_real,
            codebert_model=codebert_model,
            hf_cache_dir=hf_cache_dir,
            doc2vec_model_path=doc2vec_model_path,
            node_texts=node_texts
        )
        tmp.append({"id": g["id"], "G": G, "h0": feats, "label": int(g["label"]), "api_nodes": api_nodes})

    # Then construct NEG_GRAPHS lists (other graphs with different labels)
    samples = []
    for i, gi in enumerate(tmp):
        negs = []
        for j, gj in enumerate(tmp):
            if i == j: continue
            if gj["label"] != gi["label"]:
                negs.append({"SCG": gj["G"], "h0": gj["h0"], "LABEL": gj["label"]})
        samples.append({
            "SCG": gi["G"], "h0": gi["h0"], "LABEL": gi["label"],
            "NUM_CLASSES": num_classes, "NEG_GRAPHS": negs, "API_NODES": gi["api_nodes"]
        })

    ctx["DATASET"] = samples
    ctx["NUM_CLASSES"] = num_classes
    return ctx
