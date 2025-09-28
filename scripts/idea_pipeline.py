#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import csv

import networkx as nx
import yaml

# ---- fortify_scg imports (with graceful fallbacks) ----
from fortify_scg.pipeline import Pipeline
from fortify_scg.scg import (
    composite_centrality,
    select_sensitive_nodes,   # may be unused in this script but kept for compat
    backward_slice,
    augment_edges,
    AugInputs,
)
from fortify_scg.keywords import compile_patterns, is_sensitive

# hypergraph / embedding steps are optional; we import with fallbacks
try:
    from fortify_scg.hypergraph import build_hypergraph, adaptive_prune, nodes_on_hyperedges
except Exception:  # pragma: no cover
    build_hypergraph = adaptive_prune = nodes_on_hyperedges = None

try:
    from fortify_scg.embed import build_node_features_with_switch
except Exception:  # pragma: no cover
    try:
        # Fallback to basic features if the switch version is not provided
        from fortify_scg.embed import build_node_features as build_node_features_with_switch
    except Exception:
        build_node_features_with_switch = None

# Optional training/eval (not needed for the 3-step demo but kept for completeness)
try:
    from fortify_scg.train import train_rgcn_contrastive
except Exception:  # pragma: no cover
    train_rgcn_contrastive = None

try:
    from fortify_scg.eval import precision_recall_f1_acc, save_metrics_table, plot_metrics_bar
except Exception:  # pragma: no cover
    precision_recall_f1_acc = save_metrics_table = plot_metrics_bar = None

pipeline = Pipeline()


# ----------------------------
# Pipeline steps (core 3 steps)
# ----------------------------
@pipeline.register("load_data", "Load PDG/CFG/AST from JSON")
def load_data(ctx: dict) -> dict:
    cfg_path = Path(ctx["_cfg"]["inputs"]["pdg_json"])
    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Build PDG/CFG/AST
    PDG = nx.DiGraph()
    for n in data["nodes"]:
        PDG.add_node(n["id"], **n)
    for u, v in data["pdg_edges"]:
        PDG.add_edge(u, v)

    CFG = nx.DiGraph()
    CFG.add_edges_from(data.get("cfg_edges", []))

    AST = nx.DiGraph()
    AST.add_edges_from(data.get("ast_edges", []))

    DEF_USE = {k: set(v) for k, v in data.get("def_use", {}).items()}
    INTERPROC = set(tuple(x) for x in data.get("interproc_pairs", []))

    ctx["PDG"] = PDG
    ctx["CFG"] = CFG
    ctx["AST"] = AST
    ctx["DEF_USE"] = DEF_USE
    ctx["INTERPROC"] = INTERPROC
    return ctx


@pipeline.register("preprocess", "Centrality + seed selection")
def preprocess(ctx: dict) -> dict:
    # centrality weights: default 0.5/0.3/0.2; allow override via config
    w = ctx["_cfg"].get("centrality_weights", [0.5, 0.3, 0.2])
    dc, cc, bc = (float(w[0]), float(w[1]), float(w[2])) if isinstance(w, (list, tuple)) and len(w) == 3 else (0.5, 0.3, 0.2)

    C = composite_centrality(ctx["PDG"], dc, cc, bc)

    scores = sorted(((n, C[n]) for n in ctx["PDG"].nodes), key=lambda x: x[1], reverse=True)
    pct = float(ctx["_cfg"].get("tau_top_percent", 10)) / 100.0
    k = max(1, int(len(scores) * pct))
    top = {n for n, _ in scores[:k]}

    # keyword-sensitive hits
    api_hits = {
        n
        for n, d in ctx["PDG"].nodes(data=True)
        if is_sensitive(d.get("api") or d.get("name"), ctx["KEYWORDS_COMPILED"])
    }

    seeds = top | api_hits
    ctx["centrality"] = C
    ctx["seeds"] = seeds
    ctx["api_hits"] = list(api_hits)  # JSON-friendly
    return ctx


@pipeline.register("build_scg", "Slice + multi-type edge augmentation")
def build_scg(ctx: dict) -> dict:
    sliceG = backward_slice(ctx["PDG"], ctx["seeds"])
    aux = AugInputs(
        cfg=ctx["CFG"],
        ast=ctx["AST"],
        def_use=ctx["DEF_USE"],
        interproc_pairs=ctx["INTERPROC"],
    )
    SCG = augment_edges(sliceG, aux)
    ctx["SCG"] = SCG

    etypes = {}
    for _, _, d in SCG.edges(data=True):
        et = str(d.get("etype"))
        etypes[et] = etypes.get(et, 0) + 1

    ctx["scg_stats"] = {"nodes": SCG.number_of_nodes(), "edges": SCG.number_of_edges(), "etype_hist": etypes}
    return ctx


# ---------------------------------------------
# Optional extra steps (safe if deps are absent)
# ---------------------------------------------
if build_hypergraph is not None:

    @pipeline.register("build_hypergraph", "Hyperedge construction + adaptive pruning")
    def build_hg(ctx: dict) -> dict:
        paths = ctx.get("PATHS", None)
        H = build_hypergraph(ctx["SCG"], paths=paths)
        Hp = adaptive_prune(H, centrality=ctx["centrality"], thresh_scale=0.1)
        ctx["H_raw"] = H
        ctx["H_pruned"] = Hp

        kw = ctx["KEYWORDS_COMPILED"]
        api_nodes = {
            n
            for n, d in ctx["PDG"].nodes(data=True)
            if is_sensitive(d.get("api") or d.get("name"), kw)
        }
        ctx["sensitive_nodes_4_2"] = nodes_on_hyperedges(Hp) | api_nodes
        ctx["API_NODES"] = api_nodes
        return ctx

if build_node_features_with_switch is not None:

    @pipeline.register("embed_nodes", "Context-aware + lightweight embeddings; build h0")
    def embed_nodes(ctx: dict) -> dict:
        sens = ctx.get("sensitive_nodes_4_2", set())
        feats = build_node_features_with_switch(
            ctx["SCG"].nodes(),
            sens,
            use_real=bool(ctx["_cfg"].get("use_real_embeddings", False)),
            codebert_model=ctx["_cfg"].get("codebert_model", "microsoft/codebert-base"),
            hf_cache_dir=ctx["_cfg"].get("hf_cache_dir", ".cache/huggingface"),
            doc2vec_model_path=ctx["_cfg"].get("doc2vec_model_path", ".cache/doc2vec/doc2vec.model"),
            node_texts={},  # provide per-node texts here if available
        )
        # JSON-friendly
        ctx["h0"] = {k: (v.tolist() if hasattr(v, "tolist") else list(v)) for k, v in feats.items()}
        ctx["feature_dim"] = len(next(iter(ctx["h0"].values()))) if ctx["h0"] else 0
        return ctx

if train_rgcn_contrastive is not None:

    @pipeline.register("build_model", "Prepare configs for RGCN+Contrastive")
    def build_model(ctx: dict) -> dict:
        ctx.setdefault("HP", {"L": 2, "hidden": [256, 256], "tau": 0.5, "K": 8, "epochs": 3, "seed": 0, "gamma": 1.5})
        return ctx

    @pipeline.register("train", "Run RGCN + risk-attention + contrastive learning")
    def train(ctx: dict) -> dict:
        return train_rgcn_contrastive(ctx)

    @pipeline.register("evaluate", "Simple eval placeholder (reports pred probs)")
    def evaluate(ctx: dict) -> dict:
        return ctx

if precision_recall_f1_acc is not None and train_rgcn_contrastive is not None:

    @pipeline.register("evaluate_dataset", "Compute dataset-level metrics and export CSV/plots")
    def evaluate_dataset(ctx: dict) -> dict:
        """
        Evaluate on ctx['DATASET'].
        Writes two files under out_dir:
          - preds.csv     (columns: y_true,y_hat)
          - metrics.csv   (columns: metric,value)
        Optionally plots a metrics bar if plot helper is available.
        """
        dataset = ctx["DATASET"]
        out_dir = Path(ctx["_cfg"].get("out_dir", "out"))
        out_dir.mkdir(parents=True, exist_ok=True)

        preds = []
        for sample in dataset:
            subctx = dict(ctx)
            subctx.update({k: sample[k] for k in ["SCG", "h0", "LABEL", "NUM_CLASSES", "NEG_GRAPHS"] if k in sample})
            subctx["API_NODES"] = set(sample.get("API_NODES", []))
            subctx.setdefault("HP", {"epochs": 1, "K": 2})
            subctx = train_rgcn_contrastive(subctx)
            preds.append({"y_true": int(sample["LABEL"]), "y_hat": int(subctx["pred"]["label_hat"])})

        y_true = [r["y_true"] for r in preds]
        y_hat = [r["y_hat"] for r in preds]
        num_classes = int(dataset[0]["NUM_CLASSES"])
        metrics = precision_recall_f1_acc(y_true, y_hat, num_classes)

        # --- Write preds.csv (homogeneous columns) ---
        preds_csv = out_dir / "preds.csv"
        with preds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["y_true", "y_hat"])
            w.writeheader()
            w.writerows(preds)

        # --- Write metrics.csv (homogeneous columns) ---
        metrics_csv = out_dir / "metrics.csv"
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in metrics.items():
                w.writerow([k, v])

        # Optional: plot metrics bar if available
        fig_path = None
        if plot_metrics_bar:
            fig_path = plot_metrics_bar(str(out_dir), metrics)

        ctx["dataset_metrics"] = metrics
        ctx["dataset_metrics_paths"] = {
            "preds_csv": str(preds_csv),
            "metrics_csv": str(metrics_csv),
            "metrics_plot": fig_path,
        }
        return ctx


# ----------------------------
# Dataset loader (MUST be before main)
# ----------------------------
@pipeline.register("load_dataset_json", "Load a tiny dataset and build per-graph h0 features")
def load_dataset_json(ctx: dict) -> dict:
    # Read dataset file
    dpath = Path(ctx["_cfg"].get("dataset_json", "examples/dataset_min.json"))
    if not dpath.exists():
        return ctx

    data = json.loads(dpath.read_text(encoding="utf-8"))
    num_classes = int(data["num_classes"])

    # switches
    use_real = bool(ctx["_cfg"].get("use_real_embeddings", False))
    codebert_model = ctx["_cfg"].get("codebert_model", "microsoft/codebert-base")
    hf_cache_dir = ctx["_cfg"].get("hf_cache_dir", ".cache/huggingface")
    doc2vec_model_path = ctx["_cfg"].get("doc2vec_model_path", ".cache/doc2vec/doc2vec.model")

    kw = ctx.get("KEYWORDS_COMPILED", [])
    tmp = []
    for g in data["graphs"]:
        G = nx.DiGraph()
        for n in g["nodes"]:
            G.add_node(n["id"], **n)
        for u, v, etype in g["edges"]:
            G.add_edge(u, v, etype=etype)

        api_nodes = set(g.get("api_nodes", [])) or {
            n["id"] for n in g["nodes"] if is_sensitive(n.get("api") or n.get("id"), kw)
        }
        node_texts = g.get("node_texts", {})

        if build_node_features_with_switch is None:
            feats = {n: [0.0] * 16 for n in G.nodes()}  # fallback feature
        else:
            feats = build_node_features_with_switch(
                G.nodes(), api_nodes,
                use_real=use_real,
                codebert_model=codebert_model,
                hf_cache_dir=hf_cache_dir,
                doc2vec_model_path=doc2vec_model_path,
                node_texts=node_texts
            )
        tmp.append({"id": g["id"], "G": G, "h0": feats, "label": int(g["label"]), "api_nodes": api_nodes})

    # Build samples with negatives of opposite labels
    samples = []
    for i, gi in enumerate(tmp):
        negs = []
        for j, gj in enumerate(tmp):
            if i == j:
                continue
            if gj["label"] != gi["label"]:
                negs.append({"SCG": gj["G"], "h0": gj["h0"], "LABEL": gj["label"]})
        samples.append({
            "SCG": gi["G"], "h0": gi["h0"], "LABEL": gi["label"],
            "NUM_CLASSES": num_classes, "NEG_GRAPHS": negs, "API_NODES": gi["api_nodes"]
        })

    ctx["DATASET"] = samples
    ctx["NUM_CLASSES"] = num_classes
    return ctx


# ----------------------------
# Helpers
# ----------------------------
def _load_keywords(cfg: dict) -> list[str] | dict:
    """Load sensitive keywords from json file or api_set in config."""
    kw_path = cfg.get("sensitive_keywords_path", "configs/sensitive_keywords.json")
    try:
        if kw_path and os.path.exists(kw_path):
            with open(kw_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return cfg.get("api_set", [])


def _clean_for_json(obj):
    """Best-effort conversion to make ctx JSON-serializable."""
    # Handle common problematic types quickly
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, re.Pattern):
        return obj.pattern
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    # NetworkX graphs: skip/convert externally
    if isinstance(obj, (nx.Graph, nx.DiGraph)):
        return f"<Graph|V={obj.number_of_nodes()} E={obj.number_of_edges()}>"
    # Fallback to string
    return str(obj)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=str, default="load_data,preprocess,build_scg")
    parser.add_argument("--export", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    order = [s.strip() for s in args.run.split(",") if s.strip()]

    # Build ctx
    ctx: dict = {"_cfg": cfg}

    # Load & compile keywords
    ctx["KEYWORDS"] = _load_keywords(cfg)
    ctx["KEYWORDS_COMPILED"] = compile_patterns(ctx["KEYWORDS"])

    # Run pipeline
    ctx = pipeline.run(order, ctx)

    # Export (JSON + optional SCG edgelist)
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        serial = dict(ctx)
        # Remove heavy/non-serializable graphs
        for k in ("PDG", "CFG", "AST", "SCG", "H_raw", "H_pruned"):
            serial.pop(k, None)
        # Remove compiled regex; keep raw KEYWORDS
        serial.pop("KEYWORDS_COMPILED", None)

        # Make the remainder JSON-friendly
        serial = _clean_for_json(serial)
        export_path.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")

        # If SCG exists, also export an edgelist alongside JSON (with etype)
        if "SCG" in ctx:
            edgelist = export_path.with_suffix(".scg.edgelist")
            nx.write_edgelist(ctx["SCG"], edgelist, data=["etype"])

    print("Done.")


if __name__ == "__main__":
    main()
