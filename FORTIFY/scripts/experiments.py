#!/usr/bin/env python
"""
Command-line entry for Section 5.5 experiments.
Examples:
  # Prepare dataset from JSON
  python scripts/experiments.py --config configs/default.yaml --prep

  # 5.5.1 Edge ablation + tau sweep
  python scripts/experiments.py --config configs/default.yaml --edge-ablation
  python scripts/experiments.py --config configs/default.yaml --tau-sweep 5 10 20

  # 5.5.2 Runtime scaling (synthetic)
  python scripts/experiments.py --config configs/default.yaml --runtime

  # 5.5.3 Embedding variants
  python scripts/experiments.py --config configs/default.yaml --embedding-variants

  # 5.5.4 Robustness (k-fold)
  python scripts/experiments.py --config configs/default.yaml --kfold 5

  # 5.5.6 RGCN/Contrastive ablation
  python scripts/experiments.py --config configs/default.yaml --rgcn-abl
"""
from __future__ import annotations
import argparse
import yaml
import sys
import os
from pathlib import Path
import json
import networkx as nx

from fortify_scg.experiments import (
    scg_edge_ablation,
    tau_sweep,
    runtime_scaling,
    hypergraph_embedding_variants,
    kfold,
    rgcn_contrastive_ablation,
)
from fortify_scg.eval import plot_metrics_bar
from fortify_scg.pipeline import Pipeline
from fortify_scg.keywords import compile_patterns, is_sensitive

# ---- embed import with fallback (so minimal env still runs) ----
try:
    from fortify_scg.embed import build_node_features_with_switch
except Exception:
    try:
        from fortify_scg.embed import build_node_features as build_node_features_with_switch  # type: ignore
    except Exception:
        build_node_features_with_switch = None  # type: ignore


def _load_dataset(cfg):
    """Mirror scripts/idea_pipeline.py::load_dataset_json behavior (standalone)."""
    dpath = Path(cfg.get("dataset_json", "examples/dataset_min.json"))
    data = json.loads(dpath.read_text(encoding="utf-8"))
    num_classes = int(data["num_classes"])

    use_real = bool(cfg.get("use_real_embeddings", False))
    codebert_model = cfg.get("codebert_model", "microsoft/codebert-base")
    hf_cache_dir = cfg.get("hf_cache_dir", ".cache/huggingface")
    doc2vec_model_path = cfg.get("doc2vec_model_path", ".cache/doc2vec/doc2vec.model")

    # keywords
    kw_path = cfg.get("sensitive_keywords_path", "configs/sensitive_keywords.json")
    try:
        kw = json.loads(Path(kw_path).read_text(encoding="utf-8"))
    except Exception:
        kw = cfg.get("api_set", [])
    rgx = compile_patterns(kw)

    samples = []
    tmp = []
    for g in data["graphs"]:
        G = nx.DiGraph()
        for n in g["nodes"]:
            G.add_node(n["id"], **n)
        for u, v, etype in g["edges"]:
            G.add_edge(u, v, etype=etype)

        api_nodes = set(g.get("api_nodes", [])) or {
            n["id"] for n in g["nodes"] if is_sensitive(n.get("api") or n.get("id"), rgx)
        }
        node_texts = g.get("node_texts", {})

        if build_node_features_with_switch is None:
            feats = {n: [0.0] * 16 for n in G.nodes()}  # fallback features
        else:
            feats = build_node_features_with_switch(
                G.nodes(),
                api_nodes,
                use_real=use_real,
                codebert_model=codebert_model,
                hf_cache_dir=hf_cache_dir,
                doc2vec_model_path=doc2vec_model_path,
                node_texts=node_texts,
            )
        tmp.append({"id": g["id"], "G": G, "h0": feats, "label": int(g["label"]), "api_nodes": api_nodes})

    for i, gi in enumerate(tmp):
        negs = []
        for j, gj in enumerate(tmp):
            if i == j:
                continue
            if gj["label"] != gi["label"]:
                negs.append({"SCG": gj["G"], "h0": gj["h0"], "LABEL": gj["label"]})
        samples.append(
            {
                "SCG": gi["G"],
                "h0": gi["h0"],
                "LABEL": gi["label"],
                "NUM_CLASSES": num_classes,
                "NEG_GRAPHS": negs,
                "API_NODES": gi["api_nodes"],
            }
        )
    return samples, num_classes


def _dataset_loader_via_idea_pipeline(ctx):
    """
    Loader used by tau_sweep: safely import load_dataset_json whether we run
    as `python scripts/experiments.py` or `python -m scripts.experiments`.
    """
    try:
        # when run as a module: project root on sys.path, scripts is a package
        from scripts.idea_pipeline import load_dataset_json as ldz  # type: ignore
    except ModuleNotFoundError:
        # when run directly: cwd/scripts is sys.path[0], so import sibling
        from idea_pipeline import load_dataset_json as ldz  # type: ignore
    return ldz(ctx)["DATASET"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--edge-ablation", action="store_true")
    ap.add_argument("--tau-sweep", nargs="*", type=int)
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--embedding-variants", action="store_true")
    ap.add_argument("--kfold", type=int)
    ap.add_argument("--rgcn-abl", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8").read())
    out_dir = cfg.get("out_dir", "out")
    os.makedirs(out_dir, exist_ok=True)

    if args.prep or any([args.edge_ablation, args.tau_sweep is not None, args.embedding_variants, args.kfold, args.rgcn_abl]):
        dataset, num_classes = _load_dataset(cfg)

    if args.edge_ablation:
        res = scg_edge_ablation(dataset, out_dir)
        print("Edge ablation:", res)

    if args.tau_sweep is not None:
        # Provide a loader that is re-evaluated per tau (uses idea_pipeline's loader).
        def loader(ctx):
            return _dataset_loader_via_idea_pipeline(ctx)

        taus = [int(x) for x in args.tau_sweep] if args.tau_sweep else (5, 10, 20)
        tau_sweep(loader, cfg, out_dir, taus=taus)
        print("Tau sweep done.")

    if args.runtime:
        runtime_scaling(out_dir)
        print("Runtime scaling done.")

    if args.embedding_variants:
        hypergraph_embedding_variants(dataset, out_dir)
        print("Embedding variants done.")

    if args.kfold:
        kfold(dataset, args.kfold, out_dir)
        print(f"{args.kfold}-fold results done.")

    if args.rgcn_abl:
        rgcn_contrastive_ablation(dataset, out_dir)
        print("RGCN/contrastive ablation done.")


if __name__ == "__main__":
    main()

