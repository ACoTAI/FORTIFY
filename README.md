# FORTIFY-SCG: Reproducible SCG Construction & Pipeline

This repository accompanies the paper's Section 4 (FORTIFY model) and provides a **reproducible** implementation of **4.1 Sliced Combined Graph (SCG) Construction** plus an extensible pipeline for subsequent phases (hypergraph embedding, RGCN, risk-sensitive attention, and contrastive learning).

## Features
- Composite centrality (Eq. 3) with default weights `0.5/0.3/0.2`.
- Sensitive node selection (Top-10% + sensitive API tags).
- Backward slicing on PDG until input boundaries.
- Multi-type edge augmentation (Cascading, Local Degree, Global Degree, Structural).
- Clean CLI and config with `pyproject.toml` packaging.
- Unit tests for core SCG behaviors.

## Install
```bash
pip install -e .
```

## Quickstart
Prepare your PDG/CFG/AST and metadata (see `examples/minipdg.json`) and run:
```bash
python scripts/idea_pipeline.py --config configs/default.yaml --run load_data,preprocess,build_scg --export out/context.json
```

## Config
See `configs/default.yaml` for entry keys expected in the runtime context (PDG/CFG/AST...).

## Cite
See `CITATION.cff`.


## 4.2 Hypergraph & Embeddings
To run hypergraph construction (Eq. 4–5) and build initial node features (Eq. 6–8):
```bash
python scripts/idea_pipeline.py --config configs/default.yaml   --run load_data,preprocess,build_scg,build_hypergraph,embed_nodes   --export out/context_4_2.json
```
The output includes:
- `H_pruned` hyperedges count and retained nodes,
- `h0` initial 512-d features per node (CodeBERT placeholder + Doc2Vec placeholder).


## 4.3 RGCN + Risk-Sensitive Attention + Contrastive Learning
**Expected runtime data** (you provide these before `build_model`/`train`):
- `LABEL`: int (ground-truth class id of current graph)
- `NUM_CLASSES`: int (e.g., 2 for vulnerable vs. non)
- `NEG_GRAPHS`: list of dicts, each with keys `SCG`, `h0`, `LABEL` (used as negatives if label != `LABEL`)
- `API_NODES`: set of node ids considered sensitive (risk boost)

Run end-to-end (demo with your own negatives and labels):
```bash
python scripts/idea_pipeline.py --config configs/default.yaml   --run load_data,preprocess,build_scg,build_hypergraph,embed_nodes,build_model,train,evaluate   --export out/context_4_3.json
```
Outputs:
- `train_log` with CE/InfoNCE losses, positive similarity, and lambda schedule per epoch.
- `pred` with predicted probabilities for the current graph.


## Real Embeddings (CodeBERT/Doc2Vec)
Toggle via config:
```yaml
use_real_embeddings: true
codebert_model: microsoft/codebert-base
hf_cache_dir: .cache/huggingface
doc2vec_model_path: .cache/doc2vec/doc2vec.model
```
Pass `node_texts` (mapping node_id->code snippet) if you want textual context; otherwise the code falls back to placeholders for missing text or missing backends.

## Dataset-level Evaluation
Run an end-to-end dataset evaluation (expects `DATASET` injected in ctx):
```bash
python scripts/idea_pipeline.py --config configs/default.yaml   --run load_data,preprocess,build_scg,build_hypergraph,embed_nodes,build_model,train,evaluate,evaluate_dataset   --export out/context_full.json
```
This produces `out/metrics.csv` and `out/metrics.png`.

## Continuous Integration
A GitHub Actions workflow at `.github/workflows/ci.yml` runs `pytest` on push/PR to ensure reproducibility.


## Minimal Dataset Quickstart
We provide a tiny dataset at `examples/dataset_min.json` (3 toy graphs, 2 classes). Run:
```bash
python scripts/idea_pipeline.py --config configs/default.yaml   --run load_dataset_json,build_model,evaluate_dataset   --export out/dataset_eval.json
```
This will compute features per graph (using real embeddings if enabled, else fallbacks), build negatives automatically, and export `out/metrics.csv` & `out/metrics.png`.


## Sensitive Keywords Integration
- Provide sensitive API/function names via `configs/sensitive_keywords.json` (already prefilled with your list, supports `*` wildcards).
- The pipeline compiles these into regex and uses them to:
  1) Select sensitive seeds in **preprocess** (4.1.1), union with top-10% centrality nodes.
  2) Expand `sensitive_nodes_4_2` in **build_hypergraph** (4.2), also populating `API_NODES` for risk attention β_ij.
  3) Auto-mark API nodes in **load_dataset_json** when `api_nodes` not explicitly given.
- You can override the path in config:  
  `sensitive_keywords_path: configs/sensitive_keywords.json`


## Experiments (Sec. 5.5)
Run with the tiny dataset (you can replace with your own). Examples:

### 5.5.1 SCG Edge-Type Ablation & τ Sweep
```bash
python scripts/experiments.py --config configs/default.yaml --prep --edge-ablation
python scripts/experiments.py --config configs/default.yaml --tau-sweep 5 10 20
```

### 5.5.2 Runtime & Resource
```bash
python scripts/experiments.py --config configs/default.yaml --runtime
```

### 5.5.3 Embedding/Hypergraph Strategy Variants
```bash
python scripts/experiments.py --config configs/default.yaml --embedding-variants
```

### 5.5.4 Robustness (k-fold)
```bash
python scripts/experiments.py --config configs/default.yaml --kfold 5
```

### 5.5.6 RGCN & Contrastive Ablation
```bash
python scripts/experiments.py --config configs/default.yaml --rgcn-abl
```
All results will be saved under `out/` as CSVs (and some PNG plots).
