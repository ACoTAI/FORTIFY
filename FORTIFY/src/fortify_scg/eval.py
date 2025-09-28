"""
Dataset-level evaluation utilities: compute Precision/Recall/F1/Accuracy (macro & micro),
save CSV of per-graph predictions and aggregate metrics, and draw a simple bar chart.
"""
from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    M = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[int(t), int(p)] += 1
    return M

def precision_recall_f1_acc(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    M = confusion_matrix(y_true, y_pred, num_classes)
    # per-class precision/recall/f1
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = M[c, c]
        fp = M[:, c].sum() - tp
        fn = M[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    acc = (np.trace(M) / (M.sum() + 1e-12))
    metrics = {
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "accuracy": float(acc)
    }
    return metrics

def save_metrics_table(out_dir: str, rows: List[Dict]):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.csv")
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("empty\n")
        return path
    fields = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

def plot_metrics_bar(out_dir: str, metrics: Dict[str, float]):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.png")
    keys = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
    vals = [metrics[k] for k in keys]
    plt.figure()
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Dataset Metrics")
    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight")
    plt.close()
    return path
