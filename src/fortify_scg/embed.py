from __future__ import annotations
from typing import Dict, Iterable, Set, Tuple
import hashlib
import numpy as np

def _rng_from_key(key: str) -> np.random.Generator:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n

def codebert_embed(node_id: str, window_tokens: int = 128, raw_text: str | None = None) -> np.ndarray:
    """
    Placeholder for CodeBERT. We produce a deterministic 768-d vector using a RNG seeded by node_id.
    (In a full environment, replace with actual transformer inference.)
    """
    rng = _rng_from_key(f"codebert::{node_id}")
    return rng.standard_normal(768)

def doc2vec_embed(node_id: str, raw_text: str | None = None) -> np.ndarray:
    """
    Placeholder for Doc2Vec. Deterministic 256-d vector seeded by node_id.
    (In a full environment, replace with gensim Doc2Vec inference.)
    """
    rng = _rng_from_key(f"doc2vec::{node_id}")
    return rng.standard_normal(256)

def linear_project(x: np.ndarray, out_dim: int, key: str) -> np.ndarray:
    # Deterministic random projection matrix based on key
    rng = _rng_from_key(f"proj::{key}::{x.shape[0]}->{out_dim}")
    W = rng.standard_normal((out_dim, x.shape[0])) / np.sqrt(x.shape[0])
    return W @ x

def build_node_features(nodes: Iterable[str],
                        sensitive_nodes: Set[str],
                        zero_mask_key: str = "zero_mask") -> Dict[str, np.ndarray]:
    """
    Build h_i^(0) = concat( z_api_i , z_var_i ), where:
      - z_api_i: 768-d CodeBERT -> projected to 256, normalized; use learned zero mask if missing
      - z_var_i: 256-d Doc2Vec -> normalized
      - final feature: 512-d
    For nodes not in sensitive_nodes, z_api_i is replaced by a deterministic 'zero mask' vector.
    """
    feats: Dict[str, np.ndarray] = {}
    zero_mask = linear_project(np.zeros(768), 256, key=zero_mask_key)  # results all zeros
    for nid in nodes:
        if nid in sensitive_nodes:
            z_api = codebert_embed(nid)  # 768
            z_api = linear_project(z_api, 256, key="api_proj")
            z_api = l2norm(z_api)
        else:
            z_api = zero_mask

        z_var = doc2vec_embed(nid)  # 256
        z_var = l2norm(z_var)

        h0 = np.concatenate([z_api, z_var], axis=0)  # 512
        feats[nid] = h0.astype(np.float32)
    return feats


# === Real model backends (optional) ===
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

try:
    from gensim.models.doc2vec import Doc2Vec
    _GENSIM_AVAILABLE = True
except Exception:
    _GENSIM_AVAILABLE = False

_CODEBERT_CACHE = {}
_DOC2VEC_CACHE = {}

def _load_codebert(model_name: str, cache_dir: str):
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available")
    key = (model_name, cache_dir)
    if key not in _CODEBERT_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        mdl = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        mdl.eval()
        _CODEBERT_CACHE[key] = (tok, mdl)
    return _CODEBERT_CACHE[key]

def _embed_codebert_text(text: str, model_name: str, cache_dir: str) -> np.ndarray:
    tok, mdl = _load_codebert(model_name, cache_dir)
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        out = mdl(**enc).last_hidden_state  # [1, T, 768]
        z = out.mean(dim=1).squeeze(0).cpu().numpy()  # 768
    return z

def _load_doc2vec(path: str):
    if not _GENSIM_AVAILABLE:
        raise RuntimeError("gensim not available")
    if path not in _DOC2VEC_CACHE:
        _DOC2VEC_CACHE[path] = Doc2Vec.load(path)
    return _DOC2VEC_CACHE[path]

def _embed_doc2vec_text(text: str, path: str) -> np.ndarray:
    mdl = _load_doc2vec(path)
    return mdl.infer_vector(text.split())  # 256 by your training config; else will be whatever the model provides

def build_node_features_with_switch(
    nodes: Iterable[str],
    sensitive_nodes: Set[str],
    use_real: bool = False,
    codebert_model: str = "microsoft/codebert-base",
    hf_cache_dir: str = ".cache/huggingface",
    doc2vec_model_path: str = ".cache/doc2vec/doc2vec.model",
    node_texts: Dict[str, str] | None = None,
    zero_mask_key: str = "zero_mask",
) -> Dict[str, np.ndarray]:
    """
    When `use_real` is True, try to produce embeddings with real backends:
      - CodeBERT(mean last layer) for sensitive nodes using node_texts[nid]
      - Doc2Vec for all nodes using node_texts[nid]
    If backends are unavailable or text is missing, fall back to deterministic placeholders.
    """
    feats: Dict[str, np.ndarray] = {}
    zero_mask = linear_project(np.zeros(768), 256, key=zero_mask_key)

    for nid in nodes:
        # API channel
        if nid in sensitive_nodes and use_real and node_texts and nid in node_texts:
            try:
                z_api = _embed_codebert_text(node_texts[nid], model_name=codebert_model, cache_dir=hf_cache_dir)
                z_api = linear_project(z_api, 256, key="api_proj")
                z_api = l2norm(z_api)
            except Exception:
                z_api = linear_project(codebert_embed(nid), 256, key="api_proj")
                z_api = l2norm(z_api)
        else:
            z_api = zero_mask

        # Var channel
        if use_real and node_texts and nid in node_texts:
            try:
                z_var = _embed_doc2vec_text(node_texts[nid], doc2vec_model_path)
                # Project to 256 only if model's vector size != 256
                if z_var.shape[0] != 256:
                    z_var = linear_project(z_var, 256, key="var_proj")
                z_var = l2norm(z_var)
            except Exception:
                z_var = l2norm(doc2vec_embed(nid))
        else:
            z_var = l2norm(doc2vec_embed(nid))

        feats[nid] = np.concatenate([z_api, z_var], axis=0).astype(np.float32)
    return feats
