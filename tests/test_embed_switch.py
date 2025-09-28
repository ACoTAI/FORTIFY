import numpy as np
from fortify_scg.embed import build_node_features_with_switch

def test_embed_switch_fallback():
    nodes = ["a","b"]
    sens = {"a"}
    feats = build_node_features_with_switch(nodes, sens, use_real=True, node_texts={"a":"int strcpy(char* d, char* s);", "b":"int x=0;"})
    assert feats["a"].shape[0] == 512 and feats["b"].shape[0] == 512
