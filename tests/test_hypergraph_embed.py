import networkx as nx
from fortify_scg.hypergraph import build_hypergraph, adaptive_prune, nodes_on_hyperedges
from fortify_scg.embed import build_node_features

def test_hypergraph_and_embed():
    G = nx.DiGraph()
    G.add_nodes_from(["a","b","c","d"])
    G.add_edges_from([("a","b"),("b","c"),("c","d")])
    H = build_hypergraph(G, paths=[["a","b","c","d"]])
    assert len(H.hyperedges) >= 2
    Cp = {n: 1.0 for n in G.nodes}
    Hp = adaptive_prune(H, Cp)
    sens = nodes_on_hyperedges(Hp)
    feats = build_node_features(G.nodes(), sens)
    assert all(v.shape[0] == 512 for v in feats.values())
