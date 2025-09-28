import json, networkx as nx
from pathlib import Path
from fortify_scg.scg import composite_centrality, select_sensitive_nodes, backward_slice, augment_edges, AugInputs

def test_scg_minimal(tmp_path: Path):
    nodes = [
        {"id":"n1","api":"","bb":"b1","func":"f","is_input_boundary": True},
        {"id":"n2","api":"strcpy","bb":"b1","func":"f","is_input_boundary": False},
        {"id":"n3","api":"","bb":"b1","func":"f","is_input_boundary": False},
    ]
    PDG = nx.DiGraph()
    for n in nodes: PDG.add_node(n["id"], **n)
    PDG.add_edges_from([("n1","n2"),("n3","n2")])
    C = composite_centrality(PDG)
    seeds = select_sensitive_nodes(PDG, C, 0.10, {"strcpy"})
    assert "n2" in seeds
    sliceG = backward_slice(PDG, seeds)
    assert "n1" in sliceG and "n2" in sliceG
    aux = AugInputs(cfg=PDG, ast=PDG, def_use={"n2":{"n3"}}, interproc_pairs=set())
    SCG = augment_edges(sliceG, aux)
    assert SCG.number_of_nodes() >= 2
