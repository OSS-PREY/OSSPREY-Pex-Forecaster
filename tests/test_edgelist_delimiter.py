"""Regression test: a '#' in a node label must not corrupt the edgelist.

Edgelists use "##" as their delimiter. A contributor literally named "Ran#"
produced "Ran###Ran###0", which parses as ["Ran", "#Ran", "#0"] -- networkx then
raised "Failed to convert weight data #0 to type <class 'int'>" and killed the
whole forecast for gohugoio/hugo.
"""
import networkx as nx
import pytest

from decalfc.pipeline.create_networks import _safe_labels, EDGELIST_DELIMITER
from decalfc.pipeline.network_features import _edge_weight


def test_hash_in_node_label_survives_a_write_read_round_trip(tmp_path):
    g = nx.DiGraph()
    g.add_edge("Ran#", "some/file.go", weight=3)

    path = tmp_path / "proj__0.edgelist"
    nx.write_edgelist(_safe_labels(g), path, delimiter=EDGELIST_DELIMITER, data=["weight"])

    # Every line must split into exactly 3 fields with an integer weight.
    for line in path.read_text().splitlines():
        parts = line.split(EDGELIST_DELIMITER)
        assert len(parts) == 3, f"delimiter collision in {line!r}"
        assert int(parts[2]) == 3

    # comments="*" matches the production readers: the default comment char is
    # "#", which would truncate every line at the "##" delimiter itself.
    back = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=str,
                            comments="*", delimiter=EDGELIST_DELIMITER,
                            data=(("weight", _edge_weight),))
    assert back.number_of_edges() == 1
    assert back["Ran%23"]["some/file.go"]["weight"] == 3


def test_labels_without_hash_are_left_alone(tmp_path):
    g = nx.DiGraph()
    g.add_edge("alice", "main.go", weight=1)
    assert set(_safe_labels(g).nodes()) == {"alice", "main.go"}


def test_edge_weight_never_raises():
    assert _edge_weight("4") == 4
    assert _edge_weight("#0") == 0          # legacy corrupt file
    assert _edge_weight("nonsense") == 1    # degrades, does not kill the run
