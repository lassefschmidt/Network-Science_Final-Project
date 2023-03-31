# import external packages
import networkx as nx # graph data
import numpy as np
import pandas as pd

def load_raw():
    """
    helper function that performs all data loading
    """
    # read node features
    node_info = (pd.read_csv("data/cora.content", sep = "\t", index_col = 0, header = None)
        .rename(columns = {1434: "label"})
        .rename_axis("node")
        .sort_index()
    )

    # read edgelist
    edgelist = pd.read_csv("data/cora.cites", sep = "\t", header = None).rename(columns = {0: "target", 1: "source"})

    # reindex all nodes to go from 0, ..., number of nodes
    node_idx_mapping = {old: new for new, old in node_info.reset_index()["node"].items()}

    node_info = (node_info
        .reset_index()
        .assign(node = lambda df_: [node_idx_mapping[node] for node in df_.node])
        .set_index("node")
    )

    edgelist = (edgelist
        .assign(source = lambda df_: [node_idx_mapping[node] for node in df_.source])
        .assign(target = lambda df_: [node_idx_mapping[node] for node in df_.target])
    )
    
    # replace class label names with class indices
    class_to_idx_dict = {label: idx for idx, label in enumerate(sorted(node_info.label.unique()))}
    idx_to_class_dict = {idx: label for idx, label in enumerate(sorted(node_info.label.unique()))}

    node_info = node_info.assign(label = lambda df_: [class_to_idx_dict[label] for label in df_.label])
    
    return node_info, edgelist, class_to_idx_dict, idx_to_class_dict


def init_nx_graph(edgelist):
    return nx.DiGraph([(u, v) for u, v in zip(edgelist.source.values, edgelist.target.values)])