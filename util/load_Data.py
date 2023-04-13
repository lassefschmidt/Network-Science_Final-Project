# import external packages
import networkx as nx # graph data
import numpy as np
import pandas as pd
import copy 
import random
import networkx as nx
import itertools
from sklearn.model_selection import train_test_split

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

def generate_pos_edges(G, edgelist, val_ratio, test_ratio, seed=42):
    """
    generate pos edges for validation set, trim training graph respectively
    and ensure that it remains fully connected
    """
    # how many positive edges we want to sample for test data divided by 2 cause
    # is both number for validation and test
    val_pos_edges_num = int(len(G.edges) * val_ratio)
    test_pos_edges_num = int(len(G.edges) * test_ratio)
    random.seed(seed)
    val_pos_edges = pd.DataFrame(data = None, columns = edgelist.columns)
    test_pos_edges = pd.DataFrame(data = None, columns = edgelist.columns)
    G_train = copy.deepcopy(G)
    G_trainval = copy.deepcopy(G)

    # start reducing the graph to sample nodes of validation set
    train_pos_edges = edgelist
    sampled, removed = 0, 0
    while (removed < val_pos_edges_num) and (sampled < val_pos_edges_num * 10):
        sampled += 1
        random_edge = random.sample(list(train_pos_edges.index.values), 1)[0] # sample one random edge
        target, source = train_pos_edges.loc[random_edge].values # unpack edge
        if (G_train.degree(target) > 1 and G_train.degree(source) > 1): # only remove edge if both nodes have degree >1
            G_train.remove_edge(source, target)
            train_pos_edges = train_pos_edges.drop(index = random_edge, inplace = False)
            val_pos_edges.loc[random_edge] = [target, source]
            removed += 1
        else:
            continue
    
    # start reducing the graph to sample nodes of test set
    sampled, removed = 0, 0
    while (removed < test_pos_edges_num) and (sampled < test_pos_edges_num * 10):
        sampled += 1
        random_edge = random.sample(list(train_pos_edges.index.values), 1)[0] # sample one random edge
        target, source = train_pos_edges.loc[random_edge].values # unpack edge
        if (G_train.degree(target) > 1 and G_train.degree(source) > 1): # only remove edge if both nodes have degree >1
            G_train.remove_edge(source, target)
            G_trainval.remove_edge(source, target)
            train_pos_edges = train_pos_edges.drop(index = random_edge, inplace = False)
            test_pos_edges.loc[random_edge] = [target, source]
            removed += 1
        else:
            continue
    
    # adding labels
    train_pos_edges['y'] = 1
    val_pos_edges['y'] = 1
    test_pos_edges['y'] = 1

    # print key stats
    print(f"Number of positive edges for training: {len(train_pos_edges)}")
    print(f"Number of positive edges for validation: {len(val_pos_edges)}")
    print(f"Number of positive edges for test: {len(test_pos_edges)}")
    print(f"Number of edges in original graph: {G.number_of_edges()}")
    print(f"Number of edges in training graph: {G_train.number_of_edges()}")

    return G_train, G_trainval, train_pos_edges, val_pos_edges, test_pos_edges

def generate_neg_edges(edgelist, num_train, num_val, num_test, seed=42):
    """
    Generate non-existing edges for train, validation, and test sets.
    """
    # Get unique nodes from the edgelist
    nodes = set(edgelist.source).union(edgelist.target)

    # Generate all possible combinations of nodes
    all_edges = pd.DataFrame(list(itertools.combinations(nodes, 2)), columns=['source', 'target'])

    # Filter out existing edges
    existing_edges = set(map(tuple, edgelist[['source', 'target']].values))
    all_edges['is_existing'] = all_edges.apply(lambda x: (x.source, x.target) in existing_edges, axis=1)
    non_existing_edges = all_edges[~all_edges.is_existing].drop('is_existing', axis=1)

    # Split non-existing edges into train, validation, and test sets
    val_test_edges, train_neg_edges = train_test_split(non_existing_edges, train_size=num_train, random_state=seed)
    val_neg_edges, test_neg_edges = train_test_split(val_test_edges, train_size=num_val, test_size=num_test, random_state=seed)

    # adding labels
    train_neg_edges['y'] = 0
    val_neg_edges['y'] = 0
    test_neg_edges['y'] = 0
    return train_neg_edges, val_neg_edges, test_neg_edges

def load(val_ratio = 0.2, test_ratio = 0.1):
    """
    helper function that performs all loading + data cleaning (direct input for deep learning)
    """
    node_info, edgelist, class_to_idx_dict, idx_to_class_dict = load_raw()
    
    # remove completely empty columns from node_info
    empty_cols = node_info.columns[node_info.nunique() == 1].values
    node_info = node_info.drop(columns = empty_cols, inplace = False)
    # build graph
    G = init_nx_graph(edgelist)

    # generate train and validation data (positive edges)
    G_train, G_trainval, train_pos_edges, val_pos_edges, test_pos_edges = generate_pos_edges(G, edgelist, val_ratio, test_ratio)

    # generate train and validation data (negative edges)
    train_neg_edges, val_neg_edges, test_neg_edges = generate_neg_edges(edgelist, len(train_pos_edges), len(val_pos_edges), len(test_pos_edges))

    # append to dataframe
    train = pd.concat([train_pos_edges, train_neg_edges]).sort_index()
    val = pd.concat([val_pos_edges, val_neg_edges]).sort_index()
    test = pd.concat([test_pos_edges, test_neg_edges]).sort_index()

    # sort edge lists (so lower numbered node is always in first column)
    # to change
    train = train[["source", "target"]].apply(lambda x: np.sort(x), axis = 1, raw = True)
    val = train[["source", "target"]].apply(lambda x: np.sort(x), axis = 1, raw = True)
    test  = test[[ "source", "target"]].apply(lambda x: np.sort(x), axis = 1, raw = True)

    return (G, G_train, G_trainval, node_info, train, val, test)

def split_frame(df):
    # split into X and y and drop node columns
    if "y" in df:
        y = df.loc[:, "y"]
        X = copy.deepcopy(df)
        X.drop(["source", "target", "y"], axis = 1, inplace = True)
        return X, y
    else:
        X = copy.deepcopy(df)
        X.drop(["source", "target"], axis = 1, inplace = True)
        return X


