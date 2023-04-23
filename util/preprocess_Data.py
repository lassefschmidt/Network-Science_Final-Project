# parse & handle data
from apyori import apriori # generate decision rules
from itertools import combinations
import networkx as nx # graph data
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
import sknetwork
import numpy as np
import json

def clean_edgelist(edgelist):
    """
    Remove edges from edgelist where source node == target node
    """
    return edgelist.loc[(edgelist.node1 != edgelist.node2)]

def fetch_graph(edgelist):
    """
    Create a graph based on an edgelist. Make sure that it doesn't contain edges where source node == target node
    """
    if "y" in edgelist:
        edgelist = edgelist.loc[(edgelist.y == 1)]
    
    return nx.from_pandas_edgelist(edgelist, "node1", "node2")

def get_gcc(G):
    """
    check if graph is connected -- if not, return greatest connected component subgraph
    """
    # Is the given graph connected?
    connected = nx.is_connected(G) # check if the graph is connected or not
    if connected:
        print("The graph is connected")
        return G
    
    print("The graph is not connected")
    
    # Find the number of connected components
    num_of_cc = nx.number_connected_components(G)
    print("Number of connected components: {}".format(num_of_cc))
    
    # Get the greatest connected component subgraph
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes)
    node_fraction = gcc.number_of_nodes() / float(G.number_of_nodes())
    edge_fraction = gcc.number_of_edges() / float(G.number_of_edges())
    
    print("Fraction of nodes in GCC: {:.3f}".format(node_fraction))
    print("Fraction of edges in GCC: {:.3f}".format(edge_fraction))

    return gcc

def get_decision_rules(edgelist, node_info):
    """
    Extract decision rules based on keyword embedding of graph.
    """
    # get dense representation of keywords per node
    kws_per_node = (node_info
        .assign(kws = lambda df_: [sorted(set([df_.columns[idx] for idx, val in enumerate(df_.loc[i]) if val != 0])) for i in df_.index])
        .kws.to_frame()
        .assign(num_kws = lambda df_: [len(kw) for kw in df_.kws])
    )

    # create baskets
    baskets = []
    for u, v in edgelist:
        basket = kws_per_node.kws.loc[u] + kws_per_node.kws.loc[v]
        baskets.append(basket)

    # generate decision rules for sets of keywords that were combined at least 10 times
    association_rules = apriori(baskets, min_support=20/len(baskets), min_confidence=0.2, min_lift=3, max_length=3)
    association_results = list(association_rules)

    # store them in a dict
    rule_finder_dict = dict()
    rule_values_dict = dict()
    for item in association_results:
        pair = item[0]
        items = [x for x in pair]
        source = items[0]
        target = items[1]
        supp = item[1] * len(baskets)
        conf = item[2][0][2]
        lift = item[2][0][3]

        # add to dicts
        rule_finder_dict[source] = rule_finder_dict.get(source, []) + [target]
        rule_values_dict[(source, target)] = {"support": supp, "confidence": conf, "lift": lift}

    return kws_per_node, rule_finder_dict, rule_values_dict

def get_friendlink(G, max_power = 6):

    # initialise where we store result
    res = dict()
    
    # initialise adjacency matrix sorted by node_id (ascending)
    nodelist = sorted(list(G.nodes()))
    adj = nx.adjacency_matrix(G, nodelist = nodelist)

    # initialise normalizing coefficient
    normalizing_coef = 1

    # get FriendLink score for second to max_power power of adj matrix
    for k in range(2, max_power + 1):
        mat = (adj ** k).tocoo()
        scaling = 1/(k-1)
        normalizing_coef *= (G.number_of_nodes() - k)

        for i, j, num_paths in zip(mat.row, mat.col, mat.data):
            if i >= j: # only compute for lower diagonal (undirected graph)
                continue
            u, v = nodelist[i], nodelist[j]
            w = scaling * (num_paths / normalizing_coef)

            if (u, v) not in res:
                res[(u, v)] = w
            else:
                res[(u, v)] += w
        
    return res

def get_kat_idx_edges(G, beta = 0.001, max_power = 6):

    # initialise where we store result
    res = dict()
    
    # initialise adjacency matrix sorted by node_id (ascending)
    nodelist = sorted(list(G.nodes()))
    adj = nx.adjacency_matrix(G, nodelist = nodelist)

    # get edge katz index for each power of adj matrix
    for k in range(1, max_power + 1):
        mat = (adj ** k).tocoo()

        for i, j, num_paths in zip(mat.row, mat.col, mat.data):
            if i >= j: # only compute for lower diagonal (undirected graph)
                continue
            u, v = nodelist[i], nodelist[j]
            w = num_paths * (beta**k)

            if (u, v) not in res:
                res[(u, v)] = w
            else:
                res[(u, v)] += w
        
    return res

def get_sknetwork_features(G, ebunch, feature_name):

    # get objects we need to compute sknetwork features
    nodelist = sorted(list(G.nodes()))
    adj = nx.adjacency_matrix(G, nodelist = nodelist).toarray()

    # get sknetwork feature
    if feature_name == "SaltonIndex":
        scorer = sknetwork.linkpred.SaltonIndex()
    elif feature_name == "SorensenIndex":
        scorer = sknetwork.linkpred.SorensenIndex()
    elif feature_name == "HubPromotedIndex":
        scorer = sknetwork.linkpred.HubPromotedIndex()
    elif feature_name == "HubDepressedIndex":
        scorer =  sknetwork.linkpred.HubDepressedIndex()

    # fit and predict
    scorer = scorer.fit(adj)
    return scorer.predict(ebunch)

def get_n2v(G):
    # precompute probabilities and generate walks
    node2vec = Node2Vec(G, dimensions=4, walk_length=80, num_walks=10, workers=1, p=1.25, q=0.25)

    # embed nodes
    model = node2vec.fit(window=10)

    # edge embedding
    l2_edges_embs = WeightedL2Embedder(keyed_vectors=model.wv)

    return l2_edges_embs

def get_simrank(G, G_train, edgelist_test, edgelist_trainval):
    """Returns simrank of G and G_train
    G, G_train graphs
    edgelist_test, edgelist_trainval list of edges in test and train
    """
    simrank_G_full = nx.simrank_similarity(G)
    simrank_G = dict()
    for u, v in zip(edgelist_test.source, edgelist_test.target):
        simrank_G[(str(u) +"_"+ str(v))] = simrank_G_full[u][v]

    simrank_G_train_full = nx.simrank_similarity(G_train)
    simrank_G_train = dict()
    for u, v in zip(edgelist_trainval.source, edgelist_trainval.target):
        simrank_G_train[str(u) +"_"+ str(v)] = simrank_G_train_full[u][v]    

    return simrank_G, simrank_G_train

def rooted_pagerank(G, node, d = 0.85, epsilon = 1e-4):
    """ Returns rooted pagerank vector
    g graph
    node root
    d damping coefficient
    """
    ordered_nodes = sorted(G.nodes())
    root = ordered_nodes.index(node)
    adj = nx.to_numpy_array(G, nodelist = ordered_nodes)
    m = np.copy(adj)

    for i in range(len(G)):
        row_norm = np.linalg.norm(m[i], ord = 1)
        if row_norm != 0:
            m[i] = m[i] / row_norm

    m = m.transpose()

    rootvec = np.zeros(len(G))
    rootvec[root] = 1

    vect = np.random.rand(len(G))
    vect = vect / np.linalg.norm(vect, ord = 1)
    last_vect = np.ones(len(G)) * 100 # to ensure that does not hit epsilon randomly in first step

    iterations = 0
    while np.linalg.norm(vect - last_vect, ord = 2) > epsilon:
        last_vect = vect.copy()
        vect = d * np.matmul(m, vect) + (1 - d) * rootvec
        iterations += 1

    eigenvector = vect / np.linalg.norm(vect, ord = 1)

    eigen_dict = {}
    for i in range(len(ordered_nodes)):
        eigen_dict[ordered_nodes[i]] = eigenvector[i]

    return eigen_dict

def pagerank_avg(G, edgelist):
    pagerank_scores = nx.pagerank(G, alpha=0.3)
    PR1 = []
    # calculate the pagerank average scores
    for idx, row in edgelist.iterrows():
        pr1 = (pagerank_scores[row["source"]] + pagerank_scores[row["target"]])/2
        PR1.append((row["source"], row["target"], pr1))
    return PR1

def pagerank_sqdiff(G, edgelist):
    pagerank_scores = nx.pagerank(G, alpha=0.3)
    PR2 = []
    # calculate the rooted pagerank squared difference scores
    for idx, row in edgelist.iterrows():
        pr2 = (pagerank_scores[row["source"]] - pagerank_scores[row["target"]])**2
        PR2.append((row["source"], row["target"], pr2))
    return PR2

def feature_extractor(edgelist, G, node_info, simrank_test, simrank_trainval, pagerank_test, pagerank_trainval, n2v, trainval = None):
    """
    Enrich edgelist with graph-based edge features
    (e.g. resource allocation index, jaccard coefficient, etc.)
    and similarity metrics based on node-level keyword embedding

    Features that didn't work out: 
    HITS algorithm, eigenvector/katz/common-neighbor/load centrality, voterank, CF/SCF enhanced RA (huge overfit),
    dispersion, cosine similarity of embeddings
    """
    # create an undirected copy of the graph 
    G_undirected = G.to_undirected()

    # helper function to transform networkx generator objects into feature dicts
    def transform_generator_to_dict(generator_obj):
        result = dict()
        for (u, v, value) in generator_obj:
            result[(u, v)] = value
        return result
    
    # helper function to get CF- and SCF-enhanced features (see https://doi.org/10.1016/j.physa.2021.126107)
    def enhance_CF(edge, feature_dict, feature_func):
        (u, v) = edge
        # get neighbors of each node
        neighbors_u = [(n, v) for n in G_undirected.neighbors(u) if n != v]
        neighbors_v = [(n, u) for n in G_undirected.neighbors(v) if n != u]
        # compute similarity of neighbors of source (target) with target (source)
        sim_neighbors_u_to_v = sum([get_sim(edge, feature_dict, feature_func) for edge in neighbors_u])
        sim_neighbors_v_to_u = sum([get_sim(edge, feature_dict, feature_func) for edge in neighbors_v])
        return sim_neighbors_u_to_v + sim_neighbors_v_to_u
    
    def enhance_SCF(edge, feature_dict, feature_func):
        (u, v) = edge
        sim_neighbors = enhance_CF(edge, feature_dict, feature_func)
        sim_edge = sum([get_sim(edge, feature_dict, feature_func) for edge in [(u,v), (v,u)]])
        return sim_neighbors + sim_edge
    
    def get_sim(edge, feature_dict, feature_func):
        if edge not in feature_dict:
            feature_dict[edge] = sum([value for (_, _, value) in feature_func(G_undirected, [edge])])
        return feature_dict[edge]
    
    def enhance(edge, feature_dict, feature_func, enhance_type):
        (u, v) = edge
        # remove current edge to avoid overfitting (if it exists)
        try:
            G_undirected.remove_edge(u, v)
            edge_removed = True
        except nx.NetworkXError:
            edge_removed = False
        
        # get feature
        if enhance_type == "CF":
            result = enhance_CF(edge, feature_dict, feature_func)
        elif enhance_type == "SCF":
            result = enhance_SCF(edge, feature_dict, feature_func)

        # add current edge back to graph if it existed
        if edge_removed:
            G_undirected.add_edge(u, v)

        return result

    def read_simrank_json(u, v):
        key = str(u)+"_"+str(v)
        if key in simrank_test.keys():
            return simrank_test[key]
        elif key in simrank_trainval.keys():
            return simrank_trainval[key]
    
    def read_pagerank_json(u, v):
        key = str(u)+"_"+str(v)
        if key in pagerank_test.keys():
            return pagerank_test[key]
        elif key in pagerank_trainval.keys():
            return pagerank_trainval[key]

    # compute graph-based node features
    DCT = nx.degree_centrality(G)
    BCT = nx.betweenness_centrality(G)
    # compute graph-based edge features
    ebunch = [(u, v) for u, v in zip(edgelist.source, edgelist.target)]
    RA  = transform_generator_to_dict(nx.resource_allocation_index(G_undirected, ebunch))
    JCC = transform_generator_to_dict(nx.jaccard_coefficient(G_undirected, ebunch))
    AA  = transform_generator_to_dict(nx.adamic_adar_index(G_undirected, ebunch))
    PA  = transform_generator_to_dict(nx.preferential_attachment(G_undirected, ebunch))
    CNC  = transform_generator_to_dict(nx.common_neighbor_centrality(G_undirected, ebunch))
    PR1 = transform_generator_to_dict(pagerank_avg(G, edgelist))
    PR2 = transform_generator_to_dict(pagerank_sqdiff(G, edgelist))
    katz_idx = get_kat_idx_edges(G, beta = 0.05, max_power = 6)
    friendLink = get_friendlink(G, max_power = 6)

    # append new columns
    return (edgelist
        # node_info features
        .assign(nodeInfo_dupl  = lambda df_: [1 if (node_info.loc[u].values == node_info.loc[v].values).all() else 0 for u, v in zip(df_.source, df_.target)])
        #.assign(nodeInfo_CS    = lambda df_: [cosine_similarity(node_info.loc[u], node_info.loc[v]) for u, v in zip(df_.node1, df_.node2)])
        .assign(nodeInfo_diff  = lambda df_: [sum(abs(node_info.loc[u] - node_info.loc[v])) for u, v in zip(df_.source, df_.target)])
        # node features
        .assign(source_DCT  = lambda df_: [DCT[node] for node in df_.source])
        .assign(target_DCT  = lambda df_: [DCT[node] for node in df_.target])
        .assign(BCT_diff    = lambda df_: [BCT[v]- BCT[u] for u, v in zip(df_.source, df_.target)])
        # local edge features
        #.assign(graph_distance = lambda df_: [nx.shortest_path_length(G, source = u, target = v) for u, v in zip(df_.source, df_.target)])
        .assign(CNC    = lambda df_: [CNC[edge] for edge in zip(df_.source, df_.target)])
        .assign(RA     = lambda df_: [RA[edge]  for edge in zip(df_.source, df_.target)])
        .assign(CF_RA  = lambda df_: [enhance(edge,  RA, nx.resource_allocation_index, "CF") for edge in zip(df_.source, df_.target)])
        .assign(SCF_RA = lambda df_: [enhance(edge, RA, nx.resource_allocation_index, "SCF") for edge in zip(df_.source, df_.target)])
        .assign(JCC    = lambda df_: [JCC[edge] for edge in zip(df_.source, df_.target)])
        .assign(AA     = lambda df_: [AA[edge]  for edge in zip(df_.source, df_.target)])
        .assign(PA     = lambda df_: [PA[edge]  for edge in zip(df_.source, df_.target)])
        .assign(PA_log = lambda df_: np.log(df_.PA))
        .assign(CF_PA  = lambda df_: [enhance(edge,  PA, nx.preferential_attachment, "CF") for edge in zip(df_.source, df_.target)])
        .assign(SCF_PA = lambda df_: [enhance(edge, PA, nx.preferential_attachment, "SCF") for edge in zip(df_.source, df_.target)])
        .assign(SaI    = get_sknetwork_features(G_undirected, ebunch, "SaltonIndex"))
        .assign(SoI    = get_sknetwork_features(G_undirected, ebunch, "SorensenIndex"))
        .assign(HProm  = get_sknetwork_features(G_undirected, ebunch, "HubPromotedIndex"))
        .assign(HDem   = get_sknetwork_features(G_undirected, ebunch, "HubDepressedIndex"))
        # global edge features
        .assign(katz_idx = lambda df_: [katz_idx.get((u, v), 0) for u, v in zip(df_.source, df_.target)])
        .assign(sim_rank = lambda df_: [read_simrank_json(u, v) for u, v in zip(df_.source, df_.target)])
        .assign(root_pagerank = lambda df_: [read_pagerank_json(u, v) for u, v in zip(df_.source, df_.target)])
        .assign(node2vec_1 = lambda df_: [n2v[(str(u), str(v))][0] for u, v in zip(df_.source, df_.target)])
        .assign(node2vec_2 = lambda df_: [n2v[(str(u), str(v))][1] for u, v in zip(df_.source, df_.target)])
        .assign(node2vec_3 = lambda df_: [n2v[(str(u), str(v))][2] for u, v in zip(df_.source, df_.target)])
        .assign(node2vec_4 = lambda df_: [n2v[(str(u), str(v))][3] for u, v in zip(df_.source, df_.target)])
        # quasi-local edge features
        .assign(friendLink = lambda df_: [friendLink.get((u, v), 0) for u, v in zip(df_.source, df_.target)])
        # pagerank based features
        .assign(PR1 = lambda df_:  [PR1[edge] for edge in zip(df_.source, df_.target)])
        .assign(PR2 = lambda df_:  [PR2[edge] for edge in zip(df_.source, df_.target)])
    )