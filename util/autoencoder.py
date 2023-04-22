# import own scripts
import util.load_Data as loadData

# import external packages
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE, VGAE, APPNP
import torch_geometric.transforms as T
from ray import tune, air
from ray.tune import JupyterNotebookReporter
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import networkx as nx # graph data
from node2vec import Node2Vec

# ignore warnings that show in every raytune run
import warnings
warnings.simplefilter(action = "ignore", category = np.VisibleDeprecationWarning)

def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # same for pytorch
    random_seed = 1 # or any of your favorite number 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scaling_factor, num_propagations, teleport_probability, dropout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.scaling_factor = scaling_factor
        self.propagate = APPNP(K = num_propagations, alpha = teleport_probability, dropout = dropout)

    def forward(self, x, edge_index):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x,p=2,dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)
        return x, x_
    
def enrich_node_info(G, node_info):

    # generate random walks for node2vec
    node2vec = Node2Vec(G, dimensions=4, walk_length=80, num_walks=10, workers=1, p=1.25, q=0.25)

    # compute graph-based node features
    DCT = nx.degree_centrality(G)
    BCT = nx.betweenness_centrality(G)
    KCT = nx.katz_centrality(G, alpha = 0.01)
    PR  = nx.pagerank(G)
    HUB, AUTH = nx.hits(G)
    N2V = node2vec.fit(window=10)

    return (node_info
        .assign(KEYS = lambda df_: df_.sum(axis = 1))
        .assign(DCT  = lambda df_: [DCT[node]  for node in df_.index])
        .assign(BCT  = lambda df_: [BCT[node]  for node in df_.index])
        .assign(KCT  = lambda df_: [KCT[node]  for node in df_.index])
        .assign(PR   = lambda df_: [PR[node]   for node in df_.index])
        .assign(HUB  = lambda df_: [HUB[node]  for node in df_.index])
        .assign(AUTH = lambda df_: [AUTH[node] for node in df_.index])
        .assign(N2V  = lambda df_: [N2V.wv[node][0]  for node in df_.index])
        .assign(N2V  = lambda df_: [N2V.wv[node][1]  for node in df_.index])
        .assign(N2V  = lambda df_: [N2V.wv[node][2]  for node in df_.index])
        .assign(N2V  = lambda df_: [N2V.wv[node][3]  for node in df_.index])
    )
    

def load(val_ratio = 0.2, test_ratio = 0.1):
    # load data
    
    (G, G_train, G_trainval, node_info, train_tf, val_tf, trainval_tf, test_tf) = loadData.load(val_ratio, test_ratio)
        # get train and validation masks
    print(f"sum of train pos edges: {((trainval_tf['y'] == 1) & (trainval_tf['train_mask'] == 1)).sum()}")
    print(f"sum of train neg edges: {((trainval_tf['y'] == 0) & (trainval_tf['train_mask'] == 1)).sum()}")
    print(f"sum of val pos edges: {((trainval_tf['y'] == 1) & (trainval_tf['val_mask'] == 1)).sum()}")
    print(f"sum of val neg edges: {((trainval_tf['y'] == 0) & (trainval_tf['val_mask'] == 1)).sum()}")
    
    #removed
    """trainval_tf = (trainval_tf
        .assign(train_mask = lambda df_: [True if idx in train_tf.index else False for idx in df_.index])
        .assign(val_mask = lambda df_: ~df_.train_mask)
    )"""

    print(f"sum of train pos edges: {((trainval_tf['y'] == 1) & (trainval_tf['train_mask'] == 1)).sum()}")
    print(f"sum of train neg edges: {((trainval_tf['y'] == 0) & (trainval_tf['train_mask'] == 1)).sum()}")
    print(f"sum of val pos edges: {((trainval_tf['y'] == 1) & (trainval_tf['val_mask'] == 1)).sum()}")
    print(f"sum of val neg edges: {((trainval_tf['y'] == 0) & (trainval_tf['val_mask'] == 1)).sum()}")
    # enrich node_info
    print("Enriching node features...")
    node_info_train = enrich_node_info(G_train, node_info)
    node_info_trainval = enrich_node_info(G_trainval, node_info)
    # initialise PyTorch Geometric Dataset
    print("Create PyTorch Geometric dataset...")
    data = Data(
                # node features
                x = torch.tensor(node_info_train.values, dtype = torch.float32),
                x_trainval = torch.tensor(node_info_trainval.values, dtype = torch.float32),
                # train edges
                train_edges = torch.tensor(
                    trainval_tf.loc[trainval_tf.train_mask == 1][["source", "target"]].values
                ).T,
                train_pos_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 1) & (trainval_tf.train_mask == 1)][["source", "target"]].values
                ).T,
                train_neg_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 0) & (trainval_tf.train_mask == 1)][["source", "target"]].values
                ).T,
                # val edges
                val_edges = torch.tensor(
                    trainval_tf.loc[trainval_tf.val_mask == 1][["source", "target"]].values
                ).T,
                val_pos_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 1) & (trainval_tf.val_mask == 1)][["source", "target"]].values
                ).T,
                val_neg_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 0) & (trainval_tf.val_mask == 1)][["source", "target"]].values
                ).T,
                # trainval edges                
                trainval_edges = torch.tensor(
                    trainval_tf[["source", "target"]].values
                ).T,
                trainval_pos_edges = torch.tensor(
                    trainval_tf.loc[trainval_tf.y == 1][["source", "target"]].values
                ).T,
                # test edges
                test_edges = torch.tensor(
                    test_tf.values
                ).T)
    
    #trainval_tf.drop(columns = ['train_mask', 'val_mask'], axis = 1, inplace  = True)
    # preprocess data
    data = T.NormalizeFeatures()(data)

    return data, (G, G_train, G_trainval, node_info, train_tf, val_tf, trainval_tf, test_tf)

def get_device(model = None):
    # where we want to run the model (so this code can run on cpu, gpu, multiple gpus depending on system)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1 and model is not None:
            model = nn.DataParallel(model)
    if model is not None:
        return device, model.to(device)
    return device

def train_validate(config):
    # ensure reproduction
    set_reproducible()
    
    # how many epochs we want to train for (at maximum)
    max_epochs = int(config["max_epochs"])

    # load data
    data = config["data"]

    # model initialisation
    if config["model"] == "VGNAE":
        model = VGAE(Encoder(data.x.size()[1], config["enc_channels"], config["scaling"],
                             config["num_prop"], config["teleport"], config["dropout"]))

    # initialise device
    device, model = get_device(model)

    # move data to device
    data.to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["wd"])

    # metrics
    max_val_auc = 0

    # helper function
    def validate(pos_edges, neg_edges):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edges)

        # get preds
        pos_y = z.new_ones(pos_edges.size(1))
        neg_y = z.new_zeros(neg_edges.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred_probas = model.decoder(z, pos_edges, sigmoid=True)
        neg_pred_probas = model.decoder(z, neg_edges, sigmoid=True)
        pred_probas = torch.cat([pos_pred_probas, neg_pred_probas], dim=0)

        y, pred_probas = y.detach().cpu().numpy(), pred_probas.detach().cpu().numpy()
        pred = (pd.DataFrame(pred_probas)
            .rename(columns = {0: "pred_probas"})
            .assign(pred = lambda df_: (df_.pred_probas > df_.pred_probas.median()).astype(int))
        )
        # compute scores
        auc = roc_auc_score(y, pred.pred)
        acc = accuracy_score(y, pred.pred)

        return auc, acc
    
    for epoch in range(1, max_epochs + 1):
        ##TRAINING##
        model.train()
        optimizer.zero_grad()

        # forward + backward + optimize
        z = model.encode(data.x, data.train_pos_edges)
        loss = model.recon_loss(z, data.train_pos_edges)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

        # compute stats
        trn_loss = loss.item()
        trn_auc, trn_acc = validate(data.train_pos_edges, data.train_neg_edges)

        ##VALIDATION##
        model.eval()
        with torch.no_grad():
            # forward
            z = model.encode(data.x, data.train_pos_edges)
            loss = model.recon_loss(z, data.val_pos_edges, data.val_neg_edges)
            loss = loss + (1 / data.num_nodes) * model.kl_loss()

            # compute stats
            val_loss = loss.item()
            val_auc, val_acc = validate(data.val_pos_edges, data.val_neg_edges)
        
        ##SAVE current best models##
        if config["save"]:
            if val_auc >= max_val_auc:
                max_val_auc = val_auc
                path = os.path.abspath("")+"\\autoencoder.pt"
                torch.save(model.state_dict(), path)

        ##REPORT##
        if config["verbose"]:          
            print('Epoch: [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train AUC: {:.4f}, Val AUC: {:.4f}'.format(epoch, max_epochs,
                                                                                                                    trn_loss, val_loss,
                                                                                                                    trn_auc, val_auc))

        if config["ray"]:
            tune.report(trn_loss = trn_loss, val_loss = val_loss,
                        trn_auc = trn_auc, val_auc = val_auc, max_val_auc = max_val_auc,
                        trn_acc = trn_acc, val_acc = val_acc)
            
def get_embeddings(model, x, pos_edges):
    embeddings = model.encode(x, pos_edges)
    return embeddings.detach().cpu().numpy()

def get_similarity(model, x, pos_edges, pred_edges):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, pos_edges)
    pred = model.decoder(z, pred_edges, sigmoid = True)
    return pred.detach().cpu().numpy()
            
def trial_str_creator(trial):
    """
    Trial name creator for ray tune logging.
    """
    model = trial.config["model"]
    lr    = trial.config["lr"]
    wd    = trial.config["wd"]
    return f"{model}_{lr}_{wd}_{trial.trial_id}"

def run_ray_experiment(train_func, config, ray_path, num_samples, metric_columns, parameter_columns):

    reporter = JupyterNotebookReporter(
        metric_columns = metric_columns,
        parameter_columns= parameter_columns,
        max_column_length = 15,
        max_progress_rows = 20,
        max_report_frequency = 10, # refresh output table every ten seconds
        print_intermediate_tables = True
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"CPU": 4, "GPU": 0}
        ),
        tune_config = tune.TuneConfig(
            metric = "trn_loss",
            mode = "min",
            num_samples = num_samples,
            trial_name_creator = trial_str_creator,
            trial_dirname_creator = trial_str_creator,
            ),
        run_config = air.RunConfig(
            local_dir = ray_path,
            progress_reporter = reporter,
            verbose = 1),
        param_space = config
    )

    result_grid = tuner.fit()
    
    return result_grid

def open_validate_ray_experiment(experiment_path, trainable):
    # open & read experiment folder
    print(f"Loading results from {experiment_path}...")
    restored_tuner = tune.Tuner.restore(experiment_path, trainable = trainable, resume_unfinished = False)
    result_grid = restored_tuner.get_results()
    print("Done!\n")

    # Check if there have been errors
    if result_grid.errors:
        print(f"At least one of the {len(result_grid)} trials failed!")
    else:
        print(f"No errors! Number of terminated trials: {len(result_grid)}")
        
    return restored_tuner, result_grid