import torch.nn.functional as F
import torch as th

from model.gnn import SAGE, GAT, GIN, GCN

def get_gnn_model(config):
    if config.model == 'graphsage':
        model = SAGE(config.in_feats, config.n_hidden, config.n_classes, config.gnn_layers, F.relu, config.batch_size, config.num_workers, config.dropout)
    elif config.model == 'gat':
        model = GAT(config.in_feats, config.n_hidden, config.n_classes, config.gnn_layers, F.relu, config.batch_size, config.num_workers, config.dropout)
    elif config.model == 'gin':
        model = GIN(config.in_feats, config.n_hidden, config.n_classes, config.gnn_layers, F.relu, config.batch_size, config.num_workers, config.dropout)
    elif config.model == 'gcn':
        model = GCN(config.in_feats, config.n_hidden, config.n_classes, config.gnn_layers, F.relu, config.batch_size, config.num_workers, config.dropout)
    return model

def load_trained_gnn_model(model, model_path, device):
    print("load model from: ", model_path)
    state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model