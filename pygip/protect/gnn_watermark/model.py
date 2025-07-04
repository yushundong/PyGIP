import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from snnl import soft_nearest_neighbor_loss

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

class WatermarkedGNN(nn.Module):
    def __init__(self, base_model):
        super(WatermarkedGNN, self).__init__()
        self.gnn = base_model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data, key_inputs=None):
        if isinstance(data, list):
            # batch of graphs
            outputs = [self.gnn(g.x, g.edge_index) for g in data]
            preds = torch.stack([o.mean(dim=0) for o in outputs])
        else:
            preds = self.gnn(data.x, data.edge_index)

        if key_inputs is not None:
            key_outs = [self.gnn(g.x, g.edge_index) for g in key_inputs]
            key_preds = torch.stack([o.mean(dim=0) for o in key_outs])
            return preds, key_preds

        return preds

    def compute_loss(self, data_list, key_inputs, key_labels):
        preds, key_preds = self(data_list, key_inputs)

        data_labels = torch.tensor([data.y.item() for data in data_list])
        loss_cls = self.loss_fn(preds, data_labels)

        embeddings = torch.stack([
            self.gnn.get_embeddings(g.x, g.edge_index).mean(dim=0) for g in data_list + key_inputs
        ])
        all_labels = torch.cat([data_labels, key_labels])

        loss_snnl = soft_nearest_neighbor_loss(embeddings, all_labels, temperature=0.1)

        total_loss = loss_cls - 0.5 * loss_snnl
        return total_loss
