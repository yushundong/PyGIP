import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from .snnl import soft_nearest_neighbor_loss


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch) 
        return self.linear(x)

    def get_embeddings(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class WatermarkedGNN(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.gnn = base_model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        return self.gnn(data.x, data.edge_index, data.batch)

    def compute_loss(self, data_list, key_inputs, key_labels):
        device = next(self.parameters()).device

       
        loader = DataLoader(data_list, batch_size=len(data_list))
        data = next(iter(loader)).to(device)

        preds = self(data)
        data_labels = data.y.to(device)
        loss_cls = self.loss_fn(preds, data_labels)

        key_inputs = [d.to(device) for d in key_inputs]
        loader = DataLoader(key_inputs, batch_size=len(key_inputs))
        key_data = next(iter(loader))

        embeddings = self.gnn.get_embeddings(
            torch.cat([data.x, key_data.x], dim=0),
            torch.cat([data.edge_index, key_data.edge_index], dim=1),
            torch.cat([data.batch, key_data.batch + data.batch.max() + 1], dim=0)
        )

        combined_labels = torch.cat([data_labels, key_labels.to(device)], dim=0)
        loss_snnl = soft_nearest_neighbor_loss(embeddings, combined_labels, temperature=0.1)

        return loss_cls - 0.5 * loss_snnl

