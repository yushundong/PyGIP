import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        layers = []
        h = hidden
        if num_layers <= 1:
            layers.append(nn.Linear(in_dim, h))
        else:
            layers.append(nn.Linear(in_dim, h))
            for _ in range(num_layers - 2):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(h, h))
        self.net = nn.Sequential(*layers)
        self.dropout = dropout

    def forward(self, x, edge_index): 
        return self.net(x)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # node embeddings


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=3, heads=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden, heads=heads, concat=False))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def get_encoder(arch: str, in_dim: int, hidden: int, num_layers: int = 3, dropout: float = 0.5):
    arch = arch.lower()
    if arch == "gcn":
        return GCN(in_dim, hidden, num_layers=num_layers, dropout=dropout)
    if arch in ("sage", "graphsage"):
        return GraphSAGE(in_dim, hidden, num_layers=num_layers, dropout=dropout)
    if arch == "gat":
        return GAT(in_dim, hidden, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown arch: {arch}")


# Decoder for link prediction
class DotProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index):
        # z: node embeddings [N, d]
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)  # logits for edges
