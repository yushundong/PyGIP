# Graph classification (GC) model for ENZYMES using GraphSAGE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE_GC(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.5, pool: str = "mean"):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.pool = pool

        convs = [SAGEConv(in_dim, hidden)]
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden, hidden))
        self.convs = nn.ModuleList(convs)

        self.cls = nn.Linear(hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.convs:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        nn.init.xavier_uniform_(self.cls.weight)
        if self.cls.bias is not None:
            nn.init.zeros_(self.cls.bias)

    def _pool(self, x, batch):
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        return global_mean_pool(x, batch)  # extend to "sum"/"max" if needed

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        g = self._pool(x, batch)
        out = self.cls(g)
        return out


def get_model(arch: str, in_dim: int, hidden: int, num_classes: int,
              num_layers: int = 3, dropout: float = 0.5, pool: str = "mean"):
    a = arch.lower().strip()
    if a in ("graphsage", "sage", "gsage"):
        return GraphSAGE_GC(in_dim, hidden, num_classes,
                            num_layers=num_layers, dropout=dropout, pool=pool)
    raise ValueError(f"Unsupported arch for graph classification: {arch}")
