import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.attack.base import BaseAttack


# --- Minimal GCN model ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)


class MyCustomAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = {"Cora"}

    def __init__(self, dataset, attack_node_fraction=0.1, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        data = self.graph_data.to(self.device)

        # 1. Train target (victim) model
        victim = self._train_model(data)

        # 2. Query victim for soft labels
        victim.eval()
        with torch.no_grad():
            soft_labels = F.softmax(victim(data.x, data.edge_index), dim=1)

        # 3. Train surrogate model on victim’s outputs
        surrogate = self._train_model(data, labels=soft_labels)

        # 4. Evaluate surrogate accuracy
        acc = self._evaluate(surrogate, data)

        return {"surrogate_accuracy": acc}

    # === Helpers ===
    def _train_model(self, data, labels=None):
        model = GCN(self.num_features, 16, self.num_classes).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(100):  # keep it short
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)

            if labels is None:  # train victim
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            else:  # train surrogate
                loss = F.mse_loss(out[data.train_mask], labels[data.train_mask])

            loss.backward()
            opt.step()
        return model

    def _evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = correct / int(data.test_mask.sum())
        return acc.item()
