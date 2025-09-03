from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base import BaseDefense
from models.attack.my_custom_attack import MyCustomAttack


# --- Simple GCN model (used for defense) ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)


class MyCustomDefense(BaseDefense):
    supported_api_types = {"pyg"}
    supported_datasets = {"Cora"}

    def __init__(self, dataset, defense_node_fraction: float, model_path: str = None):
        super().__init__(dataset, defense_node_fraction)
        self.model_path = model_path
        print("MyCustomDefense initialized")

    def defend(self) -> Dict[str, Any]:
        data = self.graph_data.to(self.device)

        # Step 1: Train target model (victim)
        target_model = self._train_model(data)

        # Step 2: Run attack BEFORE defense
        attack = MyCustomAttack(self.dataset, self.attack_node_fraction)
        print("Running attack BEFORE defense...")
        attack_results_before = attack.attack()
        
        # Step 3: Train defense model (adversarial training)
        defense_model = self._train_model(data, adv_training=True)

        # Step 4: Run attack AFTER defense
        attack = MyCustomAttack(self.dataset, self.attack_node_fraction)
        print("Running attack AFTER defense...")
        attack_results_after = attack.attack()

        return {
            "attack_before_defense": attack_results_before,
            "attack_after_defense": attack_results_after,
        }

    def _train_model(self, data, adv_training=False):
        model = GCN(self.num_features, 16, self.num_classes).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(100):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)

            if adv_training:
                with torch.no_grad():
                    soft_labels = F.softmax(out, dim=1)
                ce_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                adv_loss = F.mse_loss(out[data.train_mask], soft_labels[data.train_mask])
                loss = ce_loss + 0.5 * adv_loss
            else:
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            opt.step()

        return model

def run(self) -> Dict[str, Any]:
    return self.defend()
