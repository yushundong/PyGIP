"""
attacks/genie_model_extraction.py

Model Extraction attack implementation adapted from the GENIE reproduction code.
Implements an attack class compatible with the PyGIP BaseAttack API described in the README.
"""

from typing import Optional, Dict, Any
import torch
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Adjust these imports to the actual PyGIP module locations in the repo.
# Example (change if necessary):
# from pygip.core.base import BaseAttack
# from pygip.data.dataset import Dataset
try:
    from pygip.core.base import BaseAttack  # try canonical import
    from pygip.data.dataset import Dataset
except Exception:
    # Fallback names used in the README; adjust when integrating
    from pyGIP.base import BaseAttack  # placeholder - replace with real path
    from pyGIP.dataset import Dataset  # placeholder

# If the repo's BaseAttack is under a different package path, update above.
# The rest of the code implements a self-contained extraction flow.

class GenieModelExtraction(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = set()  # supports all datasets by default

    def __init__(self, dataset: Dataset, attack_node_fraction: float = 0.05, model_path: Optional[str] = None):
        super().__init__(dataset, attack_node_fraction, model_path)
        # You can add extra parameters (e.g. surrogate hyperparams) here.
        self.query_ratio = attack_node_fraction
        # surrogate params
        self.surrogate_epochs = 50
        self.surrogate_lr = 0.01
        self.hidden_dim = 64

    def attack(self) -> Dict[str, Any]:
        """Run the model extraction attack and return metrics dict."""
        print(f"[GenieModelExtraction] Running on device {self.device}")
        # Access graph from self.graph_data (PyG Data)
        data = self.graph_data
        num_nodes = data.num_nodes
        # Build features (if not present, generate node2vec or random features)
        if getattr(data, "x", None) is None:
            print("[GenieModelExtraction] No node features found. Using random features.")
            data.x = torch.randn((num_nodes, 64))

        # Load teacher/watermarked model: respect model_path if provided
        teacher_model = self._load_model()
        if teacher_model is None:
            raise RuntimeError("Could not load teacher model for extraction")

        teacher_model.eval()
        device = self.device
        data = data.to(device)
        x = data.x.to(device)
        full_edge_index = data.edge_index.to(device)

        # Sample edges to query teacher (positive edges)
        pos_edge_index = full_edge_index  # for simplicity, sample subset below
        num_pos = pos_edge_index.size(1)
        sample_size = max(1, int(num_pos * self.query_ratio))
        cols = random.sample(range(num_pos), sample_size)
        sampled_pos = pos_edge_index[:, cols].to(device)

        # split train/val for surrogate
        pos_list = [(u.item(), v.item()) for u, v in zip(sampled_pos[0], sampled_pos[1])]
        if len(pos_list) < 2:
            train_pos = pos_list; val_pos = pos_list
        else:
            train_pos, val_pos = train_test_split(pos_list, test_size=0.2, random_state=42)
        train_pos_index = torch.tensor(train_pos, dtype=torch.long).t().contiguous().to(device)
        val_pos_index = torch.tensor(val_pos, dtype=torch.long).t().contiguous().to(device)

        # Query teacher for labels (teacher must implement encode/decode API)
        teacher_logits_train = self._query_teacher(teacher_model, train_pos_index, x, full_edge_index)
        teacher_logits_val = self._query_teacher(teacher_model, val_pos_index, x, full_edge_index)

        # Convert logits to probabilities/binary labels for surrogate training
        train_targets = (torch.sigmoid(teacher_logits_train) > 0.5).float()
        val_targets = (torch.sigmoid(teacher_logits_val) > 0.5).float()

        # Train surrogate (simple PyTorch loop)
        surrogate = self._train_surrogate(x, full_edge_index, train_pos_index, train_targets,
                                         val_pos_index, val_targets)

        # Evaluate surrogate on random negatives
        from torch_geometric.utils import negative_sampling
        neg_edges = negative_sampling(edge_index=full_edge_index, num_nodes=num_nodes, num_neg_samples=sampled_pos.size(1)).to(device)
        test_auc = self._eval_surrogate_auc(surrogate, full_edge_index, sampled_pos.to(device), neg_edges, x)

        results = {
            "dataset": self.dataset.dataset_name if hasattr(self.dataset, "dataset_name") else "unknown",
            "query_ratio": self.query_ratio,
            "surrogate_test_auc": float(test_auc)
        }
        return results

    def _load_model(self):
        """Load teacher/watermarked model from model_path or dataset default."""
        if not self.model_path:
            print("[GenieModelExtraction] No model_path passed. Attempting dataset default (not implemented).")
            return None
        # load checkpoint (user/maintainer may need to adapt path & loading)
        ckpt = torch.load(self.model_path, map_location=self.device)
        # you may need to reconstruct model architecture depending on checkpoint
        # For integration: prefer using a loader utility in PyGIP if available
        try:
            # If model_state exists in checkpoint
            state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            # The model class must be available; here we assume a GCNLinkPredictor class
            from models.gcn_link_predictor import GCNLinkPredictor
            in_ch = getattr(self.dataset, "num_features", 64)
            model = GCNLinkPredictor(in_channels=in_ch, hidden_channels=self.hidden_dim).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            print("[GenieModelExtraction] Failed to reconstruct teacher model:", e)
            return None

    @torch.no_grad()
    def _query_teacher(self, teacher, edge_label_index, features, full_edge_index):
        teacher.eval()
        z = teacher.encode(features, full_edge_index)
        logits = teacher.decode(z, edge_label_index)
        return logits.view(-1)

    def _train_surrogate(self, x, full_edge_index, train_edge_index, train_targets, val_edge_index, val_targets):
        # Minimal surrogate: same API as your local script, simplified
        from models.gcn_link_predictor import GCNLinkPredictor
        device = self.device
        model = GCNLinkPredictor(in_channels=x.size(1), hidden_channels=self.hidden_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.surrogate_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        best_auc = 0.0
        for epoch in range(1, self.surrogate_epochs + 1):
            model.train()
            opt.zero_grad()
            z = model.encode(x, full_edge_index)
            logits = model.decode(z, train_edge_index).view(-1)
            loss = criterion(logits, train_targets.to(device))
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def _eval_surrogate_auc(self, model, full_edge_index, pos_edge_index, neg_edge_index, features):
        model.eval()
        z = model.encode(features, full_edge_index)
        pos_score = torch.sigmoid(model.decode(z, pos_edge_index)).view(-1).cpu().numpy()
        neg_score = torch.sigmoid(model.decode(z, neg_edge_index)).view(-1).cpu().numpy()
        import numpy as np
        y_true = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])
        y_pred = np.concatenate([pos_score, neg_score])
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return float("nan")
