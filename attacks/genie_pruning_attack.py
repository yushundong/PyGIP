"""
attacks/genie_pruning_attack.py

Pruning attack class compatible with PyGIP BaseAttack API.
"""

from typing import Optional, Dict, Any
import torch
import os
import torch.nn.utils.prune as prune
from sklearn.metrics import roc_auc_score

# Adjust these imports to the actual PyGIP module locations in the repo.
try:
    from pygip.core.base import BaseAttack
    from pygip.data.dataset import Dataset
except Exception:
    from pyGIP.base import BaseAttack  # placeholder
    from pyGIP.dataset import Dataset  # placeholder

import torch_geometric.nn as pyg_nn

class GeniePruningAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(self, dataset: Dataset, attack_node_fraction: float = 0.1, model_path: Optional[str] = None,
                 prune_ratio: float = 0.2, save_pruned: bool = False):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.prune_ratio = prune_ratio
        self.save_pruned = save_pruned

    def attack(self) -> Dict[str, Any]:
        device = self.device
        data = self.graph_data.to(device)
        # Load model
        model = self._load_model()
        if model is None:
            raise RuntimeError("Could not load model for pruning attack")

        model.to(device)
        # Collect pruning targets (GCNConv -> .lin.weight usually)
        params_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, pyg_nn.GCNConv):
                if hasattr(module, "lin"):
                    params_to_prune.append((module.lin, "weight"))

        if len(params_to_prune) == 0:
            raise RuntimeError("No GCNConv linear layers found to prune. Check model architecture.")

        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=self.prune_ratio)

        # Evaluate model (test AUC and watermark AUC if watermark data exists)
        test_auc, wm_auc = self._evaluate_model(model, data)
        results = {
            "dataset": self.dataset.dataset_name if hasattr(self.dataset, "dataset_name") else "unknown",
            "prune_ratio": self.prune_ratio,
            "test_auc": float(test_auc),
            "watermark_auc": float(wm_auc) if wm_auc is not None else None
        }
        # optionally save pruned model
        if self.save_pruned and self.model_path:
            out_path = os.path.splitext(self.model_path)[0] + f"_pruned_{int(self.prune_ratio*100)}.pth"
            torch.save(model.state_dict(), out_path)
            results["pruned_model_path"] = out_path
        return results

    def _load_model(self):
        # As in model_extraction, use dataset or a provided path
        if not self.model_path:
            print("[GeniePruningAttack] No model path provided.")
            return None
        ckpt = torch.load(self.model_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            from models.gcn_link_predictor import GCNLinkPredictor
            in_ch = getattr(self.dataset, "num_features", 64)
            model = GCNLinkPredictor(in_channels=in_ch, hidden_channels=64).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            print("[GeniePruningAttack] Failed to load model:", e)
            return None

    def _evaluate_model(self, model, data):
        model.eval()
        device = self.device
        # if dataset uses train/test splits, use those edges; otherwise, build negative sampling
        train_pos = getattr(data, "train_pos_edge_index", None)
        test_pos = getattr(data, "test_pos_edge_index", None)
        if train_pos is None or test_pos is None:
            # try to use full edges for a simple evaluation
            full_edge_index = data.edge_index
            test_pos = full_edge_index

        z = model.encode(data.x.to(device), getattr(data, "train_pos_edge_index", data.edge_index).to(device))
        pos_logits = model.decode(z, test_pos.to(device)).view(-1).cpu().detach()
        # generate negatives - naive random negs if not provided
        from torch_geometric.utils import negative_sampling
        neg = negative_sampling(edge_index=data.edge_index.to(device), num_nodes=data.num_nodes, num_neg_samples=pos_logits.size(0)).to(device)
        neg_logits = model.decode(z, neg).view(-1).cpu().detach()

        y = torch.cat([torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))]).numpy()
        preds = torch.cat([pos_logits, neg_logits]).numpy()
        try:
            auc = roc_auc_score(y, preds)
        except Exception:
            auc = float("nan")

        # watermark evaluation (if watermark data attached to dataset)
        wm_auc = None
        if hasattr(self.dataset, "watermark_edges") and hasattr(self.dataset, "watermark_labels"):
            with torch.no_grad():
                z_wm = model.encode(data.x.to(device), getattr(data, "train_pos_edge_index", data.edge_index).to(device))
                wm_preds = model.decode(z_wm, self.dataset.watermark_edges.to(device)).view(-1).cpu().numpy()
                try:
                    wm_auc = roc_auc_score(self.dataset.watermark_labels.cpu().numpy(), wm_preds)
                except Exception:
                    wm_auc = float("nan")
        return auc, wm_auc
