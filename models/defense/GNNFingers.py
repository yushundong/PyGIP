"""GNNFingers ownership verification defense.

This module implements the core workflow described in the GNNFingers paper:

1. Train (or load) an owner model on the supplied graph dataset.
2. Optimise lightweight fingerprint vectors for sampled node pairs using the
   intermediate representations of the owner model.
3. Persist and verify those fingerprints against a suspect model by checking
   whether the same hidden-space patterns are reproduced.

The implementation follows the structure expected by :class:`BaseDefense` so it
integrates cleanly with the rest of PyGIP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from models.defense.base import BaseDefense


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FingerprintRecord:
    """Description of a single fingerprint pair.

    Attributes
    ----------
    anchor_i, anchor_j:
        Indices of the anchor nodes used when deriving the fingerprint. Each
        refers to a node in the original transductive graph provided by the
        dataset.
    fingerprint_i, fingerprint_j:
        Normalised tensors capturing the expected hidden representation pattern
        for ``anchor_i`` and ``anchor_j`` respectively.
    same_class:
        Whether the anchor nodes share the same ground-truth class label. The
        sign of the verification score depends on this flag, matching the
        objective in the paper which uses both positive and negative pairs.
    radius:
        The hop radius that was considered when sampling the anchor nodes. This
        is metadata only (the current implementation operates on the global
        transductive graph) but is persisted for reproducibility.
    """

    anchor_i: int
    anchor_j: int
    fingerprint_i: Tensor
    fingerprint_j: Tensor
    same_class: bool
    radius: int

    def serialise(self) -> Dict[str, object]:
        """Return a CPU serialisable payload for persistence."""

        return {
            "anchor_i": self.anchor_i,
            "anchor_j": self.anchor_j,
            "same_class": self.same_class,
            "radius": self.radius,
            "fingerprint_i": self.fingerprint_i.detach().cpu(),
            "fingerprint_j": self.fingerprint_j.detach().cpu(),
        }


# ---------------------------------------------------------------------------
# Backbone helper
# ---------------------------------------------------------------------------


class _GNNFingersGCN(nn.Module):
    """Two-layer GCN that exposes intermediate representations."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_hidden: bool = False,
        layer_index: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with optional intermediate activations."""

        hidden_states: List[Tensor] = []

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        hidden_states.append(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.conv2(h, edge_index)
        hidden_states.append(out)

        if return_hidden:
            idx = max(0, min(layer_index, len(hidden_states) - 1))
            return out, hidden_states[idx]
        return out, torch.empty(0, device=out.device)


# ---------------------------------------------------------------------------
# Main defense
# ---------------------------------------------------------------------------


class GNNFingers(BaseDefense):
    """Implementation of the GNNFingers fingerprinting defense."""

    supported_api_types = {"pyg"}

    def __init__(
        self,
        dataset,
        attack_node_fraction: float,
        *,
        fingerprint_budget: int = 64,
        fingerprint_lr: float = 0.75,
        fingerprint_steps: int = 16,
        fingerprint_layer: int = 1,
        fingerprint_radius: int = 2,
        hidden_channels: int = 128,
        owner_epochs: int = 200,
        verification_threshold: float = 0.6,
        model_path: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        super().__init__(dataset, attack_node_fraction)

        if dataset.api_type != "pyg":
            raise ValueError("GNNFingers currently supports datasets loaded with the PyG API.")
        if not isinstance(dataset.graph_data, Data):
            raise TypeError("Expected dataset.graph_data to be an instance of torch_geometric.data.Data.")

        self.data = dataset.graph_data
        self.fingerprint_budget = int(fingerprint_budget)
        self.fingerprint_lr = fingerprint_lr
        self.fingerprint_steps = int(fingerprint_steps)
        self.fingerprint_layer = int(fingerprint_layer)
        self.fingerprint_radius = int(fingerprint_radius)
        self.hidden_channels = int(hidden_channels)
        self.owner_epochs = int(owner_epochs)
        self.verification_threshold = float(verification_threshold)
        self.model_path = model_path
        self.save_dir = save_dir

        self._fingerprints: List[FingerprintRecord] = []
        self._target_model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def defend(self) -> Dict[str, object]:
        """Run the full GNNFingers workflow and return summary metrics."""

        model = self._load_model()
        if model is None:
            model = self._train_target_model()

        fingerprints = self._build_fingerprints(model)
        verification = self.verify(model, fingerprints=fingerprints)

        metrics = {
            "target_accuracy": self._evaluate_model(model, mask_attr="test_mask"),
            "fingerprint_count": len(fingerprints),
            "verification": verification,
        }

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            registry_path = os.path.join(self.save_dir, "fingerprints.pt")
            self.register(registry_path, fingerprints)
            metrics["registry_path"] = registry_path

        return metrics

    def register(self, path: str, fingerprints: Optional[Sequence[FingerprintRecord]] = None) -> None:
        """Persist fingerprints to ``path`` using ``torch.save``."""

        records = list(fingerprints) if fingerprints is not None else self._fingerprints
        if not records:
            raise ValueError("No fingerprints available to register. Run `defend()` first or provide a list.")

        payload = [record.serialise() for record in records]
        torch.save(payload, path)

    def verify(
        self,
        suspect_model: nn.Module,
        *,
        fingerprints: Optional[Sequence[FingerprintRecord]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        """Verify model ownership against the provided fingerprint registry."""

        if suspect_model is None:
            raise ValueError("`suspect_model` must be a trained torch.nn.Module instance")

        records = list(fingerprints) if fingerprints is not None else self._fingerprints
        if not records:
            raise ValueError("Verification requires at least one fingerprint record.")

        data = self._data_to_device()
        suspect_model = suspect_model.to(self.device)
        suspect_model.eval()

        with torch.no_grad():
            logits, hidden = self._forward_with_hidden(suspect_model, data)

        mask = getattr(data, "test_mask", None)
        accuracy = None
        if mask is not None and mask.numel() == data.num_nodes:
            accuracy = self._accuracy_from_logits(logits, data.y, mask)

        per_pair_scores: List[float] = []
        expected_signs: List[float] = []
        for record in records:
            hi = hidden[record.anchor_i]
            hj = hidden[record.anchor_j]
            fi = record.fingerprint_i.to(self.device)
            fj = record.fingerprint_j.to(self.device)

            score_i = F.cosine_similarity(hi.unsqueeze(0), fi.unsqueeze(0), dim=-1).item()
            score_j = F.cosine_similarity(hj.unsqueeze(0), fj.unsqueeze(0), dim=-1).item()

            sign = 1.0 if record.same_class else -1.0
            per_pair_scores.append(0.5 * (score_i + sign * score_j))
            expected_signs.append(sign)

        used_threshold = self.verification_threshold if threshold is None else float(threshold)
        mean_score = float(torch.tensor(per_pair_scores).mean().item())
        verdict = mean_score >= used_threshold

        return {
            "mean_score": mean_score,
            "threshold": used_threshold,
            "verified": verdict,
            "pair_scores": per_pair_scores,
            "expected_signs": expected_signs,
            "suspect_accuracy": accuracy,
        }

    # ------------------------------------------------------------------
    # BaseDefense hooks
    # ------------------------------------------------------------------
    def _load_model(self) -> Optional[nn.Module]:
        """Load a pre-trained owner model if ``model_path`` is provided."""

        if not self.model_path:
            return None
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"No model checkpoint found at: {self.model_path}")

        model = _GNNFingersGCN(self.num_features, self.hidden_channels, self.num_classes)
        state = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state)
        self._target_model = model.to(self.device)
        return self._target_model

    def _train_target_model(self) -> nn.Module:
        """Train the owner model used to derive fingerprints."""

        data = self._data_to_device()
        model = _GNNFingersGCN(self.num_features, self.hidden_channels, self.num_classes).to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        train_mask = getattr(data, "train_mask", None)
        if train_mask is None or int(train_mask.sum()) == 0:
            raise ValueError("Dataset requires a populated `train_mask` for supervised training.")

        val_mask = getattr(data, "val_mask", None)
        best_state: Optional[Dict[str, Tensor]] = None
        best_val = float("-inf")

        for _ in range(self.owner_epochs):
            model.train()
            optimiser.zero_grad()
            logits, _ = model(data.x, data.edge_index, return_hidden=True, layer_index=self.fingerprint_layer)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
            loss.backward()
            optimiser.step()

            if val_mask is not None and int(val_mask.sum()) > 0:
                model.eval()
                with torch.no_grad():
                    val_logits, _ = model(
                        data.x, data.edge_index, return_hidden=True, layer_index=self.fingerprint_layer
                    )
                val_acc = self._accuracy_from_logits(val_logits, data.y, val_mask)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        self._target_model = model
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(model.state_dict(), self.model_path)

        return model

    def _train_defense_model(self) -> nn.Module:
        """Alias for :meth:`_train_target_model` as the owner model is the defense."""

        if self._target_model is None:
            return self._train_target_model()
        return self._target_model

    def _train_surrogate_model(self) -> Optional[nn.Module]:
        """GNNFingers does not train a separate surrogate model."""

        return None

    # ------------------------------------------------------------------
    # Fingerprint construction
    # ------------------------------------------------------------------
    def _build_fingerprints(self, model: nn.Module) -> List[FingerprintRecord]:
        """Generate fingerprints following Algorithms 3–5 of the paper."""

        data = self._data_to_device()
        model.eval()
        with torch.no_grad():
            _, hidden = self._forward_with_hidden(model, data)

        node_pairs = self._sample_node_pairs(self.fingerprint_budget)
        records: List[FingerprintRecord] = []

        for anchor_i, anchor_j in node_pairs:
            hi = hidden[anchor_i]
            hj = hidden[anchor_j]
            same_class = bool(data.y[anchor_i].item() == data.y[anchor_j].item())
            fi, fj = self._optimise_pair(hi, hj, same_class)

            records.append(
                FingerprintRecord(
                    anchor_i=int(anchor_i),
                    anchor_j=int(anchor_j),
                    fingerprint_i=fi.detach().cpu(),
                    fingerprint_j=fj.detach().cpu(),
                    same_class=same_class,
                    radius=self.fingerprint_radius,
                )
            )

        self._fingerprints = records
        return records

    def _sample_node_pairs(self, budget: int) -> List[Tuple[int, int]]:
        """Sample node pairs respecting the requested class ratio."""

        labels = self.data.y.cpu()
        nodes = torch.arange(self.num_nodes)
        same_budget = int(round(budget * 0.5))
        diff_budget = budget - same_budget

        rng = torch.Generator().manual_seed(int(torch.randint(0, 1_000_000, (1,)).item()))
        pairs: List[Tuple[int, int]] = []
        same_count = 0
        diff_count = 0
        attempts = 0
        max_attempts = max(1000, budget * 50)

        while len(pairs) < budget and attempts < max_attempts:
            attempts += 1
            idx = torch.randint(0, nodes.numel(), (2,), generator=rng)
            i, j = int(nodes[idx[0]]), int(nodes[idx[1]])
            if i == j:
                continue
            same = labels[i].item() == labels[j].item()
            if same and same_count < same_budget:
                pairs.append((i, j))
                same_count += 1
            elif (not same) and diff_count < diff_budget:
                pairs.append((i, j))
                diff_count += 1

        if len(pairs) < budget:
            raise RuntimeError("Unable to sample the requested number of node pairs with the available labels.")

        return pairs

    def _optimise_pair(self, hi: Tensor, hj: Tensor, same_class: bool) -> Tuple[Tensor, Tensor]:
        """Optimise fingerprint vectors for the supplied hidden representations."""

        device = hi.device
        hi = hi.detach()
        hj = hj.detach()

        fi = nn.Parameter(F.normalize(torch.randn_like(hi), p=2, dim=0))
        fj = nn.Parameter(F.normalize(torch.randn_like(hj), p=2, dim=0))
        optimiser = torch.optim.SGD([fi, fj], lr=self.fingerprint_lr)
        target_sign = 1.0 if same_class else -1.0

        for _ in range(self.fingerprint_steps):
            optimiser.zero_grad()

            sim_i = F.cosine_similarity(fi.unsqueeze(0), hi.unsqueeze(0), dim=-1)
            sim_j = F.cosine_similarity(fj.unsqueeze(0), hj.unsqueeze(0), dim=-1)

            loss = -(sim_i + target_sign * sim_j).mean()
            loss.backward()
            optimiser.step()

            with torch.no_grad():
                fi.copy_(self._project_vector(fi, device=device))
                fj.copy_(self._project_vector(fj, device=device))

        return fi.detach(), fj.detach()

    @staticmethod
    def _project_vector(vec: Tensor, *, device: torch.device) -> Tensor:
        r"""Project ``vec`` onto the unit ``\ell_2`` ball."""

        norm = vec.norm(p=2)
        if norm.item() == 0:
            return torch.zeros_like(vec, device=device)
        return vec / norm

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _data_to_device(self) -> Data:
        data = self.data.clone() if hasattr(self.data, "clone") else self.data
        return data.to(self.device)

    def _forward_with_hidden(self, model: nn.Module, data: Data) -> Tuple[Tensor, Tensor]:
        """Call ``model`` and return logits and the selected hidden layer."""

        forward = getattr(model, "forward")
        try:
            logits, hidden = forward(
                data.x, data.edge_index, return_hidden=True, layer_index=self.fingerprint_layer
            )
        except TypeError:
            logits, hidden = forward(data.x, data.edge_index, return_hidden=True)

        if hidden is None or hidden.numel() == 0:
            raise RuntimeError(
                "Model must support `return_hidden=True` and return non-empty hidden representations for fingerprinting."
            )
        return logits, hidden

    def _accuracy_from_logits(self, logits: Tensor, labels: Tensor, mask: Tensor) -> float:
        pred = logits.argmax(dim=-1)
        mask = mask.to(logits.device)
        return float((pred[mask] == labels[mask]).float().mean().item())

    def _evaluate_model(self, model: nn.Module, mask_attr: str = "test_mask") -> Optional[float]:
        data = self._data_to_device()
        mask = getattr(data, mask_attr, None)
        if mask is None or int(mask.sum()) == 0:
            return None
        model.eval()
        with torch.no_grad():
            logits, _ = self._forward_with_hidden(model, data)
        return self._accuracy_from_logits(logits, data.y, mask)


__all__ = ["GNNFingers"]
