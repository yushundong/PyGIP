"""
GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks
Defense implementation following PyGIP framework conventions.

Path: pygip/defense/gnn_fingers_defense.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import copy
import random
import numpy as np
from abc import ABC, abstractmethod

from .base import BaseDefense
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from datasets import Dataset
from .gnn_fingers_models import (
    GCN, GCNMean, GCNDiff, GCNLinkPredictor, 
    Univerifier, get_model_for_task
)
from utils.gnn_fingers_utils import (
    calculate_aruc, plot_robustness_uniqueness_curve,
    create_obfuscated_models, evaluate_fingerprint_verification
)
from .gnn_fingers_protect import (
    FingerprintConstructor, NodeFingerprint, 
    GraphFingerprint, LinkPredictionFingerprint, GraphMatchingFingerprint
)


class GNNFingersDefense(BaseDefense):
    """
    GNNFingers defense mechanism for verifying GNN model ownership.
    
    This defense creates fingerprints that can identify pirated/obfuscated models
    while preserving the original model's utility.
    """
    
    supported_api_types = {"pyg"}
    supported_datasets = {"Cora", "Citeseer", "PubMed", "PROTEINS", "AIDS", "MUTAG", 
                         "CoraGNNFingers", "CiteseerGNNFingers", "PubMedGNNFingers", 
                         "ProteinsGNNFingers", "AidsGNNFingers", "MutagGNNFingers",
                         "PyGIPDatasetAdapter"}
    
    def __init__(self, dataset: Dataset, 
                 task_type: str = "node_classification",
                 num_fingerprints: int = 64,
                 fingerprint_params: Optional[Dict] = None,
                 univerifier_params: Optional[Dict] = None,
                 training_params: Optional[Dict] = None,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize GNNFingers defense.
        
        Args:
            dataset: PyGIP Dataset instance
            task_type: Type of GNN task ("node_classification", "graph_classification", 
                      "link_prediction", "graph_matching")
            num_fingerprints: Number of fingerprints to create
            fingerprint_params: Parameters for fingerprint construction
            univerifier_params: Parameters for univerifier model
            training_params: Training parameters
            device: Computing device
        """
        # We don't use attack_node_fraction for fingerprinting, so set to None
        super().__init__(dataset, attack_node_fraction=None, device=device)
        
        self.task_type = task_type
        self.num_fingerprints = num_fingerprints
        
        # Default parameters
        default_fingerprint_params = self._get_default_fingerprint_params()
        default_univerifier_params = self._get_default_univerifier_params()
        default_training_params = self._get_default_training_params()
        
        # Merge provided parameters with defaults
        self.fingerprint_params = default_fingerprint_params.copy()
        if fingerprint_params:
            self.fingerprint_params.update(fingerprint_params)
            
        self.univerifier_params = default_univerifier_params.copy()
        if univerifier_params:
            self.univerifier_params.update(univerifier_params)
            
        self.training_params = default_training_params.copy()
        if training_params:
            self.training_params.update(training_params)
        
        # Initialize components
        self.target_model = None
        self.fingerprint_constructor = None
        self.univerifier = None
        self.positive_models = []  # Pirated models
        self.negative_models = []  # Independent models
        
        # Training state
        self.training_history = []
        self.converged = False
        self.flag = 0  # Algorithm 1 flag for alternating optimization
        
        self._initialize_fingerprint_constructor()
    
    def _get_default_fingerprint_params(self) -> Dict:
        """Get default fingerprint construction parameters."""
        base_params = {
            'num_fingerprints': self.num_fingerprints,
            'edge_prob': 0.2,
        }
        
        if self.task_type == "node_classification":
            base_params.update({
                'num_nodes': 32,
                'feature_dim': self.num_features
            })
        elif self.task_type == "graph_classification":
            base_params.update({
                'num_fingerprints': self.num_fingerprints,
                'min_nodes': 8,
                'max_nodes': 25,
                'feature_dim': self.num_features
            })
        elif self.task_type == "link_prediction":
            base_params.update({
                'num_nodes': 32,
                'feature_dim': self.num_features,
                'num_edge_samples': 64
            })
        elif self.task_type == "graph_matching":
            base_params.update({
                'num_fingerprint_pairs': self.num_fingerprints,
                'min_nodes': 6,
                'max_nodes': 20,
                'feature_dim': self.num_features
            })
        
        return base_params
    
    def _get_default_univerifier_params(self) -> Dict:
        """Get default univerifier parameters."""
        return {
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3,
            'activation': 'leaky_relu'
        }
    
    def _get_default_training_params(self) -> Dict:
        """Get default training parameters."""
        return {
            'epochs_total': 100,
            'e1': 1,  # Fingerprint optimization epochs per iteration
            'e2': 1,  # Univerifier optimization epochs per iteration
            'alpha': 0.01,  # Fingerprint learning rate
            'beta': 0.001,  # Univerifier learning rate
            'convergence_threshold': 0.001
        }
    
    def _initialize_fingerprint_constructor(self):
        """Initialize fingerprint constructor based on task type."""
        if self.task_type == "node_classification":
            self.fingerprint_constructor = NodeFingerprint(
                num_nodes=self.fingerprint_params['num_nodes'],
                feature_dim=self.fingerprint_params['feature_dim'],
                edge_prob=self.fingerprint_params['edge_prob'],
                device=self.device
            )
        elif self.task_type == "graph_classification":
            self.fingerprint_constructor = GraphFingerprint(
                num_fingerprints=self.fingerprint_params['num_fingerprints'],
                min_nodes=self.fingerprint_params['min_nodes'],
                max_nodes=self.fingerprint_params['max_nodes'],
                feature_dim=self.fingerprint_params['feature_dim'],
                edge_prob=self.fingerprint_params['edge_prob'],
                device=self.device
            )
        elif self.task_type == "link_prediction":
            self.fingerprint_constructor = LinkPredictionFingerprint(
                num_nodes=self.fingerprint_params['num_nodes'],
                feature_dim=self.fingerprint_params['feature_dim'],
                edge_prob=self.fingerprint_params['edge_prob'],
                num_edge_samples=self.fingerprint_params['num_edge_samples'],
                device=self.device
            )
        elif self.task_type == "graph_matching":
            self.fingerprint_constructor = GraphMatchingFingerprint(
                num_fingerprint_pairs=self.fingerprint_params['num_fingerprint_pairs'],
                min_nodes=self.fingerprint_params['min_nodes'],
                max_nodes=self.fingerprint_params['max_nodes'],
                feature_dim=self.fingerprint_params['feature_dim'],
                edge_prob=self.fingerprint_params['edge_prob'],
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def defend(self, attack_method: str = "comprehensive") -> Dict:
        """
        Main defense method implementing GNNFingers framework.
        
        Args:
            attack_method: Type of attack scenario to defend against
                          ("comprehensive", "fine_tuning", "distillation", "partial_retraining")
        
        Returns:
            Dict containing defense results and metrics
        """
        print(f"Starting GNNFingers defense for {self.task_type}")
        print(f"Dataset: {self.dataset.dataset_name}")
        print(f"Attack method: {attack_method}")
        
        # Step 1: Train target model
        print("\n=== Step 1: Training Target Model ===")
        self.target_model = self._train_target_model()
        
        # Step 2: Initialize univerifier
        print("\n=== Step 2: Initializing Univerifier ===")
        self._initialize_univerifier()
        
        # Step 3: Prepare suspect models (simulating attack scenarios)
        print("\n=== Step 3: Preparing Suspect Models ===")
        num_positive, num_negative = self._get_model_counts(attack_method)
        self._prepare_suspect_models(num_positive, num_negative, attack_method)
        
        # Step 4: Train fingerprinting system using Algorithm 1
        print("\n=== Step 4: Training Fingerprinting System ===")
        self._train_fingerprinting_system()
        
        # Step 5: Evaluate defense
        print("\n=== Step 5: Evaluating Defense ===")
        results = self._evaluate_defense()
        
        print(f"\n=== Defense Results ===")
        print(f"AUC Score: {results['auc']:.4f}")
        print(f"ARUC Score: {results['aruc']:.4f}")
        if results['threshold_results']:
            best_result = max(results['threshold_results'], key=lambda x: x['accuracy'])
            print(f"Best Verification Accuracy: {best_result['accuracy']:.4f}")
        
        return results
    
    def _get_model_counts(self, attack_method: str) -> Tuple[int, int]:
        """Get number of positive and negative models based on attack method."""
        if attack_method == "comprehensive":
            return 100, 100  # Full-scale evaluation
        elif attack_method in ["fine_tuning", "distillation", "partial_retraining"]:
            return 50, 50   # Focused evaluation
        else:
            return 20, 20   # Quick evaluation
    
    def _train_target_model(self) -> nn.Module:
        """Train the target model that we want to protect."""
        print("Training target model...")
        
        # Get appropriate model architecture
        model = get_model_for_task(
            task_type=self.task_type,
            input_dim=self.num_features,
            hidden_dim=64,
            output_dim=self.num_classes,
            num_layers=2
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Training logic based on task type
        if self.task_type == "node_classification":
            model = self._train_node_classification_model(model, optimizer)
        elif self.task_type == "graph_classification":
            model = self._train_graph_classification_model(model, optimizer)
        elif self.task_type == "link_prediction":
            model = self._train_link_prediction_model(model, optimizer)
        elif self.task_type == "graph_matching":
            model = self._train_graph_matching_model(model, optimizer)
        
        print("Target model training completed")
        return model
    
    def _train_node_classification_model(self, model: nn.Module, optimizer) -> nn.Module:
        """Train node classification model."""
        data = self.graph_data.to(self.device)
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(data.x, data.edge_index).argmax(dim=1)
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return model
    
    def _train_graph_classification_model(self, model: nn.Module, optimizer) -> nn.Module:
        """Train graph classification model."""
        # Use dataset dataloaders
        try:
            train_loader = self.dataset.get_dataloader(split="train", batch_size=32, shuffle=True)
            val_loader = self.dataset.get_dataloader(split="val", batch_size=32, shuffle=False)
        except Exception as e:
            print(f"WARNING: Failed to get dataloaders for graph classification ({e}), skipping training")
            return model

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
        best_val_acc = 0.0
        best_state = None

        for epoch in range(200):
            model.train()
            total_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                y = batch.y.view(-1).long()
                if y.numel() > 0 and y.min().item() != 0:
                    y = y - y.min()
                loss = F.nll_loss(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            scheduler.step()

            if epoch % 20 == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        pred = out.argmax(dim=1)
                        y_true = batch.y.view(-1).long()
                        if y_true.numel() > 0 and y_true.min().item() != 0:
                            y_true = y_true - y_true.min()
                        correct += pred.eq(y_true).sum().item()
                        total += y_true.size(0)
                val_acc = correct / total if total > 0 else 0.0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                avg_loss = total_loss / max(num_batches, 1)
                print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')

        if best_state is not None:
            model.load_state_dict(best_state)

        return model
    
    def _train_link_prediction_model(self, model: nn.Module, optimizer) -> nn.Module:
        """Train link prediction model."""
        # Ensure dataset has edge splits
        try:
            data = self.graph_data
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                if hasattr(self.dataset, 'prepare_for_link_prediction'):
                    self.dataset.prepare_for_link_prediction()
                    data = self.dataset.graph_data
                else:
                    from torch_geometric.utils import train_test_split_edges, to_undirected, remove_self_loops
                    data.edge_index, _ = remove_self_loops(data.edge_index)
                    data.edge_index = to_undirected(data.edge_index)
                    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
                    self.graph_data = data
        except Exception as e:
            print(f"WARNING: Failed to prepare link prediction splits ({e}), skipping training")
            return model

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        best_val_auc = 0.0
        best_state = None

        def evaluate_auc(m):
            from sklearn.metrics import roc_auc_score, average_precision_score
            m.eval()
            with torch.no_grad():
                emb = m.get_embeddings(data.x.to(self.device), data.train_pos_edge_index.to(self.device))
                pos_pred = m.predict_links(emb, data.val_pos_edge_index.to(self.device))
                neg_pred = m.predict_links(emb, data.val_neg_edge_index.to(self.device))
                pred = torch.cat([pos_pred, neg_pred]).detach().cpu().numpy()
                labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
                try:
                    return roc_auc_score(labels, pred)
                except Exception:
                    return 0.5

        for epoch in range(200):
            model.train()
            total_loss = 0.0
            num_batches = 0

            # Create negatives each epoch
            try:
                neg_edge_index = negative_sampling(
                    edge_index=data.train_pos_edge_index.to(self.device),
                    num_nodes=data.x.size(0),
                    num_neg_samples=data.train_pos_edge_index.size(1),
                    method='sparse'
                )
            except Exception:
                # Fallback dense method
                from torch_geometric.utils import negative_sampling as neg_samp
                neg_edge_index = neg_samp(
                    edge_index=data.train_pos_edge_index.to(self.device),
                    num_nodes=data.x.size(0),
                    num_neg_samples=data.train_pos_edge_index.size(1)
                )

            batch_size = 512
            pos_edges = data.train_pos_edge_index.t()
            neg_edges = neg_edge_index.t()
            max_batches = min(pos_edges.size(0), neg_edges.size(0)) // batch_size
            for i in range(min(max_batches, 10)):
                start = i * batch_size
                end = (i + 1) * batch_size
                optimizer.zero_grad()
                pos_batch = pos_edges[start:end].t().to(self.device)
                neg_batch = neg_edges[start:end].t().to(self.device)
                pos_pred = model(data.x.to(self.device), data.train_pos_edge_index.to(self.device), pos_batch)
                neg_pred = model(data.x.to(self.device), data.train_pos_edge_index.to(self.device), neg_batch)
                pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
                neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            scheduler.step()

            if epoch % 20 == 0:
                val_auc = evaluate_auc(model)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = copy.deepcopy(model.state_dict())
                avg_loss = total_loss / max(num_batches, 1)
                print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}')

        if best_state is not None:
            model.load_state_dict(best_state)

        return model
    
    def _train_graph_matching_model(self, model: nn.Module, optimizer) -> nn.Module:
        """Train graph matching model (pairwise similarity regression)."""
        # Build pairs from dataset
        try:
            all_pairs = self.dataset.create_graph_pairs(num_pairs=600)
        except Exception as e:
            print(f"WARNING: Failed to create graph pairs for matching ({e}), skipping training")
            return model

        # Split pairs
        num_pairs = len(all_pairs)
        indices = list(range(num_pairs))
        random.shuffle(indices)
        train_size = int(0.7 * num_pairs)
        val_size = int(0.15 * num_pairs)

        train_pairs = [all_pairs[i] for i in indices[:train_size]]
        val_pairs = [all_pairs[i] for i in indices[train_size:train_size + val_size]]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        best_val_mse = float('inf')
        best_state = None

        for epoch in range(150):
            model.train()
            total_loss = 0.0
            batches = 0
            random.shuffle(train_pairs)
            for (graph1, graph2), sim in train_pairs[:200]:  # limit per epoch for speed
                try:
                    optimizer.zero_grad()
                    batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long, device=self.device)
                    batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long, device=self.device)
                    d1 = Data(x=graph1.x.to(self.device), edge_index=graph1.edge_index.to(self.device), batch=batch1)
                    d2 = Data(x=graph2.x.to(self.device), edge_index=graph2.edge_index.to(self.device), batch=batch2)
                    pred = model(d1, d2)
                    target = torch.tensor([sim], dtype=torch.float, device=self.device)
                    loss = F.mse_loss(pred.unsqueeze(0), target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batches += 1
                except Exception:
                    continue

            scheduler.step()

            if epoch % 20 == 0:
                model.eval()
                val_mse = 0.0
                cnt = 0
                with torch.no_grad():
                    for (graph1, graph2), sim in val_pairs[:100]:
                        try:
                            batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long, device=self.device)
                            batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long, device=self.device)
                            d1 = Data(x=graph1.x.to(self.device), edge_index=graph1.edge_index.to(self.device), batch=batch1)
                            d2 = Data(x=graph2.x.to(self.device), edge_index=graph2.edge_index.to(self.device), batch=batch2)
                            pred = model(d1, d2)
                            target = torch.tensor([sim], dtype=torch.float, device=self.device)
                            val_mse += F.mse_loss(pred.unsqueeze(0), target).item()
                            cnt += 1
                        except Exception:
                            continue
                val_mse = val_mse / max(cnt, 1)
                avg_loss = total_loss / max(batches, 1)
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_state = copy.deepcopy(model.state_dict())
                print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val MSE: {val_mse:.4f}')

        if best_state is not None:
            model.load_state_dict(best_state)

        return model
    
    def _initialize_univerifier(self):
        """Initialize the univerifier (binary classifier)."""
        # Get sample output to determine input dimension
        sample_output = self.fingerprint_constructor.get_model_outputs(self.target_model)
        input_dim = sample_output.size(0)
        
        self.univerifier = Univerifier(
            input_dim=input_dim,
            hidden_dims=self.univerifier_params['hidden_dims'],
            dropout=self.univerifier_params['dropout']
        ).to(self.device)
        
        print(f"Univerifier initialized with input dimension: {input_dim}")
    
    def _prepare_suspect_models(self, num_positive: int, num_negative: int, attack_method: str):
        """Prepare positive (pirated) and negative (independent) models."""
        print(f"Creating {num_positive} positive and {num_negative} negative models...")
        
        # Create positive models (pirated versions)
        self.positive_models = create_obfuscated_models(
            target_model=self.target_model,
            dataset=self.dataset,
            task_type=self.task_type,
            num_models=num_positive,
            attack_method=attack_method,
            device=self.device
        )
        
        # Create negative models (independent models)
        self.negative_models = []
        for i in range(num_negative):
            # Create independent model with random architecture
            hidden_dim = random.choice([32, 64, 128])
            num_layers = random.choice([2, 3, 4])
            
            neg_model = get_model_for_task(
                task_type=self.task_type,
                input_dim=self.num_features,
                hidden_dim=hidden_dim,
                output_dim=self.num_classes,
                num_layers=num_layers
            ).to(self.device)
            
            # Train independently
            optimizer = torch.optim.Adam(neg_model.parameters(), lr=0.01)
            self._train_independent_model(neg_model, optimizer)
            
            self.negative_models.append(neg_model)
        
        print(f"Created {len(self.positive_models)} positive and {len(self.negative_models)} negative models")
    
    def _train_independent_model(self, model: nn.Module, optimizer):
        """Train an independent model (not derived from target)."""
        if self.task_type == "node_classification":
            data = self.graph_data.to(self.device)
            
            for epoch in range(random.randint(50, 150)):
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
        elif self.task_type == "graph_matching":
            # Train on a small set of random pairs for diversity
            try:
                pairs = self.dataset.create_graph_pairs(num_pairs=200)
            except Exception:
                return
            for epoch in range(random.randint(40, 120)):
                random.shuffle(pairs)
                for (graph1, graph2), sim in pairs[:50]:
                    try:
                        optimizer.zero_grad()
                        batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long, device=self.device)
                        batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long, device=self.device)
                        d1 = Data(x=graph1.x.to(self.device), edge_index=graph1.edge_index.to(self.device), batch=batch1)
                        d2 = Data(x=graph2.x.to(self.device), edge_index=graph2.edge_index.to(self.device), batch=batch2)
                        pred = model(d1, d2)
                        target = torch.tensor([sim], dtype=torch.float, device=self.device)
                        loss = F.mse_loss(pred.unsqueeze(0), target)
                        loss.backward()
                        optimizer.step()
                    except Exception:
                        continue
                if epoch > 30 and random.random() < 0.03:
                    break
        elif self.task_type == "graph_classification":
            try:
                train_loader = self.dataset.get_dataloader(split="train", batch_size=32, shuffle=True)
            except Exception:
                return
            for epoch in range(random.randint(50, 150)):
                for batch in train_loader:
                    batch = batch.to(self.device)
                    model.train()
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    y = batch.y.view(-1).long()
                    if y.numel() > 0 and y.min().item() != 0:
                        y = y - y.min()
                    loss = F.nll_loss(out, y)
                    loss.backward()
                    optimizer.step()
                if epoch > 50 and random.random() < 0.02:
                    break
        elif self.task_type == "link_prediction":
            data = self.graph_data
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                if hasattr(self.dataset, 'prepare_for_link_prediction'):
                    self.dataset.prepare_for_link_prediction()
                    data = self.dataset.graph_data
                else:
                    return
            for epoch in range(random.randint(50, 150)):
                model.train()
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=data.train_pos_edge_index.to(self.device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(1000, data.train_pos_edge_index.size(1)),
                        method='sparse'
                    )
                except Exception:
                    from torch_geometric.utils import negative_sampling as neg_samp
                    neg_edge_index = neg_samp(
                        edge_index=data.train_pos_edge_index.to(self.device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(1000, data.train_pos_edge_index.size(1))
                    )
                batch_size = 256
                pos_edges = data.train_pos_edge_index.t()
                neg_edges = neg_edge_index.t()
                num_batches = min(pos_edges.size(0), neg_edges.size(0)) // batch_size
                for i in range(min(num_batches, 5)):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    optimizer.zero_grad()
                    pos_batch = pos_edges[start:end].t().to(self.device)
                    neg_batch = neg_edges[start:end].t().to(self.device)
                    pos_pred = model(data.x.to(self.device), data.train_pos_edge_index.to(self.device), pos_batch)
                    neg_pred = model(data.x.to(self.device), data.train_pos_edge_index.to(self.device), neg_batch)
                    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
                    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                    loss = pos_loss + neg_loss
                    loss.backward()
                    optimizer.step()
                if epoch > 50 and random.random() < 0.02:
                    break
        # Add other task implementations as needed
    
    def _train_fingerprinting_system(self):
        """Train fingerprinting system using Algorithm 1 (Joint alternating optimization)."""
        print("Training fingerprinting system with Algorithm 1...")
        
        univerifier_optimizer = torch.optim.Adam(
            self.univerifier.parameters(), 
            lr=self.training_params['beta']
        )
        
        epoch = 0
        while epoch < self.training_params['epochs_total'] and not self.converged:
            # Collect fingerprint outputs from all models
            fingerprint_outputs = self._collect_fingerprint_outputs()
            
            # Calculate unified loss
            loss, predictions, labels = self._calculate_unified_loss(fingerprint_outputs)
            
            if self.flag == 0:
                # Update fingerprints for e1 epochs
                for _ in range(self.training_params['e1']):
                    self.fingerprint_constructor.optimize_fingerprint(
                        loss=loss,
                        alpha=self.training_params['alpha'],
                        target_model=self.target_model,
                        positive_models=self.positive_models,
                        negative_models=self.negative_models,
                        univerifier=self.univerifier
                    )
                self.flag = 1
                operation = "Fingerprints"
            else:
                # Update univerifier for e2 epochs
                for _ in range(self.training_params['e2']):
                    univerifier_optimizer.zero_grad()
                    
                    # Recalculate loss for current fingerprints
                    fingerprint_outputs = self._collect_fingerprint_outputs()
                    loss, predictions, labels = self._calculate_unified_loss(fingerprint_outputs)
                    
                    loss.backward()
                    univerifier_optimizer.step()
                
                self.flag = 0
                operation = "Univerifier"
            
            # Calculate accuracy
            if predictions is not None and labels is not None:
                acc = (predictions.argmax(dim=1) == labels).float().mean()
            else:
                acc = 0.0
            
            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | {operation:12} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
            
            self.training_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'accuracy': acc.item(),
                'operation': operation
            })
            
            # Check convergence
            if len(self.training_history) >= 20:
                recent_losses = [h['loss'] for h in self.training_history[-10:]]
                if max(recent_losses) - min(recent_losses) < self.training_params['convergence_threshold']:
                    self.converged = True
                    print(f"Converged at epoch {epoch}")
            
            epoch += 1
    
    def _collect_fingerprint_outputs(self) -> Dict:
        """Collect outputs from all models using fingerprints."""
        try:
            # Target model output
            target_out = self.fingerprint_constructor.get_model_outputs(self.target_model)
            
            # Sample models to avoid memory issues
            positive_sample = random.sample(
                self.positive_models, 
                min(50, len(self.positive_models))
            )
            negative_sample = random.sample(
                self.negative_models, 
                min(50, len(self.negative_models))
            )
            
            # Positive model outputs
            positive_outs = []
            for pos_model in positive_sample:
                try:
                    pos_out = self.fingerprint_constructor.get_model_outputs(pos_model)
                    if pos_out is not None and pos_out.numel() > 0:
                        positive_outs.append(pos_out)
                except:
                    continue
            
            # Negative model outputs
            negative_outs = []
            for neg_model in negative_sample:
                try:
                    neg_out = self.fingerprint_constructor.get_model_outputs(neg_model)
                    if neg_out is not None and neg_out.numel() > 0:
                        negative_outs.append(neg_out)
                except:
                    continue
            
            return {
                'target': target_out,
                'positive': positive_outs,
                'negative': negative_outs
            }
        except Exception as e:
            print(f"Error collecting fingerprint outputs: {e}")
            return {
                'target': torch.randn(10, device=self.device),
                'positive': [],
                'negative': []
            }
    
    def _calculate_unified_loss(self, fingerprint_outputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate unified loss L as per Algorithm 1."""
        all_outputs = []
        labels = []
        
        # Target model (positive)
        if 'target' in fingerprint_outputs and fingerprint_outputs['target'] is not None:
            all_outputs.append(fingerprint_outputs['target'])
            labels.append(1)
        
        # Positive models
        for pos_out in fingerprint_outputs.get('positive', []):
            all_outputs.append(pos_out)
            labels.append(1)
        
        # Negative models
        for neg_out in fingerprint_outputs.get('negative', []):
            all_outputs.append(neg_out)
            labels.append(0)
        
        if len(all_outputs) < 2:
            # Return dummy values when insufficient data
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            dummy_pred = torch.tensor([[0.5, 0.5]], requires_grad=True, device=self.device)
            dummy_labels = torch.tensor([0], dtype=torch.long, device=self.device)
            return dummy_loss, dummy_pred, dummy_labels
        
        # Ensure all outputs have same size
        min_size = min(out.size(0) for out in all_outputs if out.numel() > 0)
        all_outputs = [out[:min_size] for out in all_outputs if out.numel() > 0]
        
        if not all_outputs:
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            dummy_pred = torch.tensor([[0.5, 0.5]], requires_grad=True, device=self.device)
            dummy_labels = torch.tensor([0], dtype=torch.long, device=self.device)
            return dummy_loss, dummy_pred, dummy_labels
        
        batch_outputs = torch.stack(all_outputs)
        batch_labels = torch.tensor(labels[:len(all_outputs)], dtype=torch.long, device=self.device)
        
        # Get univerifier predictions
        predictions = self.univerifier(batch_outputs)
        
        # Calculate unified loss
        loss = F.cross_entropy(predictions, batch_labels)
        
        return loss, predictions, batch_labels
    
    def _evaluate_defense(self) -> Dict:
        """Evaluate the defense performance."""
        # Create fresh test models
        test_positive_models = create_obfuscated_models(
            target_model=self.target_model,
            dataset=self.dataset,
            task_type=self.task_type,
            num_models=10,
            attack_method="comprehensive",
            device=self.device
        )
        
        test_negative_models = []
        for _ in range(10):
            model = get_model_for_task(
                task_type=self.task_type,
                input_dim=self.num_features,
                hidden_dim=random.choice([32, 64, 128]),
                output_dim=self.num_classes,
                num_layers=random.choice([2, 3])
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            self._train_independent_model(model, optimizer)
            test_negative_models.append(model)
        
        # Evaluate verification performance
        return evaluate_fingerprint_verification(
            univerifier=self.univerifier,
            fingerprint_constructor=self.fingerprint_constructor,
            positive_models=test_positive_models,
            negative_models=test_negative_models,
            device=self.device
        )
    
    def verify_ownership(self, suspect_model: nn.Module, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Verify if a suspect model is pirated from our target model.
        
        Args:
            suspect_model: Model to verify
            threshold: Decision threshold
        
        Returns:
            Tuple of (is_pirated, confidence_score)
        """
        try:
            suspect_outputs = self.fingerprint_constructor.get_model_outputs(suspect_model)
            
            self.univerifier.eval()
            with torch.no_grad():
                prediction = self.univerifier(suspect_outputs.unsqueeze(0))
                confidence = prediction[0, 1].item()  # Positive class probability
            
            is_pirated = confidence > threshold
            return is_pirated, confidence
        
        except Exception as e:
            print(f"Error in ownership verification: {e}")
            return False, 0.0
    
    def _load_model(self):
        """Load a pre-trained model (PyGIP interface requirement)."""
        # Implementation for loading pre-trained models
        pass
    
    def _train_defense_model(self):
        """Train defense model (PyGIP interface requirement)."""
        return self._train_fingerprinting_system()
    
    def _train_surrogate_model(self):
        """Train surrogate model (PyGIP interface requirement)."""
        # For GNNFingers, this would be the suspect models
        return self._prepare_suspect_models(50, 50, "comprehensive")