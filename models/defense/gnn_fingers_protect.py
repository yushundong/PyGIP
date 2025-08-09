"""
Core fingerprinting construction and verification algorithms for GNNFingers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from typing import List, Tuple, Dict, Optional, Union
import copy
import random
import numpy as np
from abc import ABC, abstractmethod


class FingerprintConstructor(ABC):
    """Abstract base class for fingerprint construction."""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
    
    @abstractmethod
    def get_model_outputs(self, model: nn.Module) -> torch.Tensor:
        """Get model outputs for fingerprints."""
        pass
    
    @abstractmethod
    def optimize_fingerprint(self, loss: torch.Tensor, alpha: float, 
                           target_model: nn.Module, positive_models: List[nn.Module], 
                           negative_models: List[nn.Module], univerifier: Optional[nn.Module] = None):
        """Optimize fingerprint based on loss."""
        pass


class NodeFingerprint(FingerprintConstructor):
    """Fingerprint constructor for node classification tasks."""
    
    def __init__(self, num_nodes: int = 32, feature_dim: int = 1433, 
                 edge_prob: float = 0.15, device: torch.device = torch.device('cpu')):
        super().__init__(device)
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.edge_prob = edge_prob
        self.fingerprint = self._create_random_graph()

    def _create_random_graph(self) -> Data:
        """Create random graph fingerprint."""
        x = torch.randn(self.num_nodes, self.feature_dim, 
                       requires_grad=True, device=self.device)

        # Initialize adjacency with specified probability
        adj_prob = torch.rand(self.num_nodes, self.num_nodes)
        adj_matrix = (adj_prob < self.edge_prob).float()
        adj_matrix = torch.triu(adj_matrix, diagonal=1)
        adj_matrix = adj_matrix + adj_matrix.t()

        # Ensure connectivity
        for i in range(min(5, self.num_nodes - 1)):
            j = (i + 1) % self.num_nodes
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        edge_index = adj_matrix.nonzero().t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def get_model_outputs(self, model: nn.Module, num_sampled_nodes: int = 10, require_grad: bool = False) -> torch.Tensor:
        """Get model outputs for sampled nodes."""
        model.eval()
        if require_grad:
            outputs = model(self.fingerprint.x.to(self.device), 
                            self.fingerprint.edge_index.to(self.device))
        else:
            with torch.no_grad():
                outputs = model(self.fingerprint.x.to(self.device), 
                                self.fingerprint.edge_index.to(self.device))
        num_nodes = min(num_sampled_nodes, outputs.size(0))
        sampled_indices = torch.randperm(outputs.size(0))[:num_nodes]
        return outputs[sampled_indices].flatten()

    def optimize_fingerprint(self, loss: torch.Tensor, alpha: float,
                           target_model: nn.Module, positive_models: List[nn.Module],
                           negative_models: List[nn.Module], univerifier: Optional[nn.Module] = None):
        """Optimize node features and graph structure."""
        if self.fingerprint.x.requires_grad:
            params_to_optimize = [self.fingerprint.x]
            optimizer = torch.optim.Adam(params_to_optimize, lr=alpha)
            
            optimizer.zero_grad()
            # Recalculate loss for current fingerprint with gradient
            all_outputs = []
            labels = []
            # Target
            try:
                out = self.get_model_outputs(target_model, require_grad=True)
                all_outputs.append(out); labels.append(1)
            except:
                pass
            for pos_model in random.sample(positive_models, min(8, len(positive_models))):
                try:
                    out = self.get_model_outputs(pos_model, require_grad=True)
                    all_outputs.append(out); labels.append(1)
                except:
                    continue
            for neg_model in random.sample(negative_models, min(8, len(negative_models))):
                try:
                    out = self.get_model_outputs(neg_model, require_grad=True)
                    all_outputs.append(out); labels.append(0)
                except:
                    continue
            if len(all_outputs) >= 2 and univerifier is not None:
                min_size = min(t.size(0) for t in all_outputs)
                batch_outputs = torch.stack([t[:min_size] for t in all_outputs])
                batch_labels = torch.tensor(labels[:len(all_outputs)], dtype=torch.long, device=self.device)
                preds = univerifier(batch_outputs)
                current_loss = F.cross_entropy(preds, batch_labels)
                current_loss.backward()
                # Apply edge update strategy using gradients on x
                self._update_graph_structure()
                optimizer.step()

    def _collect_model_outputs(self, target_model: nn.Module, 
                             positive_models: List[nn.Module], 
                             negative_models: List[nn.Module]) -> Tuple[List, List]:
        """Collect outputs from all models."""
        all_outputs = []
        labels = []

        # Target model
        try:
            target_out = self.get_model_outputs(target_model)
            if target_out is not None and target_out.numel() > 0:
                all_outputs.append(target_out)
                labels.append(1)
        except:
            pass

        # Sample models to avoid memory issues
        pos_sample = random.sample(positive_models, min(8, len(positive_models)))
        for pos_model in pos_sample:
            try:
                pos_out = self.get_model_outputs(pos_model)
                if pos_out is not None and pos_out.numel() > 0:
                    all_outputs.append(pos_out)
                    labels.append(1)
            except:
                continue

        neg_sample = random.sample(negative_models, min(8, len(negative_models)))
        for neg_model in neg_sample:
            try:
                neg_out = self.get_model_outputs(neg_model)
                if neg_out is not None and neg_out.numel() > 0:
                    all_outputs.append(neg_out)
                    labels.append(0)
            except:
                continue

        return all_outputs, labels

    def _update_graph_structure(self):
        """Update graph structure using edge ranking algorithm."""
        if not hasattr(self.fingerprint, 'x') or self.fingerprint.x.grad is None:
            return

        num_nodes = self.fingerprint.x.size(0)
        if num_nodes <= 1:
            return

        # Calculate node importance from gradients
        node_importance = torch.norm(self.fingerprint.x.grad, dim=1)

        # Create current adjacency matrix
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)
        if hasattr(self.fingerprint, 'edge_index') and self.fingerprint.edge_index.size(1) > 0:
            adj_matrix[self.fingerprint.edge_index[0], self.fingerprint.edge_index[1]] = 1

        # Calculate edge gradients approximation
        edge_gradients = torch.zeros_like(adj_matrix)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edge_gradients[i, j] = (node_importance[i] + node_importance[j]) / 2
                edge_gradients[j, i] = edge_gradients[i, j]

        # Rank edges by absolute gradient values
        edge_importance = torch.abs(edge_gradients)

        # Get top-K edges for modification
        K = max(1, int(0.1 * max(self.fingerprint.edge_index.size(1), num_nodes)))

        flat_importance = edge_importance.view(-1)
        top_k_values, top_k_indices = torch.topk(flat_importance, K)

        # Convert back to (i,j) coordinates
        top_k_edges = [(idx.item() // num_nodes, idx.item() % num_nodes)
                       for idx in top_k_indices]

        # Apply edge flipping rules
        for i, j in top_k_edges:
            if i != j:  # No self-loops
                edge_exists = adj_matrix[i, j].item() == 1
                gradient_positive = edge_gradients[i, j].item() >= 0

                if edge_exists and not gradient_positive:
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
                elif not edge_exists and gradient_positive:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        # Ensure connectivity
        self._ensure_graph_connectivity(adj_matrix, num_nodes)

        # Update edge index
        self.fingerprint.edge_index = adj_matrix.nonzero().t().contiguous()

    def _ensure_graph_connectivity(self, adj_matrix: torch.Tensor, num_nodes: int):
        """Ensure the graph remains connected."""
        current_edges = adj_matrix.sum().item()

        if current_edges < num_nodes - 1:
            for i in range(min(num_nodes - 1, 5)):
                j = (i + 1) % num_nodes
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1


class GraphFingerprint(FingerprintConstructor):
    """Fingerprint constructor for graph classification tasks."""
    
    def __init__(self, num_fingerprints: int = 64, min_nodes: int = 8, max_nodes: int = 25,
                 feature_dim: int = 1, edge_prob: float = 0.2, 
                 device: torch.device = torch.device('cpu')):
        super().__init__(device)
        self.num_fingerprints = num_fingerprints
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.edge_prob = edge_prob
        self.fingerprints = self._create_random_graphs()

    def _create_random_graphs(self) -> List[Data]:
        """Create multiple random graph fingerprints."""
        fingerprints = []
        for i in range(self.num_fingerprints):
            num_nodes = random.randint(self.min_nodes, self.max_nodes)

            if self.feature_dim > 0:
                x = torch.randn(num_nodes, self.feature_dim, 
                              requires_grad=True, device=self.device)
            else:
                x = torch.ones(num_nodes, 1, requires_grad=True, device=self.device)

            # Create adjacency matrix
            adj_prob = torch.rand(num_nodes, num_nodes)
            adj_matrix = (adj_prob < self.edge_prob).float()
            adj_matrix = torch.triu(adj_matrix, diagonal=1)
            adj_matrix = adj_matrix + adj_matrix.t()

            # Ensure connectivity
            for j in range(min(3, num_nodes-1)):
                adj_matrix[j, (j+1) % num_nodes] = 1
                adj_matrix[(j+1) % num_nodes, j] = 1

            edge_index = adj_matrix.nonzero().t().contiguous()
            fingerprints.append(Data(x=x, edge_index=edge_index))

        return fingerprints

    def get_model_outputs(self, model: nn.Module, require_grad: bool = False) -> torch.Tensor:
        """Get concatenated outputs from all fingerprint graphs."""
        model.eval()
        outputs = []

        if require_grad:
            for fp in self.fingerprints:
                batch = torch.zeros(fp.x.size(0), dtype=torch.long, device=self.device)
                fp_device = Data(x=fp.x.to(self.device), edge_index=fp.edge_index.to(self.device))
                out = model(fp_device.x, fp_device.edge_index, batch)
                outputs.append(out.squeeze())
        else:
            with torch.no_grad():
                for fp in self.fingerprints:
                    batch = torch.zeros(fp.x.size(0), dtype=torch.long, device=self.device)
                    fp_device = Data(x=fp.x.to(self.device), edge_index=fp.edge_index.to(self.device))
                    out = model(fp_device.x, fp_device.edge_index, batch)
                    outputs.append(out.squeeze())

        return torch.cat(outputs)

    def optimize_fingerprint(self, loss: torch.Tensor, alpha: float,
                           target_model: nn.Module, positive_models: List[nn.Module],
                           negative_models: List[nn.Module], univerifier: Optional[nn.Module] = None):
        """Optimize multiple graph fingerprints."""
        params = []
        for fp in self.fingerprints:
            if fp.x.requires_grad:
                params.append(fp.x)
        
        if params:
            optimizer = torch.optim.Adam(params, lr=alpha)
            optimizer.zero_grad()
            # Recompute loss with gradient
            if univerifier is not None:
                outputs = []
                labels = []
                try:
                    out = self.get_model_outputs(target_model, require_grad=True)
                    outputs.append(out); labels.append(1)
                except:
                    pass
                for pos_model in random.sample(positive_models, min(8, len(positive_models))):
                    try:
                        outputs.append(self.get_model_outputs(pos_model, require_grad=True)); labels.append(1)
                    except:
                        continue
                for neg_model in random.sample(negative_models, min(8, len(negative_models))):
                    try:
                        outputs.append(self.get_model_outputs(neg_model, require_grad=True)); labels.append(0)
                    except:
                        continue
                if len(outputs) >= 2:
                    min_size = min(t.size(0) for t in outputs)
                    batch_outputs = torch.stack([t[:min_size] for t in outputs])
                    batch_labels = torch.tensor(labels[:len(outputs)], dtype=torch.long, device=self.device)
                    preds = univerifier(batch_outputs)
                    current_loss = F.cross_entropy(preds, batch_labels)
                    current_loss.backward()
            # Edge update based on gradients
            for fp in self.fingerprints:
                self._apply_edge_ranking_algorithm(fp)
            optimizer.step()

    def _apply_edge_ranking_algorithm(self, graph_data: Data):
        """Apply edge ranking and flipping algorithm to a single graph."""
        if not hasattr(graph_data, 'x') or graph_data.x.grad is None:
            return

        num_nodes = graph_data.x.size(0)
        if num_nodes <= 1:
            return

        # Similar to NodeFingerprint update
        node_importance = torch.norm(graph_data.x.grad, dim=1)
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)
        if hasattr(graph_data, 'edge_index') and graph_data.edge_index.size(1) > 0:
            adj_matrix[graph_data.edge_index[0], graph_data.edge_index[1]] = 1
        edge_gradients = torch.zeros_like(adj_matrix)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edge_gradients[i, j] = (node_importance[i] + node_importance[j]) / 2
                edge_gradients[j, i] = edge_gradients[i, j]
        edge_importance = torch.abs(edge_gradients)
        K = max(1, int(0.1 * max(graph_data.edge_index.size(1), num_nodes)))
        flat_importance = edge_importance.view(-1)
        _, top_k_indices = torch.topk(flat_importance, K)
        top_k_edges = [(idx.item() // num_nodes, idx.item() % num_nodes) for idx in top_k_indices]
        for i, j in top_k_edges:
            if i != j:
                exists = adj_matrix[i, j].item() == 1
                grad_pos = edge_gradients[i, j].item() >= 0
                if exists and not grad_pos:
                    adj_matrix[i, j] = 0; adj_matrix[j, i] = 0
                elif not exists and grad_pos:
                    adj_matrix[i, j] = 1; adj_matrix[j, i] = 1
        # ensure connectivity
        if adj_matrix.sum().item() < num_nodes - 1:
            for i in range(min(num_nodes - 1, 3)):
                j = (i + 1) % num_nodes
                adj_matrix[i, j] = 1; adj_matrix[j, i] = 1
        graph_data.edge_index = adj_matrix.nonzero().t().contiguous()


class LinkPredictionFingerprint(FingerprintConstructor):
    """Fingerprint constructor for link prediction tasks."""
    
    def __init__(self, num_nodes: int = 32, feature_dim: int = 1433,
                 edge_prob: float = 0.2, num_edge_samples: int = 64,
                 device: torch.device = torch.device('cpu')):
        super().__init__(device)
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.edge_prob = edge_prob
        self.num_edge_samples = num_edge_samples

        self.fingerprint = self._create_random_graph()
        self.edge_pairs = self._create_edge_pairs()

    def _create_random_graph(self) -> Data:
        """Create random graph for link prediction."""
        x = torch.randn(self.num_nodes, self.feature_dim, 
                       requires_grad=True, device=self.device)

        adj_prob = torch.rand(self.num_nodes, self.num_nodes)
        adj_matrix = (adj_prob < self.edge_prob).float()
        adj_matrix = torch.triu(adj_matrix, diagonal=1)
        adj_matrix = adj_matrix + adj_matrix.t()

        # Ensure strong connectivity
        for i in range(min(8, self.num_nodes - 1)):
            j = (i + 1) % self.num_nodes
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        edge_index = adj_matrix.nonzero().t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def _create_edge_pairs(self) -> torch.Tensor:
        """Create edge pairs for link prediction."""
        pairs = []

        # Add existing edges (positive samples)
        if self.fingerprint.edge_index.size(1) > 0:
            existing_edges = self.fingerprint.edge_index.t()
            unique_edges = []
            seen = set()
            for edge in existing_edges:
                edge_tuple = tuple(sorted([edge[0].item(), edge[1].item()]))
                if edge_tuple not in seen:
                    seen.add(edge_tuple)
                    unique_edges.append([edge[0].item(), edge[1].item()])
            
            num_pos = min(self.num_edge_samples // 2, len(unique_edges))
            pos_pairs = random.sample(unique_edges, num_pos)
            pairs.extend(pos_pairs)

        # Add non-existing edges (negative samples)
        existing_set = set()
        if self.fingerprint.edge_index.size(1) > 0:
            edges = self.fingerprint.edge_index.t().cpu().numpy()
            existing_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)

        while len(pairs) < self.num_edge_samples:
            i, j = random.sample(range(self.num_nodes), 2)
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple not in existing_set and [i, j] not in pairs and [j, i] not in pairs:
                pairs.append([i, j])

        return torch.tensor(pairs[:self.num_edge_samples], dtype=torch.long, device=self.device).t()

    def get_model_outputs(self, model: nn.Module, require_grad: bool = False) -> torch.Tensor:
        """Get model outputs for link prediction fingerprints."""
        model.eval()
        model_device = next(model.parameters()).device
        fingerprint_x = self.fingerprint.x.to(model_device)
        fingerprint_edge_index = self.fingerprint.edge_index.to(model_device)
        edge_pairs = self.edge_pairs.to(model_device)
        if require_grad:
            embeddings = model.get_embeddings(fingerprint_x, fingerprint_edge_index)
            link_probs = model.predict_links(embeddings, edge_pairs)
        else:
            with torch.no_grad():
                embeddings = model.get_embeddings(fingerprint_x, fingerprint_edge_index)
                link_probs = model.predict_links(embeddings, edge_pairs)
        return link_probs.flatten()

    def optimize_fingerprint(self, loss: torch.Tensor, alpha: float,
                           target_model: nn.Module, positive_models: List[nn.Module],
                           negative_models: List[nn.Module], univerifier: Optional[nn.Module] = None):
        """Optimize link prediction fingerprint."""
        if self.fingerprint.x.requires_grad:
            params_to_optimize = [self.fingerprint.x]
            optimizer = torch.optim.Adam(params_to_optimize, lr=alpha)
            optimizer.zero_grad()
            # Recompute univerifier loss with gradient if provided
            if univerifier is not None:
                outputs = []
                labels = []
                try:
                    outputs.append(self.get_model_outputs(target_model, require_grad=True)); labels.append(1)
                except:
                    pass
                for pos_model in random.sample(positive_models, min(8, len(positive_models))):
                    try:
                        outputs.append(self.get_model_outputs(pos_model, require_grad=True)); labels.append(1)
                    except:
                        continue
                for neg_model in random.sample(negative_models, min(8, len(negative_models))):
                    try:
                        outputs.append(self.get_model_outputs(neg_model, require_grad=True)); labels.append(0)
                    except:
                        continue
                if len(outputs) >= 2:
                    min_size = min(t.size(0) for t in outputs)
                    batch_outputs = torch.stack([t[:min_size] for t in outputs])
                    batch_labels = torch.tensor(labels[:len(outputs)], dtype=torch.long, device=self.device)
                    preds = univerifier(batch_outputs)
                    current_loss = F.cross_entropy(preds, batch_labels)
                    current_loss.backward()
            optimizer.step()


class GraphMatchingFingerprint(FingerprintConstructor):
    """Fingerprint constructor for graph matching tasks."""
    
    def __init__(self, num_fingerprint_pairs: int = 64, min_nodes: int = 6, max_nodes: int = 20,
                 feature_dim: int = 1, edge_prob: float = 0.2, 
                 device: torch.device = torch.device('cpu')):
        super().__init__(device)
        self.num_fingerprint_pairs = num_fingerprint_pairs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.edge_prob = edge_prob
        self.fingerprint_pairs = self._create_random_graph_pairs()

    def _create_random_graph_pairs(self) -> List[Tuple[Data, Data]]:
        """Create pairs of random graphs for matching."""
        fingerprint_pairs = []

        for i in range(self.num_fingerprint_pairs):
            graph1 = self._create_single_graph()

            if random.random() < 0.5:  # 50% similar graphs
                graph2 = self._create_similar_graph(graph1)
            else:
                graph2 = self._create_single_graph()

            fingerprint_pairs.append((graph1, graph2))

        return fingerprint_pairs

    def _create_single_graph(self) -> Data:
        """Create a single random graph."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        if self.feature_dim > 0:
            x = torch.randint(0, 5, (num_nodes, self.feature_dim), 
                            dtype=torch.float, requires_grad=True, device=self.device)
        else:
            x = torch.ones(num_nodes, 1, requires_grad=True, device=self.device)

        # Create molecular-like structure
        edge_list = []
        for i in range(num_nodes - 1):
            edge_list.extend([[i, i+1], [i+1, i]])

        num_extra_edges = int(self.edge_prob * num_nodes * (num_nodes - 1) / 2)
        for _ in range(num_extra_edges):
            n1, n2 = random.sample(range(num_nodes), 2)
            edge_list.extend([[n1, n2], [n2, n1]])

        edge_set = set(tuple(edge) for edge in edge_list)
        edge_list = list(edge_set)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return Data(x=x, edge_index=edge_index)

    def _create_similar_graph(self, base_graph: Data) -> Data:
        """Create a graph similar to the base graph."""
        base_nodes = base_graph.x.size(0)
        num_nodes = base_nodes + random.randint(-2, 2)
        num_nodes = max(self.min_nodes, min(self.max_nodes, num_nodes))

        x = torch.randint(0, 5, (num_nodes, self.feature_dim), 
                        dtype=torch.float, requires_grad=True, device=self.device)

        # Copy some structural patterns
        edge_list = []
        min_nodes_to_copy = min(num_nodes, base_nodes)
        for i in range(min_nodes_to_copy - 1):
            edge_list.extend([[i, i+1], [i+1, i]])

        # Add some variations
        num_extra_edges = random.randint(0, num_nodes // 2)
        for _ in range(num_extra_edges):
            n1, n2 = random.sample(range(num_nodes), 2)
            edge_list.extend([[n1, n2], [n2, n1]])

        edge_set = set(tuple(edge) for edge in edge_list)
        edge_list = list(edge_set)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return Data(x=x, edge_index=edge_index)

    def get_model_outputs(self, model: nn.Module, require_grad: bool = False) -> torch.Tensor:
        """Get model outputs for graph matching fingerprints."""
        model.eval()
        outputs = []

        for graph1, graph2 in self.fingerprint_pairs:
            try:
                model_device = next(model.parameters()).device
                batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long, device=model_device)
                batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long, device=model_device)
                data1 = Data(x=graph1.x.to(model_device), edge_index=graph1.edge_index.to(model_device), batch=batch1)
                data2 = Data(x=graph2.x.to(model_device), edge_index=graph2.edge_index.to(model_device), batch=batch2)
                if require_grad:
                    similarity = model.forward(data1, data2)
                else:
                    with torch.no_grad():
                        similarity = model.forward(data1, data2)
                if isinstance(similarity, torch.Tensor):
                    if similarity.dim() == 0:
                        outputs.append(similarity.unsqueeze(0))
                    else:
                        outputs.append(similarity)
                else:
                    outputs.append(torch.tensor([similarity], device=model_device))
            except Exception as e:
                model_device = next(model.parameters()).device
                outputs.append(torch.tensor([0.5], device=model_device))

        if not outputs:
            model_device = next(model.parameters()).device
            return torch.tensor([0.5] * self.num_fingerprint_pairs, device=model_device)
            
        return torch.cat(outputs)

    def optimize_fingerprint(self, loss: torch.Tensor, alpha: float,
                           target_model: nn.Module, positive_models: List[nn.Module],
                           negative_models: List[nn.Module], univerifier: Optional[nn.Module] = None):
        """Optimize graph matching fingerprints."""
        params = []
        for graph1, graph2 in self.fingerprint_pairs:
            if graph1.x.requires_grad:
                params.append(graph1.x)
            if graph2.x.requires_grad:
                params.append(graph2.x)
        
        if params:
            optimizer = torch.optim.Adam(params, lr=alpha)
            optimizer.zero_grad()
            if univerifier is not None:
                outputs = []
                labels = []
                try:
                    outputs.append(self.get_model_outputs(target_model, require_grad=True)); labels.append(1)
                except:
                    pass
                for pos_model in random.sample(positive_models, min(8, len(positive_models))):
                    try:
                        outputs.append(self.get_model_outputs(pos_model, require_grad=True)); labels.append(1)
                    except:
                        continue
                for neg_model in random.sample(negative_models, min(8, len(negative_models))):
                    try:
                        outputs.append(self.get_model_outputs(neg_model, require_grad=True)); labels.append(0)
                    except:
                        continue
                if len(outputs) >= 2:
                    min_size = min(t.size(0) for t in outputs)
                    batch_outputs = torch.stack([t[:min_size] for t in outputs])
                    batch_labels = torch.tensor(labels[:len(outputs)], dtype=torch.long, device=self.device)
                    preds = univerifier(batch_outputs)
                    current_loss = F.cross_entropy(preds, batch_labels)
                    current_loss.backward()
            optimizer.step()


def create_fingerprint_constructor(task_type: str, dataset_info: Dict, 
                                 fingerprint_params: Dict, 
                                 device: torch.device) -> FingerprintConstructor:
    """
    Factory function to create appropriate fingerprint constructor.
    
    Args:
        task_type: Type of GNN task
        dataset_info: Dictionary containing dataset information
        fingerprint_params: Parameters for fingerprint construction
        device: Computing device
    
    Returns:
        Appropriate fingerprint constructor
    """
    if task_type == "node_classification":
        return NodeFingerprint(
            num_nodes=fingerprint_params.get('num_nodes', 32),
            feature_dim=dataset_info.get('num_features', 1433),
            edge_prob=fingerprint_params.get('edge_prob', 0.15),
            device=device
        )
    elif task_type == "graph_classification":
        return GraphFingerprint(
            num_fingerprints=fingerprint_params.get('num_fingerprints', 64),
            min_nodes=fingerprint_params.get('min_nodes', 8),
            max_nodes=fingerprint_params.get('max_nodes', 25),
            feature_dim=dataset_info.get('num_features', 1),
            edge_prob=fingerprint_params.get('edge_prob', 0.2),
            device=device
        )
    elif task_type == "link_prediction":
        return LinkPredictionFingerprint(
            num_nodes=fingerprint_params.get('num_nodes', 32),
            feature_dim=dataset_info.get('num_features', 1433),
            edge_prob=fingerprint_params.get('edge_prob', 0.2),
            num_edge_samples=fingerprint_params.get('num_edge_samples', 64),
            device=device
        )
    elif task_type == "graph_matching":
        return GraphMatchingFingerprint(
            num_fingerprint_pairs=fingerprint_params.get('num_fingerprint_pairs', 64),
            min_nodes=fingerprint_params.get('min_nodes', 6),
            max_nodes=fingerprint_params.get('max_nodes', 20),
            feature_dim=dataset_info.get('num_features', 1),
            edge_prob=fingerprint_params.get('edge_prob', 0.2),
            device=device
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


class FingerprintOptimizer:
    """Optimizer for fingerprint construction using Algorithm 1."""
    
    def __init__(self, fingerprint_constructor: FingerprintConstructor,
                 univerifier: nn.Module, device: torch.device):
        self.fingerprint_constructor = fingerprint_constructor
        self.univerifier = univerifier
        self.device = device
        self.flag = 0
        self.training_history = []
        self.converged = False

    def optimize(self, target_model: nn.Module, positive_models: List[nn.Module], 
                negative_models: List[nn.Module], epochs_total: int = 100,
                e1: int = 1, e2: int = 1, alpha: float = 0.01, beta: float = 0.001,
                convergence_threshold: float = 0.001) -> Dict:
        """
        Run Algorithm 1: Joint alternating optimization.
        
        Args:
            target_model: Target model to protect
            positive_models: List of pirated models
            negative_models: List of independent models
            epochs_total: Total training epochs
            e1: Fingerprint optimization epochs per iteration
            e2: Univerifier optimization epochs per iteration
            alpha: Fingerprint learning rate
            beta: Univerifier learning rate
            convergence_threshold: Convergence threshold
        
        Returns:
            Training history and results
        """
        print(f"Starting Algorithm 1 optimization...")
        print(f"Total epochs: {epochs_total}, e1={e1}, e2={e2}, alpha={alpha}, beta={beta}")

        univerifier_optimizer = torch.optim.Adam(self.univerifier.parameters(), lr=beta)
        epoch = 0

        while epoch < epochs_total and not self.converged:
            # Get fingerprint outputs from all models
            fingerprint_outputs = self._collect_fingerprint_outputs(
                target_model, positive_models, negative_models
            )

            if not fingerprint_outputs:
                print("Warning: No fingerprint outputs collected")
                break

            # Calculate unified loss L
            loss, predictions, labels = self._calculate_unified_loss(fingerprint_outputs)

            if self.flag == 0:
                # Update fingerprints for e1 epochs
                for _ in range(e1):
                    self.fingerprint_constructor.optimize_fingerprint(
                        loss, alpha, target_model, positive_models, negative_models
                    )
                self.flag = 1
                operation = "Fingerprints"
            else:
                # Update univerifier for e2 epochs
                for _ in range(e2):
                    univerifier_optimizer.zero_grad()

                    # Recalculate loss for current fingerprints
                    fingerprint_outputs = self._collect_fingerprint_outputs(
                        target_model, positive_models, negative_models
                    )
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
                if max(recent_losses) - min(recent_losses) < convergence_threshold:
                    self.converged = True
                    print(f"Converged at epoch {epoch}")

            epoch += 1

        return {
            'training_history': self.training_history,
            'converged': self.converged,
            'final_epoch': epoch
        }

    def _collect_fingerprint_outputs(self, target_model: nn.Module, 
                                   positive_models: List[nn.Module],
                                   negative_models: List[nn.Module]) -> Dict:
        """Collect outputs from all models using fingerprints."""
        try:
            # Target model output
            target_out = self.fingerprint_constructor.get_model_outputs(target_model)

            # Sample models to avoid memory issues
            positive_sample = random.sample(positive_models, min(50, len(positive_models)))
            negative_sample = random.sample(negative_models, min(50, len(negative_models)))

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
            return {}

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