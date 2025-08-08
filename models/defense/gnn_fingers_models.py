"""
GNN model implementations for GNNFingers framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from typing import List, Optional, Union
import copy


class GCN(nn.Module):
    """Graph Convolutional Network for Node Classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNMean(nn.Module):
    """Graph Convolutional Network with Mean Pooling for Graph Classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GCNMean, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class GCNLinkPredictor(nn.Module):
    """Graph Convolutional Network for Link Prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super(GCNLinkPredictor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Link prediction decoder
        self.link_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_pairs=None):
        embeddings = self.get_embeddings(x, edge_index)

        if edge_pairs is not None:
            return self.predict_links(embeddings, edge_pairs)
        else:
            return embeddings

    def get_embeddings(self, x, edge_index):
        """Get node embeddings through GCN layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def predict_links(self, embeddings, edge_pairs):
        """Predict link probabilities for given node pairs."""
        source_emb = embeddings[edge_pairs[0]]
        target_emb = embeddings[edge_pairs[1]]

        pair_emb = torch.cat([source_emb, target_emb], dim=1)
        link_logits = self.link_decoder(pair_emb)
        return torch.sigmoid(link_logits.squeeze())


class GCNDiff(nn.Module):
    """Graph Convolutional Network with Difference Pooling for Graph Matching."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GCNDiff, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Graph matching layers
        self.matching_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, data1, data2):
        """Forward pass for graph matching."""
        emb1 = self.get_graph_embedding(data1.x, data1.edge_index, data1.batch)
        emb2 = self.get_graph_embedding(data2.x, data2.edge_index, data2.batch)

        # Compute difference-based features
        diff_features = torch.abs(emb1 - emb2)
        similarity = self.matching_layers(diff_features)
        return similarity.squeeze()

    def get_graph_embedding(self, x, edge_index, batch):
        """Get graph-level embedding."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x


class Univerifier(nn.Module):
    """Universal Verification mechanism - Binary classifier for ownership verification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 dropout: float = 0.3, activation: str = 'leaky_relu'):
        super(Univerifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Final binary classification layer
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning probability simplex."""
        logits = self.network(x)
        return F.softmax(logits, dim=1)  # Returns {(o+, o-) | o- + o+ = 1}


def get_model_for_task(task_type: str, input_dim: int, hidden_dim: int, 
                      output_dim: int, num_layers: int = 2) -> nn.Module:
    """
    Factory function to get appropriate model for task type.
    
    Args:
        task_type: Type of GNN task
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of layers
    
    Returns:
        Appropriate GNN model for the task
    """
    if task_type == "node_classification":
        return GCN(input_dim, hidden_dim, output_dim, num_layers)
    elif task_type == "graph_classification":
        return GCNMean(input_dim, hidden_dim, output_dim, num_layers)
    elif task_type == "link_prediction":
        return GCNLinkPredictor(input_dim, hidden_dim, num_layers)
    elif task_type == "graph_matching":
        return GCNDiff(input_dim, hidden_dim, 1, num_layers)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


class ModelObfuscator:
    """Utility class for creating obfuscated versions of target models."""
    
    @staticmethod
    def fine_tune_model(model: nn.Module, data, task_type: str, epochs: int = 20, 
                       lr: float = 0.01, device: torch.device = torch.device('cpu')):
        """Create fine-tuned version of model."""
        fine_tuned_model = copy.deepcopy(model).to(device)
        optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=lr)

        fine_tuned_model.train()
        
        if task_type == "node_classification":
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = fine_tuned_model(data.x.to(device), data.edge_index.to(device))
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
                loss.backward()
                optimizer.step()
        
        # Add other task types as needed
        
        return fine_tuned_model
    
    @staticmethod
    def partial_retrain_model(model: nn.Module, data, task_type: str, 
                             layers_to_retrain: int = 1, epochs: int = 20, 
                             lr: float = 0.01, device: torch.device = torch.device('cpu')):
        """Create partially retrained version of model."""
        retrained_model = copy.deepcopy(model).to(device)
        
        # Reinitialize last K layers
        if hasattr(retrained_model, 'convs'):
            for i in range(min(layers_to_retrain, len(retrained_model.convs))):
                layer_idx = -(i + 1)
                retrained_model.convs[layer_idx].reset_parameters()

        # Freeze other layers
        for param in retrained_model.parameters():
            param.requires_grad = False

        # Unfreeze layers to retrain
        if hasattr(retrained_model, 'convs'):
            for i in range(min(layers_to_retrain, len(retrained_model.convs))):
                layer_idx = -(i + 1)
                for param in retrained_model.convs[layer_idx].parameters():
                    param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, retrained_model.parameters()), 
            lr=lr
        )

        retrained_model.train()
        
        if task_type == "node_classification":
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = retrained_model(data.x.to(device), data.edge_index.to(device))
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
                loss.backward()
                optimizer.step()

        return retrained_model
    
    @staticmethod
    def distill_model(teacher_model: nn.Module, data, task_type: str, 
                     input_dim: int, hidden_dim: int, output_dim: int,
                     epochs: int = 200, lr: float = 0.01, temperature: float = 4.0,
                     device: torch.device = torch.device('cpu')):
        """Create knowledge-distilled version of model."""
        student_model = get_model_for_task(
            task_type=task_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3
        ).to(device)
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

        teacher_model.eval()
        student_model.train()

        if task_type == "node_classification":
            for epoch in range(epochs):
                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = teacher_model(data.x.to(device), data.edge_index.to(device))

                student_outputs = student_model(data.x.to(device), data.edge_index.to(device))

                teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
                student_soft = F.log_softmax(student_outputs / temperature, dim=1)
                distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

                hard_loss = F.nll_loss(student_outputs[data.train_mask], 
                                     data.y[data.train_mask].to(device))
                total_loss = 0.7 * distill_loss + 0.3 * hard_loss

                total_loss.backward()
                optimizer.step()

        return student_model