import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_add_pool
from typing import List, Optional, Union
from torch_geometric.utils import negative_sampling
import random
import copy


class GCN(nn.Module):
    
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
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def predict_links(self, embeddings, edge_pairs):
        source_emb = embeddings[edge_pairs[0]]
        target_emb = embeddings[edge_pairs[1]]

        pair_emb = torch.cat([source_emb, target_emb], dim=1)
        link_logits = self.link_decoder(pair_emb)
        return torch.sigmoid(link_logits.squeeze())


class GCNDiff(nn.Module):
    
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

        # Graph classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def forward_matching(self, data1, data2):
        emb1 = self.get_graph_embedding(data1.x, data1.edge_index, data1.batch)
        emb2 = self.get_graph_embedding(data2.x, data2.edge_index, data2.batch)
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Compute similarity
        similarity = self.classifier(combined)
        return torch.sigmoid(similarity)

    def get_graph_embedding(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x


class GCNDiffGraphMatching(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GCNDiffGraphMatching, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Graph matching layers - specifically designed for similarity prediction
        self.matching_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # Single output for similarity score
        )

    def forward(self, data1, data2):
        emb1 = self.get_graph_embedding(data1.x, data1.edge_index, data1.batch)
        emb2 = self.get_graph_embedding(data2.x, data2.edge_index, data2.batch)
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Compute similarity score (0 to 1)
        similarity = self.matching_classifier(combined)
        return torch.sigmoid(similarity)
    
    def forward_matching(self, data1, data2):
        return self.forward(data1, data2)

    def get_graph_embedding(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x


class GraphSage(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super(GraphSage, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSageLinkPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 2, dropout: float = 0.5):
        super(GraphSageLinkPredictor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, edge_pairs=None):
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Get node embeddings
        node_embeddings = x
        
        if edge_pairs is not None:
            return self.predict_links(node_embeddings, edge_pairs)
        else:
            return node_embeddings

    def get_embeddings(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def predict_links(self, node_embeddings, edge_pairs):
        """Predict link probabilities for given edge pairs."""
        # Extract source and target node embeddings
        src_nodes = edge_pairs[0]
        dst_nodes = edge_pairs[1]
        
        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]
        
        # Concatenate source and target embeddings
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict link probability
        link_prob = self.link_predictor(edge_features)
        return torch.sigmoid(link_prob)


class GraphSageMean(nn.Module):
    """GraphSage with Mean Pooling for Graph Classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GraphSageMean, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

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


class GraphSageDiff(nn.Module):
    """GraphSage with Difference-based similarity for Graph Classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GraphSageDiff, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Graph classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, batch):
        """Forward pass for graph classification."""
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def forward_matching(self, data1, data2):
        """Forward pass for graph matching (legacy method)."""
        emb1 = self.get_graph_embedding(data1.x, data1.edge_index, data1.batch)
        emb2 = self.get_graph_embedding(data2.x, data2.edge_index, data2.batch)
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Compute similarity
        similarity = self.classifier(combined)
        return torch.sigmoid(similarity)

    def get_graph_embedding(self, x, edge_index, batch):
        """Get graph-level embedding."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x


class SimGNN(nn.Module):
    """SimGNN for Graph Classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.5):
        super(SimGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Attention mechanism for graph classification
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Graph classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, batch):
        """Forward pass for graph classification."""
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def forward_matching(self, data1, data2):
        """Forward pass for graph matching (legacy method)."""
        # Get embeddings for both graphs
        emb1 = self.get_graph_embedding(data1.x, data1.edge_index, data1.batch)
        emb2 = self.get_graph_embedding(data2.x, data2.edge_index, data2.batch)
        
        # Apply attention mechanism
        emb1 = emb1.unsqueeze(0)  # Add batch dimension for attention
        emb2 = emb2.unsqueeze(0)
        
        attn_out, _ = self.attention(emb1, emb2, emb2)
        attn_out = attn_out.squeeze(0)
        
        # Concatenate embeddings
        combined = torch.cat([emb1.squeeze(0), attn_out], dim=1)
        
        # Compute similarity
        similarity = self.classifier(combined)
        return torch.sigmoid(similarity)

    def get_graph_embedding(self, x, edge_index, batch):
        """Get graph-level embedding."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x


class SimGNNGraphMatching(nn.Module):
    """SimGNN for Graph Matching."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.5):
        super(SimGNNGraphMatching, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Attention mechanism for graph matching
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Graph matching layers - specifically designed for similarity prediction
        self.matching_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single output for similarity score
        )

    def forward(self, data1, data2):
        """Forward pass for graph matching."""
        # Get embeddings for both graphs
        emb1 = self.get_graph_embedding(data1.x, data1.edge_index, data1.batch)
        emb2 = self.get_graph_embedding(data2.x, data2.edge_index, data2.batch)
        
        # Apply attention mechanism
        emb1 = emb1.unsqueeze(0)  # Add batch dimension for attention
        emb2 = emb2.unsqueeze(0)
        
        attn_out, _ = self.attention(emb1, emb2, emb2)
        attn_out = attn_out.squeeze(0)
        
        # Concatenate embeddings
        combined = torch.cat([emb1.squeeze(0), attn_out], dim=1)
        
        # Compute similarity
        similarity = self.matching_classifier(combined)
        return torch.sigmoid(similarity)
    
    def forward_matching(self, data1, data2):
        """Alternative forward pass for graph matching (for compatibility)."""
        return self.forward(data1, data2)

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
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64, 32], 
                 dropout: float = 0.3, activation: str = 'leaky_relu'):
        super(Univerifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)  # Changed from BatchNorm1d to LayerNorm for stability
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
                       output_dim: int, num_layers: int, device: Optional[torch.device] = None) -> nn.Module:
    """Get appropriate model for the task."""
    # Automatic device selection: GPU if available, else CPU
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    if task_type == "node_classification":
        return GCN(input_dim, hidden_dim, output_dim, num_layers).to(device)
    elif task_type == "graph_classification":
        return GCNMean(input_dim, hidden_dim, output_dim, num_layers).to(device)
    elif task_type == "link_prediction":
        return GCNLinkPredictor(input_dim, hidden_dim, num_layers).to(device)
    elif task_type == "graph_matching":
        return GCNDiffGraphMatching(input_dim, hidden_dim, output_dim, num_layers).to(device)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def get_model_with_architecture(task_type: str, architecture: str, input_dim: int, 
                               hidden_dim: int, output_dim: int, num_layers: int = 2) -> nn.Module:
    """
    Factory function to get model with specific architecture for task type.
    
    Args:
        task_type: Type of GNN task
        architecture: Model architecture ('GCN', 'GraphSage', 'SimGNN')
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of layers
    
    Returns:
        Appropriate GNN model with specified architecture
    """
    if task_type == "node_classification":
        if architecture.upper() == "GCN":
            return GCN(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GRAPHSAGE":
            return GraphSage(input_dim, hidden_dim, output_dim, num_layers)
        else:
            return GCN(input_dim, hidden_dim, output_dim, num_layers)
    
    elif task_type == "graph_classification":
        if architecture.upper() == "GCNMEAN":
            return GCNMean(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GCNDIFF":
            return GCNDiff(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GRAPHSAGEMEAN":
            return GraphSageMean(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GRAPHSAGEDIFF":
            return GraphSageDiff(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "SIMGNN":
            return SimGNN(input_dim, hidden_dim, output_dim, num_layers)
        else:
            return GCNMean(input_dim, hidden_dim, output_dim, num_layers)
    
    elif task_type == "link_prediction":
        if architecture.upper() == "GCN":
            return GCNLinkPredictor(input_dim, hidden_dim, num_layers)
        elif architecture.upper() == "GRAPHSAGE":
            return GraphSageLinkPredictor(input_dim, hidden_dim, output_dim, num_layers)
        else:
            return GCNLinkPredictor(input_dim, hidden_dim, num_layers)
    
    elif task_type == "graph_matching":
        # For graph matching, always use a proper graph matching model
        # Map the architecture to the appropriate graph matching model
        if architecture.upper() == "GCNMEAN":
            return GCNDiffGraphMatching(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GCNDIFF":
            return GCNDiffGraphMatching(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GRAPHSAGEMEAN":
            return GraphSageDiff(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "GRAPHSAGEDIFF":
            return GraphSageDiff(input_dim, hidden_dim, output_dim, num_layers)
        elif architecture.upper() == "SIMGNN":
            return SimGNNGraphMatching(input_dim, hidden_dim, output_dim, num_layers)
        else:
            return GCNDiffGraphMatching(input_dim, hidden_dim, output_dim, num_layers)
    
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
        elif task_type == "graph_classification":
            # Expect data to be a dataset-like object providing dataloaders
            try:
                train_loader = data.get_dataloader(split="train", batch_size=32, shuffle=True)
            except Exception:
                return fine_tuned_model
            for epoch in range(epochs):
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = fine_tuned_model(batch.x, batch.edge_index, batch.batch)
                    # Ensure output and target have matching batch sizes
                    if out.size(0) != batch.y.size(0):
                        # If batch sizes don't match, use the smaller one
                        min_size = min(out.size(0), batch.y.size(0))
                        out = out[:min_size]
                        batch_y = batch.y[:min_size]
                    else:
                        batch_y = batch.y
                    # Ensure labels are in valid range [0, num_classes-1]
                    batch_y = batch_y.view(-1).long()
                    if batch_y.numel() > 0:
                        # Shift labels to start from 0 if they don't already
                        batch_y = batch_y - batch_y.min()
                        # Clamp to valid range
                        batch_y = torch.clamp(batch_y, 0, out.size(1) - 1)
                    loss = F.nll_loss(out, batch_y)
                    loss.backward()
                    optimizer.step()
        elif task_type == "link_prediction":
            # Expect data to be PyG Data with edge splits
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                return fine_tuned_model
            for epoch in range(epochs):
                # Sample negatives each epoch
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=data.train_pos_edge_index.to(device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(1000, data.train_pos_edge_index.size(1)),
                        method='sparse'
                    )
                except Exception:
                    from torch_geometric.utils import negative_sampling as neg_samp
                    neg_edge_index = neg_samp(
                        edge_index=data.train_pos_edge_index.to(device),
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
                    pos_batch = pos_edges[start:end].t().to(device)
                    neg_batch = neg_edges[start:end].t().to(device)
                    pos_pred = fine_tuned_model(data.x.to(device), data.train_pos_edge_index.to(device), pos_batch)
                    neg_pred = fine_tuned_model(data.x.to(device), data.train_pos_edge_index.to(device), neg_batch)
                    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
                    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                    loss = pos_loss + neg_loss
                    loss.backward()
                    optimizer.step()
        elif task_type == "graph_matching":
            # Expect data to be a list of pairs: [((g1,g2), sim), ...]
            train_pairs = data
            for epoch in range(epochs):
                random.shuffle(train_pairs)
                for (graph1, graph2), sim in train_pairs[:50]:
                    optimizer.zero_grad()
                    batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long, device=device)
                    batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long, device=device)
                    d1 = type(graph1)(x=graph1.x.to(device), edge_index=graph1.edge_index.to(device), batch=batch1)
                    d2 = type(graph2)(x=graph2.x.to(device), edge_index=graph2.edge_index.to(device), batch=batch2)
                    
                    # Use forward_matching if available, otherwise use regular forward
                    if hasattr(fine_tuned_model, 'forward_matching'):
                        pred = fine_tuned_model.forward_matching(d1, d2)
                    else:
                        # For models without forward_matching, use regular forward with batch
                        # Check if the model has a forward method that takes data1, data2
                        if hasattr(fine_tuned_model, 'forward') and fine_tuned_model.forward.__code__.co_argcount == 3:
                            # Model expects (self, data1, data2) - use it directly
                            pred = fine_tuned_model(d1, d2)
                        else:
                            # Fallback to individual forward calls
                            pred1 = fine_tuned_model(d1.x, d1.edge_index, d1.batch)
                            pred2 = fine_tuned_model(d2.x, d2.edge_index, d2.batch)
                            # Combine predictions (simple approach)
                            pred = (pred1 + pred2) / 2
                    
                    # For graph matching, use binary cross-entropy loss for similarity prediction
                    # Ensure target is in [0,1] range and prediction is properly shaped
                    target = torch.tensor([sim], dtype=torch.float, device=device).clamp(0, 1)
                    pred = pred.squeeze().clamp(1e-7, 1-1e-7)  # Avoid log(0) or log(1)
                    
                    # Dynamic tensor shape handling for graph matching
                    if pred.dim() == 0:  # scalar prediction
                        pred = pred.unsqueeze(0)  # Make it [1] to match target [1]
                    
                    # Ensure pred and target have the same shape
                    if pred.shape != target.shape:
                        if pred.numel() == 1 and target.numel() == 1:
                            pred = pred.view_as(target)
                    
                    loss = F.binary_cross_entropy(pred, target)
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

        # Also unfreeze classifier for graph classification
        if task_type == "graph_classification" and hasattr(retrained_model, 'classifier'):
            for p in retrained_model.classifier.parameters():
                p.requires_grad = True

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
        elif task_type == "graph_classification":
            try:
                train_loader = data.get_dataloader(split="train", batch_size=32, shuffle=True)
            except Exception:
                return retrained_model
            for epoch in range(epochs):
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = retrained_model(batch.x, batch.edge_index, batch.batch)
                    
                    # Ensure labels are in valid range [0, num_classes-1]
                    batch_y = batch.y.view(-1).long()
                    if batch_y.numel() > 0:
                        # Shift labels to start from 0 if they don't already
                        batch_y = batch_y - batch_y.min()
                        # Clamp to valid range
                        batch_y = torch.clamp(batch_y, 0, out.size(1) - 1)
                    loss = F.nll_loss(out, batch_y)
                    loss.backward()
                    optimizer.step()
        elif task_type == "link_prediction":
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                return retrained_model
            for epoch in range(epochs):
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=data.train_pos_edge_index.to(device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(1000, data.train_pos_edge_index.size(1)),
                        method='sparse'
                    )
                except Exception:
                    from torch_geometric.utils import negative_sampling as neg_samp
                    neg_edge_index = neg_samp(
                        edge_index=data.train_pos_edge_index.to(device),
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
                    pos_batch = pos_edges[start:end].t().to(device)
                    neg_batch = neg_edges[start:end].t().to(device)
                    pos_pred = retrained_model(data.x.to(device), data.train_pos_edge_index.to(device), pos_batch)
                    neg_pred = retrained_model(data.x.to(device), data.train_pos_edge_index.to(device), neg_batch)
                    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
                    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                    loss = pos_loss + neg_loss
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
        elif task_type == "graph_classification":
            try:
                train_loader = data.get_dataloader(split="train", batch_size=32, shuffle=True)
            except Exception:
                return student_model
            for epoch in range(epochs):
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = teacher_model(batch.x, batch.edge_index, batch.batch)
                    student_outputs = student_model(batch.x, batch.edge_index, batch.batch)
                    teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
                    student_soft = F.log_softmax(student_outputs / temperature, dim=1)
                    distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
                    
                    # Ensure labels are in valid range [0, num_classes-1]
                    batch_y = batch.y.view(-1).long()
                    if batch_y.numel() > 0:
                        # Shift labels to start from 0 if they don't already
                        batch_y = batch_y - batch_y.min()
                        # Clamp to valid range
                        batch_y = torch.clamp(batch_y, 0, student_outputs.size(1) - 1)
                    hard_loss = F.nll_loss(student_outputs, batch_y)
                    total_loss = 0.7 * distill_loss + 0.3 * hard_loss
                    total_loss.backward()
                    optimizer.step()
        elif task_type == "link_prediction":
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                return student_model
            for epoch in range(epochs):
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=data.train_pos_edge_index.to(device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(800, data.train_pos_edge_index.size(1)),
                        method='sparse'
                    )
                except Exception:
                    from torch_geometric.utils import negative_sampling as neg_samp
                    neg_edge_index = neg_samp(
                        edge_index=data.train_pos_edge_index.to(device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(800, data.train_pos_edge_index.size(1))
                    )
                batch_size = 256
                pos_edges = data.train_pos_edge_index.t()
                neg_edges = neg_edge_index.t()
                num_batches = min(pos_edges.size(0), neg_edges.size(0)) // batch_size
                for i in range(min(num_batches, 5)):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    optimizer.zero_grad()
                    pos_batch = pos_edges[start:end].t().to(device)
                    neg_batch = neg_edges[start:end].t().to(device)
                    with torch.no_grad():
                        teacher_pos = teacher_model(data.x.to(device), data.train_pos_edge_index.to(device), pos_batch)
                        teacher_neg = teacher_model(data.x.to(device), data.train_pos_edge_index.to(device), neg_batch)
                    student_pos = student_model(data.x.to(device), data.train_pos_edge_index.to(device), pos_batch)
                    student_neg = student_model(data.x.to(device), data.train_pos_edge_index.to(device), neg_batch)
                    distill_loss = (F.mse_loss(student_pos, teacher_pos.detach()) + F.mse_loss(student_neg, teacher_neg.detach())) / 2
                    hard_loss = (F.binary_cross_entropy(student_pos, torch.ones_like(student_pos)) + F.binary_cross_entropy(student_neg, torch.zeros_like(student_neg))) / 2
                    total_loss = 0.7 * distill_loss + 0.3 * hard_loss
                    total_loss.backward()
                    optimizer.step()

        return student_model
    
    @staticmethod
    def prune_model(model: nn.Module, data, task_type: str, 
                    pruning_ratio: float = 0.3, epochs: int = 50,
                    device: torch.device = torch.device('cpu')):
        """Create pruned version of model by removing less important connections."""
        # Create a copy of the model for pruning
        pruned_model = copy.deepcopy(model).to(device)
        
        # Apply pruning to convolutional layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, (GCNConv, SAGEConv)):
                # For GCNConv and SAGEConv, we need to access the underlying linear layer
                if hasattr(module, 'lin'):
                    # Access the linear layer's weight
                    weight = module.lin.weight.data
                    num_params = weight.numel()
                    num_to_prune = int(num_params * pruning_ratio)
                    
                    # Find the smallest absolute values to prune
                    flat_weights = weight.abs().flatten()
                    threshold = torch.kthvalue(flat_weights, num_to_prune)[0]
                    
                    # Create mask for pruning
                    mask = (weight.abs() > threshold).float()
                    module.lin.weight.data = module.lin.weight.data * mask
                elif hasattr(module, 'weight'):
                    # Direct weight access if available
                    weight = module.weight.data
                    num_params = weight.numel()
                    num_to_prune = int(num_params * pruning_ratio)
                    
                    # Find the smallest absolute values to prune
                    flat_weights = weight.abs().flatten()
                    threshold = torch.kthvalue(flat_weights, num_to_prune)[0]
                    
                    # Create mask for pruning
                    mask = (weight.abs() > threshold).float()
                    module.weight.data = module.weight.data * mask
        
        # Fine-tune the pruned model
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.001)
        
        if task_type == "node_classification":
            for epoch in range(epochs):
                pruned_model.train()
                optimizer.zero_grad()
                out = pruned_model(data.x.to(device), data.edge_index.to(device))
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
                loss.backward()
                optimizer.step()
        elif task_type == "graph_classification":
            try:
                train_loader = data.get_dataloader(split="train", batch_size=32, shuffle=True)
            except Exception:
                return pruned_model
            for epoch in range(epochs):
                for batch in train_loader:
                    batch = batch.to(device)
                    pruned_model.train()
                    optimizer.zero_grad()
                    out = pruned_model(batch.x, batch.edge_index, batch.batch)
                    # Ensure labels are in valid range [0, num_classes-1]
                    y = batch.y.view(-1).long()
                    
                    # Ensure output and target have matching batch sizes
                    if out.size(0) != y.size(0):
                        min_size = min(out.size(0), y.size(0))
                        out = out[:min_size]
                        y = y[:min_size]
                    
                    if y.numel() > 0:
                        # Shift labels to start from 0 if they don't already
                        y = y - y.min()
                        # Clamp to valid range
                        y = torch.clamp(y, 0, out.size(1) - 1)
                    loss = F.nll_loss(out, y)
                    loss.backward()
                    optimizer.step()
        elif task_type == "link_prediction":
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                return pruned_model
            for epoch in range(epochs):
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=data.train_pos_edge_index.to(device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(800, data.train_pos_edge_index.size(1)),
                        method='sparse'
                    )
                except Exception:
                    from torch_geometric.utils import negative_sampling as neg_samp
                    neg_edge_index = neg_samp(
                        edge_index=data.train_pos_edge_index.to(device),
                        num_nodes=data.x.size(0),
                        num_neg_samples=min(800, data.train_pos_edge_index.size(1))
                    )
                batch_size = 256
                pos_edges = data.train_pos_edge_index.t()
                neg_edges = neg_edge_index.t()
                num_batches = min(pos_edges.size(0), neg_edges.size(0)) // batch_size
                for i in range(min(num_batches, 5)):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    optimizer.zero_grad()
                    pos_batch = pos_edges[start:end].t().to(device)
                    neg_batch = neg_edges[start:end].t().to(device)
                    pos_pred = pruned_model(data.x.to(device), data.train_pos_edge_index.to(device), pos_batch)
                    neg_pred = pruned_model(data.x.to(device), data.train_pos_edge_index.to(device), neg_batch)
                    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
                    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                    loss = pos_loss + neg_loss
                    loss.backward()
                    optimizer.step()
        elif task_type == "graph_matching":
            try:
                pairs = data.create_graph_pairs(num_pairs=300)
            except Exception:
                return pruned_model
            for epoch in range(epochs):
                random.shuffle(pairs)
                for (graph1, graph2), sim in pairs[:50]:
                    try:
                        optimizer.zero_grad()
                        batch1 = torch.zeros(graph1.x.size(0), dtype=torch.long, device=device)
                        batch2 = torch.zeros(graph2.x.size(0), dtype=torch.long, device=device)
                        d1 = type(graph1)(x=graph1.x.to(device), edge_index=graph1.edge_index.to(device), batch=batch1)
                        d2 = type(graph2)(x=graph2.x.to(device), edge_index=graph2.edge_index.to(device), batch=batch2)
                        
                        # Use forward_matching if available, otherwise use regular forward
                        if hasattr(pruned_model, 'forward_matching'):
                            pred = pruned_model.forward_matching(d1, d2)
                        else:
                            # For models without forward_matching, use regular forward with batch
                            # Check if the model has a forward method that takes data1, data2
                            if hasattr(pruned_model, 'forward') and pruned_model.forward.__code__.co_argcount == 3:
                                # Model expects (self, data1, data2) - use it directly
                                pred = pruned_model(d1, d2)
                            else:
                                # Fallback to individual forward calls
                                pred1 = pruned_model(d1.x, d1.edge_index, d1.batch)
                                pred2 = pruned_model(d2.x, d2.edge_index, d2.batch)
                                # Combine predictions (simple approach)
                                pred = (pred1 + pred2) / 2
                        
                        # For graph matching, use binary cross-entropy loss for similarity prediction
                        target = torch.tensor([sim], dtype=torch.float, device=device).clamp(0, 1)
                        pred = pred.squeeze().clamp(1e-7, 1-1e-7)  # Avoid log(0) or log(1)
                        loss = F.binary_cross_entropy(pred, target)
                        loss.backward()
                        optimizer.step()
                    except Exception:
                        continue
                if epoch > 30 and random.random() < 0.03:
                    break
        
        return pruned_model
    
    @staticmethod
    def create_comprehensive_obfuscated_models(model: nn.Module, data, task_type: str,
                                             input_dim: int, hidden_dim: int, output_dim: int,
                                             num_models_per_method: int = 1,
                                             device: torch.device = torch.device('cpu')) -> dict:
        """
        Create obfuscated models using all 4 attacking methods for comprehensive testing.
        
        Args:
            model: Original target model to obfuscate
            data: Dataset for training
            task_type: Type of GNN task
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_models_per_method: Number of models to create per method
            device: Computing device
        
        Returns:
            Dictionary containing obfuscated models for each method
        """
        print(f"Creating comprehensive obfuscated models for {task_type} task...")
        print(f"Using all 4 attacking methods: fine_tuning, partial_retraining, distillation, pruning")
        
        obfuscated_models = {
            'fine_tuning': [],
            'partial_retraining': [],
            'distillation': [],
            'pruning': []
        }
        
        # Create fine-tuned models
        print("Creating fine-tuned models...")
        for i in range(num_models_per_method):
            fine_tuned = ModelObfuscator.fine_tune_model(
                model, data, task_type, epochs=20, lr=0.01, device=device
            )
            obfuscated_models['fine_tuning'].append(fine_tuned)
        
        # Create partially retrained models
        print("Creating partially retrained models...")
        for i in range(num_models_per_method):
            retrained = ModelObfuscator.partial_retrain_model(
                model, data, task_type, layers_to_retrain=1, epochs=20, lr=0.01, device=device
            )
            obfuscated_models['partial_retraining'].append(retrained)
        
        # Create distilled models
        print("Creating distilled models...")
        for i in range(num_models_per_method):
            distilled = ModelObfuscator.distill_model(
                model, data, task_type, input_dim, hidden_dim, output_dim,
                epochs=200, lr=0.01, temperature=4.0, device=device
            )
            obfuscated_models['distillation'].append(distilled)
        
        # Create pruned models
        print("Creating pruned models...")
        for i in range(num_models_per_method):
            pruned = ModelObfuscator.prune_model(
                model, data, task_type, pruning_ratio=0.3, epochs=50, device=device
            )
            obfuscated_models['pruning'].append(pruned)
        
        print(f"Successfully created {num_models_per_method} obfuscated models for each of the 4 attacking methods")
        return obfuscated_models
    
    @staticmethod
    def create_quick_obfuscated_models(model: nn.Module, data, task_type: str,
                                       input_dim: int, hidden_dim: int, output_dim: int,
                                       device: torch.device = torch.device('cpu')) -> dict:
        """
        Create obfuscated models using all 4 attacking methods for quick testing.
        Uses reduced epochs and simpler configurations for faster execution.
        
        Args:
            model: Original target model to obfuscate
            data: Dataset for training
            task_type: Type of GNN task
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            device: Computing device
        
        Returns:
            Dictionary containing obfuscated models for each method
        """
        print(f"Creating quick obfuscated models for {task_type} task...")
        print(f"Using all 4 attacking methods with reduced epochs for faster execution")
        
        obfuscated_models = {
            'fine_tuning': [],
            'partial_retraining': [],
            'distillation': [],
            'pruning': []
        }
        
        # Create fine-tuned models (quick version)
        print("Creating quick fine-tuned models...")
        fine_tuned = ModelObfuscator.fine_tune_model(
            model, data, task_type, epochs=5, lr=0.01, device=device
        )
        obfuscated_models['fine_tuning'].append(fine_tuned)
        
        # Create partially retrained models (quick version)
        print("Creating quick partially retrained models...")
        retrained = ModelObfuscator.partial_retrain_model(
            model, data, task_type, layers_to_retrain=1, epochs=5, lr=0.01, device=device
        )
        obfuscated_models['partial_retraining'].append(retrained)
        
        # Create distilled models (quick version)
        print("Creating quick distilled models...")
        distilled = ModelObfuscator.distill_model(
            model, data, task_type, input_dim, hidden_dim, output_dim,
            epochs=50, lr=0.01, temperature=4.0, device=device
        )
        obfuscated_models['distillation'].append(distilled)
        
        # Create pruned models (quick version)
        print("Creating quick pruned models...")
        pruned = ModelObfuscator.prune_model(
            model, data, task_type, pruning_ratio=0.2, epochs=10, device=device
        )
        obfuscated_models['pruning'].append(pruned)
        
        print("Successfully created quick obfuscated models for all 4 attacking methods")
        return obfuscated_models