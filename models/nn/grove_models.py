"""
Grove GNN Models (embedding based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GINConv
import os


class BaseGNNModel(nn.Module):
    """
    Base class for GNN models.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, dropout=0.5, name=None):
        """
        Initialize the base GNN model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            dropout: Dropout probability
            name: Name of the model
        """
        super(BaseGNNModel, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.name = name or self.__class__.__name__
        
        # Initialize weights after model creation
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize the weights of the model using Kaiming initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        raise NotImplementedError("Forward method must be implemented by subclasses")
    
    def save(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'in_feats': self.in_feats,
            'h_feats': self.h_feats,
            'out_feats': self.out_feats,
            'dropout': self.dropout,
            'name': self.name
        }, path)
        
        
    @classmethod
    def load(cls, path, device=None):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            in_feats=checkpoint['in_feats'],
            h_feats=checkpoint['h_feats'],
            out_feats=checkpoint['out_feats'],
            dropout=checkpoint['dropout'],
            name=checkpoint.get('name')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


class GATModel(BaseGNNModel):
    """
    Graph Attention Network model for grove.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, num_heads=4, num_layers=3, dropout=0.5, name=None):
        """
        Initialize the GAT model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout probability
            name: Name of the model
        """
        super(GATModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Multi-head attention layers
        self.gat1 = GATConv(
            in_feats, h_feats, heads=num_heads, dropout=dropout
        )
        
        self.gat2 = GATConv(
            h_feats * num_heads, h_feats, heads=num_heads, dropout=dropout
        )
        
        # Final layer for output (embeddings layer)
        self.gat3 = GATConv(
            h_feats * num_heads, h_feats, heads=1, concat=False, dropout=dropout
        )
        
        # Dense layer for classification
        self.classifier = nn.Linear(h_feats, out_feats)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Sample neighbors for each layer, fixed at 10
        edge_index1 = self._sample_neighbors(edge_index, 10)
        edge_index2 = self._sample_neighbors(edge_index, 10)
        edge_index3 = self._sample_neighbors(edge_index, 10)
        
        # First GAT layer with multi-head attention
        h = self.gat1(x, edge_index1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GAT layer
        h = self.gat2(h, edge_index2)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Third GAT layer (embeddings layer)
        embeddings = self.gat3(h, edge_index3)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        
        # Dense layer for classification
        predictions = self.classifier(embeddings)
        predictions = F.log_softmax(predictions, dim=1)
        
        return embeddings, predictions
    
    def _sample_neighbors(self, edge_index, num_neighbors):
        """
        Sample a fixed number of neighbors for each node.
        
        Args:
            edge_index: Edge index tensor
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Sampled edge index tensor
        """
        if not self.training:
            return edge_index
            
        # Get unique source and target nodes
        source_nodes = torch.unique(edge_index[0])
        target_nodes = torch.unique(edge_index[1])
        all_nodes = torch.unique(torch.cat([source_nodes, target_nodes]))
        
        new_edges_list = []
        
        for node in all_nodes:
            # Get neighbors (both incoming and outgoing)
            outgoing_mask = edge_index[0] == node
            incoming_mask = edge_index[1] == node
            
            outgoing_neighbors = edge_index[1][outgoing_mask]
            incoming_neighbors = edge_index[0][incoming_mask]
            
            # Combine and get unique neighbors (handle empty tensors)
            if len(outgoing_neighbors) > 0 and len(incoming_neighbors) > 0:
                neighbors = torch.unique(torch.cat([outgoing_neighbors, incoming_neighbors]))
            elif len(outgoing_neighbors) > 0:
                neighbors = torch.unique(outgoing_neighbors)
            elif len(incoming_neighbors) > 0:
                neighbors = torch.unique(incoming_neighbors)
            else:
                continue  # No neighbors found
            
            # Remove self-loops if present
            neighbors = neighbors[neighbors != node]
            
            if len(neighbors) == 0:
                # If no neighbors, skip this node
                continue
            elif len(neighbors) >= num_neighbors:
                # Sample without replacement
                indices = torch.randperm(len(neighbors))[:num_neighbors]
                sampled_neighbors = neighbors[indices]
            else:
                # Sample with replacement if we don't have enough neighbors
                indices = torch.randint(0, len(neighbors), (num_neighbors,))
                sampled_neighbors = neighbors[indices]
            
            # Create edges in both directions to maintain graph structure
            for neighbor in sampled_neighbors:
                new_edges_list.append([node.item(), neighbor.item()])
                new_edges_list.append([neighbor.item(), node.item()])
        
        if len(new_edges_list) == 0:
            # Return original edge_index if no sampling was possible
            return edge_index
            
        # Convert to tensor and transpose to get proper shape [2, num_edges]
        new_edges = torch.tensor(new_edges_list, dtype=edge_index.dtype, device=edge_index.device).t()
        
        # Remove duplicate edges
        new_edges = torch.unique(new_edges, dim=1)
        
        return new_edges


class GINModel(BaseGNNModel):
    """
    Graph Isomorphism Network model for grove.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, num_layers=3, dropout=0.5, name=None):
        """
        Initialize the GIN model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            num_layers: Number of GIN layers
            dropout: Dropout probability
            name: Name of the model
        """
        super(GINModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        self.num_layers = num_layers
        
        # MLP for each GIN layer
        self.mlp1 = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        
        # Final MLP for embeddings
        self.mlp3 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        
        # Graph convolution layers
        self.gin1 = GINConv(self.mlp1)
        self.gin2 = GINConv(self.mlp2)
        self.gin3 = GINConv(self.mlp3)
        
        # Dense layer for classification
        self.classifier = nn.Linear(h_feats, out_feats)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Sample neighbors for each layer, fixed at 10
        edge_index1 = self._sample_neighbors(edge_index, 10)
        edge_index2 = self._sample_neighbors(edge_index, 10)
        edge_index3 = self._sample_neighbors(edge_index, 10)
        
        # First GIN layer
        h = self.gin1(x, edge_index1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GIN layer
        h = self.gin2(h, edge_index2)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Third GIN layer
        embeddings = self.gin3(h, edge_index3)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        
        # Dense layer for classification
        predictions = self.classifier(embeddings)
        predictions = F.log_softmax(predictions, dim=1)
        
        return embeddings, predictions
    
    def _sample_neighbors(self, edge_index, num_neighbors):
        """
        Sample a fixed number of neighbors for each node.
        
        Args:
            edge_index: Edge index tensor
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Sampled edge index tensor
        """
        if not self.training:
            return edge_index
            
        # Get unique source and target nodes
        source_nodes = torch.unique(edge_index[0])
        target_nodes = torch.unique(edge_index[1])
        all_nodes = torch.unique(torch.cat([source_nodes, target_nodes]))
        
        new_edges_list = []
        
        for node in all_nodes:
            # Get neighbors (both incoming and outgoing)
            outgoing_mask = edge_index[0] == node
            incoming_mask = edge_index[1] == node
            
            outgoing_neighbors = edge_index[1][outgoing_mask]
            incoming_neighbors = edge_index[0][incoming_mask]
            
            # Combine and get unique neighbors (handle empty tensors)
            if len(outgoing_neighbors) > 0 and len(incoming_neighbors) > 0:
                neighbors = torch.unique(torch.cat([outgoing_neighbors, incoming_neighbors]))
            elif len(outgoing_neighbors) > 0:
                neighbors = torch.unique(outgoing_neighbors)
            elif len(incoming_neighbors) > 0:
                neighbors = torch.unique(incoming_neighbors)
            else:
                continue  # No neighbors found
            
            # Remove self-loops if present
            neighbors = neighbors[neighbors != node]
            
            if len(neighbors) == 0:
                # If no neighbors, skip this node
                continue
            elif len(neighbors) >= num_neighbors:
                # Sample without replacement
                indices = torch.randperm(len(neighbors))[:num_neighbors]
                sampled_neighbors = neighbors[indices]
            else:
                # Sample with replacement if we don't have enough neighbors
                indices = torch.randint(0, len(neighbors), (num_neighbors,))
                sampled_neighbors = neighbors[indices]
            
            # Create edges in both directions to maintain graph structure
            for neighbor in sampled_neighbors:
                new_edges_list.append([node.item(), neighbor.item()])
                new_edges_list.append([neighbor.item(), node.item()])
        
        if len(new_edges_list) == 0:
            # Return original edge_index if no sampling was possible
            return edge_index
            
        # Convert to tensor and transpose to get proper shape [2, num_edges]
        new_edges = torch.tensor(new_edges_list, dtype=edge_index.dtype, device=edge_index.device).t()
        
        new_edges = torch.unique(new_edges, dim=1)
        
        return new_edges


class GraphSAGEModel(BaseGNNModel):
    """
    GraphSAGE model for grove.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, num_layers=2, dropout=0.5, aggregator_type='mean', name=None):
        """
        Initialize the GraphSAGE model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            aggregator_type: Aggregator type
            name: Name of the model
        """
        super(GraphSAGEModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        self.num_layers = num_layers
        self.aggregator_type = aggregator_type
        
        self.sage1 = SAGEConv(in_feats, h_feats, normalize=True, aggr=aggregator_type)
        self.sage2 = SAGEConv(h_feats, h_feats, normalize=True, aggr=aggregator_type)
        
        self.classifier = nn.Linear(h_feats, out_feats)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        x, edge_index = data.x, data.edge_index
        
        edge_index1 = self._sample_neighbors(edge_index, 25)
        edge_index2 = self._sample_neighbors(edge_index, 10)
        
        # First GraphSAGE layer
        h = self.sage1(x, edge_index1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GraphSAGE layer (last hidden layer)
        embeddings = self.sage2(h, edge_index2)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        
        # Dense layer for classification
        predictions = self.classifier(embeddings)
        predictions = F.log_softmax(predictions, dim=1)
        
        return embeddings, predictions
    
    def _sample_neighbors(self, edge_index, num_neighbors):
        """
        Sample a fixed number of neighbors for each node.
        
        Args:
            edge_index: Edge index tensor
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Sampled edge index tensor
        """
        if not self.training:
            return edge_index
            
        # Get unique source and target nodes
        source_nodes = torch.unique(edge_index[0])
        target_nodes = torch.unique(edge_index[1])
        all_nodes = torch.unique(torch.cat([source_nodes, target_nodes]))
        
        new_edges_list = []
        
        for node in all_nodes:
            # Get neighbors (both incoming and outgoing)
            outgoing_mask = edge_index[0] == node
            incoming_mask = edge_index[1] == node
            
            outgoing_neighbors = edge_index[1][outgoing_mask]
            incoming_neighbors = edge_index[0][incoming_mask]
            
            # Combine and get unique neighbors (handle empty tensors)
            if len(outgoing_neighbors) > 0 and len(incoming_neighbors) > 0:
                neighbors = torch.unique(torch.cat([outgoing_neighbors, incoming_neighbors]))
            elif len(outgoing_neighbors) > 0:
                neighbors = torch.unique(outgoing_neighbors)
            elif len(incoming_neighbors) > 0:
                neighbors = torch.unique(incoming_neighbors)
            else:
                continue  # No neighbors found
            
            # Remove self-loops if present
            neighbors = neighbors[neighbors != node]
            
            if len(neighbors) == 0:
                # If no neighbors, skip this node
                continue
            elif len(neighbors) >= num_neighbors:
                # Sample without replacement
                indices = torch.randperm(len(neighbors))[:num_neighbors]
                sampled_neighbors = neighbors[indices]
            else:
                # Sample with replacement if we don't have enough neighbors
                indices = torch.randint(0, len(neighbors), (num_neighbors,))
                sampled_neighbors = neighbors[indices]
            
            # Create edges in both directions to maintain graph structure
            for neighbor in sampled_neighbors:
                new_edges_list.append([node.item(), neighbor.item()])
                new_edges_list.append([neighbor.item(), node.item()])
        
        if len(new_edges_list) == 0:
            # Return original edge_index if no sampling was possible
            return edge_index
            
        # Convert to tensor and transpose to get proper shape [2, num_edges]
        new_edges = torch.tensor(new_edges_list, dtype=edge_index.dtype, device=edge_index.device).t()
        
        # Remove duplicate edges
        new_edges = torch.unique(new_edges, dim=1)
        
        return new_edges


class GNNModel(BaseGNNModel):
    """
    GNN model wrapper.
    """
    def __init__(self, model_type, in_feats, h_feats, out_feats, num_features=None, num_classes=None, dropout=0.5, num_heads=4, num_layers=None, name=None):
        """
        Initialize the GNN model.
        
        Args:
            model_type: Type of GNN architecture ('gat', 'gin', or 'sage')
            in_feats: Input feature dimension (can use num_features instead)
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension (can use num_classes instead)
            num_features: Alternative to in_feats (for compatibility)
            num_classes: Alternative to out_feats (for compatibility)
            dropout: Dropout probability
            num_heads: Number of attention heads for GAT
            num_layers: Number of layers for the model (if None, uses defaults: 3 for GAT/GIN, 2 for SAGE)
            name: Name of the model
        """
        # Allow both naming conventions for compatibility
        in_feats = in_feats or num_features
        out_feats = out_feats or num_classes
        
        if in_feats is None or out_feats is None:
            raise ValueError("Either (in_feats, out_feats) or (num_features, num_classes) must be provided")
            
        super(GNNModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        
        self.model_type = model_type.lower()
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Set default num_layers based on model type 
        if num_layers is None:
            if self.model_type in ['gat', 'gin']:
                self.num_layers = 3
            elif self.model_type == 'sage':
                self.num_layers = 2
        
        # Initialize the appropriate model based on model_type
        if self.model_type == 'gat':
            self.model = GATModel(in_feats, h_feats, out_feats, num_heads=num_heads, num_layers=self.num_layers, dropout=dropout, name=name)
        elif self.model_type == 'gin':
            self.model = GINModel(in_feats, h_feats, out_feats, num_layers=self.num_layers, dropout=dropout, name=name)
        elif self.model_type == 'sage':
            self.model = GraphSAGEModel(in_feats, h_feats, out_feats, num_layers=self.num_layers, dropout=dropout, name=name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        return self.model(data)
    
    def save(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state with wrapper metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'in_feats': self.in_feats,
            'h_feats': self.h_feats,
            'out_feats': self.out_feats,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'name': self.name
        }, path)
        
    @classmethod
    def load(cls, path, device=None):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            model_type=checkpoint['model_type'],
            in_feats=checkpoint['in_feats'],
            h_feats=checkpoint['h_feats'],
            out_feats=checkpoint['out_feats'],
            dropout=checkpoint['dropout'],
            num_heads=checkpoint.get('num_heads', 4),
            num_layers=checkpoint.get('num_layers'),
            name=checkpoint.get('name')
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


def load_model(model_name, in_feats, h_feats, out_feats):
    """
    Load a model based on its name.
    
    Args:
        model_name: Name of the model (GAT, GIN, GraphSAGE)
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        
    Returns:
        Loaded model
    """
    if model_name == "GAT":
        return GATModel(in_feats, h_feats, out_feats)
    elif model_name == "GIN":
        return GINModel(in_feats, h_feats, out_feats)
    elif model_name == "GraphSAGE":
        return GraphSAGEModel(in_feats, h_feats, out_feats)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: GAT, GIN, GraphSAGE")


__all__ = ['BaseGNNModel', 'GATModel', 'GINModel', 'GraphSAGEModel', 'GNNModel', 'load_model'] 