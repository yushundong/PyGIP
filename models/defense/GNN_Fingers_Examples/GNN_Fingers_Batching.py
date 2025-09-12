import sys
sys.path.append('.')
import importlib
import numpy as np
from tqdm import tqdm
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Cora, PubMed
#from models.nn import GraphSAGE
#from dgl.dataloading import NeighborSampler, NodeCollator
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj, dense_to_sparse
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data as PyGData
from models.defense.base import BaseDefense


class LearnableGraphFingerprint(nn.Module):
    """
    A learnable graph fingerprint that converts PyG Data components to learnable parameters
    """
    def __init__(self, num_nodes, feature_dim):
        super(LearnableGraphFingerprint, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # Initialize node features as learnable parameters
        self.x = nn.Parameter(torch.randn(num_nodes, feature_dim))
        
        # Initialize adjacency matrix as learnable parameters (dense representation)
        self.adj_matrix = nn.Parameter(torch.zeros(num_nodes, num_nodes))
    
    @classmethod
    def from_pyg_data(cls, x, edge_index, num_nodes, feature_dim):
        """Create learnable fingerprint from PyG Data components"""
        fingerprint = cls(num_nodes, feature_dim)
        
        # Initialize node features
        fingerprint.x.data = x.clone()
        
        # Initialize adjacency matrix from edge_index
        with torch.no_grad():
            # Convert sparse edge_index to dense adjacency matrix
            dense_adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
            fingerprint.adj_matrix.data = dense_adj
        return fingerprint
    
    def forward(self, return_pyg_data=True):
        """Return the graph structure using straight-through estimator"""
        # Get discrete adjacency matrix (0.0 or 1.0) using straight-through estimator
        adj_binary = (self.adj_matrix > 0.5).float()
        adj_binary_st = adj_binary + (self.adj_matrix - self.adj_matrix.detach())
        
        # Convert dense adjacency to sparse edge_index
        edge_index, edge_attr = dense_to_sparse(adj_binary_st)
        
        if return_pyg_data:
            # Return as PyG Data object
            return PyGData(x=self.x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            # Return raw components
            return self.x, edge_index, adj_binary
    
    def get_discrete_adjacency(self):
        """Get the actual discrete adjacency matrix (for verification)"""
        with torch.no_grad():
            return (self.adj_matrix > 0.5).float()
    
    def to_pyg_data(self):
        """Convert to PyG Data object (without gradient tracking)"""
        with torch.no_grad():
            adj_binary = (self.adj_matrix > 0.5).float()
            edge_index, edge_attr = dense_to_sparse(adj_binary)
            return PyGData(x=self.x.detach(), edge_index=edge_index, edge_attr=edge_attr)
    
    def get_original_components(self):
        """Get the original PyG Data components (for debugging)"""
        with torch.no_grad():
            adj_binary = (self.adj_matrix > 0.5).float()
            edge_index, edge_attr = dense_to_sparse(adj_binary)
            return self.x.detach(), edge_index


class Univerifier(nn.Module):
    """
    Unified Verification Mechanism - Binary classifier that takes concatenated outputs
    from suspect models and predicts whether they are pirated or irrelevant.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64,32,16,8,4]):
        super(Univerifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.softmax(self.classifier(x), dim=1)

class GNNFingers(BaseDefense):
    """
    GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks
    Implementation based on the paper by You et al. (2024)
    """
    supported_api_types = {"dgl"}
    
    def __init__(self, dataset, attack_node_fraction=0.2, device=None, attack_name=None,
                 num_fingerprints=64, fingerprint_nodes=32, lambda_threshold=0.7,  
                 fingerprint_update_epochs=5, univerifier_update_epochs=3, 
                 fingerprint_lr=0.01, univerifier_lr=0.001, top_k_ratio=0.1, 
                 epochs=100, batch_size=32, num_neighbors=[10, 5]):
        """
        Initialize GNNFingers defense framework
        
        Parameters
        ----------
        dataset : Dataset
            The original dataset containing the graph to defend
        attack_node_fraction : float
            Fraction of nodes to consider for attack
        device : torch.device
            Device to run computations on
        attack_name : str
            Name of the attack class to use
        num_fingerprints : int
            Number of graph fingerprints to generate
        fingerprint_nodes : int
            Number of nodes in each fingerprint graph
        lambda_threshold : float
            Threshold for Univerifier classification
        fingerprint_update_epochs: int
            Number of Epochs to update Fingerprint
        univerifier_update_epochs: int
            Number of Epochs to update Univerifier
        fingerprint_lr: float
            Learning rate for fingerprint update
        univerifier_lr: float
            Learning rate for Univerifier update
        top_k_ratio: float
            top k gradients of fingerprint adjacency matrix to select
        epochs: int
            total number of epochs to run experiment
        batch_size : int
            Batch size for training
        num_neighbors : list
            Number of neighbors to sample at each layer
        """
        super().__init__(dataset, attack_node_fraction, device)
        self.attack_name = attack_name or "ModelExtractionAttack0"
        self.dataset = dataset
        self.graph = dataset.graph_data
        
        # Extract dataset properties
        self.node_number = dataset.num_nodes
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes
        self.attack_node_number = int(self.node_number * attack_node_fraction)
        
        # Training parameters
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors

        # Convert DGL to PyG data
        self.pyg_data = self._dgl_to_pyg(self.graph)
        
        # Extract features and labels
        self.features = self.pyg_data.x
        self.labels = self.pyg_data.y
        
        # Extract masks
        self.train_mask = self.pyg_data.train_mask
        self.test_mask = self.pyg_data.test_mask
        
        # GNNFingers parameters
        self.num_fingerprints = num_fingerprints
        self.fingerprint_nodes = fingerprint_nodes
        self.lambda_threshold = lambda_threshold
        
        # Initialize components
        self.target_gnn = None
        self.positive_gnns = []  # Pirated GNNs
        self.negative_gnns = []  # Irrelevant GNNs
        self.graph_fingerprints = None
        self.univerifier = None

        self.fingerprint_lr = fingerprint_lr
        self.fingerprint_update_epochs = fingerprint_update_epochs
        self.univerifier_update_epochs = univerifier_update_epochs
        self.univerifier_lr = univerifier_lr
        self.top_k_ratio = top_k_ratio
        self.epochs = epochs
        
        # Move tensors to device
        if self.device != 'cpu':
            self.graph = self.graph.to(self.device)
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.train_mask = self.train_mask.to(self.device)
            self.test_mask = self.test_mask.to(self.device)
    
    def _dgl_to_pyg(self, dgl_graph):
        """Convert DGL graph to PyTorch Geometric Data object"""
        # Extract edge indices
        edge_index = torch.stack(dgl_graph.edges())
        x = dgl_graph.ndata.get('feat')
        y = dgl_graph.ndata.get('label')

        train_mask = dgl_graph.ndata.get('train_mask')
        val_mask = dgl_graph.ndata.get('val_mask')
        test_mask = dgl_graph.ndata.get('test_mask')

        data = PyGData(x=x, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        return data
   
    def _create_dataloaders(self, graph_data, batch_size=None, num_neighbors=None):
        """
        Create train and test dataloaders for PyG data
        
        Parameters
        ----------
        graph_data : PyG Data
            The graph data to create loaders for
        batch_size : int, optional
            Batch size (defaults to self.batch_size)
        num_neighbors : list, optional
            Number of neighbors to sample (defaults to self.num_neighbors)
        
        Returns
        -------
        train_loader : NeighborLoader
            Training dataloader
        test_loader : NeighborLoader
            Test dataloader
        """
        batch_size = batch_size or self.batch_size
        num_neighbors = num_neighbors or self.num_neighbors
        
        train_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
            input_nodes=graph_data.train_mask,
        )
        
        test_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes=graph_data.test_mask,
        )
        
        return train_loader, test_loader

    def defend(self, attack_name=None):
        """
        Main defense workflow for GNNFingers
        """
        attack_name = attack_name or self.attack_name
        AttackClass = self._get_attack_class(attack_name)
        print(f"Using attack method: {attack_name}")
        
        # Step 1: Train target model
        print("Training target GNN...")
        self.target_gnn = self._train_gnn_model(self.pyg_data, "Target GNN")
        
        # Step 2: Prepare positive and negative GNNs
        print("Preparing positive (pirated) GNNs...")
        self.positive_gnns = self._prepare_positive_gnns(self.target_gnn, num_models=50)
        
        print("Preparing negative (irrelevant) GNNs...")
        self.negative_gnns = self._prepare_negative_gnns(num_models=50)
        
        # Step 3: Initialize graph fingerprints
        print("Initializing graph fingerprints...")
        self.graph_fingerprints = self._initialize_graph_fingerprints()
        
        # Step 4: Initialize Univerifier
        output_dim = self._get_output_dimension(self.target_gnn, self.graph_fingerprints[0])
        self.univerifier = Univerifier(input_dim=output_dim * self.num_fingerprints)
        self.univerifier = self.univerifier.to(self.device)
        
        # Step 5: Joint learning of fingerprints and Univerifier
        print("Joint learning of fingerprints and Univerifier...")
        self._joint_learning_alternating()
        
        # Step 6: Attack target model
        print("Attacking target model...")
        attack = AttackClass(self.dataset, attack_node_fraction=0.2)
        attack_results = attack.attack()
        suspect_model = attack.net2 if hasattr(attack, 'net2') else None
        
        # Step 7: Verify ownership
        if suspect_model is not None:
            print("Verifying ownership of suspect model...")
            verification_result = self._verify_ownership(suspect_model)
            print(f"Ownership verification result: {verification_result}")
            
            return {
                "attack_results": attack_results,
                "verification_result": verification_result,
                "target_accuracy": self._evaluate_model(self.target_gnn, self.pyg_data),
                "suspect_accuracy": self._evaluate_model(suspect_model, self.pyg_data)
            }
        
        return {"attack_results": attack_results, "verification_result": "No suspect model found"}
    
    def _train_gnn_model(self, data, model_name="GNN", epochs=100):
        """Train a GNN model on the given data using batched training"""
        model = GCNConvGNN(
            in_channels=data.x.size(1),
            hidden_channels=128,
            out_channels=self.label_number,
            num_layers=3
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Create dataloaders using the helper function
        train_loader, test_loader = self._create_dataloaders(data)
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluate on test set
            if epoch % 10 == 0:
                test_acc = self._evaluate_model_with_loader(model, test_loader)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = copy.deepcopy(model)
                
                print(f"{model_name} Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={test_acc:.4f}")
        
        print(f"{model_name} trained with best accuracy: {best_acc:.4f}")
        return best_model
    
    def _evaluate_model_with_loader(self, model, test_loader):
        """Evaluate model accuracy using a test dataloader"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                pred = out.argmax(dim=1)
                
                # Only count test nodes
                test_mask = batch.test_mask if hasattr(batch, 'test_mask') else torch.ones(batch.num_nodes, dtype=bool)
                correct += (pred[test_mask] == batch.y[test_mask]).sum().item()
                total += test_mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    def _prepare_positive_gnns(self, target_model, num_models=50):
        """Prepare pirated GNNs using obfuscation techniques"""
        positive_models = []
        
        for i in range(num_models):
            # Apply different obfuscation techniques
            if i % 3 == 0:
                # Fine-tuning with batched training
                layers_to_finetune = random.randint(1, 3)
                model = self._fine_tune_model(copy.deepcopy(target_model), self.pyg_data, 
                                            epochs=10, num_layers_to_finetune=layers_to_finetune)
            elif i % 3 == 1:
                # Partial retraining with batched training
                layers_to_retrain = random.randint(1, 3)
                model = self._partial_retrain_model(copy.deepcopy(target_model), self.pyg_data, 
                                                epochs=15, num_layers_to_retrain=layers_to_retrain)
            else:
                # Distillation with batched training
                temperature = random.choice([1.5, 2.0, 3.0, 4.0])
                model = self._distill_model(target_model, self.pyg_data, 
                                        epochs=30, temperature=temperature)
            
            positive_models.append(model)
        
        return positive_models
    
    def _prepare_negative_gnns(self, num_models=50):
        """Prepare irrelevant GNNs"""
        negative_models = []
        
        for i in range(num_models):
            # Train from scratch with different architectures or data
            if i % 2 == 0:
                # Different architecture
                model = self._train_different_architecture(self.pyg_data)
            else:
                # Different training data (subset)
                model = self._train_on_subset(self.pyg_data)
            
            negative_models.append(model)
        
        return negative_models
    
    def _fine_tune_model(self, model, data, epochs=10, num_layers_to_finetune=1):
        """Fine-tune a model using batched training"""
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last K layers for fine-tuning
        if hasattr(model, 'convs'):
            total_layers = len(model.convs)
            layers_to_finetune = min(num_layers_to_finetune, total_layers)
            
            for i in range(total_layers - layers_to_finetune, total_layers):
                for param in model.convs[i].parameters():
                    param.requires_grad = True
        
        # Create dataloader using helper function
        train_loader, _ = self._create_dataloaders(data)
        
        # Only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Unfreeze all parameters for future use
        for param in model.parameters():
            param.requires_grad = True
        
        return model
    
    def _partial_retrain_model(self, model, data, epochs=10, num_layers_to_retrain=2):
        """Partially retrain a model with random initialization of K layers before resuming training"""
        # Randomly initialize selected K layers
        if hasattr(model, 'convs'):
            # For models with convs attribute (like GCNConvGNN, GATConvGNN)
            total_layers = len(model.convs)
            layers_to_retrain = min(num_layers_to_retrain, total_layers)
            
            # Randomly select K layers to retrain
            layer_indices = random.sample(range(total_layers), layers_to_retrain)
            
            print(f"Partially retraining layers: {layer_indices}")
            
            for idx in layer_indices:
                model.convs[idx].reset_parameters()  # Random reinitialization
        
        # Train the entire model (both retrained and original layers)
        train_loader, test_loader = self._create_dataloaders(data)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Track best model
            if epoch % 5 == 0:
                acc = self._evaluate_model_with_loader(model, test_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
        
        print(f"Partial retraining completed. Best accuracy: {best_acc:.4f}")
        return best_model if best_model is not None else model
    
    def _distill_model(self, teacher_model, data, epochs=30, temperature=2.0):
        """Distill knowledge using batched training"""
        # Create student model with different architecture
        if isinstance(teacher_model, GCNConvGNN):
            # If teacher is GCN, use GAT as student
            student_model = GATConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=96,  # Different hidden size
                out_channels=self.label_number,
                num_layers=2,        # Different number of layers
                heads=3              # Different number of heads
            ).to(self.device)
        else:
            # If teacher is GAT or other, use GCN as student
            student_model = GCNConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,  # Different hidden size
                out_channels=self.label_number,
                num_layers=3         # Different number of layers
            ).to(self.device)
        
        # Create dataloader using helper function
        train_loader, test_loader = self._create_dataloaders(data)
        
        optimizer = optim.Adam(student_model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Combined loss: KL divergence for distillation + cross entropy for ground truth
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()
        
        teacher_model.eval()
        
        best_acc = 0
        best_student = None
        
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Get teacher predictions (with temperature scaling)
                with torch.no_grad():
                    teacher_logits = teacher_model(batch.x, batch.edge_index)
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                
                # Get student predictions
                student_logits = student_model(batch.x, batch.edge_index)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                
                # Distillation loss (KL divergence between teacher and student)
                distill_loss = kl_loss(student_log_probs[batch.train_mask], 
                                    teacher_probs[batch.train_mask]) * (temperature ** 2)
                
                # Student's own classification loss
                class_loss = ce_loss(student_logits[batch.train_mask], 
                                    batch.y[batch.train_mask])
                
                # Combined loss (weighted sum)
                loss = 0.7 * distill_loss + 0.3 * class_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Track best student model
            if epoch % 5 == 0:
                student_model.eval()
                test_acc = self._evaluate_model_with_loader(student_model, test_loader)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_student = copy.deepcopy(student_model)
                
                print(f"Distillation Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={test_acc:.4f}")
        
        print(f"Distillation completed. Best student accuracy: {best_acc:.4f}")
        return best_student if best_student is not None else student_model
    
    def _train_different_architecture(self, data):
        """Train a model with different architecture for negative GNNs"""
        # Use opposite architecture of target model
        if isinstance(self.target_gnn, GCNConvGNN):
            model = GATConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,
                out_channels=self.label_number,
                num_layers=2,
                heads=4
            ).to(self.device)
        else:
            model = GCNConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,
                out_channels=self.label_number,
                num_layers=3
            ).to(self.device)
    
        return self._train_gnn_model_with_data(model, data, epochs=50)
    
    def _train_on_subset(self, data, subset_ratio=0.7):
        """Train on a subset of the data"""
        # Create subset mask
        num_train = int(data.train_mask.sum().item() * subset_ratio)
        subset_mask = torch.zeros_like(data.train_mask)
        train_indices = data.train_mask.nonzero(as_tuple=True)[0]
        selected_indices = random.sample(range(len(train_indices)), min(num_train, len(train_indices)))
        subset_mask[train_indices[selected_indices]] = True
        
        # Create subset data
        subset_data = PyGData(
            x=data.x, 
            edge_index=data.edge_index, 
            y=data.y,
            train_mask=subset_mask,
            test_mask=data.test_mask
        )
        
        # Use opposite architecture of target model
        if isinstance(self.target_gnn, GCNConvGNN):
            model = GCNConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,
                out_channels=self.label_number,
                num_layers=2
            ).to(self.device)
        else:
            model = GATConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,
                out_channels=self.label_number,
                num_layers=3,
                heads=4
            ).to(self.device)
        
        return self._train_gnn_model_with_data(model, subset_data, epochs=50)
    
    def _train_gnn_model_with_data(self, model, data, epochs=100):
        """Train a specific model on specific data using batched training"""
        # Create dataloaders using helper function
        train_loader, test_loader = self._create_dataloaders(data)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                acc = self._evaluate_model_with_loader(model, test_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
        
        return best_model
    
    def _initialize_graph_fingerprints(self):
        """Initialize graph fingerprints as learnable parameters from PyG Data"""
        fingerprints = nn.ModuleList()  # Use ModuleList to properly register parameters
        feature_dim = self.features.size(1) if self.features is not None else 16
        
        for _ in range(self.num_fingerprints):
            # Initialize random graph using Erdos-Renyi model
            edge_index = erdos_renyi_graph(self.fingerprint_nodes, 0.3)
            
            # Initialize node features
            if self.features is not None:
                x = torch.randn(self.fingerprint_nodes, self.features.size(1))
            else:
                x = torch.randn(self.fingerprint_nodes, 16)
            
            # Convert to learnable fingerprint
            fingerprint = LearnableGraphFingerprint.from_pyg_data(
                x, edge_index, self.fingerprint_nodes, feature_dim
            ).to(self.device)
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    def _get_output_dimension(self, model, fingerprint_data):
        """Get the output dimension of a model for a given fingerprint"""
        model.eval()
        with torch.no_grad():
            output = model(fingerprint_data.x.to(self.device), 
                          fingerprint_data.edge_index.to(self.device))
            return output.size(1)
    
    def _verify_ownership(self, suspect_model):
        """Verify if a suspect model is pirated from the target model"""
        # Get outputs for all fingerprints
        target_outputs = []
        suspect_outputs = []
        
        for fingerprint in self.graph_fingerprints:
            self.target_gnn.eval()
            suspect_model.eval()
            # Get fingerprint as PyG Data object (without gradient tracking)
            fingerprint = fingerprint.to_pyg_data()
            
            with torch.no_grad():
                target_out = self.target_gnn(fingerprint.x.to(self.device), 
                                           fingerprint.edge_index.to(self.device))
                suspect_out = suspect_model(fingerprint.x.to(self.device), 
                                          fingerprint.edge_index.to(self.device))
            
            target_outputs.append(target_out)
            suspect_outputs.append(suspect_out)
        
        # Concatenate outputs
        target_concat = torch.cat(target_outputs, dim=0).view(1, -1)
        suspect_concat = torch.cat(suspect_outputs, dim=0).view(1, -1)
        
        # Get Univerifier prediction
        self.univerifier.eval()
        with torch.no_grad():
            prediction = self.univerifier(suspect_concat)
            confidence = prediction[0, 1].item()  # Probability of being pirated
        
        return confidence > self.lambda_threshold, confidence
    
    def _evaluate_model(self, model, data):
        """Evaluate model accuracy"""
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
            total = data.test_mask.sum().item()
            return correct / total if total > 0 else 0

    def _update_adjacency_discrete(self, fingerprint, grad_adj):
        """
        Update discrete adjacency matrix based on gradients
        """
        # Get current discrete adjacency
        current_adj = fingerprint.get_discrete_adjacency()
        
        # Get absolute gradient values and flatten
        grad_abs = torch.abs(grad_adj)
        grad_abs_flat = grad_abs.view(-1)
        
        # Determine top-K edges to consider for flipping
        k = int(self.top_k_ratio * self.fingerprint_nodes * self.fingerprint_nodes)
        topk_values, topk_indices = torch.topk(grad_abs_flat, k)
        
        # Convert flat indices to row, col indices
        rows = topk_indices // self.fingerprint_nodes
        cols = topk_indices % self.fingerprint_nodes
        
        # Update edges based on gradient signs
        with torch.no_grad():
            for idx in range(k):
                row, col = rows[idx], cols[idx]
                grad_val = grad_adj[row, col]
                
                # Current edge existence (0 or 1)
                current_edge = current_adj[row, col]
                
                # Apply update rules:
                if current_edge > 0.5 and grad_val <= 0:
                    # Edge exists and gradient is negative → remove edge
                    fingerprint.adj_matrix.data[row, col] = 0.0
                elif current_edge < 0.5 and grad_val >= 0:
                    # Edge doesn't exist and gradient is positive → add edge
                    fingerprint.adj_matrix.data[row, col] = 1.0

    def _update_fingerprints_discrete(self, loss, top_k_ratio=0.1):
        """
        Update graph fingerprints using gradients
        """
        # Compute gradients for all fingerprints
        gradients_adj = []
        gradients_x = []
        
        for fingerprint in self.graph_fingerprints:
            # Compute gradients for adjacency matrix
            grad_adj = torch.autograd.grad(
                loss, fingerprint.adj_matrix, 
                retain_graph=True, create_graph=False
            )[0]
            
            # Compute gradients for node features
            grad_x = torch.autograd.grad(
                loss, fingerprint.x,
                retain_graph=True, create_graph=False
            )[0]
            
            gradients_adj.append(grad_adj)
            gradients_x.append(grad_x)
        
        # Update each fingerprint
        for i, fingerprint in enumerate(self.graph_fingerprints):
            grad_adj = gradients_adj[i]
            grad_x = gradients_x[i]
            
            # Update node features with clipping
            with torch.no_grad():
                fingerprint.x.data += self.fingerprint_lr * grad_x
                
                # Clip node features to reasonable range
                if self.features is not None:
                    min_val = self.features.min().item()
                    max_val = self.features.max().item()
                    fingerprint.x.data = torch.clamp(fingerprint.x.data, min_val, max_val)
                else:
                    fingerprint.x.data = torch.clamp(fingerprint.x.data, -3, 3)
            
            # Update adjacency matrix using discrete strategy
            self._update_adjacency_discrete(fingerprint, grad_adj, top_k_ratio)

    def visualize_fingerprint_evolution(self, epoch):
        """Visualize how fingerprints evolve during training"""
        if epoch % 20 == 0:  # Visualize every 20 epochs
            print(f"\n=== Fingerprint Evolution at Epoch {epoch} ===")
            
            for i, fingerprint in enumerate(self.graph_fingerprints[:2]):  # First 2 only
                x, edge_index = fingerprint.get_original_components()
                current_adj = fingerprint.get_discrete_adjacency()
                
                # Calculate statistics
                num_edges = current_adj.sum().item()
                sparsity = 1 - (num_edges / (self.fingerprint_nodes * self.fingerprint_nodes))
                
                print(f"Fingerprint {i}: {num_edges} edges, sparsity: {sparsity:.3f}")
                
                # Feature statistics
                feature_mean = x.mean().item()
                feature_std = x.std().item()
                print(f"  Features: mean={feature_mean:.3f}, std={feature_std:.3f}")

    def _joint_learning_alternating(self):
        """
        Joint learning with alternating optimization algorithm
        """
        
        # Prepare all models and labels
        all_models = [self.target_gnn] + self.positive_gnns + self.negative_gnns
        labels = torch.cat([
            torch.ones(len(self.positive_gnns) + 1),  # Target + positive models
            torch.zeros(len(self.negative_gnns))      # Negative models
        ]).long().to(self.device)
        
        # Flag to alternate between fingerprint and univerifier updates
        update_fingerprints = True
        
        for epoch in range(self.epochs):
            # Forward pass through all models using the actual discrete structure
            all_outputs = []
            for model in all_models:
                model_outputs = []
                for fingerprint in self.graph_fingerprints:
                    model.eval()
                    
                    # Get fingerprint as PyG Data object (this uses straight-through estimator)
                    fingerprint_data = fingerprint(return_pyg_data=True)
                    
                    with torch.no_grad():
                        # Pass through the model
                        output = model(fingerprint_data.x, fingerprint_data.edge_index)
                        model_outputs.append(output)
                
                # Concatenate all fingerprint outputs
                concatenated = torch.cat(model_outputs, dim=0).view(1, -1)
                all_outputs.append(concatenated)
            
            # Stack all outputs
            all_outputs = torch.cat(all_outputs, dim=0)
            
            # Univerifier prediction
            univerifier_out = self.univerifier(all_outputs)
            
            # Calculate joint loss
            loss = 0
            for i, model in enumerate(all_models):
                if i < len(self.positive_gnns) + 1:  # Target + positive models
                    # log o_+(F) and log o_+(F_+) terms
                    loss += torch.log(univerifier_out[i, 1] + 1e-10)
                else:  # Negative models
                    # log o_-(F_-) term
                    loss += torch.log(1 - univerifier_out[i, 1] + 1e-10)
            
            loss = -loss  # Negative log likelihood (minimize negative log likelihood)
            
            # Alternating optimization
            if update_fingerprints:
                # Phase 1: Update fingerprints for e1 epochs
                for e in range(self.fingerprint_update_epochs):
                    self._update_fingerprints_discrete(loss, self.top_k_ratio)
                
                update_fingerprints = False
                print(f"Epoch {epoch}: Updated fingerprints, Loss: {loss.item():.4f}")
                
            else:
                # Phase 2: Update Univerifier for e2 epochs
                univerifier_optimizer = optim.Adam(self.univerifier.parameters(), lr=self.univerifier_lr)
                
                for e in range(self.univerifier_update_epochs):
                    univerifier_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    univerifier_optimizer.step()
                
                update_fingerprints = True
                print(f"Epoch {epoch}: Updated Univerifier, Loss: {loss.item():.4f}")
            
            # Calculate accuracy every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    preds = univerifier_out.argmax(dim=1)
                    acc = (preds == labels).float().mean().item()
                    
                    # Calculate true positive and true negative rates
                    tp_mask = (preds == 1) & (labels == 1)
                    tn_mask = (preds == 0) & (labels == 0)
                    
                    tp_rate = tp_mask.float().mean().item() if (labels == 1).sum() > 0 else 0
                    tn_rate = tn_mask.float().mean().item() if (labels == 0).sum() > 0 else 0
                    
                    print(f"Epoch {epoch}, Acc: {acc:.4f}, TP: {tp_rate:.4f}, TN: {tn_rate:.4f}")
            
            # Visualize fingerprint evolution
            if epoch % 20 == 0:
                self.visualize_fingerprint_evolution(epoch)


    def _verify_ownership_detailed(self, suspect_model):
        """Detailed verification for debugging purposes only"""
        suspect_outputs = []
        target_outputs = []
        
        for fingerprint in self.graph_fingerprints:
            suspect_model.eval()
            self.target_gnn.eval()
            
            fingerprint_data = fingerprint.to_pyg_data()
            
            with torch.no_grad():
                suspect_out = suspect_model(
                    fingerprint_data.x.to(self.device), 
                    fingerprint_data.edge_index.to(self.device)
                )
                suspect_outputs.append(suspect_out)
                
                target_out = self.target_gnn(
                    fingerprint_data.x.to(self.device),
                    fingerprint_data.edge_index.to(self.device)
                )
                target_outputs.append(target_out)
        
        suspect_concat = torch.cat(suspect_outputs, dim=0).view(1, -1)
        target_concat = torch.cat(target_outputs, dim=0).view(1, -1)
        
        self.univerifier.eval()
        with torch.no_grad():
            suspect_prediction = self.univerifier(suspect_concat)
            suspect_confidence = suspect_prediction[0, 1].item()
            
            target_prediction = self.univerifier(target_concat)
            target_confidence = target_prediction[0, 1].item()
        
        output_similarity = F.cosine_similarity(suspect_concat, target_concat).item()
        is_pirated = suspect_confidence > self.lambda_threshold
        
        # Return detailed info for debugging, but main method keeps original interface
        return {
            'is_pirated': is_pirated,
            'confidence': suspect_confidence,
            'target_confidence': target_confidence,
            'output_similarity': output_similarity,
            'lambda_threshold': self.lambda_threshold
        }

    
# GNN Model Definitions
class GCNConvGNN(nn.Module):
    """GCN-based GNN model"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCNConvGNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training, p=0.5)
        return x

class GATConvGNN(nn.Module):

    """GAT-based GNN model"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4):
        super(GATConvGNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training, p=0.6)
        return x


if __name__ == "__main__":

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = Cora(api_type="dgl", path="./data")
    print(f"Loaded dataset: {dataset}")
    
    # Initialize defense
    defense = GNNFingers(
        dataset=dataset,
        device=device,
        num_fingerprints=32,
        fingerprint_nodes=64,
        epochs=100
    )
    
    # Run defense
    results = defense.defend()
    
    # Print results
    print("\n=== Defense Results ===")
    print(f"Target Accuracy: {results.get('target_accuracy', 0):.4f}")
    print(f"Suspect Accuracy: {results.get('suspect_accuracy', 0):.4f}")
    print(f"Verification Result: {results.get('verification_result', 'Unknown')}")