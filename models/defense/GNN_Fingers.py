import importlib
import numpy as np
from tqdm import tqdm
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import dataset
#from models.nn import GraphSAGE
#from dgl.dataloading import NeighborSampler, NodeCollator
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj, dense_to_sparse
#from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.data import Data as PyGData
from models.defense.base import BaseDefense


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
                 batch_size=32, num_neighbors=[5, 5]):
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
        # self.batch_size = batch_size
        # self.num_neighbors = num_neighbors

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
   
   
    # def _create_dataloaders(self, graph_data):
    #     """Create train and test dataloaders with neighbor sampling"""
    #     # For DGL graphs
    #     if hasattr(graph_data, 'ndata'):
    #         # DGL graph
    #         sampler = NeighborSampler(self.num_neighbors)
    #         train_nids = graph_data.ndata['train_mask'].nonzero(as_tuple=True)[0].to(self.device)
    #         test_nids = graph_data.ndata['test_mask'].nonzero(as_tuple=True)[0].to(self.device)
            
    #         train_collator = NodeCollator(graph_data, train_nids, sampler)
    #         test_collator = NodeCollator(graph_data, test_nids, sampler)
            
    #         train_dataloader = DataLoader(
    #             train_collator.dataset,
    #             batch_size=self.batch_size,
    #             shuffle=True,
    #             collate_fn=train_collator.collate,
    #             drop_last=False
    #         )
            
    #         test_dataloader = DataLoader(
    #             test_collator.dataset,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             collate_fn=test_collator.collate,
    #             drop_last=False
    #         )
            
    #         return train_dataloader, test_dataloader
        
    #     else:
    #         # PyG data
    #         train_loader = NeighborLoader(
    #             graph_data,
    #             num_neighbors=self.num_neighbors,
    #             batch_size=self.batch_size,
    #             shuffle=True,
    #             input_nodes=graph_data.train_mask,
    #         )
            
    #         test_loader = NeighborLoader(
    #             graph_data,
    #             num_neighbors=self.num_neighbors,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             input_nodes=graph_data.test_mask,
    #         )
            
    #         return train_loader, test_loader

    def _get_attack_class(self, attack_name):
        """Dynamically import and return the specified attack class"""
        try:
            attack_module = importlib.import_module('models.attack')
            attack_class = getattr(attack_module, attack_name)
            return attack_class
        except (ImportError, AttributeError) as e:
            print(f"Error loading attack class '{attack_name}': {e}")
            print("Falling back to ModelExtractionAttack0")
            attack_module = importlib.import_module('models.attack')
            return getattr(attack_module, "ModelExtractionAttack0")
    
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
        self._joint_learning()
        
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
        """Train a GNN model on the given data"""
        model = GCNConvGNN(
            in_channels=data.x.size(1),
            hidden_channels=128,
            out_channels=self.label_number,
            num_layers=3
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Evaluate ONLY on test nodes
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out[data.test_mask].argmax(dim=1)
                correct = (pred == data.y[data.test_mask]).sum().item()
                total = data.test_mask.sum().item()
                acc = correct / total if total > 0 else 0
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
    
        print(f"{model_name} trained with accuracy: {best_acc:.4f}")
        return best_model
    
    def _prepare_positive_gnns(self, target_model, num_models=50):
        """Prepare pirated GNNs using obfuscation techniques"""
        positive_models = []
        
        for i in range(num_models):
            # Apply different obfuscation techniques
            if i % 3 == 0:
                # Fine-tuning - fine-tune different numbers of layers
                layers_to_finetune = random.randint(1, 3)
                model = self._fine_tune_model(copy.deepcopy(target_model), self.pyg_data, 
                                            epochs=10, num_layers_to_finetune=layers_to_finetune)
            elif i % 3 == 1:
                # Partial retraining - retrain different numbers of layers
                layers_to_retrain = random.randint(1, 3)
                model = self._partial_retrain_model(copy.deepcopy(target_model), self.pyg_data, 
                                                epochs=15, num_layers_to_retrain=layers_to_retrain)
            else:
                # Distillation - use different temperatures and architectures
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
        """Fine-tune a model on the same data, but only update the last K layers"""
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last K layers for fine-tuning
        if hasattr(model, 'convs'):
            # For models with convs attribute (like GCNConvGNN, GATConvGNN)
            total_layers = len(model.convs)
            layers_to_finetune = min(num_layers_to_finetune, total_layers)
            
            for i in range(total_layers - layers_to_finetune, total_layers):
                for param in model.convs[i].parameters():
                    param.requires_grad = True
        else:
            # For other model types, try to find the last layers
            all_params = list(model.parameters())
            layers_to_finetune = min(num_layers_to_finetune, len(all_params))
            
            for param in all_params[-layers_to_finetune:]:
                param.requires_grad = True
        
        # Only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Count how many parameters are being fine-tuned
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Fine-tuning {num_trainable}/{num_total} parameters ({num_trainable/num_total*100:.1f}%)")
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            optimizer.step()
        
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
            
            # Randomly select K layers to retrain (not necessarily the last ones)
            layer_indices = random.sample(range(total_layers), layers_to_retrain)
            
            print(f"Partially retraining layers: {layer_indices}")
            
            for idx in layer_indices:
                model.convs[idx].reset_parameters()  # Random reinitialization
        else:
            # For other model types, try to find and reset random layers
            # This is a fallback approach
            all_layers = list(model.children())
            total_layers = len(all_layers)
            layers_to_retrain = min(num_layers_to_retrain, total_layers)
            
            layer_indices = random.sample(range(total_layers), layers_to_retrain)
            
            for idx in layer_indices:
                if hasattr(all_layers[idx], 'reset_parameters'):
                    all_layers[idx].reset_parameters()
        
        # Train the entire model (both retrained and original layers)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            optimizer.step()
            
            # Track best model
            if epoch % 5 == 0:
                acc = self._evaluate_model(model, data)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
        
        print(f"Partial retraining completed. Best accuracy: {best_acc:.4f}")
        return best_model if best_model is not None else model
    def _distill_model(self, teacher_model, data, epochs=30, temperature=2.0):
        """Distill knowledge from teacher to student model with different architecture"""
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
        
        optimizer = optim.Adam(student_model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Combined loss: KL divergence for distillation + cross entropy for ground truth
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()
        
        teacher_model.eval()
        
        best_acc = 0
        best_student = None
        
        for epoch in range(epochs):
            student_model.train()
            optimizer.zero_grad()
            
            # Get teacher predictions (with temperature scaling)
            with torch.no_grad():
                teacher_logits = teacher_model(data.x, data.edge_index)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            
            # Get student predictions
            student_logits = student_model(data.x, data.edge_index)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            # Distillation loss (KL divergence between teacher and student)
            distill_loss = kl_loss(student_log_probs[data.train_mask], 
                                teacher_probs[data.train_mask]) * (temperature ** 2)
            
            # Student's own classification loss
            class_loss = ce_loss(student_logits[data.train_mask], 
                                data.y[data.train_mask])
            
            # Combined loss (weighted sum)
            loss = 0.7 * distill_loss + 0.3 * class_loss
            
            loss.backward()
            optimizer.step()
            
            # Track best student model
            if epoch % 5 == 0:
                student_model.eval()
                with torch.no_grad():
                    out = student_model(data.x, data.edge_index)
                    pred = out[data.test_mask].argmax(dim=1)
                    correct = (pred == data.y[data.test_mask]).sum().item()
                    total = data.test_mask.sum().item()
                    acc = correct / total if total > 0 else 0
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_student = copy.deepcopy(student_model)
                
                print(f"Distillation Epoch {epoch}: Loss={loss.item():.4f}, "
                    f"Distill={distill_loss.item():.4f}, Class={class_loss.item():.4f}, "
                    f"Acc={acc:.4f}")
        
        print(f"Distillation completed. Best student accuracy: {best_acc:.4f}")
        return best_student if best_student is not None else student_model
    
    def _train_different_architecture(self, data):
        """Train a model with different architecture for negative GNNs"""
        # Use opposite architecture of target model
        if isinstance(self.target_gnn, GCNConv):
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
        if isinstance(self.target_gnn, GCNConv):
            model = GCNConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,
                out_channels=self.label_number,
                num_layers=2,
                heads=4
            ).to(self.device)
        else:
            model = GATConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=64,
                out_channels=self.label_number,
                num_layers=3
            ).to(self.device)
        
        return self._train_gnn_model_with_data(model, subset_data, epochs=50)
    
    def _train_gnn_model_with_data(self, model, data, epochs=100):
        """Train a specific model on specific data"""
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                acc = self._evaluate_model(model, data)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
        
        return best_model
    
    def _initialize_graph_fingerprints(self):
        """Initialize graph fingerprints for node classification task"""
        fingerprints = []
        
        for _ in range(self.num_fingerprints):
            # Initialize random graph using Erdos-Renyi model
            edge_index = erdos_renyi_graph(self.fingerprint_nodes, 0.3)
            
            # Initialize node features
            if self.features is not None:
                x = torch.randn(self.fingerprint_nodes, self.features.size(1))
            else:
                x = torch.randn(self.fingerprint_nodes, 16)  # Default feature dimension
            
            # Create PyG Data object
            fingerprint_data = PyGData(x=x, edge_index=edge_index)
            fingerprints.append(fingerprint_data)
        
        return fingerprints
    
    def _get_output_dimension(self, model, fingerprint_data):
        """Get the output dimension of a model for a given fingerprint"""
        model.eval()
        with torch.no_grad():
            output = model(fingerprint_data.x.to(self.device), 
                          fingerprint_data.edge_index.to(self.device))
            return output.size(1)
    
    def _joint_learning(self, epochs=100, fingerprint_lr=0.01, univerifier_lr=0.001):
        """Joint learning of graph fingerprints and Univerifier"""
        fingerprint_optimizer = optim.Adam([fp.x for fp in self.graph_fingerprints] + 
                                          [fp.edge_index for fp in self.graph_fingerprints], 
                                          lr=fingerprint_lr)
        univerifier_optimizer = optim.Adam(self.univerifier.parameters(), lr=univerifier_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare all models for training
        all_models = [self.target_gnn] + self.positive_gnns + self.negative_gnns
        labels = torch.cat([
            torch.ones(len(self.positive_gnns) + 1),  # Target + positive models
            torch.zeros(len(self.negative_gnns))      # Negative models
        ]).long().to(self.device)
        
        for epoch in range(epochs):
            # Forward pass through all models
            all_outputs = []
            for model in all_models:
                model_outputs = []
                for fingerprint in self.graph_fingerprints:
                    model.eval()
                    with torch.no_grad():
                        output = model(fingerprint.x.to(self.device), 
                                      fingerprint.edge_index.to(self.device))
                        model_outputs.append(output)
                
                # Concatenate all fingerprint outputs
                concatenated = torch.cat(model_outputs, dim=0).view(1, -1)
                all_outputs.append(concatenated)
            
            # Stack all outputs
            all_outputs = torch.cat(all_outputs, dim=0)
            
            # Univerifier prediction
            univerifier_out = self.univerifier(all_outputs)
            
            # Calculate loss
            loss = criterion(univerifier_out, labels)
            
            # Backward pass
            fingerprint_optimizer.zero_grad()
            univerifier_optimizer.zero_grad()
            loss.backward()
            
            # Update fingerprints and Univerifier
            fingerprint_optimizer.step()
            univerifier_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def _verify_ownership(self, suspect_model):
        """Verify if a suspect model is pirated from the target model"""
        # Get outputs for all fingerprints
        target_outputs = []
        suspect_outputs = []
        
        for fingerprint in self.graph_fingerprints:
            self.target_gnn.eval()
            suspect_model.eval()
            
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


    # def _joint_learning_alternating(self, e1=5, e2=5, alpha=0.01, beta=0.001):
    #     """
    #     Implementation of Joint Learning Approach
        
    #     Parameters:
    #     e1: number of fingerprint update epochs
    #     e2: number of Univerifier update epochs  
    #     alpha: learning rate for fingerprints
    #     beta: learning rate for Univerifier
    #     """
    #     flag = 0  # 0: update fingerprints, 1: update Univerifier
    #     convergence_threshold = 1e-4
    #     prev_loss = float('inf')
    #     convergence_count = 0
        
    #     print("Starting alternating optimization...")
        
    #     while convergence_count < 3:  # Converge if loss doesn't improve for 3 cycles
    #         # Compute total loss L (lines 4-10)
    #         L = 0
    #         all_models = [self.target_gnn] + self.positive_gnns + self.negative_gnns
            
    #         for model_idx, model in enumerate(all_models):
    #             # Get concatenated outputs from all fingerprints for this model
    #             fingerprint_outputs = []
    #             for fingerprint in self.graph_fingerprints:
    #                 model.eval()
    #                 with torch.no_grad():
    #                     output = model(fingerprint.x.to(self.device), 
    #                                 fingerprint.edge_index.to(self.device))
    #                     fingerprint_outputs.append(output)
                
    #             # Concatenate all fingerprint outputs (flattened)
    #             concatenated = torch.cat([out.view(-1) for out in fingerprint_outputs]).unsqueeze(0)
                
    #             # Get Univerifier prediction
    #             univerifier_out = self.univerifier(concatenated)
    #             o_plus = univerifier_out[0, 1]  # Probability of being pirated
                
    #             # Accumulate loss according to algorithm
    #             if model_idx < len(self.positive_gnns) + 1:  # Target or positive model
    #                 L += torch.log(o_plus + 1e-10)
    #             else:  # Negative model
    #                 L += torch.log(1 - o_plus + 1e-10)
            
    #         current_loss = -L.item()  # Negative since we're maximizing
            
    #         # Check convergence
    #         if abs(prev_loss - current_loss) < convergence_threshold:
    #             convergence_count += 1
    #         else:
    #             convergence_count = 0
    #         prev_loss = current_loss
            
    #         print(f"Cycle loss: {current_loss:.6f}, Flag: {flag}, Convergence count: {convergence_count}")
            
    #         # Alternating optimization (lines 11-21)
    #         if flag == 0:
    #             # Update fingerprints for e1 epochs
    #             print(f"Updating fingerprints for {e1} epochs...")
    #             for e in range(e1):
    #                 self._update_fingerprints_single_epoch(alpha)
    #             flag = 1
    #         else:
    #             # Update Univerifier for e2 epochs
    #             print(f"Updating Univerifier for {e2} epochs...")
    #             for e in range(e2):
    #                 self._update_univerifier_single_epoch(beta)
    #             flag = 0
        
    #     print("Alternating optimization converged!")



    # def _update_fingerprints_single_epoch(self, alpha=0.01, top_k_edges=10):
    #     """Update fingerprints for one epoch"""
    #     attribute_ranges = self._get_attribute_ranges()
        
    #     for fingerprint in self.graph_fingerprints:
    #         # Make copies that require gradients
    #         x_tensor = fingerprint.x.clone().detach().requires_grad_(True)
    #         adj_dense = to_dense_adj(fingerprint.edge_index, 
    #                             max_num_nodes=self.fingerprint_nodes)[0]
    #         adj_tensor = adj_dense.clone().detach().requires_grad_(True)
            
    #         # Compute loss for this fingerprint
    #         loss = self._compute_single_fingerprint_loss(x_tensor, adj_tensor)
            
    #         # Compute gradients
    #         if x_tensor.grad is not None:
    #             x_tensor.grad.zero_()
    #         if adj_tensor.grad is not None:
    #             adj_tensor.grad.zero_()
            
    #         loss.backward()
            
    #         # Update node attributes with gradient and clipping
    #         if x_tensor.grad is not None:
    #             new_x = x_tensor + alpha * x_tensor.grad
    #             fingerprint.x = self._clip_attributes(new_x.detach(), attribute_ranges)
            
    #         # Update adjacency matrix using paper's discrete method
    #         if adj_tensor.grad is not None:
    #             self._update_adjacency_discrete(fingerprint, adj_tensor, alpha, top_k_edges)

    # def _update_univerifier_single_epoch(self, beta=0.001):
    #     """Update Univerifier for one epoch"""
    #     optimizer = optim.Adam(self.univerifier.parameters(), lr=beta)
        
    #     # Compute loss for all models
    #     L = 0
    #     all_models = [self.target_gnn] + self.positive_gnns + self.negative_gnns
        
    #     for model_idx, model in enumerate(all_models):
    #         # Get concatenated outputs from all fingerprints
    #         fingerprint_outputs = []
    #         for fingerprint in self.graph_fingerprints:
    #             model.eval()
    #             with torch.no_grad():
    #                 output = model(fingerprint.x.to(self.device), 
    #                             fingerprint.edge_index.to(self.device))
    #                 fingerprint_outputs.append(output)
            
    #         concatenated = torch.cat([out.view(-1) for out in fingerprint_outputs]).unsqueeze(0)
            
    #         # Univerifier prediction
    #         univerifier_out = self.univerifier(concatenated)
    #         o_plus = univerifier_out[0, 1]
            
    #         # Accumulate loss
    #         if model_idx < len(self.positive_gnns) + 1:  # Target or positive
    #             L += torch.log(o_plus + 1e-10)
    #         else:  # Negative
    #             L += torch.log(1 - o_plus + 1e-10)
        
    #     # Optimization step
    #     optimizer.zero_grad()
    #     (-L).backward()  # Minimize negative log likelihood
    #     optimizer.step()

    # def _update_adjacency_discrete(self, fingerprint, adj_tensor, alpha, top_k_edges):
    #     """Update adjacency matrix using paper's discrete optimization rules"""
    #     adj_grad = adj_tensor.grad
        
    #     if adj_grad is None:
    #         return
        
    #     # Get top-K edges with largest absolute gradient values
    #     flat_grad = adj_grad.view(-1)
    #     flat_abs_grad = torch.abs(flat_grad)
    #     top_values, top_indices = torch.topk(flat_abs_grad, min(top_k_edges, flat_abs_grad.numel()))
        
    #     current_adj = adj_tensor.detach().clone()
    #     current_adj.requires_grad_(False)
        
    #     for idx in top_indices:
    #         if top_values[idx] < 1e-8:  # Skip very small gradients
    #             continue
                
    #         # Convert flat index to (u, v) coordinates
    #         u = idx // self.fingerprint_nodes
    #         v = idx % self.fingerprint_nodes
            
    #         if u >= self.fingerprint_nodes or v >= self.fingerprint_nodes:
    #             continue
            
    #         grad_value = adj_grad[u, v].item()
    #         current_value = current_adj[u, v].item()
            
    #         # Apply paper's rules:
    #         # 1. If edge exists and gradient <= 0: remove edge
    #         # 2. If edge doesn't exist and gradient >= 0: add edge
    #         if current_value > 0.5:  # Edge exists
    #             if grad_value <= 0:
    #                 current_adj[u, v] = 0
    #                 current_adj[v, u] = 0  # Undirected graph
    #         else:  # Edge doesn't exist
    #             if grad_value >= 0:
    #                 current_adj[u, v] = 1
    #                 current_adj[v, u] = 1  # Undirected graph
        
    #     # Convert back to sparse and update fingerprint
    #     new_edge_index = dense_to_sparse(current_adj)[0]
    #     fingerprint.edge_index = new_edge_index

    # def _compute_single_fingerprint_loss(self, x_tensor, adj_tensor):
    #     """Compute loss contribution from a single fingerprint"""
    #     edge_index_sparse = dense_to_sparse(adj_tensor)[0]
    #     loss = 0
        
    #     # All models
    #     all_models = [self.target_gnn] + self.positive_gnns + self.negative_gnns
        
    #     for model_idx, model in enumerate(all_models):
    #         model_out = model(x_tensor, edge_index_sparse)
    #         concat_out = model_out.view(1, -1)
    #         univerifier_out = self.univerifier(concat_out)
    #         o_plus = univerifier_out[0, 1]
            
    #         if model_idx < len(self.positive_gnns) + 1:  # Target or positive
    #             loss += torch.log(o_plus + 1e-10)
    #         else:  # Negative
    #             loss += torch.log(1 - o_plus + 1e-10)
        
    #     return loss


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
    



