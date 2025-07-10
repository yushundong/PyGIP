"""
Grove Model Stealing Attack implementation.
Includes all advanced attack components: graph reconstruction, model pruning, and attack variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
from typing import Dict, Any, Optional, Tuple
import copy
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from models.attack.base import BaseAttack
from models.nn.grove_models import GATModel, GINModel, GraphSAGEModel, BaseGNNModel
from datasets import Dataset


class SurrogateEmbeddingModel(nn.Module):
    """
    Surrogate model that outputs embeddings to match target model embeddings.
    """
    
    def __init__(self, base_model: nn.Module, output_dim: int):
        """
        Initialize surrogate embedding model.
        
        Args:
            base_model: Base GNN model (GATModel, GINModel, or GraphSAGEModel)
            output_dim: Dimension of target embeddings to match
        """
        super(SurrogateEmbeddingModel, self).__init__()
        self.base_model = base_model
        self.output_dim = output_dim
        
        # Create a dedicated embedding projection layer
        # Get the output dimension from the base model
        if hasattr(base_model, 'h_feats'):
            # For Grove models with h_feats attribute
            base_output_dim = base_model.h_feats
        elif hasattr(base_model, 'out_channels'):
            # For other models
            base_output_dim = base_model.out_channels
        else:
            # Default fallback
            base_output_dim = 128
            
        self.embedding_projection = nn.Linear(base_output_dim, output_dim)
        
        # Initialize the projection layer
        nn.init.xavier_uniform_(self.embedding_projection.weight)
        nn.init.zeros_(self.embedding_projection.bias)
        
    def forward(self, data):
        """
        Forward pass returning embeddings.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            embeddings: Node embeddings matching target dimension
        """
        # Get embeddings from base model
        # Models return (embeddings, predictions)
        if hasattr(self.base_model, 'forward') and isinstance(self.base_model, BaseGNNModel):
            embeddings, _ = self.base_model(data)
        else:
            # Fallback for other models
            embeddings = self.base_model(data)
        
        # Project to target dimension
        target_embeddings = self.embedding_projection(embeddings)
        
        return target_embeddings
    
    def get_embedding_dimension(self):
        """
        Get the output embedding dimension.
        
        Returns:
            Output embedding dimension
        """
        return self.output_dim


class GroveAttack(BaseAttack):
    """
    Stealing Attack implementation.
    
    Supports Type I (original structure) and Type II (reconstructed structure) attacks
    with various attack sophistication levels.
    """
    
    def __init__(self, 
                 dataset: Dataset, 
                 attack_node_fraction: float, 
                 model_path: str = None,
                 attack_type: str = "type_i",
                 surrogate_architecture: str = "gat",
                 recovery_from: str = "embedding",
                 structure: str = "original",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 hidden_dim: int = 256,
                 num_epochs: int = 200,
                 learning_rate: float = 0.001,
                 pruning_ratio: float = 0.1):
        """
        Initialize Grove attack.
        
        Args:
            dataset: PyGIP Dataset object
            attack_node_fraction: Fraction of nodes to use for attack
            model_path: Path to pre-trained target model
            attack_type: Type of attack ("type_i" or "type_ii")
            surrogate_architecture: Architecture for surrogate model ("gat", "graphsage", "gin")
            recovery_from: What to recover from target ("embedding", "prediction")
            structure: Graph structure to use ("original", "idgl", "knn", "random")
            device: Device to run on
            hidden_dim: Hidden dimension for models
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            pruning_ratio: Ratio of parameters to prune in advanced attacks
        """
        # Model storage
        self.target_model = None
        self.surrogate_model = None
        self.classifier = None
        
        # Training data splits
        self.query_nodes = None
        self.val_nodes = None
        self.test_nodes = None
        
        # Store dataset and initialize required attributes BEFORE parent init
        self.dataset = dataset
        self.graph = getattr(dataset, 'graph', None)
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Ensure we have edge_index for PyTorch Geometric operations BEFORE parent init
        self._ensure_pyg_format()
        
        self.attack_type = attack_type
        self.surrogate_architecture = surrogate_architecture.lower()
        self.recovery_from = recovery_from.lower()
        self.structure = structure.lower()
        self.pruning_ratio = pruning_ratio
        
        # Prepare attack data splits BEFORE parent init
        self._prepare_attack_data()
        
        # Initialize base class
        super().__init__(dataset, attack_node_fraction, model_path)
        
    def _ensure_pyg_format(self):
        """
        Ensure the dataset has PyTorch Geometric format attributes.
        """
        # Check if dataset has edge_index directly (PyTorch Geometric format)
        if hasattr(self.dataset, 'edge_index'):
            self.edge_index = self.dataset.edge_index
        elif self.graph is not None and hasattr(self.graph, 'edge_index'):
            # Already in PyTorch Geometric format
            self.edge_index = self.graph.edge_index
        elif self.graph is not None and hasattr(self.graph, 'edges'):
            # DGL graph - convert to edge_index
            edge_index = torch.stack(self.graph.edges())
            self.edge_index = edge_index
        else:
            # Fallback - create edge_index from available data
            print("Warning: No edge information found, creating empty edge_index")
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        
    def _prepare_attack_data(self):
        """
        Prepare data splits for the attack following Grove specification.
        
        Creates four disjoint sets:
        - Target Model Training Set (target_train): 40% of nodes
        - Surrogate Model Query Set (surrogate_train): 40% of nodes
        - Test Set (test): 10% of nodes
        - Verification Set (verification): 10% of nodes
        
        All sets are completely disjoint (non-overlapping).
        """
        # Get all available nodes (not just train_mask)
        total_nodes = torch.arange(self.dataset.node_number)
        
        # Calculate split sizes based on total nodes
        total_nodes_count = len(total_nodes)
        num_target_train = int(total_nodes_count * 0.4)    # 40% for target training
        num_surrogate_query = int(total_nodes_count * 0.4)  # 40% for surrogate queries
        num_test = int(total_nodes_count * 0.1)            # 10% for test
        num_verification = total_nodes_count - num_target_train - num_surrogate_query - num_test  # Remaining for verification
        
        # Ensure we have valid splits
        if num_target_train + num_surrogate_query + num_test + num_verification != total_nodes_count:
            print(f"Warning: Split sizes don't sum to total nodes, adjusting...")
            num_verification = total_nodes_count - num_target_train - num_surrogate_query - num_test
        
        # Random split with fixed seed for reproducibility
        torch.manual_seed(42)
        perm = torch.randperm(total_nodes_count)
        
        # Create disjoint splits
        self.target_train_nodes = total_nodes[perm[:num_target_train]]
        self.query_nodes = total_nodes[perm[num_target_train:num_target_train + num_surrogate_query]]
        self.test_nodes = total_nodes[perm[num_target_train + num_surrogate_query:num_target_train + num_surrogate_query + num_test]]
        self.verification_nodes = total_nodes[perm[num_target_train + num_surrogate_query + num_test:]]
        
        # For validation during surrogate training, use a subset of query nodes
        num_val = max(1, int(len(self.query_nodes) * 0.15))  # 15% of query nodes for validation
        val_perm = torch.randperm(len(self.query_nodes))
        self.val_nodes = self.query_nodes[val_perm[:num_val]]
        
        print(f"Grove attack data splits prepared:")
        print(f"  Target training nodes: {len(self.target_train_nodes)} (40%)")
        print(f"  Surrogate query nodes: {len(self.query_nodes)} (40%)")
        print(f"  Test nodes: {len(self.test_nodes)} (10%)")
        print(f"  Verification nodes: {len(self.verification_nodes)} (10%)")
        print(f"  Validation nodes (subset of query): {len(self.val_nodes)}")
        print(f"  Total nodes: {total_nodes_count}")
        
        # Validate splits are disjoint
        all_splits = torch.cat([self.target_train_nodes, self.query_nodes, self.test_nodes, self.verification_nodes])
        if len(torch.unique(all_splits)) != len(all_splits):
            raise ValueError("Data splits are not disjoint!")
        
        # Validate we have enough nodes for meaningful splits
        if len(self.query_nodes) == 0:
            raise ValueError("No query nodes available for attack")
        if len(self.target_train_nodes) == 0:
            raise ValueError("No target training nodes available")
        if len(self.test_nodes) == 0:
            raise ValueError("No test nodes available")
        if len(self.verification_nodes) == 0:
            raise ValueError("No verification nodes available")
        
    def _load_model(self, model_path: str):
        """
        Load a pre-trained target model.
        
        Args:
            model_path: Path to the model file
        """
        print(f"Loading target model from: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create target model based on architecture
        if 'gat' in model_path.lower():
            self.target_model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
        elif 'sage' in model_path.lower() or 'graphsage' in model_path.lower():
            self.target_model = GraphSAGEModel(self.feature_number, self.hidden_dim, self.label_number)
        elif 'gin' in model_path.lower():
            self.target_model = GINModel(self.feature_number, self.hidden_dim, self.label_number)
        else:
            # Default to GAT
            self.target_model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.target_model.load_state_dict(checkpoint)
        
        self.target_model.to(self.device)
        self.target_model.eval()
        
    def _train_target_model(self):
        """
        Train the target model if not provided.
        Uses the target training nodes (40% of all nodes) as per Grove specification.
        """
        print("Training target model...")
        
        # Create target model
        self.target_model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
        self.target_model.to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.target_model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.target_model.train()
            optimizer.zero_grad()
            
            # Forward pass
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            
            # Models return (embeddings, predictions)
            embeddings, predictions = self.target_model(data)
            
            # Compute loss on target training nodes (40% of all nodes)
            loss = F.cross_entropy(predictions[self.target_train_nodes], self.labels[self.target_train_nodes])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                train_acc = (predictions[self.target_train_nodes].argmax(dim=1) == 
                           self.labels[self.target_train_nodes]).float().mean().item()
            
            if epoch % 20 == 0:
                print(f"TARGET TRAINING: Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")
        
        self.target_model.eval()
        print("Target model training completed.")
        
    def _train_attack_model(self):
        """
        Train the surrogate attack model.
        """
        print(f"Training surrogate model using {self.surrogate_architecture} architecture...")
        
        # Create surrogate model 
        if self.surrogate_architecture == "gat":
            base_model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
        elif self.surrogate_architecture == "graphsage":
            base_model = GraphSAGEModel(self.feature_number, self.hidden_dim, self.label_number)
        elif self.surrogate_architecture == "gin":
            base_model = GINModel(self.feature_number, self.hidden_dim, self.label_number)
        else:
            raise ValueError(f"Unsupported surrogate architecture: {self.surrogate_architecture}. Supported: gat, graphsage, gin")
        
        # Get target embedding dimension
        target_embedding_dim = self._get_target_embedding_dim()
        
        # Create surrogate embedding model
        self.surrogate_model = SurrogateEmbeddingModel(base_model, target_embedding_dim)
        self.surrogate_model.to(self.device)
        
        # ALWAYS create classifier 
        self.classifier = nn.Linear(target_embedding_dim, self.label_number).to(self.device)
        
        # Separate optimizers for surrogate and classifier
        surrogate_optimizer = optim.Adam(self.surrogate_model.parameters(), lr=self.learning_rate)
        classifier_optimizer = optim.SGD(self.classifier.parameters(), lr=0.01)
        
        # Loss functions
        embedding_loss_fn = nn.MSELoss()
        classification_loss_fn = nn.CrossEntropyLoss()
        
        # Prepare data with structure modification
        data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
        data = self._apply_structure_modification(data)
        
        # Query target model to get target response
        original_data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
        with torch.no_grad():
            target_embeddings, target_predictions = self._query_target_model(original_data)
        
        # Determine what to recover from target (print once before training)
        if self.recovery_from == "embedding":
            target_response = target_embeddings[self.query_nodes]
            print(f"Using embeddings as target response (dim: {target_response.shape[1]})")
        else:
            target_response = target_predictions[self.query_nodes]
            print(f"Using predictions as target response (dim: {target_response.shape[1]})")
        
        # Training loop 
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.surrogate_model.train()
            self.classifier.train()
            
            try:
                # Forward pass through surrogate model
                surrogate_embeddings = self.surrogate_model(data)
                
                # Different training strategies based on recovery method
                if self.recovery_from == "embedding":
                    # Embedding recovery: train surrogate to match target embeddings
                    embedding_loss = torch.sqrt(embedding_loss_fn(surrogate_embeddings[self.query_nodes], target_response))
                    
                    # Update surrogate model
                    surrogate_optimizer.zero_grad()
                    embedding_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.surrogate_model.parameters(), max_norm=1.0)
                    surrogate_optimizer.step()
                    
                    # Classification loss (separate from embedding loss, using detached embeddings)
                    with torch.no_grad():
                        surrogate_embeddings_detached = self.surrogate_model(data)
                    
                    logits = self.classifier(surrogate_embeddings_detached.detach())
                    classification_loss = classification_loss_fn(logits[self.query_nodes], self.labels[self.query_nodes].long())
                    
                    # Update classifier
                    classifier_optimizer.zero_grad()
                    classification_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                    classifier_optimizer.step()
                    
                    main_loss = embedding_loss
                    
                else:  # self.recovery_from == "prediction"
                    # Prediction recovery: train surrogate to match target predictions
                    # Get surrogate predictions
                    surrogate_predictions = self.classifier(surrogate_embeddings)
                    
                    # Prediction loss (KL divergence or cross-entropy with soft targets)
                    prediction_loss = F.kl_div(
                        F.log_softmax(surrogate_predictions[self.query_nodes], dim=1),
                        target_response,
                        reduction='batchmean'
                    )
                    
                    # Update both surrogate model and classifier together for prediction recovery
                    surrogate_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()
                    prediction_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.surrogate_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                    surrogate_optimizer.step()
                    classifier_optimizer.step()
                    
                    main_loss = prediction_loss
                    classification_loss = prediction_loss  # Same loss for both
                
                # Calculate accuracy
                with torch.no_grad():
                    if self.recovery_from == "prediction":
                        logits = surrogate_predictions
                    else:
                        logits = self.classifier(surrogate_embeddings.detach())
                    train_acc = (logits[self.query_nodes].argmax(dim=1) == self.labels[self.query_nodes]).float().mean().item()
                                               
                # Validation and logging 
                if epoch % 20 == 0:
                    val_acc = self._validate_surrogate_acc()
                    print(f'Epoch {epoch:05d} , Loss {main_loss.item():.4f} , Train Acc {train_acc:.4f} , Class Loss {classification_loss.item():.4f}')
                    
                    if epoch % 20 == 0 and epoch > 0:
                        print(f'Val Acc {val_acc:.4f}')
                    
                    # Use validation accuracy for early stopping
                    val_loss = 1.0 - val_acc # Assuming higher accuracy is better, so lower 1-acc is better
                    if val_loss < best_loss:
                        best_loss = val_loss
                        
            except Exception as e:
                print(f"WARNING: Training error at epoch {epoch}: {e}")
                # Continue training with dummy values
                continue
        
        print("Surrogate model training completed.")
        
    def _get_target_embedding_dim(self) -> int:
        """
        Get the embedding dimension of the target model.
        
        Returns:
            Target embedding dimension
        """
        # Query target model to get embedding dimension
        with torch.no_grad():
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            target_embeddings, _ = self._query_target_model(data)
            return target_embeddings.shape[1]
            
    def _query_target_model(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the target model to get embeddings and predictions.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Tuple of (embeddings, predictions)
        """
        self.target_model.eval()
        
        # Forward pass through target model
        with torch.no_grad():
            # Models return (embeddings, predictions)
            embeddings, predictions = self.target_model(data)
            
            # Apply softmax to predictions
            predictions = F.softmax(predictions, dim=1)
            
        return embeddings, predictions
        
    def _validate_surrogate(self) -> float:
        """
        Validate the surrogate model.
        
        Returns:
            Validation loss
        """
        if len(self.val_nodes) == 0:
            return float('inf')
            
        self.surrogate_model.eval()
        if self.classifier is not None:
            self.classifier.eval()
        
        with torch.no_grad():
            # Prepare data
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            data = self._apply_structure_modification(data)
            
            # Query target model
            target_embeddings, target_predictions = self._query_target_model(data)
            
            # Get surrogate outputs
            surrogate_embeddings = self.surrogate_model(data)
            
            # Compute validation loss based on recovery type
            if self.recovery_from == "embedding":
                val_loss = F.mse_loss(surrogate_embeddings[self.val_nodes], 
                                    target_embeddings[self.val_nodes])
            else: # self.recovery_from == "prediction"
                if self.classifier is not None:
                    surrogate_predictions = self.classifier(surrogate_embeddings)
                    # Use CrossEntropyLoss for prediction recovery, target is argmax of target_predictions
                    val_loss = F.cross_entropy(surrogate_predictions[self.val_nodes], 
                                             target_predictions[self.val_nodes].argmax(dim=1))
                else:
                    val_loss = float('inf')
            
            # Also compute classification accuracy for validation if prediction recovery
            if self.recovery_from == "prediction" and self.classifier is not None:
                # Re-run surrogate_predictions if it wasn't computed in the else block above, to ensure accurate val_acc
                # (although it should be)
                surrogate_predictions = self.classifier(surrogate_embeddings) 
                val_acc = (surrogate_predictions[self.val_nodes].argmax(dim=1) == 
                          self.labels[self.val_nodes]).float().mean().item()
                # Use classification accuracy as validation metric for early stopping
                val_loss = 1.0 - val_acc # Convert accuracy to a 'loss' for consistent early stopping logic
        
        return val_loss
    
    def _validate_surrogate_acc(self) -> float:
        """
        Validate the surrogate model and return accuracy.
        
        Returns:
            Validation accuracy
        """
        if len(self.val_nodes) == 0:
            return 0.0
            
        self.surrogate_model.eval()
        if self.classifier is not None:
            self.classifier.eval()
        
        with torch.no_grad():
            # Prepare data
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            data = self._apply_structure_modification(data)
            
            # Get surrogate outputs
            surrogate_embeddings = self.surrogate_model(data)
            
            # Compute classification accuracy for validation
            if self.classifier is not None:
                surrogate_predictions = self.classifier(surrogate_embeddings)
                val_acc = (surrogate_predictions[self.val_nodes].argmax(dim=1) == 
                          self.labels[self.val_nodes]).float().mean().item()
                return val_acc
        
        return 0.0
        
    def attack(self, attack_method: str = "simple") -> Dict[str, Any]:
        """
        Execute the model stealing attack.
        
        Args:
            attack_method: Type of attack ("simple", "fine_tuning", "double_extraction", "distribution_shift", "pruned")
            
        Returns:
            Dictionary containing attack results and metrics
        """
        print(f"Executing Grove {self.attack_type} attack using {attack_method} method...")
        
        # Execute attack based on method
        if attack_method == "simple":
            self._train_attack_model()
        elif attack_method == "fine_tuning":
            self._run_fine_tuning_attack()
        elif attack_method == "double_extraction":
            self._run_double_extraction_attack()
        elif attack_method == "distribution_shift":
            self._run_distribution_shift_attack()
        elif attack_method == "pruned":
            self._run_pruned_attack()
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        # Evaluate attack success
        results = self._evaluate_attack()
        
        # Add attack configuration to results
        results.update({
            'attack_type': self.attack_type,
            'surrogate_architecture': self.surrogate_architecture,
            'recovery_from': self.recovery_from,
            'attack_method': attack_method,
            'attack_node_fraction': self.attack_node_fraction,
            'num_epochs': self.num_epochs
        })
        
        return results
        
    def _evaluate_attack(self) -> Dict[str, Any]:
        """
        Evaluate the success of the attack.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating attack success...")
        
        # Prepare ORIGINAL data for evaluation (not modified structure)
        original_data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
        
        # Get target and surrogate outputs ON ORIGINAL STRUCTURE
        with torch.no_grad():
            # Target predictions on original structure
            target_embeddings, target_predictions = self._query_target_model(original_data)
            target_predictions = F.softmax(target_predictions, dim=1)
            
            # Surrogate embeddings on original structure (always compute)
            self.surrogate_model.eval()
            surrogate_embeddings = self.surrogate_model(original_data)
            
            # Surrogate predictions on original structure
            if self.classifier is not None:
                self.classifier.eval()
                surrogate_logits = self.classifier(surrogate_embeddings)
                surrogate_predictions = F.softmax(surrogate_logits, dim=1)
            else:
                # This should not happen anymore since we always train a classifier
                print("Warning: No classifier available - this should not happen")
                surrogate_predictions = target_predictions  # Fallback
        
        # Calculate metrics
        results = {}
        
        # Fidelity: Agreement between target and surrogate predictions ON ORIGINAL STRUCTURE
        if len(self.test_nodes) > 0:
            target_pred_labels = target_predictions[self.test_nodes].argmax(dim=1)
            surrogate_pred_labels = surrogate_predictions[self.test_nodes].argmax(dim=1)
            fidelity = (target_pred_labels == surrogate_pred_labels).float().mean().item()
            results['fidelity'] = fidelity
        
        # Accuracy: Performance on ground truth labels
        if len(self.test_nodes) > 0:
            target_accuracy = (target_predictions[self.test_nodes].argmax(dim=1) == 
                             self.labels[self.test_nodes]).float().mean().item()
            surrogate_accuracy = (surrogate_predictions[self.test_nodes].argmax(dim=1) == 
                                self.labels[self.test_nodes]).float().mean().item()
            
            results['target_accuracy'] = target_accuracy
            results['surrogate_accuracy'] = surrogate_accuracy
            results['accuracy_gap'] = abs(target_accuracy - surrogate_accuracy)
        
        # Embedding similarity (cosine similarity) - This is the main metric for embedding recovery
        if target_embeddings.shape[1] == surrogate_embeddings.shape[1]:
            target_emb_norm = F.normalize(target_embeddings, p=2, dim=1)
            surrogate_emb_norm = F.normalize(surrogate_embeddings, p=2, dim=1)
            cosine_sim = torch.sum(target_emb_norm * surrogate_emb_norm, dim=1).mean().item()
            results['embedding_cosine_similarity'] = cosine_sim
        
        # L2 distance between embeddings
        if target_embeddings.shape[1] == surrogate_embeddings.shape[1]:
            l2_distance = F.mse_loss(target_embeddings, surrogate_embeddings).item()
            results['embedding_l2_distance'] = l2_distance
        
        print(f"Attack evaluation completed:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def _run_fine_tuning_attack(self, fine_tune_ratio: float = 0.5, fine_tune_lr_factor: float = 0.1):
        """
        Run fine-tuning attack: Train initial surrogate, then fine-tune on subset.
        
        Args:
            fine_tune_ratio: Fraction of nodes to use for fine-tuning
            fine_tune_lr_factor: Learning rate reduction factor for fine-tuning
        """
        print("Running fine-tuning attack...")
        
        # Step 1: Train initial surrogate model
        print("Step 1: Training initial surrogate model...")
        self._train_attack_model()
        
        # Step 2: Fine-tune on subset of data
        print("Step 2: Fine-tuning on subset...")
        
        # Select subset for fine-tuning
        num_fine_tune = int(len(self.query_nodes) * fine_tune_ratio)
        fine_tune_indices = torch.randperm(len(self.query_nodes))[:num_fine_tune]
        fine_tune_nodes = self.query_nodes[fine_tune_indices]
        
        # Fine-tune with lower learning rate
        optimizer = optim.Adam(self.surrogate_model.parameters(), 
                             lr=self.learning_rate * fine_tune_lr_factor)
        
        for epoch in range(self.num_epochs // 4):  # Fewer epochs for fine-tuning
            self.surrogate_model.train()
            optimizer.zero_grad()
            
            # Prepare data
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            
            # Query target model
            with torch.no_grad():
                target_embeddings, target_predictions = self._query_target_model(data)
            
            # Get surrogate outputs
            surrogate_embeddings = self.surrogate_model(data)
            
            # Compute loss only on fine-tune nodes
            if self.recovery_from == "embedding":
                loss = F.mse_loss(surrogate_embeddings[fine_tune_nodes], 
                                target_embeddings[fine_tune_nodes])
            else:
                if self.classifier is not None:
                    surrogate_predictions = self.classifier(surrogate_embeddings)
                    loss = F.cross_entropy(surrogate_predictions[fine_tune_nodes], 
                                         target_predictions[fine_tune_nodes].argmax(dim=1))
                else:
                    break
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"FINE-TUNING: Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("Fine-tuning attack completed.")
    
    def _run_double_extraction_attack(self):
        """
        Run double extraction attack: Use first surrogate as target for second surrogate.
        """
        print("Running double extraction attack...")
        
        # Step 1: Train first surrogate model
        print("Step 1: Training first surrogate model...")
        self._train_attack_model()
        
        # Save first surrogate
        first_surrogate = self.surrogate_model
        first_classifier = getattr(self, 'classifier', None)
        
        # Step 2: Use first surrogate as new target
        print("Step 2: Training second surrogate using first as target...")
        
        # Create second surrogate
        target_embedding_dim = self._get_target_embedding_dim()
        
        if self.surrogate_architecture == "gat":
            base_model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
        elif self.surrogate_architecture == "graphsage":
            base_model = GraphSAGEModel(self.feature_number, self.hidden_dim, self.label_number)
        elif self.surrogate_architecture == "gin":
            base_model = GINModel(self.feature_number, self.hidden_dim, self.label_number)
        
        second_surrogate = SurrogateEmbeddingModel(base_model, target_embedding_dim)
        second_surrogate.to(self.device)
        
        # Train second surrogate using first as target
        optimizer = optim.Adam(second_surrogate.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.num_epochs):
            second_surrogate.train()
            first_surrogate.eval()
            optimizer.zero_grad()
            
            # Prepare data
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            
            # Get first surrogate outputs as target
            with torch.no_grad():
                first_embeddings = first_surrogate(data)
                if first_classifier is not None:
                    first_predictions = F.softmax(first_classifier(first_embeddings), dim=1)
                else:
                    first_predictions = first_embeddings  # Use embeddings as predictions
            
            # Get second surrogate outputs
            second_embeddings = second_surrogate(data)
            
            # Compute loss
            loss = F.mse_loss(second_embeddings[self.query_nodes], 
                            first_embeddings[self.query_nodes])
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"DOUBLE EXTRACTION: Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Replace surrogate with second surrogate
        self.surrogate_model = second_surrogate
        print("Double extraction attack completed.")
    
    def _run_distribution_shift_attack(self, shift_intensity: float = 0.3):
        """
        Run distribution shift attack: Apply distribution shift to features during training.
        
        Args:
            shift_intensity: Intensity of the distribution shift
        """
        print(f"Running distribution shift attack with intensity {shift_intensity}...")
        
        # Create surrogate model
        if self.surrogate_architecture == "gat":
            base_model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
        elif self.surrogate_architecture == "graphsage":
            base_model = GraphSAGEModel(self.feature_number, self.hidden_dim, self.label_number)
        elif self.surrogate_architecture == "gin":
            base_model = GINModel(self.feature_number, self.hidden_dim, self.label_number)
        
        target_embedding_dim = self._get_target_embedding_dim()
        self.surrogate_model = SurrogateEmbeddingModel(base_model, target_embedding_dim)
        self.surrogate_model.to(self.device)
        
        optimizer = optim.Adam(self.surrogate_model.parameters(), lr=self.learning_rate)

        # Always create a classifier for evaluation, even if not used in training
        if not hasattr(self, 'classifier') or self.classifier is None:
            self.classifier = nn.Linear(target_embedding_dim, self.label_number).to(self.device)
        
        # Add classifier to optimizer only if we're training with prediction recovery
        if self.recovery_from == "prediction":
            optimizer.add_param_group({'params': self.classifier.parameters()})

        # Training loop with distribution shift
        for epoch in range(self.num_epochs):
            self.surrogate_model.train()
            optimizer.zero_grad()
            
            # Ensure features require gradients for distribution shift
            features_with_grad = self.features.clone().detach().requires_grad_(True)

            # Apply distribution shift to features
            shifted_features = self._apply_distribution_shift(features_with_grad, shift_intensity)
            
            # Prepare data with shifted features
            data = Data(x=shifted_features, edge_index=self.edge_index).to(self.device)
            original_data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            
            # Query target model on original data
            with torch.no_grad():
                target_embeddings, target_predictions = self._query_target_model(original_data)
            
            # Get surrogate outputs on shifted data
            surrogate_embeddings = self.surrogate_model(data)
            
            # Compute loss
            if self.recovery_from == "embedding":
                loss = F.mse_loss(surrogate_embeddings[self.query_nodes], 
                                target_embeddings[self.query_nodes])
            else:
                surrogate_predictions = self.classifier(surrogate_embeddings)
                loss = F.cross_entropy(surrogate_predictions[self.query_nodes], 
                                     target_predictions[self.query_nodes].argmax(dim=1))
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"DISTRIBUTION SHIFT: Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # If we were doing embedding recovery, now train a classifier for evaluation
        if self.recovery_from == "embedding":
            print("Training classifier for evaluation...")
            classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
            
            for epoch in range(self.num_epochs // 4):  # Fewer epochs for classifier training
                self.surrogate_model.eval()
                self.classifier.train()
                classifier_optimizer.zero_grad()
                
                # Prepare data
                data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
                
                # Query target model for labels
                with torch.no_grad():
                    _, target_predictions = self._query_target_model(data)
                    surrogate_embeddings = self.surrogate_model(data)
                
                # Train classifier on surrogate embeddings
                surrogate_predictions = self.classifier(surrogate_embeddings)
                loss = F.cross_entropy(surrogate_predictions[self.query_nodes], 
                                     target_predictions[self.query_nodes].argmax(dim=1))
                
                loss.backward()
                classifier_optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"CLASSIFIER TRAINING: Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("Distribution shift attack completed.")
    
    def _apply_distribution_shift(self, features: torch.Tensor, shift_intensity: float) -> torch.Tensor:
        """
        Apply distribution shift to features.
        
        Args:
            features: Original features
            shift_intensity: Intensity of the shift
            
        Returns:
            Shifted features
        """
        # Detach statistics to prevent gradient flow through them if features itself is not meant to be differentiable.
        features_std = features.std().detach()
        features_mean = features.mean(dim=0).detach()

        # Add noise to features
        noise = torch.randn_like(features) * shift_intensity * features_std
        shifted_features = features + noise
        
        # Scale features
        scale_factor = 1.0 + shift_intensity * (torch.rand(1) - 0.5)
        shifted_features = shifted_features * scale_factor
        
        # Add bias
        bias = torch.randn(features.shape[1]) * shift_intensity * features_mean
        shifted_features = shifted_features + bias.to(features.device)
        
        return shifted_features
    
    def _run_pruned_attack(self, pruning_type: str = "magnitude"):
        """
        Run pruned attack: Train surrogate then apply model pruning.
        
        Args:
            pruning_type: Type of pruning ("random", "magnitude")
        """
        print(f"Running pruned attack with {pruning_type} pruning...")
        
        # Step 1: Train initial surrogate model
        print("Step 1: Training initial surrogate model...")
        self._train_attack_model()
        
        # Step 2: Apply pruning to surrogate model
        print(f"Step 2: Applying {pruning_type} pruning...")
        
        if pruning_type == "random":
            self.surrogate_model = self._random_prune_model(self.surrogate_model)
        elif pruning_type == "magnitude":
            self.surrogate_model = self._magnitude_prune_model(self.surrogate_model)
        else:
            print(f"Unknown pruning type: {pruning_type}, using magnitude")
            self.surrogate_model = self._magnitude_prune_model(self.surrogate_model)
        
        print("Pruned attack completed.")
    
    # =====================================================================
    # Private Helper Methods for Other Grove Attack Components
    # =====================================================================
    
    def _random_prune_model(self, model: nn.Module) -> nn.Module:
        """
        Apply random pruning to a model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model (copy)
        """
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning to parameters 
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:  # Only prune trainable parameters
                bitmask = torch.rand_like(param) > self.pruning_ratio
                with torch.no_grad():
                    param.copy_(torch.mul(param, bitmask.float()))
        
        return pruned_model
    
    def _magnitude_prune_model(self, model: nn.Module) -> nn.Module:
        """
        Apply magnitude-based pruning (remove smallest weights).
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model (copy)
        """
        pruned_model = copy.deepcopy(model)
        
        # Collect all parameters and their magnitudes
        all_params = []
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:
                all_params.append(param.abs().flatten())
        
        # Calculate global threshold
        if all_params:
            all_weights = torch.cat(all_params)
            threshold = torch.quantile(all_weights, self.pruning_ratio)
            
            # Apply pruning based on threshold
            for name, param in pruned_model.named_parameters():
                if param.requires_grad:
                    with torch.no_grad():
                        mask = param.abs() > threshold
                        param.copy_(torch.mul(param, mask.float()))
        
        return pruned_model
    
    def _knn_graph_reconstruction(self, data: Data, k: int = 10, metric: str = 'cosine') -> Data:
        """
        Reconstruct graph using K-nearest neighbors.
        
        Args:
            data: Original data with features
            k: Number of nearest neighbors
            metric: Distance metric
            
        Returns:
            Data with KNN-based graph structure
        """
        features = data.x.cpu().numpy()
        
        # Build KNN graph
        knn_graph = kneighbors_graph(
            features, 
            n_neighbors=k, 
            metric=metric,
            mode='connectivity',
            include_self=False
        )
        
        # Make symmetric
        knn_graph = (knn_graph + knn_graph.T) / 2
        knn_graph.data = np.ones_like(knn_graph.data)
        
        # Convert to edge_index
        coo = knn_graph.tocoo()
        edge_index = torch.stack([
            torch.from_numpy(coo.row),
            torch.from_numpy(coo.col)
        ], dim=0).long()
        
        # Create new data object
        reconstructed_data = Data(
            x=data.x,
            edge_index=edge_index.to(data.x.device),
            y=data.y if hasattr(data, 'y') else None
        )
        
        return reconstructed_data
    
    def _random_graph_reconstruction(self, data: Data, edge_prob: float = 0.1, preserve_degree: bool = True) -> Data:
        """
        Reconstruct graph with random structure.
        
        Args:
            data: Original data
            edge_prob: Probability of edge existence
            preserve_degree: Whether to preserve original degree distribution
            
        Returns:
            Data with random graph structure
        """
        num_nodes = data.x.shape[0]
        
        if preserve_degree and hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
            # Preserve original number of edges
            original_num_edges = data.edge_index.shape[1]
            
            # Sample random edges
            all_possible_edges = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    all_possible_edges.append([i, j])
            
            # Randomly select edges
            if len(all_possible_edges) > 0:
                selected_indices = np.random.choice(
                    len(all_possible_edges), 
                    size=min(original_num_edges // 2, len(all_possible_edges)), 
                    replace=False
                )
                
                selected_edges = [all_possible_edges[i] for i in selected_indices]
                
                # Create edge_index (undirected)
                edge_list = []
                for edge in selected_edges:
                    edge_list.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
                
                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # Use edge probability
            adj_matrix = torch.rand(num_nodes, num_nodes) < edge_prob
            adj_matrix = adj_matrix.triu(diagonal=1)  # Upper triangular
            adj_matrix = adj_matrix + adj_matrix.t()  # Make symmetric
            
            edge_index, _ = dense_to_sparse(adj_matrix.float())
        
        # Create new data object
        reconstructed_data = Data(
            x=data.x,
            edge_index=edge_index.to(data.x.device),
            y=data.y if hasattr(data, 'y') else None
        )
        
        return reconstructed_data
    
    def _idgl_graph_reconstruction(self, data: Data, 
                                  epsilon: float = 0.65,
                                  num_pers: int = 8,
                                  max_iter: int = 10) -> Data:
        """
        IDGL-based graph reconstruction (simplified version).
        
        Args:
            data: Original data with features
            epsilon: Threshold for edge connections
            num_pers: Number of perspectives
            max_iter: Maximum iterations
            
        Returns:
            Data with IDGL-reconstructed graph structure
        """
        features = data.x
        num_nodes = features.shape[0]
        
        # Multi-perspective weighted cosine similarity
        weight_tensor = torch.randn(num_pers, features.shape[1], device=features.device)
        weight_tensor = F.normalize(weight_tensor, p=2, dim=1)
        
        # Apply multi-perspective transformation
        expand_weight_tensor = weight_tensor.unsqueeze(1)  # [num_pers, 1, input_dim]
        context_fc = features.unsqueeze(0) * expand_weight_tensor  # [num_pers, num_nodes, input_dim]
        
        # Normalize features for each perspective
        context_norm = F.normalize(context_fc, p=2, dim=-1)  # [num_pers, num_nodes, input_dim]
        
        # Compute similarity matrix for each perspective and average
        similarity = torch.matmul(context_norm, context_norm.transpose(-1, -2))  # [num_pers, num_nodes, num_nodes]
        similarity = similarity.mean(0)  # Average over perspectives: [num_nodes, num_nodes]
        
        # Apply threshold
        adj = torch.where(similarity > epsilon, similarity, torch.zeros_like(similarity))
        
        # Make symmetric
        adj = (adj + adj.t()) / 2
        
        # Convert to edge_index format
        edge_index, _ = dense_to_sparse(adj)
        
        # Create new data object with reconstructed structure
        reconstructed_data = Data(
            x=data.x,
            edge_index=edge_index.to(data.x.device),
            y=data.y if hasattr(data, 'y') else None
        )
        
        return reconstructed_data
    
    def _apply_structure_modification(self, data: Data) -> Data:
        """
        Apply structure modification based on attack type and structure parameter.
        
        Args:
            data: Original data
            
        Returns:
            Data with modified structure
        """
        if self.attack_type == "type_i" or self.structure == "original":
            # Type I attack: use original structure
            return data
        else:
            # Type II attack: reconstruct structure
            print(f"Reconstructing graph structure using {self.structure} method...")
            
            if self.structure == 'idgl':
                return self._idgl_graph_reconstruction(data)
            elif self.structure == 'knn':
                return self._knn_graph_reconstruction(data, k=10)
            elif self.structure == 'random':
                return self._random_graph_reconstruction(data, edge_prob=0.1)
            else:
                print(f"Unknown reconstruction method: {self.structure}, using original")
                return data
    
    def _compute_embedding_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor, 
                                    metric: str = 'cosine') -> float:
        """
        Compute similarity between two embedding sets.
        
        Args:
            emb1: First embedding set
            emb2: Second embedding set
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            sim = F.cosine_similarity(emb1, emb2, dim=1).mean().item()
        elif metric == 'euclidean':
            sim = -torch.norm(emb1 - emb2, dim=1).mean().item()
        elif metric == 'manhattan':
            sim = -torch.norm(emb1 - emb2, p=1, dim=1).mean().item()
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return sim
    
    def _compute_prediction_fidelity(self, pred1: torch.Tensor, pred2: torch.Tensor) -> float:
        """
        Compute prediction fidelity between two models.
        
        Args:
            pred1: Predictions from first model
            pred2: Predictions from second model
            
        Returns:
            Fidelity score (fraction of agreement)
        """
        # Convert to predicted classes
        class1 = pred1.argmax(dim=1)
        class2 = pred2.argmax(dim=1)
        
        # Compute agreement
        fidelity = (class1 == class2).float().mean().item()
        
        return fidelity
    
    def _tsne_projection(self, embeddings: torch.Tensor, 
                        perplexity: float = 30.0, 
                        n_components: int = 2) -> torch.Tensor:
        """
        Apply t-SNE projection to embeddings.
        
        Args:
            embeddings: Input embeddings
            perplexity: t-SNE perplexity parameter
            n_components: Number of components for t-SNE
            
        Returns:
            Projected embeddings
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        projected_embeddings = tsne.fit_transform(embeddings_np)
        
        return torch.tensor(projected_embeddings, dtype=embeddings.dtype, device=embeddings.device)
    
    def _pca_projection(self, embeddings: torch.Tensor, 
                       n_components: Optional[int] = None) -> torch.Tensor:
        """
        Apply PCA projection to embeddings.
        
        Args:
            embeddings: Input embeddings
            n_components: Number of PCA components
            
        Returns:
            Projected embeddings
        """
        if n_components is None:
            n_components = min(embeddings.shape[0], embeddings.shape[1], 50)
        
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        projected_embeddings = pca.fit_transform(embeddings_np)
        
        return torch.tensor(projected_embeddings, dtype=embeddings.dtype, device=embeddings.device)
    
    def _gaussian_noise_shift(self, data: torch.Tensor, intensity: float = 0.3) -> torch.Tensor:
        """
        Apply Gaussian noise to create distribution shift.
        
        Args:
            data: Input data tensor
            intensity: Intensity of distribution shift
            
        Returns:
            Data with Gaussian noise applied
        """
        noise = torch.randn_like(data) * intensity * data.std()
        return data + noise
    
    def _feature_dropout_shift(self, data: torch.Tensor, intensity: float = 0.3) -> torch.Tensor:
        """
        Apply feature dropout to create distribution shift.
        
        Args:
            data: Input data tensor
            intensity: Intensity of shift
            
        Returns:
            Data with random features dropped
        """
        mask = torch.rand_like(data) > intensity
        return data * mask.float() 