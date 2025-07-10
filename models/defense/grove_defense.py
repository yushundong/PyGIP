"""
Grove Ownership Verification Defense implementation.
Based on Grove's CSim ownership verification system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from typing import Dict, Any, Tuple, List
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import joblib

from models.defense.base import BaseDefense
from models.nn.grove_models import GATModel, GINModel, GraphSAGEModel, BaseGNNModel
from datasets import Dataset


class SimilarityModel:
    """
    CSim - Similarity model for verifying if a suspect model is a surrogate or independent
    using embedding similarity analysis.
    """
    
    def __init__(self, 
                 target_model_name: str,
                 device: str = 'cuda',
                 random_state: int = 42):
        """
        Initialize CSim similarity model.
        
        Args:
            target_model_name: Name/ID of the target model this CSim is for
            device: Device to use for computations
            random_state: Random state for reproducibility
        """
        self.target_model_name = target_model_name
        self.device = device
        self.random_state = random_state
        self.classifier = None
        self.is_trained = False
        
    def _compute_distance_vector(self, 
                                embedding1: torch.Tensor, 
                                embedding2: torch.Tensor) -> np.ndarray:
        """
        Compute element-wise squared distance vector between two embeddings.
        
        Args:
            embedding1: First embedding [num_nodes, embedding_dim]
            embedding2: Second embedding [num_nodes, embedding_dim]
            
        Returns:
            Distance vectors [num_nodes, embedding_dim]
        """
        # Ensure embeddings are on CPU and converted to numpy
        emb1 = embedding1.detach().cpu().numpy() if torch.is_tensor(embedding1) else embedding1
        emb2 = embedding2.detach().cpu().numpy() if torch.is_tensor(embedding2) else embedding2
        
        # Compute element-wise squared distance
        distance_vector = (emb1 - emb2) ** 2
        
        return distance_vector
    
    def prepare_training_data(self,
                            target_embeddings: torch.Tensor,
                            surrogate_embeddings_list: List[torch.Tensor],
                            independent_embeddings_list: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for CSim using embeddings.
        
        Args:
            target_embeddings: Target model embeddings
            surrogate_embeddings_list: List of surrogate model embeddings
            independent_embeddings_list: List of independent model embeddings
            
        Returns:
            Tuple of (X_train, y_train) where X_train is distance vectors and y_train is labels
        """
        print("Preparing training data for CSim...")
        
        target_emb_np = target_embeddings.detach().cpu().numpy()
        
        distance_vectors = []
        labels = []
        
        # Positive samples: (target, surrogate) pairs
        print(f"Processing {len(surrogate_embeddings_list)} surrogate models for positive samples...")
        for i, surrogate_embeddings in enumerate(surrogate_embeddings_list):
            surrogate_emb_np = surrogate_embeddings.detach().cpu().numpy()
            
            # Ensure same shape
            if target_emb_np.shape != surrogate_emb_np.shape:
                min_nodes = min(target_emb_np.shape[0], surrogate_emb_np.shape[0])
                target_emb_crop = target_emb_np[:min_nodes]
                surrogate_emb_crop = surrogate_emb_np[:min_nodes]
            else:
                target_emb_crop = target_emb_np
                surrogate_emb_crop = surrogate_emb_np
            
            # Compute distance vectors
            dist_vectors = self._compute_distance_vector(target_emb_crop, surrogate_emb_crop)
            
            distance_vectors.extend(dist_vectors)
            labels.extend([1] * len(dist_vectors))  # 1 = similar (positive)
            
            print(f"  Surrogate {i+1}: {len(dist_vectors)} positive samples")
        
        # Negative samples: (target, independent) pairs
        print(f"Processing {len(independent_embeddings_list)} independent models for negative samples...")
        for i, independent_embeddings in enumerate(independent_embeddings_list):
            independent_emb_np = independent_embeddings.detach().cpu().numpy()
            
            # Ensure same shape
            if target_emb_np.shape != independent_emb_np.shape:
                min_nodes = min(target_emb_np.shape[0], independent_emb_np.shape[0])
                target_emb_crop = target_emb_np[:min_nodes]
                independent_emb_crop = independent_emb_np[:min_nodes]
            else:
                target_emb_crop = target_emb_np
                independent_emb_crop = independent_emb_np
            
            # Compute distance vectors
            dist_vectors = self._compute_distance_vector(target_emb_crop, independent_emb_crop)
            
            distance_vectors.extend(dist_vectors)
            labels.extend([0] * len(dist_vectors))  # 0 = not similar (negative)
            
            print(f"  Independent {i+1}: {len(dist_vectors)} negative samples")
        
        X_train = np.array(distance_vectors)
        y_train = np.array(labels)
        
        print(f"Training data prepared:")
        print(f"  Total samples: {len(X_train)}")
        print(f"  Positive samples (surrogate): {np.sum(y_train)}")
        print(f"  Negative samples (independent): {len(y_train) - np.sum(y_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        return X_train, y_train

    def train(self, X_train: np.ndarray, y_train: np.ndarray, use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train the CSim classifier.
        
        Args:
            X_train: Training features (distance vectors)
            y_train: Training labels (0=independent, 1=surrogate)
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Training results
        """
        print("Training CSim classifier...")
        
        if use_grid_search:
            # Grid search for best hyperparameters

            param_grid = {
                'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 128)],
                'activation': ['tanh', 'relu'],
                'random_state': [self.random_state],
                'max_iter': [1000],
                'early_stopping': [True],
                'validation_fraction': [0.1],
                'n_iter_no_change': [20]
            }
            grid_search = GridSearchCV(
                MLPClassifier(),
                param_grid,
                cv=10,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.classifier = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(128,),
                activation='relu',
                random_state=self.random_state,
                max_iter=1000
            )
            
            self.classifier.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Training accuracy
        train_accuracy = self.classifier.score(X_train, y_train)
        
        results = {
            'train_accuracy': train_accuracy,
            'num_samples': len(X_train),
            'num_features': X_train.shape[1],
            'positive_samples': np.sum(y_train),
            'negative_samples': len(y_train) - np.sum(y_train)
        }
        
        print(f"CSim training completed. Training accuracy: {train_accuracy:.4f}")
        
        return results

    def verify(self, target_embeddings: torch.Tensor, suspect_embeddings: torch.Tensor, 
              threshold: float = 0.5) -> Dict[str, Any]:
        """
        Verify if suspect model is a surrogate or independent.
        
        Args:
            target_embeddings: Target model embeddings
            suspect_embeddings: Suspect model embeddings
            threshold: Classification threshold
            
        Returns:
            Verification results
        """
        if not self.is_trained:
            raise ValueError("CSim model must be trained before verification")
        
        print("Verifying model ownership...")
        
        # Prepare embeddings
        target_emb_np = target_embeddings.detach().cpu().numpy()
        suspect_emb_np = suspect_embeddings.detach().cpu().numpy()
        
        # Ensure same shape
        if target_emb_np.shape != suspect_emb_np.shape:
            min_nodes = min(target_emb_np.shape[0], suspect_emb_np.shape[0])
            target_emb_crop = target_emb_np[:min_nodes]
            suspect_emb_crop = suspect_emb_np[:min_nodes]
        else:
            target_emb_crop = target_emb_np
            suspect_emb_crop = suspect_emb_np
        
        # Compute distance vectors
        distance_vectors = self._compute_distance_vector(target_emb_crop, suspect_emb_crop)
        
        # Predict
        predictions = self.classifier.predict(distance_vectors)
        probabilities = self.classifier.predict_proba(distance_vectors)
        
        # Aggregate results
        surrogate_probability = np.mean(probabilities[:, 1])  # Probability of being surrogate
        is_surrogate = surrogate_probability > threshold
        
        results = {
            'is_surrogate': is_surrogate,
            'surrogate_probability': surrogate_probability,
            'confidence': abs(surrogate_probability - 0.5) * 2,  # Confidence score
            'threshold': threshold,
            'num_nodes_verified': len(distance_vectors)
        }
        
        print(f"Verification completed:")
        print(f"  Is surrogate: {is_surrogate}")
        print(f"  Surrogate probability: {surrogate_probability:.4f}")
        print(f"  Confidence: {results['confidence']:.4f}")
        
        return results

    def save(self, save_path: str):
        """
        Save the CSim model.
        
        Args:
            save_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'target_model_name': self.target_model_name,
            'device': self.device,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, save_path)
        print(f"CSim model saved to: {save_path}")

    def load(self, load_path: str):
        """
        Load the CSim model.
        
        Args:
            load_path: Path to load the model from
        """
        model_data = joblib.load(load_path)
        
        self.classifier = model_data['classifier']
        self.target_model_name = model_data['target_model_name']
        self.device = model_data['device']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"CSim model loaded from: {load_path}")


class GroveDefense(BaseDefense):
    """
    Grove Ownership Verification Defense implementation extending PyGIP's BaseDefense.
    
    Uses CSim similarity model to detect if suspect models are surrogates (stolen) or independent.
    """
    
    def __init__(self, 
                 dataset: Dataset, 
                 attack_node_fraction: float,   
                 target_model_path: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 hidden_dim: int = 256,
                 num_epochs: int = 200,
                 learning_rate: float = 0.001,
                 verification_threshold: float = 0.5):
        """
        Initialize Grove defense.
        
        Args:
            dataset: PyGIP Dataset object
            attack_node_fraction: Fraction of nodes used in attacks
            target_model_path: Path to pre-trained target model
            device: Device to run on
            hidden_dim: Hidden dimension for models
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            verification_threshold: Threshold for ownership verification
        """
        # Store parameters
        self.target_model_path = target_model_path
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.verification_threshold = verification_threshold
        
        # Model storage
        self.target_model = None
        self.independent_models = []
        self.surrogate_models = []
        self.csim_model = None
        
        # Verification data
        self.verification_nodes = None
        
        # Ensure we have edge_index for PyTorch Geometric operations BEFORE parent init
        # We need to set dataset first for _ensure_pyg_format to work
        self.dataset = dataset
        self.graph = getattr(dataset, 'graph', None)
        self._ensure_pyg_format()
        
        # Prepare verification data BEFORE parent init
        self._prepare_verification_data()
        
        # Now call parent init
        super().__init__(dataset, attack_node_fraction)
        
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
        
    def _prepare_verification_data(self):
        """
        Prepare verification data split according to Grove specification.
        
        Creates four disjoint sets:
        - Target Model Training Set (target_train): 40% of nodes
        - Surrogate Model Query Set (surrogate_train): 40% of nodes
        - Test Set (test): 10% of nodes
        - Verification Set (verification): 10% of nodes
        
        All sets are completely disjoint (non-overlapping).
        """
        # Get all available nodes (not just test_mask)
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
        
        # Random split with fixed seed for reproducibility (same as attack)
        torch.manual_seed(42)
        perm = torch.randperm(total_nodes_count)
        
        # Create disjoint splits (same as attack implementation)
        self.target_train_nodes = total_nodes[perm[:num_target_train]]
        self.surrogate_query_nodes = total_nodes[perm[num_target_train:num_target_train + num_surrogate_query]]
        self.test_nodes = total_nodes[perm[num_target_train + num_surrogate_query:num_target_train + num_surrogate_query + num_test]]
        self.verification_nodes = total_nodes[perm[num_target_train + num_surrogate_query + num_test:]]
        
        print(f"Grove defense data splits prepared:")
        print(f"  Target training nodes: {len(self.target_train_nodes)} (40%)")
        print(f"  Surrogate query nodes: {len(self.surrogate_query_nodes)} (40%)")
        print(f"  Test nodes: {len(self.test_nodes)} (10%)")
        print(f"  Verification nodes: {len(self.verification_nodes)} (10%)")
        print(f"  Total nodes: {total_nodes_count}")
        
        # Validate splits are disjoint
        all_splits = torch.cat([self.target_train_nodes, self.surrogate_query_nodes, self.test_nodes, self.verification_nodes])
        if len(torch.unique(all_splits)) != len(all_splits):
            raise ValueError("Data splits are not disjoint!")
        
        # Validate we have enough nodes for meaningful splits
        if len(self.verification_nodes) == 0:
            raise ValueError("No verification nodes available")
        if len(self.target_train_nodes) == 0:
            raise ValueError("No target training nodes available")
        if len(self.test_nodes) == 0:
            raise ValueError("No test nodes available")
        if len(self.surrogate_query_nodes) == 0:
            raise ValueError("No surrogate query nodes available")
        
    def _load_model(self, model_path: str = None):
        """
        Load the target model.
        
        Args:
            model_path: Path to the target model
        """
        if model_path is None:
            model_path = self.target_model_path
            
        if model_path is None:
            print("No target model path provided, training new model...")
            self._train_target_model()
            return
            
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
        Train the target model.
        Uses the target training nodes (40% of all nodes) as per Grove specification.
        """
        print("Training target model...")
        
        # Create target model using Grove models
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
            if isinstance(self.target_model, BaseGNNModel):
                embeddings, predictions = self.target_model(data)
                out = predictions
            else:
                out = self.target_model(data)
            
            # Compute loss on target training nodes (40% of all nodes)
            loss = F.cross_entropy(out[self.target_train_nodes], self.labels[self.target_train_nodes])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.target_model.eval()
        print("Target model training completed.")
        
    def _train_defense_model(self):
        """
        Train the CSim defense model.
        """
        print("Training CSim defense model...")
        
        # Ensure we have target model
        if self.target_model is None:
            self._load_model()
        
        # Get target embeddings
        target_embeddings = self._get_model_embeddings(self.target_model)
        
        # Get independent model embeddings
        independent_embeddings_list = []
        for model in self.independent_models:
            embeddings = self._get_model_embeddings(model)
            independent_embeddings_list.append(embeddings)
        
        # Get surrogate model embeddings
        surrogate_embeddings_list = []
        for model in self.surrogate_models:
            embeddings = self._get_model_embeddings(model)
            surrogate_embeddings_list.append(embeddings)
        
        # Create and train CSim model
        self.csim_model = SimilarityModel(
            target_model_name="target_model",
            device=self.device
        )
        
        # Prepare training data
        X_train, y_train = self.csim_model.prepare_training_data(
            target_embeddings,
            surrogate_embeddings_list,
            independent_embeddings_list
        )
        
        # Train CSim
        training_results = self.csim_model.train(X_train, y_train, use_grid_search=True)
        
        return training_results
        
    def _train_surrogate_model(self):
        """
        Train surrogate models for testing defense.
        """
        print("Training surrogate models for defense testing...")
        
        # Train multiple surrogate models with different architectures
        architectures = ['gat', 'gin', 'graphsage']
        
        for arch in architectures:
            print(f"Training {arch} surrogate model...")
            
            # Create surrogate model using Grove models
            if arch == 'gat':
                model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
            elif arch == 'gin':
                model = GINModel(self.feature_number, self.hidden_dim, self.label_number)
            elif arch == 'graphsage':
                model = GraphSAGEModel(self.feature_number, self.hidden_dim, self.label_number)
            
            model.to(self.device)
            
            # Train with different random initialization
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            for epoch in range(self.num_epochs // 2):  # Train for fewer epochs
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
                
                # Models return (embeddings, predictions)
                if isinstance(model, BaseGNNModel):
                    embeddings, predictions = model(data)
                    out = predictions
                else:
                    out = model(data)
                
                # Compute loss on surrogate query nodes (40% of all nodes)
                loss = F.cross_entropy(out[self.surrogate_query_nodes], self.labels[self.surrogate_query_nodes])
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            model.eval()
            self.surrogate_models.append(model)
            
        print(f"Trained {len(self.surrogate_models)} surrogate models")
        
        # Also train independent models
        print("Training independent models...")
        
        for arch in architectures:
            print(f"Training {arch} independent model...")
            
            # Create independent models
            if arch == 'gat':
                model = GATModel(self.feature_number, self.hidden_dim, self.label_number)
            elif arch == 'gin':
                model = GINModel(self.feature_number, self.hidden_dim, self.label_number)
            elif arch == 'graphsage':
                model = GraphSAGEModel(self.feature_number, self.hidden_dim, self.label_number)
            
            model.to(self.device)
            
            # Train independently with different random seed
            torch.manual_seed(42 + len(self.independent_models))
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            for epoch in range(self.num_epochs // 2):
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
                
                # Models return (embeddings, predictions)
                if isinstance(model, BaseGNNModel):
                    embeddings, predictions = model(data)
                    out = predictions
                else:
                    out = model(data)
                
                # Compute loss on target training nodes (different from surrogate training)
                loss = F.cross_entropy(out[self.target_train_nodes], self.labels[self.target_train_nodes])
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            model.eval()
            self.independent_models.append(model)
            
        print(f"Trained {len(self.independent_models)} independent models")
        
    def _get_model_embeddings(self, model: nn.Module) -> torch.Tensor:
        """
        Get embeddings from a model.
        
        Args:
            model: The model to get embeddings from
            
        Returns:
            Model embeddings
        """
        model.eval()
        
        with torch.no_grad():
            data = Data(x=self.features, edge_index=self.edge_index).to(self.device)
            
            # Get embeddings (use verification nodes)
            # Check if model is SurrogateEmbeddingModel (expects Data object) or regular model (expects x, edge_index)
            if hasattr(model, 'base_model'):
                # SurrogateEmbeddingModel - pass Data object
                embeddings = model(data)
            elif isinstance(model, BaseGNNModel):
                # Models return (embeddings, predictions)
                embeddings, _ = model(data)
            else:
                # Regular model - pass x and edge_index
                embeddings = model(data)
            
            # Return embeddings for verification nodes
            return embeddings[self.verification_nodes]
        
    def defend(self, suspect_models: List[nn.Module] = None) -> Dict[str, Any]:
        """
        Execute the Grove ownership verification defense.
        
        Args:
            suspect_models: List of suspect models to verify
            
        Returns:
            Dictionary containing defense results
        """
        print("Executing Grove ownership verification defense...")
        
        # Load target model
        self._load_model()
        
        # Train surrogate and independent models if not provided
        if len(self.surrogate_models) == 0 or len(self.independent_models) == 0:
            self._train_surrogate_model()
        
        # Train CSim defense model
        training_results = self._train_defense_model()
        
        # Verify suspect models
        verification_results = []
        
        if suspect_models is None:
            # Test with our trained surrogate and independent models
            suspect_models = self.surrogate_models + self.independent_models
            expected_labels = [True] * len(self.surrogate_models) + [False] * len(self.independent_models)
        else:
            expected_labels = [None] * len(suspect_models)
        
        print(f"Verifying {len(suspect_models)} suspect models...")
        
        target_embeddings = self._get_model_embeddings(self.target_model)
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, suspect_model in enumerate(suspect_models):
            suspect_embeddings = self._get_model_embeddings(suspect_model)
            
            # Verify using CSim
            verification_result = self.csim_model.verify(
                target_embeddings, 
                suspect_embeddings, 
                threshold=self.verification_threshold
            )
            
            # Add expected label if available
            if expected_labels[i] is not None:
                verification_result['expected_surrogate'] = expected_labels[i]
                verification_result['correct_prediction'] = (
                    verification_result['is_surrogate'] == expected_labels[i]
                )
                
                if verification_result['correct_prediction']:
                    correct_predictions += 1
                total_predictions += 1
            
            verification_results.append(verification_result)
            
            print(f"  Model {i+1}: {'Surrogate' if verification_result['is_surrogate'] else 'Independent'} "
                  f"(prob: {verification_result['surrogate_probability']:.4f})")
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results = {
            'training_results': training_results,
            'verification_results': verification_results,
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'verification_threshold': self.verification_threshold,
            'num_surrogate_models': len(self.surrogate_models),
            'num_independent_models': len(self.independent_models)
        }
        
        print(f"Grove defense completed:")
        print(f"  Overall accuracy: {overall_accuracy:.4f}")
        print(f"  Correct predictions: {correct_predictions}/{total_predictions}")
        
        return results 