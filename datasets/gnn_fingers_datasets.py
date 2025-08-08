"""
Dataset extensions and utilities for GNNFingers framework.
"""

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import train_test_split_edges, to_undirected, remove_self_loops
import torch_geometric.transforms as T
from typing import List, Tuple, Dict, Optional
import random
import os

from .datasets import Dataset


class GNNFingersDatasetMixin:
    """Mixin class providing GNNFingers-specific dataset functionality."""
    
    def prepare_for_link_prediction(self):
        """Prepare dataset for link prediction tasks."""
        if hasattr(self, 'graph_data'):
            # Remove self-loops and make undirected
            self.graph_data.edge_index, _ = remove_self_loops(self.graph_data.edge_index)
            self.graph_data.edge_index = to_undirected(self.graph_data.edge_index)
            
            # Split edges for link prediction
            self.graph_data = train_test_split_edges(self.graph_data, val_ratio=0.1, test_ratio=0.2)
            
            print(f"Link prediction splits:")
            print(f"  Train edges: {self.graph_data.train_pos_edge_index.size(1)}")
            print(f"  Val edges: {self.graph_data.val_pos_edge_index.size(1)}")
            print(f"  Test edges: {self.graph_data.test_pos_edge_index.size(1)}")
    
    def create_graph_pairs(self, num_pairs: int = 500) -> List[Tuple]:
        """
        Create pairs of graphs for graph matching tasks.
        
        Args:
            num_pairs: Number of graph pairs to create
            
        Returns:
            List of (graph_pair, similarity_label) tuples
        """
        if not hasattr(self, 'graph_dataset') or self.graph_dataset is None:
            raise ValueError("Graph dataset not available for pair creation")
        
        print(f"Creating {num_pairs} graph pairs for matching...")
        
        pairs = []
        labels = []
        
        for i in range(num_pairs):
            idx1, idx2 = random.sample(range(len(self.graph_dataset)), 2)
            graph1 = self.graph_dataset[idx1]
            graph2 = self.graph_dataset[idx2]
            
            # Create similarity based on graph properties
            if hasattr(graph1, 'y') and hasattr(graph2, 'y'):
                # Same class = higher similarity
                if graph1.y.item() == graph2.y.item():
                    similarity = random.uniform(0.6, 1.0)
                else:
                    similarity = random.uniform(0.0, 0.4)
            else:
                # Random similarity
                similarity = random.uniform(0.0, 1.0)
            
            pairs.append((graph1, graph2))
            labels.append(similarity)
        
        return list(zip(pairs, labels))
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                      split: str = "train") -> DataLoader:
        """
        Get DataLoader for graph-level tasks.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            split: Which split to use ("train", "val", "test")
            
        Returns:
            DataLoader instance
        """
        if not hasattr(self, 'graph_dataset') or self.graph_dataset is None:
            raise ValueError("Graph dataset not available for DataLoader creation")
        
        # Simple split for demonstration
        total_size = len(self.graph_dataset)
        
        if split == "train":
            indices = list(range(int(0.7 * total_size)))
        elif split == "val":
            indices = list(range(int(0.7 * total_size), int(0.85 * total_size)))
        else:  # test
            indices = list(range(int(0.85 * total_size), total_size))
        
        subset = [self.graph_dataset[i] for i in indices]
        
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


class CoraGNNFingers(Dataset, GNNFingersDatasetMixin):
    """Cora dataset with GNNFingers extensions."""
    
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def get_name(self):
        return "Cora"
    
    def load_pyg_data(self):
        """Load Cora dataset for PyG."""
        print("Loading Cora dataset...")
        
        dataset = Planetoid(root=os.path.join(self.path, 'Cora'), 
                           name='Cora', transform=T.NormalizeFeatures())
        
        self.graph_dataset = dataset
        self.graph_data = dataset[0]
        
        # Set metadata
        self.num_nodes = self.graph_data.x.size(0)
        self.num_features = self.graph_data.x.size(1)
        self.num_classes = dataset.num_classes
        
        print(f"Cora dataset loaded: {self.num_nodes} nodes, {self.num_features} features, {self.num_classes} classes")
    
    def load_dgl_data(self):
        """Load Cora dataset for DGL."""
        raise NotImplementedError("DGL loading not implemented for GNNFingers datasets")


class CiteseerGNNFingers(Dataset, GNNFingersDatasetMixin):
    """Citeseer dataset with GNNFingers extensions."""
    
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def get_name(self):
        return "Citeseer"
    
    def load_pyg_data(self):
        """Load Citeseer dataset for PyG."""
        print("Loading Citeseer dataset...")
        
        dataset = Planetoid(root=os.path.join(self.path, 'Citeseer'), 
                           name='Citeseer', transform=T.NormalizeFeatures())
        
        self.graph_dataset = dataset
        self.graph_data = dataset[0]
        
        # Set metadata
        self.num_nodes = self.graph_data.x.size(0)
        self.num_features = self.graph_data.x.size(1)
        self.num_classes = dataset.num_classes
        
        print(f"Citeseer dataset loaded: {self.num_nodes} nodes, {self.num_features} features, {self.num_classes} classes")
    
    def load_dgl_data(self):
        """Load Citeseer dataset for DGL."""
        raise NotImplementedError("DGL loading not implemented for GNNFingers datasets")


class PubMedGNNFingers(Dataset, GNNFingersDatasetMixin):
    """PubMed dataset with GNNFingers extensions."""
    
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def get_name(self):
        return "PubMed"
    
    def load_pyg_data(self):
        """Load PubMed dataset for PyG."""
        print("Loading PubMed dataset...")
        
        dataset = Planetoid(root=os.path.join(self.path, 'PubMed'), 
                           name='PubMed', transform=T.NormalizeFeatures())
        
        self.graph_dataset = dataset
        self.graph_data = dataset[0]
        
        # Set metadata
        self.num_nodes = self.graph_data.x.size(0)
        self.num_features = self.graph_data.x.size(1)
        self.num_classes = dataset.num_classes
        
        print(f"PubMed dataset loaded: {self.num_nodes} nodes, {self.num_features} features, {self.num_classes} classes")
    
    def load_dgl_data(self):
        """Load PubMed dataset for DGL."""
        raise NotImplementedError("DGL loading not implemented for GNNFingers datasets")


class ProteinsGNNFingers(Dataset, GNNFingersDatasetMixin):
    """PROTEINS dataset with GNNFingers extensions."""
    
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def get_name(self):
        return "PROTEINS"
    
    def load_pyg_data(self):
        """Load PROTEINS dataset for PyG."""
        print("Loading PROTEINS dataset...")
        
        try:
            dataset = TUDataset(root=os.path.join(self.path, 'PROTEINS'), name='PROTEINS')
            print(f"SUCCESS: Real PROTEINS dataset loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"WARNING: PROTEINS dataset not available ({e}), creating synthetic protein graphs...")
            dataset = self._create_synthetic_protein_dataset()
        
        self.graph_dataset = dataset
        # Set a representative graph so base class can infer metadata
        if len(dataset) > 0:
            self.graph_data = dataset[0]
        
        # Check and add node features if missing
        if hasattr(dataset, 'num_node_features') and dataset.num_node_features == 0:
            print("Adding node features based on node degrees...")
            for data in dataset:
                if not hasattr(data, 'x') or data.x is None:
                    row, col = data.edge_index
                    deg = torch.zeros(data.num_nodes, dtype=torch.float)
                    deg = deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
                    data.x = deg.unsqueeze(1)
        
        # Set metadata
        self.num_nodes = 0  # Graph-level dataset
        self.num_features = getattr(dataset, 'num_node_features', dataset[0].x.size(1) if hasattr(dataset[0], 'x') else 1)
        self.num_classes = getattr(dataset, 'num_classes', len(set(data.y.item() for data in dataset if hasattr(data, 'y'))))
        
        print(f"PROTEINS dataset ready: {len(dataset)} graphs, {self.num_features} features, {self.num_classes} classes")
    
    def load_dgl_data(self):
        """Load PROTEINS dataset for DGL."""
        raise NotImplementedError("DGL loading not implemented for GNNFingers datasets")
    
    def _create_synthetic_protein_dataset(self):
        """Create high-quality synthetic protein-like dataset."""
        graphs = []
        num_graphs = 1113  # Match PROTEINS dataset size
        
        # Define amino acid properties (simplified)
        amino_acids = {
            'A': [0, 1, 0, 0],  # Alanine: small, hydrophobic
            'R': [1, 0, 1, 0],  # Arginine: large, charged, hydrophilic
            'N': [1, 0, 0, 1],  # Asparagine: medium, polar
            'D': [1, 0, 1, 1],  # Aspartic acid: medium, charged, hydrophilic
            'C': [0, 1, 0, 0],  # Cysteine: small, can form disulfide bonds
        }
        
        for i in range(num_graphs):
            # Protein-like sizes
            num_nodes = random.randint(15, 50)
            
            # Create amino acid sequence
            sequence = [random.choice(list(amino_acids.keys())) for _ in range(num_nodes)]
            x = torch.tensor([amino_acids[aa] for aa in sequence], dtype=torch.float)
            
            # Create realistic protein structure
            edge_list = []
            
            # Primary structure (backbone connections)
            for j in range(num_nodes - 1):
                edge_list.extend([[j, j+1], [j+1, j]])
            
            # Secondary structure (alpha helices, beta sheets)
            if num_nodes > 6:
                # Alpha helix pattern (i to i+4 connections)
                helix_start = random.randint(0, num_nodes//2)
                helix_length = min(random.randint(4, 8), num_nodes - helix_start - 4)
                for j in range(helix_start, helix_start + helix_length - 3):
                    if j + 3 < num_nodes:
                        edge_list.extend([[j, j+3], [j+3, j]])
            
            # Tertiary structure (disulfide bonds, hydrophobic interactions)
            num_tertiary = random.randint(1, min(5, num_nodes//6))
            for _ in range(num_tertiary):
                n1, n2 = random.sample(range(num_nodes), 2)
                if abs(n1 - n2) > 3:  # Non-local connections
                    edge_list.extend([[n1, n2], [n2, n1]])
            
            # Remove duplicates
            edge_set = set(tuple(sorted(edge)) for edge in edge_list)
            edge_list = [[min(e), max(e)] for e in edge_set] + [[max(e), min(e)] for e in edge_set]
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty((2, 0), dtype=torch.long)
            
            # Binary classification based on structural properties
            helix_ratio = locals().get('helix_length', 0) / num_nodes
            connectivity = len(edge_set) / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
            
            y = torch.tensor([1 if helix_ratio > 0.3 or connectivity > 0.15 else 0], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        
        class MockDataset:
            def __init__(self, data_list):
                self.data_list = data_list
                self.num_node_features = data_list[0].x.size(1)
                self.num_classes = len(set(data.y.item() for data in data_list))
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
        
        return MockDataset(graphs)


class AidsGNNFingers(Dataset, GNNFingersDatasetMixin):
    """AIDS dataset with GNNFingers extensions."""
    
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def get_name(self):
        return "AIDS"
    
    def load_pyg_data(self):
        """Load AIDS dataset for PyG."""
        print("Loading AIDS dataset...")
        
        try:
            dataset = TUDataset(root=os.path.join(self.path, 'AIDS'), name='AIDS')
            print(f"SUCCESS: Real AIDS dataset loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"WARNING: AIDS dataset not available ({e}), creating synthetic chemical graphs...")
            dataset = self._create_synthetic_chemical_dataset()
        
        self.graph_dataset = dataset
        # Set a representative graph so base class can infer metadata
        if len(dataset) > 0:
            self.graph_data = dataset[0]
        
        # Check and add node features if missing
        if hasattr(dataset, 'num_node_features') and dataset.num_node_features == 0:
            print("Adding node features based on atom types...")
            for data in dataset:
                if not hasattr(data, 'x') or data.x is None:
                    # Create realistic chemical atom features
                    num_atoms = 5  # C, N, O, S, P
                    atom_types = torch.randint(0, num_atoms, (data.num_nodes, 1), dtype=torch.float)
                    data.x = atom_types
        
        # Set metadata
        self.num_nodes = 0  # Graph-level dataset
        self.num_features = getattr(dataset, 'num_node_features', dataset[0].x.size(1) if hasattr(dataset[0], 'x') else 1)
        self.num_classes = getattr(dataset, 'num_classes', len(set(data.y.item() for data in dataset if hasattr(data, 'y'))))
        
        print(f"AIDS dataset ready: {len(dataset)} graphs, {self.num_features} features, {self.num_classes} classes")
    
    def load_dgl_data(self):
        """Load AIDS dataset for DGL."""
        raise NotImplementedError("DGL loading not implemented for GNNFingers datasets")
    
    def _create_synthetic_chemical_dataset(self):
        """Create high-quality synthetic chemical-like dataset."""
        graphs = []
        num_graphs = 2000  # Match AIDS dataset size
        
        for i in range(num_graphs):
            num_nodes = random.randint(8, 30)
            num_atom_types = 5  # C, N, O, S, P
            x = torch.randint(0, num_atom_types, (num_nodes, 1)).float()
            
            # Create realistic molecular structure
            edge_list = []
            
            # Backbone structure
            for j in range(num_nodes - 1):
                edge_list.extend([[j, j+1], [j+1, j]])
            
            # Add rings and side chains
            num_extra_edges = random.randint(num_nodes//4, num_nodes//2)
            for _ in range(num_extra_edges):
                n1, n2 = random.sample(range(num_nodes), 2)
                if abs(n1 - n2) > 1:  # Avoid too many adjacent connections
                    edge_list.extend([[n1, n2], [n2, n1]])
            
            # Remove duplicates
            edge_set = set(tuple(sorted(edge)) for edge in edge_list)
            edge_list = [[min(e), max(e)] for e in edge_set] + [[max(e), min(e)] for e in edge_set]
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty((2, 0), dtype=torch.long)
            
            # Binary classification (active vs inactive compounds)
            y = torch.tensor([random.randint(0, 1)], dtype=torch.long)
            
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        
        class MockDataset:
            def __init__(self, data_list):
                self.data_list = data_list
                self.num_node_features = data_list[0].x.size(1)
                self.num_classes = len(set(data.y.item() for data in data_list))
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
        
        return MockDataset(graphs)


class MutagGNNFingers(Dataset, GNNFingersDatasetMixin):
    """MUTAG dataset with GNNFingers extensions."""
    
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def get_name(self):
        return "MUTAG"
    
    def load_pyg_data(self):
        """Load MUTAG dataset for PyG."""
        print("Loading MUTAG dataset...")
        
        try:
            dataset = TUDataset(root=os.path.join(self.path, 'MUTAG'), name='MUTAG')
            print(f"SUCCESS: Real MUTAG dataset loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"WARNING: MUTAG dataset not available ({e}), creating synthetic molecular graphs...")
            dataset = self._create_synthetic_molecular_dataset()
        
        self.graph_dataset = dataset
        
        # Set metadata
        self.num_nodes = 0  # Graph-level dataset
        self.num_features = getattr(dataset, 'num_node_features', dataset[0].x.size(1) if hasattr(dataset[0], 'x') else 7)
        self.num_classes = getattr(dataset, 'num_classes', 2)  # MUTAG is binary
        
        print(f"MUTAG dataset ready: {len(dataset)} graphs, {self.num_features} features, {self.num_classes} classes")
    
    def load_dgl_data(self):
        """Load MUTAG dataset for DGL."""
        raise NotImplementedError("DGL loading not implemented for GNNFingers datasets")
    
    def _create_synthetic_molecular_dataset(self):
        """Create synthetic molecular dataset similar to MUTAG."""
        graphs = []
        num_graphs = 188  # Match MUTAG dataset size
        
        for i in range(num_graphs):
            num_nodes = random.randint(10, 28)  # MUTAG size range
            
            # MUTAG has 7-dimensional node features
            x = torch.randn(num_nodes, 7)
            
            # Create molecular graph structure
            edge_list = []
            
            # Create ring structures
            if num_nodes >= 6:
                ring_size = random.randint(5, 7)
                for j in range(ring_size):
                    edge_list.extend([[j, (j+1) % ring_size], [(j+1) % ring_size, j]])
            
            # Add side chains
            for j in range(ring_size if num_nodes >= 6 else 0, num_nodes - 1):
                edge_list.extend([[j, j+1], [j+1, j]])
            
            # Add some cross-connections
            num_cross = random.randint(0, min(3, num_nodes//5))
            for _ in range(num_cross):
                n1, n2 = random.sample(range(num_nodes), 2)
                if abs(n1 - n2) > 2:
                    edge_list.extend([[n1, n2], [n2, n1]])
            
            edge_set = set(tuple(edge) for edge in edge_list)
            edge_list = list(edge_set)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty((2, 0), dtype=torch.long)
            
            # Binary mutagenicity classification
            y = torch.tensor([random.randint(0, 1)], dtype=torch.long)
            
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        
        class MockDataset:
            def __init__(self, data_list):
                self.data_list = data_list
                self.num_node_features = 7
                self.num_classes = 2
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
        
        return MockDataset(graphs)


def get_gnnfingers_dataset(dataset_name: str, api_type: str = 'pyg', path: str = './data'):
    """
    Factory function to get GNNFingers-compatible datasets.
    
    Args:
        dataset_name: Name of the dataset
        api_type: API type ('pyg' or 'dgl')
        path: Path to store dataset files
        
    Returns:
        Dataset instance with GNNFingers extensions
    """
    dataset_name = dataset_name.upper()
    
    if dataset_name == "CORA":
        return CoraGNNFingers(api_type=api_type, path=path)
    elif dataset_name == "CITESEER":
        return CiteseerGNNFingers(api_type=api_type, path=path)
    elif dataset_name == "PUBMED":
        return PubMedGNNFingers(api_type=api_type, path=path)
    elif dataset_name == "PROTEINS":
        return ProteinsGNNFingers(api_type=api_type, path=path)
    elif dataset_name == "AIDS":
        return AidsGNNFingers(api_type=api_type, path=path)
    elif dataset_name == "MUTAG":
        return MutagGNNFingers(api_type=api_type, path=path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: CORA, CITESEER, PUBMED, PROTEINS, AIDS, MUTAG")


def create_multi_task_dataset(datasets: List[str], task_types: List[str], 
                             api_type: str = 'pyg', path: str = './data') -> Dict:
    """
    Create a multi-task dataset configuration for comprehensive GNNFingers evaluation.
    
    Args:
        datasets: List of dataset names
        task_types: List of task types for each dataset
        api_type: API type
        path: Data path
        
    Returns:
        Dictionary mapping task types to datasets
    """
    if len(datasets) != len(task_types):
        raise ValueError("Number of datasets must match number of task types")
    
    multi_task_config = {}
    
    for dataset_name, task_type in zip(datasets, task_types):
        try:
            dataset = get_gnnfingers_dataset(dataset_name, api_type, path)
            
            # Prepare dataset for specific task
            if task_type == "link_prediction":
                dataset.prepare_for_link_prediction()
            elif task_type == "graph_matching":
                # Graph matching requires pair creation
                pass  # Handled during training
            
            if task_type not in multi_task_config:
                multi_task_config[task_type] = []
            
            multi_task_config[task_type].append((dataset_name, dataset))
            
        except Exception as e:
            print(f"Failed to load {dataset_name} for {task_type}: {e}")
            continue
    
    return multi_task_config


def validate_dataset_compatibility(dataset, task_type: str) -> bool:
    """
    Validate if dataset is compatible with specified task type.
    
    Args:
        dataset: Dataset instance
        task_type: Type of GNN task
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        if task_type == "node_classification":
            # Check for node-level labels and masks
            return (hasattr(dataset.graph_data, 'y') and 
                   hasattr(dataset.graph_data, 'train_mask') and
                   dataset.graph_data.y.size(0) == dataset.graph_data.x.size(0))
        
        elif task_type == "graph_classification":
            # Check for graph-level labels
            return (hasattr(dataset, 'graph_dataset') and 
                   hasattr(dataset.graph_dataset[0], 'y'))
        
        elif task_type == "link_prediction":
            # Check for edge information
            return (hasattr(dataset.graph_data, 'edge_index') and
                   dataset.graph_data.edge_index.size(1) > 0)
        
        elif task_type == "graph_matching":
            # Check for graph-level data
            return (hasattr(dataset, 'graph_dataset') and 
                   len(dataset.graph_dataset) >= 2)
        
        else:
            return False
            
    except Exception as e:
        print(f"Error validating dataset compatibility: {e}")
        return False


def print_dataset_info(dataset, task_type: str):
    """
    Print comprehensive dataset information.
    
    Args:
        dataset: Dataset instance
        task_type: Type of GNN task
    """
    print(f"\nDATASET INFORMATION")
    print(f"{'='*40}")
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Task Type: {task_type.replace('_', ' ').title()}")
    print(f"API Type: {dataset.api_type}")
    print(f"{'='*40}")
    
    if task_type == "node_classification":
        print(f"Nodes: {dataset.num_nodes:,}")
        print(f"Edges: {dataset.graph_data.edge_index.size(1):,}")
        print(f"Features: {dataset.num_features}")
        print(f"Classes: {dataset.num_classes}")
        
        if hasattr(dataset.graph_data, 'train_mask'):
            train_nodes = dataset.graph_data.train_mask.sum().item()
            val_nodes = dataset.graph_data.val_mask.sum().item()
            test_nodes = dataset.graph_data.test_mask.sum().item()
            print(f"Train/Val/Test: {train_nodes}/{val_nodes}/{test_nodes}")
    
    elif task_type in ["graph_classification", "graph_matching"]:
        print(f"Graphs: {len(dataset.graph_dataset):,}")
        print(f"Node Features: {dataset.num_features}")
        print(f"Classes: {dataset.num_classes}")
        
        # Graph size statistics
        if hasattr(dataset.graph_dataset, '__getitem__'):
            sizes = [dataset.graph_dataset[i].num_nodes for i in range(min(100, len(dataset.graph_dataset)))]
            print(f"Avg Graph Size: {np.mean(sizes):.1f} +/- {np.std(sizes):.1f} nodes")
    
    elif task_type == "link_prediction":
        print(f"Nodes: {dataset.num_nodes:,}")
        print(f"Features: {dataset.num_features}")
        
        if hasattr(dataset.graph_data, 'train_pos_edge_index'):
            train_edges = dataset.graph_data.train_pos_edge_index.size(1)
            val_edges = dataset.graph_data.val_pos_edge_index.size(1)
            test_edges = dataset.graph_data.test_pos_edge_index.size(1)
            print(f"Train/Val/Test Edges: {train_edges}/{val_edges}/{test_edges}")
    
    print(f"{'='*40}")
    
    # Compatibility check
    is_compatible = validate_dataset_compatibility(dataset, task_type)
    compatibility_status = "COMPATIBLE" if is_compatible else "INCOMPATIBLE"
    print(f"Task Compatibility: {compatibility_status}")
    
    if not is_compatible:
        print("WARNING: This dataset may not work properly with the specified task type.")
    
    print(f"{'='*40}\n")