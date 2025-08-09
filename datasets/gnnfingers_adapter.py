"""
Adapter to make GNNFingers work with existing PyGIP dataset structure.

This adapter allows GNNFingers to work with the existing PyGIP datasets
like Cora(api_type='dgl') while maintaining compatibility.
"""

import torch
from torch_geometric.data import Data
from typing import Optional, Union


class PyGIPDatasetAdapter:
    """
    Adapter class to make existing PyGIP datasets work with GNNFingers.
    
    This converts DGL-based datasets to PyG format for GNNFingers compatibility
    while preserving the original PyGIP interface.
    """
    
    def __init__(self, pygip_dataset):
        """
        Initialize adapter with PyGIP dataset.
        
        Args:
            pygip_dataset: Original PyGIP dataset (e.g., Cora(api_type='dgl'))
        """
        self.original_dataset = pygip_dataset
        self.dataset_name = getattr(pygip_dataset, 'dataset_name', 'Unknown')
        
        # Set metadata first (use PyGIP Dataset metadata when available)
        self.num_nodes = getattr(pygip_dataset, 'num_nodes', getattr(pygip_dataset, 'node_number', 0))
        self.num_features = getattr(pygip_dataset, 'num_features', getattr(pygip_dataset, 'feature_number', 0))
        self.num_classes = getattr(pygip_dataset, 'num_classes', getattr(pygip_dataset, 'label_number', 0))
        
        # Convert to PyG format for GNNFingers
        self.graph_data = self._convert_to_pyg()
        self.graph_dataset = None  # For graph-level tasks, would need dataset list
        
        # API type
        self.api_type = 'pyg'  # Adapter always outputs PyG format
    
    def _convert_to_pyg(self) -> Data:
        """Convert DGL graph to PyG Data format."""
        try:
            # Prefer PyGIP's graph_data when present
            if hasattr(self.original_dataset, 'graph_data') and self.original_dataset.graph_data is not None:
                try:
                    # Detect DGLGraph via duck typing to avoid hard dependency
                    dgl_graph = self.original_dataset.graph_data
                    # DGL graph has .edges() and .ndata
                    if hasattr(dgl_graph, 'edges') and hasattr(dgl_graph, 'ndata'):
                        src, dst = dgl_graph.edges()
                        edge_index = torch.stack([src, dst], dim=0).long()
                        x = dgl_graph.ndata.get('feat')
                        y = dgl_graph.ndata.get('label')
                        train_mask = dgl_graph.ndata.get('train_mask')
                        val_mask = dgl_graph.ndata.get('val_mask')
                        test_mask = dgl_graph.ndata.get('test_mask')
                        if x is None:
                            x = torch.randn(self.num_nodes, max(1, self.num_features))
                        if y is None:
                            y = torch.zeros(self.num_nodes).long()
                        if train_mask is None:
                            train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                        if val_mask is None:
                            val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                        if test_mask is None:
                            test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                        return Data(x=x, edge_index=edge_index, y=y,
                                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
                except Exception:
                    pass
                # If it's already a PyG Data, just use it
                if isinstance(self.original_dataset.graph_data, Data):
                    return self.original_dataset.graph_data
            
            # Legacy attribute path
            if hasattr(self.original_dataset, 'graph') and self.original_dataset.graph is not None:
                dgl_graph = self.original_dataset.graph
                src, dst = dgl_graph.edges()
                edge_index = torch.stack([src, dst], dim=0).long()
                x = getattr(self.original_dataset, 'features', None)
                y = getattr(self.original_dataset, 'labels', None)
                if x is None:
                    x = torch.randn(self.num_nodes, max(1, self.num_features))
                else:
                    x = x.float()
                if y is None:
                    y = torch.zeros(self.num_nodes).long()
                train_mask = getattr(self.original_dataset, 'train_mask', torch.zeros(self.num_nodes).bool())
                val_mask = getattr(self.original_dataset, 'val_mask', torch.zeros(self.num_nodes).bool())
                test_mask = getattr(self.original_dataset, 'test_mask', torch.zeros(self.num_nodes).bool())
                return Data(x=x, edge_index=edge_index, y=y,
                            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

            # Fallback: create synthetic data
            print("WARNING: No graph data found, creating synthetic data")
            return self._create_synthetic_data()

        except Exception as e:
            print(f"WARNING: Error converting dataset ({e}), creating synthetic data")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> Data:
        """Create synthetic data as fallback."""
        num_nodes = max(100, self.num_nodes)
        num_features = max(10, self.num_features)
        num_classes = max(2, self.num_classes)
        
        # Create random graph
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        edge_index = torch.unique(edge_index, dim=1)
        
        # Create features and labels
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # Create masks
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:train_size] = True
        val_mask[train_size:train_size + val_size] = True
        test_mask[train_size + val_size:] = True
        
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        # Update metadata
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        
        return data
    
    def get_name(self):
        """Get dataset name."""
        return self.dataset_name
    
    def prepare_for_link_prediction(self):
        """Prepare dataset for link prediction tasks."""
        from torch_geometric.utils import train_test_split_edges, to_undirected, remove_self_loops
        
        # Remove self-loops and make undirected
        self.graph_data.edge_index, _ = remove_self_loops(self.graph_data.edge_index)
        self.graph_data.edge_index = to_undirected(self.graph_data.edge_index)
        
        # Split edges for link prediction
        self.graph_data = train_test_split_edges(self.graph_data, val_ratio=0.1, test_ratio=0.2)
        
        print(f"Link prediction splits:")
        print(f"  Train edges: {self.graph_data.train_pos_edge_index.size(1)}")
        print(f"  Val edges: {self.graph_data.val_pos_edge_index.size(1)}")
        print(f"  Test edges: {self.graph_data.test_pos_edge_index.size(1)}")


def adapt_pygip_dataset(dataset_name: str, api_type: str = 'dgl'):
    """
    Factory function to adapt existing PyGIP datasets for GNNFingers.
    
    Args:
        dataset_name: Name of the PyGIP dataset
        api_type: API type for the original dataset
    
    Returns:
        Adapted dataset compatible with GNNFingers
    """
    try:
        # Import existing PyGIP datasets
        if dataset_name.upper() == 'CORA':
            from datasets import Cora
            original_dataset = Cora(api_type=api_type)
        elif dataset_name.upper() == 'PUBMED':
            from datasets import PubMed  
            original_dataset = PubMed(api_type=api_type)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for adaptation")
        
        print(f"SUCCESS: Loaded original PyGIP {dataset_name} dataset")
        
        # Create adapter
        adapted_dataset = PyGIPDatasetAdapter(original_dataset)
        print(f"SUCCESS: Adapted {dataset_name} for GNNFingers compatibility")
        
        return adapted_dataset
        
    except Exception as e:
        print(f"ERROR: Failed to adapt {dataset_name}: {e}")
        raise


def test_adaptation():
    """Test the dataset adaptation functionality."""
    print("Testing PyGIP Dataset Adaptation")
    print("=" * 50)
    
    datasets_to_test = ['Cora', 'PubMed']
    
    for dataset_name in datasets_to_test:
        try:
            print(f"\nTesting {dataset_name} adaptation...")
            adapted_dataset = adapt_pygip_dataset(dataset_name, api_type='dgl')
            
            print(f"  Dataset name: {adapted_dataset.get_name()}")
            print(f"  Nodes: {adapted_dataset.num_nodes}")
            print(f"  Features: {adapted_dataset.num_features}")
            print(f"  Classes: {adapted_dataset.num_classes}")
            print(f"  Graph data shape: {adapted_dataset.graph_data.x.shape}")
            print(f"  Edge index shape: {adapted_dataset.graph_data.edge_index.shape}")
            print(f"SUCCESS: {dataset_name} adaptation successful")
            
        except Exception as e:
            print(f"ERROR: {dataset_name} adaptation failed: {e}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    test_adaptation()