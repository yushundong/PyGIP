#!/usr/bin/env python3
"""
Test script for PyGIP Dataset Adaptation functionality.

This script tests the adapter functionality that allows GNNFingers to work
with existing PyGIP datasets.

Usage:
    python examples/test_adapter.py
"""

import sys
import os

# Add project root to path to import PyGIP modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.gnnfingers_adapter import adapt_pygip_dataset


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


def main():
    """Main function to run the adapter tests."""
    print("PyGIP Dataset Adapter Test Suite")
    print("=" * 40)
    print("This script tests the adapter functionality for GNNFingers")
    print("=" * 40)
    
    test_adaptation()
    
    print("\n" + "=" * 40)
    print("ADAPTER TESTS COMPLETED!")
    print("=" * 40)


if __name__ == "__main__":
    main()
