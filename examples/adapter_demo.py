#!/usr/bin/env python3
"""
Demo script showing how to use GNNFingers with existing PyGIP datasets.

This demonstrates the adapter functionality that allows GNNFingers to work
with existing PyGIP datasets like Cora(api_type='dgl').

Usage:
    python examples/adapter_demo.py
"""

import torch
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path to import PyGIP modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Original PyGIP imports (as in your existing test.py)
from datasets import Cora, PubMed
from models.attack import ModelExtractionAttack0 as MEA

# GNNFingers adapter import
try:
    from datasets.gnnfingers_adapter import PyGIPDatasetAdapter, adapt_pygip_dataset
    from models.defense.gnn_fingers_defense import GNNFingersDefense
    ADAPTER_AVAILABLE = True
except ImportError as e:
    print(f"Adapter not available: {e}")
    ADAPTER_AVAILABLE = False


def demo_original_pygip_workflow():
    """Show the original PyGIP workflow (preserved exactly)."""
    print("=" * 25 + " ORIGINAL PYGIP WORKFLOW " + "=" * 25)
    
    # Your existing code (unchanged)
    dataset = Cora(api_type='dgl')
    print(dataset)

    mea = MEA(dataset, attack_node_fraction=0.1)
    result = mea.attack()
    
    print("SUCCESS: Original PyGIP workflow completed")
    return result


def demo_gnnfingers_with_adapter():
    """Show how to use GNNFingers with existing PyGIP datasets via adapter."""
    if not ADAPTER_AVAILABLE:
        print("ERROR: GNNFingers adapter not available")
        return
    
    print("\n" + "=" * 25 + " GNNFINGERS WITH ADAPTER " + "=" * 25)
    
    # Step 1: Load original PyGIP dataset (your existing way)
    print("Step 1: Loading original PyGIP dataset...")
    original_dataset = Cora(api_type='dgl')
    print(f"SUCCESS: Loaded original Cora dataset: {original_dataset}")
    
    # Step 2: Adapt for GNNFingers compatibility
    print("\nStep 2: Adapting dataset for GNNFingers...")
    adapted_dataset = PyGIPDatasetAdapter(original_dataset)
    print(f"SUCCESS: Adapted dataset:")
    print(f"  - Name: {adapted_dataset.get_name()}")
    print(f"  - Nodes: {adapted_dataset.num_nodes}")
    print(f"  - Features: {adapted_dataset.num_features}")
    print(f"  - Classes: {adapted_dataset.num_classes}")
    print(f"  - API Type: {adapted_dataset.api_type}")
    
    # Step 3: Use GNNFingers defense
    print("\nStep 3: Using GNNFingers defense...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    defense = GNNFingersDefense(
        dataset=adapted_dataset,
        task_type="node_classification",
        num_fingerprints=32,  # Reduced for demo
        training_params={'epochs_total': 20},  # Quick demo
        device=device
    )
    
    print("SUCCESS: GNNFingers defense initialized with adapted dataset")
    
    # Step 4: Run fingerprinting (quick mode)
    print("\nStep 4: Running fingerprinting defense...")
    results = defense.defend(attack_method="fine_tuning")
    
    # Step 5: Show results
    print("\nStep 5: Results:")
    if results:
        print(f"  - AUC Score: {results.get('auc', 0):.4f}")
        print(f"  - ARUC Score: {results.get('aruc', 0):.4f}")
        if results.get('threshold_results'):
            best_result = max(results['threshold_results'], key=lambda x: x['accuracy'])
            print(f"  - Best Accuracy: {best_result['accuracy']:.4f}")
    
    print("SUCCESS: GNNFingers with adapter completed successfully!")
    return results


def demo_both_workflows():
    """Run both the original PyGIP workflow and the GNNFingers adapter workflow."""
    print("=" * 60)
    print("DEMONSTRATING BOTH WORKFLOWS")
    print("=" * 60)
    
    # Run original workflow
    original_result = demo_original_pygip_workflow()
    
    # Run GNNFingers adapter workflow
    adapter_result = demo_gnnfingers_with_adapter()
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPARISON")
    print("=" * 60)
    print("Original PyGIP workflow:")
    print(f"  - Status: {'SUCCESS' if original_result else 'FAILED'}")
    print(f"  - Result: {original_result}")
    
    print("\nGNNFingers with adapter workflow:")
    print(f"  - Status: {'SUCCESS' if adapter_result else 'FAILED'}")
    print(f"  - Result: {adapter_result}")
    
    return original_result, adapter_result


def main():
    """Main function to run the demo."""
    print("PyGIP GNNFingers Adapter Demo")
    print("=" * 40)
    print("This demo shows how to use GNNFingers with existing PyGIP datasets")
    print("=" * 40)
    
    # Check if GNNFingers is available
    if not ADAPTER_AVAILABLE:
        print("WARNING: GNNFingers adapter not available")
        print("Running only original PyGIP workflow...")
        demo_original_pygip_workflow()
        return
    
    # Run both workflows
    demo_both_workflows()
    
    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 40)
    print("\nKey Benefits of the Adapter:")
    print("1. Seamless integration with existing PyGIP datasets")
    print("2. No need to modify existing PyGIP code")
    print("3. GNNFingers defense capabilities on PyGIP datasets")
    print("4. Maintains backward compatibility")


if __name__ == "__main__":
    main()
