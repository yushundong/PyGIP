#!/usr/bin/env python3
"""
GNNFingers usage examples and demonstrations.

This script demonstrates how to use GNNFingers for different GNN tasks
within the PyGIP framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import warnings
warnings.filterwarnings('ignore')

from datasets.gnn_fingers_datasets import get_gnnfingers_dataset
from models.defense.gnn_fingers_defense import GNNFingersDefense
from utils.gnn_fingers_utils import print_defense_summary


def example_node_classification():
    """Example: Node classification with Cora dataset."""
    print("=" * 20 + " NODE CLASSIFICATION EXAMPLE " + "=" * 20)
    print("Demonstrating GNNFingers for node classification using Cora dataset.")
    
    # Load dataset
    dataset = get_gnnfingers_dataset("Cora", api_type='pyg')
    print(f"Loaded Cora dataset: {dataset.num_nodes} nodes, {dataset.num_features} features")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize GNNFingers defense
    defense = GNNFingersDefense(
        dataset=dataset,
        task_type="node_classification",
        num_fingerprints=32,  # Reduced for demo
        fingerprint_params={'edge_prob': 0.15},
        univerifier_params={'hidden_dims': [64, 32, 16]},
        training_params={'epochs_total': 30, 'alpha': 0.01, 'beta': 0.001},
        device=device
    )
    
    print("SUCCESS: GNNFingers defense initialized")
    
    # Run defense (quick mode)
    print("Running fingerprinting defense...")
    results = defense.defend(attack_method="fine_tuning")
    
    # Print results
    print_defense_summary(results, "node_classification", "Cora")
    
    # Demonstrate individual model verification
    print("\nTesting individual model verification:")
    if defense.positive_models:
        test_model = defense.positive_models[0]
        is_pirated, confidence = defense.verify_ownership(test_model)
        print(f"  Pirated model: Detected={is_pirated}, Confidence={confidence:.4f}")
    
    if defense.negative_models:
        test_model = defense.negative_models[0]
        is_pirated, confidence = defense.verify_ownership(test_model)
        print(f"  Independent model: Detected={is_pirated}, Confidence={confidence:.4f}")
    
    print("Node classification example completed!")
    return results


def example_graph_classification():
    """Example: Graph classification with PROTEINS dataset."""
    print("\n" + "=" * 20 + " GRAPH CLASSIFICATION EXAMPLE " + "=" * 20)
    print("Demonstrating GNNFingers for graph classification using PROTEINS dataset.")
    
    # Load dataset
    dataset = get_gnnfingers_dataset("PROTEINS", api_type='pyg')
    print(f"Loaded PROTEINS dataset: {len(dataset.graph_dataset)} graphs")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize GNNFingers defense
    defense = GNNFingersDefense(
        dataset=dataset,
        task_type="graph_classification",
        num_fingerprints=32,  # Reduced for demo
        fingerprint_params={'min_nodes': 8, 'max_nodes': 20, 'edge_prob': 0.2},
        training_params={'epochs_total': 30},
        device=device
    )
    
    print("GNNFingers defense initialized for graph classification")
    
    # Run defense (quick mode)
    print("Running fingerprinting defense...")
    results = defense.defend(attack_method="fine_tuning")
    
    # Print results
    print_defense_summary(results, "graph_classification", "PROTEINS")
    
    print("Graph classification example completed!")
    return results


def example_link_prediction():
    """Example: Link prediction with Cora dataset."""
    print("\n" + "=" * 20 + " LINK PREDICTION EXAMPLE " + "=" * 20)
    print("Demonstrating GNNFingers for link prediction using Cora dataset.")
    
    # Load dataset
    dataset = get_gnnfingers_dataset("Cora", api_type='pyg')
    
    # Prepare for link prediction
    dataset.prepare_for_link_prediction()
    print(f"Prepared Cora for link prediction")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize GNNFingers defense
    defense = GNNFingersDefense(
        dataset=dataset,
        task_type="link_prediction",
        num_fingerprints=32,  # Reduced for demo
        fingerprint_params={'num_edge_samples': 32},
        training_params={'epochs_total': 30},
        device=device
    )
    
    print("GNNFingers defense initialized for link prediction")
    
    # Run defense (quick mode)
    print("Running fingerprinting defense...")
    results = defense.defend(attack_method="fine_tuning")
    
    # Print results
    print_defense_summary(results, "link_prediction", "Cora")
    
    print("Link prediction example completed!")
    return results


def example_graph_matching():
    """Example: Graph matching with AIDS dataset."""
    print("\n" + "=" * 20 + " GRAPH MATCHING EXAMPLE " + "=" * 20)
    print("Demonstrating GNNFingers for graph matching using AIDS dataset.")
    
    # Load dataset
    dataset = get_gnnfingers_dataset("AIDS", api_type='pyg')
    print(f"Loaded AIDS dataset: {len(dataset.graph_dataset)} graphs")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize GNNFingers defense
    defense = GNNFingersDefense(
        dataset=dataset,
        task_type="graph_matching",
        num_fingerprints=32,  # Reduced for demo
        fingerprint_params={'num_fingerprint_pairs': 32, 'min_nodes': 6, 'max_nodes': 15},
        training_params={'epochs_total': 30},
        device=device
    )
    
    print("SUCCESS: GNNFingers defense initialized for graph matching")
    
    # Run defense (quick mode)
    print("Running fingerprinting defense...")
    results = defense.defend(attack_method="fine_tuning")
    
    # Print results
    print_defense_summary(results, "graph_matching", "AIDS")
    
    print("Graph matching example completed!")
    return results


def example_custom_parameters():
    """Example: Using custom parameters for advanced configuration."""
    print("\n" * 20 + " CUSTOM PARAMETERS EXAMPLE " + "" * 20)
    print("Demonstrating GNNFingers with custom advanced parameters.")
    
    # Load dataset
    dataset = get_gnnfingers_dataset("Cora", api_type='pyg')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Custom fingerprint parameters
    custom_fingerprint_params = {
        'num_nodes': 48,      # Larger fingerprint graphs
        'edge_prob': 0.25,    # Higher connectivity
    }
    
    # Custom univerifier parameters  
    custom_univerifier_params = {
        'hidden_dims': [256, 128, 64, 32],  # Deeper network
        'dropout': 0.4,       # Higher dropout
        'activation': 'leaky_relu'
    }
    
    # Custom training parameters
    custom_training_params = {
        'epochs_total': 50,   # More training epochs
        'e1': 2,              # More fingerprint updates per iteration
        'e2': 1,              # Standard univerifier updates
        'alpha': 0.008,       # Lower fingerprint learning rate
        'beta': 0.002,        # Higher univerifier learning rate
        'convergence_threshold': 0.0005  # Stricter convergence
    }
    
    # Initialize with custom parameters
    defense = GNNFingersDefense(
        dataset=dataset,
        task_type="node_classification",
        num_fingerprints=64,  # More fingerprints
        fingerprint_params=custom_fingerprint_params,
        univerifier_params=custom_univerifier_params,
        training_params=custom_training_params,
        device=device
    )
    
    print("GNNFingers initialized with custom parameters")
    print("   - Fingerprint nodes: 48")
    print("   - Univerifier layers: [256, 128, 64, 32]")
    print("   - Training epochs: 50")
    print("   - Fingerprint updates per iteration: 2")
    
    # Run defense
    print("Running advanced fingerprinting defense...")
    results = defense.defend(attack_method="comprehensive")
    
    # Print results
    print_defense_summary(results, "node_classification", "Cora")
    
    print("Custom parameters example completed!")
    return results


def example_model_verification_workflow():
    """Example: Complete model verification workflow."""
    print("\n" * 20 + " MODEL VERIFICATION WORKFLOW " + "" * 20)
    print("Demonstrating complete GNNFingers model verification workflow.")
    
    # Load dataset and setup
    dataset = get_gnnfingers_dataset("Cora", api_type='pyg')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize defense
    defense = GNNFingersDefense(
        dataset=dataset,
        task_type="node_classification",
        num_fingerprints=32,
        training_params={'epochs_total': 20},
        device=device
    )
    
    print("Step 1: Training fingerprinting system...")
    results = defense.defend(attack_method="fine_tuning")
    
    print(f"\nStep 2: Model verification workflow...")
    print("=" * 50)
    
    # Test different models
    test_cases = [
        ("Target Model", defense.target_model),
        ("Pirated Model", defense.positive_models[0] if defense.positive_models else None),
        ("Independent Model", defense.negative_models[0] if defense.negative_models else None)
    ]
    
    for model_type, model in test_cases:
        if model is not None:
            print(f"\nTesting {model_type}:")
            
            # Test with different thresholds
            thresholds = [0.3, 0.5, 0.7, 0.9]
            for threshold in thresholds:
                is_pirated, confidence = defense.verify_ownership(model, threshold=threshold)
                status = "PIRATED" if is_pirated else "CLEAN"
                print(f"   Threshold {threshold:.1f}: {status:>7} (confidence: {confidence:.4f})")
    
    print("\nStep 3: Batch verification example...")
    print("=" * 50)
    
    # Simulate batch verification
    if defense.positive_models and defense.negative_models:
        test_models = defense.positive_models[:3] + defense.negative_models[:3]
        true_labels = [1, 1, 1, 0, 0, 0]  # 1=pirated, 0=independent
        
        correct_detections = 0
        for i, (model, true_label) in enumerate(zip(test_models, true_labels)):
            is_pirated, confidence = defense.verify_ownership(model)
            predicted_label = 1 if is_pirated else 0
            correct = predicted_label == true_label
            
            if correct:
                correct_detections += 1
            
            print(f"   Model {i+1}: True={true_label}, Pred={predicted_label}, "
                  f"Conf={confidence:.4f}, {'SUCCESS' if correct else 'FAILED'}")
        
        accuracy = correct_detections / len(test_models)
        print(f"\n   Batch Verification Accuracy: {accuracy:.2%}")
    
    print("SUCCESS: Model verification workflow completed!")
    return results


def interactive_demo():
    """Interactive demo allowing user to choose examples."""
    print("\n" + "=" * 20 + " INTERACTIVE GNNFINGERS DEMO " + "=" * 20)
    print("Welcome to the GNNFingers Interactive Demo!")
    print("\nAvailable examples:")
    print("1. Node Classification (Cora)")
    print("2. Graph Classification (PROTEINS)")  
    print("3. Link Prediction (Cora)")
    print("4. Graph Matching (AIDS)")
    print("5. Custom Parameters")
    print("6. Model Verification Workflow")
    print("7. Run All Examples")
    print("0. Exit")
    
    examples_map = {
        1: ("Node Classification", example_node_classification),
        2: ("Graph Classification", example_graph_classification),
        3: ("Link Prediction", example_link_prediction),
        4: ("Graph Matching", example_graph_matching),
        5: ("Custom Parameters", example_custom_parameters),
        6: ("Model Verification", example_model_verification_workflow),
        7: ("All Examples", None)  # Special case
    }
    
    while True:
        try:
            choice = input("\nSelect example (0-7): ").strip()
            
            if choice == '0':
                print("Thanks for trying GNNFingers!")
                break
            
            choice_int = int(choice)
            
            if choice_int == 7:
                # Run all examples
                print("\nRunning all examples...")
                for i in range(1, 7):
                    name, func = examples_map[i]
                    print(f"\n{'='*80}")
                    print(f"Running Example {i}: {name}")
                    print(f"{'='*80}")
                    try:
                        func()
                    except Exception as e:
                        print(f"ERROR: Example {i} failed: {e}")
                print("\nSUCCESS: All examples completed!")
            
            elif choice_int in examples_map:
                name, func = examples_map[choice_int]
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print(f"{'='*60}")
                func()
                print(f"{'='*60}")
            
            else:
                print("ERROR: Invalid choice. Please select 0-7.")
        
        except ValueError:
            print("ERROR: Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    """Main function."""
    print("GNNFingers Examples and Demonstrations")
    print("=" * 60)
    print("This script demonstrates GNNFingers capabilities within PyGIP framework.")
    print("=" * 60)
    
    # Check if running interactively
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_demo()
    else:
        # Run a quick demonstration
        print("Running quick demonstration of GNNFingers capabilities...\n")
        
        try:
            # Quick node classification example
            print("Quick Node Classification Demo")
            print("-" * 40)
            example_node_classification()
            
            print("\n" + "=" * 60)
            print("SUCCESS: Quick demonstration completed!")
            print("\nFor interactive mode, run:")
            print("  python examples/gnn_fingers_example.py --interactive")
            print("\nFor comprehensive testing, run:")
            print("  python test.py --all --quick")
            
        except Exception as e:
            print(f"ERROR: Demo failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()