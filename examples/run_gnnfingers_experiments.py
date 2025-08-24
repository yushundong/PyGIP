#!/usr/bin/env python3
"""
GNNFingers Experiment Runner for PyGIP

This script runs comprehensive GNNFingers experiments for all task-dataset-model combinations.
It supports both quick and full training modes and can run individual experiments or all 22 tests.

Usage:
    python examples/run_gnnfingers_experiments.py --all --quick
    python examples/run_gnnfingers_experiments.py --all --full
    python examples/run_gnnfingers_experiments.py --task node_classification --dataset Cora --model GCN --quick
"""

import sys
import os

# Add project root to path to import PyGIP modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import Cora, PubMed
from models.attack import ModelExtractionAttack0 as MEA
import argparse
import torch
import warnings
import datetime
import copy
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional

warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*torch-cluster.*")
warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
warnings.filterwarnings("ignore", message=".*torch-sparse.*")
warnings.filterwarnings('ignore')

# Original PyGIP workflow (preserved for compatibility)
dataset = Cora(api_type='dgl')
print(dataset)

mea = MEA(dataset, attack_node_fraction=0.1)
mea.attack()

try:
    from models.defense.gnn_fingers_models import get_model_for_task, ModelObfuscator, Univerifier
    from models.defense.gnn_fingers_defense import GNNFingersDefense
    from datasets.gnn_fingers_datasets import get_gnnfingers_dataset, print_dataset_info
    from datasets.gnnfingers_adapter import PyGIPDatasetAdapter, adapt_pygip_dataset
    from utils.gnn_fingers_utils import (
        print_defense_summary, generate_defense_report, 
        save_defense_results, plot_robustness_uniqueness_curve
    )
    GNNFINGERS_AVAILABLE = True
    print("GNNFingers modules loaded successfully")
except ImportError as e:
    GNNFINGERS_AVAILABLE = False
    print(f"GNNFingers not available: {e}")


def setup_device():
    """Setup computing device."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return device


def run_single_experiment_with_mode(task_type, dataset_name, model_architecture, device, quick_mode=False, experiment_num=1):
    """Run a single experiment in the specified mode (quick or full)."""
    try:
        # Clear CUDA cache before starting experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared before experiment")
        # Load dataset
        if dataset_name.upper() in ['CORA', 'CITESEER']:
            try:
                print(f"Attempting to use PyGIP {dataset_name} dataset...")
                adapted_dataset = adapt_pygip_dataset(dataset_name, api_type='dgl')
                print(f"Successfully adapted PyGIP {dataset_name} dataset")
            except Exception as e:
                print(f"PyGIP adapter failed: {e}")
                print(f"Using native GNNFingers {dataset_name} dataset...")
                adapted_dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
        else:
            adapted_dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
        
        print(f"Dataset loaded: {dataset_name}")
        print_dataset_info(adapted_dataset, task_type)
        
        # Configure based on quick/full mode
        if quick_mode:
            num_fingerprints = 32
            training_epochs = 50
            print(f"Quick mode: {num_fingerprints} fingerprints, {training_epochs} epochs")
        else:
            num_fingerprints = 64
            training_epochs = 100
            print(f"Full mode: {num_fingerprints} fingerprints, {training_epochs} epochs")
        
        # Initialize defense with specific model architecture
        print(f"Initializing GNNFingers defense for {task_type} on {dataset_name} with {model_architecture}...")
        defense = GNNFingersDefense(
            task_type=task_type,
            dataset=adapted_dataset,
            model_name=model_architecture  # Use the specific architecture
        )
        
        # Set the specific model architecture (for backward compatibility)
        if hasattr(defense, 'model_architecture'):
            defense.model_architecture = model_architecture
            print(f"Model architecture set to: {model_architecture}")
        
        print("Defense initialized successfully")
        
        # Run defense with comprehensive attack method
        print(f"Starting defense training for {task_type} on {dataset_name} with {model_architecture}...")
        start_time = datetime.datetime.now()
        result = defense.defend(attack_method="comprehensive")  # Use comprehensive attack method
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(f"Defense training completed in {training_time}")
        
        # Store results
        experiment_result = {
            'task_type': task_type,
            'dataset_name': dataset_name,
            'model_architecture': model_architecture,
            'result': result,
            'training_time': str(training_time),
            'status': 'SUCCESS'
        }
        
        # Save model weights
        mode_suffix = 'quick' if quick_mode else 'full'
        save_path = f"./weights/gnnfingers_{task_type}_{dataset_name.lower()}_{model_architecture.lower()}_{mode_suffix}.pth"
        os.makedirs("./weights", exist_ok=True)
        
        torch.save({
            'target_model_state_dict': defense.target_model.state_dict(),
            'univerifier_state_dict': defense.univerifier.state_dict(),
            'fingerprint_constructor': defense.fingerprint_constructor,
            'training_history': defense.training_history,
            'results': result,
            'task_type': task_type,
            'dataset_name': dataset_name,
            'model_architecture': model_architecture,
            'mode': mode_suffix,
            'timestamp': datetime.datetime.now().isoformat()
        }, save_path)
        
        # Save individual experiment results immediately
        individual_result = {
            'experiment_info': {
                'experiment_number': experiment_num,
                'task_type': task_type,
                'dataset_name': dataset_name,
                'model_architecture': model_architecture,
                'mode': mode_suffix,
                'attack_method': 'comprehensive',
                'timestamp': datetime.datetime.now().isoformat(),
                'training_time': str(training_time)
            },
            'performance_metrics': result,
            'training_history': defense.training_history,
            'model_path': save_path
        }
        
        # Save individual experiment result
        os.makedirs("./gnnfinger_results_json", exist_ok=True)
        individual_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        individual_filename = f"gnnfingers_{task_type}_{dataset_name.lower()}_{model_architecture.lower()}_{mode_suffix}_{individual_timestamp}.json"
        individual_path = f"./gnnfinger_results_json/{individual_filename}"
        
        import json
        with open(individual_path, 'w') as f:
            json.dump(individual_result, f, indent=2, default=str)
        
        print(f"SUCCESS: {task_type} - {dataset_name} ({model_architecture}) - {mode_suffix.upper()} mode: AUC={result['auc']:.4f}, ARUC={result['aruc']:.4f}")
        print(f"   Model saved to: {save_path}")
        print(f"   Individual results saved to: {individual_path}")
        
        return experiment_result
        
    except Exception as e:
        print(f"ERROR: {task_type} - {dataset_name} ({model_architecture}) - {'QUICK' if quick_mode else 'FULL'} mode failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'task_type': task_type,
            'dataset_name': dataset_name,
            'model_architecture': model_architecture,
            'mode': 'quick' if quick_mode else 'full',
            'error': str(e),
            'status': 'FAILED'
        }


def run_all_gnnfingers_experiments(quick_mode=False):
    """Run all 22 comprehensive GNNFingers experiments."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers not available. Cannot run experiments.")
        return None
    
    print("\nRunning all 22 comprehensive GNNFingers experiments")
    print("=" * 80)
    print("This covers all task-dataset-model combinations from the test matrix")
    print("=" * 80)
    
    # Complete test matrix based on supported datasets
    test_matrix = [
        
        # Node Classification - Cora (2 tests)
        ("node_classification", "Cora", "GCN"),
        ("node_classification", "Cora", "Graphsage"),
        
        # Node Classification - Citeseer (2 tests)
        ("node_classification", "Citeseer", "GCN"),
        ("node_classification", "Citeseer", "Graphsage"),
        
        # Link Prediction - Cora (2 tests)
        ("link_prediction", "Cora", "GCN"),
        ("link_prediction", "Cora", "Graphsage"),
        
        # Link Prediction - Citeseer (2 tests)
        ("link_prediction", "Citeseer", "GCN"),
        ("link_prediction", "Citeseer", "Graphsage"),

        # Graph Matching - AIDS (3 tests)
        ("graph_matching", "AIDS", "GCNMean"),
        ("graph_matching", "AIDS", "GCNDiff"),
        ("graph_matching", "AIDS", "SimGNN"),
        
        # Graph Matching - PROTEINS (3 tests)
        ("graph_matching", "PROTEINS", "GCNMean"),
        ("graph_matching", "PROTEINS", "GCNDiff"),
        ("graph_matching", "PROTEINS", "SimGNN"),

        # Graph Classification - PROTEINS (4 tests)
        ("graph_classification", "PROTEINS", "GCNMean"),
        ("graph_classification", "PROTEINS", "GCNDiff"),
        ("graph_classification", "PROTEINS", "GraphsageMean"),
        ("graph_classification", "PROTEINS", "GraphsageDiff"),
        
        # Graph Classification - AIDS (4 tests)
        ("graph_classification", "AIDS", "GCNMean"),
        ("graph_classification", "AIDS", "GCNDiff"),
        ("graph_classification", "AIDS", "GraphsageMean"),
        ("graph_classification", "AIDS", "GraphsageDiff"),
    ]
    
    print(f"Total experiments to run: {len(test_matrix)}")
    print("\nTest Matrix:")
    print(f"{'Task':<25} {'Dataset':<12} {'Model':<15} {'Status':<10}")
    print("-" * 70)
    
    for task, dataset, model in test_matrix:
        print(f"{task:<25} {dataset:<12} {model:<15} {'PENDING':<10}")
    
    print("\n" + "=" * 80)
    
    device = setup_device()
    results_summary = {}
    successful_experiments = 0
    failed_experiments = 0
    
    for i, (task_type, dataset_name, model_architecture) in enumerate(test_matrix, 1):
        print(f"\n{'='*20} Experiment {i}/22: {task_type} - {dataset_name} ({model_architecture}) {'='*20}")
        
        # Clear CUDA cache before each experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared before experiment")
        
        try:
            # Load dataset
            if dataset_name.upper() in ['CORA', 'CITESEER']:
                try:
                    print(f"Attempting to use PyGIP {dataset_name} dataset...")
                    adapted_dataset = adapt_pygip_dataset(dataset_name, api_type='dgl')
                    print(f"Successfully adapted PyGIP {dataset_name} dataset")
                except Exception as e:
                    print(f"PyGIP adapter failed: {e}")
                    print(f"Using native GNNFingers {dataset_name} dataset...")
                    adapted_dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
            else:
                adapted_dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
            
            print(f"Dataset loaded: {dataset_name}")
            print_dataset_info(adapted_dataset, task_type)
            
            # Configure based on quick/full mode
            if quick_mode:
                num_fingerprints = 32
                training_epochs = 50
                print(f"Quick mode: {num_fingerprints} fingerprints, {training_epochs} epochs")
            else:
                num_fingerprints = 64
                training_epochs = 100
                print(f"Full mode: {num_fingerprints} fingerprints, {training_epochs} epochs")
            
            # Initialize defense with specific model architecture
            print(f"Initializing GNNFingers defense for {task_type} on {dataset_name} with {model_architecture}...")
            defense = GNNFingersDefense(
                task_type=task_type,
                dataset=adapted_dataset,
                model_name=model_architecture  # Use the specific architecture
            )
            
            # Set the specific model architecture (for backward compatibility)
            if hasattr(defense, 'model_architecture'):
                defense.model_architecture = model_architecture
                print(f"Model architecture set to: {model_architecture}")
            
            print("Defense initialized successfully")
            
            # Run comprehensive defense (always uses all 4 attacking methods)
            print(f"Starting comprehensive defense training for {task_type} on {dataset_name} with {model_architecture}...")
            start_time = datetime.datetime.now()
            result = defense.defend(attack_method="comprehensive")
            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            print(f"Defense training completed in {training_time}")
            
            # Store results
            test_key = f"{task_type}_{dataset_name}_{model_architecture}"
            results_summary[test_key] = {
                    'task_type': task_type,
                'dataset_name': dataset_name,
                'model_architecture': model_architecture,
                'result': result,
                'training_time': str(training_time),
                    'status': 'SUCCESS'
                }
            
            # Save model weights
            save_path = f"./weights/gnnfingers_{task_type}_{dataset_name.lower()}_{model_architecture.lower()}.pth"
            os.makedirs("./weights", exist_ok=True)
            
            torch.save({
                'target_model_state_dict': defense.target_model.state_dict(),
                'univerifier_state_dict': defense.univerifier.state_dict(),
                'fingerprint_constructor': defense.fingerprint_constructor,
                'training_history': defense.training_history,
                'results': result,
                    'task_type': task_type,
                'dataset_name': dataset_name,
                'model_architecture': model_architecture,
                'timestamp': datetime.datetime.now().isoformat()
            }, save_path)
            
            # Save individual experiment results immediately
            individual_result = {
                'experiment_info': {
                    'task_type': task_type,
                    'dataset_name': dataset_name,
                    'model_architecture': model_architecture,
                    'mode': 'quick' if quick_mode else 'full',
                    'attack_method': 'comprehensive',
                    'timestamp': datetime.datetime.now().isoformat(),
                    'training_time': str(training_time)
                },
                'performance_metrics': result,
                'training_history': defense.training_history,
                'model_path': save_path
            }
            
            # Save individual experiment result
            os.makedirs("./gnnfinger_results_json", exist_ok=True)
            individual_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            individual_filename = f"gnnfingers_{task_type}_{dataset_name.lower()}_{model_architecture.lower()}_{('quick' if quick_mode else 'full')}_{individual_timestamp}.json"
            individual_path = f"./gnnfinger_results_json/{individual_filename}"
            
            import json
            with open(individual_path, 'w') as f:
                json.dump(individual_result, f, indent=2, default=str)
            
            print(f"SUCCESS: {task_type} - {dataset_name} ({model_architecture}): AUC={result['auc']:.4f}, ARUC={result['aruc']:.4f}")
            print(f"   Model saved to: {save_path}")
            print(f"   Individual results saved to: {individual_path}")
            successful_experiments += 1
                
        except Exception as e:
            print(f"ERROR: {task_type} - {dataset_name} ({model_architecture}) failed: {e}")
            import traceback
            traceback.print_exc()
            
            test_key = f"{task_type}_{dataset_name}_{model_architecture}"
            results_summary[test_key] = {
                'task_type': task_type,
                'dataset_name': dataset_name,
                'model_architecture': model_architecture,
                'error': str(e),
                'status': 'FAILED'
            }
            failed_experiments += 1
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EXPERIMENTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(test_matrix)}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success rate: {successful_experiments/len(test_matrix)*100:.1f}%")
    
    if successful_experiments > 0:
        print(f"\nSuccessful Experiments:")
        print(f"{'Task':<25} {'Dataset':<12} {'Model':<15} {'AUC':<8} {'ARUC':<8}")
        print("-" * 75)
        
        for test_key, test_result in results_summary.items():
            if test_result['status'] == 'SUCCESS':
                task = test_result['task_type']
                dataset = test_result['task_type']
                model = test_result['model_architecture']
                result = test_result['result']
                auc = f"{result.get('auc', 0):.3f}"
                aruc = f"{result.get('aruc', 0):.3f}"
                print(f"{task:<25} {dataset:<12} {model:<15} {auc:<8} {aruc:<8}")
    
    if failed_experiments > 0:
        print(f"\nFailed Experiments:")
        print(f"{'Task':<25} {'Dataset':<12} {'Model':<15} {'Error':<30}")
        print("-" * 85)
        
        for test_key, test_result in results_summary.items():
            if test_result['status'] == 'FAILED':
                task = test_result['task_type']
                dataset = test_result['dataset_name']
                model = test_result['model_architecture']
                error = test_result['error'][:27] + "..." if len(test_result['error']) > 30 else test_result['error']
                print(f"{task:<25} {dataset:<12} {model:<15} {error:<30}")
    
    # Save comprehensive results
    os.makedirs("./gnnfinger_results_json", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"./gnnfinger_results_json/all_experiments_{timestamp}.json"
    
    comprehensive_results = {
        'experiment_type': 'all_22_experiments',
        'timestamp': datetime.datetime.now().isoformat(),
        'configuration': {
            'mode': 'quick' if quick_mode else 'full',
            'attack_method': 'comprehensive',
            'total_experiments': len(test_matrix),
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments/len(test_matrix)*100
        },
        'test_matrix': test_matrix,
        'results': results_summary
    }
    
    import json
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\nAll comprehensive results saved to: {results_path}")
    return comprehensive_results


def main():
    """Main function to run GNNFingers tests."""
    # Automatic device selection: GPU if available, else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
        print("GPU not available, using CPU")
    
    print("GNNFingers Test Suite for PyGIP")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"GNNFingers available: {True}")
    print("=" * 60)
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GNNFingers Test Suite for PyGIP')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Use quick training mode')
    parser.add_argument('--full', action='store_true', help='Use full training mode')
    
    # Individual task options
    parser.add_argument('--task', type=str, choices=['node_classification', 'graph_classification', 'link_prediction', 'graph_matching'], 
                       help='Run specific task type')
    parser.add_argument('--dataset', type=str, help='Dataset name for individual task (e.g., Cora, PROTEINS, AIDS)')
    parser.add_argument('--model', type=str, help='Model architecture for individual task (e.g., GCN, GCNMean, GCNDiff)')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if len(sys.argv) == 1:
        print("GNNFingers Test Suite - Available Commands:")
        print("  --all --quick          : Run all experiments in quick mode")
        print("  --all --full           : Run all experiments in full mode")
        print("  --task TASK --dataset DATASET --model MODEL [--quick] : Run specific task")
        print("  --help                 : Show detailed help message")
        print()
        print("Examples:")
        print("  python examples/run_gnnfingers_experiments.py --all --quick")
        print("  python examples/run_gnnfingers_experiments.py --all --full")
        print("  python examples/run_gnnfingers_experiments.py --task node_classification --dataset Cora --model GCN --quick")
        print("  python examples/run_gnnfingers_experiments.py --task graph_classification --dataset PROTEINS --model GCNMean --quick")
        print("  python examples/run_gnnfingers_experiments.py --task link_prediction --dataset Cora --model GCN --quick")
        print("  python examples/run_gnnfingers_experiments.py --task graph_matching --dataset AIDS --model GCNMean --quick")
        return
    
    if args.all and args.quick:
        print("Running all experiments in quick mode...")
        run_all_gnnfingers_experiments(quick_mode=True)
    elif args.all and args.full:
        print("Running all experiments in full mode...")
        run_all_gnnfingers_experiments(quick_mode=False)
    elif args.task and args.dataset and args.model:
        # Run individual task
        print(f"Running individual task: {args.task} on {args.dataset} with {args.model}")
        quick_mode = args.quick
        mode_str = "quick" if quick_mode else "full"
        print(f"Training mode: {mode_str}")
        
        # Validate task-dataset-model combination
        valid_combinations = {
            'node_classification': ['Cora', 'Citeseer', 'PubMed'],
            'graph_classification': ['PROTEINS', 'AIDS', 'MUTAG'],
            'link_prediction': ['Cora', 'Citeseer', 'PubMed'],
            'graph_matching': ['AIDS', 'PROTEINS']
        }
        
        if args.task not in valid_combinations:
            print(f"ERROR: Invalid task type: {args.task}")
            return
        
        if args.dataset not in valid_combinations[args.task]:
            print(f"ERROR: Dataset {args.dataset} is not valid for task {args.task}")
            print(f"Valid datasets for {args.task}: {valid_combinations[args.task]}")
            return
        
        # Validate model architecture for the task
        valid_models = {
            'node_classification': ['GCN', 'Graphsage'],
            'graph_classification': ['GCNMean', 'GCNDiff', 'GraphsageMean', 'GraphsageDiff'],
            'link_prediction': ['GCN', 'Graphsage'],
            'graph_matching': ['GCNMean', 'GCNDiff', 'SimGNN']
        }
        
        if args.model not in valid_models[args.task]:
            print(f"ERROR: Model {args.model} is not valid for task {args.task}")
            print(f"Valid models for {args.task}: {valid_models[args.task]}")
            return
        
        print(f"Validation passed. Running {args.task} on {args.dataset} with {args.model} in {mode_str} mode...")
        
        try:
            # Run the individual experiment
            result = run_single_experiment_with_mode(
                task_type=args.task,
                dataset_name=args.dataset,
                model_architecture=args.model,
                device=device,
                quick_mode=quick_mode,
                experiment_num=1
            )
            
            if result:
                print(f"\nSUCCESS: Individual task completed successfully!")
                print(f"Task: {args.task}")
                print(f"Dataset: {args.dataset}")
                print(f"Model: {args.model}")
                print(f"Mode: {mode_str}")
                print(f"Results saved to: ./gnnfinger_results_json/")
                print(f"Model saved to: ./weights/")
            else:
                print(f"\nFAILED: Individual task failed!")
                
        except Exception as e:
            print(f"ERROR: Failed to run individual task: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No valid command specified. Use --help for available options.")


if __name__ == "__main__":
    main()
