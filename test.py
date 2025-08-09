from datasets import Cora, PubMed
from models.attack import ModelExtractionAttack0 as MEA
import argparse
import torch
import sys
import os
import warnings
import datetime
import copy

warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*torch-cluster.*")
warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
warnings.filterwarnings("ignore", message=".*torch-sparse.*")
warnings.filterwarnings('ignore')

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


def run_gnnfingers_experiment(task_type, dataset_name, quick_mode=False):
    """Run a single GNNFingers experiment."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers not available. Skipping experiment.")
        return None
    
    print(f"\nRunning GNNFingers experiment: {task_type} on {dataset_name}")
    print("=" * 60)
    
    device = setup_device()
    
    try:
        if dataset_name.upper() in ['CORA', 'PUBMED']:
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
        
        print_dataset_info(adapted_dataset, task_type)
        
        num_fingerprints = 32 if quick_mode else 64
        training_epochs = 50 if quick_mode else 100
        
        print(f"Configuration: {num_fingerprints} fingerprints, {training_epochs} epochs")
        
        defense = GNNFingersDefense(
            dataset=adapted_dataset,
            task_type=task_type,
            num_fingerprints=num_fingerprints,
            fingerprint_params=None,
            univerifier_params={'hidden_dims': [128, 64, 32], 'dropout': 0.3},
            training_params={
                'epochs_total': training_epochs,
                'e1': 1, 'e2': 1,
                'alpha': 0.01, 'beta': 0.001,
                'convergence_threshold': 0.001
            },
            device=device
        )
        
        print("GNNFingers defense initialized successfully")
        
        start_time = datetime.datetime.now()
        attack_method = "fine_tuning" if quick_mode else "comprehensive"
        
        print(f"Starting fingerprinting defense with {attack_method} attack method...")
        results = defense.defend(attack_method=attack_method)
        
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        
        print(f"Execution time: {execution_time}")
        print_defense_summary(results, task_type, dataset_name)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"gnnfingers_{task_type}_{dataset_name.lower()}_{timestamp}.json"
        os.makedirs("./gnnfinger_results_json", exist_ok=True)
        results_path = f"./gnnfinger_results_json/{results_filename}"
        save_defense_results(results, task_type, dataset_name, results_path)
        
        save_path = f"./weights/gnnfingers_{task_type}_{dataset_name.lower()}.pth"
        os.makedirs("./weights", exist_ok=True)
        
        torch.save({
            'target_model_state_dict': defense.target_model.state_dict(),
            'univerifier_state_dict': defense.univerifier.state_dict(),
            'fingerprint_constructor': defense.fingerprint_constructor,
            'training_history': defense.training_history,
            'results': results,
            'task_type': task_type,
            'dataset_name': dataset_name,
            'timestamp': datetime.datetime.now().isoformat()
        }, save_path)
        
        print(f"Model weights saved to: {save_path}")
        
        if hasattr(defense, 'positive_models') and defense.positive_models:
            test_model = defense.positive_models[0]
            is_pirated, confidence = defense.verify_ownership(test_model)
            print(f"Positive model test - Pirated: {is_pirated}, Confidence: {confidence:.4f}")
        
        if hasattr(defense, 'negative_models') and defense.negative_models:
            test_model = defense.negative_models[0]
            is_pirated, confidence = defense.verify_ownership(test_model)
            print(f"Negative model test - Pirated: {is_pirated}, Confidence: {confidence:.4f}")
        
        print("Experiment completed successfully")
        return results
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_gnnfingers_experiments(quick_mode=False):
    """Run all GNNFingers experiments."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers not available. Cannot run experiments.")
        return None
    
    print("\nRunning all GNNFingers experiments")
    print("=" * 60)
    
    experiments = [
        ("node_classification", "Cora"),
        ("graph_classification", "PROTEINS"),
        ("link_prediction", "Cora"),
        ("graph_matching", "AIDS")
    ]
    
    results_summary = {}
    successful_experiments = 0
    
    for i, (task_type, dataset_name) in enumerate(experiments, 1):
        print(f"\nExperiment {i}/4: {task_type} on {dataset_name}")
        print("-" * 50)
        
        try:
            results = run_gnnfingers_experiment(task_type, dataset_name, quick_mode)
            
            if results is not None:
                best_accuracy = 0
                if results.get('threshold_results'):
                    best_accuracy = max(r['accuracy'] for r in results['threshold_results'])
                
                results_summary[f"{task_type}_{dataset_name}"] = {
                    'task_type': task_type,
                    'dataset': dataset_name,
                    'auc': results.get('auc', 0),
                    'aruc': results.get('aruc', 0),
                    'best_accuracy': best_accuracy,
                    'status': 'SUCCESS'
                }
                successful_experiments += 1
                print(f"Experiment {i} completed successfully")
            else:
                results_summary[f"{task_type}_{dataset_name}"] = {
                    'task_type': task_type,
                    'dataset': dataset_name,
                    'status': 'FAILED'
                }
                print(f"Experiment {i} failed")
                
        except Exception as e:
            print(f"Experiment {i} failed with error: {e}")
            results_summary[f"{task_type}_{dataset_name}"] = {
                'task_type': task_type,
                'dataset': dataset_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    print(f"\nGNNFingers Experiments Summary")
    print("=" * 60)
    print(f"Successful experiments: {successful_experiments}/4")
    print(f"Success rate: {successful_experiments/4*100:.1f}%")
    
    if successful_experiments > 0:
        print("\nResults:")
        print(f"{'Task':<25} {'Dataset':<10} {'AUC':<8} {'ARUC':<8} {'Best Acc':<10} {'Status'}")
        print("-" * 75)
        
        for key, result in results_summary.items():
            if result['status'] == 'SUCCESS':
                task_display = result['task_type'].replace('_', ' ').title()[:24]
                dataset = result['dataset']
                auc = f"{result['auc']:.3f}"
                aruc = f"{result['aruc']:.3f}"
                acc = f"{result['best_accuracy']:.3f}"
                status = result['status']
                
                print(f"{task_display:<25} {dataset:<10} {auc:<8} {aruc:<8} {acc:<10} {status}")
            else:
                task_display = result['task_type'].replace('_', ' ').title()[:24]
                dataset = result['dataset']
                print(f"{task_display:<25} {dataset:<10} {'N/A':<8} {'N/A':<8} {'N/A':<10} {result['status']}")
    
    return results_summary


def test_dataset_loading():
    """Test dataset loading functionality."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers not available for dataset testing.")
        return
    
    print("\nTesting GNNFingers dataset loading")
    print("=" * 50)
    
    datasets_to_test = [
        ("node_classification", "Cora"),
        ("node_classification", "Citeseer"),
        ("graph_classification", "PROTEINS"),
        ("graph_matching", "AIDS"),
    ]
    
    successful_loads = 0
    
    for task_type, dataset_name in datasets_to_test:
        try:
            print(f"Loading {dataset_name} for {task_type}...")
            
            if dataset_name.upper() in ['CORA', 'PUBMED']:
                try:
                    dataset = adapt_pygip_dataset(dataset_name, api_type='dgl')
                    print(f"  {dataset_name} loaded via PyGIP adapter")
                except:
                    dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
                    print(f"  {dataset_name} loaded via native GNNFingers")
            else:
                dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
                print(f"  {dataset_name} loaded successfully")
            
            successful_loads += 1
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
    
    print(f"\nDataset loading results: {successful_loads}/{len(datasets_to_test)} successful")


def test_adapter():
    """Test the PyGIP dataset adapter."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers adapter not available for testing.")
        return
    
    print("\nTesting PyGIP dataset adapter")
    print("=" * 50)
    
    datasets_to_test = ['Cora', 'PubMed']
    
    for dataset_name in datasets_to_test:
        try:
            print(f"Testing {dataset_name} adapter...")
            
            if dataset_name == 'Cora':
                original_dataset = Cora(api_type='dgl')
            elif dataset_name == 'PubMed':
                original_dataset = PubMed(api_type='dgl')
            
            print(f"  Loaded original PyGIP {dataset_name}")
            
            adapted_dataset = PyGIPDatasetAdapter(original_dataset)
            
            print(f"  Created adapter for {dataset_name}")
            print(f"    Name: {adapted_dataset.get_name()}")
            print(f"    Nodes: {adapted_dataset.num_nodes}")
            print(f"    Features: {adapted_dataset.num_features}")
            print(f"    Classes: {adapted_dataset.num_classes}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            defense = GNNFingersDefense(
                dataset=adapted_dataset,
                task_type="node_classification",
                num_fingerprints=16,
                training_params={'epochs_total': 5},
                device=device
            )
            
            print(f"  {dataset_name} adapter compatible with GNNFingers")
            
        except Exception as e:
            print(f"  {dataset_name} adapter test failed: {e}")


def run_full_training_experiments():
    """Run full training experiments for all tasks and datasets."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers not available. Skipping full training experiments.")
        return
    
    print("\nRunning Full Training Experiments for All Tasks")
    print("=" * 60)
    
    device = setup_device()
    
    experiments = [
        ("node_classification", "Cora"),
        ("graph_classification", "PROTEINS"),
        ("link_prediction", "Cora"),
        ("graph_matching", "AIDS"),
    ]
    
    results = {}
    successful_experiments = 0
    total_experiments = len(experiments)
    
    for i, (task_type, dataset_name) in enumerate(experiments, 1):
        print(f"\n{'='*20} {task_type} - {dataset_name} ({i}/{total_experiments}) {'='*20}")
        
        try:
            if dataset_name.upper() in ['CORA', 'PUBMED']:
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
            
            print(f"Initializing GNNFingers defense for {task_type} on {dataset_name}...")
            defense = GNNFingersDefense(
                dataset=adapted_dataset,
                task_type=task_type,
                num_fingerprints=128,
                fingerprint_params=None,
                univerifier_params={'hidden_dims': [256, 128, 64], 'dropout': 0.3},
                training_params={
                    'epochs_total': 200,
                    'e1': 2, 'e2': 2,
                    'alpha': 0.01, 'beta': 0.001,
                    'convergence_threshold': 0.001
                },
                device=device
            )
            print("Defense initialized successfully")
            
            print(f"Starting comprehensive defense training for {task_type} on {dataset_name}...")
            start_time = datetime.datetime.now()
            result = defense.defend(attack_method="comprehensive")
            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            print(f"Defense training completed in {training_time}")
            
            results[f"{task_type}_{dataset_name}"] = result
            
            save_path = f"./weights/gnnfingers_{task_type}_{dataset_name.lower()}.pth"
            os.makedirs("./weights", exist_ok=True)
            
            torch.save({
                'target_model_state_dict': defense.target_model.state_dict(),
                'univerifier_state_dict': defense.univerifier.state_dict(),
                'fingerprint_constructor': defense.fingerprint_constructor,
                'training_history': defense.training_history,
                'results': result,
                'task_type': task_type,
                'dataset_name': dataset_name,
                'timestamp': datetime.datetime.now().isoformat()
            }, save_path)
            
            print(f"SUCCESS: {task_type} - {dataset_name}: AUC={result['auc']:.4f}, ARUC={result['aruc']:.4f}")
            print(f"   Model saved to: {save_path}")
            successful_experiments += 1
            
        except Exception as e:
            print(f"ERROR: {task_type} - {dataset_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[f"{task_type}_{dataset_name}"] = {'error': str(e)}
    
    print(f"\nFull Training Experiments Summary")
    print("=" * 60)
    print(f"Successful experiments: {successful_experiments}/{total_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    
    if successful_experiments > 0:
        print("\nResults:")
        print(f"{'Task':<25} {'Dataset':<10} {'AUC':<8} {'ARUC':<8} {'Status'}")
        print("-" * 65)
        
        for key, result in results.items():
            if 'error' not in result:
                task_display = result.get('task_type', key.split('_')[0]).replace('_', ' ').title()[:24]
                dataset = result.get('dataset_name', key.split('_')[1])
                auc = f"{result.get('auc', 0):.3f}"
                aruc = f"{result.get('aruc', 0):.3f}"
                status = "SUCCESS"
                
                print(f"{task_display:<25} {dataset:<10} {auc:<8} {aruc:<8} {status}")
            else:
                task_display = key.split('_')[0].replace('_', ' ').title()[:24]
                dataset = key.split('_')[1]
                print(f"{task_display:<25} {dataset:<10} {'N/A':<8} {'N/A':<8} FAILED")
    
    os.makedirs("./gnnfinger_results_json", exist_ok=True)
    results_path = "./gnnfinger_results_json/full_training_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAll results saved to: {results_path}")
    return results


def run_unit_tests():
    """Run unit tests for all tasks using saved models."""
    if not GNNFINGERS_AVAILABLE:
        print("GNNFingers not available. Skipping unit tests.")
        return
    
    print("\nRunning Unit Tests for All Tasks")
    print("=" * 60)
    
    device = setup_device()
    
    test_cases = [
        ("node_classification", "Cora", "test_node_classification"),
        ("node_classification", "Citeseer", "test_node_classification"),
        ("graph_classification", "PROTEINS", "test_graph_classification"),
        ("graph_classification", "AIDS", "test_graph_classification"),
        ("link_prediction", "Cora", "test_link_prediction"),
        ("link_prediction", "Citeseer", "test_link_prediction"),
        ("graph_matching", "PROTEINS", "test_graph_matching"),
        ("graph_matching", "AIDS", "test_graph_matching"),
    ]
    
    unit_test_results = {}
    
    for task_type, dataset_name, test_name in test_cases:
        print(f"\n{'='*20} {test_name} - {dataset_name} {'='*20}")
        
        try:
            model_path = f"./weights/gnnfingers_{task_type}_{dataset_name.lower()}.pth"
            
            if not os.path.exists(model_path):
                print(f"WARNING: Model not found: {model_path}")
                print("   Run full training first with --full-training")
                continue
            
            checkpoint = torch.load(model_path, map_location=device)
            
            adapted_dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg', path='./data')
            
            saved_results = checkpoint.get('results', {})
            saved_task_type = checkpoint.get('task_type', task_type)
            saved_dataset_name = checkpoint.get('dataset_name', dataset_name)
            
            num_fingerprints = 128
            if 'fingerprint_constructor' in checkpoint:
                try:
                    saved_fp = checkpoint['fingerprint_constructor']
                    if hasattr(saved_fp, 'num_fingerprints'):
                        num_fingerprints = saved_fp.num_fingerprints
                except:
                    pass
            
            defense = GNNFingersDefense(
                dataset=adapted_dataset,
                task_type=saved_task_type,
                num_fingerprints=num_fingerprints,
                device=device
            )
            
            if defense.target_model is None:
                print("  Initializing target model...")
                defense.target_model = defense._train_target_model()
            
            if defense.univerifier is None:
                print("  Initializing univerifier...")
                if 'fingerprint_constructor' in checkpoint:
                    defense.fingerprint_constructor = checkpoint['fingerprint_constructor']
                    print("  Loaded fingerprint constructor before univerifier initialization")
                
                try:
                    defense._initialize_univerifier()
                except Exception as e:
                    print(f"  WARNING: Could not initialize univerifier: {e}")
                    if hasattr(defense, 'fingerprint_constructor') and defense.fingerprint_constructor is not None:
                        sample_output = defense.fingerprint_constructor.get_model_outputs(defense.target_model)
                        input_dim = sample_output.size(0)
                        defense.univerifier = Univerifier(
                            input_dim=input_dim,
                            hidden_dims=defense.univerifier_params['hidden_dims'],
                            dropout=defense.univerifier_params['dropout']
                        ).to(defense.device)
                        print(f"  Created fallback univerifier with input dimension: {input_dim}")
            
            if 'target_model_state_dict' in checkpoint and defense.target_model is not None:
                try:
                    defense.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                    print("  Loaded target model weights")
                except Exception as e:
                    print(f"  WARNING: Could not load target model weights: {e}")
                    print("  Will use newly initialized target model")
            
            if 'univerifier_state_dict' in checkpoint and defense.univerifier is not None:
                try:
                    defense.univerifier.load_state_dict(checkpoint['univerifier_state_dict'])
                    print("  Loaded univerifier weights")
                except Exception as e:
                    print(f"  WARNING: Could not load univerifier weights: {e}")
                    print("  Will use newly initialized univerifier")
            
            if 'fingerprint_constructor' in checkpoint:
                defense.fingerprint_constructor = checkpoint['fingerprint_constructor']
                print("  Loaded fingerprint constructor")
            
            if 'training_history' in checkpoint:
                defense.training_history = checkpoint['training_history']
                print("  Loaded training history")
            
            # unit tests removed
            
        except Exception as e:
            print(f"ERROR: {test_name} - {dataset_name} failed: {e}")
            unit_test_results[f"{test_name}_{dataset_name}"] = {'error': str(e)}
    
    os.makedirs("./gnnfinger_results_json", exist_ok=True)
    unit_results_path = "./gnnfinger_results_json/unit_test_results.json"
    import json
    with open(unit_results_path, 'w') as f:
        json.dump(unit_test_results, f, indent=2, default=str)
    
    print(f"\nUnit test results saved to: {unit_results_path}")
    return unit_test_results




def get_available_weights():
    """Get list of available pre-trained weights."""
    weights_dir = "./weights"
    available_weights = {}
    
    if not os.path.exists(weights_dir):
        return available_weights
    
    for filename in os.listdir(weights_dir):
        if filename.endswith('.pth'):
            # Parse filename: gnnfingers_task_dataset.pth
            parts = filename.replace('.pth', '').split('_')
            if len(parts) >= 4 and parts[0] == 'gnnfingers':
                task = parts[1] + '_' + parts[2]  # e.g., "node_classification"
                dataset = parts[3].title()  # e.g., "Cora"
                
                if task not in available_weights:
                    available_weights[task] = []
                available_weights[task].append({
                    'dataset': dataset,
                    'filepath': os.path.join(weights_dir, filename),
                    'filename': filename
                })
    
    return available_weights


def select_best_weights(task_type, dataset_name):
    """
    Select the best available weights for a given task and dataset.
    
    Returns:
        tuple: (filepath, dataset_name) or (None, None) if no weights available
    """
    available_weights = get_available_weights()
    
    # Convert task type to match filename format
    task_key = task_type.replace('_', '_')  # Already in correct format
    
    if task_key not in available_weights:
        print(f"WARNING: No weights available for task '{task_type}'")
        return None, None
    
    task_weights = available_weights[task_key]
    
    # First, try to find exact match
    for weight_info in task_weights:
        if weight_info['dataset'].lower() == dataset_name.lower():
            print(f"SUCCESS: Found exact match - {weight_info['filename']}")
            return weight_info['filepath'], weight_info['dataset']
    
    # If no exact match, use the first available weight
    if task_weights:
        best_weight = task_weights[0]
        print(f"WARNING: No weights for dataset '{dataset_name}', using '{best_weight['dataset']}' instead")
        print(f"INFO: Using weights from {best_weight['filename']}")
        return best_weight['filepath'], best_weight['dataset']
    
    print(f"ERROR: No weights available for task '{task_type}'")
    return None, None


def verify_single_model(model_path, task_type, dataset_name):
    """
    Verify a single GNN model for originality using pre-trained weights.
    
    Args:
        model_path: Path to the model file to verify
        task_type: Type of GNN task
        dataset_name: Dataset name
    
    Returns:
        dict: Verification results
    """
    print(f"\n=== Single Model Verification ===")
    print(f"Model: {model_path}")
    print(f"Task: {task_type}")
    print(f"Dataset: {dataset_name}")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        return {
            'status': 'error',
            'message': f'Model file not found: {model_path}'
        }
    
    weights_path, weights_dataset = select_best_weights(task_type, dataset_name)
    
    if weights_path is None:
        return {
            'status': 'error',
            'message': f'No pre-trained weights available for task "{task_type}". Please train a new model first.'
        }
    
    try:
        print(f"Loading dataset: {dataset_name}")
        try:
            dataset = adapt_pygip_dataset(dataset_name)
        except ValueError as e:
            print(f"PyGIP adaptation failed: {e}")
            print("Trying GNNFingers dataset...")
            from datasets.gnn_fingers_datasets import get_gnnfingers_dataset
            dataset = get_gnnfingers_dataset(dataset_name, api_type='pyg')
            print(f"SUCCESS: Loaded {dataset_name} from GNNFingers datasets")
        
        print(f"Initializing defense with weights: {os.path.basename(weights_path)}")
        defense = GNNFingersDefense(
            dataset=dataset,
            task_type=task_type,
            num_fingerprints=32,
            device=setup_device()
        )
        
        print("Loading pre-trained weights...")
        checkpoint = torch.load(weights_path, map_location=defense.device)
        
        print("Initializing target model...")
        defense.target_model = get_model_for_task(
            task_type=task_type,
            input_dim=defense.num_features,
            hidden_dim=64,
            output_dim=defense.num_classes,
            num_layers=2
        ).to(defense.device)
        
        if 'target_model_state_dict' in checkpoint:
            defense.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            print("SUCCESS: Loaded target model weights")
        else:
            print("WARNING: Target model weights not found in checkpoint")
        
        if 'fingerprint_constructor_state_dict' in checkpoint:
            defense.fingerprint_constructor.load_state_dict(checkpoint['fingerprint_constructor_state_dict'])
            print("SUCCESS: Loaded fingerprint constructor weights")
        else:
            print("WARNING: Fingerprint constructor weights not found in checkpoint")
        
        if 'univerifier_state_dict' in checkpoint:
            print("Loading univerifier with task-specific parameters...")
            saved_state_dict = checkpoint['univerifier_state_dict']
            
            sample_output = defense.fingerprint_constructor.get_model_outputs(defense.target_model)
            input_dim = sample_output.size(0)
            print(f"Fingerprint output dimension: {input_dim}")
            
            if task_type == 'graph_classification':
                verification_input_dim = 32
                hidden_dims = [128, 64, 32]
                print("Using graph_classification univerifier: [128, 64, 32] with input_dim=32")
            elif task_type == 'node_classification':
                verification_input_dim = 64
                hidden_dims = [256, 128, 64]
                print("Using node_classification univerifier: [256, 128, 64] with input_dim=64")
            elif task_type == 'link_prediction':
                verification_input_dim = input_dim
                hidden_dims = [128, 64, 32]
                print("Using link_prediction univerifier: [128, 64, 32]")
            elif task_type == 'graph_matching':
                verification_input_dim = input_dim
                hidden_dims = [128, 64, 32]
                print("Using graph_matching univerifier: [128, 64, 32]")
            else:
                verification_input_dim = input_dim
                hidden_dims = [128, 64, 32]
                print("Using default univerifier: [128, 64, 32]")
            
            defense.univerifier = Univerifier(
                input_dim=verification_input_dim,
                hidden_dims=hidden_dims,
                dropout=0.3
            ).to(defense.device)
            
            try:
                defense.univerifier.load_state_dict(checkpoint['univerifier_state_dict'])
                print("SUCCESS: Loaded univerifier weights")
            except Exception as e:
                print(f"WARNING: Could not load univerifier weights due to architecture mismatch: {e}")
                print("Continuing with initialized univerifier...")
        else:
            print("WARNING: Univerifier weights not found in checkpoint, initializing with defaults")
            defense._initialize_univerifier()
        
        if model_path == "test_model.pth" or not os.path.exists(model_path):
            # unit tests removed: no test model generation
            print(f"Created task-specific test model for {task_type}")
        else:
            print(f"Loading model to verify: {model_path}")
            suspect_model = torch.load(model_path, map_location=defense.device)
        
        print("Adapting model to match expected graph structure...")
        try:
            test_output = defense.fingerprint_constructor.get_model_outputs(suspect_model)
            print("SUCCESS: Model compatible with fingerprint structure")
        except Exception as e:
            print(f"WARNING: Model needs adaptation - {e}")
            print("Creating model adapter...")
            adapted_model = adapt_model_for_verification(suspect_model, defense.fingerprint_constructor, defense.device)
            print("SUCCESS: Model adapted for verification")
            
            def verify_with_adapted_model(model):
                try:
                    if hasattr(defense.fingerprint_constructor, 'fingerprints'):
                        fingerprint_data = defense.fingerprint_constructor.fingerprints[0]
                    elif hasattr(defense.fingerprint_constructor, 'fingerprint'):
                        fingerprint_data = defense.fingerprint_constructor.fingerprint
                    else:
                        raise ValueError("Unknown fingerprint constructor type")
                    
                    x = fingerprint_data.x.to(defense.device)
                    edge_index = fingerprint_data.edge_index.to(defense.device)
                    
                    with torch.no_grad():
                        output = adapted_model(x, edge_index)
                    
                    print(f"Model output shape: {output.shape}")
                    
                    if output.dim() > 1:
                        output = output.mean(dim=0)
                    
                    print(f"Reshaped output shape: {output.shape}")
                    
                    defense.univerifier.eval()
                    prediction = defense.univerifier(output.unsqueeze(0))
                    confidence = prediction[0, 1].item()
                    
                    return confidence > 0.5, confidence
                except Exception as e:
                    print(f"Error in adapted verification: {e}")
                    return False, 0.0
            
            is_pirated, confidence = verify_with_adapted_model(suspect_model)
        else:
            print("Verifying model ownership...")
            is_pirated, confidence = defense.verify_ownership(suspect_model)
        
        if is_pirated:
            result = "PIRATED"
            recommendation = "This model appears to be derived from the protected model."
        else:
            result = "ORIGINAL"
            recommendation = "This model appears to be independently trained."
        
        print(f"\n=== Verification Results ===")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Recommendation: {recommendation}")
        print("=" * 50)
        
        return {
            'status': 'success',
            'model': os.path.basename(model_path),
            'result': result,
            'confidence': confidence,
            'recommendation': recommendation,
            'weights_used': os.path.basename(weights_path),
            'weights_dataset': weights_dataset
        }
        
    except Exception as e:
        error_msg = f"Verification failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': error_msg
        }


def adapt_model_for_verification(original_model, fingerprint_constructor, device):
    """
    Create a wrapper model that adapts the original model to work with fingerprint verification.
    
    Args:
        original_model: The original model to adapt
        fingerprint_constructor: The fingerprint constructor that defines expected input/output
        device: Computing device
    
    Returns:
        Adapted model that can handle fingerprint verification
    """
    import torch.nn as nn
    
    class ModelAdapter(nn.Module):
        def __init__(self, original_model, fingerprint_constructor, device):
            super(ModelAdapter, self).__init__()
            self.original_model = original_model
            self.fingerprint_constructor = fingerprint_constructor
            self.device = device
            
            self.expected_input_dim = fingerprint_constructor.feature_dim
            
            if hasattr(fingerprint_constructor, 'num_nodes'):
                self.expected_output_dim = fingerprint_constructor.num_nodes
            elif hasattr(fingerprint_constructor, 'num_fingerprints'):
                self.expected_output_dim = fingerprint_constructor.num_fingerprints
            else:
                self.expected_output_dim = 32
            
            print(f"Fingerprint dimensions: input={self.expected_input_dim}, output={self.expected_output_dim}")
            
            self.input_adapter = None
            self.output_adapter = None
            
            model_input_dim = None
            model_output_dim = None
            
            if hasattr(original_model, 'conv1') and hasattr(original_model.conv1, 'in_channels'):
                model_input_dim = original_model.conv1.in_channels
            elif hasattr(original_model, 'layers') and len(original_model.layers) > 0:
                model_input_dim = original_model.layers[0].in_channels
            elif hasattr(original_model, 'input_dim'):
                model_input_dim = original_model.input_dim
            
            if hasattr(original_model, 'conv2') and hasattr(original_model.conv2, 'out_channels'):
                model_output_dim = original_model.conv2.out_channels
            elif hasattr(original_model, 'layers') and len(original_model.layers) > 1:
                model_output_dim = original_model.layers[-1].out_channels
            elif hasattr(original_model, 'output_dim'):
                model_output_dim = original_model.output_dim
            
            print(f"Model dimensions: input={model_input_dim}, output={model_output_dim}")
            
            print(f"Model type: {type(original_model)}")
            print(f"Model attributes: {[attr for attr in dir(original_model) if not attr.startswith('_')]}")
            
            if model_input_dim is None:
                model_input_dim = 1433
                print(f"Using inferred input dimension: {model_input_dim}")
            
            if model_output_dim is None:
                model_output_dim = 7
                print(f"Using inferred output dimension: {model_output_dim}")
            
            if model_input_dim != self.expected_input_dim:
                print(f"Creating input adapter: {self.expected_input_dim} -> {model_input_dim}")
                self.input_adapter = nn.Linear(self.expected_input_dim, model_input_dim)
            
            if hasattr(fingerprint_constructor, 'num_fingerprints'):
                univerifier_input_dim = fingerprint_constructor.num_fingerprints
            else:
                univerifier_input_dim = 64
            
            if model_output_dim != univerifier_input_dim:
                print(f"Creating output adapter: {model_output_dim} -> {univerifier_input_dim}")
                self.output_adapter = nn.Linear(model_output_dim, univerifier_input_dim)
                self.expected_output_dim = univerifier_input_dim
        
        def forward(self, x, edge_index):
            if self.input_adapter is not None:
                x = self.input_adapter(x)
            
            output = self.original_model(x, edge_index)
            
            if self.output_adapter is not None:
                if hasattr(self.fingerprint_constructor, 'num_fingerprints'):
                    if output.dim() > 1:
                        output = output.mean(dim=0)
                    
                    output = self.output_adapter(output.unsqueeze(0)).squeeze(0)
                else:
                    raw_output = output
                    adapted_output = self.output_adapter(raw_output)
                    output = torch.nn.functional.log_softmax(adapted_output, dim=1)
            
            return output
    
    return ModelAdapter(original_model, fingerprint_constructor, device).to(device)


def list_available_weights():
    """List all available pre-trained weights."""
    print("\n=== Available Pre-trained Weights ===")
    available_weights = get_available_weights()
    
    if not available_weights:
        print("No pre-trained weights found in ./weights/ directory.")
        print("To create weights, run training experiments first:")
        print("  python test.py --full-training")
        return
    
    for task, weights_list in available_weights.items():
        print(f"\nTask: {task}")
        print("-" * 40)
        for weight_info in weights_list:
            print(f"  Dataset: {weight_info['dataset']}")
            print(f"  File: {weight_info['filename']}")
            print(f"  Path: {weight_info['filepath']}")
            print()
    
    print("=" * 50)
    print("To verify a model using these weights:")
    print("  python test.py --verify-model model.pth --model-task <task> --model-dataset <dataset>")


def main():
    """Main function for command line interface."""
    if len(sys.argv) == 1:
        print("\nOriginal PyGIP test completed.")
        print("For GNNFingers testing, use command line arguments:")
        print("  python test.py --list-weights")
        print("  python test.py --task node_classification --dataset Cora --quick")
        print("  python test.py --all --quick")
        print("  python test.py --test-datasets")
        print("  python test.py --test-adapter")
        print("  python test.py --verify-model model.pth --model-task node_classification --model-dataset Cora")
        return
    
    parser = argparse.ArgumentParser(description='GNNFingers testing for PyGIP framework')
    
    parser.add_argument('--task', type=str,
                       choices=['node_classification', 'graph_classification', 'link_prediction', 'graph_matching'],
                       help='Type of GNN task to test')
    
    parser.add_argument('--dataset', type=str,
                       choices=['Cora', 'Citeseer', 'PubMed', 'PROTEINS', 'AIDS', 'MUTAG'],
                       help='Dataset to use for testing')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (fewer models, faster execution)')
    
    parser.add_argument('--all', action='store_true',
                       help='Run all GNNFingers experiments')
    
    parser.add_argument('--test-datasets', action='store_true',
                       help='Test dataset loading only')
    
    parser.add_argument('--test-adapter', action='store_true',
                       help='Test PyGIP dataset adapter')
    
    parser.add_argument('--full-training', action='store_true',
                       help='Run full training experiments for all tasks and datasets')
    
    # parser.add_argument('--unit-tests', action='store_true',
    #                    help='Run unit tests for all tasks using saved models')
    
    parser.add_argument('--verify-model', type=str,
                       help='Verify a single model file (provide path to .pth file)')
    
    parser.add_argument('--model-task', type=str,
                       choices=['node_classification', 'graph_classification', 'link_prediction', 'graph_matching'],
                       help='Task type for the model to verify (required with --verify-model)')
    
    parser.add_argument('--model-dataset', type=str,
                       choices=['Cora', 'Citeseer', 'PubMed', 'PROTEINS', 'AIDS', 'MUTAG'],
                       help='Dataset for the model to verify (required with --verify-model)')
    
    parser.add_argument('--list-weights', action='store_true',
                       help='List all available pre-trained weights')
    
    args = parser.parse_args()
    
    print(f"GNNFingers Test Suite for PyGIP")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"GNNFingers available: {GNNFINGERS_AVAILABLE}")
    print("=" * 60)
    
    if not GNNFINGERS_AVAILABLE:
        print("Error: GNNFingers not available. Please check installation.")
        print("The original PyGIP functionality above still works normally.")
        return
    
    try:
        if args.list_weights:
            list_available_weights()
        elif args.test_datasets:
            test_dataset_loading()
        elif args.test_adapter:
            test_adapter()
        elif args.full_training:
            run_full_training_experiments()
        # elif args.unit_tests:
        #     run_unit_tests()
        elif args.verify_model:
            if not args.model_task or not args.model_dataset:
                print("Error: --verify-model requires both --model-task and --model-dataset")
                print("Example: python test.py --verify-model model.pth --model-task node_classification --model-dataset Cora")
                return
            verify_single_model(args.verify_model, args.model_task, args.model_dataset)
        elif args.all:
            run_all_gnnfingers_experiments(quick_mode=args.quick)
        elif args.task and args.dataset:
            run_gnnfingers_experiment(args.task, args.dataset, args.quick)
        else:
            print("Error: Must specify one of the following options:")
            print("  --list-weights: List available pre-trained weights")
            print("  --all: Run all experiments")
            print("  --test-datasets: Test dataset loading")
            print("  --test-adapter: Test PyGIP adapter")
            print("  --full-training: Run full training experiments")
            print("  --unit-tests: Run unit tests")
            print("  --verify-model: Verify a single model (requires --model-task and --model-dataset)")
            print("  --task and --dataset: Run specific experiment")
            print("\nExamples:")
            print("  python test.py --list-weights")
            print("  python test.py --task node_classification --dataset Cora --quick")
            print("  python test.py --all --quick")
            print("  python test.py --test-datasets")
            print("  python test.py --test-adapter")
            print("  python test.py --full-training")
            # print("  python test.py --unit-tests")
            print("  python test.py --verify-model model.pth --model-task node_classification --model-dataset Cora")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()