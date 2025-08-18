"""
Utility functions and metrics for GNNFingers framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from typing import List, Dict, Tuple, Optional
import copy
import random

from models.defense.gnn_fingers_models import ModelObfuscator, get_model_for_task
from models.defense.gnn_fingers_protect import FingerprintConstructor


def calculate_aruc(robustness_scores: List[float], uniqueness_scores: List[float]) -> float:
    """
    Calculate Area Under Robustness-Uniqueness Curve (ARUC).
    
    Args:
        robustness_scores: List of robustness (TPR) scores
        uniqueness_scores: List of uniqueness (TNR) scores
    
    Returns:
        ARUC score
    """
    if len(robustness_scores) > 1 and len(uniqueness_scores) > 1:
        aruc = np.trapz(uniqueness_scores, robustness_scores)
        return abs(aruc)
    else:
        return 0.5


def plot_robustness_uniqueness_curve(results: Dict, title_suffix: str = "", 
                                    save_path: Optional[str] = None):
    """
    Plot Robustness-Uniqueness curve.
    
    Args:
        results: Results dictionary containing threshold_results
        title_suffix: Additional title text
        save_path: Path to save the plot
    """
    if not results.get('threshold_results'):
        print("No results to plot")
        return

    thresholds = [r['threshold'] for r in results['threshold_results']]
    robustness = [r['robustness'] for r in results['threshold_results']]
    uniqueness = [r['uniqueness'] for r in results['threshold_results']]

    plt.figure(figsize=(12, 5))

    # Plot 1: Robustness & Uniqueness vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, robustness, 'b-o', label='Robustness (TPR)', linewidth=2, markersize=6)
    plt.plot(thresholds, uniqueness, 'r-s', label='Uniqueness (TNR)', linewidth=2, markersize=6)
    plt.xlabel('Threshold lambda', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Robustness & Uniqueness vs Threshold\n{title_suffix}', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([min(thresholds) - 0.05, max(thresholds) + 0.05])
    plt.ylim([0, 1.05])

    # Plot 2: Robustness-Uniqueness Curve
    plt.subplot(1, 2, 2)
    plt.plot(robustness, uniqueness, 'g-^', linewidth=2, markersize=6)
    plt.xlabel('Robustness (True Positive Rate)', fontsize=12)
    plt.ylabel('Uniqueness (True Negative Rate)', fontsize=12)
    plt.title(f'Robustness-Uniqueness Curve\nARUC = {results["aruc"]:.3f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def evaluate_fingerprint_verification(univerifier: nn.Module, 
                                    fingerprint_constructor: FingerprintConstructor,
                                    positive_models: List[nn.Module],
                                    negative_models: List[nn.Module],
                                    device: torch.device,
                                    thresholds: Optional[List[float]] = None) -> Dict:
    """
    Evaluate fingerprint verification performance across multiple thresholds.
    
    Args:
        univerifier: Trained univerifier model
        fingerprint_constructor: Fingerprint constructor
        positive_models: List of positive (pirated) models
        negative_models: List of negative (independent) models
        device: Computing device
        thresholds: List of thresholds to evaluate
    
    Returns:
        Dictionary containing evaluation results
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    print("Evaluating fingerprint verification...")
    
    all_confidences = []
    true_labels = []

    # Evaluate positive models
    print("Evaluating positive models...")
    for i, pos_model in enumerate(positive_models):
        try:
            confidence = verify_single_model(
                univerifier, fingerprint_constructor, pos_model, device
            )
            all_confidences.append(confidence)
            true_labels.append(1)
            print(f"  Positive model {i+1}: confidence = {confidence:.3f}")
        except Exception as e:
            print(f"  Error evaluating positive model {i+1}: {e}")
            continue

    # Evaluate negative models
    print("Evaluating negative models...")
    for i, neg_model in enumerate(negative_models):
        try:
            confidence = verify_single_model(
                univerifier, fingerprint_constructor, neg_model, device
            )
            all_confidences.append(confidence)
            true_labels.append(0)
            print(f"  Negative model {i+1}: confidence = {confidence:.3f}")
        except Exception as e:
            print(f"  Error evaluating negative model {i+1}: {e}")
            continue

    if len(all_confidences) == 0:
        return {'auc': 0.5, 'aruc': 0.5, 'threshold_results': []}

    # Calculate AUC
    if len(set(true_labels)) > 1:
        auc_score = roc_auc_score(true_labels, all_confidences)
    else:
        auc_score = 0.5

    # Calculate metrics for each threshold
    robustness_scores = []
    uniqueness_scores = []
    threshold_results = []

    for threshold in thresholds:
        tp = sum(1 for i, conf in enumerate(all_confidences)
                if true_labels[i] == 1 and conf > threshold)
        fp = sum(1 for i, conf in enumerate(all_confidences)
                if true_labels[i] == 0 and conf > threshold)
        tn = sum(1 for i, conf in enumerate(all_confidences)
                if true_labels[i] == 0 and conf <= threshold)
        fn = sum(1 for i, conf in enumerate(all_confidences)
                if true_labels[i] == 1 and conf <= threshold)

        robustness = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
        uniqueness = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        accuracy = (tp + tn) / len(all_confidences)

        robustness_scores.append(robustness)
        uniqueness_scores.append(uniqueness)

        threshold_results.append({
            'threshold': threshold,
            'robustness': robustness,
            'uniqueness': uniqueness,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })

    # Calculate ARUC
    aruc = calculate_aruc(robustness_scores, uniqueness_scores)

    results = {
        'auc': auc_score,
        'aruc': aruc,
        'threshold_results': threshold_results,
        'all_confidences': all_confidences,
        'true_labels': true_labels
    }

    return results


def verify_single_model(univerifier: nn.Module, fingerprint_constructor: FingerprintConstructor,
                       model: nn.Module, device: torch.device) -> float:
    """
    Verify ownership of a single model.
    
    Args:
        univerifier: Trained univerifier
        fingerprint_constructor: Fingerprint constructor
        model: Model to verify
        device: Computing device
    
    Returns:
        Confidence score (0-1)
    """
    try:
        model_outputs = fingerprint_constructor.get_model_outputs(model)
        
        # Dynamic shape handling for different task types
        if model_outputs.dim() == 2 and model_outputs.size(1) > 1:
            # For graph classification, link prediction, etc. - flatten the outputs
            model_outputs = model_outputs.flatten()
        elif model_outputs.dim() == 1:
            # Already 1D, keep as is
            pass
        else:
            # For other cases, ensure it's 1D
            model_outputs = model_outputs.view(-1)
        
        # Special handling for link prediction to improve discrimination
        # Normalize and scale the outputs to make differences more pronounced
        if hasattr(fingerprint_constructor, 'task_type') and fingerprint_constructor.task_type == "link_prediction":
            # Apply normalization and scaling for better discrimination
            model_outputs = (model_outputs - model_outputs.mean()) / (model_outputs.std() + 1e-8)
            model_outputs = model_outputs * 2.0  # Scale up differences
        
        univerifier.eval()
        with torch.no_grad():
            prediction = univerifier(model_outputs.unsqueeze(0))
            
            # Dynamic output handling for different univerifier architectures
            if prediction.dim() == 2 and prediction.size(1) >= 2:
                # Standard 2D output with multiple classes - use positive class probability
                confidence = prediction[0, 1].item()
            elif prediction.dim() == 2 and prediction.size(1) == 1:
                # 2D output with single class - use sigmoid for binary classification
                confidence = torch.sigmoid(prediction[0, 0]).item()
            elif prediction.dim() == 1:
                # 1D output - use sigmoid for binary classification
                confidence = torch.sigmoid(prediction[0]).item()
            else:
                # Fallback - assume it's a binary output
                confidence = torch.sigmoid(prediction).item()
        
        return confidence
    except Exception as e:
        print(f"Error in single model verification: {e}")
        return 0.0


def create_obfuscated_models(target_model: nn.Module, dataset, task_type: str, 
                           num_models: int, attack_method: str, 
                           device: torch.device) -> List[nn.Module]:
    """
    Create obfuscated versions of target model for testing.
    
    Args:
        target_model: Original model to obfuscate
        dataset: Dataset for training
        task_type: Type of GNN task
        num_models: Number of models to create
        attack_method: Attack method ("comprehensive", "fine_tuning", etc.)
        device: Computing device
    
    Returns:
        List of obfuscated models
    """
    print(f"Creating {num_models} obfuscated models using {attack_method}...")
    
    obfuscated_models = []
    
    if attack_method == "comprehensive":
        # Mix of all obfuscation techniques
        fine_tune_count = num_models // 4
        retrain_count = num_models // 4
        distill_count = num_models // 4
        prune_count = num_models - fine_tune_count - retrain_count - distill_count
        
        methods = [
            ("fine_tuning", fine_tune_count),
            ("partial_retraining", retrain_count),
            ("distillation", distill_count),
            ("pruning", prune_count)
        ]
    elif attack_method == "fine_tuning":
        methods = [("fine_tuning", num_models)]
    elif attack_method == "partial_retraining":
        methods = [("partial_retraining", num_models)]
    elif attack_method == "distillation":
        methods = [("distillation", num_models)]
    else:
        # Default to fine-tuning
        methods = [("fine_tuning", num_models)]
    
    for method, count in methods:
        for i in range(count):
            try:
                # Prepare task-specific training data handle
                if task_type == "graph_classification":
                    data_handle = dataset  # provides get_dataloader
                elif task_type == "graph_matching":
                    # Build a small set of pairs for obfuscation
                    try:
                        pairs = dataset.create_graph_pairs(num_pairs=400)
                    except Exception:
                        pairs = []
                    data_handle = pairs
                else:
                    data_handle = dataset.graph_data

                if method == "fine_tuning":
                    model = ModelObfuscator.fine_tune_model(
                        target_model, data_handle, task_type, epochs=20, device=device
                    )
                elif method == "partial_retraining":
                    model = ModelObfuscator.partial_retrain_model(
                        target_model, data_handle, task_type, 
                        layers_to_retrain=random.choice([1, 2]), epochs=20, device=device
                    )
                elif method == "distillation":
                    # Determine output dimension for tasks lacking explicit num_classes
                    out_dim = dataset.num_classes if hasattr(dataset, 'num_classes') and dataset.num_classes else (1 if task_type in ["link_prediction", "graph_matching"] else 2)
                    
                    # Use the same hidden dimension as the target model to avoid tensor shape mismatches
                    target_hidden_dim = target_model.convs[0].out_channels if hasattr(target_model, 'convs') and len(target_model.convs) > 0 else 64
                    
                    model = ModelObfuscator.distill_model(
                        target_model, data_handle, task_type,
                        dataset.num_features if hasattr(dataset, 'num_features') else target_model.convs[0].in_channels,
                        target_hidden_dim,  # Use target model's hidden dimension
                        out_dim, epochs=100, device=device
                    )
                elif method == "pruning":
                    model = ModelObfuscator.prune_model(
                        target_model, data_handle, task_type,
                        random.choice([0.1, 0.2, 0.3]), # Example pruning ratio
                        epochs=20, device=device
                    )
                else:
                    continue
                
                obfuscated_models.append(model)
                
                if (i + 1) % 10 == 0:
                    print(f"  {method}: {i + 1}/{count} completed")
                    
            except Exception as e:
                print(f"  Error creating {method} model {i+1}: {e}")
                continue
    
    print(f"Successfully created {len(obfuscated_models)} obfuscated models")
    return obfuscated_models


def calculate_model_similarity(model1: nn.Module, model2: nn.Module, 
                              fingerprint_constructor: FingerprintConstructor) -> float:
    """
    Calculate similarity between two models using fingerprints.
    
    Args:
        model1: First model
        model2: Second model
        fingerprint_constructor: Fingerprint constructor
    
    Returns:
        Similarity score (0-1)
    """
    try:
        output1 = fingerprint_constructor.get_model_outputs(model1)
        output2 = fingerprint_constructor.get_model_outputs(model2)
        
        # Cosine similarity
        similarity = F.cosine_similarity(output1.unsqueeze(0), output2.unsqueeze(0))
        return similarity.item()
    except Exception as e:
        print(f"Error calculating model similarity: {e}")
        return 0.0


def generate_defense_report(results: Dict, task_type: str, dataset_name: str) -> str:
    """
    Generate a comprehensive defense report.
    
    Args:
        results: Evaluation results dictionary
        task_type: Type of GNN task
        dataset_name: Name of dataset used
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("GNNFINGERS DEFENSE EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"Task Type: {task_type.replace('_', ' ').title()}")
    report.append(f"Dataset: {dataset_name}")
    report.append("")
    
    # Overall metrics
    report.append("OVERALL PERFORMANCE METRICS:")
    report.append(f"  - AUC Score: {results.get('auc', 0):.4f}")
    report.append(f"  - ARUC Score: {results.get('aruc', 0):.4f}")
    
    if results.get('threshold_results'):
        best_result = max(results['threshold_results'], key=lambda x: x['accuracy'])
        report.append(f"  - Best Verification Accuracy: {best_result['accuracy']:.4f} at threshold {best_result['threshold']:.2f}")
        report.append("")
        
        # Detailed threshold analysis
        report.append("THRESHOLD ANALYSIS:")
        report.append("Threshold | Robustness | Uniqueness | Accuracy | TP | FP | TN | FN")
        report.append("-" * 70)
        
        for result in results['threshold_results']:
            report.append(f"{result['threshold']:.2f}      | "
                         f"{result['robustness']:.3f}      | "
                         f"{result['uniqueness']:.3f}      | "
                         f"{result['accuracy']:.3f}    | "
                         f"{result['tp']:2d} | {result['fp']:2d} | "
                         f"{result['tn']:2d} | {result['fn']:2d}")
    
    report.append("")
    report.append("INTERPRETATION:")
    
    auc_score = results.get('auc', 0)
    if auc_score >= 0.9:
        report.append("  - Excellent fingerprinting performance")
    elif auc_score >= 0.8:
        report.append("  - Good fingerprinting performance") 
    elif auc_score >= 0.7:
        report.append("  - Moderate fingerprinting performance")
    else:
        report.append("  - Poor fingerprinting performance - needs improvement")
    
    aruc_score = results.get('aruc', 0)
    if aruc_score >= 0.8:
        report.append("  - High robustness-uniqueness balance")
    elif aruc_score >= 0.6:
        report.append("  - Moderate robustness-uniqueness balance")
    else:
        report.append("  - Low robustness-uniqueness balance")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def save_defense_results(results: Dict, task_type: str, dataset_name: str, 
                        save_path: str):
    """
    Save defense results to file.
    
    Args:
        results: Results dictionary
        task_type: Type of GNN task
        dataset_name: Dataset name
        save_path: Path to save results
    """
    import json
    import datetime
    
    save_data = {
        'task_type': task_type,
        'dataset_name': dataset_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'results': {
            'auc': results.get('auc', 0),
            'aruc': results.get('aruc', 0),
            'threshold_results': results.get('threshold_results', [])
        }
    }
    
    try:
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to: {save_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


class GNNFingersMetrics:
    """Class for computing various GNNFingers-specific metrics."""
    
    @staticmethod
    def fidelity_score(target_model: nn.Module, suspect_model: nn.Module,
                      test_data, device: torch.device) -> float:
        """
        Calculate fidelity score between target and suspect models.
        
        Args:
            target_model: Target model
            suspect_model: Suspect model
            test_data: Test data
            device: Computing device
        
        Returns:
            Fidelity score (0-1)
        """
        target_model.eval()
        suspect_model.eval()
        
        try:
            with torch.no_grad():
                if hasattr(test_data, 'x'):  # Node classification
                    target_pred = target_model(test_data.x.to(device), 
                                             test_data.edge_index.to(device))
                    suspect_pred = suspect_model(test_data.x.to(device), 
                                               test_data.edge_index.to(device))
                    
                    target_labels = target_pred.argmax(dim=1)
                    suspect_labels = suspect_pred.argmax(dim=1)
                    
                    fidelity = (target_labels == suspect_labels).float().mean()
                    return fidelity.item()
                else:
                    return 0.0
        except Exception as e:
            print(f"Error calculating fidelity: {e}")
            return 0.0
    
    @staticmethod
    def extraction_accuracy(target_model: nn.Module, extracted_model: nn.Module,
                          test_data, device: torch.device) -> float:
        """
        Calculate extraction accuracy of the extracted model.
        
        Args:
            target_model: Original target model
            extracted_model: Extracted model
            test_data: Test data
            device: Computing device
        
        Returns:
            Extraction accuracy (0-1)
        """
        extracted_model.eval()
        
        try:
            with torch.no_grad():
                if hasattr(test_data, 'test_mask'):  # Node classification
                    pred = extracted_model(test_data.x.to(device), 
                                         test_data.edge_index.to(device))
                    pred_labels = pred.argmax(dim=1)
                    
                    accuracy = (pred_labels[test_data.test_mask] == 
                              test_data.y[test_data.test_mask].to(device)).float().mean()
                    return accuracy.item()
                else:
                    return 0.0
        except Exception as e:
            print(f"Error calculating extraction accuracy: {e}")
            return 0.0


def validate_fingerprint_quality(fingerprint_constructor: FingerprintConstructor,
                                model: nn.Module, device: torch.device) -> Dict:
    """
    Validate the quality of generated fingerprints.
    
    Args:
        fingerprint_constructor: Fingerprint constructor to validate
        model: Model to test fingerprints on
        device: Computing device
    
    Returns:
        Dictionary containing quality metrics
    """
    try:
        # Test fingerprint consistency
        output1 = fingerprint_constructor.get_model_outputs(model)
        output2 = fingerprint_constructor.get_model_outputs(model)
        
        consistency = F.cosine_similarity(output1.unsqueeze(0), output2.unsqueeze(0)).item()
        
        # Test fingerprint distinctiveness
        if hasattr(fingerprint_constructor, 'fingerprints'):
            # Multiple fingerprints case
            num_fingerprints = len(fingerprint_constructor.fingerprints)
        else:
            num_fingerprints = 1
        
        # Test output variance
        output_std = output1.std().item()
        output_mean = output1.mean().item()
        
        return {
            'consistency': consistency,
            'num_fingerprints': num_fingerprints,
            'output_std': output_std,
            'output_mean': output_mean,
            'output_size': output1.size(0)
        }
    except Exception as e:
        print(f"Error validating fingerprint quality: {e}")
        return {
            'consistency': 0.0,
            'num_fingerprints': 0,
            'output_std': 0.0,
            'output_mean': 0.0,
            'output_size': 0
        }


def benchmark_gnnfingers_performance(defense_instance, test_models: List[nn.Module],
                                   test_labels: List[int], device: torch.device) -> Dict:
    """
    Benchmark GNNFingers performance against various attacks.
    
    Args:
        defense_instance: GNNFingers defense instance
        test_models: List of test models
        test_labels: List of labels (1 for pirated, 0 for independent)
        device: Computing device
    
    Returns:
        Comprehensive benchmark results
    """
    results = {
        'total_models': len(test_models),
        'pirated_models': sum(test_labels),
        'independent_models': len(test_labels) - sum(test_labels),
        'verification_results': []
    }
    
    print(f"Benchmarking {len(test_models)} models...")
    
    confidences = []
    predictions = []
    
    for i, (model, true_label) in enumerate(zip(test_models, test_labels)):
        try:
            is_pirated, confidence = defense_instance.verify_ownership(model)
            
            confidences.append(confidence)
            predictions.append(1 if is_pirated else 0)
            
            results['verification_results'].append({
                'model_id': i,
                'true_label': true_label,
                'predicted_label': 1 if is_pirated else 0,
                'confidence': confidence,
                'correct': (1 if is_pirated else 0) == true_label
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(test_models)} models processed")
                
        except Exception as e:
            print(f"  Error processing model {i}: {e}")
            continue
    
    # Calculate overall metrics
    if len(confidences) > 0 and len(set(test_labels)) > 1:
        auc_score = roc_auc_score(test_labels[:len(confidences)], confidences)
        accuracy = sum(r['correct'] for r in results['verification_results']) / len(results['verification_results'])
        
        results['overall_metrics'] = {
            'auc': auc_score,
            'accuracy': accuracy,
            'processed_models': len(confidences)
        }
    else:
        results['overall_metrics'] = {
            'auc': 0.5,
            'accuracy': 0.0,
            'processed_models': 0
        }
    
    return results


def create_synthetic_dataset_for_task(task_type: str, num_nodes: int = 1000, 
                                     num_features: int = 100, num_classes: int = 5,
                                     device: torch.device = torch.device('cpu')):
    """
    Create synthetic dataset for testing GNNFingers on different tasks.
    
    Args:
        task_type: Type of GNN task
        num_nodes: Number of nodes
        num_features: Number of node features
        num_classes: Number of classes
        device: Computing device
    
    Returns:
        Synthetic dataset compatible with PyGIP Dataset format
    """
    from torch_geometric.data import Data
    
    # Generate random graph
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))  # Random edges
    edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates
    
    x = torch.randn(num_nodes, num_features)
    
    if task_type in ["node_classification"]:
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # Create train/val/test masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:int(0.6 * num_nodes)] = True
        val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
        test_mask[int(0.8 * num_nodes):] = True
        
        data = Data(x=x, edge_index=edge_index, y=y, 
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    elif task_type in ["graph_classification", "graph_matching"]:
        # For graph-level tasks, create a single graph with graph-level label
        y = torch.randint(0, num_classes, (1,))
        data = Data(x=x, edge_index=edge_index, y=y)
    
    else:  # link_prediction
        y = torch.randint(0, 2, (edge_index.size(1),))  # Binary edge labels
        data = Data(x=x, edge_index=edge_index, y=y)
    
    return data.to(device)


def print_defense_summary(results: Dict, task_type: str, dataset_name: str):
    """
    Print a concise summary of defense results.
    
    Args:
        results: Results dictionary
        task_type: Type of GNN task  
        dataset_name: Dataset name
    """
    print(f"\nGNNFINGERS DEFENSE SUMMARY")
    print(f"{'='*50}")
    print(f"Task: {task_type.replace('_', ' ').title()}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    
    auc = results.get('auc', 0)
    aruc = results.get('aruc', 0)
    
    print(f"AUC Score: {auc:.4f}")
    print(f"ARUC Score: {aruc:.4f}")
    
    if results.get('threshold_results'):
        best_result = max(results['threshold_results'], key=lambda x: x['accuracy'])
        print(f"Best Accuracy: {best_result['accuracy']:.4f} (threshold: {best_result['threshold']:.2f})")
    
    # Performance assessment
    if auc >= 0.9:
        assessment = "EXCELLENT"
    elif auc >= 0.8:
        assessment = "GOOD"
    elif auc >= 0.7:
        assessment = "MODERATE"
    else:
        assessment = "NEEDS IMPROVEMENT"
    
    print(f"Performance: {assessment}")
    print(f"{'='*50}")