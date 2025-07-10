"""
Comprehensive example demonstrating various Grove attack and defense scenarios.
This file serves as a detailed guide for using the Grove framework.
"""
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.attack.grove_attack import GroveAttack
from models.defense.grove_defense import GroveDefense
from datasets import ACM, CoauthorCS, PubMed, CitationFullDBLP, Citeseer, AmazonPhoto

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTACK_NODE_FRACTION = 0.4
NUM_EPOCHS = 200
LEARNING_RATE = 0.001

def run_attack_scenario(dataset_name, dataset_obj, attack_config, method_name):
    """Helper function to run an attack scenario and print results."""
    print(f"\n=== Running {method_name} Attack on {dataset_name} ===")
    attack_instance = GroveAttack(
        dataset=dataset_obj,
        device=DEVICE,
        attack_node_fraction=ATTACK_NODE_FRACTION,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        **attack_config
    )
    results = attack_instance.attack(attack_method=method_name)
    print(f"  Fidelity: {results.get('fidelity', 'N/A'):.4f}")
    print(f"  Target Accuracy: {results.get('target_accuracy', 'N/A'):.4f}")
    print(f"  Surrogate Accuracy: {results.get('surrogate_accuracy', 'N/A'):.4f}")
    print(f"  Accuracy Gap: {results.get('accuracy_gap', 'N/A'):.4f}")
    if 'embedding_cosine_similarity' in results:
        print(f"  Embedding Cosine Similarity: {results['embedding_cosine_similarity']:.4f}")
    if 'embedding_l2_distance' in results:
        print(f"  Embedding L2 Distance: {results['embedding_l2_distance']:.4f}")
    return results

def run_defense_scenario(dataset_name, dataset_obj):
    """Helper function to run a defense scenario and print results."""
    print(f"\n=== Running Defense on {dataset_name} ===")
    defense_instance = GroveDefense(dataset_obj, attack_node_fraction=ATTACK_NODE_FRACTION, device=DEVICE, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    defense_results = defense_instance.defend()
    print(f"  Defense Results: {defense_results}")
    return defense_results

def main():
    print(f"Running Grove example on device: {DEVICE}")
    
    # --- Load Datasets ---
    print("\nLoading datasets...")
    datasets = {
        "ACM": ACM(api_type='torch_geometric', path='./downloads/'),
        "CoauthorCS": CoauthorCS(api_type='torch_geometric', path='./downloads/'),
        "PubMed": PubMed(api_type='torch_geometric', path='./downloads/'),
        "CitationFullDBLP": CitationFullDBLP(api_type='torch_geometric', path='./downloads/'),
        "Citeseer": Citeseer(api_type='torch_geometric', path='./downloads/'),
        "AmazonPhoto": AmazonPhoto(api_type='torch_geometric', path='./downloads/'),
    }

    for name, ds in datasets.items():
        print(f"  {name} loaded: {ds.node_number} nodes, {ds.feature_number} features, {ds.label_number} classes")

    # --- Attack Scenarios ---

    # Scenario 1: Basic Type I Attack (Embedding Recovery, GAT, Original Structure)
    run_attack_scenario(
        "ACM", datasets["ACM"],
        {"attack_type": "type_i", "surrogate_architecture": "gat", "recovery_from": "embedding", "structure": "original"},
        "simple"
    )

    # Scenario 2: Type II Attack (Prediction Recovery, GIN, IDGL Structure)
    run_attack_scenario(
        "CoauthorCS", datasets["CoauthorCS"],
        {"attack_type": "type_ii", "surrogate_architecture": "gin", "recovery_from": "prediction", "structure": "idgl"},
        "fine_tuning"
    )

    # Scenario 3: Double Extraction Attack (Embedding Recovery, GraphSAGE, KNN Structure)
    run_attack_scenario(
        "PubMed", datasets["PubMed"],
        {"attack_type": "type_ii", "surrogate_architecture": "graphsage", "recovery_from": "embedding", "structure": "knn"},
        "double_extraction"
    )

    # Scenario 4: Distribution Shift Attack (Prediction Recovery, GAT, Random Structure)
    run_attack_scenario(
        "CitationFullDBLP", datasets["CitationFullDBLP"],
        {"attack_type": "type_ii", "surrogate_architecture": "gat", "recovery_from": "prediction", "structure": "random"},
        "distribution_shift"
    )

    # Scenario 5: Pruned Attack (Embedding Recovery, GIN, Original Structure)
    run_attack_scenario(
        "Citeseer", datasets["Citeseer"],
        {"attack_type": "type_i", "surrogate_architecture": "gin", "recovery_from": "embedding", "structure": "original"},
        "pruned"
    )

    # --- Defense Scenarios ---

    # Defense on ACM dataset
    run_defense_scenario("ACM", datasets["ACM"])

    # Defense on CoauthorCS dataset
    run_defense_scenario("CoauthorCS", datasets["CoauthorCS"])

    print("\n=== All Grove Example Scenarios Completed ===")

if __name__ == "__main__":
    main() 