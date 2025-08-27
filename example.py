import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datasets
import dgl
import numpy as np
from models.defense.graph_pruning import GraphPruningDefense

CONFIG = {
    "datasets_to_run": ["Cora", "CiteSeer","PubMed"],
    # we can Implements and compares several edge pruning strategies as "random", "degree_low", "degree_high"
    "pruning_strategy": "degree_low",
    "initial_pruning_ratio": 0.05, # we start pruning from 5%
    "pruning_step": 0.02, #then we increase the pruning ratio by 2%
    "max_accuracy_drop": 0.03, # if the accuracy drops by more than 3%, we stop
}

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_summary_plot(all_results, max_drop_threshold, strategy):
    """Generates and saves a plot summarizing results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for dataset_name, results in all_results.items():
        baseline_acc = results[0.0]
        ratios = sorted(results.keys())
        accuracies = [results[r] for r in ratios]
        
        ratios_percent = [r * 100 for r in ratios]
        accuracies_percent = [acc * 100 for acc in accuracies]

        line, = ax.plot(ratios_percent, accuracies_percent, marker='o', linestyle='-', label=f'{dataset_name} Accuracy')
        ax.axhline(y=baseline_acc * 100, color=line.get_color(), linestyle='--', label=f'{dataset_name} Baseline ({baseline_acc*100:.2f}%)')

    ax.set_title(f'Effective Range of Graph Pruning (Strategy: {strategy.replace("_", " ").title()})', fontsize=16)
    ax.set_xlabel('Pruning Ratio (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_path = f"pruning_summary_{strategy}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nSummary plot saved to: {save_path}")


def main():
    """Main function to run the multi-dataset experiment."""
    set_seed(42)
    all_results_for_plot = {}
    datasets_to_run = CONFIG["datasets_to_run"]
    strategy = CONFIG["pruning_strategy"]
    num_total_steps = len(datasets_to_run) + 1

    for i, dataset_name in enumerate(datasets_to_run):
        step_num = i + 1
        print("\n" + "="*80)
        print(f"      STEP [{step_num}/{num_total_steps}]: Exp for {dataset_name.upper()} (Strategy: {strategy})")
        print("="*80)

        print(f"\n   - Sub-step [1/3]: Loading {dataset_name} and calculating baseline...")
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class()
        
        baseline_defense = GraphPruningDefense(dataset, pruning_ratio=0.0, pruning_strategy=strategy)
        baseline_results = baseline_defense.defend()
        baseline_accuracy = baseline_results['defended_accuracy']
        
        print(f"\n   - Sub-step [2/3]: Searching for optimal pruning ratio...")
        current_ratio = CONFIG["initial_pruning_ratio"]
        run_results = {0.0: baseline_accuracy}
        
        while True:
            defense = GraphPruningDefense(dataset, pruning_ratio=current_ratio, pruning_strategy=strategy)
            results_dict = defense.defend()
            current_accuracy = results_dict['defended_accuracy']
            run_results[current_ratio] = current_accuracy
            accuracy_drop = baseline_accuracy - current_accuracy

            if accuracy_drop <= CONFIG["max_accuracy_drop"]:
                current_ratio = round(current_ratio + CONFIG["pruning_step"], 2)
            else:
                print(f"      -> STOPPING. Drop of {accuracy_drop*100:.2f}% exceeded threshold of {CONFIG['max_accuracy_drop']*100:.2f}%.")
                break
            if current_ratio >= 0.8:
                print("      -> STOPPING. Reached maximum pruning limit.")
                break
        
        all_results_for_plot[dataset_name] = run_results
        print(f"\n   - Sub-step [3/3]: Experiment for {dataset_name} complete.")

    step_num = num_total_steps
    print("\n" + "="*80)
    print(f"      STEP [{step_num}/{num_total_steps}]: Generating Overall Summary Plot")
    print("="*80)
    generate_summary_plot(all_results_for_plot, CONFIG["max_accuracy_drop"], strategy)
    print("="*80)

if __name__ == "__main__":
    main()