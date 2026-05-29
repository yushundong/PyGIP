import torch
from datasets import Cora
from models.defense import RandomWM

torch.manual_seed(42)

dataset = Cora(api_type='dgl')
print(f"Device: cpu")
print(f"Dataset: Cora | Nodes: {dataset.num_nodes} | Features: {dataset.num_features} | Classes: {dataset.num_classes}\n")

defense = RandomWM(
    dataset=dataset,
    attack_node_fraction=0.1,
    wm_node=50,
    pr=0.2,
    pg=0.2,
)

print("=== Running RandomWM Defense ===\n")
metrics = defense.defend()

print("\n=== Defense Metrics ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
