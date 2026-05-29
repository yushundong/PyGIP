import torch
from datasets import Cora
from models.attack import ModelExtractionAttack0

torch.manual_seed(42)

dataset = Cora(api_type='dgl')

print(f"Dataset: Cora | Nodes: {dataset.num_nodes} | Features: {dataset.num_features} | Classes: {dataset.num_classes}")
print(f"Attack fraction: 10% ({int(dataset.num_nodes * 0.1)} nodes queried)")
print()

attack = ModelExtractionAttack0(dataset, attack_node_fraction=0.1)
attack.attack()
