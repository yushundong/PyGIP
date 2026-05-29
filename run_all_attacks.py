import torch
from datasets import Cora
from models.attack import (
    ModelExtractionAttack1,
    ModelExtractionAttack2,
    ModelExtractionAttack3,
    ModelExtractionAttack4,
    ModelExtractionAttack5,
    AdvMEA,
)
from utils.metrics import GraphNeuralNetworkMetric

torch.manual_seed(42)

dataset = Cora(api_type='dgl')
print(f"Dataset: Cora | Nodes: {dataset.num_nodes} | Features: {dataset.num_features} | Classes: {dataset.num_classes}")
print(f"(MEA0 baseline already run: Fidelity=0.6505, Accuracy=0.6276)\n")

results = {"MEA0 (baseline)": (0.6505, 0.6276)}

attacks = [
    ("MEA1 (shadow graph)",        ModelExtractionAttack1, dict(attack_node_fraction=0.1)),
    ("MEA2 (identity features)",   ModelExtractionAttack2, dict(attack_node_fraction=0.1)),
    ("MEA3 (subgraph merge)",      ModelExtractionAttack3, dict(attack_node_fraction=0.1)),
    ("MEA4 (distance linking)",    ModelExtractionAttack4, dict(attack_node_fraction=0.1)),
    ("MEA5 (distance linking v2)", ModelExtractionAttack5, dict(attack_node_fraction=0.1)),
    ("AdvMEA (k-hop + sampling)",  AdvMEA,                 dict(attack_node_fraction=0.1)),
]

for name, AttackClass, kwargs in attacks:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print('='*60)
    try:
        atk = AttackClass(dataset, **kwargs)
        out = atk.attack()
        # AdvMEA returns (metrics, net); others store best_performance_metrics inline
        if isinstance(out, tuple):
            metrics = out[0]
        else:
            # MEA1-5 print results internally; retrieve from atk if available
            metrics = None

        if metrics is not None and hasattr(metrics, 'fidelity'):
            results[name] = (metrics.fidelity, metrics.accuracy)
        elif hasattr(atk, 'net2'):
            results[name] = ("done", "see above")
        else:
            results[name] = ("done", "see above")
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = ("error", str(e)[:60])

print(f"\n\n{'='*60}")
print("SUMMARY — All MEAs on Cora")
print('='*60)
print(f"{'Attack':<30} {'Fidelity':>10} {'Accuracy':>10}")
print('-'*52)
for name, (fid, acc) in results.items():
    if isinstance(fid, float):
        print(f"{name:<30} {fid:>10.4f} {acc:>10.4f}")
    else:
        print(f"{name:<30} {fid:>10} {acc:>10}")
