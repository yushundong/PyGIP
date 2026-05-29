import os
import torch
from datasets import Cora
from models.defense import GNNFingers
from models.defense.GNNFingers import FPConfig

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = Cora(api_type='pyg')
print(f"Dataset: Cora | Nodes: {dataset.num_nodes} | Features: {dataset.num_features} | Classes: {dataset.num_classes}\n")

cfg = FPConfig(
    P=16,          # fingerprint graphs (paper uses 64)
    n_nodes=16,    # nodes per fingerprint graph
    m_readout=16,  # readout dimension
    depth=2,
    x_step=5e-3,
    topK_ratio=0.05,
    iters=100,     # joint training iterations (paper uses 1000)
    alt_I_steps=1,
    alt_V_steps=1,
    update_A=True,
    update_X=True,
)

os.makedirs("ckpts", exist_ok=True)
os.makedirs("registry", exist_ok=True)

defense = GNNFingers(
    dataset=dataset,
    fingerprint=cfg,
    hidden_channels=64,
    depth=2,
    owner_epochs=50,           # paper uses 200
    verification_threshold=0.5,
    n_pos=10, n_neg=10,        # paper uses 200/200
    pos_ops=("finetune_last", "finetune_all", "partial_reinit", "prune", "distill"),
    neg_archs=("gcn", "sage"),
    model_path="ckpts/owner.pt",
    save_dir="registry",
    device=device,
)

print("=== Running GNNFingers Defense ===\n")
metrics = defense.defend()

print("\n=== Defense Metrics ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print("\n=== Owner Verification ===")
owner, _, _ = defense._load_or_train_owner(dataset.graph_data, device)
result = defense.verify(owner)
print(f"  o_plus (owner score): {result['o_plus']:.4f}")
print(f"  verified: {result['verified']}  (threshold: {result['threshold']})")
