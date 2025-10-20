# test1.py
# Minimal paper-style run on Cora/Citeseer (node classification).
# Requires: Dataset(api_type='pyg') that exposes .graph_data with train/val/test masks.

import torch
from datasets.datasets import Cora, CiteSeer
from models.defense.GNNFingers import GNNFingers, FPConfig

def run(dataset_cls):
    ds = dataset_cls(api_type='pyg')        # library-guideline: PyG variant
    cfg = FPConfig(
        P=64, n_nodes=32, m_readout=32, depth=3,
        x_step=1e-2, topK_ratio=0.03, iters=1000,
        alt_I_steps=1, alt_V_steps=1, update_A=True, update_X=True
    )
    defense = GNNFingers(
        dataset=ds,
        fingerprint=cfg,
        hidden_channels=128,
        depth=3,
        owner_epochs=200,
        verification_threshold=0.5,
        n_pos=200, n_neg=200,
        pos_ops=("finetune_last", "finetune_all", "partial_reinit", "prune", "distill"),
        neg_archs=("gcn", "sage"),
        model_path="ckpts/owner.pt",
        save_dir="registry"
    )
    metrics = defense.defend()
    print("== Paper-style metrics ==")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Optional: quick self-verify the owner (should be 'pirated' == True)
    owner_model, _, _ = defense._load_or_train_owner(ds.graph_data, defense.get_device())
    print("Self-verify (owner):", defense.verify(owner_model))

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== Cora ===")
    run(Cora)
    # print("=== Citeseer ===")
    # run(Citeseer)