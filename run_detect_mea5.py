"""
Run MEA5 (best attacker) then test whether GNNFingers and RandomWM catch the stolen model.
"""
import os
import torch
import torch.nn as nn
import dgl
from datasets import Cora
from models.attack import ModelExtractionAttack5
from models.defense import GNNFingers, RandomWM
from models.defense.GNNFingers import FPConfig

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── 1. Run MEA5 and keep the extracted model ────────────────────────────────
print("=" * 60)
print("Step 1: Run MEA5 (best attacker — 85.9% fidelity)")
print("=" * 60)
dataset_dgl = Cora(api_type='dgl')
atk = ModelExtractionAttack5(dataset_dgl, attack_node_fraction=0.1)
atk.attack()
stolen_dgl = atk.net2
stolen_dgl.eval()
print()


# ── 2. GNNFingers detection ─────────────────────────────────────────────────
print("=" * 60)
print("Step 2: GNNFingers — can it detect the stolen model?")
print("=" * 60)

# GNNFingers verify() expects PyG-style forward(x, edge_index).
# Wrap the DGL GCN with an adapter.
class DGLGCNAdapter(nn.Module):
    def __init__(self, dgl_gcn):
        super().__init__()
        self.model = dgl_gcn

    def forward(self, x, edge_index):
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.size(0))
        g = dgl.add_self_loop(g).to(x.device)
        return self.model(g, x)

stolen_adapted = DGLGCNAdapter(stolen_dgl)

dataset_pyg = Cora(api_type='pyg')
registry_path = "registry/fingerprints.pt"

if os.path.exists(registry_path):
    print(f"  Using existing registry: {registry_path}")
    gnn_defense = GNNFingers(dataset=dataset_pyg, fingerprint=None,
                              hidden_channels=64, depth=2,
                              n_pos=0, n_neg=0, owner_epochs=0,
                              model_path="ckpts/owner.pt", save_dir="registry",
                              device=device)
else:
    print("  No registry found — running GNNFingers defend() first...")
    cfg = FPConfig(P=16, n_nodes=16, m_readout=16, depth=2,
                   x_step=5e-3, topK_ratio=0.05, iters=100,
                   alt_I_steps=1, alt_V_steps=1, update_A=True, update_X=True)
    gnn_defense = GNNFingers(dataset=dataset_pyg, fingerprint=cfg,
                              hidden_channels=64, depth=2, owner_epochs=50,
                              n_pos=10, n_neg=10, model_path="ckpts/owner.pt",
                              save_dir="registry", device=device)
    gnn_defense.defend()

owner, _, _ = gnn_defense._load_or_train_owner(dataset_pyg.graph_data, device)
owner_result  = gnn_defense.verify(owner)
stolen_result = gnn_defense.verify(stolen_adapted)

print(f"\n  Owner  — o_plus: {owner_result['o_plus']:.4f}  verified: {owner_result['verified']}")
print(f"  Stolen — o_plus: {stolen_result['o_plus']:.4f}  verified: {stolen_result['verified']}"
      f"  (τ={stolen_result['threshold']})")

if stolen_result['verified']:
    print("  >> DETECTED: GNNFingers flagged the MEA5 stolen model.")
else:
    print("  >> MISSED: GNNFingers did not flag the MEA5 stolen model.")
print()


# ── 3. RandomWM detection ───────────────────────────────────────────────────
print("=" * 60)
print("Step 3: RandomWM — does the stolen model fail the watermark test?")
print("=" * 60)

wm_defense = RandomWM(dataset=dataset_dgl, attack_node_fraction=0.1,
                      wm_node=50, pr=0.2, pg=0.2)

print("\n  Training watermarked defense model (GraphSAGE)...")
defense_model = wm_defense._train_defense_model()

# _evaluate_watermark uses the dataloader path (for GraphSAGE-style models)
legit_wm_acc = wm_defense._evaluate_watermark(defense_model)
# _evaluate_attack_on_watermark dispatches on model class name
stolen_wm_acc = wm_defense._evaluate_attack_on_watermark(stolen_dgl)

print(f"\n  Legitimate defense model (GraphSAGE) — watermark accuracy: {legit_wm_acc:.4f}")
print(f"  MEA5 stolen model (DGL GCN)          — watermark accuracy: {stolen_wm_acc:.4f}")
print(f"  Random baseline (1/7 classes)         — ~0.143")

gap = legit_wm_acc - stolen_wm_acc
if gap > 0.1:
    print(f"\n  >> DETECTED: ownership gap = {gap:.4f} — stolen model fails watermark.")
else:
    print(f"\n  >> MISSED: gap = {gap:.4f} — watermark not effective against MEA5.")


# ── 4. Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  MEA5 attack fidelity            : 0.8592")
print(f"  GNNFingers owner  o_plus        : {owner_result['o_plus']:.4f}  ({'' if owner_result['verified'] else 'NOT '}verified)")
print(f"  GNNFingers stolen o_plus        : {stolen_result['o_plus']:.4f}  ({'' if stolen_result['verified'] else 'NOT '}verified)")
print(f"  RandomWM  legit  watermark acc  : {legit_wm_acc:.4f}")
print(f"  RandomWM  stolen watermark acc  : {stolen_wm_acc:.4f}")
