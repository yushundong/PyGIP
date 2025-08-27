"""
Evaluate a trained Univerifier on LP positives ({target ∪ F+}) and negatives (F−)
using saved LP fingerprints. Produces Robustness/Uniqueness, Mean Test Accuracy and ARUC.
"""

import argparse, glob, json, math, os, torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, to_undirected

from gcn_lp import get_encoder, DotProductDecoder

import torch.nn as nn

class FPVerifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


def list_paths_from_globs(globs_str):
    globs = [g.strip() for g in globs_str.split(",") if g.strip()]
    paths = []
    for g in globs:
        paths.extend(glob.glob(g))
    return sorted(paths)

def get_lp_encoder(arch: str, in_dim: int, hidden: int, layers: int = 3):
    return get_encoder(arch, in_dim, hidden, num_layers=layers, dropout=0.5)

def load_encoder_from_pt(pt_path: str, in_dim: int):
    meta_path = pt_path.replace(".pt", ".json")
    j = json.load(open(meta_path, "r"))
    enc = get_lp_encoder(j["arch"], in_dim, j["hidden"], layers=j.get("layers", 3))
    enc.load_state_dict(torch.load(pt_path, map_location="cpu"))
    enc.eval()
    return enc


@torch.no_grad()
def forward_on_fp(encoder, decoder, fp):
    X = fp["X"]
    A = fp["A"]
    n = X.size(0)

    # Binarize & symmetrize adjacency; build undirected edge_index
    A_bin = (A > 0.5).float()
    A_sym = torch.maximum(A_bin, A_bin.t())
    edge_index = dense_to_sparse(A_sym)[0]
    if edge_index.numel() == 0:
        idx = torch.arange(n, dtype=torch.long)
        edge_index = torch.stack([idx, (idx + 1) % n], dim=0)
    edge_index = to_undirected(edge_index)

    z = encoder(X, edge_index)

    sel = fp["node_idx"]
    if sel.numel() == 1:
        u = sel
        v = torch.tensor([(sel.item() + 1) % n], dtype=torch.long)
    else:
        u = sel
        v = torch.roll(sel, shifts=-1, dims=0)
    probe_edge = torch.stack([u, v], dim=0)

    logits = decoder(z, probe_edge)
    return logits

@torch.no_grad()
def concat_for_model(encoder, decoder, fps):
    parts = [forward_on_fp(encoder, decoder, fp) for fp in fps]
    return torch.cat(parts, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fingerprints_path', type=str, default='fingerprints/fingerprints_lp.pt')
    ap.add_argument('--verifier_path', type=str, default='fingerprints/univerifier_lp.pt')
    ap.add_argument('--target_path', type=str, default='models/target_model_lp.pt')
    ap.add_argument('--target_meta', type=str, default='models/target_meta_lp.json')
    ap.add_argument('--positives_glob', type=str,
                    default='models/positives/lp_ftpr_*.pt,models/positives/distill_lp_*.pt')
    ap.add_argument('--negatives_glob', type=str, default='models/negatives/negative_lp_*.pt')
    ap.add_argument('--out_plot', type=str, default='plots/citeseer_lp_aruc.png')
    ap.add_argument('--save_csv', type=str, default='',
                    help='Optional: path to save thresholds/robustness/uniqueness CSV')
    args = ap.parse_args()

    ds = Planetoid(root="data", name="CiteSeer")
    in_dim = ds.num_features

    # Load fingerprints (with node_idx)
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = int(pack.get("ver_in_dim", 0))

    decoder = DotProductDecoder()

    tmeta = json.load(open(args.target_meta, "r"))
    target_enc = get_lp_encoder(tmeta["arch"], in_dim, tmeta["hidden"], layers=tmeta.get("layers", 3))
    target_enc.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target_enc.eval()

    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models_pos = [target_enc] + [load_encoder_from_pt(p, in_dim) for p in pos_paths]
    models_neg = [load_encoder_from_pt(n, in_dim) for n in neg_paths]

    z0 = concat_for_model(models_pos[0], decoder, fps)
    D = z0.numel()
    if ver_in_dim_saved and ver_in_dim_saved != D:
        raise RuntimeError(f"Verifier input mismatch: D={D} vs ver_in_dim_saved={ver_in_dim_saved}")

    V = FPVerifier(D)
    ver_path = Path(args.verifier_path)
    if ver_path.exists():
        V.load_state_dict(torch.load(str(ver_path), map_location='cpu'))
        print(f"Loaded verifier from {ver_path}")
    elif "verifier" in pack:
        V.load_state_dict(pack["verifier"])
        print("Loaded verifier from fingerprints pack.")
    else:
        raise FileNotFoundError(
            f"No verifier found at {args.verifier_path} and no 'verifier' key in {args.fingerprints_path}"
        )
    V.eval()

    with torch.no_grad():
        pos_scores = []
        for enc in models_pos:
            z = concat_for_model(enc, decoder, fps).unsqueeze(0)  # [1, D]
            pos_scores.append(float(V(z)))
        neg_scores = []
        for enc in models_neg:
            z = concat_for_model(enc, decoder, fps).unsqueeze(0)
            neg_scores.append(float(V(z)))

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    ts = np.linspace(0.0, 1.0, 201)
    robustness = np.array([(pos_scores >= t).mean() for t in ts])  # TPR on positives
    uniqueness = np.array([(neg_scores <  t).mean() for t in ts])  # TNR on negatives
    overlap = np.minimum(robustness, uniqueness)
    # Accuracy at each threshold
    Npos, Nneg = len(pos_scores), len(neg_scores)
    acc_curve = np.array([((pos_scores >= t).sum() + (neg_scores < t).sum()) / (Npos + Nneg)
                        for t in ts])
    mean_test_acc = float(acc_curve.mean())


    aruc = np.trapz(overlap, ts)

    # Best threshold (maximize min(robustness, uniqueness))
    idx_best = int(np.argmax(overlap))
    t_best = float(ts[idx_best])
    rob_best = float(robustness[idx_best])
    uniq_best = float(uniqueness[idx_best])
    acc_best = 0.5 * (rob_best + uniq_best)

    print(f"Mean Test Accuracy (avg over thresholds) = {mean_test_acc:.4f}")
    print(f"Models: +{len(models_pos)} | -{len(models_neg)} | D={D}")
    print(f"ARUC = {aruc:.4f}")
    print(f"Best threshold = {t_best:.3f} | Robustness={rob_best:.3f} | Uniqueness={uniq_best:.3f} | Acc={acc_best:.3f}")

    if args.save_csv:
        import csv
        Path(os.path.dirname(args.save_csv)).mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['threshold', 'robustness', 'uniqueness', 'min_curve', 'accuracy'])
            for t, r, u, s, a in zip(ts, robustness, uniqueness, shade, acc_curve):
                w.writerow([f"{t:.5f}", f"{r:.6f}", f"{u:.6f}", f"{s:.6f}", f"{a:.6f}"])
        print(f"Saved CSV to {args.save_csv}")

    # Plot
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax.set_title(f"CiteSeer link-prediction • ARUC={aruc:.3f}", fontsize=14)
    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.plot(ts, robustness, color="#ff0000", linewidth=2.0, label="Robustness (TPR)")
    ax.plot(ts, uniqueness, color="#0000ff", linestyle="--", linewidth=2.0, label="Uniqueness (TNR)")
    overlap = np.minimum(robustness, uniqueness)
    ax.fill_between(ts, overlap, color="#bbbbbb", alpha=0.25, label="Overlap (ARUC region)")

    # best-threshold vertical line
    # ax.axvline(t_best, color="0.4", linewidth=2.0, alpha=0.6)

    ax.set_xlabel("Threshold (τ)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=11)

    leg = ax.legend(loc="lower left", frameon=True, framealpha=0.85,
                    facecolor="white", edgecolor="0.8")

    plt.tight_layout()
    plt.savefig(args.out_plot, bbox_inches="tight")
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
