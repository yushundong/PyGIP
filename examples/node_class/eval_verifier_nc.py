"""
Evaluate a trained Univerifier on positives ({target ∪ F+}) and negatives (F−)
using saved fingerprints. Produces Robustness/Uniqueness, ARUC, Mean Test Accuracy, KL Divergence.
"""

import argparse, glob, json, math, torch, os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from gcn_nc import get_model
import torch.nn.functional as F

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


@torch.no_grad()
def forward_on_fp(model, fp):

    X = fp["X"]
    A = fp["A"]
    idx = fp["node_idx"]

    A_bin = (A > 0.5).float()
    A_sym = torch.triu(A_bin, diagonal=1)
    A_sym = A_sym + A_sym.t()
    edge_index = dense_to_sparse(A_sym)[0]

    if edge_index.numel() == 0:
        n = X.size(0)
        edge_index = torch.arange(n, dtype=torch.long).repeat(2, 1)

    logits = model(X, edge_index)
    sel = logits[idx, :]
    return sel.reshape(-1)


@torch.no_grad()
def concat_for_model(model, fps):
    parts = [forward_on_fp(model, fp) for fp in fps]
    return torch.cat(parts, dim=0)


def list_paths_from_globs(globs_str):
    globs = [g.strip() for g in globs_str.split(",") if g.strip()]
    paths = []
    for g in globs:
        paths.extend(glob.glob(g))
    return sorted(paths)


def load_model_from_pt(pt_path, in_dim):
    meta_path = pt_path.replace(".pt", ".json")
    j = json.load(open(meta_path, "r"))
    m = get_model(j["arch"], in_dim, j["hidden"], j["num_classes"])
    m.load_state_dict(torch.load(pt_path, map_location="cpu"))
    m.eval()
    return m

# KL divergence helpers 
def softmax_logits(x):
    return F.softmax(x, dim=-1)

@torch.no_grad()
def forward_nc_logits(model, fp):
    X, A, idx = fp["X"], fp["A"], fp["node_idx"]
    A_bin = (A > 0.5).float()
    A_sym = torch.triu(A_bin, diagonal=1); A_sym = A_sym + A_sym.t()
    edge_index = dense_to_sparse(A_sym)[0]
    if edge_index.numel() == 0:
        n = X.size(0)
        edge_index = torch.arange(n, dtype=torch.long).repeat(2, 1)
    logits = model(X, edge_index)
    return logits[idx, :]

def sym_kl(p, q, eps=1e-8):
    """
    Symmetric KL
    """
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    kl1 = (p * (p.log() - q.log())).sum(dim=-1)
    kl2 = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl1 + kl2)

@torch.no_grad()
def model_nc_kl_to_target(suspect, target, fps):
    """
    Average symmetric KL over all fingerprints.
    """
    vals = []
    for fp in fps:
        t = softmax_logits(forward_nc_logits(target, fp))
        s = softmax_logits(forward_nc_logits(suspect, fp))
        d = sym_kl(s, t)
        vals.append(d.mean().item())
    return float(np.mean(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fingerprints_path', type=str, default='fingerprints/fingerprints_nc.pt')
    ap.add_argument('--verifier_path', type=str, default='fingerprints/univerifier_nc.pt')
    ap.add_argument('--target_path', type=str, default='models/target_model_nc.pt')
    ap.add_argument('--target_meta', type=str, default='models/target_meta_nc.json')
    ap.add_argument('--positives_glob', type=str,
                    default='models/positives/nc_ftpr_*.pt,models/positives/distill_nc_*.pt')
    ap.add_argument('--negatives_glob', type=str, default='models/negatives/negative_nc_*.pt')
    ap.add_argument('--out_plot', type=str, default='plots/cora_nc_aruc.png')   
    ap.add_argument('--out_plot_kl', type=str, default='plots/cora_nc_kl.png')
    ap.add_argument('--save_csv', type=str, default='',
                    help='Optional: path to save thresholds/robustness/uniqueness CSV')
    args = ap.parse_args()

    ds = Planetoid(root="data/cora", name="Cora")
    in_dim = ds.num_features
    num_classes = ds.num_classes

    # Load fingerprints (with node_idx)
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = int(pack.get("ver_in_dim", 0))

    # Load models (target + positives + negatives)
    tmeta = json.load(open(args.target_meta, "r"))
    target = get_model(tmeta["arch"], in_dim, tmeta["hidden"], tmeta["num_classes"])
    target.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target.eval()

    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models_pos = [target] + [load_model_from_pt(p, in_dim) for p in pos_paths]
    models_neg = [load_model_from_pt(n, in_dim) for n in neg_paths]

    # Infer verifier input dim from a probe concat
    z0 = concat_for_model(models_pos[0], fps)
    D = z0.numel()
    if ver_in_dim_saved and ver_in_dim_saved != D:
        raise RuntimeError(f"Verifier input mismatch: D={D} vs ver_in_dim_saved={ver_in_dim_saved}")

    V = FPVerifier(D)
    V.load_state_dict(torch.load(args.verifier_path, map_location='cpu'))
    V.eval()

    with torch.no_grad():
        pos_scores = []
        for m in models_pos:
            z = concat_for_model(m, fps).unsqueeze(0)
            pos_scores.append(float(V(z)))
        neg_scores = []
        for m in models_neg:
            z = concat_for_model(m, fps).unsqueeze(0)
            neg_scores.append(float(V(z)))

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # Sweep thresholds
    ts = np.linspace(0.0, 1.0, 201)
    robustness = np.array([(pos_scores >= t).mean() for t in ts])  # TPR on positives
    uniqueness = np.array([(neg_scores <  t).mean() for t in ts])  # TNR on negatives
    overlap = np.minimum(robustness, uniqueness)

    # Mean Test Accuracy at each threshold 
    Npos = len(pos_scores)
    Nneg = len(neg_scores)
    acc_curve = np.array([((pos_scores >= t).sum() + (neg_scores < t).sum()) / (Npos + Nneg)
                        for t in ts])

    mean_test_acc = float(acc_curve.mean())

    aruc = np.trapz(overlap, ts)

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
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['threshold', 'robustness', 'uniqueness', 'min_curve', 'accuracy'])
            for t, r, u, s, a in zip(ts, robustness, uniqueness, overlap, acc_curve):
                w.writerow([f"{t:.5f}", f"{r:.6f}", f"{u:.6f}", f"{s:.6f}", f"{a:.6f}"])
        print(f"Saved CSV to {args.save_csv}")

    # ARUC Plot
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


    # KL divergence 
    pos_divs = [model_nc_kl_to_target(m, target, fps) for m in models_pos[1:]]  # exclude target itself
    neg_divs = [model_nc_kl_to_target(m, target, fps) for m in models_neg]
    pos_divs, neg_divs = np.array(pos_divs), np.array(neg_divs)

    print(f"[KL] F+ mean±std = {pos_divs.mean():.4f}±{pos_divs.std():.4f} | "
          f"F- mean±std = {neg_divs.mean():.4f}±{neg_divs.std():.4f}")

    os.makedirs(os.path.dirname(args.out_plot_kl), exist_ok=True)
    plt.figure(figsize=(4.8, 3.2), dpi=160)
    bins = 30
    plt.hist(pos_divs, bins=bins, density=True, alpha=0.35, color="r", label="Surrogate GNN")
    plt.hist(neg_divs, bins=bins, density=True, alpha=0.35, color="b", label="Irrelevant GNN")
    plt.title("Node Classification")
    plt.xlabel("KL Divergence"); plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot_kl, bbox_inches="tight")
    print(f"Saved KL plot to {args.out_plot_kl}")


if __name__ == "__main__":
    main()
