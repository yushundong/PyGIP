# Fingerprint generation for GRAPH CLASSIFICATION on ENZYMES.

import argparse, glob, json, random, time, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dense_to_sparse, to_undirected

from gsage_gc import get_model


def set_seed(s: int):
    random.seed(s); torch.manual_seed(s)


def load_meta(path):
    with open(path, 'r') as f:
        return json.load(f)


def list_paths_from_globs(globs: List[str]) -> List[str]:
    out = []
    for g in globs:
        out.extend(glob.glob(g))
    return sorted(out)


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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def load_model_from_pt(pt_path: str, in_dim: int, num_classes: int):
    meta = json.load(open(pt_path.replace('.pt', '.json'), 'r'))
    m = get_model(meta["arch"], in_dim, meta["hidden"], num_classes,
                  num_layers=meta.get("layers", 3), dropout=meta.get("dropout", 0.5), pool="mean")
    m.load_state_dict(torch.load(pt_path, map_location='cpu'))
    m.eval()
    return m, meta


@torch.no_grad()
def forward_on_fp(model, fp):
    X = fp["X"]
    A = fp["A"]
    n = X.size(0)

    # binarize & symmetrize adjacency -> edge_index
    A_bin = (A > 0.5).float()
    A_sym = torch.maximum(A_bin, A_bin.t())
    edge_index = dense_to_sparse(A_sym)[0]
    if edge_index.numel() == 0:
        idx = torch.arange(n, dtype=torch.long)
        edge_index = torch.stack([idx, (idx + 1) % n], dim=0)
    edge_index = to_undirected(edge_index)

    # single-graph batch vector of zeros
    batch = X.new_zeros(n, dtype=torch.long)
    logits = model(X, edge_index, batch=batch)
    return logits.squeeze(0)


def concat_for_model(model, fingerprints):
    vecs = [forward_on_fp(model, fp) for fp in fingerprints]
    return torch.cat(vecs, dim=-1)


def compute_loss(models_pos, models_neg, fingerprints, V):
    z_pos = [concat_for_model(m, fingerprints) for m in models_pos]
    z_neg = [concat_for_model(m, fingerprints) for m in models_neg]
    if not z_pos or not z_neg:
        raise RuntimeError("Need both positive and negative models.")
    Zp = torch.stack(z_pos)
    Zn = torch.stack(z_neg)

    yp = V(Zp).clamp(1e-6, 1-1e-6)
    yn = V(Zn).clamp(1e-6, 1-1e-6)
    L = torch.log(yp).mean() + torch.log(1 - yn).mean()
    return L, Zp, Zn


def feature_ascent_step(models_pos, models_neg, fingerprints, V, alpha=0.01):
    # ascent on X only
    for fp in fingerprints:
        fp["X"].requires_grad_(True)
        fp["A"].requires_grad_(False)

    L, _, _ = compute_loss(models_pos, models_neg, fingerprints, V)
    grads = torch.autograd.grad(
        L, [fp["X"] for fp in fingerprints],
        retain_graph=False, create_graph=False, allow_unused=True
    )
    with torch.no_grad():
        for fp, g in zip(fingerprints, grads):
            if g is None:
                g = torch.zeros_like(fp["X"])
            fp["X"].add_(alpha * g)
            fp["X"].clamp_(-5.0, 5.0)


def edge_flip_candidates(A: torch.Tensor, budget: int):
    n = A.size(0)
    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    scores = torch.abs(0.5 - A[tri_i, tri_j])
    order = torch.argsort(scores)
    picks = order[:min(budget, len(order))]
    return tri_i[picks], tri_j[picks]


def edge_flip_step(models_pos, models_neg, fingerprints, V, flip_k=8):
    for fp in fingerprints:
        A = fp["A"]
        i_idx, j_idx = edge_flip_candidates(A, flip_k * 4)

        with torch.no_grad():
            base_L, _, _ = compute_loss(models_pos, models_neg, fingerprints, V)

        gains = []
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            with torch.no_grad():
                old = float(A[i, j])
                new = 1.0 - old
                A[i, j] = new; A[j, i] = new
                L_try, _, _ = compute_loss(models_pos, models_neg, fingerprints, V)
                gains.append((float(L_try - base_L), i, j, old))
                A[i, j] = old; A[j, i] = old

        gains.sort(key=lambda x: x[0], reverse=True)
        with torch.no_grad():
            for g, i, j, old in gains[:flip_k]:
                A[i, j] = 1.0 - old; A[j, i] = 1.0 - old
        A.clamp_(0.0, 1.0)


def train_verifier_step(models_pos, models_neg, fingerprints, V, opt):
    L, Zp, Zn = compute_loss(models_pos, models_neg, fingerprints, V)
    loss = -L  # maximize L
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        yp = (V(Zp) >= 0.5).float().mean().item()
        yn = (V(Zn) < 0.5).float().mean().item()
        acc = 0.5 * (yp + yn)
    return float(L.item()), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_path', default='models/target_model_gc.pt')
    ap.add_argument('--target_meta', default='models/target_meta_gc.json')
    ap.add_argument('--positives_glob', default='models/positives/gc_ftpr_*.pt,models/positives/distill_gc_*.pt')
    ap.add_argument('--negatives_glob', default='models/negatives/negative_gc_*.pt')
    ap.add_argument('--out', default='fingerprints/fingerprints_gc.pt')

    ap.add_argument('--P', type=int, default=64)          # number of fingerprints (graphs)
    ap.add_argument('--n', type=int, default=32)           # nodes per fingerprint graph
    ap.add_argument('--iters', type=int, default=1000)       # alternations
    ap.add_argument('--e1', type=int, default=1)          # fingerprint updates per alternation
    ap.add_argument('--e2', type=int, default=1)          # verifier updates per alternation
    ap.add_argument('--alpha_x', type=float, default=0.01)
    ap.add_argument('--flip_k', type=int, default=8)      # edge flips per fp per step
    ap.add_argument('--verifier_lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    Path('fingerprints').mkdir(parents=True, exist_ok=True)

    # Dataset dims for model reconstruction
    ds = TUDataset(root='data/ENZYMES', name='ENZYMES',
                   use_node_attr=True, transform=NormalizeFeatures())
    in_dim = ds.num_features
    num_classes = ds.num_classes

    meta_t = load_meta(args.target_meta)
    target = get_model(meta_t.get("arch", "gsage"), in_dim, meta_t.get("hidden", 64), num_classes,
                       num_layers=meta_t.get("layers", 3), dropout=meta_t.get("dropout", 0.5), pool="mean")
    target.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    target.eval()

    pos_paths = list_paths_from_globs([g.strip() for g in args.positives_glob.split(',') if g.strip()])
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models_pos = [target] + [load_model_from_pt(p, in_dim, num_classes)[0] for p in pos_paths]
    models_neg = [load_model_from_pt(npath, in_dim, num_classes)[0] for npath in neg_paths]

    print(f"[loaded] positives={len(models_pos)} (incl. target) | negatives={len(models_neg)}")

    # Initialize fingerprints: small random X, A near 0.5, symmetric
    fingerprints = []
    for _ in range(args.P):
        X = torch.randn(args.n, in_dim) * 0.1
        A = torch.rand(args.n, args.n) * 0.2 + 0.4
        A = torch.triu(A, diagonal=1); A = A + A.t()
        torch.diagonal(A).zero_()
        fingerprints.append({"X": X, "A": A})

    # Univerifier
    ver_in_dim = args.P * num_classes
    V = FPVerifier(ver_in_dim)
    optV = torch.optim.Adam(V.parameters(), lr=args.verifier_lr)

    flag = 0
    for it in range(1, args.iters + 1):
        if flag == 0:
            for _ in range(args.e1):
                feature_ascent_step(models_pos, models_neg, fingerprints, V, alpha=args.alpha_x)
                edge_flip_step(models_pos, models_neg, fingerprints, V, flip_k=args.flip_k)
            flag = 1
        else:
            diag_acc = None
            for _ in range(args.e2):
                Lval, acc = train_verifier_step(models_pos, models_neg, fingerprints, V, optV)
                diag_acc = acc
            flag = 0
        if it % 10 == 0 and 'diag_acc' in locals() and diag_acc is not None:
            print(f"[Iter {it}] verifier acc={diag_acc:.3f}")

    # Save
    clean_fps = [{"X": fp["X"].detach().clone(), "A": fp["A"].detach().clone()} for fp in fingerprints]
    torch.save(
        {"fingerprints": clean_fps, "verifier": V.state_dict(), "ver_in_dim": ver_in_dim},
        args.out
    )
    print(f"Saved {args.out}")
    print("Time taken (min): ", (time.time() - t0) / 60.0)


if __name__ == '__main__':
    main()
