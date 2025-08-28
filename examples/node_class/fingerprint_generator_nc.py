import argparse, glob, json, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, to_undirected
import time


def set_seed(s):
    random.seed(s); torch.manual_seed(s)

def load_meta(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_model(arch: str, in_dim: int, hidden: int, num_classes: int):
    from gcn_nc import get_model as _get
    return _get(arch, in_dim, hidden, num_classes)

def list_paths_from_globs(globs: List[str]) -> List[str]:
    out = []
    for g in globs:
        out.extend(glob.glob(g))
    return sorted(out)

class FPVerifier(nn.Module):
    # Arch: [128, 64, 32] + LeakyReLU, sigmoid output
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

def load_model_from_pair(pt_path: str, in_dim: int):
    meta = json.load(open(pt_path.replace('.pt', '.json'), 'r'))
    m = get_model(meta["arch"], in_dim, meta["hidden"], meta["num_classes"])
    m.load_state_dict(torch.load(pt_path, map_location='cpu'))
    m.eval()
    return m, meta

def forward_on_fp(model, fp):
    A = fp["A"]
    A_bin = (A > 0.5).float()
    A_sym = torch.maximum(A_bin, A_bin.t())
    edge_index = dense_to_sparse(A_sym)[0]
    if edge_index.numel() == 0:
        edge_index = torch.arange(fp["X"].size(0)).repeat(2,1)
    edge_index = to_undirected(edge_index)
    logits = model(fp["X"], edge_index)
    return logits.mean(dim=0)

def concat_for_model(model, fingerprints):
    vecs = [forward_on_fp(model, fp) for fp in fingerprints]
    return torch.cat(vecs, dim=-1)

def compute_loss(models_pos, models_neg, fingerprints, V):
    z_pos = []
    for m in models_pos:
        z_pos.append(concat_for_model(m, fingerprints))
    z_neg = []
    for m in models_neg:
        z_neg.append(concat_for_model(m, fingerprints))
    if len(z_pos) == 0 or len(z_neg) == 0:
        raise RuntimeError("Need both positive and negative models.")
    Zp = torch.stack(z_pos)
    Zn = torch.stack(z_neg)

    yp = V(Zp).clamp(1e-6, 1-1e-6)
    yn = V(Zn).clamp(1e-6, 1-1e-6)

    L = torch.log(yp).mean() + torch.log(1 - yn).mean()
    return L, Zp, Zn

def feature_ascent_step(models_pos, models_neg, fingerprints, V, alpha=0.01):
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
    # Rank-and-flip edges by gain in the full loss L when flipping entries in ONE fp at a time
    for fp_idx, fp in enumerate(fingerprints):
        A = fp["A"]
        i_idx, j_idx = edge_flip_candidates(A, flip_k * 4)  # candidate pool
        with torch.no_grad():
            base_L, _, _ = compute_loss(models_pos, models_neg, fingerprints, V)

        gains = []
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            with torch.no_grad():
                old = float(A[i, j])
                new = 1.0 - old
                # toggle in place
                A[i, j] = new; A[j, i] = new
                L_try, _, _ = compute_loss(models_pos, models_neg, fingerprints, V)
                gain = float(L_try - base_L)
                gains.append((gain, i, j, old))
                # revert
                A[i, j] = old; A[j, i] = old

        # Flip the best k edges for this fingerprint
        gains.sort(key=lambda x: x[0], reverse=True)
        with torch.no_grad():
            for g, i, j, old in gains[:flip_k]:
                new = 1.0 - old
                A[i, j] = new; A[j, i] = new
        A.clamp_(0.0, 1.0)

def train_verifier_step(models_pos, models_neg, fingerprints, V, opt):
    L, Zp, Zn = compute_loss(models_pos, models_neg, fingerprints, V)
    loss = -L
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        yp = (V(Zp) >= 0.5).float().mean().item()
        yn = (V(Zn) < 0.5).float().mean().item()
        acc = 0.5 * (yp + yn)
    return float(L.item()), acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_path', default='models/target_model_nc.pt')
    ap.add_argument('--target_meta', default='models/target_meta_nc.json')
    ap.add_argument('--positives_glob', default='models/positives/nc_ftpr_*.pt,models/positives/distill_nc_*.pt')
    ap.add_argument('--negatives_glob', default='models/negatives/negative_nc_*.pt')

    # Hyperparams
    ap.add_argument('--P', type=int, default=64)             
    ap.add_argument('--n', type=int, default=32)             # nodes per fingerprint
    ap.add_argument('--iters', type=int, default=1000)       # alternating iterations
    ap.add_argument('--verifier_lr', type=float, default=1e-3)  # learning rate for V
    ap.add_argument('--e1', type=int, default=1)             # epochs for fingerprint updates per alternation
    ap.add_argument('--e2', type=int, default=1)             # epochs for verifier updates per alternation
    ap.add_argument('--alpha_x', type=float, default=0.01)   # step size for feature ascent
    ap.add_argument('--flip_k', type=int, default=4)         # edges flipped per step per fingerprint
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--m', type=int, default=64)       # sampled nodes per fingerprint


    args = ap.parse_args()

    set_seed(args.seed)
    Path('fingerprints').mkdir(parents=True, exist_ok=True)

    # Dataset dims
    ds = Planetoid(root='data/cora', name='Cora')
    in_dim = ds.num_features
    num_classes = ds.num_classes

    # Load {f} and F+ into "positives"; F- separately
    meta_t = load_meta(args.target_meta)
    target = get_model(meta_t["arch"], in_dim, meta_t["hidden"], meta_t["num_classes"])
    target.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    target.eval()

    pos_globs = [g.strip() for g in args.positives_glob.split(',') if g.strip()]
    pos_paths = list_paths_from_globs(pos_globs)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models_pos = [target]
    for p in pos_paths:
        m,_ = load_model_from_pair(p, in_dim)
        models_pos.append(m)

    models_neg = []
    for npath in neg_paths:
        m,_ = load_model_from_pair(npath, in_dim)
        models_neg.append(m)

    print(f"[loaded] positives={len(models_pos)} (incl. target) | negatives={len(models_neg)}")

    # Initialize fingerprints with small random X, sparse A near 0.5
    fingerprints = []
    if args.m > args.n:
        raise ValueError(f"--m ({args.m}) must be <= --n ({args.n})")

    for _ in range(args.P):
        X = torch.randn(args.n, in_dim) * 0.1
        A = torch.rand(args.n, args.n) * 0.2 + 0.4
        A = torch.triu(A, diagonal=1)
        A = A + A.t()
        torch.diagonal(A).zero_()
        idx = torch.randperm(args.n)[:args.m]
        fingerprints.append({"X": X, "A": A, "node_idx": idx})


    ver_in_dim = args.P * args.m * num_classes
    V = FPVerifier(ver_in_dim)
    optV = torch.optim.Adam(V.parameters(), lr=args.verifier_lr)

    flag = 0
    for it in range(1, args.iters + 1):
        if flag == 0:
            # Update fingerprints (features + edges), e1 times
            for _ in range(args.e1):
                feature_ascent_step(models_pos, models_neg, fingerprints, V, alpha=args.alpha_x)
                edge_flip_step(models_pos, models_neg, fingerprints, V, flip_k=args.flip_k)
            flag = 1
        else:
            # Update verifier, e2 times
            diag_acc = None
            for _ in range(args.e2):
                Lval, acc = train_verifier_step(models_pos, models_neg, fingerprints, V, optV)
                diag_acc = acc
            flag = 0

        if it % 10 == 0 and 'diag_acc' in locals() and diag_acc is not None:
            print(f"[Iter {it}] verifier acc={diag_acc:.3f} (diagnostic)")

    clean_fps = []
    for fp in fingerprints:
        clean_fps.append({
            "X": fp["X"].detach().clone(),
            "A": fp["A"].detach().clone(),
            "node_idx": fp["node_idx"].detach().clone(),
        })
    torch.save(
        {"fingerprints": clean_fps, "verifier": V.state_dict(), "ver_in_dim": ver_in_dim},
        "fingerprints/fingerprints_nc.pt"
    )

    print("Saved fingerprints/fingerprints_nc.pt")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    print("Time taken: ", (end_time - start_time)/60)

