# Fingerprint generation & Univerifier training for LINK PREDICTION on CiteSeer.
#  - loads LP encoders + dot-product decoder
#  - feature vector per model = concatenated EDGE logits over P fingerprints (each contributes m logits)


import argparse, glob, json, math, random, time, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, to_undirected
from gcn_lp import get_encoder, DotProductDecoder

def set_seed(s):
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

def get_lp_encoder(arch: str, in_dim: int, hidden: int, layers: int = 3):
    return get_encoder(arch, in_dim, hidden, num_layers=layers, dropout=0.5)

def load_encoder_from_pt(pt_path: str, in_dim: int):
    meta = json.load(open(pt_path.replace('.pt', '.json'), 'r'))
    enc = get_lp_encoder(meta["arch"], in_dim, meta["hidden"], layers=meta.get("layers", 3))
    enc.load_state_dict(torch.load(pt_path, map_location='cpu'))
    enc.eval()
    return enc, meta

# LP fingerprint forward: encoder -> embeddings -> decoder over probe edges
def forward_on_fp(encoder, decoder, fp):
    X = fp["X"]
    A = fp["A"]
    n = X.size(0)

    A_bin = (A > 0.5).float()
    A_sym = torch.maximum(A_bin, A_bin.t())
    edge_index = dense_to_sparse(A_sym)[0]
    if edge_index.numel() == 0:
        idx = torch.arange(n, dtype=torch.long)
        edge_index = torch.stack([idx, (idx + 1) % n], dim=0)
    edge_index = to_undirected(edge_index)

    # node embeddings
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

def concat_for_model(encoder, decoder, fingerprints):
    vecs = [forward_on_fp(encoder, decoder, fp) for fp in fingerprints]
    return torch.cat(vecs, dim=-1)

def compute_loss(encoders_pos, encoders_neg, fingerprints, V, decoder):
    z_pos = [concat_for_model(e, decoder, fingerprints) for e in encoders_pos]
    z_neg = [concat_for_model(e, decoder, fingerprints) for e in encoders_neg]
    if not z_pos or not z_neg:
        raise RuntimeError("Need both positive and negative models.")
    Zp = torch.stack(z_pos)
    Zn = torch.stack(z_neg)

    yp = V(Zp).clamp(1e-6, 1-1e-6)
    yn = V(Zn).clamp(1e-6, 1-1e-6)
    L = torch.log(yp).mean() + torch.log(1 - yn).mean()
    return L, Zp, Zn

def feature_ascent_step(encoders_pos, encoders_neg, fingerprints, V, decoder, alpha=0.01):
    # ascent on X only
    for fp in fingerprints:
        fp["X"].requires_grad_(True)
        fp["A"].requires_grad_(False)

    L, _, _ = compute_loss(encoders_pos, encoders_neg, fingerprints, V, decoder)
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

def edge_flip_step(encoders_pos, encoders_neg, fingerprints, V, decoder, flip_k=8):
    for fp_idx, fp in enumerate(fingerprints):
        A = fp["A"]
        i_idx, j_idx = edge_flip_candidates(A, flip_k * 4)  # candidate pool
        with torch.no_grad():
            base_L, _, _ = compute_loss(encoders_pos, encoders_neg, fingerprints, V, decoder)

        gains = []
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            with torch.no_grad():
                old = float(A[i, j])
                new = 1.0 - old
                # toggle in place
                A[i, j] = new; A[j, i] = new
                L_try, _, _ = compute_loss(encoders_pos, encoders_neg, fingerprints, V, decoder)
                gain = float(L_try - base_L)
                gains.append((gain, i, j, old))
                # revert
                A[i, j] = old; A[j, i] = old

        gains.sort(key=lambda x: x[0], reverse=True)
        with torch.no_grad():
            for g, i, j, old in gains[:flip_k]:
                new = 1.0 - old
                A[i, j] = new; A[j, i] = new
        A.clamp_(0.0, 1.0)

def train_verifier_step(encoders_pos, encoders_neg, fingerprints, V, decoder, opt):
    # maximize L wrt V (via minimizing -L)
    L, Zp, Zn = compute_loss(encoders_pos, encoders_neg, fingerprints, V, decoder)
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
    ap.add_argument('--target_path', default='models/target_model_lp.pt')
    ap.add_argument('--target_meta', default='models/target_meta_lp.json')
    ap.add_argument('--positives_glob', default='models/positives/lp_ftpr_*.pt,models/positives/distill_lp_*.pt')
    ap.add_argument('--negatives_glob', default='models/negatives/negative_lp_*.pt')

    ap.add_argument('--P', type=int, default=64)               # number of fingerprints
    ap.add_argument('--n', type=int, default=32)                # nodes per fingerprint
    ap.add_argument('--iters', type=int, default=1000)            # alternations
    ap.add_argument('--verifier_lr', type=float, default=1e-3)
    ap.add_argument('--e1', type=int, default=1)               # fingerprint update epochs per alternation
    ap.add_argument('--e2', type=int, default=1)               # verifier update epochs per alternation
    ap.add_argument('--alpha_x', type=float, default=0.01)     # feature ascent step
    ap.add_argument('--flip_k', type=int, default=8)           # edges flipped per fp per step
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--m', type=int, default=4)                # probed edges per fingerprint (via node_idx size)
    args = ap.parse_args()

    start_time = time.time()
    set_seed(args.seed)
    Path('fingerprints').mkdir(parents=True, exist_ok=True)

    ds = Planetoid(root='data', name='CiteSeer')
    in_dim = ds.num_features

    meta_t = load_meta(args.target_meta)
    target_enc = get_lp_encoder(meta_t["arch"], in_dim, meta_t["hidden"], layers=meta_t.get("layers", 3))
    target_enc.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    target_enc.eval()

    pos_globs = [g.strip() for g in args.positives_glob.split(',') if g.strip()]
    pos_paths = list_paths_from_globs(pos_globs)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    enc_pos = [target_enc] + [load_encoder_from_pt(p, in_dim)[0] for p in pos_paths]
    enc_neg = [load_encoder_from_pt(npath, in_dim)[0] for npath in neg_paths]
    decoder = DotProductDecoder()

    print(f"[loaded] positives={len(enc_pos)} (incl. target) | negatives={len(enc_neg)}")

    if args.m > args.n:
        raise ValueError(f"--m ({args.m}) must be <= --n ({args.n})")

    fingerprints = []
    for _ in range(args.P):
        X = torch.randn(args.n, in_dim) * 0.1
        A = torch.rand(args.n, args.n) * 0.2 + 0.4
        A = torch.triu(A, diagonal=1)
        A = A + A.t()
        torch.diagonal(A).zero_()
        idx = torch.randperm(args.n)[:args.m]
        fingerprints.append({"X": X, "A": A, "node_idx": idx})

    # Univerifier
    ver_in_dim = args.P * args.m
    V = FPVerifier(ver_in_dim)
    optV = torch.optim.Adam(V.parameters(), lr=args.verifier_lr)

    flag = 0
    for it in range(1, args.iters + 1):
        if flag == 0:
            # Update fingerprints (features + edges)
            for _ in range(args.e1):
                feature_ascent_step(enc_pos, enc_neg, fingerprints, V, decoder, alpha=args.alpha_x)
                edge_flip_step(enc_pos, enc_neg, fingerprints, V, decoder, flip_k=args.flip_k)
            flag = 1
        else:
            # Update verifier
            diag_acc = None
            for _ in range(args.e2):
                Lval, acc = train_verifier_step(enc_pos, enc_neg, fingerprints, V, decoder, optV)
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
        "fingerprints/fingerprints_lp.pt"
    )

    print("Saved fingerprints/fingerprints_lp.pt")
    end_time = time.time()
    print("Time taken: ", (end_time - start_time)/60)


if __name__ == '__main__':
    main()
