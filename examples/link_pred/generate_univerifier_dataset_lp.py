"""
Build a Univerifier dataset from saved LP fingerprints.
Label 1 for positives ({target ∪ F+}) and 0 for negatives (F−).
Outputs a .pt with:
    - X: [N_models, D] where D = P * m   (m = probed edges per fingerprint)
    - y: [N_models] float tensor with 1.0 (positive) or 0.0 (negative)
"""

import argparse, glob, json, torch
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, to_undirected

from gcn_lp import get_encoder, DotProductDecoder


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


# LP fingerprint forward: encoder -> embeddings -> dot-product over probe edges
@torch.no_grad()
def forward_on_fp(encoder, decoder, fp):
    X = fp["X"]
    A = fp["A"]
    n = X.size(0)

    # Binarize & symmetrize adjacency, build edge_index
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
    ap.add_argument("--fingerprints_path", type=str, default="fingerprints/fingerprints_lp.pt")
    ap.add_argument("--target_path", type=str, default="models/target_model_lp.pt")
    ap.add_argument("--target_meta", type=str, default="models/target_meta_lp.json")
    ap.add_argument("--positives_glob", type=str,
                    default="models/positives/lp_ftpr_*.pt,models/positives/distill_lp_*.pt")
    ap.add_argument("--negatives_glob", type=str, default="models/negatives/negative_lp_*.pt")
    ap.add_argument("--out", type=str, default="fingerprints/univerifier_dataset_lp.pt")
    args = ap.parse_args()

    ds = Planetoid(root="data", name="CiteSeer")
    in_dim = ds.num_features

    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = pack.get("ver_in_dim", None)

    decoder = DotProductDecoder()

    tmeta = json.load(open(args.target_meta, "r"))
    target_enc = get_lp_encoder(tmeta["arch"], in_dim, tmeta["hidden"], layers=tmeta.get("layers", 3))
    target_enc.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target_enc.eval()

    # Positives & negatives
    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    encoders = [target_enc] + [load_encoder_from_pt(p, in_dim) for p in pos_paths] + \
               [load_encoder_from_pt(n, in_dim) for n in neg_paths]
    labels = [1.0] * (1 + len(pos_paths)) + [0.0] * len(neg_paths)

    # Build feature matrix X and labels y
    with torch.no_grad():
        z0 = concat_for_model(encoders[0], decoder, fps)
        D = z0.numel()
        if ver_in_dim_saved is not None and D != int(ver_in_dim_saved):
            raise RuntimeError(
                f"Verifier input mismatch: dataset dim {D} vs saved ver_in_dim {ver_in_dim_saved}"
            )

        X_rows = [z0] + [concat_for_model(enc, decoder, fps) for enc in encoders[1:]]
        X = torch.stack(X_rows, dim=0).float()         # [N, D]
        y = torch.tensor(labels, dtype=torch.float32)  # [N]

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y}, args.out)
    print(f"Saved {args.out} with {X.shape[0]} rows; dim={X.shape[1]}")
    print(f"Positives: {int(sum(labels))} | Negatives: {len(labels) - int(sum(labels))}")


if __name__ == "__main__":
    main()
