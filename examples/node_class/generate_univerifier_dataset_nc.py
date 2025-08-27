"""
Build a Univerifier dataset from saved fingerprints.
Label 1 for positives ({target ∪ F+}) and 0 for negatives (F−).
Outputs: a .pt file with:
    - X: [N_models, D] tensor, where D = P * m * num_classes
    - y: [N_models] float tensor with 1.0 (positive) or 0.0 (negative)
"""

import argparse, glob, json, torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from gcn_nc import get_model

@torch.no_grad()
def forward_on_fp(model, fp):
    X = fp["X"]
    A = fp["A"]
    idx = fp["node_idx"]

    # Binarize & symmetrize adjacency
    A_bin = (A > 0.5).float()
    A_sym = torch.triu(A_bin, diagonal=1)
    A_sym = A_sym + A_sym.t()
    edge_index = dense_to_sparse(A_sym)[0]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints_path", type=str, default="fingerprints/fingerprints_nc.pt")
    ap.add_argument("--target_path", type=str, default="models/target_model_nc.pt")
    ap.add_argument("--target_meta", type=str, default="models/target_meta_nc.json")
    ap.add_argument("--positives_glob", type=str,
                    default="models/positives/nc_ftpr_*.pt,models/positives/distill_nc_*.pt")
    ap.add_argument("--negatives_glob", type=str, default="models/negatives/negative_nc_*.pt")
    ap.add_argument("--out", type=str, default="fingerprints/univerifier_dataset_nc.pt")
    args = ap.parse_args()

    # Dataset dims (for model reconstruction)
    ds = Planetoid(root="data/cora", name="Cora")
    in_dim = ds.num_features
    num_classes = ds.num_classes

    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = pack.get("ver_in_dim", None)


    tmeta = json.load(open(args.target_meta, "r"))
    target = get_model(tmeta["arch"], in_dim, tmeta["hidden"], tmeta["num_classes"])
    target.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target.eval()

    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models = [target]
    labels = [1.0]

    for p in pos_paths:
        models.append(load_model_from_pt(p, in_dim)); labels.append(1.0)
    for n in neg_paths:
        models.append(load_model_from_pt(n, in_dim)); labels.append(0.0)

    with torch.no_grad():
        z0 = concat_for_model(models[0], fps)
        D = z0.numel()
        if ver_in_dim_saved is not None and D != int(ver_in_dim_saved):
            raise RuntimeError(
                f"Verifier input mismatch: dataset dim {D} vs saved ver_in_dim {ver_in_dim_saved}"
            )

        X_rows = [z0] + [concat_for_model(m, fps) for m in models[1:]]
        X = torch.stack(X_rows, dim=0).float()
        y = torch.tensor(labels, dtype=torch.float32)

    torch.save({"X": X, "y": y}, args.out)
    print(f"Saved {args.out} with {X.shape[0]} rows; dim={X.shape[1]} (num_classes={num_classes})")
    print(f"Positives: {int(sum(labels))} | Negatives: {len(labels) - int(sum(labels))}")

if __name__ == "__main__":
    main()
