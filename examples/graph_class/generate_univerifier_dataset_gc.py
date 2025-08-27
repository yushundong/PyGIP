"""
Build a Univerifier dataset from saved GC fingerprints on ENZYMES.
Label 1 for positives ({target ∪ F+}) and 0 for negatives (F−).
"""

import argparse, glob, json, torch
from pathlib import Path
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import dense_to_sparse, to_undirected
from torch_geometric.transforms import NormalizeFeatures

from gsage_gc import get_model


def list_paths_from_globs(globs_str):
    globs = [g.strip() for g in globs_str.split(",") if g.strip()]
    paths = []
    for g in globs:
        paths.extend(glob.glob(g))
    return sorted(paths)


def load_model_from_pt(pt_path, in_dim, num_classes):
    meta_path = pt_path.replace(".pt", ".json")
    j = json.load(open(meta_path, "r"))
    m = get_model(
        j["arch"], in_dim, j["hidden"], num_classes,
        num_layers=j.get("layers", 3), dropout=j.get("dropout", 0.5), pool="mean"
    )
    m.load_state_dict(torch.load(pt_path, map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def forward_on_fp(model, fp):
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

    batch = X.new_zeros(n, dtype=torch.long)
    logits = model(X, edge_index, batch=batch)
    return logits.squeeze(0)


@torch.no_grad()
def concat_for_model(model, fps):
    parts = [forward_on_fp(model, fp) for fp in fps]
    return torch.cat(parts, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints_path", type=str, default="fingerprints/fingerprints_gc.pt")
    ap.add_argument("--target_path", type=str, default="models/target_model_gc.pt")
    ap.add_argument("--target_meta", type=str, default="models/target_meta_gc.json")
    ap.add_argument("--positives_glob", type=str,
                    default="models/positives/gc_ftpr_*.pt,models/positives/distill_gc_*.pt")
    ap.add_argument("--negatives_glob", type=str, default="models/negatives/negative_gc_*.pt")
    ap.add_argument("--out", type=str, default="fingerprints/univerifier_dataset_gc.pt")
    args = ap.parse_args()

    ds = TUDataset(root="data/ENZYMES", name="ENZYMES", use_node_attr=True, transform=NormalizeFeatures())
    in_dim = ds.num_features
    num_classes = ds.num_classes

    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = pack.get("ver_in_dim", None)

    tmeta = json.load(open(args.target_meta, "r"))
    target = get_model(
        tmeta.get("arch", "gsage"), in_dim, tmeta.get("hidden", 64), num_classes,
        num_layers=tmeta.get("layers", 3), dropout=tmeta.get("dropout", 0.5), pool="mean"
    )
    target.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target.eval()

    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models = [target] + [load_model_from_pt(p, in_dim, num_classes) for p in pos_paths] + \
             [load_model_from_pt(n, in_dim, num_classes) for n in neg_paths]
    labels = [1.0] * (1 + len(pos_paths)) + [0.0] * len(neg_paths)

    # Build feature matrix X and labels y
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

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y}, args.out)
    print(f"Saved {args.out} with {X.shape[0]} rows; dim={X.shape[1]}")
    print(f"Positives: {int(sum(labels))} | Negatives: {len(labels) - int(sum(labels))}")


if __name__ == "__main__":
    main()
