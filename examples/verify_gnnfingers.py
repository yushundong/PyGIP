import os
import sys
import glob
import argparse
import torch

from datasets import Cora
from models.defense import GNNFingers
from models.defense.GNNFingers import GCN 

def _load_dataset():
    return Cora(api_type='pyg')

def _make_suspect(arch: str, in_ch: int, hidden: int, out_ch: int, depth: int):
    arch = arch.lower()
    if arch == "gcn":
        return GCN(in_ch, hidden, out_ch, depth=depth)
    raise ValueError(f"Unsupported arch '{arch}'. Supported: gcn")

def verify_single(defense: GNNFingers, suspect_sd_path: str, arch: str, hidden: int, depth: int):
    data = defense.dataset.graph_data
    in_ch = data.num_features
    out_ch = int(data.y.max().item()) + 1
    model = _make_suspect(arch, in_ch, hidden, out_ch, depth).to(defense.device).eval()
    state = torch.load(suspect_sd_path, map_location="cpu")
    model.load_state_dict(state)
    return defense.verify(model)

def main():
    ap = argparse.ArgumentParser(description="Verify models with a saved GNNFingers registry.")
    ap.add_argument("--registry", default="registry/fingerprints.pt",
                    help="Path to fingerprints registry (.pt). Default: registry/fingerprints.pt")
    ap.add_argument("--suspect", default=None,
                    help="Path to a single suspect state_dict (.pt).")
    ap.add_argument("--suspects_dir", default=None,
                    help="Directory with suspect .pt files (verified in sorted order).")
    ap.add_argument("--arch", default="gcn", help="Suspect backbone arch (default: gcn)")
    ap.add_argument("--hidden", type=int, default=128, help="Suspect hidden width (default: 128)")
    ap.add_argument("--depth", type=int, default=3, help="Suspect depth (default: 3)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Verification threshold τ (default: 0.5)")
    ap.add_argument("--owner", action="store_true", help="Also verify the owner (sanity).")
    args = ap.parse_args()

    if (args.suspect is None) and (args.suspects_dir is None) and (not args.owner):
        print("Nothing to do: pass --suspect or --suspects_dir or --owner.", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = _load_dataset()
    defense = GNNFingers(
        dataset=dataset,
        fingerprint=None,
        n_pos=0, n_neg=0,
        owner_epochs=0,
        model_path="ckpts/owner.pt",
        save_dir=os.path.dirname(args.registry) or ".",
        device=device,
    )

    # Load registry payload directly (don’t re-train)
    payload = torch.load(args.registry, map_location="cpu")

    if args.owner:
        owner, _, _ = defense._load_or_train_owner(dataset.graph_data, device)
        r = defense.verify(owner, fingerprints=payload, threshold=args.threshold)
        print(f"OWNER, o_plus={r['o_plus']:.4f}, verified={r['verified']} (τ={r['threshold']})")

    if args.suspect:
        r = verify_single(defense, args.suspect, args.arch, args.hidden, args.depth)
        print(f"{os.path.basename(args.suspect)}, o_plus={r['o_plus']:.4f}, verified={r['verified']} (τ={r['threshold']})")

    if args.suspects_dir:
        paths = sorted(glob.glob(os.path.join(args.suspects_dir, "*.pt")))
        if not paths:
            print(f"No .pt files found under {args.suspects_dir}", file=sys.stderr)
            sys.exit(1)
        print("model_path,o_plus,verified,threshold")
        for p in paths:
            r = verify_single(defense, p, args.arch, args.hidden, args.depth)
            print(f"{os.path.basename(p)},{r['o_plus']:.6f},{int(r['verified'])},{r['threshold']:.2f}")

if __name__ == "__main__":
    main()
