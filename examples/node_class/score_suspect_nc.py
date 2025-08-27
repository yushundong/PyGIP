import argparse, json, os, torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

#  Univerifier 
class FPVerifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 64),     nn.LeakyReLU(),
            nn.Linear(64, 32),      nn.LeakyReLU(),
            nn.Linear(32, 1),       nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

def load_json(p): 
    with open(p, "r") as f: return json.load(f)

def edge_index_from_A(A: torch.Tensor) -> torch.Tensor:
    A_bin = (A > 0.5).float()
    A_sym = torch.triu(A_bin, diagonal=1); A_sym = A_sym + A_sym.t()
    ei = dense_to_sparse(A_sym)[0]
    if ei.numel() == 0:
        n = A.size(0); ei = torch.arange(n).repeat(2,1)
    return ei

@torch.no_grad()
def build_z_nc(model: nn.Module, fps):
    parts = []
    for fp in fps:
        X, A, idx = fp["X"], fp["A"], fp["node_idx"]
        ei = edge_index_from_A(A)
        logits = model(X, ei)
        parts.append(logits[idx, :].reshape(-1))
    return torch.cat(parts, dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints_path", default="fingerprints/fingerprints_nc.pt")
    ap.add_argument("--verifier_path", default="fingerprints/univerifier_nc.pt",
                    help="If missing, load 'verifier' from fingerprints pack.")
    ap.add_argument("--suspect_pt", required=True)
    ap.add_argument("--suspect_meta", required=False, default="")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--in_dim", type=int, default=1433)
    ap.add_argument("--num_classes", type=int, default=7)
    args = ap.parse_args()

    device = torch.device(args.device)
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]; ver_in_dim = int(pack.get("ver_in_dim", 0))

    # Build suspect NC model
    from gcn_nc import get_model
    meta = load_json(args.suspect_meta) if args.suspect_meta else {}
    arch   = meta.get("arch", "gcn")
    hidden = int(meta.get("hidden", 64))
    in_dim = int(meta.get("in_dim", args.in_dim))
    num_classes = int(meta.get("num_classes", args.num_classes))
    model = get_model(arch, in_dim, hidden, num_classes).to(device)
    model.load_state_dict(torch.load(args.suspect_pt, map_location="cpu"))
    model.eval()

    z = build_z_nc(model, fps)
    D = z.numel()
    if ver_in_dim and ver_in_dim != D:
        raise RuntimeError(f"Dim mismatch: verifier expects {ver_in_dim}, got {D}.")

    # Load verifier
    V = FPVerifier(D).to(device)
    if os.path.isfile(args.verifier_path):
        V.load_state_dict(torch.load(args.verifier_path, map_location="cpu"))
        src = args.verifier_path
    else:
        if "verifier" not in pack: raise FileNotFoundError("No verifier found.")
        V.load_state_dict(pack["verifier"]); src = f"{args.fingerprints_path}:[verifier]"
    V.eval()

    with torch.no_grad():
        s = float(V(z.view(1, -1).to(device)).item())
    verdict = "OWNED (positive)" if s >= args.threshold else "NOT-OWNED (negative)"
    print(f"Score={s:.6f} | τ={args.threshold:.3f} -> {verdict}")

if __name__ == "__main__":
    main()
