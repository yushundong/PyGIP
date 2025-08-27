"""
Trains the Univerifier on features built from fingerprints (MLP: [128,64,32] + LeakyReLU).
Loads X,y from generate_univerifier_dataset.py and saves weights + a tiny meta JSON.
"""

import argparse, json, torch, time
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='fingerprints/univerifier_dataset_nc.pt')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--fingerprints_path', type=str, default='fingerprints/fingerprints_nc.pt')
    ap.add_argument('--out', type=str, default='fingerprints/univerifier_nc.pt')
    args = ap.parse_args()

    # Load dataset
    pack = torch.load(args.dataset, map_location='cpu')
    X = pack['X'].float().detach()
    y = pack['y'].float().view(-1, 1).detach()
    N, D = X.shape

    try:
        fp_pack = torch.load(args.fingerprints_path, map_location='cpu')
        ver_in_dim = int(fp_pack.get('ver_in_dim', D))
        if ver_in_dim != D:
            raise RuntimeError(f'Input dim mismatch: dataset dim {D} vs ver_in_dim {ver_in_dim}')
    except FileNotFoundError:
        pass

    # Train/val split
    n_val = max(1, int(args.val_split * N))
    perm = torch.randperm(N)
    idx_tr, idx_val = perm[:-n_val], perm[-n_val:]
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]

    # Model/optim
    V = FPVerifier(D)
    opt = torch.optim.Adam(V.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        V.train(); opt.zero_grad()
        p = V(X_tr)
        loss = F.binary_cross_entropy(p, y_tr)
        loss.backward(); opt.step()

        with torch.no_grad():
            V.eval()
            pv = V(X_val)
            val_loss = F.binary_cross_entropy(pv, y_val)
            val_acc = ((pv >= 0.5).float() == y_val).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in V.state_dict().items()}

        if ep % 20 == 0 or ep == args.epochs:
            print(f'Epoch {ep:03d} | train_bce {loss.item():.4f} '
                  f'| val_bce {val_loss.item():.4f} | val_acc {val_acc:.4f}')

    # Save best
    if best_state is None:
        best_state = V.state_dict()
    Path('fingerprints').mkdir(exist_ok=True, parents=True)
    torch.save(best_state, args.out)
    with open(args.out.replace('.pt', '_meta.json'), 'w') as f:
        json.dump({'in_dim': D, 'hidden': [128, 64, 32], 'act': 'LeakyReLU'}, f)
    print(f'Saved {args.out} | Best Val Acc {best_acc:.4f} | Input dim D={D}')

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("time taken: ", (end_time-start_time)/60 )

