import argparse, json, random, torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.datasets import Planetoid
from gcn_nc import get_model

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_masks(n, train_p=0.7, val_p=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_tr = int(train_p*n); n_va = int(val_p*n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    mtr = torch.zeros(n, dtype=torch.bool); mtr[tr]=True
    mva = torch.zeros(n, dtype=torch.bool); mva[va]=True
    mte = torch.zeros(n, dtype=torch.bool); mte[te]=True
    return mtr, mva, mte

def train_epoch(model, data, opt, mask):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask]); loss.backward(); opt.step()
    return float(loss.item())

@torch.no_grad()
def eval_mask(model, data, mask):
    model.eval(); out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    return float((pred[mask]==data.y[mask]).float().mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='sage', help='gcn or sage (unrelated to target)')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=9999, help='use a NEW seed unseen by the Univerifier')
    ap.add_argument('--out_dir', default='models/suspects')
    ap.add_argument('--name', default='neg_nc_seed9999')
    args = ap.parse_args()

    set_seed(args.seed)
    ds = Planetoid(root='data/cora', name='Cora')
    data = ds[0]
    mtr, mva, mte = make_masks(data.num_nodes, 0.7, 0.1, seed=args.seed)
    data.train_mask, data.val_mask, data.test_mask = mtr, mva, mte

    model = get_model(args.arch, ds.num_features, args.hidden, ds.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val, best_state = -1.0, None
    for _ in range(args.epochs):
        _ = train_epoch(model, data, opt, data.train_mask)
        val = eval_mask(model, data, data.val_mask)
        if val > best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pt = f"{args.out_dir}/{args.name}.pt"
    meta = {
        "arch": args.arch, "hidden": args.hidden,
        "in_dim": ds.num_features, "num_classes": ds.num_classes,
        "seed": args.seed, "note": "never-seen negative suspect"
    }
    torch.save(model.state_dict(), pt)
    with open(pt.replace('.pt','.json'), 'w') as f: json.dump(meta, f)
    print(f"[saved] {pt} (val_acc={best_val:.4f})")

if __name__ == '__main__':
    main()
