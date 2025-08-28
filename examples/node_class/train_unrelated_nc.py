
"""
Negative models: different random seeds and/or architectures trained from scratch on the same train split.
"""
import argparse, torch, random, json
from pathlib import Path
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from gcn_nc import get_model

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_masks(num_nodes, train_p=0.7, val_p=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(num_nodes, generator=g)
    n_train = int(train_p * num_nodes)
    n_val = int(val_p * num_nodes)
    train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]; test_idx = idx[n_train+n_val:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[train_idx]=True
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool); val_mask[val_idx]=True
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool); test_mask[test_idx]=True
    return train_mask, val_mask, test_mask

def train_epoch(model, data, optimizer, mask):
    model.train(); optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask])
    loss.backward(); optimizer.step()
    return float(loss.item())

@torch.no_grad()
def eval_mask(model, data, mask):
    model.eval(); out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    return float((pred[mask]==data.y[mask]).float().mean())

def save_model(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists
    torch.save(model.state_dict(), str(path))
    with open(str(path).replace('.pt', '.json'), 'w') as f:
        json.dump(meta, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=50)
    ap.add_argument('--archs', type=str, default='gcn,sage')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out_dir', type=str, default='models/negatives')  # <-- where to save
    args = ap.parse_args()

    dataset = Planetoid(root='data/cora', name='Cora')
    data = dataset[0]

    saved = []
    arch_list = args.archs.split(',')

    for i in range(args.count):
        seed_i = args.seed + i
        set_seed(seed_i)
        train_mask, val_mask, test_mask = make_masks(data.num_nodes, 0.7, 0.1, seed=seed_i)
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

        arch = arch_list[i % len(arch_list)]
        model = get_model(arch, data.num_features, 64, dataset.num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val, best_state = -1, None
        for ep in range(args.epochs):
            loss = train_epoch(model, data, opt, data.train_mask)
            val = eval_mask(model, data, data.val_mask)
            if val > best_val:
                best_val, best_state = val, {k:v.cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state)

        out_path = Path(args.out_dir) / f"negative_nc_{i:03d}.pt"
        meta = {"arch": arch, "hidden": 64, "num_classes": dataset.num_classes, "seed": seed_i}
        save_model(model, out_path, meta)
        
        saved.append(str(out_path))
        print(f"Saved negative {i} arch={arch} val={best_val:.4f} -> {out_path}")

if __name__ == '__main__':
    main()
