import argparse, torch, copy, random, json
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

def reinit_last_layer(model):
    last = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    if last is not None:
        for p in last.parameters():
            if p.dim() > 1: torch.nn.init.xavier_uniform_(p)
            else: torch.nn.init.zeros_(p)

def reinit_all(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        if hasattr(m, 'reset_parameters'):
            try: m.reset_parameters()
            except: pass

def save_model(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    with open(str(path).replace('.pt','.json'),'w') as f:
        json.dump(meta, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_path', type=str, default='models/target_model_nc.pt')
    ap.add_argument('--meta_path', type=str, default='models/target_meta_nc.json')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_variants', type=int, default=100)
    ap.add_argument('--out_dir', type=str, default='models/positives')
    args = ap.parse_args()

    set_seed(args.seed)
    with open(args.meta_path,'r') as f:
        meta = json.load(f)

    dataset = Planetoid(root='data/cora', name='Cora')
    data = dataset[0]
    train_mask, val_mask, test_mask = make_masks(data.num_nodes, 0.7, 0.1, seed=args.seed)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    target = get_model(meta["arch"], data.num_features, meta["hidden"], meta["num_classes"])
    target.load_state_dict(torch.load(args.target_path, map_location='cpu'))

    saved = []
    for i in range(args.num_variants):
        kind = i % 4  # 0:FT-last,1:FT-all,2:PR-last,3:PR-all
        m = get_model(meta["arch"], data.num_features, meta["hidden"], meta["num_classes"])
        m.load_state_dict(target.state_dict())

        if kind == 2:      reinit_last_layer(m)
        elif kind == 3:    reinit_all(m)

        if kind in (0,2):
            for name,p in m.named_parameters():
                p.requires_grad = ('conv2' in name) or ('mlp.3' in name)
        else:
            for p in m.parameters(): p.requires_grad=True

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr, weight_decay=args.wd)
        best_val, best_state = -1, None
        for _ in range(args.epochs):
            _ = train_epoch(m, data, opt, data.train_mask)
            val = eval_mask(m, data, data.val_mask)
            if val > best_val:
                best_val, best_state = val, {k:v.cpu().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best_state)
        out_path = f"{args.out_dir}/nc_ftpr_{i:03d}.pt"
        meta_out = {"arch": meta["arch"], "hidden": meta["hidden"], "num_classes": meta["num_classes"], "pos_kind": ["ft_last","ft_all","pr_last","pr_all"][kind]}
        save_model(m, out_path, meta_out)
        saved.append(out_path)
        print(f"Saved {out_path}  val={best_val:.4f}")

    print(f"Total FT/PR positives saved: {len(saved)}")

if __name__ == '__main__':
    main()
