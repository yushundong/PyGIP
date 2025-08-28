import argparse, json, random, torch, torch.nn.functional as F
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph
from gcn_nc import get_model

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_masks_like(train_seed=0):
    ds = Planetoid(root='data/cora', name='Cora')
    data = ds[0]
    g = torch.Generator().manual_seed(train_seed)
    idx = torch.randperm(data.num_nodes, generator=g)
    n_tr = int(0.7*data.num_nodes); n_va = int(0.1*data.num_nodes)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    mtr = torch.zeros(data.num_nodes, dtype=torch.bool); mtr[tr]=True
    mva = torch.zeros(data.num_nodes, dtype=torch.bool); mva[va]=True
    mte = torch.zeros(data.num_nodes, dtype=torch.bool); mte[te]=True
    data.train_mask, data.val_mask, data.test_mask = mtr, mva, mte
    return ds, data

@torch.no_grad()
def teacher_logits_on_nodes(model, x, edge_index, nodes):
    model.eval()
    out = model(x, edge_index)
    return out[nodes]

def sample_node_subgraph(num_nodes, low=0.5, high=0.8):
    k = int(random.uniform(low, high) * num_nodes)
    idx = torch.randperm(num_nodes)[:k]
    return idx.sort().values

def kd_loss(student_logits, teacher_logits):
    return F.mse_loss(student_logits, teacher_logits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_path', default='models/target_meta_nc.json')
    ap.add_argument('--target_path', default='models/target_model_nc.pt')
    ap.add_argument('--archs', default='gat,sage')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--count_per_arch', type=int, default=50)
    ap.add_argument('--out_dir', type=str, default='models/positives')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.meta_path,'r') as f:
        meta = json.load(f)
    ds, data = make_masks_like(train_seed=args.seed)
    in_dim, num_classes = ds.num_features, ds.num_classes

    teacher = get_model(meta['arch'], in_dim, meta['hidden'], num_classes)
    teacher.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    teacher.eval()

    archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    saved = []
    for arch in archs:
        for i in range(args.count_per_arch):
            student = get_model(arch, in_dim, 64, num_classes)
            opt = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)

            for _ in range(args.epochs):
                student.train(); opt.zero_grad()
                idx = sample_node_subgraph(data.num_nodes, 0.5, 0.8)
                e_idx, _ = subgraph(idx, data.edge_index, relabel_nodes=True)
                x_sub = data.x[idx]
                with torch.no_grad():
                    t_logits = teacher_logits_on_nodes(teacher, data.x, data.edge_index, idx)
                s_logits = student(x_sub, e_idx)
                loss = kd_loss(s_logits, t_logits)
                loss.backward(); opt.step()

            out_pt = f'{args.out_dir}/distill_nc_{arch}_{i:03d}.pt'
            torch.save(student.state_dict(), out_pt)
            with open(out_pt.replace('.pt','.json'),'w') as f:
                json.dump({"arch": arch, "hidden": 64, "num_classes": num_classes, "pos_kind": "distill"}, f)
            saved.append(out_pt)
            print(f"[distill-nc] saved {out_pt}")
    print(f"Saved {len(saved)} distilled positives.")

if __name__ == '__main__':
    main()
