# Distill LINK PREDICTION students on CiteSeer from a trained LP teacher
# Teacher/Student: encoder (GCN/SAGE/GAT) + dot-product decoder

import argparse, json, random, torch
from pathlib import Path
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph, negative_sampling
from gcn_lp import get_encoder, DotProductDecoder


def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def sample_node_subset(num_nodes: int, low: float = 0.5, high: float = 0.8):
    k = max(2, int(random.uniform(low, high) * num_nodes))
    idx = torch.randperm(num_nodes)[:k]
    return idx.sort().values


@torch.no_grad()
def teacher_edge_logits(teacher_enc, teacher_dec, x, edge_index, pos_edge, neg_edge, device):
    teacher_enc.eval()
    z_t = teacher_enc(x.to(device), edge_index.to(device))
    t_pos = teacher_dec(z_t, pos_edge.to(device))
    t_neg = teacher_dec(z_t, neg_edge.to(device))
    return t_pos.detach(), t_neg.detach()


def kd_loss(student_logits, teacher_logits, kind: str = "mse"):
    if kind == "mse":
        return F.mse_loss(student_logits, teacher_logits)
    elif kind == "bce_soft":
        with torch.no_grad():
            soft = torch.sigmoid(teacher_logits)
        return F.binary_cross_entropy_with_logits(student_logits, soft)
    else:
        raise ValueError(f"Unknown distill loss kind: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_path', default='models/target_meta_lp.json')
    ap.add_argument('--target_path', default='models/target_model_lp.pt')
    ap.add_argument('--archs', default='gat,sage')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--count_per_arch', type=int, default=50)
    ap.add_argument('--out_dir', type=str, default='models/positives')
    ap.add_argument('--student_hidden', type=int, default=64)
    ap.add_argument('--student_layers', type=int, default=3)
    ap.add_argument('--distill_loss', choices=['mse', 'bce_soft'], default='mse')
    ap.add_argument('--sub_low', type=float, default=0.5)       # subgraph ratio lower bound
    ap.add_argument('--sub_high', type=float, default=0.8)      # subgraph ratio upper bound
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    arch_t = meta.get('arch', 'gcn')
    hidden_t = meta.get('hidden', 64)
    layers_t = meta.get('layers', 3)

    dataset = Planetoid(root='data', name='CiteSeer')
    data = dataset[0]

    teacher_enc = get_encoder(arch_t, dataset.num_node_features, hidden_t,
                              num_layers=layers_t, dropout=0.5)
    teacher_enc.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    teacher_enc.to(device).eval()
    t_dec = DotProductDecoder().to(device)

    archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    saved = []

    for arch in archs:
        for i in range(args.count_per_arch):
            student = get_encoder(arch, dataset.num_node_features, args.student_hidden,
                                  num_layers=args.student_layers, dropout=0.5).to(device)
            s_dec = DotProductDecoder().to(device)
            opt = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)

            for _ in range(args.epochs):
                student.train(); opt.zero_grad()

                # sample a subgraph (50–80% nodes by default)
                idx = sample_node_subset(data.num_nodes, args.sub_low, args.sub_high)
                e_idx, _ = subgraph(idx, data.edge_index, relabel_nodes=True)
                if e_idx.numel() == 0 or e_idx.size(1) == 0:
                    continue

                x_sub = data.x[idx]

                # positives = subgraph edges; negatives = sampled non-edges
                pos_edge = e_idx
                neg_edge = negative_sampling(
                    edge_index=pos_edge,
                    num_nodes=x_sub.size(0),
                    num_neg_samples=pos_edge.size(1),
                    method='sparse'
                )

                t_pos, t_neg = teacher_edge_logits(
                    teacher_enc, t_dec, x_sub, e_idx, pos_edge, neg_edge, device
                )

                z_s = student(x_sub.to(device), e_idx.to(device))
                s_pos = s_dec(z_s, pos_edge.to(device))
                s_neg = s_dec(z_s, neg_edge.to(device))

                s_all = torch.cat([s_pos, s_neg], dim=0)
                t_all = torch.cat([t_pos, t_neg], dim=0)
                loss = kd_loss(s_all, t_all, kind=args.distill_loss)

                loss.backward(); opt.step()

            out_pt = f'{args.out_dir}/distill_lp_{arch}_{i:03d}.pt'
            torch.save(student.state_dict(), out_pt)
            with open(out_pt.replace('.pt', '.json'), 'w') as f:
                json.dump({
                    "task": "link_prediction",
                    "dataset": "CiteSeer",
                    "arch": arch,
                    "hidden": args.student_hidden,
                    "layers": args.student_layers,
                    "pos_kind": "distill",
                    "teacher_arch": arch_t,
                    "teacher_hidden": hidden_t,
                    "teacher_layers": layers_t,
                    "distill_loss": args.distill_loss
                }, f, indent=2)

            saved.append(out_pt)
            print(f"[distill] saved {out_pt}")

    print(f"Saved {len(saved)} distilled LP positives.")


if __name__ == '__main__':
    main()
