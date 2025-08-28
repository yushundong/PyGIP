# Positive (pirated) models for GRAPH CLASSIFICATION on ENZYMES via DISTILLATION.
# Teacher: trained GC model loaded from target_model_gc.pt
# Students: GraphSAGE via get_model

import argparse, json, random, torch
from pathlib import Path

import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

from gsage_gc import get_model


def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def kd_loss(student_logits, teacher_logits):
    return F.mse_loss(student_logits, teacher_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_path', default='models/target_meta_gc.json')
    ap.add_argument('--target_path', default='models/target_model_gc.pt')
    ap.add_argument('--archs', default='gsage')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--count_per_arch', type=int, default=100)
    ap.add_argument('--out_dir', type=str, default='models/positives')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--student_hidden', type=int, default=64)
    ap.add_argument('--student_layers', type=int, default=3)
    ap.add_argument('--student_dropout', type=float, default=0.5)
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TUDataset(root='data/ENZYMES', name='ENZYMES',
                        use_node_attr=True, transform=NormalizeFeatures())
    in_dim = dataset.num_features
    num_classes = dataset.num_classes
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Teacher GC model
    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    arch_t   = meta.get('arch', 'gsage')
    hidden_t = meta.get('hidden', 64)
    layers_t = meta.get('layers', 3)
    drop_t   = meta.get('dropout', 0.5)

    teacher = get_model(arch_t, in_dim, hidden_t, num_classes,
                        num_layers=layers_t, dropout=drop_t, pool="mean").to(device)
    teacher.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    teacher.eval()

    archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    saved = []

    for arch in archs:
        for i in range(args.count_per_arch):
            # fresh student
            student = get_model(arch, in_dim, args.student_hidden, num_classes,
                                num_layers=args.student_layers,
                                dropout=args.student_dropout, pool="mean").to(device)
            opt = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)

            for _ in range(args.epochs):
                student.train()
                for batch in loader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        t_logits = teacher(batch.x, batch.edge_index, batch=batch.batch)  # [B, C]
                    s_logits = student(batch.x, batch.edge_index, batch=batch.batch)      # [B, C]
                    loss = kd_loss(s_logits, t_logits)
                    opt.zero_grad(); loss.backward(); opt.step()

            # save student
            out_pt = f'{args.out_dir}/distill_gc_{arch}_{i:03d}.pt'
            torch.save(student.state_dict(), out_pt)
            with open(out_pt.replace('.pt', '.json'), 'w') as f:
                json.dump({
                    "task": "graph_classification",
                    "dataset": "ENZYMES",
                    "arch": arch,
                    "hidden": args.student_hidden,
                    "layers": args.student_layers,
                    "dropout": args.student_dropout,
                    "pos_kind": "distill",
                    "teacher_arch": arch_t,
                    "teacher_hidden": hidden_t,
                    "teacher_layers": layers_t,
                    "teacher_dropout": drop_t
                }, f, indent=2)

            saved.append(out_pt)
            print(f"[distill-gc] saved {out_pt}")

    print(f"Saved {len(saved)} distilled GC positives.")


if __name__ == '__main__':
    main()
