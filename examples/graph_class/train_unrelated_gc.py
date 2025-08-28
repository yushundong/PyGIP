# Train NEGATIVE (unrelated) GRAPH-CLASSIFICATION models on ENZYMES from scratch.

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

from gsage_gc import get_model


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(val_ratio * n))
    n_test = int(round(test_ratio * n))
    n_train = n - n_val - n_test
    idx_tr = perm[:n_train].tolist()
    idx_va = perm[n_train:n_train + n_val].tolist()
    idx_te = perm[n_train + n_val:].tolist()
    return idx_tr, idx_va, idx_te


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_graphs = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(1, total_graphs)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        loss = F.cross_entropy(out, batch.y)
        pred = out.argmax(dim=-1)
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
        total_loss += float(loss.item()) * batch.num_graphs
    acc = correct / max(1, total)
    return acc, (total_loss / max(1, total))


def main():
    ap = argparse.ArgumentParser(description="Train unrelated GC (negative) models on ENZYMES")
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--archs', type=str, default='gsage')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--start_index', type=int, default=50)

    args = ap.parse_args()

    device = torch.device(args.device)
    Path("models/negatives").mkdir(parents=True, exist_ok=True)

    dataset_full = TUDataset(root='data/ENZYMES', name='ENZYMES',
                             use_node_attr=True, transform=NormalizeFeatures())
    in_dim = dataset_full.num_features
    num_classes = dataset_full.num_classes

    arch_list = [a.strip() for a in args.archs.split(',') if a.strip()]
    saved = []

    for i in range(args.count):
        idx = args.start_index + i
        seed_i = args.seed + idx
        set_seed(seed_i)

        n_graphs = len(dataset_full)
        tr_idx, va_idx, te_idx = split_indices(n_graphs, args.val_ratio, args.test_ratio, seed=seed_i)
        train_set = dataset_full[tr_idx]
        val_set   = dataset_full[va_idx]
        test_set  = dataset_full[te_idx]

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        arch = arch_list[idx % len(arch_list)]
        model = get_model(arch, in_dim, args.hidden, num_classes,
                          num_layers=args.layers, dropout=args.dropout, pool="mean").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val, best_state = -1.0, None
        for ep in range(1, args.epochs + 1):
            _ = train_one_epoch(model, train_loader, opt, device)
            val_acc, _ = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if ep % 20 == 0 or ep == args.epochs:
                print(f"[neg {idx:03d} | {arch}] epoch {ep:03d} | val acc {val_acc:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        test_acc, test_loss = evaluate(model, test_loader, device)

        out_path = f"models/negatives/negative_gc_{idx:03d}.pt"
        torch.save(model.state_dict(), out_path)
        meta = {
            "task": "graph_classification",
            "dataset": "ENZYMES",
            "arch": arch,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "seed": seed_i,
            "val_acc": float(best_val),
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
        }
        with open(out_path.replace('.pt', '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        saved.append(out_path)
        print(f"Saved NEGATIVE {idx:03d} arch={arch} best_val_acc={best_val:.4f} "
              f"test_acc={test_acc:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
