# Graph classification on ENZYMES using GraphSAGE.

import argparse
import json
import os
import random
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from gsage_gc import get_model


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from collections import defaultdict

def split_indices_stratified(y, val_ratio=0.1, test_ratio=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    by_cls = defaultdict(list)
    for i, yi in enumerate(y.tolist()):
        by_cls[int(yi)].append(i)
    tr, va, te = [], [], []
    for cls, idxs in by_cls.items():
        idxs = torch.tensor(idxs)[torch.randperm(len(idxs), generator=g)].tolist()
        n = len(idxs)
        n_val = int(round(val_ratio * n))
        n_test = int(round(test_ratio * n))
        n_train = n - n_val - n_test
        tr += idxs[:n_train]
        va += idxs[n_train:n_train+n_val]
        te += idxs[n_train+n_val:]
    return tr, va, te


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0
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
    correct = 0
    total = 0
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        loss = F.cross_entropy(out, batch.y)
        pred = out.argmax(dim=-1)
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
        total_loss += float(loss.item()) * batch.num_graphs
    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gsage", choices=["gsage", "graphsage", "sage"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Dataset: ENZYMES (graph classification) ---
    dataset = TUDataset(root="data/ENZYMES", name="ENZYMES", use_node_attr=True, transform=NormalizeFeatures())
    num_graphs = len(dataset)
    in_dim = dataset.num_features
    num_classes = dataset.num_classes

    # split indices
    y_all = torch.tensor([data.y.item() for data in dataset])
    tr_idx, va_idx, te_idx = split_indices_stratified(y_all, args.val_ratio, args.test_ratio, args.seed)
    train_set = dataset[tr_idx]
    val_set = dataset[va_idx]
    test_set = dataset[te_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = get_model(
        args.arch,
        in_dim=in_dim,
        hidden=args.hidden,
        num_classes=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
        pool="mean",
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_acc, val_loss = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"val acc {val_acc:.4f} | val loss {val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_loss = evaluate(model, test_loader, device)
    print(f"Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save target GC model + meta (GC-specific filenames)
    torch.save(model.state_dict(), "models/target_model_gc.pt")
    with open("models/target_meta_gc.json", "w") as f:
        json.dump(
            {
                "task": "graph_classification",
                "dataset": "ENZYMES",
                "arch": args.arch,
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "metrics": {"val_acc": float(best_val_acc), "test_acc": float(test_acc)},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
