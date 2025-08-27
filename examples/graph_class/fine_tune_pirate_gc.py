# Create positive (pirated) GC models on ENZYMES by fine-tuning / partial-retraining
# a trained target GraphSAGE GC model.

import argparse, json, random, copy
from pathlib import Path

import torch
import torch.nn as nn
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
    idx_va = perm[n_train:n_train+n_val].tolist()
    idx_te = perm[n_train+n_val:].tolist()
    return idx_tr, idx_va, idx_te


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_graphs = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward(); optimizer.step()
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


def reinit_classifier(model: nn.Module):
    if not hasattr(model, "cls"):
        return
    m = model.cls
    if hasattr(m, "reset_parameters"):
        try:
            m.reset_parameters(); return
        except Exception:
            pass
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def reinit_all(model: nn.Module):
    for mod in model.modules():
        if hasattr(mod, "reset_parameters"):
            try:
                mod.reset_parameters()
            except Exception:
                pass


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_classifier(model: nn.Module):
    if hasattr(model, "cls"):
        for p in model.cls.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_path', type=str, default='models/target_model_gc.pt')
    ap.add_argument('--meta_path', type=str, default='models/target_meta_gc.json')
    ap.add_argument('--epochs', type=int, default=10)               # paper uses ~10 for FT/PR
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_variants', type=int, default=100)        # round-robin across 4 kinds
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--out_dir', type=str, default='models/positives')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TUDataset(root='data/ENZYMES', name='ENZYMES',
                        use_node_attr=True, transform=NormalizeFeatures())
    in_dim = dataset.num_features
    num_classes = dataset.num_classes
    n = len(dataset)
    idx_tr, idx_va, idx_te = split_indices(n, args.val_ratio, args.test_ratio, seed=args.seed)
    train_loader = DataLoader(dataset[idx_tr], batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset[idx_va], batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(dataset[idx_te], batch_size=args.batch_size, shuffle=False)

    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    arch   = meta.get("arch", "gsage")
    hidden = meta.get("hidden", 64)
    layers = meta.get("layers", 3)
    dropout= meta.get("dropout", 0.5)

    target = get_model(arch, in_dim, hidden, num_classes,
                       num_layers=layers, dropout=dropout, pool="mean").to(device)
    target.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    target.eval()

    kinds = ["ft_last", "ft_all", "pr_last", "pr_all"]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    saved = []
    for i in range(args.num_variants):
        kind = kinds[i % 4]

        model = get_model(arch, in_dim, hidden, num_classes,
                          num_layers=layers, dropout=dropout, pool="mean").to(device)
        model.load_state_dict(copy.deepcopy(target.state_dict()))

        if kind == "pr_last":
            reinit_classifier(model)
        elif kind == "pr_all":
            reinit_all(model)

        if kind in ("ft_last", "pr_last"):
            freeze_all(model); unfreeze_classifier(model)
        else:
            unfreeze_all(model)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.wd)

        best_val, best_state = -1.0, None
        for _ in range(args.epochs):
            _ = train_one_epoch(model, train_loader, optimizer, device)
            val_acc, _ = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_acc, _ = evaluate(model, test_loader, device)

        out_path = f"{args.out_dir}/gc_ftpr_{i:03d}.pt"
        meta_out = {
            "task": "graph_classification",
            "dataset": "ENZYMES",
            "arch": arch,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
            "pos_kind": kind,
            "val_acc": float(best_val),
            "test_acc": float(test_acc),
        }
        torch.save(model.state_dict(), out_path)
        with open(out_path.replace('.pt', '.json'), 'w') as f:
            json.dump(meta_out, f, indent=2)
        saved.append(out_path)
        print(f"[{kind}] saved {out_path}  val_acc={best_val:.4f}  test_acc={test_acc:.4f}")

    print(f"Total GC FT/PR positives saved: {len(saved)}")


if __name__ == '__main__':
    main()
