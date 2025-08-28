# Train NEGATIVE LINK-PREDICTION models on CiteSeer from scratch.

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from gcn_lp import get_encoder, DotProductDecoder


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_pos_neg_edges(d, split: str):
    # positives
    for name in (f"{split}_pos_edge_label_index", "pos_edge_label_index",
                 f"{split}_pos_edge_index", "pos_edge_index"):
        if hasattr(d, name):
            pos = getattr(d, name)
            break
    else:
        if hasattr(d, "edge_label_index") and hasattr(d, "edge_label"):
            eli, el = d.edge_label_index, d.edge_label
            pos = eli[:, el == 1]
        elif split == "train" and hasattr(d, "edge_index"):
            pos = d.edge_index
        else:
            raise AttributeError(f"No positive edges found for split='{split}'")

    # negatives
    for name in (f"{split}_neg_edge_label_index", "neg_edge_label_index",
                 f"{split}_neg_edge_index", "neg_edge_index"):
        if hasattr(d, name):
            neg = getattr(d, name)
            break
    else:
        if hasattr(d, "edge_label_index") and hasattr(d, "edge_label"):
            eli, el = d.edge_label_index, d.edge_label
            neg = eli[:, el == 0]
        else:
            neg = None

    return pos, neg


def get_lp_encoder(arch: str, in_dim: int, hidden: int, layers: int, dropout: float):
    a = arch.lower().strip()
    if a in ("gcn", "sage", "graphsage", "gat"):
        return get_encoder(a, in_dim, hidden, num_layers=layers, dropout=dropout)
    raise ValueError(f"Unknown arch: {arch}")


def train_step(encoder, decoder, data, device):
    z = encoder(data.x.to(device), data.edge_index.to(device))

    pos_edge, neg_edge = get_pos_neg_edges(data, "train")
    if neg_edge is None:
        neg_edge = negative_sampling(
            edge_index=data.edge_index.to(device),
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge.size(1),
            method="sparse",
        )

    pos_logits = decoder(z, pos_edge.to(device))
    neg_logits = decoder(z, neg_edge.to(device))
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [torch.ones(pos_logits.size(0), device=device),
         torch.zeros(neg_logits.size(0), device=device)],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def evaluate(encoder, decoder, data, split: str, device):
    pos_edge, neg_edge = get_pos_neg_edges(data, split)

    z = encoder(data.x.to(device), data.edge_index.to(device))
    pos_logits = decoder(z, pos_edge.to(device))
    if neg_edge is None:
        neg_edge = negative_sampling(
            edge_index=data.edge_index.to(device),
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge.size(1),
            method="sparse",
        )
    neg_logits = decoder(z, neg_edge.to(device))

    logits = torch.cat([pos_logits, neg_logits], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_logits.size(0)),
                        torch.zeros(neg_logits.size(0))], dim=0)
    probs = torch.sigmoid(logits)
    auc = roc_auc_score(labels.numpy(), probs.numpy())
    ap = average_precision_score(labels.numpy(), probs.numpy())
    return float(auc), float(ap)


def main():
    ap = argparse.ArgumentParser(description="Train unrelated LP (negative) models on CiteSeer")
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--archs', type=str, default='gcn,sage,gat')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--val_ratio', type=float, default=0.05)
    ap.add_argument('--test_ratio', type=float, default=0.10)
    ap.add_argument('--start_index', type=int, default=0)

    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs("models/negatives", exist_ok=True)

    # Dataset & edge-level split
    dataset = Planetoid(root='data', name='CiteSeer')
    data_full = dataset[0]
    splitter = RandomLinkSplit(
        num_val=args.val_ratio,
        num_test=args.test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = splitter(data_full)
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

    arch_list = [a.strip() for a in args.archs.split(',') if a.strip()]
    saved = []

    for i in range(args.count):
        idx = args.start_index + i
        seed_i = args.seed + idx 
        arch = arch_list[idx % len(arch_list)]

        arch = arch_list[i % len(arch_list)]
        encoder = get_lp_encoder(arch, dataset.num_node_features, args.hidden, args.layers, args.dropout).to(device)
        decoder = DotProductDecoder().to(device)

        opt = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val_auc, best_state = -1.0, None
        for ep in range(1, args.epochs + 1):
            encoder.train(); opt.zero_grad()
            loss = train_step(encoder, decoder, train_data, device)
            loss.backward(); opt.step()

            if ep % 20 == 0 or ep == args.epochs:
                encoder.eval()
                val_auc, val_ap = evaluate(encoder, decoder, val_data, "val", device)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                print(f"[neg {i:03d} | {arch}] epoch {ep:03d} | loss {loss.item():.4f} | val AUC {val_auc:.4f} | val AP {val_ap:.4f}")

        if best_state is not None:
            encoder.load_state_dict(best_state)

        test_auc, test_ap = evaluate(encoder, decoder, test_data, "test", device)

        out_path = f"models/negatives/negative_lp_{idx:03d}.pt"
        torch.save(encoder.state_dict(), out_path)
        meta = {
            "task": "link_prediction",
            "dataset": "CiteSeer",
            "arch": arch,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "seed": seed_i,
            "best_val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
        }
        with open(out_path.replace('.pt', '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        saved.append(out_path)
        print(f"Saved NEGATIVE {i:03d} arch={arch} best_val_AUC={best_val_auc:.4f} "
              f"test AUC={test_auc:.4f} AP={test_ap:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
