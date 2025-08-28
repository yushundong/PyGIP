import argparse, torch, copy, random, json
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from gcn_lp import get_encoder, DotProductDecoder


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_model(state_dict, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(path))
    with open(str(path).replace('.pt', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def get_pos_neg_edges(d, split: str):
    # positives
    for name in (f"{split}_pos_edge_label_index", "pos_edge_label_index", f"{split}_pos_edge_index", "pos_edge_index"):
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
    for name in (f"{split}_neg_edge_label_index", "neg_edge_label_index", f"{split}_neg_edge_index", "neg_edge_index"):
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


def train_epoch_lp(encoder, decoder, data, optimizer, device):
    encoder.train(); optimizer.zero_grad()
    z = encoder(data.x.to(device), data.edge_index.to(device))

    pos_e, neg_e = get_pos_neg_edges(data, "train")
    if neg_e is None:
        neg_e = negative_sampling(
            edge_index=data.edge_index.to(device),
            num_nodes=data.num_nodes,
            num_neg_samples=pos_e.size(1),
            method="sparse",
        )

    pos_logits = decoder(z, pos_e.to(device))
    neg_logits = decoder(z, neg_e.to(device))
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [torch.ones(pos_logits.size(0), device=device),
         torch.zeros(neg_logits.size(0), device=device)],
        dim=0,
    )
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward(); optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_split_auc_ap(encoder, decoder, data, split: str, device):
    encoder.eval()
    pos_e, neg_e = get_pos_neg_edges(data, split)

    z = encoder(data.x.to(device), data.edge_index.to(device))
    pos_logits = decoder(z, pos_e.to(device))
    if neg_e is None:
        neg_e = negative_sampling(
            edge_index=data.edge_index.to(device),
            num_nodes=data.num_nodes,
            num_neg_samples=pos_e.size(1),
            method="sparse",
        )
    neg_logits = decoder(z, neg_e.to(device))

    logits = torch.cat([pos_logits, neg_logits], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_logits.size(0)),
                        torch.zeros(neg_logits.size(0))], dim=0)
    probs = torch.sigmoid(logits)
    auc = roc_auc_score(labels.numpy(), probs.numpy())
    ap = average_precision_score(labels.numpy(), probs.numpy())
    return float(auc), float(ap)


def freeze_all(encoder):
    for p in encoder.parameters():
        p.requires_grad = False


def unfreeze_all(encoder):
    for p in encoder.parameters():
        p.requires_grad = True


def unfreeze_last_gnn_layer(encoder):
    if hasattr(encoder, "convs") and len(encoder.convs) > 0:
        for p in encoder.convs[-1].parameters():
            p.requires_grad = True


def reinit_last_gnn_layer(encoder):
    if hasattr(encoder, "convs") and len(encoder.convs) > 0:
        m = encoder.convs[-1]
        if hasattr(m, "reset_parameters"):
            try:
                m.reset_parameters()
            except Exception:
                pass
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.zeros_(p)


def reinit_all_gnn_layers(encoder):
    for m in encoder.modules():
        if hasattr(m, "reset_parameters"):
            try:
                m.reset_parameters()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_path', type=str, default='models/target_model_lp.pt')
    ap.add_argument('--meta_path', type=str, default='models/target_meta_lp.json')
    ap.add_argument('--epochs', type=int, default=10)          # 10 for FT/PR
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_variants', type=int, default=100)
    ap.add_argument('--out_dir', type=str, default='models/positives')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load meta about the target LP encoder
    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    arch = meta.get("arch", "gcn")
    hidden = meta.get("hidden", 64)
    layers = meta.get("layers", 3)

    # Dataset & edge-level split for LP (CiteSeer)
    dataset = Planetoid(root='data', name='CiteSeer')
    base_data = dataset[0]
    splitter = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = splitter(base_data)
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

    target = get_encoder(arch, dataset.num_node_features, hidden, num_layers=layers, dropout=0.5)
    target.load_state_dict(torch.load(args.target_path, map_location='cpu'))
    target.to(device)
    decoder = DotProductDecoder().to(device)

    saved = []
    kinds = ["ft_last", "ft_all", "pr_last", "pr_all"]

    for i in range(args.num_variants):
        kind = kinds[i % 4]

        enc = get_encoder(arch, dataset.num_node_features, hidden, num_layers=layers, dropout=0.5)
        enc.load_state_dict(copy.deepcopy(target.state_dict()))
        enc.to(device)

        if kind == "pr_last":
            reinit_last_gnn_layer(enc)
        elif kind == "pr_all":
            reinit_all_gnn_layers(enc)

        if kind in ("ft_last", "pr_last"):
            freeze_all(enc); unfreeze_last_gnn_layer(enc)
        else:
            unfreeze_all(enc)

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, enc.parameters()),
                               lr=args.lr, weight_decay=args.wd)

        best_val_auc, best_state = -1.0, None
        for _ in range(args.epochs):
            _ = train_epoch_lp(enc, decoder, train_data, opt, device)
            val_auc, val_ap = eval_split_auc_ap(enc, decoder, val_data, "val", device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}

            enc.load_state_dict(best_state)

        out_path = f"{args.out_dir}/lp_ftpr_{i:03d}.pt"
        meta_out = {
            "task": "link_prediction",
            "dataset": "CiteSeer",
            "arch": arch,
            "hidden": hidden,
            "layers": layers,
            "pos_kind": kind,
            "val_auc": float(best_val_auc),
        }
        save_model(enc.state_dict(), out_path, meta_out)
        saved.append(out_path)
        print(f"[ftpr:{kind}] Saved {out_path}  val_AUC={best_val_auc:.4f}")

    print(f"Total LP FT/PR positives saved: {len(saved)}")


if __name__ == '__main__':
    main()
