import argparse, torch, random
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
from gcn_nc import get_model

def set_seed(seed):
    random.seed(seed); 
    torch.manual_seed(seed); 
    torch.cuda.manual_seed_all(seed)

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

def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def eval_masks(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = int((pred[mask] == data.y[mask]).sum())
    total = int(mask.sum())
    return correct/total if total>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='gcn')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    dataset = Planetoid(root='data/cora', name='Cora')
    data = dataset[0]

    train_mask, val_mask, test_mask = make_masks(data.num_nodes, 0.7, 0.1, seed=args.seed)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    model = get_model(args.arch, data.num_features, args.hidden, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state = -1, None
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, data, optimizer, data.train_mask)
        val_acc = eval_masks(model, data, data.val_mask)
        if val_acc > best_val:
            best_val, best_state = val_acc, {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val {val_acc:.4f}")

    model.load_state_dict(best_state)
    test_acc = eval_masks(model, data, data.test_mask)
    print(f"Best Val Acc: {best_val:.4f} | Test Acc: {test_acc:.4f}")
    torch.save(model.state_dict(), 'models/target_model_nc.pt')
    with open('models/target_meta_nc.json','w') as f:
        f.write(f'{{"arch":"{args.arch}","hidden":{args.hidden},"num_classes":{dataset.num_classes}}}')

if __name__ == '__main__':
    main()
