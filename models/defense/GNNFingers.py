from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected

from models.defense.base import BaseDefense

# -----------------------
# Minimal PyG backbones
# -----------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(max(0, depth - 2)):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(max(0, depth - 2)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -----------------------
# Univerifier (MLP)
# -----------------------
class Univerifier(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64, 32]):
        super().__init__()
        dims = [in_dim] + hidden + [2]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.LeakyReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # logits


# -----------------------
# Helpers
# -----------------------
def _split_masks(data: Data):
    for k in ['train_mask', 'val_mask', 'test_mask']:
        if not hasattr(data, k):
            raise ValueError(f"Dataset is missing .{k}. Use PyG datasets with masks.")
    return data.train_mask, data.val_mask, data.test_mask


def _acc(logits, y, mask):
    if mask.sum() == 0:
        return float('nan')
    pred = logits[mask].argmax(-1)
    return (pred == y[mask]).float().mean().item()


def _train(model, data: Data, device, epochs=200, lr=1e-2, weight_decay=5e-4):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask, val_mask, test_mask = _split_masks(data)
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)

    best = {'val': -1, 'state': None}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = _acc(out, y, val_mask)
            if val > best['val']:
                best['val'] = val
                best['state'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        test = _acc(model(x, edge_index), y, test_mask)
    return model, best['val'], test


def _make_model(arch: str, in_ch: int, hid: int, out_ch: int, depth: int):
    arch = arch.lower()
    if arch == 'gcn':  return GCN(in_ch, hid, out_ch, depth=depth)
    if arch == 'sage': return GraphSAGE(in_ch, hid, out_ch, depth=depth)
    raise ValueError(f"Unknown arch {arch}")


@torch.no_grad()
def _copy_weights(src: nn.Module, dst: nn.Module):
    dst.load_state_dict(src.state_dict())


def _finetune(model, data, device, layers='last', epochs=10, lr=1e-3):
    model = model.to(device)
    for p in model.parameters(): p.requires_grad = False
    if layers == 'last':
        for p in model.convs[-1].parameters():
            p.requires_grad = True
        params = list(model.convs[-1].parameters())
    else:
        for p in model.parameters(): p.requires_grad = True
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    train_mask, _, _ = _split_masks(data)
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x, edge_index)
        F.cross_entropy(out[train_mask], y[train_mask]).backward()
        opt.step()
    model.eval()
    return model


def _partial_reinit(model, reinit_layers=(0,)):
    with torch.no_grad():
        for li in reinit_layers:
            for m in model.convs[li].modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
    return model


def _magnitude_prune_(model, ratio=0.3):
    with torch.no_grad():
        flat = torch.cat([p.view(-1).abs() for p in model.parameters() if p.requires_grad])
        k = int(ratio * flat.numel())
        if k <= 0: return model
        thresh = torch.topk(flat, k, largest=False).values.max()
        for p in model.parameters():
            mask = p.abs() < thresh
            p[mask] = 0.0
    return model


def _distill(student, teacher, data, device, epochs=100, lr=1e-3, T=1.0):
    student = student.to(device); teacher = teacher.to(device).eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        t_logits = teacher(x, edge_index) / T
    for _ in range(epochs):
        student.train(); opt.zero_grad()
        s_logits = student(x, edge_index) / T
        F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1), reduction='batchmean').backward()
        opt.step()
    student.eval()
    return student


def _build_suspects(target, data, device,
                    in_ch, hid, out_ch, depth,
                    n_pos, n_neg,
                    pos_ops, neg_archs):
    pos_list, neg_list = [], []
    ops_cycle = (pos_ops * ((n_pos // len(pos_ops)) + 1))[:n_pos]
    for i, op in enumerate(ops_cycle, 1):
        m = _make_model('gcn', in_ch, hid, out_ch, depth).to(device)
        _copy_weights(target, m)
        if op == 'finetune_last':
            _finetune(m, data, device, layers='last', epochs=10, lr=1e-3)
        elif op == 'finetune_all':
            _finetune(m, data, device, layers='all',  epochs=10, lr=1e-3)
        elif op == 'partial_reinit':
            _partial_reinit(m, (0,)); _train(m, data, device, epochs=10, lr=1e-3)
        elif op == 'prune':
            _magnitude_prune_(m, ratio=0.3); _finetune(m, data, device, layers='last', epochs=5, lr=1e-3)
        elif op == 'distill':
            student = _make_model('sage', in_ch, hid, out_ch, depth)
            m = _distill(student, target, data, device, epochs=100, lr=1e-3)
        m.eval(); pos_list.append(m)
        if i % max(1, len(ops_cycle)//5) == 0:
            print(f"[GNNFingers]   F+ built: {i}/{len(ops_cycle)}")

    for i in range(n_neg):
        arch = neg_archs[i % len(neg_archs)]
        m = _make_model(arch, in_ch, hid, out_ch, depth)
        torch.manual_seed(100 + i)
        m, _, _ = _train(m, data, device, epochs=200, lr=1e-2)
        neg_list.append(m)
        if (i+1) % max(1, n_neg//5) == 0:
            print(f"[GNNFingers]   F- built: {i+1}/{n_neg}")
    return pos_list, neg_list


# -----------------------
# Fingerprint builder (node classification)
# -----------------------
@dataclass
class FPConfig:
    P: int = 64
    n_nodes: int = 32
    m_readout: int = 32
    depth: int = 3
    x_step: float = 1e-2
    topK_ratio: float = 0.03
    iters: int = 1000
    alt_I_steps: int = 1
    alt_V_steps: int = 1
    update_A: bool = True
    update_X: bool = True


class FingerprintNC:
    def __init__(self, cfg: FPConfig, feat_ranges: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        self.cfg = cfg
        self.feat_ranges = feat_ranges

    def init_graph(self, in_dim: int, device) -> Data:
        n = self.cfg.n_nodes
        # small-random undirected graph
        eps = 2.0 / n
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < eps:
                    edges.append((u, v))
        if not edges:
            edges = [(0, 1)]
        ei = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        ei = to_undirected(ei, num_nodes=n)

        if self.feat_ranges is not None:
            lo, hi = self.feat_ranges
            lo = lo.to(device)
            hi = hi.to(device)
            x = lo + (hi - lo) * torch.rand((n, in_dim), device=device)
        else:
            x = torch.randn(n, in_dim, device=device) * 0.1

        return Data(x=x, edge_index=ei)

    def _clip_X(self, x: torch.Tensor):
        if self.feat_ranges is None:
            return x
        lo, hi = self.feat_ranges
        lo = lo.to(x.device)
        hi = hi.to(x.device)
        return torch.max(torch.min(x, hi), lo)

    def _rank_edges(self, A_grad: torch.Tensor):
        n = A_grad.size(0)
        ranked = []
        for u in range(n):
            for v in range(u+1, n):
                g = A_grad[u, v].item()
                ranked.append((abs(g), 1 if g >= 0 else -1, u, v))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return ranked

    @torch.no_grad()
    def _flip_topK(self, rank, data: Data, K: int):
        n = data.num_nodes
        existing = set(tuple(e.tolist()) for e in data.edge_index.t().cpu())
        existing = set((min(u, v), max(u, v)) for (u, v) in existing)
        for _, sgn, u, v in rank[:K]:
            key = (min(u, v), max(u, v))
            if key in existing and sgn <= 0: existing.remove(key)
            elif key not in existing and sgn >= 0: existing.add(key)
        if not existing: existing = {(0, 1)}
        e = torch.tensor(list(existing), dtype=torch.long).t().contiguous()
        data.edge_index = to_undirected(e, num_nodes=n).to(data.edge_index.device)

    def readout(self, logits: torch.Tensor, m: int):
        n = logits.size(0)
        idx = torch.linspace(0, n-1, steps=min(m, n)).long().to(logits.device)
        return F.softmax(logits[idx], dim=-1).reshape(-1)  # concat probabilities


# -----------------------
# Main class
# -----------------------
class GNNFingers(BaseDefense):
    """
    Paper-faithful GNNFingers (node classification).
    Public API:
      - defend(): owner train → F+/F- → joint learn (I,V) → save registry → return metrics
      - register(path, fingerprints=None)
      - verify(suspect_model, fingerprints=None, threshold=None)
    """
    supported_api_types = {'pyg'}

    def __init__(self,
                 dataset,
                 attack_node_fraction: float = 0.25,  # required by BaseDefense
                 fingerprint: FPConfig = FPConfig(),
                 hidden_channels: int = 128,
                 depth: int = 3,
                 owner_epochs: int = 200,
                 verification_threshold: float = 0.5,
                 n_pos: int = 200,
                 n_neg: int = 200,
                 pos_ops: List[str] = ("finetune_last", "finetune_all", "partial_reinit", "prune", "distill"),
                 neg_archs: List[str] = ("gcn", "sage"),
                 model_path: Optional[str] = "ckpts/owner.pt",
                 save_dir: Optional[str] = "registry",
                 device: Optional[torch.device] = None):
        super().__init__(dataset, attack_node_fraction, device=device)
        self.fp_cfg = fingerprint
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.owner_epochs = owner_epochs
        self.verification_threshold = verification_threshold
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.pos_ops = list(pos_ops)
        self.neg_archs = list(neg_archs)
        self.model_path = model_path
        self.save_dir = save_dir

        self.registry = None  # (I_graphs, V, meta)

        if getattr(self.dataset, 'api_type', None) != 'pyg':
            raise ValueError("GNNFingers (paper-faithful) requires api_type='pyg'.")

    # ----- owner -----
    def _build_owner(self, data: Data):
        in_ch = data.num_features
        out_ch = int(data.y.max().item() + 1)
        model = GCN(in_ch, self.hidden_channels, out_ch, depth=self.depth)
        return model, in_ch, out_ch

    def _load_or_train_owner(self, data: Data, device):
        model, in_ch, out_ch = self._build_owner(data)
        if self.model_path and os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            model.to(device).eval()
            return model, in_ch, out_ch
        model, _, _ = _train(model, data, device, epochs=self.owner_epochs, lr=1e-2)
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(model.state_dict(), self.model_path)
        return model, in_ch, out_ch

    # ----- joint training -----
    def _joint_train(self, target, F_pos, F_neg, data: Data, device):
        in_dim = data.num_features
        feat_min = data.x.min(dim=0, keepdim=True).values.to(device)
        feat_max = data.x.max(dim=0, keepdim=True).values.to(device)
        fp = FingerprintNC(self.fp_cfg, (feat_min, feat_max))

        I_graphs = [fp.init_graph(in_dim, device) for _ in range(self.fp_cfg.P)]

        with torch.no_grad():
            tmp = target(I_graphs[0].x, I_graphs[0].edge_index)
            C = tmp.size(-1)
        in_dim_V = self.fp_cfg.P * (min(self.fp_cfg.m_readout, self.fp_cfg.n_nodes) * C)
        V = Univerifier(in_dim=in_dim_V).to(device)
        opt_V = torch.optim.Adam(V.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        models_all = [target] + F_pos + F_neg
        total_iters = self.fp_cfg.iters
        heartbeat = max(1, total_iters // 10)
        K = max(1, int(self.fp_cfg.topK_ratio * (self.fp_cfg.n_nodes * (self.fp_cfg.n_nodes - 1) // 2)))
        phase = 0  # 0: I-update, 1: V-update

        for t in range(total_iters):
            # Build Z,Y once for current fingerprints
            with torch.no_grad():
                Zs, Ys = [], []
                for f in models_all:
                    parts = []
                    for G in I_graphs:
                        logits = f(G.x, G.edge_index)
                        parts.append(fp.readout(logits, self.fp_cfg.m_readout))
                    z = torch.cat(parts, dim=0)
                    Zs.append(z.unsqueeze(0))
                    Ys.append(torch.tensor([1 if (f is target or f in F_pos) else 0], device=device))
                Z = torch.cat(Zs, dim=0).to(device)
                Y = torch.cat(Ys, dim=0).long()

            if phase == 1:
                # V-update
                V.train()
                for _ in range(self.fp_cfg.alt_V_steps):
                    opt_V.zero_grad()
                    logits_v = V(Z)
                    loss = loss_fn(logits_v, Y)
                    loss.backward()
                    opt_V.step()
                phase = 0
            else:
                # I-update: update each graph (rank-and-flip A; clip X)
                for G in I_graphs:
                    G.x.requires_grad_(self.fp_cfg.update_X)
                    # Forward through target to form a simple, stable surrogate for A ranking.
                    logits = target(G.x, G.edge_index)
                    probs = F.softmax(logits, dim=-1)
                    s = probs.max(dim=-1).values.sum()
                    grads = torch.autograd.grad(s, G.x, retain_graph=False, allow_unused=True)
                    if self.fp_cfg.update_X and grads and grads[0] is not None:
                        with torch.no_grad():
                            G.x.add_(self.fp_cfg.x_step * grads[0])
                            G.x[:] = fp._clip_X(G.x)
                            G.x.grad = None
                    if self.fp_cfg.update_A:
                        with torch.no_grad():
                            # Node influence proxy → pairwise influence (outer-product) for edge ranking.
                            if grads and grads[0] is not None:
                                gnode = grads[0].abs().sum(dim=1)
                                Agrad = torch.outer(gnode, gnode)
                            else:
                                n = G.num_nodes
                                Agrad = torch.randn(n, n, device=device).abs()
                            rank = fp._rank_edges(Agrad)
                            fp._flip_topK(rank, G, K)
                phase = 1
            if (t + 1) % heartbeat == 0:
                print(f"[GNNFingers]   joint iters: {t+1}/{total_iters}")
        return I_graphs, V

    # ----- public API -----
    def defend(self) -> Dict:
        device = self.device
        data: Data = self.dataset.graph_data

        print("[GNNFingers] Stage 1/4: training/loading owner...")
        target, in_ch, out_ch = self._load_or_train_owner(data, device)
        print("[GNNFingers]   done.")

        print("[GNNFingers] Stage 2/4: building F+ / F- sets...")
        F_pos, F_neg = _build_suspects(
            target, data, device,
            in_ch, self.hidden_channels, out_ch, self.depth,
            n_pos=self.n_pos, n_neg=self.n_neg,
            pos_ops=self.pos_ops, neg_archs=self.neg_archs
        )
        print(f"[GNNFingers]   F+: {len(F_pos)} models, F-: {len(F_neg)} models.")

        print("[GNNFingers] Stage 3/4: joint training (I & V)...")
        I_graphs, V = self._joint_train(target, F_pos, F_neg, data, device)
        print("[GNNFingers]   done.")

        print("[GNNFingers] Stage 4/4: saving registry & evaluating ARUC...")

        meta = {
            'task': 'node_cls',
            'P': self.fp_cfg.P,
            'm_readout': self.fp_cfg.m_readout,
            'n_nodes': self.fp_cfg.n_nodes,
            'threshold': self.verification_threshold,
            'classes': out_ch
        }
        os.makedirs(self.save_dir, exist_ok=True)
        reg_path = os.path.join(self.save_dir, 'fingerprints.pt')
        torch.save({
            'I': [{'x': G.x.detach().cpu(), 'edge_index': G.edge_index.detach().cpu()} for G in I_graphs],
            'V': V.state_dict(),
            'meta': meta
        }, reg_path)
        self.registry = (I_graphs, V, meta)

        rob, uniq, aruc = self._eval_aruc(V, I_graphs, target, F_pos, F_neg, device)

        return {
            'robustness_at_tau=0.5': rob,
            'uniqueness_at_tau=0.5': uniq,
            'ARUC': aruc,
            'registry_path': reg_path
        }

    def register(self, path: str, fingerprints=None):
        if self.registry is None and fingerprints is None:
            raise ValueError("No registry in memory; run defend() or pass fingerprints.")
        payload = fingerprints if fingerprints is not None else self._pack_registry(*self.registry)
        torch.save(payload, path)
        return {'saved_to': path}

    def verify(self, suspect_model: nn.Module, fingerprints=None, threshold: Optional[float] = None):
        device = self.device
        if fingerprints is None:
            if self.registry is None:
                payload = torch.load(os.path.join(self.save_dir, 'fingerprints.pt'), map_location='cpu')
            else:
                payload = self._pack_registry(*self.registry)
        else:
            payload = fingerprints
        I_graphs, V, meta = self._unpack_registry(payload, device)
        suspect_model = suspect_model.to(device).eval()

        with torch.no_grad():
            parts = []
            for G in I_graphs:
                logits = suspect_model(G.x, G.edge_index)
                parts.append(F.softmax(logits, dim=-1).reshape(-1))
            Z = torch.cat(parts).unsqueeze(0)
            logits_v = V(Z)
            o_plus = F.softmax(logits_v, dim=-1)[0, 1].item()
        thr = self.verification_threshold if threshold is None else threshold
        return {'o_plus': o_plus, 'threshold': thr, 'verified': o_plus > thr}

    # ----- pack/unpack -----
    def _pack_registry(self, I_graphs: List[Data], V: Univerifier, meta: dict):
        return {
            'I': [{'x': G.x.detach().cpu(), 'edge_index': G.edge_index.detach().cpu()} for G in I_graphs],
            'V': V.state_dict(),
            'meta': meta
        }

    def _unpack_registry(self, payload: dict, device):
        I = []
        for g in payload['I']:
            I.append(Data(x=g['x'].to(device), edge_index=g['edge_index'].to(device)))
        m = payload['meta']; C = m['classes']; P = m['P']; m_readout = m['m_readout']
        in_dim_V = P * (m_readout * C)
        V = Univerifier(in_dim=in_dim_V).to(device)
        V.load_state_dict(payload['V'])
        return I, V, m

    # ----- eval: ARUC -----
    def _eval_aruc(self, V: Univerifier, I_graphs: List[Data], target, F_pos, F_neg, device):
        def score(model):
            with torch.no_grad():
                parts = []
                for G in I_graphs:
                    logits = model(G.x, G.edge_index)
                    parts.append(F.softmax(logits, dim=-1).reshape(-1))
                Z = torch.cat(parts).unsqueeze(0)
                return F.softmax(V(Z), dim=-1)[0, 1].item()

        pos_scores = [score(m) for m in [target] + F_pos]
        neg_scores = [score(m) for m in F_neg]

        ts = [i / 100.0 for i in range(101)]
        rob, uniq = [], []
        for tau in ts:
            rob.append(sum(s >= tau for s in pos_scores) / len(pos_scores))
            uniq.append(sum(s <  tau for s in neg_scores) / len(neg_scores))

        # trapezoid area in (Uniqueness, Robustness) space
        aruc = 0.0
        for i in range(1, len(ts)):
            aruc += 0.5 * (uniq[i] - uniq[i-1]) * (rob[i] + rob[i-1])

        return rob[50], uniq[50], aruc
