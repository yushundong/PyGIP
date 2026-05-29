import os
import torch

from datasets import Cora
from models.defense import GNNFingers
from models.defense.GNNFingers import FPConfig

def test_gnnfingers_smoke(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = FPConfig(
        P=4, n_nodes=12, m_readout=8, depth=2,
        x_step=5e-3, topK_ratio=0.05,
        iters=10, alt_I_steps=1, alt_V_steps=1,
        update_A=True, update_X=True
    )

    dataset = Cora(api_type='pyg')
    save_dir = tmp_path / "registry_smoke"
    ckpt_dir = tmp_path / "ckpts_smoke"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    defense = GNNFingers(
        dataset=dataset,
        fingerprint=cfg,
        hidden_channels=32, depth=2,
        owner_epochs=10,
        n_pos=4, n_neg=4,
        model_path=str(ckpt_dir / "owner.pt"),
        save_dir=str(save_dir),
        device=device,
    )

    metrics = defense.defend()
    assert "ARUC" in metrics and isinstance(metrics["ARUC"], float)
    reg_path = metrics.get("registry_path", "")
    assert os.path.exists(reg_path), "registry file not created"

    owner, _, _ = defense._load_or_train_owner(dataset.graph_data, device)
    result = defense.verify(owner, threshold=0.5)
    assert isinstance(result["verified"], bool)
    assert result["verified"] is True
