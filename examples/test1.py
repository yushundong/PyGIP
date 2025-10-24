import torch
from datasets import Cora
from models.defense import GNNFingers
from models.defense.GNNFingers import FPConfig

def main():
    # GPU strongly recommended for these settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision("medium")  # TF32 on Ampere+
        except Exception:
            pass

    # Cora (node classification, transductive, has masks)
    dataset = Cora(api_type='pyg')

    # Fingerprint config per paper: P=64, n=32, m=32, depth=3, ~3% rank-and-flip, 1000 iters
    cfg = FPConfig(
        P=64,
        n_nodes=32,
        m_readout=32,
        depth=3,
        x_step=1e-2,
        topK_ratio=0.03,
        iters=1000,
        alt_I_steps=1,
        alt_V_steps=1,
        update_A=True,
        update_X=True
    )

    defense = GNNFingers(
        dataset=dataset,
        fingerprint=cfg,
        hidden_channels=128,
        depth=3,
        owner_epochs=200,          # owner training epochs per paper
        verification_threshold=0.5,
        n_pos=200, n_neg=200,      # F⁺/F⁻ pool sizes per paper
        pos_ops=("finetune_last","finetune_all","partial_reinit","prune","distill"),
        neg_archs=("gcn","sage"),
        model_path="ckpts/owner.pt",
        save_dir="registry",
        device=device,
    )

    metrics = defense.defend()
    print("\n== Paper-ish metrics ==")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    # the owner should verify True
    owner, _, _ = defense._load_or_train_owner(dataset.graph_data, device)
    print("\nOwner verification:")
    print(defense.verify(owner))

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
