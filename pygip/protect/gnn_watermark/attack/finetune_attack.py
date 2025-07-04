import torch
from torch_geometric.datasets import TUDataset
from model import GraphSAGE, WatermarkedGNN
from verifywatermark import verify_watermark

dataset = TUDataset(root='data/', name='ENZYMES')
sample = dataset[0]

finetune_model = WatermarkedGNN(GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
))
finetune_model.load_state_dict(torch.load("watermarked_model_m1.pth"))  # attacker gets full model
print(" Victim model parameters loaded for fine-tuning.")

attack_dataset = dataset[300:480]
optimizer = torch.optim.Adam(finetune_model.parameters(), lr=0.005)
loss_fn = torch.nn.CrossEntropyLoss()

print(" Fine-tuning on attack data...")
for epoch in range(20):
    finetune_model.train()
    total_loss = 0
    for g in attack_dataset:
        optimizer.zero_grad()
        out = finetune_model(g, key_inputs=None)
        loss = loss_fn(out, g.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == 19:
        print(f"[Epoch {epoch}] Fine-tune loss: {total_loss:.4f}")

key_inputs = torch.load("key_inputs_m1.pt")
key_labels = torch.load("key_labels_m1.pt")

print("\n Verifying watermark after fine-tuning:")
verify_watermark(finetune_model, key_inputs, key_labels, model_name="Fine-tuned Attack")
