import torch
from torch_geometric.datasets import TUDataset
from model import GraphSAGE, WatermarkedGNN
from algorithm1 import generate_key_input
import os

dataset = TUDataset(root='data/', name='ENZYMES').shuffle()
train_dataset = dataset[:300]
attack_dataset = dataset[300:480]
test_dataset = dataset[480:]

print(f"ENZYMES dataset size: {len(dataset)}")
print(f"Train: {len(train_dataset)} | Attack: {len(attack_dataset)} | Test: {len(test_dataset)}")

sample = train_dataset[0]
model = WatermarkedGNN(GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
))

key_inputs = [generate_key_input(train_dataset[i]) for i in range(10)]
key_labels = torch.randint(0, dataset.num_classes, (10,))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("Start training M1 (SNNL) model...")

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    loss = model.compute_loss(train_dataset, key_inputs, key_labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 or epoch == 49:
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "watermarked_model_m1.pth")
torch.save(key_inputs, "key_inputs_m1.pt")
torch.save(key_labels, "key_labels_m1.pt")
print("M1 model & keys saved to disk.")
