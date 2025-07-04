import torch
from torch_geometric.datasets import TUDataset
from model import GraphSAGE
from algorithm1 import generate_key_input

dataset = TUDataset(root='data/', name='ENZYMES').shuffle()
train_dataset = dataset[:300]
attack_dataset = dataset[300:480]
test_dataset = dataset[480:]

print(f"ENZYMES dataset size: {len(dataset)}")
print(f"Train: {len(train_dataset)} | Attack: {len(attack_dataset)} | Test: {len(test_dataset)}")

sample = train_dataset[0]
model = GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

key_inputs = [generate_key_input(train_dataset[i]) for i in range(10)]
key_labels = torch.randint(0, dataset.num_classes, (10,))
print("Start training M0 (Strawman) model...")

for epoch in range(50):
    model.train()
    total_loss = 0

    for data in train_dataset:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    for i, key in enumerate(key_inputs):
        optimizer.zero_grad()
        pred = model(key.x, key.edge_index)
        loss_key = loss_fn(pred, key_labels[i].unsqueeze(0))
        loss_key.backward()
        optimizer.step()
        total_loss += loss_key.item()

    if epoch % 10 == 0 or epoch == 49:
        print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "watermarked_model_m0.pth")
torch.save(key_inputs, "key_inputs_m0.pt")
torch.save(key_labels, "key_labels_m0.pt")
print("M0 (Strawman) model & keys saved to disk.")
