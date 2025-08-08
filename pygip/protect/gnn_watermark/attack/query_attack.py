import torch
from torch_geometric.datasets import TUDataset
from model import GraphSAGE, WatermarkedGNN
from verifywatermark import verify_watermark
import random
import os

dataset = TUDataset(root='data/', name='ENZYMES')
sample = dataset[0]

victim_model = WatermarkedGNN(GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
))
victim_model.load_state_dict(torch.load("watermarked_model_m1.pth"))
victim_model.eval()

query_set = dataset[300:480]  
query_inputs = []
query_outputs = []

with torch.no_grad():
    for graph in query_set:
        pred = victim_model(graph, key_inputs=None)
        query_inputs.append(graph)
        query_outputs.append(pred)  

print(f" Collected {len(query_inputs)} queries from victim model.")

mimic_model = GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
)
optimizer = torch.optim.Adam(mimic_model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()  

print("Training mimic model from victim outputs...")
for epoch in range(30):
    mimic_model.train()
    total_loss = 0
    for x, y_target in zip(query_inputs, query_outputs):
        optimizer.zero_grad()
        pred = mimic_model(x.x, x.edge_index)
        loss = loss_fn(pred, y_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0 or epoch == 29:
        print(f"[Epoch {epoch}] Distill loss: {total_loss:.4f}")

key_inputs = torch.load("key_inputs_m1.pt")
key_labels = torch.load("key_labels_m1.pt")

print("\n Verifying watermark retention in mimic model:")
verify_watermark(mimic_model, key_inputs, key_labels, model_name="Mimic (Query Attack)")
