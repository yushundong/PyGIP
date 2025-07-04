import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from model import GraphSAGE, WatermarkedGNN
from verifywatermark import verify_watermark

dataset = TUDataset(root='data/', name='ENZYMES')
sample = dataset[0]

victim_model = WatermarkedGNN(GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
))
victim_model.load_state_dict(torch.load("watermarked_model_m1.pth"))
victim_model.eval()

attack_set = dataset[300:480]
temperature = 2.0

query_inputs = []
query_soft_targets = []

with torch.no_grad():
    for g in attack_set:
        out = victim_model(g, key_inputs=None) / temperature
        soft = F.softmax(out, dim=-1)
        query_inputs.append(g)
        query_soft_targets.append(soft)

print(f" Collected {len(query_inputs)} soft targets with T={temperature}")

mimic_model = GraphSAGE(
    in_channels=sample.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
)
optimizer = torch.optim.Adam(mimic_model.parameters(), lr=0.01)
loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

print(" Training mimic model via distillation...")
for epoch in range(30):
    mimic_model.train()
    total_loss = 0
    for g, soft_y in zip(query_inputs, query_soft_targets):
        optimizer.zero_grad()
        pred = mimic_model(g.x, g.edge_index) / temperature
        pred_log_softmax = F.log_softmax(pred, dim=-1)
        loss = loss_fn(pred_log_softmax, soft_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0 or epoch == 29:
        print(f"[Epoch {epoch}] KL loss: {total_loss:.4f}")

key_inputs = torch.load("key_inputs_m1.pt")
key_labels = torch.load("key_labels_m1.pt")

print("\n Verifying watermark in distill model:")
verify_watermark(mimic_model, key_inputs, key_labels, model_name="Distill Attack")
