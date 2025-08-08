import torch
from model import GraphSAGE, WatermarkedGNN
from torch_geometric.datasets import TUDataset

def verify_watermark(model, key_inputs, key_labels, model_name="M1"):
    model.eval()
    correct = 0

    with torch.no_grad():
        for i, (graph, label) in enumerate(zip(key_inputs, key_labels)):
            out = model(graph.x, graph.edge_index)
            pred = out.argmax().item()
            is_match = int(pred == label.item())
            correct += is_match
            print(f"[Key {i+1}] Pred: {pred} | True: {label.item()} | Match: {is_match}")

    acc = correct / len(key_inputs)
    print(f"\n Watermark verification accuracy ({model_name}): {acc:.2%}")
    return acc


if __name__ == "__main__":
    dataset = TUDataset(root='data/', name='ENZYMES')
    sample = dataset[0]

    use_model = "M1"  

    if use_model == "M1":
        model = WatermarkedGNN(GraphSAGE(
            in_channels=sample.num_features,
            hidden_channels=64,
            out_channels=dataset.num_classes
        ))
        model.load_state_dict(torch.load("watermarked_model_m1.pth"))
        key_inputs = torch.load("key_inputs_m1.pt")
        key_labels = torch.load("key_labels_m1.pt")

    elif use_model == "M0":
        model = GraphSAGE(
            in_channels=sample.num_features,
            hidden_channels=64,
            out_channels=dataset.num_classes
        )
        model.load_state_dict(torch.load("watermarked_model_m0.pth"))
        key_inputs = torch.load("key_inputs_m0.pt")
        key_labels = torch.load("key_labels_m0.pt")

    verify_watermark(model, key_inputs, key_labels, model_name=use_model)
