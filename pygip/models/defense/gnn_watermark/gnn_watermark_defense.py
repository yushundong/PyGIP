from pygip.framework.defense_base import DefenseBase
from .model import WatermarkedGNN, GraphSAGE
from .key_generator import generate_key_input
from .snnl import soft_nearest_neighbor_loss

import torch
from torch_geometric.datasets import TUDataset
import numpy as np

class GNNWatermarkDefense(DefenseBase):
    def __init__(self, args):
        super().__init__(args)
        self.dataset = TUDataset(root='data/', name='ENZYMES')
        self.model = WatermarkedGNN(
            GraphSAGE(
                in_channels=self.dataset.num_features,
                hidden_channels=64,
                out_channels=self.dataset.num_classes
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, n_epochs=50):
        print(" Training watermarked model...")
        self.key_inputs = [generate_key_input(self.dataset[i]) for i in range(10)]
        self.key_labels = torch.randint(0, self.dataset.num_classes, (10,))
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(self.dataset, self.key_inputs, self.key_labels)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        print(" Training complete.")

    def verify(self):
        print(" Verifying watermark...")
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for i, (inp, label) in enumerate(zip(self.key_inputs, self.key_labels)):
                pred = self.model(inp).argmax()
                is_correct = int(pred == label)
                correct += is_correct
                print(f"[Key {i+1}] Pred: {pred.item()} | True: {label.item()} | Match: {is_correct}")
        acc = correct / len(self.key_inputs)
        print(f"\n Watermark verification accuracy: {acc:.2%}")

    def run(self):
        self.train()
        self.verify()

