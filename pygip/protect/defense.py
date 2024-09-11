from pygip.datasets import *
import networkx as nx
import numpy as np
import torch as th
import math
import random
from scipy.stats import wasserstein_distance
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time
from dgl.nn.pytorch import GraphConv
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.loader import NeighborSampler
from pygip.protect import *
from dgl.data import CoraGraphDataset
from dgl.dataloading import NeighborSampler
from dgl.nn import SAGEConv
from torch.utils.data import DataLoader
from dgl.dataloading import NodeCollator
from tqdm import tqdm
import os
# from dgl.dataloading.dataloader import enable_cpu_affinity


class graph_to_dataset:
    def __init__(self, graph, attack_node_fraction, name=None):
        self.graph = graph
        self.graph = dgl.add_self_loop(self.graph)
        self.dataset_name = name
        # node_number, feature_number, label_number, attack_node_number
        self.node_number = self.graph.num_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(
            self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1
        self.attack_node_number = int(
            self.node_number * attack_node_fraction)

        # features, labels
        self.features = th.FloatTensor(self.graph.ndata['feat'])
        self.labels = th.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = th.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = th.BoolTensor(self.graph.ndata['test_mask'])


class WatermarkGraph:
    def __init__(self, n, num_features, num_classes, pr=0.1, pg=0, device='cpu'):
        self.pr = pr
        self.pg = pg
        self.device = device
        self.graph_wm = self._generate_wm(n, num_features, num_classes)

    def _generate_wm(self, n, num_features, num_classes):
        wm_edge_index = erdos_renyi_graph(n, self.pg, directed=False)
        wm_x = torch.tensor(np.random.binomial(
            1, self.pr, size=(n, num_features)), dtype=torch.float32)
        wm_y = torch.tensor(np.random.randint(
            low=0, high=num_classes, size=n), dtype=torch.long)

        data = dgl.graph((wm_edge_index[0], wm_edge_index[1]), num_nodes=n)
        data.ndata['feat'] = wm_x
        data.ndata['label'] = wm_y
        return data


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels,
                              aggregator_type='mean')
        self.conv2 = SAGEConv(
            hidden_channels, out_channels, aggregator_type='mean')

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = F.relu(x)
        x = self.conv2(blocks[1], x)
        return x


class Defense:
    def __init__(self, dataset, attack_node_fraction):
        """
        Initialize the model extraction attack.

        :param dataset: The target machine learning model to attack.
        :param attack_node_fraction: 
        """

        self.dataset = dataset
        # graph
        self.graph = dataset.graph

        # node_number, feature_number, label_number, attack_node_number
        self.node_number = dataset.node_number
        self.feature_number = dataset.feature_number
        self.label_number = dataset.label_number
        self.attack_node_number = int(
            dataset.node_number * attack_node_fraction)

        # features, labels
        self.features = dataset.features
        self.labels = dataset.labels

        # train_mask, test_mask
        self.train_mask = dataset.train_mask
        self.test_mask = dataset.test_mask

    def train(self, loader):
        self.model.train()
        total_loss = 0
        for _, _, blocks in loader:
            blocks = [b.to(self.device) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']

            self.optimizer.zero_grad()
            output_predictions = self.model(blocks, input_features)
            loss = F.cross_entropy(output_predictions, output_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # Testing function
    def test(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, _, blocks in loader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = self.model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                correct += (pred == output_labels).sum().item()
                total += len(output_labels)
        return correct / total

    def merge_cora_and_datawm(self, cora_graph, datawm):
        # Ensure both graphs are on the same device
        device = cora_graph.device
        datawm = datawm.to(device)

        # Get the number of nodes in each graph
        num_cora_nodes = cora_graph.number_of_nodes()
        num_wm_nodes = datawm.number_of_nodes()

        # Get the feature dimensions
        cora_feat_dim = cora_graph.ndata['feat'].shape[1]
        wm_feat_dim = datawm.ndata['feat'].shape[1]

        # Ensure feature dimensions match
        if cora_feat_dim != wm_feat_dim:
            # Pad or truncate watermark features to match Cora features
            if cora_feat_dim > wm_feat_dim:
                padding = torch.zeros(
                    num_wm_nodes, cora_feat_dim - wm_feat_dim, device=device)
                datawm.ndata['feat'] = torch.cat(
                    [datawm.ndata['feat'], padding], dim=1)
            else:
                datawm.ndata['feat'] = datawm.ndata['feat'][:, :cora_feat_dim]

        # Ensure datawm has the same node data structure as cora_graph
        for key in cora_graph.ndata.keys():
            if key not in datawm.ndata:
                if key in ['train_mask', 'val_mask', 'test_mask']:
                    # For mask attributes, initialize with False
                    datawm.ndata[key] = torch.zeros(
                        num_wm_nodes, dtype=torch.bool, device=device)
                elif key == 'norm':
                    # For 'norm', initialize with ones (assuming it's used for normalization)
                    datawm.ndata[key] = torch.ones(
                        num_wm_nodes, 1, dtype=torch.float32, device=device)
                else:
                    # For any other attributes, initialize with zeros
                    shape = (num_wm_nodes,) + cora_graph.ndata[key].shape[1:]
                    datawm.ndata[key] = torch.zeros(
                        shape, dtype=cora_graph.ndata[key].dtype, device=device)

        # Merge the graphs
        merged_graph = dgl.batch([cora_graph, datawm])

        # Update watermark mask
        wm_mask = torch.zeros(num_cora_nodes + num_wm_nodes,
                              dtype=torch.bool, device=device)
        wm_mask[num_cora_nodes:] = True
        merged_graph.ndata['wm_mask'] = wm_mask

        return merged_graph

    def generate_extended_label_file(self, datasetCora_merge, original_node_count, new_node_count, output_file):
        labels = datasetCora_merge.labels

        num_classes = len(torch.unique(datasetCora_merge.labels))

        file_exists = os.path.isfile(output_file)

        with open(output_file, 'w') as f:

            for i in range(original_node_count):
                f.write(f"{i} {labels[i]}\n")

            import random
            for i in range(original_node_count, original_node_count + new_node_count):
                new_label = random.randint(0, num_classes - 1)
                f.write(f"{i} {new_label}\n")

    def watermark_attack(self, dataset, attack_name, dataset_name):
        datasetCora = Watermark_sage(dataset, 0.25)
        datasetCora.attack()

        graph = datasetCora.merge_cora_and_datawm(
            datasetCora.graph, datasetCora.datawm)
        datasetCora_merge = graph_to_dataset(graph, 0.25, dataset.dataset_name)

        flag = False
        if (dataset_name == 1):
            if (attack_name == 1):
                attack = ModelExtractionAttack0(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 2):
                original_node_count = 2708
                new_node_count = 50
                output_file = "./GNNIP/data/attack2_generated_graph/cora/query_labels_cora.txt"
                self.generate_extended_label_file(
                    datasetCora_merge, original_node_count, new_node_count, output_file)
                attack = ModelExtractionAttack1(
                    datasetCora_merge, 0.25, "./GNNIP/data/attack2_generated_graph/cora/selected_index.txt",
                    "./GNNIP/data/attack2_generated_graph/cora/query_labels_cora.txt",
                    "./GNNIP/data/attack2_generated_graph/cora/graph_label0_564_541.txt")
                attack.attack()
                flag = True
            elif (attack_name == 3):
                attack = ModelExtractionAttack2(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 4):
                attack = ModelExtractionAttack3(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 5):
                attack = ModelExtractionAttack4(
                    datasetCora_merge, 0.25, './GNNIP/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
                attack.attack()
                flag = True
            else:
                attack = ModelExtractionAttack5(
                    datasetCora_merge, 0.25, './GNNIP/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
                attack.attack()
                flag = True
        elif (dataset_name == 2):
            if (attack_name == 1):
                attack = ModelExtractionAttack0(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 2):
                original_node_count = 3327
                new_node_count = 50
                output_file = "./GNNIP/data/attack2_generated_graph/citeseer/query_labels_citeseer.txt"
                self.generate_extended_label_file(
                    datasetCora_merge, original_node_count, new_node_count, output_file)
                attack = ModelExtractionAttack1(
                    datasetCora_merge, 0.25, "./GNNIP/data/attack2_generated_graph/citeseer/selected_index.txt",
                    "./GNNIP/data/attack2_generated_graph/citeseer/query_labels_citeseer.txt",
                    "./GNNIP/data/attack2_generated_graph/citeseer/graph_label0_604_525.txt")
                attack.attack()
                flag = True
            elif (attack_name == 3):
                attack = ModelExtractionAttack2(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 4):
                attack = ModelExtractionAttack3(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 5):
                attack = ModelExtractionAttack4(
                    datasetCora_merge, 0.25, './GNNIP/models/attack_3_subgraph_shadow_model_citeseer_6966.pkl')
                attack.attack()
                flag = True
            else:
                attack = ModelExtractionAttack5(
                    datasetCora_merge, 0.25, './pygip/models/attack_3_subgraph_shadow_model_citeseer_6966.pkl')
                attack.attack()
                flag = True
        elif (dataset_name == 3):
            if (attack_name == 1):
                attack = ModelExtractionAttack0(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 2):
                original_node_count = 19717
                new_node_count = 50
                output_file = "./GNNIP/data/attack2_generated_graph/pubmed/query_labels_pubmed.txt"
                self.generate_extended_label_file(
                    datasetCora_merge, original_node_count, new_node_count, output_file)
                attack = ModelExtractionAttack1(
                    datasetCora_merge, 0.25, "./GNNIP/data/attack2_generated_graph/pubmed/selected_index.txt",
                    "./GNNIP/data/attack2_generated_graph/pubmed/query_labels_pubmed.txt",
                    "./GNNIP/data/attack2_generated_graph/pubmed/graph_label0_0.657_667_.txt")
                attack.attack()
                flag = True
            elif (attack_name == 3):
                attack = ModelExtractionAttack2(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 4):
                attack = ModelExtractionAttack3(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 5):
                attack = ModelExtractionAttack4(
                    datasetCora_merge, 0.25, './GNNIP/models/attack_3_subgraph_shadow_model_pubmed_8063.pkl')
                attack.attack()
                flag = True
            else:
                flag = True
                attack = ModelExtractionAttack5(
                    datasetCora_merge, 0.25, './GNNIP/models/attack_3_subgraph_shadow_model_pubmed_8063.pkl')
                attack.attack()
        if (flag == True):
            datawm = datasetCora.datawm
            datasetCora_wm = graph_to_dataset(
                datawm, 0.25, dataset.dataset_name)
            datasetCora_wm.test_mask = torch.ones_like(
                datasetCora_wm.test_mask, dtype=torch.bool)
            net = Gcn_Net(attack.feature_number, attack.label_number)
            evaluation = GraphNeuralNetworkMetric(
                0, 0, net, datasetCora_wm.graph, datasetCora_wm.features, datasetCora_wm.test_mask, datasetCora_wm.labels)
            evaluation.evaluate()
            print("Watermark Graph - Accuracy:", evaluation.accuracy)


class Watermark_sage(Defense):
    def __init__(self, dataset, attack_node_fraction, wm_node=50, pr=0.1, pg=0, device="cpu"):
        super().__init__(dataset, attack_node_fraction)
        #
        self.wm_node = wm_node
        self.pr = pr
        self.pg = pg
        self.device = device

        self.model = GraphSAGE(
            in_channels=self.graph.ndata['feat'].shape[1], hidden_channels=128, out_channels=7)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def erdos_renyi_graph(self, n, p, directed=False):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if torch.rand(1).item() < p:
                    edges.append([i, j])
                    if not directed:
                        edges.append([j, i])

        if not edges:
            i, j = torch.randint(0, n, (2,))
            while i == j:
                j = torch.randint(0, n, (1,))
            edges = [[i.item(), j.item()], [j.item(), i.item()]]

        edges = torch.tensor(edges, dtype=torch.long).t()
        return edges

    def attack(self):
        # enable_cpu_affinity()
        data_wm = WatermarkGraph(n=self.wm_node, num_features=self.feature_number,
                                 num_classes=self.label_number, pr=self.pr, pg=self.pg, device=self.device).graph_wm

        self.datawm = data_wm

        # Choose devices
        data_wm = data_wm.to(self.device)
        data_wm.ndata['feat'] = data_wm.ndata['feat'].to(self.device)
        data_wm.ndata['label'] = data_wm.ndata['label'].to(self.device)

        # Create NeighborSampler and DataLoader for DGL
        sampler = NeighborSampler([5, 5])
        train_nids = self.graph.ndata['train_mask'].nonzero(as_tuple=True)[0]
        val_nids = self.graph.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_nids = self.graph.ndata['test_mask'].nonzero(as_tuple=True)[0]
        wm_nids = torch.arange(data_wm.number_of_nodes())

        train_collator = NodeCollator(self.graph, train_nids, sampler)
        val_collator = NodeCollator(self.graph, val_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)
        wm_collator = NodeCollator(data_wm, wm_nids, sampler)

        train_dataloader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False
        )

        val_dataloader = DataLoader(
            val_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=val_collator.collate,
            drop_last=False
        )

        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )

        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=False,
            collate_fn=wm_collator.collate,
            drop_last=False
        )

        # Create GraphSAGE model
        self.model = self.model.to(self.device)
        self.graph = self.graph.to(self.device)

        # Ensure features and labels are on the correct device
        self.graph.ndata['feat'] = self.graph.ndata['feat'].to(self.device)
        self.graph.ndata['label'] = self.graph.ndata['label'].to(self.device)

        # Training and evaluation
        for epoch in tqdm(range(1, 51)):
            loss = self.train(train_dataloader)
            val_acc = self.test(val_dataloader)
            test_acc = self.test(test_dataloader)

        nonmarked_acc = self.test(wm_dataloader)

        marked_acc = self.test(test_dataloader)
        print(f'Marked Acc: {marked_acc:.4f}')

        for epoch in tqdm(range(1, 16)):
            loss = self.train(wm_dataloader)
            test_acc = self.test(wm_dataloader)

        # Final results
        marked_acc = self.test(test_dataloader)
        watermark_acc = self.test(wm_dataloader)
        print('Final results')
        print('Non-Marked Acc: {:.4f}, Marked Acc: {:.4f}, Watermark Acc: {:.4f}'.format(
            nonmarked_acc, marked_acc, watermark_acc))
