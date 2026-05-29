import os
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from tqdm import tqdm

from models.attack.base import BaseAttack
from models.nn import GCN, ShadowNet, AttackNet
from utils.metrics import GraphNeuralNetworkMetric


class ModelExtractionAttack(BaseAttack):
    supported_api_types = {"dgl"}
    supported_datasets = {"Cora", "CiteSeer", "PubMed"}

    def __init__(self, dataset, attack_node_fraction, model_path=None, alpha=0.8):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.alpha = alpha
        self.graph = dataset.graph_data.to(self.device)
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata['train_mask']
        self.test_mask = self.graph.ndata['test_mask']

        # meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # attack params
        self.attack_node_num = int(dataset.num_nodes * attack_node_fraction)

        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    def _train_target_model(self):
        """
        Train the target model (GCN) on the original graph.
        """
        # Initialize GNN model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        # Training loop
        for epoch in range(200):
            self.net1.train()

            # Forward pass
            logits = self.net1(self.graph, self.features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation (optional)
            if epoch % 20 == 0:
                self.net1.eval()
                with torch.no_grad():
                    logits_val = self.net1(self.graph, self.features)
                    logp_val = F.log_softmax(logits_val, dim=1)
                    pred = logp_val.argmax(dim=1)
                    acc_val = (pred[self.test_mask] == self.labels[self.test_mask]).float().mean()
                    # You could print validation accuracy here

        return self.net1

    def _load_model(self, model_path):
        """
        Load a pre-trained model from a file.
        """
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        self.net1.load_state_dict(torch.load(model_path))
        self.net1.eval()
        return self.net1

    def attack(self):
        raise NotImplementedError


class ModelExtractionAttack0(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None, alpha=0.8):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.alpha = alpha

    def get_nonzero_indices(self, matrix_row):
        return np.where(matrix_row != 0)[0]

    def attack(self):
        """
        Main attack procedure.

        1. Samples a subset of nodes (`sub_graph_node_index`) for querying.
        2. Synthesizes features for neighboring nodes and their neighbors.
        3. Builds a sub-graph, trains a new GCN on it, and evaluates
           fidelity & accuracy w.r.t. the target model.
        """
        try:
            torch.cuda.empty_cache()
            g = self.graph.clone().to(self.device)
            g_matrix = g.adjacency_matrix().to_dense().cpu().numpy()
            del g

            sub_graph_node_index = np.random.choice(
                self.num_nodes, self.attack_node_num, replace=False).tolist()

            batch_size = 32
            features_query = self.features.clone()

            syn_nodes = []
            for node_index in sub_graph_node_index:
                one_step_node_index = self.get_nonzero_indices(g_matrix[node_index]).tolist()
                syn_nodes.extend(one_step_node_index)

                for first_order_node_index in one_step_node_index:
                    two_step_node_index = self.get_nonzero_indices(g_matrix[first_order_node_index]).tolist()
                    syn_nodes.extend(two_step_node_index)

            sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
            total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))

            # Process synthetic nodes in batches
            for i in range(0, len(sub_graph_syn_node_index), batch_size):
                batch_indices = sub_graph_syn_node_index[i:i + batch_size]

                for node_index in batch_indices:
                    features_query[node_index] = 0
                    one_step_node_index = self.get_nonzero_indices(g_matrix[node_index]).tolist()
                    one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))

                    num_one_step = len(one_step_node_index)
                    if num_one_step > 0:
                        for first_order_node_index in one_step_node_index:
                            this_node_degree = len(self.get_nonzero_indices(g_matrix[first_order_node_index]))
                            features_query[node_index] += (
                                    self.features[first_order_node_index] * self.alpha /
                                    torch.sqrt(torch.tensor(num_one_step * this_node_degree, device=self.device))
                            )

                    two_step_nodes = []
                    for first_order_node_index in one_step_node_index:
                        two_step_nodes.extend(self.get_nonzero_indices(g_matrix[first_order_node_index]).tolist())

                    total_two_step_node_index = list(set(two_step_nodes) - set(one_step_node_index))
                    total_two_step_node_index = list(
                        set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

                    num_two_step = len(total_two_step_node_index)
                    if num_two_step > 0:
                        for second_order_node_index in total_two_step_node_index:
                            this_node_first_step_nodes = self.get_nonzero_indices(
                                g_matrix[second_order_node_index]).tolist()
                            this_node_second_step_nodes = set()

                            for nodes_in_this_node in this_node_first_step_nodes:
                                this_node_second_step_nodes.update(
                                    self.get_nonzero_indices(g_matrix[nodes_in_this_node]).tolist())

                            this_node_second_step_nodes = this_node_second_step_nodes - set(this_node_first_step_nodes)
                            this_node_second_degree = len(this_node_second_step_nodes)

                            if this_node_second_degree > 0:
                                features_query[node_index] += (
                                        self.features[second_order_node_index] * (1 - self.alpha) /
                                        torch.sqrt(
                                            torch.tensor(num_two_step * this_node_second_degree, device=self.device))
                                )

                torch.cuda.empty_cache()

            # Update masks
            for i in range(self.num_nodes):
                if i in sub_graph_node_index:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                elif i in sub_graph_syn_node_index:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0
                else:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0

            # Create subgraph adjacency matrix
            sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
            for sub_index in range(len(total_sub_nodes)):
                sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

            del g_matrix

            sub_train_mask = self.train_mask[total_sub_nodes]
            sub_features = features_query[total_sub_nodes]
            sub_labels = self.labels[total_sub_nodes]

            # Get query labels
            self.net1.eval()
            with torch.no_grad():
                g = self.graph.to(self.device)
                logits_query = self.net1(g, features_query)
                _, labels_query = torch.max(logits_query, dim=1)
                sub_labels_query = labels_query[total_sub_nodes]
                del logits_query

            # Create DGL graph
            sub_g = nx.from_numpy_array(sub_g)
            sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
            sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
            sub_g = DGLGraph(sub_g)
            sub_g = sub_g.to(self.device)

            degs = sub_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            sub_g.ndata['norm'] = norm.unsqueeze(1)

            # Train extraction model
            net = GCN(self.num_features, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(200)):
                net.train()
                logits = net(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_mask], sub_labels_query[sub_train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net, g, self.features, self.test_mask, self.labels, labels_query
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

            self.net2 = net

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack1(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction):
        super().__init__(dataset, attack_node_fraction)
        self.attack_node_num = 700
        current_dir = os.path.dirname(os.path.abspath(__file__))
        generated_graph_dataset_path = os.path.join(current_dir, 'data', 'attack2_generated_graph',
                                                    dataset.__class__.__name__.lower())
        self.selected_node_file = os.path.join(generated_graph_dataset_path, "selected_index.txt")
        self.query_label_file = os.path.join(generated_graph_dataset_path, "query_labels.txt")
        self.shadow_graph_file = os.path.join(generated_graph_dataset_path, "graph_label.txt")

    def attack(self):
        """
        Main attack procedure.

        1. Reads selected nodes from file for training (attack) nodes.
        2. Reads query labels from another file.
        3. Builds a shadow graph from the given adjacency matrix file.
        4. Trains a shadow model on the selected nodes, then evaluates
           fidelity & accuracy against the original target graph.
        """
        try:
            torch.cuda.empty_cache()

            with open(self.selected_node_file, "r") as selected_node_file:
                attack_nodes = [int(line.strip()) for line in selected_node_file]

            # Identify the test nodes
            testing_nodes = [i for i in range(self.num_nodes) if i not in attack_nodes]

            attack_features = self.features[attack_nodes]

            # Update masks
            for i in range(self.num_nodes):
                if i in attack_nodes:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                else:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0

            sub_test_mask = self.test_mask

            with open(self.query_label_file, "r") as query_label_file:
                lines = query_label_file.readlines()
                all_query_labels = []
                attack_query = []
                for line in lines:
                    node_id, label = map(int, line.split())
                    all_query_labels.append(label)
                    if node_id in attack_nodes:
                        attack_query.append(label)

            attack_query = torch.LongTensor(attack_query).to(self.device)
            all_query_labels = torch.LongTensor(all_query_labels).to(self.device)

            with open(self.shadow_graph_file, "r") as shadow_graph_file:
                lines = shadow_graph_file.readlines()
                adj_matrix = np.zeros((self.attack_node_num, self.attack_node_num))
                for line in lines:
                    src, dst = map(int, line.split())
                    adj_matrix[src][dst] = 1
                    adj_matrix[dst][src] = 1

            g_shadow = np.asmatrix(adj_matrix)
            sub_g = nx.from_numpy_array(g_shadow)

            sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
            sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
            sub_g = DGLGraph(sub_g)
            sub_g = sub_g.to(self.device)

            degs = sub_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            sub_g.ndata['norm'] = norm.unsqueeze(1)

            # Create target graph
            adj_matrix = self.graph.adjacency_matrix().to_dense().cpu().numpy()
            sub_g_b = nx.from_numpy_array(adj_matrix)

            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(self.device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            net = ShadowNet(self.num_features, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("===================Model Extracting================================")
            for epoch in tqdm(range(200)):
                if epoch >= 3:
                    t0 = time.time()

                net.train()
                logits = net(sub_g, attack_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp, attack_query)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net, sub_g_b, self.features, self.test_mask,
                        all_query_labels, self.labels
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print(best_performance_metrics)

            self.net2 = net

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack2(ModelExtractionAttack):
    """
    ModelExtractionAttack2.

    A strategy that randomly samples a fraction of nodes as attack nodes,
    synthesizes identity features for all nodes, then trains an extraction
    model. The leftover nodes become test nodes.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        """
        Main attack procedure.

        1. Randomly select `attack_node_num` nodes as training nodes.
        2. Set up synthetic features as identity vectors for all nodes.
        3. Train a `Net_attack` model on these nodes with the queried labels.
        4. Evaluate fidelity & accuracy on a subset of leftover nodes.
        """
        try:
            torch.cuda.empty_cache()

            attack_nodes = []
            for i in range(self.attack_node_num):
                candidate_node = random.randint(0, self.num_nodes - 1)
                if candidate_node not in attack_nodes:
                    attack_nodes.append(candidate_node)

            test_num = 0
            for i in range(self.num_nodes):
                if i in attack_nodes:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                else:
                    if test_num < 1000:
                        self.test_mask[i] = 1
                        self.train_mask[i] = 0
                        test_num += 1
                    else:
                        self.test_mask[i] = 0
                        self.train_mask[i] = 0

            self.net1.eval()
            with torch.no_grad():
                logits_query = self.net1(self.graph, self.features)
                _, labels_query = torch.max(logits_query, dim=1)

            syn_features_np = np.eye(self.num_nodes)
            syn_features = torch.FloatTensor(syn_features_np).to(self.device)
            g = self.graph.to(self.device)

            degs = g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            g.ndata['norm'] = norm.unsqueeze(1)

            net_attack = AttackNet(self.num_nodes, self.num_classes).to(self.device)
            optimizer_original = torch.optim.Adam(net_attack.parameters(), lr=5e-2, weight_decay=5e-4)

            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(200)):
                if epoch >= 3:
                    t0 = time.time()

                net_attack.train()
                logits = net_attack(g, syn_features)
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp[self.train_mask.to(self.device)], labels_query[self.train_mask].to(self.device))

                optimizer_original.zero_grad()
                loss.backward()
                optimizer_original.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net_attack, g, syn_features,
                        self.test_mask.to(self.device),
                        self.labels.to(self.device),
                        labels_query.to(self.device)
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack3(ModelExtractionAttack):
    """
    ModelExtractionAttack3.

    A more complex extraction strategy that uses a "shadow graph index"
    file to build partial subgraphs and merges them. It queries selected
    nodes from a potential set and forms a combined adjacency matrix.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        """
        Main attack procedure.

        Steps:
        1. Loads indices for two subgraphs from text files.
        2. Selects `attack_node_num` nodes from the first subgraph index.
        3. Merges subgraph adjacency matrices and constructs a new graph
           with combined features.
        4. Trains a new GCN and evaluates fidelity & accuracy w.r.t. the
           original target.
        """
        try:
            torch.cuda.empty_cache()
            g_numpy = self.graph.adjacency_matrix().to_dense().cpu().numpy()

            current_dir = os.path.dirname(os.path.abspath(__file__))
            shadow_graph_dataset_path = os.path.join(current_dir, 'data', 'attack3_shadow_graph',
                                                     self.dataset.__class__.__name__.lower())

            sub_graph_index_b = []
            with open(os.path.abspath(
                    os.path.join(shadow_graph_dataset_path, 'attack_6_sub_shadow_graph_index_attack_2.txt')),
                    'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_b.append(int(ip))

            sub_graph_index_a = []
            with open(
                    os.path.abspath(os.path.join(shadow_graph_dataset_path, 'protential_1300_shadow_graph_index.txt')),
                    'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_a.append(int(ip))

            attack_node = []
            while len(attack_node) < self.attack_node_num:  # TODO potential bug: attack_node_num > all possible node
                protential_node_index = random.randint(0, len(sub_graph_index_b) - 1)
                protential_node = sub_graph_index_b[protential_node_index]
                if protential_node not in attack_node:
                    attack_node.append(int(protential_node))

            attack_features = self.features[attack_node].to(self.device)
            attack_labels = self.labels[attack_node].to(self.device)
            shadow_features = self.features[sub_graph_index_a].to(self.device)
            shadow_labels = self.labels[sub_graph_index_a].to(self.device)

            sub_graph_g_A = g_numpy[sub_graph_index_a]
            sub_graph_g_a = sub_graph_g_A[:, sub_graph_index_a]

            sub_graph_attack = g_numpy[attack_node]
            sub_graph_Attack = sub_graph_attack[:, attack_node]

            zeros_1 = np.zeros((len(attack_node), len(sub_graph_index_a)))
            zeros_2 = np.zeros((len(sub_graph_g_a), len(attack_node)))

            sub_graph_Attack = np.array(sub_graph_Attack)
            sub_graph_g_a = np.array(sub_graph_g_a)

            generated_graph_1 = np.concatenate((sub_graph_Attack, zeros_1), axis=1)
            generated_graph_2 = np.concatenate((zeros_2, sub_graph_g_a), axis=1)
            generated_graph = np.concatenate((generated_graph_1, generated_graph_2), axis=0)

            generated_features = torch.cat((attack_features, shadow_features), dim=0).to(self.device)
            generated_labels = torch.cat((attack_labels, shadow_labels), dim=0).to(self.device)

            generated_train_mask = torch.ones(len(generated_features), dtype=torch.bool, device=self.device)
            generated_test_mask = torch.ones(len(generated_features), dtype=torch.bool, device=self.device)

            generated_g = nx.from_numpy_array(generated_graph)
            generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
            generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
            generated_g = DGLGraph(generated_g)
            generated_g = generated_g.to(self.device)

            degs = generated_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            generated_g.ndata['norm'] = norm.unsqueeze(1)

            sub_graph_g_B = g_numpy[sub_graph_index_b]
            sub_graph_g_b = sub_graph_g_B[:, sub_graph_index_b]
            sub_graph_features_b = self.features[sub_graph_index_b].to(self.device)
            sub_graph_labels_b = self.labels[sub_graph_index_b].to(self.device)
            sub_graph_train_mask_b = self.train_mask[sub_graph_index_b].to(self.device)
            sub_graph_test_mask_b = self.test_mask[sub_graph_index_b].to(self.device)

            test_mask_length = min(len(sub_graph_test_mask_b), len(generated_train_mask))
            for i in range(test_mask_length):
                if i >= 140:
                    generated_train_mask[i] = 0
                    sub_graph_test_mask_b[i] = 1
                else:
                    generated_train_mask[i] = 1
                    sub_graph_test_mask_b[i] = 0

            if len(sub_graph_test_mask_b) > test_mask_length:
                sub_graph_test_mask_b[test_mask_length:] = 1

            sub_g_b = nx.from_numpy_array(sub_graph_g_b)
            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(self.device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            self.net1.eval()
            with torch.no_grad():
                logits_b = self.net1(sub_g_b, sub_graph_features_b)
                _, query_b = torch.max(logits_b, dim=1)

            net2 = GCN(self.num_features, self.num_classes).to(self.device)
            optimizer_a = torch.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(300)):
                if epoch >= 3:
                    t0 = time.time()

                net2.train()
                logits_a = net2(generated_g, generated_features)
                logp_a = F.log_softmax(logits_a, 1)
                loss_a = F.nll_loss(logp_a[generated_train_mask], generated_labels[generated_train_mask])

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net2, sub_g_b, sub_graph_features_b,
                        sub_graph_test_mask_b, sub_graph_labels_b, query_b
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

            self.net2 = net2

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack4(ModelExtractionAttack):
    """
    ModelExtractionAttack4.

    Another graph-based strategy that reads node indices from files,
    merges adjacency matrices, and links new edges based on feature similarity.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.model_path = model_path

    def attack(self):
        """
        Main attack procedure.

        1. Reads two sets of node indices from text files.
        2. Selects a fixed number of nodes from the target set for attack.
        3. Builds a combined adjacency matrix with zero blocks, then populates
           edges between shadow and attack nodes based on a distance threshold.
        4. Trains a new GCN on this combined graph and evaluates fidelity & accuracy.
        """
        try:
            torch.cuda.empty_cache()

            g_numpy = self.graph.adjacency_matrix().to_dense().cpu().numpy()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            shadow_graph_dataset_path = os.path.join(current_dir, 'data', 'attack3_shadow_graph',
                                                     self.dataset.__class__.__name__.lower())

            sub_graph_index_b = []
            with open(os.path.abspath(os.path.join(shadow_graph_dataset_path, 'target_graph_index.txt')),
                      'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_b.append(int(ip))

            sub_graph_index_a = []
            with open(
                    os.path.abspath(os.path.join(shadow_graph_dataset_path, 'protential_1200_shadow_graph_index.txt')),
                    'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_a.append(int(ip))

            attack_node_arg = 60
            attack_node = []
            while len(attack_node) < attack_node_arg:
                protential_node_index = random.randint(0, len(sub_graph_index_b) - 1)
                protential_node = sub_graph_index_b[protential_node_index]
                if protential_node not in attack_node:
                    attack_node.append(int(protential_node))

            attack_features = self.features[attack_node].cpu()
            attack_labels = self.labels[attack_node].cpu()
            shadow_features = self.features[sub_graph_index_a].cpu()
            shadow_labels = self.labels[sub_graph_index_a].cpu()

            sub_graph_g_A = np.array(g_numpy[sub_graph_index_a])
            sub_graph_g_a = np.array(sub_graph_g_A[:, sub_graph_index_a])
            sub_graph_Attack = np.zeros((len(attack_node), len(attack_node)))

            zeros_1 = np.zeros((len(attack_node), len(sub_graph_index_a)))
            zeros_2 = np.zeros((len(sub_graph_g_a), len(attack_node)))

            generated_graph = np.block([
                [sub_graph_Attack, zeros_1],
                [zeros_2, sub_graph_g_a]
            ])

            distance = []
            for i in range(100):
                index1 = i
                index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
                for index2 in index2_list:
                    distance.append(float(np.linalg.norm(
                        shadow_features[index1].cpu().numpy() -
                        shadow_features[int(index2)].cpu().numpy())))

            threshold = np.mean(distance)
            max_threshold = max(distance)

            generated_features = np.vstack((attack_features.cpu().numpy(), shadow_features.cpu().numpy()))
            generated_labels = np.concatenate([attack_labels.cpu().numpy(), shadow_labels.cpu().numpy()])

            for i in range(len(attack_features)):
                for loop in range(1000):
                    j = random.randint(0, len(shadow_features) - 1)
                    if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                    if loop > 500:
                        if np.linalg.norm(
                                generated_features[i] - generated_features[len(attack_features) + j]) < max_threshold:
                            generated_graph[i][len(attack_features) + j] = 1
                            generated_graph[len(attack_features) + j][i] = 1
                            break
                    if loop == 999:
                        print("one isolated node!")

            generated_train_mask = torch.ones(len(generated_features), dtype=torch.bool)
            generated_test_mask = torch.ones(len(generated_features), dtype=torch.bool)

            generated_features = torch.FloatTensor(generated_features).to(self.device)
            generated_labels = torch.LongTensor(generated_labels).to(self.device)
            generated_train_mask = generated_train_mask.to(self.device)
            generated_test_mask = generated_test_mask.to(self.device)

            generated_g = nx.from_numpy_array(generated_graph)
            generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
            generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
            generated_g = DGLGraph(generated_g)
            generated_g = generated_g.to(self.device)

            degs = generated_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            generated_g.ndata['norm'] = norm.unsqueeze(1)

            sub_graph_g_B = np.array(g_numpy[sub_graph_index_b])
            sub_graph_g_b = np.array(sub_graph_g_B[:, sub_graph_index_b])
            sub_graph_features_b = self.features[sub_graph_index_b].to(self.device)
            sub_graph_labels_b = self.labels[sub_graph_index_b].to(self.device)
            sub_graph_train_mask_b = self.train_mask[sub_graph_index_b].to(self.device)
            sub_graph_test_mask_b = self.test_mask[sub_graph_index_b].to(self.device)

            for i in range(len(sub_graph_test_mask_b)):
                if i >= 300:
                    sub_graph_train_mask_b[i] = 0
                    sub_graph_test_mask_b[i] = 1
                else:
                    sub_graph_train_mask_b[i] = 1
                    sub_graph_test_mask_b[i] = 0

            sub_g_b = nx.from_numpy_array(sub_graph_g_b)
            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(self.device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            self.net1.eval()
            with torch.no_grad():
                logits_b = self.net1(sub_g_b, sub_graph_features_b)
                _, query_b = torch.max(logits_b, dim=1)

            net2 = GCN(self.num_features, self.num_classes).to(self.device)
            optimizer_a = torch.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(300)):
                if epoch >= 3:
                    t0 = time.time()

                net2.train()
                logits_a = net2(generated_g, generated_features)
                logp_a = F.log_softmax(logits_a, 1)
                loss_a = F.nll_loss(logp_a[generated_train_mask], generated_labels[generated_train_mask])

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net2, sub_g_b, sub_graph_features_b,
                        sub_graph_test_mask_b, sub_graph_labels_b, query_b
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

            self.net2 = net2

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack5(ModelExtractionAttack):
    """
    ModelExtractionAttack5.

    Similar to ModelExtractionAttack4, but uses a slightly different
    strategy to link edges between nodes based on a threshold distance.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.model_path = model_path

    def attack(self):
        """
        Main attack procedure.

        1. Reads two sets of node indices (for target and shadow nodes).
        2. Builds a block adjacency matrix with all zero blocks, then links
           edges between attack nodes and shadow nodes if the feature distance
           is less than a threshold.
        3. Trains a new GCN on this combined graph and evaluates fidelity & accuracy.
        """
        try:
            torch.cuda.empty_cache()

            g_numpy = self.graph.adjacency_matrix().to_dense().cpu().numpy()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            shadow_graph_dataset_path = os.path.join(current_dir, 'data', 'attack3_shadow_graph',
                                                     self.dataset.__class__.__name__.lower())

            sub_graph_index_b = []
            with open(os.path.abspath(os.path.join(shadow_graph_dataset_path, 'target_graph_index.txt')),
                      'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_b.append(int(ip))

            sub_graph_index_a = []
            with open(
                    os.path.abspath(os.path.join(shadow_graph_dataset_path, 'protential_1200_shadow_graph_index.txt')),
                    'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_a.append(int(ip))

            attack_node = []
            while len(attack_node) < 60:
                protential_node_index = random.randint(0, len(sub_graph_index_b) - 1)
                protential_node = sub_graph_index_b[protential_node_index]
                if protential_node not in attack_node:
                    attack_node.append(int(protential_node))

            attack_features = self.features[attack_node].cpu()
            attack_labels = self.labels[attack_node].cpu()
            shadow_features = self.features[sub_graph_index_a].cpu()
            shadow_labels = self.labels[sub_graph_index_a].cpu()

            sub_graph_g_A = np.array(g_numpy[sub_graph_index_a])
            sub_graph_g_a = np.array(sub_graph_g_A[:, sub_graph_index_a])
            sub_graph_Attack = np.zeros((len(attack_node), len(attack_node)))

            zeros_1 = np.zeros((len(attack_node), len(sub_graph_index_a)))
            zeros_2 = np.zeros((len(sub_graph_g_a), len(attack_node)))

            generated_graph = np.block([
                [sub_graph_Attack, zeros_1],
                [zeros_2, sub_graph_g_a]
            ])

            distance = []
            for i in range(100):
                index1 = i
                index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
                for index2 in index2_list:
                    distance.append(float(np.linalg.norm(
                        shadow_features[index1].cpu().numpy() -
                        shadow_features[int(index2)].cpu().numpy())))

            threshold = np.mean(distance)
            max_threshold = max(distance)

            generated_features = np.vstack((attack_features.cpu().numpy(),
                                            shadow_features.cpu().numpy()))
            generated_labels = np.concatenate([attack_labels.cpu().numpy(),
                                               shadow_labels.cpu().numpy()])

            for i in range(len(attack_features)):
                for loop in range(1000):
                    j = random.randint(0, len(shadow_features) - 1)
                    feat_diff = generated_features[i] - generated_features[len(attack_features) + j]
                    dist = np.linalg.norm(feat_diff)

                    if dist < threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                    if loop > 500 and dist < max_threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                    if loop == 999:
                        print("one isolated node!")

            generated_features = torch.FloatTensor(generated_features).to(self.device)
            generated_labels = torch.LongTensor(generated_labels).to(self.device)
            generated_train_mask = torch.ones(len(generated_features), dtype=torch.bool, device=self.device)
            generated_test_mask = torch.ones(len(generated_features), dtype=torch.bool, device=self.device)

            generated_g = nx.from_numpy_array(generated_graph)
            generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
            generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
            generated_g = DGLGraph(generated_g)
            generated_g = generated_g.to(self.device)

            degs = generated_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            generated_g.ndata['norm'] = norm.unsqueeze(1)

            sub_graph_g_B = np.array(g_numpy[sub_graph_index_b])
            sub_graph_g_b = np.array(sub_graph_g_B[:, sub_graph_index_b])
            sub_graph_features_b = self.features[sub_graph_index_b].to(self.device)
            sub_graph_labels_b = self.labels[sub_graph_index_b].to(self.device)
            sub_graph_train_mask_b = self.train_mask[sub_graph_index_b].to(self.device)
            sub_graph_test_mask_b = self.test_mask[sub_graph_index_b].to(self.device)

            for i in range(len(sub_graph_test_mask_b)):
                if i >= 300:
                    sub_graph_train_mask_b[i] = 0
                    sub_graph_test_mask_b[i] = 1
                else:
                    sub_graph_train_mask_b[i] = 1
                    sub_graph_test_mask_b[i] = 0

            sub_g_b = nx.from_numpy_array(sub_graph_g_b)
            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(self.device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            self.net1.eval()
            with torch.no_grad():
                logits_b = self.net1(sub_g_b, sub_graph_features_b)
                _, query_b = torch.max(logits_b, dim=1)

            net2 = GCN(self.num_features, self.num_classes).to(self.device)
            optimizer_a = torch.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(300)):
                if epoch >= 3:
                    t0 = time.time()

                net2.train()
                logits_a = net2(generated_g, generated_features)
                logp_a = F.log_softmax(logits_a, 1)
                loss_a = F.nll_loss(logp_a[generated_train_mask],
                                    generated_labels[generated_train_mask])

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net2, sub_g_b, sub_graph_features_b,
                        sub_graph_test_mask_b, sub_graph_labels_b, query_b
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

            self.net2 = net2

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            torch.cuda.empty_cache()
            raise
