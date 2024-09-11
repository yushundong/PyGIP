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
from tqdm import tqdm


class GraphNeuralNetworkMetric:
    def __init__(self, fidelity=0, accuracy=0, model=None, graph=None, features=None, mask=None, labels=None, query_labels=None):
        """
        Initializes the Metric class with true and predicted values,
        and calculates fidelity and accuracy.
        """

        self.model = model
        self.graph = graph
        self.features = features
        self.mask = mask
        self.labels = labels
        self.query_labels = query_labels

        self.accuracy = accuracy
        self.fidelity = fidelity

    def evaluate_helper(self, model, graph, features, labels, mask):
        if model == None or graph == None or features == None or labels == None or mask == None:
            return None
        model.eval()
        with th.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = th.max(logits, dim=1)
            correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    def evaluate(self):
        self.accuracy = self.evaluate_helper(
            self.model, self.graph, self.features, self.labels, self.mask)
        self.fidelity = self.evaluate_helper(
            self.model, self.graph, self.features, self.query_labels, self.mask)

    def __str__(self):
        """
        Returns a string representation of the Metric instance,
        showing fidelity and accuracy.

        :return: String representation of the Metric instance.
        """
        return f"Fidelity: {self.fidelity}, Accuracy: {self.accuracy}"


class Gcn_Net(nn.Module):
    def __init__(self, feature_number, label_number):
        super(Gcn_Net, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x


class Net_shadow(th.nn.Module):
    def __init__(self, feature_number, label_number):
        super(Net_shadow, self).__init__()
        self.layer1 = GraphConv(feature_number, 16)
        self.layer2 = GraphConv(16, label_number)

    def forward(self, g, features):
        x = th.nn.functional.relu(self.layer1(g, features))
        # x = torch.nn.functional.dropout(x, 0.2)
        # x = F.dropout(x, training=self.training)
        x = self.layer2(g, x)
        return x


class Net_attack(nn.Module):
    def __init__(self, feature_number, label_number):
        super(Net_attack, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class ModelExtractionAttack:
    def __init__(self, dataset, attack_node_fraction, model_path=None):
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
        self.attack_node_fraction = attack_node_fraction

        # features, labels
        self.features = dataset.features
        self.labels = dataset.labels

        # train_mask, test_mask
        self.train_mask = dataset.train_mask
        self.test_mask = dataset.test_mask

        if model_path == None:
            # Train the GCN traget model.
            self.train_target_model()
        else:
            self.net1 = Gcn_Net(self.feature_number, self.label_number)
            optimizer_b = th.optim.Adam(
                self.net1.parameters(), lr=1e-2, weight_decay=5e-4)
            self.net1.load_state_dict(th.load(model_path))

    def train_target_model(self):
        # Train the GCN target model.
        focus_graph = self.graph
        degs = focus_graph.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        # if cuda != None:
        #     norm = norm.cuda()
        focus_graph.ndata['norm'] = norm.unsqueeze(1)
        self.net1 = Gcn_Net(self.feature_number, self.label_number)
        optimizer = th.optim.Adam(
            self.net1.parameters(), lr=1e-2, weight_decay=5e-4)
        dur = []

        print("=========Target Model Generating==========================")
        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            self.net1.train()
            logits = self.net1(focus_graph, self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask],
                              self.labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)


class ModelExtractionAttack0(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None, alpha=0.8):
        super().__init__(dataset, attack_node_fraction, model_path)
        #
        self.alpha = alpha

    def attack(self):
        g = self.graph.clone()
        g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())

        # Step1, sample attack_node_number nodes.
        sub_graph_node_index = []

        sub_graph_node_index = np.random.choice(
            self.node_number, self.attack_node_number, replace=False).tolist()

        sub_labels = self.labels[sub_graph_node_index]

        # generate syn nodes for this sub-graph index
        syn_nodes = []
        for node_index in sub_graph_node_index:
            # get nodes
            one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
            two_step_node_index = []
            for first_order_node_index in one_step_node_index:
                syn_nodes.append(first_order_node_index)
                two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[
                    1].tolist()
                # =============================================================================
                for second_order_node_index in two_step_node_index:
                    syn_nodes.append(second_order_node_index)
                # =============================================================================

        sub_graph_syn_node_index = list(
            set(syn_nodes) - set(sub_graph_node_index))

        total_sub_nodes = list(
            set(sub_graph_syn_node_index + sub_graph_node_index))

        np_features_query = self.features.clone()

        for node_index in sub_graph_syn_node_index:
            # initialized as zero
            np_features_query[node_index] = np_features_query[node_index] * 0
            # get one step and two steps nodes
            one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
            one_step_node_index = list(
                set(one_step_node_index).intersection(set(sub_graph_node_index)))

            total_two_step_node_index = []
            num_one_step = len(one_step_node_index)
            for first_order_node_index in one_step_node_index:
                # caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
                # new_array = features[first_order_node_index]*0.8/num_one_step
                this_node_degree = len(
                    g_matrix[first_order_node_index, :].nonzero()[1].tolist())
                np_features_query[node_index] = torch.from_numpy(np.sum(
                    [np_features_query[node_index],
                     self.features[first_order_node_index] * self.alpha / math.sqrt(num_one_step * this_node_degree)],
                    axis=0))

                two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[
                    1].tolist()
                total_two_step_node_index = list(
                    set(total_two_step_node_index + two_step_node_index) - set(one_step_node_index))
            total_two_step_node_index = list(
                set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

            num_two_step = len(total_two_step_node_index)
            for second_order_node_index in total_two_step_node_index:

                # caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
                this_node_second_step_nodes = []
                this_node_first_step_nodes = g_matrix[second_order_node_index, :].nonzero()[
                    1].tolist()
                for nodes_in_this_node in this_node_first_step_nodes:
                    this_node_second_step_nodes = list(
                        set(this_node_second_step_nodes + g_matrix[nodes_in_this_node, :].nonzero()[1].tolist()))
                this_node_second_step_nodes = list(
                    set(this_node_second_step_nodes) - set(this_node_first_step_nodes))

                this_node_second_degree = len(this_node_second_step_nodes)
                np_features_query[node_index] = torch.from_numpy(np.sum(
                    [np_features_query[node_index],
                     self.features[second_order_node_index] * (1 - self.alpha) / math.sqrt(num_two_step * this_node_second_degree)],
                    axis=0))

        features_query = th.FloatTensor(np_features_query)
        # use original features

        # generate sub-graph adj-matrix, features, labels

        total_sub_nodes = list(
            set(sub_graph_syn_node_index + sub_graph_node_index))
        sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
        for sub_index in range(len(total_sub_nodes)):
            sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

        for i in range(self.node_number):
            if i in sub_graph_node_index:
                self.test_mask[i] = 0
                self.train_mask[i] = 1
                continue
            if i in sub_graph_syn_node_index:
                self.test_mask[i] = 1
                self.train_mask[i] = 0
            else:
                self.test_mask[i] = 1
                self.train_mask[i] = 0

        sub_train_mask = self.train_mask[total_sub_nodes]
        sub_features = features_query[total_sub_nodes]
        sub_labels = self.labels[total_sub_nodes]
        # gcn_msg = fn.copy_src(src='h', out='m')
        # gcn_reduce = fn.sum(msg='m', out='h')

        sub_features = th.FloatTensor(sub_features)
        sub_labels = th.LongTensor(sub_labels)
        sub_train_mask = sub_train_mask
        sub_test_mask = self.test_mask
        # sub_g = DGLGraph(nx.from_numpy_matrix(sub_g))

        self.net1.eval()

        logits_query = self.net1(g, self.features)
        _, labels_query = th.max(logits_query, dim=1)

        sub_labels_query = labels_query[total_sub_nodes]

        # graph preprocess and calculate normalization factor
        sub_g = nx.from_numpy_array(sub_g)
        # add self loop

        sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
        sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))

        sub_g = DGLGraph(sub_g)
        n_edges = sub_g.number_of_edges()
        # normalization
        degs = sub_g.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        sub_g.ndata['norm'] = norm.unsqueeze(1)

        # create GCN model

        net = Gcn_Net(self.feature_number, self.label_number)

        optimizer = th.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
        dur = []

        print("=========Model Extracting==========================")
        best_performance_metrics = GraphNeuralNetworkMetric()
        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            net.train()
            logits = net(sub_g, sub_features)

            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[sub_train_mask],
                              sub_labels_query[sub_train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net, g, self.features, self.test_mask, self.labels, labels_query)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

        print("========================Final results:=========================================")
        print(best_performance_metrics)


class ModelExtractionAttack1(ModelExtractionAttack):

    def __init__(self, dataset, attack_node_fraction, selected_node_file, query_label_file, shadow_graph_file=None):
        super().__init__(dataset, attack_node_fraction)
        self.attack_node_number = 700
        self.selected_node_file = selected_node_file
        self.query_label_file = query_label_file
        self.shadow_graph_file = shadow_graph_file

    def attack(self):

        # read the selected node file
        selected_node_file = open(self.selected_node_file, "r")
        lines1 = selected_node_file.readlines()
        attack_nodes = []
        for line_1 in lines1:
            attack_nodes.append(int(line_1))
        selected_node_file.close()

        # find the testing node
        testing_nodes = []
        for i in range(self.node_number):
            if i not in attack_nodes:
                testing_nodes.append(i)

        attack_features = self.features[attack_nodes]

        # mark the test/train split.
        for i in range(self.node_number):
            if i in attack_nodes:
                self.test_mask[i] = 0
                self.train_mask[i] = 1
            else:
                self.test_mask[i] = 1
                self.train_mask[i] = 0

        sub_test_mask = self.test_mask

        # get their labels
        query_label_file = open(self.query_label_file, "r")
        lines2 = query_label_file.readlines()
        all_query_labels = []
        attack_query = []
        for line_2 in lines2:
            all_query_labels.append(int(line_2.split()[1]))
            if int(line_2.split()[0]) in attack_nodes:
                attack_query.append(int(line_2.split()[1]))
        query_label_file.close()

        attack_query = torch.LongTensor(attack_query)
        all_query_labels = torch.LongTensor(all_query_labels)

        # build shadow graph
        shadow_graph_file = open(self.shadow_graph_file, "r")
        lines3 = shadow_graph_file.readlines()
        adj_matrix = np.zeros(
            (self.attack_node_number, self.attack_node_number))
        for line_3 in lines3:
            list_line = line_3.split()
            adj_matrix[int(list_line[0])][int(list_line[1])] = 1
            adj_matrix[int(list_line[1])][int(list_line[0])] = 1
        shadow_graph_file.close()

        g_shadow = np.asmatrix(adj_matrix)
        sub_g = nx.from_numpy_array(g_shadow)

        # add self loop
        sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
        sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
        sub_g = DGLGraph(sub_g)
        n_edges = sub_g.number_of_edges()

        # normalization
        degs = sub_g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        sub_g.ndata['norm'] = norm.unsqueeze(1)

        # build GCN

        # todo check this
        # g = DGLGraph(data.graph)
        # g_numpy = nx.to_numpy_array(data.graph)
        sub_g_b = nx.from_numpy_array(
            np.asmatrix(self.graph.adjacency_matrix().to_dense()))

        # graph preprocess and calculate normalization factor
        # sub_g_b = nx.from_numpy_array(sub_g_b)
        # add self loop

        sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
        sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))

        sub_g_b = DGLGraph(sub_g_b)
        n_edges = sub_g_b.number_of_edges()
        # normalization
        degs = sub_g_b.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        sub_g_b.ndata['norm'] = norm.unsqueeze(1)

        # Train the DNN
        net = Net_shadow(self.feature_number, self.label_number)
        print(net)

        #
        optimizer = torch.optim.Adam(
            net.parameters(), lr=1e-2, weight_decay=5e-4)

        dur = []

        best_performance_metrics = GraphNeuralNetworkMetric()

        print("===================Model Extracting================================")

        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            net.train()
            logits = net(sub_g, attack_features)
            logp = torch.nn.functional.log_softmax(logits, dim=1)
            loss = torch.nn.functional.nll_loss(logp, attack_query)

            # weights = [1/num_0, 1/num_1, 1/num_2, 1/num_3, 1/num_4, 1/num_5, 1/num_6]
            # class_weights = th.FloatTensor(weights)
        # =============================================================================
        #     criterion = torch.nn.CrossEntropyLoss()
        #     loss = criterion(logp, attack_query)
        # =============================================================================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net, sub_g_b, self.features, self.test_mask, all_query_labels, self.labels)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

        print(best_performance_metrics)


class ModelExtractionAttack2(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):

        # sample nodes
        attack_nodes = []
        for i in range(self.attack_node_number):
            candidate_node = random.randint(0, self.node_number - 1)
            if candidate_node not in attack_nodes:
                attack_nodes.append(candidate_node)

        test_num = 0
        for i in range(self.node_number):
            if i in attack_nodes:
                self.test_mask[i] = 0
                self.train_mask[i] = 1
                continue
            else:
                if test_num < 1000:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0
                    test_num = test_num + 1
                else:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 0

        self.net1.eval()

        # Generate Label
        logits_query = self.net1(self.graph, self.features)
        _, labels_query = th.max(logits_query, dim=1)

        syn_features_np = np.eye(self.node_number)
        syn_features = th.FloatTensor(syn_features_np)

        # normalization
        degs = self.graph.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        self.graph.ndata['norm'] = norm.unsqueeze(1)

        net_attack = Net_attack(self.node_number, self.label_number)

        optimizer_original = th.optim.Adam(
            net_attack.parameters(), lr=5e-2, weight_decay=5e-4)
        dur = []

        best_performance_metrics = GraphNeuralNetworkMetric()

        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            net_attack.train()
            logits = net_attack(self.graph, syn_features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask],
                              labels_query[self.train_mask])

            optimizer_original.zero_grad()
            loss.backward()
            optimizer_original.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net_attack, self.graph, syn_features, self.test_mask, self.labels, labels_query)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)
            # print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid  {:.4f} | Time(s) {:.4f}".format(
            #     epoch, loss.item(), focus_gnn_metrics.accuracy, focus_gnn_metrics.fidelity, np.mean(dur)))

        print(best_performance_metrics)


class ModelExtractionAttack3(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        g_numpy = self.graph.adjacency_matrix().to_dense().numpy()
        sub_graph_index_b = []

        # sample nodes
        sub_graph_index_b = []
        fileObject = open('./pygip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
                          '/target_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_b.append(int(ip))
        fileObject.close()

        sub_graph_index_a = []
        fileObject = open('./pygip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
                          '/protential_1300_shadow_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_a.append(int(ip))
        fileObject.close()

        # choose attack features in graphA
        attack_node = []
        while len(attack_node) < self.attack_node_number:
            protential_node_index = random.randint(
                0, len(sub_graph_index_b) - 1)
            protential_node = sub_graph_index_b[protential_node_index]
            if protential_node not in attack_node:
                attack_node.append(int(protential_node))

        attack_features = self.features[attack_node]
        attack_labels = self.labels[attack_node]
        shadow_features = self.features[sub_graph_index_a]
        shadow_labels = self.labels[sub_graph_index_a]

        sub_graph_g_A = g_numpy[sub_graph_index_a]
        sub_graph_g_a = sub_graph_g_A[:, sub_graph_index_a]

        sub_graph_attack = g_numpy[attack_node]
        sub_graph_Attack = sub_graph_attack[:, attack_node]

        generated_graph_1 = np.hstack(
            (sub_graph_Attack, np.zeros((len(attack_node), len(sub_graph_index_a)))))
        generated_graph_2 = np.hstack(
            (np.zeros((len(sub_graph_g_a), len(attack_node))), sub_graph_g_a))
        generated_graph = np.vstack((generated_graph_1, generated_graph_2))

        # caculate the distance of this features to every node in graphA
        generated_features = np.vstack((attack_features, shadow_features))
        generated_labels = np.hstack((attack_labels, shadow_labels))

        # train two model and evaluate
        generated_train_mask = np.ones(len(generated_features))
        print("generated_train_mask", len(generated_train_mask))
        generated_test_mask = np.ones(len(generated_features))

        generated_features = th.FloatTensor(generated_features)
        generated_labels = th.LongTensor(generated_labels)
        generated_train_mask = th.ByteTensor(generated_train_mask)
        generated_test_mask = th.ByteTensor(generated_test_mask)

        generated_g = nx.from_numpy_array(generated_graph)

        generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
        generated_g.add_edges_from(
            zip(generated_g.nodes(), generated_g.nodes()))

        generated_g = DGLGraph(generated_g)
        n_edges = generated_g.number_of_edges()
        # normalization
        degs = generated_g.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        generated_g.ndata['norm'] = norm.unsqueeze(1)

        dur = []

        # Here we use a pre-trained model, you may replace the model with yours.

        # th.save(net.state_dict(), "./models/attack_3_subgraph_shadow_model_pubmed.pkl")

        # for sub_graph_B
        sub_graph_g_B = g_numpy[sub_graph_index_b]
        sub_graph_g_b = sub_graph_g_B[:, sub_graph_index_b]
        sub_graph_features_b = self.features[sub_graph_index_b]
        sub_graph_labels_b = self.labels[sub_graph_index_b]
        sub_graph_train_mask_b = self.train_mask[sub_graph_index_b]
        sub_graph_test_mask_b = self.test_mask[sub_graph_index_b]

        for i in range(len(sub_graph_test_mask_b)):
            if i >= 140:
                generated_train_mask[i] = 0
                sub_graph_test_mask_b[i] = 1
            else:
                generated_train_mask[i] = 1
                sub_graph_test_mask_b[i] = 0

        for i in range(len(sub_graph_test_mask_b)):
            if i >= 140:
                sub_graph_train_mask_b[i] = 0
                sub_graph_test_mask_b[i] = 1
            else:
                sub_graph_train_mask_b[i] = 1
                sub_graph_test_mask_b[i] = 0
        # =============================================================================
        # for i in range(len(sub_graph_test_mask_b)):
        #     #if sub_graph_test_mask_b[i] == 0:
        #     sub_graph_test_mask_b[i] = 1
        # =============================================================================

        sub_features_b = th.FloatTensor(sub_graph_features_b)
        sub_labels_b = th.LongTensor(sub_graph_labels_b)
        sub_train_mask_b = sub_graph_train_mask_b
        sub_test_mask_b = sub_graph_test_mask_b
        # sub_g_b = DGLGraph(nx.from_numpy_matrix(sub_graph_g_b))

        sub_g_b = nx.from_numpy_array(sub_graph_g_b)

        # graph preprocess and calculate normalization factor
        # sub_g_b = nx.from_numpy_array(sub_g_b)
        # add self loop

        sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
        sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))

        sub_g_b = DGLGraph(sub_g_b)
        n_edges = sub_g_b.number_of_edges()
        # normalization
        degs = sub_g_b.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        sub_g_b.ndata['norm'] = norm.unsqueeze(1)

        self.net1.eval()
        logits_b = self.net1(sub_g_b, sub_features_b)
        # logits_b = F.log_softmax(logits_b, 1)
        _, query_b = th.max(logits_b, dim=1)

        net2 = Gcn_Net(self.feature_number, self.label_number)
        optimizer_a = th.optim.Adam(
            net2.parameters(), lr=1e-2, weight_decay=5e-4)

        best_performance_metrics = GraphNeuralNetworkMetric()

        for epoch in tqdm(range(300)):
            if epoch >= 3:
                t0 = time.time()

            net2.train()
            logits_a = net2(generated_g, generated_features)
            logp_a = F.log_softmax(logits_a, 1)
            # generated_train_mask
            loss_a = F.nll_loss(logp_a[generated_train_mask],
                                generated_labels[generated_train_mask])

            optimizer_a.zero_grad()
            loss_a.backward()
            optimizer_a.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net2, sub_g_b, sub_features_b, sub_test_mask_b, sub_labels_b, query_b)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)
            # print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid  {:.4f} | Time(s) {:.4f}".format(
            #     epoch, loss_a.item(), focus_gnn_metrics.accuracy, focus_gnn_metrics.fidelity, np.mean(dur)))

        print(best_performance_metrics)


class ModelExtractionAttack4(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.model_path = model_path

    def attack(self):
        g_numpy = self.graph.adjacency_matrix().to_dense().numpy()

        # sample nodes
        sub_graph_index_b = []
        fileObject = open('./pygip/data/' + self.dataset.dataset_name +
                          '/target_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_b.append(int(ip))
        fileObject.close()

        sub_graph_index_a = []
        fileObject = open('./pygip/data/' + self.dataset.dataset_name +
                          '/protential_1200_shadow_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_a.append(int(ip))
        fileObject.close()

        attack_node_arg = 60

        # choose attack features in graphA
        attack_node = []
        while len(attack_node) < attack_node_arg:
            protential_node_index = random.randint(
                0, len(sub_graph_index_b) - 1)
            protential_node = sub_graph_index_b[protential_node_index]
            if protential_node not in attack_node:
                attack_node.append(int(protential_node))

        attack_features = self.features[attack_node]
        attack_labels = self.labels[attack_node]
        shadow_features = self.features[sub_graph_index_a]
        shadow_labels = self.labels[sub_graph_index_a]

        sub_graph_g_A = g_numpy[sub_graph_index_a]
        sub_graph_g_a = sub_graph_g_A[:, sub_graph_index_a]

        sub_graph_Attack = np.zeros((len(attack_node), len(attack_node)))

        generated_graph_1 = np.hstack(
            (sub_graph_Attack, np.zeros((len(attack_node), len(sub_graph_index_a)))))
        generated_graph_2 = np.hstack(
            (np.zeros((len(sub_graph_g_a), len(attack_node))), sub_graph_g_a))
        generated_graph = np.vstack((generated_graph_1, generated_graph_2))

        distance = []
        for i in range(100):
            index1 = i
            index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
            for index2 in index2_list:
                distance.append(float(np.linalg.norm(
                    shadow_features[index1]-shadow_features[int(index2)])))

        threshold = np.mean(distance)
        max_threshold = max(distance)

        # caculate the distance of this features to every node in graphA
        generated_features = np.vstack((attack_features, shadow_features))
        generated_labels = np.hstack((attack_labels, shadow_labels))

        for i in range(len(attack_features)):
            for loop in range(1000):
                j = random.randint(0, len(shadow_features) - 1)
            # for j in range(len(shadow_features)):
                if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < threshold:
                    generated_graph[i][len(attack_features) + j] = 1
                    generated_graph[len(attack_features) + j][i] = 1
                    break
                if loop > 500:
                    # for k in range(len(shadow_features)):
                    if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < max_threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                if loop == 999:
                    print("one isolated node!")

            # train two model and evaluate
        generated_train_mask = np.ones(len(generated_features))
        generated_test_mask = np.ones(len(generated_features))

        generated_features = th.FloatTensor(generated_features)
        generated_labels = th.LongTensor(generated_labels)
        # generated_train_mask = th.ByteTensor(generated_train_mask)
        # generated_test_mask = th.ByteTensor(generated_test_mask)

        generated_g = nx.from_numpy_array(generated_graph)

        # graph preprocess and calculate normalization factor
        # sub_g = nx.from_numpy_array(sub_g)
        # add self loop

        generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
        generated_g.add_edges_from(
            zip(generated_g.nodes(), generated_g.nodes()))

        generated_g = DGLGraph(generated_g)
        n_edges = generated_g.number_of_edges()
        # normalization
        degs = generated_g.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        generated_g.ndata['norm'] = norm.unsqueeze(1)

        dur = []

        # th.save(net.state_dict(), "./models/attack_3_subgraph_shadow_model_pubmed.pkl")

        # for sub_graph_B
        sub_graph_g_B = g_numpy[sub_graph_index_b]
        sub_graph_g_b = sub_graph_g_B[:, sub_graph_index_b]
        sub_graph_features_b = self.features[sub_graph_index_b]
        sub_graph_labels_b = self.labels[sub_graph_index_b]
        sub_graph_train_mask_b = self.train_mask[sub_graph_index_b]
        sub_graph_test_mask_b = self.test_mask[sub_graph_index_b]

        for i in range(len(sub_graph_test_mask_b)):
            if i >= 300:
                sub_graph_train_mask_b[i] = 0
                sub_graph_test_mask_b[i] = 1
            else:
                sub_graph_train_mask_b[i] = 1
                sub_graph_test_mask_b[i] = 0

        sub_features_b = th.FloatTensor(sub_graph_features_b)
        sub_labels_b = th.LongTensor(sub_graph_labels_b)
        sub_train_mask_b = sub_graph_train_mask_b
        sub_test_mask_b = sub_graph_test_mask_b
        # sub_g_b = DGLGraph(nx.from_numpy_matrix(sub_graph_g_b))

        sub_g_b = nx.from_numpy_array(sub_graph_g_b)

        # graph preprocess and calculate normalization factor
        # sub_g_b = nx.from_numpy_array(sub_g_b)
        # add self loop

        sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
        sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))

        sub_g_b = DGLGraph(sub_g_b)
        n_edges = sub_g_b.number_of_edges()
        # normalization
        degs = sub_g_b.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        sub_g_b.ndata['norm'] = norm.unsqueeze(1)

        self.net1.eval()
        logits_b = self.net1(sub_g_b, sub_features_b)
        # logits_b = F.log_softmax(logits_b, 1)
        _, query_b = th.max(logits_b, dim=1)

        net2 = Gcn_Net(self.feature_number, self.label_number)
        optimizer_a = th.optim.Adam(
            net2.parameters(), lr=1e-2, weight_decay=5e-4)
        best_performance_metrics = GraphNeuralNetworkMetric()

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

            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net2, sub_g_b, sub_features_b, sub_test_mask_b, sub_labels_b, query_b)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)
            # print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid  {:.4f} | Time(s) {:.4f}".format(
            #     epoch, loss_a.item(), focus_gnn_metrics.accuracy, focus_gnn_metrics.fidelity, np.mean(dur)))

        print(best_performance_metrics)


class ModelExtractionAttack5(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.model_path = model_path

    def attack(self):
        g_numpy = self.graph.adjacency_matrix().to_dense().numpy()

        sub_graph_index_b = []
        fileObject = open('./pygip/data/' + self.dataset.dataset_name +
                          '/target_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_b.append(int(ip))
        fileObject.close()

        sub_graph_index_a = []
        fileObject = open('./pygip/data/' + self.dataset.dataset_name +
                          '/protential_1200_shadow_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_a.append(int(ip))
        fileObject.close()

        attack_node = []
        while len(attack_node) < 60:
            protential_node_index = random.randint(
                0, len(sub_graph_index_b) - 1)
            protential_node = sub_graph_index_b[protential_node_index]
            if protential_node not in attack_node:
                attack_node.append(int(protential_node))

        attack_features = self.features[attack_node]
        attack_labels = self.labels[attack_node]
        shadow_features = self.features[sub_graph_index_a]
        shadow_labels = self.labels[sub_graph_index_a]

        sub_graph_g_A = g_numpy[sub_graph_index_a]
        sub_graph_g_a = sub_graph_g_A[:, sub_graph_index_a]

        sub_graph_Attack = np.zeros((len(attack_node), len(attack_node)))

        generated_graph_1 = np.hstack(
            (sub_graph_Attack, np.zeros((len(attack_node), len(sub_graph_index_a)))))
        generated_graph_2 = np.hstack(
            (np.zeros((len(sub_graph_g_a), len(attack_node))), sub_graph_g_a))
        generated_graph = np.vstack((generated_graph_1, generated_graph_2))

        distance = []
        for i in range(100):
            index1 = i
            index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
            for index2 in index2_list:
                distance.append(float(np.linalg.norm(
                    shadow_features[index1]-shadow_features[int(index2)])))

        threshold = np.mean(distance)
        max_threshold = max(distance)

        # caculate the distance of this features to every node in graphA
        generated_features = np.vstack((attack_features, shadow_features))
        generated_labels = np.hstack((attack_labels, shadow_labels))

        for i in range(len(attack_features)):
            for loop in range(1000):
                j = random.randint(0, len(shadow_features) - 1)
            # for j in range(len(shadow_features)):
                if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < threshold:
                    generated_graph[i][len(attack_features) + j] = 1
                    generated_graph[len(attack_features) + j][i] = 1
                    break
                if loop > 500:
                    # for k in range(len(shadow_features)):
                    if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < max_threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                if loop == 999:

                    print("one isolated node!")

        # train two model and evaluate
        generated_train_mask = np.ones(len(generated_features))
        generated_test_mask = np.ones(len(generated_features))

        generated_features = th.FloatTensor(generated_features)
        generated_labels = th.LongTensor(generated_labels)
        # generated_train_mask = th.ByteTensor(generated_train_mask)
        # generated_test_mask = th.ByteTensor(generated_test_mask)

        generated_g = nx.from_numpy_array(generated_graph)

        # graph preprocess and calculate normalization factor
        # sub_g = nx.from_numpy_array(sub_g)
        # add self loop

        generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
        generated_g.add_edges_from(
            zip(generated_g.nodes(), generated_g.nodes()))

        generated_g = DGLGraph(generated_g)
        n_edges = generated_g.number_of_edges()
        # normalization
        degs = generated_g.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        generated_g.ndata['norm'] = norm.unsqueeze(1)

        dur = []

        # th.save(net.state_dict(), "./models/attack_3_subgraph_shadow_model_pubmed.pkl")

        # for sub_graph_B
        sub_graph_g_B = g_numpy[sub_graph_index_b]
        sub_graph_g_b = sub_graph_g_B[:, sub_graph_index_b]

        sub_graph_features_b = self.features[sub_graph_index_b]

        sub_graph_labels_b = self.labels[sub_graph_index_b]

        sub_graph_train_mask_b = self.train_mask[sub_graph_index_b]

        sub_graph_test_mask_b = self.test_mask[sub_graph_index_b]

        for i in range(len(sub_graph_test_mask_b)):
            if i >= 300:
                sub_graph_train_mask_b[i] = 0
                sub_graph_test_mask_b[i] = 1
            else:
                sub_graph_train_mask_b[i] = 1
                sub_graph_test_mask_b[i] = 0
        # =============================================================================
        # for i in range(len(sub_graph_test_mask_b)):
        #     #if sub_graph_test_mask_b[i] == 0:
        #     sub_graph_test_mask_b[i] = 1
        # =============================================================================

        sub_features_b = th.FloatTensor(sub_graph_features_b)
        sub_labels_b = th.LongTensor(sub_graph_labels_b)
        sub_train_mask_b = sub_graph_train_mask_b
        sub_test_mask_b = sub_graph_test_mask_b
        # sub_g_b = DGLGraph(nx.from_numpy_matrix(sub_graph_g_b))

        sub_g_b = nx.from_numpy_array(sub_graph_g_b)

        # graph preprocess and calculate normalization factor
        # sub_g_b = nx.from_numpy_array(sub_g_b)
        # add self loop

        sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
        sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))

        sub_g_b = DGLGraph(sub_g_b)
        n_edges = sub_g_b.number_of_edges()
        # normalization
        degs = sub_g_b.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0

        sub_g_b.ndata['norm'] = norm.unsqueeze(1)

        self.net1.eval()
        logits_b = self.net1(sub_g_b, sub_features_b)
        # logits_b = F.log_softmax(logits_b, 1)
        _, query_b = th.max(logits_b, dim=1)

        net2 = Gcn_Net(self.feature_number, self.label_number)
        optimizer_a = th.optim.Adam(
            net2.parameters(), lr=1e-2, weight_decay=5e-4)
        best_performance_metrics = GraphNeuralNetworkMetric()
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
            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net2, sub_g_b, sub_features_b, sub_test_mask_b, sub_labels_b, query_b)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)
            # print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid  {:.4f} | Time(s) {:.4f}".format(
            #     epoch, loss_a.item(), focus_gnn_metrics.accuracy, focus_gnn_metrics.fidelity, np.mean(dur)))

        print(best_performance_metrics)
