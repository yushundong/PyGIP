from gnnip.datasets import *
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


class MyNet(th.nn.Module):
    """
    Input - 1433
    Output - 7
    """

    def __init__(self, in_feats, out_feats):
        super(MyNet, self).__init__()

        self.fc1 = th.nn.Linear(in_feats, 16)
        self.fc2 = th.nn.Linear(16, out_feats)

    def forward(self, x):
        x = th.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
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

        # Train the GCN traget model.
        self.train_target_model()

    def train_target_model(self):
        # Train the GCN target model.
        focus_graph = self.graph
        degs = focus_graph.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        # if cuda != None:
        #     norm = norm.cuda()
        focus_graph.ndata['norm'] = norm.unsqueeze(1)
        self.gcn_Net = Gcn_Net(self.feature_number, self.label_number)
        optimizer = th.optim.Adam(
            self.gcn_Net.parameters(), lr=1e-2, weight_decay=5e-4)
        dur = []

        print("=========Target Model Generating==========================")
        for epoch in range(200):
            if epoch >= 3:
                t0 = time.time()

            self.gcn_Net.train()
            logits = self.gcn_Net(focus_graph, self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask],
                              self.labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            # acc = evaluate(gcn_Net, g, features, labels, test_mask)
            # print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                # epoch, loss.item(), acc, np.mean(dur)))

    def evaluate_helper(self, model, g, features, labels, mask):
        model.eval()
        with th.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = th.max(logits, dim=1)
            correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    def evaluate(self, model, g, features, query_labels, labels, mask):

        return self.evaluate_helper(model, g, features, query_labels, mask), self.evaluate_helper(model, g, features, labels, mask)


class MdoelExtractionAttack0(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, alpha):
        super().__init__(dataset, attack_node_fraction)
        #
        self.alpha = alpha

    def attack(self):
        g = self.graph.clone()
        g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())

        # sample attack_node_number nodes
        sub_graph_node_index = []
        for i in range(self.attack_node_number):
            sub_graph_node_index.append(
                random.randint(0, self.node_number - 1))

        sub_labels = self.labels[sub_graph_node_index]

        # TODO?

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

        # features = th.FloatTensor(data.features)
        # labels = th.LongTensor(data.labels)
        # train_mask = th.ByteTensor(data.train_mask)
        # test_mask = th.ByteTensor(data.test_mask)
        # g = DGLGraph(data.graph)

        self.gcn_Net.eval()

        logits_query = self.gcn_Net(g, self.features)
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
        max_acc1 = 0
        max_acc2 = 0
        for epoch in range(200):
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
            acc1, acc2 = self.evaluate(
                net, g, self.features, labels_query, self.labels, self.test_mask)

            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc2 > max_acc2:
                max_acc2 = acc2
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc2, acc1, np.mean(dur)))

        print("========================Final results:=========================================")
        print("Accuracy:" + str(max_acc2) + "Fedility:" + str(max_acc1))


class MdoelExtractionAttack1(ModelExtractionAttack):

    def __init__(self, dataset, attack_node_fraction, selected_node_file, query_label_file, shadow_graph_file):
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

        max_acc1 = 0
        max_acc2 = 0

        print("===================Model Extracting================================")

        for epoch in range(200):
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

            acc1, acc2 = self.evaluate(
                net, sub_g_b, self.features, self.labels, all_query_labels, self.test_mask)

            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc2 > max_acc2:
                max_acc2 = acc2
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc1, acc2, np.mean(dur)))

        print("Final one:" + str(max_acc1) + "Fiderity: " + str(max_acc2))


class MdoelExtractionAttack2(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction):
        super().__init__(dataset, attack_node_fraction)

    def attack(self):

        # sample nodes
        attack_nodes = []
        for i in range(self.attack_node_number):
            candidate_node = random.randint(0, self.node_number - 1)
            if candidate_node not in attack_nodes:
                attack_nodes.append(candidate_node)

        #

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

        self.gcn_Net.eval()

        # Generate Label
        logits_query = self.gcn_Net(self.graph, self.features)
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

        max_acc1 = 0
        max_acc2 = 0

        for epoch in range(200):
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

            acc1, acc2 = self.evaluate(
                net_attack, self.graph, syn_features, self.labels, labels_query, self.test_mask)
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid  {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc1, acc2, np.mean(dur)))

            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc2 > max_acc2:
                max_acc2 = acc2

        print("Accuracy: " + str(acc1) + " /Fidelity: " + str(acc2))


class ModelExtractionAttack3(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path):
        super().__init__(dataset, attack_node_fraction)

    def attack(self):
        g_numpy = self.graph.adjacency_matrix().to_dense().numpy()
        sub_graph_index_b = []

        # This is to get sub_graph_index b and a
        sub_graph_index_b = []
        fileObject = open('./data/attack3_shadow_graph/' +
                          self.dataset.dataset_name + '/target_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_b.append(int(ip))
        fileObject.close()

        sub_graph_index_a = []
        fileObject = open('./data/attack3_shadow_graph/' + self.dataset.dataset_name +
                          '/protential_1300_shadow_graph_index.txt', 'r')
        contents = fileObject.readlines()
        for ip in contents:
            sub_graph_index_a.append(int(ip))
        fileObject.close()

        # choose attack features in graphA
        attack_node = []
        while len(attack_node) < attack_node_arg * self.node_number:
            protential_node_index = random.randint(
                0, len(sub_graph_index_b) - 1)
            protential_node = sub_graph_index_b[protential_node_index]
            if protential_node not in attack_node:
                attack_node.append(int(protential_node))

        attack_features = features[attack_node]
        attack_labels = labels[attack_node]
        shadow_features = features[sub_graph_index_a]
        shadow_labels = labels[sub_graph_index_a]

        sub_graph_g_A = g_numpy[sub_graph_index_a]
        sub_graph_g_a = sub_graph_g_A[:, sub_graph_index_a]

        sub_graph_attack = g_numpy[attack_node]
        sub_graph_Attack = sub_graph_attack[:, attack_node]
