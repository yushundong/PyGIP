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

            acc1 = evaluate(net, g, self.features,
                            labels_query, self.test_mask)
            acc2 = evaluate(net, g, self.features, self.labels, self.test_mask)
            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc2 > max_acc2:
                max_acc2 = acc2
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc2, acc1, np.mean(dur)))

        print("========================Final results:=========================================")
        print("Accuracy:" + str(max_acc2) + "Fedility:" + str(max_acc1))
