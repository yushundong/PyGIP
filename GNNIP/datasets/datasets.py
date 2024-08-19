from dgl.data import citation_graph as citegrh
import torch as th


class Dataset(object):
    def __init__(self):
        self.node_number = 0
        self.feature_number = 0
        self.label_number = 0

        #
        self.features = None
        self.labels = None

        #
        self.train_mask = None
        self.test_mask = None
        self.path_name = ""


class Cora(Dataset):
    def __init__(self):
        super().__init__()
        data = citegrh.load_cora()

        # graph, dataset_name
        self.graph = data[0]
        self.dataset_name = "cora"

        # node_number, feature_numbe, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(
            self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        # features, labels
        self.features = th.FloatTensor(self.graph.ndata['feat'])
        self.labels = th.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = th.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = th.BoolTensor(self.graph.ndata['test_mask'])


class Citeseer(Dataset):
    def __init__(self):
        super().__init__()
        data = citegrh.load_citeseer()

        # graph
        self.graph = data[0]
        self.dataset_name = "citeseer"

        # node_number, feature_numbe, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(
            self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        # features, labels
        self.features = th.FloatTensor(self.graph.ndata['feat'])
        self.labels = th.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = th.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = th.BoolTensor(self.graph.ndata['test_mask'])


class PubMed(Dataset):
    def __init__(self):
        super().__init__()
        data = citegrh.load_pubmed()

        # graph
        self.graph = data[0]
        self.dataset_name = 'pubmed'

        # node_number, feature_numbe, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(
            self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        # features, labels
        self.features = th.FloatTensor(self.graph.ndata['feat'])
        self.labels = th.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = th.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = th.BoolTensor(self.graph.ndata['test_mask'])
