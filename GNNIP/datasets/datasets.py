from dgl.data import citation_graph as citegrh
import torch as th


class Dataset(object):
    def __init__(self):
        self.features_ = None
        self.labels_ = None
        self.train_mask = None
        self.test_mask = None
        self.path_name = ""
    

class Cora(Dataset):
    def __init__(self):
        data = citegrh.load_cora()
        g = data[0]
        self.features = th.FloatTensor(g.ndata['feat'])
        self.labels = th.LongTensor(g.ndata['label'])
        self.train_mask = th.BoolTensor(g.ndata['train_mask'])
        self.test_mask = th.BoolTensor(g.ndata['test_mask'])


class Citesser(Dataset):
    def __init__(self):
        data = citegrh.load_citeseer()
        g = data[0]
        self.features = th.FloatTensor(g.ndata['feat'])
        self.labels = th.LongTensor(g.ndata['label'])
        self.train_mask = th.BoolTensor(g.ndata['train_mask'])
        self.test_mask = th.BoolTensor(g.ndata['test_mask'])

class PubMed(Dataset):
    def __init__(self):
        data = citegrh.load_cora()
        g = data[0]
        self.features = th.FloatTensor(g.ndata['feat'])
        self.labels = th.LongTensor(g.ndata['label'])
        self.train_mask = th.BoolTensor(g.ndata['train_mask'])
        self.test_mask = th.BoolTensor(g.ndata['test_mask'])

