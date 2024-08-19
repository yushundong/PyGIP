import dgl
import torch
from dgl import DGLGraph 
from dgl.data import citation_graph as citegrh
import numpy as np
from torch.utils.data import Dataset

from torch_geometric.datasets import Planetoid      ### Cora, CiteSeer, PubMed
from ogb.nodeproppred import DglNodePropPredDataset ### ogbn-arxiv
from torch_geometric.datasets import SNAPDataset    ### Facebook
from torch_geometric.datasets import Flickr         ### Flickr
from torch_geometric.datasets import Reddit as TGReddit
from dgl.data import RedditDataset as DGLReddit     ### Reddit
from torch_geometric.datasets import TUDataset as TGTUDataset
from dgl.data import TUDataset as DGLTUDataset      ### MUTAG, PTC, NCI1, NCI109, ENZYMES, 
                                                     ## PROTEINS, COLLAB, IMDB-BINARY
from torch_geometric.datasets import Amazon         
from dgl.data import AmazonCoBuy                    ### Computers, Photo
from torch_geometric.datasets import Yelp
from dgl.data import YelpDataset                    ### Yelp
from torch_geometric.datasets import PPI
from dgl.data import PPIDataset                     ### PROTEINS
from torch_geometric.datasets import LastFM         ### LastFM
from torch_geometric.datasets import BitcoinOTC
from dgl.data import BitcoinOTCDataset              ### Bitcoin_Alpha
# Others: Twitter, Polblogs, Tmall, ML_1M, DP, AIDS, USA, Brazil
from torch_geometric.data import Data

def pyg_to_dgl(data):
    edge_index = data.edge_index.numpy()
    src, dst = edge_index
    graph = dgl.graph((src, dst), num_nodes=data.num_nodes)

    graph.ndata['feat'] = data.x
    graph.ndata['label'] = data.y

    graph.ndata['train_mask'] = data.train_mask
    graph.ndata['val_mask'] = data.val_mask if 'val_mask' in data else torch.zeros(data.num_nodes, dtype=torch.bool)
    graph.ndata['test_mask'] = data.test_mask

    return graph

class Dataset(object):
    def __init__(self, api_type, path):
        self.dataset_name = ""
        self.node_number = 0
        self.feature_number = 0
        self.label_number = 0

        self.features = None
        self.labels = None

        self.train_mask = None
        self.test_mask = None 
        
        self.path = path
        self.api_type = api_type

           

    def load_data(self):
        raise NotImplementedError("load_data not implemented in subclasses.")
    
    def trans_to_dgl(self):
        pass
    

class Cora(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        data = citegrh.load_cora()
        self.graph = data[0]
        self.dataset_name = "cora"

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Planetoid(root=self.path, name='Cora')
        data = dataset[0]
        self.dataset_name = "cora"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, test_mask
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class Citeseer(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        data = citegrh.load_citeseer()
        self.graph = data[0]
        self.dataset_name = "citeseer"

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Planetoid(root=self.path, name='Citeseer')
        data = dataset[0]
        self.dataset_name = "citeseer"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, test_mask
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class PubMed(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        data = citegrh.load_pubmed()
        self.graph = data[0]
        self.dataset_name = "pubmed"

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Planetoid(root=self.path, name='PubMed')
        data = dataset[0]
        self.dataset_name = "pubmed"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, test_mask
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class OGBN(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        self.path = path
        self.api_type = api_type
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        data = DglNodePropPredDataset(name='ogbn-arxiv') 
        graph, labels = data[0]  
        self.graph = graph

        split_idx = data.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        node_number = graph.number_of_nodes()
        feature_number = len(graph.ndata['feat'][0])
        label_number = int(max(labels) - min(labels)) + 1

        features = torch.FloatTensor(graph.ndata['feat'])
        labels = torch.LongTensor(labels.squeeze())  # remove additional dimensions

        train_mask = torch.zeros(node_number, dtype=torch.bool)
        train_mask[train_idx] = True

        valid_mask = torch.zeros(node_number, dtype=torch.bool)
        valid_mask[valid_idx] = True

        test_mask = torch.zeros(node_number, dtype=torch.bool)
        test_mask[test_idx] = True


    def load_tg_data(self):
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]

        self.dataset_name = "ogbn-arxiv"
        self.dataset = dataset
        self.data = data
        self.feature_number = data.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        self.train_mask = self.index_to_mask(train_idx, size=data.num_nodes)
        self.val_mask = self.index_to_mask(valid_idx, size=data.num_nodes)
        self.test_mask = self.index_to_mask(test_idx, size=data.num_nodes)

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class FlickrDataset(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        self.api_type = api_type
        self.path = path
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")


    def load_dgl_data(self):
        dataset = Flickr(self.path)
        data = dataset[0]
        self.dataset_name = "flickr"
        self.graph = pyg_to_dgl(data)

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        # train_mask, val_mask, test_mask
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        dataset = Flickr(self.path)
        data = dataset[0]
        self.dataset_name = "flickr"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class FacebookDataset(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        self.api_type = api_type
        self.path = path

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = SNAPDataset(self.path, name='ego-facebook')
        
        data = dataset[0]
        self.dataset_name = "facebook"
        self.graph = pyg_to_dgl(data)

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        # train_mask, val_mask, test_mask
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        dataset = SNAPDataset(self.path, name='ego-facebook')
        data = dataset[0]

        self.dataset_name = "facebook"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class RedditDataset(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        self.api_type = api_type
        self.path = path

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = DGLReddit(raw_dir=self.path)
        self.graph = dataset[0]

        self.dataset_name = "reddit"

        # node_number, feature_number, label_number
        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        # train_mask, val_mask, test_mask
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        dataset = TGReddit(self.path)
        data = dataset[0]
        self.dataset_name = "reddit"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class TU(Dataset):
    def __init__(self, api_type, path, dataset_name):
        super().__init__()
        self.api_type = api_type
        self.path = path
        self.dataset_name = dataset_name.lower()

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = DGLTUDataset(name=self.dataset_name, raw_dir=self.path)
        data = dataset[0]
        self.graph = data

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, test_mask (val_mask not available in DGL TUDataset)
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = TGTUDataset(root=self.path, name=self.dataset_name)
        data = dataset[0]
        self.dataset_name = self.dataset_name

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, test_mask, val_mask
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.val_mask = data.val_mask

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class AmazonDataset(Dataset):
    def __init__(self, api_type, path, dataset_name):
        super().__init__()
        self.api_type = api_type
        self.path = path
        self.dataset_name = dataset_name.lower()

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = AmazonCoBuy(self.path, self.dataset_name.capitalize())
        data = dataset[0]
        self.graph = data

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, val_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.val_mask = torch.BoolTensor(self.graph.ndata.get('val_mask', torch.zeros(self.node_number, dtype=torch.bool)))
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Amazon(self.path, self.dataset_name.capitalize())
        data = dataset[0]
        self.dataset_name = self.dataset_name

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class YelpDataset(Dataset):
    def __init__(self, api_type, path):
        super().__init__()
        self.api_type = api_type
        self.path = path

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = YelpDataset(self.path)
        data = dataset[0]
        self.graph = data

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, val_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.val_mask = torch.BoolTensor(self.graph.ndata.get('val_mask', torch.zeros(self.node_number, dtype=torch.bool)))
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Yelp(self.path)
        data = dataset[0]

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class PPIDataset(Dataset):
    def __init__(self, api_type, path):
        super().__init__()
        self.api_type = api_type
        self.path = path

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = PPIDataset(self.path)
        data = dataset[0]
        self.graph = data

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, val_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.val_mask = torch.BoolTensor(self.graph.ndata.get('val_mask', torch.zeros(self.node_number, dtype=torch.bool)))
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = PPI(self.path)
        data = dataset[0]

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class LastFMDataset(Dataset):
    def __init__(self, api_type, path):
        super().__init__()
        self.api_type = api_type
        self.path = path

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = LastFM(self.path)
        data = dataset[0]
        self.graph = self.graph = pyg_to_dgl(data)

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, val_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.val_mask = torch.BoolTensor(self.graph.ndata.get('val_mask', torch.zeros(self.node_number, dtype=torch.bool)))
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = LastFM(self.path)
        data = dataset[0]

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index




class BitcoinOTCDataset(Dataset):
    def __init__(self, api_type, path, dataset_name):
        super().__init__()
        self.api_type = api_type
        self.path = path
        self.dataset_name = dataset_name.lower()

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = BitcoinOTCDataset(self.path, 'alpha')
        data = dataset[0]
        self.graph = data

        # node_number, feature_number, label_number
        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        # features, labels
        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        # train_mask, val_mask, test_mask
        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.val_mask = torch.BoolTensor(self.graph.ndata.get('val_mask', torch.zeros(self.node_number, dtype=torch.bool)))
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = BitcoinOTC(self.path, 'alpha')
        data = dataset[0]

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        # features, labels
        self.features = data.x
        self.labels = data.y

        # train_mask, val_mask, test_mask
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

        self.node_number = data.num_nodes
        self.edge_index = data.edge_index