import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import dgl
from dgl import DGLGraph 
from torch_geometric.data import Data as PyGData

from dgl.data import citation_graph as citegrh
from torch_geometric.datasets import Planetoid          ### Cora, CiteSeer, PubMed
from torch_geometric.datasets import DBLP               ### DBLP
from dgl.data import WikiCSDataset                      ### WikiCS                                         
from dgl.data import YelpDataset     
from torch_geometric.datasets import Yelp               ### YelpData
from torch_geometric.datasets import FacebookPagePage   ### Facebook
from dgl.data import FlickrDataset
from torch_geometric.datasets import Flickr             ### FlickrData
from torch_geometric.datasets import PolBlogs           ### Polblogs 
from torch_geometric.datasets import LastFMAsia         ### LastFM
from dgl.data import RedditDataset
from torch_geometric.datasets import Reddit             ### RedditData       
from dgl.data import AmazonCoBuyComputerDataset         ### Computer
from dgl.data import AmazonCoBuyPhotoDataset            ### Photo
from dgl.data import MUTAGDataset                       ### MUTAGData
from dgl.data import GINDataset                         ### Collab, NCI1, PROTEINS, PTC, IMDB-BINARY
from dgl.data import FakeNewsDataset                    ### Twitter  
                                                   

def dgl_to_tg(dgl_graph):
    edge_index = torch.stack(dgl_graph.edges())
    x = dgl_graph.ndata.get('feat')
    y = dgl_graph.ndata.get('label')
    
    train_mask = dgl_graph.ndata.get('train_mask')
    val_mask = dgl_graph.ndata.get('val_mask')
    test_mask = dgl_graph.ndata.get('test_mask')

    data = PyGData(x=x, edge_index=edge_index, y=y, 
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

def tg_to_dgl(py_g_data):
    edge_index = py_g_data.edge_index
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]))

    if py_g_data.x is not None:
        dgl_graph.ndata['feat'] = py_g_data.x
    if py_g_data.y is not None:
        dgl_graph.ndata['label'] = py_g_data.y

    if hasattr(py_g_data, 'train_mask') and py_g_data.train_mask is not None:
        dgl_graph.ndata['train_mask'] = py_g_data.train_mask
    if hasattr(py_g_data, 'val_mask') and py_g_data.val_mask is not None:
        dgl_graph.ndata['val_mask'] = py_g_data.val_mask
    if hasattr(py_g_data, 'test_mask') and py_g_data.test_mask is not None:
        dgl_graph.ndata['test_mask'] = py_g_data.test_mask

    return dgl_graph


class Dataset(object):
    def __init__(self, api_type='dgl', path='./downloads/'):
        self.api_type = api_type
        self.path = path
        self.dataset_name = ""

        self.node_number = 0
        self.feature_number = 0
        self.label_number = 0

        self.features = None
        self.labels = None
        
        #self.train_ratio = 0.8
        self.train_mask = None
        #self.var_mask = None
        self.test_mask = None 

    def load_dgl_data(self):
        raise NotImplementedError("load_dgl_data not implemented in subclasses.")
    
    def load_tg_data(self):
        raise NotImplementedError("load_dgl_data not implemented in subclasses.")
    
    def generate_train_test_masks(self):
        num_nodes = self.node_number
        indices = torch.randperm(num_nodes)
        train_size = int(self.train_ratio * num_nodes)
        val_size = (num_nodes - train_size) // 2

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        #print("Train mask sum:", train_mask.sum().item()) 
        #print("Val mask sum:", val_mask.sum().item())
        #print("Test mask sum:", test_mask.sum().item())  
    



class Cora(Dataset):
    def __init__(self, api_type='dgl', path='./'):
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

        # train_mask, test_mask, var_mask
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

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Planetoid(root=self.path, name='Citeseer')
        data = dataset[0]
        self.dataset_name = "citeseer"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class DBLPdata(Dataset):
    def __init__(self, api_type='dgl', path='./'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = DBLP(self.path)
        dblp_data = dataset[0]

        edge_index = dblp_data['author', 'to', 'paper']._mapping['edge_index'].numpy()
        num_nodes = max(edge_index[0].max(), edge_index[1].max()) + 1

        self.graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        
        author_node_indices = np.unique(edge_index[0])
        author_subgraph = dgl.node_subgraph(self.graph, author_node_indices)
        author_subgraph.ndata['feat'] = dblp_data['author'].x  # 节点特征
        author_subgraph.ndata['label'] = dblp_data['author'].y  # 节点标签
        
        self.train_mask = dblp_data['author'].train_mask
        self.test_mask = dblp_data['author'].test_mask

        self.graph = author_subgraph
        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']


    def load_tg_data(self):
        dataset = DBLP(root=self.path)
        data = dataset[0]
        self.dataset_name = "dblp"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        self.features = data.x
        self.labels = data.y

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

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        dataset = Planetoid(root=self.path, name='PubMed')
        data = dataset[0]
        self.dataset_name = "pubmed"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
      
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class WikiCS(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = WikiCSDataset(raw_dir=self.path)
        self.graph = dgl.add_self_loop(dataset[0])
        self.dataset_name = "wikics"

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        #print("Label shape:", self.labels.shape)
        #print("Example labels:", max(self.labels))

        train_mask = self.graph.ndata['train_mask']
        self.train_mask = train_mask[:, 1].bool() # originally cross-validation
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']
        #print(True in train_mask[:, 1].bool())
        #print("train shape:", self.train_mask.shape)
        #print("train_mask:", self.train_mask)

####################################################################################################
class FacebookData(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")  

    def load_dgl_data(self):
        dataset = FacebookPagePage(root=self.path)
        data = dataset[0]
        self.train_ratio = 0.8
        self.dataset_name = "facebook"

        edge_index = data.edge_index.numpy()
        self.graph = dgl.graph((edge_index[0], edge_index[1]))
        #print(f"Graph: {self.graph}")
        #print(f"Graph nodes: {self.graph.num_nodes()}")
        #print(f"Graph edges: {self.graph.num_edges()}")

        self.graph.ndata['feat'] = data.x  
        self.graph.ndata['label'] = data.y 

        self.train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        self.test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        num_train = int(self.graph.num_nodes() * self.train_ratio)
        perm = torch.randperm(self.graph.num_nodes())
        self.train_mask[perm[:num_train]] = True
        self.test_mask[perm[num_train:]] = True
        #print(f"Node feature shape: {self.graph.ndata['feat'].shape}")
        #print(f"Node label shape: {self.graph.ndata['label'].shape}")
        #print(f"Train mask shape: {self.train_mask.shape}")
        #print(f"Test mask shape: {self.test_mask.shape}")

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]  
        self.label_number = len(torch.unique(self.graph.ndata['label'])) 
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']


    def load_tg_data(self):
        dataset = FacebookPagePage(root=self.path)
        data = dataset[0]
        self.dataset_name = "facebook"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class FlickrData(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = FlickrDataset(self.path)
        self.graph = dataset[0]
        self.dataset_name = "flickr"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

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

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class PolblogsData(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")  

    def load_dgl_data(self):
        dataset = PolBlogs(root=self.path)
        data = dataset[0]
        self.train_ratio = 0.8
        self.dataset_name = "polblogs"

        edge_index = data.edge_index.numpy()
        self.graph = dgl.graph((edge_index[0], edge_index[1]))
        #print(f"Graph: {self.graph}")
        #print(f"Graph nodes: {self.graph.num_nodes()}")
        #print(f"Graph edges: {self.graph.num_edges()}")

        self.graph.ndata['feat'] = data.x  
        self.graph.ndata['label'] = data.y 

        self.train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        self.test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        num_train = int(self.graph.num_nodes() * self.train_ratio)
        perm = torch.randperm(self.graph.num_nodes())
        self.train_mask[perm[:num_train]] = True
        self.test_mask[perm[num_train:]] = True
        #print(f"Node feature shape: {self.graph.ndata['feat'].shape}")
        #print(f"Node label shape: {self.graph.ndata['label'].shape}")
        #print(f"Train mask shape: {self.train_mask.shape}")
        #print(f"Test mask shape: {self.test_mask.shape}")

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]  
        self.label_number = len(torch.unique(self.graph.ndata['label'])) 
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']


    def load_tg_data(self):
        dataset = PolBlogs(root=self.path)
        data = dataset[0]
        self.dataset_name = "polblogs"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class LastFMdata(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")  

    def load_dgl_data(self):
        dataset = LastFMAsia(root=self.path)
        data = dataset[0]
        self.train_ratio = 0.8
        self.dataset_name = "last-fm"

        edge_index = data.edge_index.numpy()
        self.graph = dgl.graph((edge_index[0], edge_index[1]))
        #print(f"Graph: {self.graph}")
        #print(f"Graph nodes: {self.graph.num_nodes()}")
        #print(f"Graph edges: {self.graph.num_edges()}")

        self.graph.ndata['feat'] = data.x  
        self.graph.ndata['label'] = data.y 

        self.train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        self.test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        num_train = int(self.graph.num_nodes() * self.train_ratio)
        perm = torch.randperm(self.graph.num_nodes())
        self.train_mask[perm[:num_train]] = True
        self.test_mask[perm[num_train:]] = True
        #print(f"Node feature shape: {self.graph.ndata['feat'].shape}")
        #print(f"Node label shape: {self.graph.ndata['label'].shape}")
        #print(f"Train mask shape: {self.train_mask.shape}")
        #print(f"Test mask shape: {self.test_mask.shape}")

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]  
        self.label_number = len(torch.unique(self.graph.ndata['label'])) 
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']


    def load_tg_data(self):
        dataset = LastFMAsia(root=self.path)
        data = dataset[0]
        self.dataset_name = "last-fm"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes # originally num_classes

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index

class RedditData(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = RedditDataset(raw_dir=self.path)
        self.graph = dataset[0]
        self.dataset_name = "reddit"

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        #print("Label shape:", self.labels.shape)
        #print("Example labels:", max(self.labels))

        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']
        #print("train shape:", self.train_mask.shape)
        #print("train_mask:", self.train_mask)

    def load_tg_data(self):
        dataset = Reddit(self.path)
        data = dataset[0]
        self.dataset_name = "reddit"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class Twitter(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")
        
    def load_dgl_data(self):
        dataset = FakeNewsDataset('gossipcop', 'bert', raw_dir=self.path)
        graph, _ = dataset[0]
        self.graph = dgl.add_self_loop(graph)
        self.dataset_name = "twitter"
        self.train_ratio = 0.8

        if hasattr(dataset, 'feature'):
            node_ids = self.graph.ndata['_ID'].numpy()  
            selected_features = dataset.feature[node_ids]
            self.graph.ndata['feat'] = selected_features.float() 

        if hasattr(dataset, 'labels'):
            selected_labels = dataset.labels[node_ids]
            self.graph.ndata['label'] = selected_labels.long() 

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        #print("Label shape:", self.labels.shape)
        #print("Example labels:", max(self.labels))

        self.generate_train_test_masks()

####################################################################################################

class MutaData(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")


    def load_dgl_data(self):
        dataset = MUTAGDataset(raw_dir=self.path)
        self.graph = dataset[0]
        self.dataset_name = "mutag"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.dataset.predict_category

        self.train_mask = self.graph.nodes[category].data['train_mask']
        self.val_mask = self.graph.nodes[category].data['train_mask']
        self.test_mask = self.graph.nodes[category].data['train_mask']


class PTC(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = GINDataset(name='PTC', raw_dir=self.path, self_loop=False)
        graph, _ =  zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "ptc"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


class NCI1(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = GINDataset(name='NCI1', raw_dir=self.path, self_loop=False)
        graph, _ =  zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "nci1"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()

####################################################################################################

class PROTEINS(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = GINDataset(name='PROTEINS', raw_dir=self.path, self_loop=False)
        graph, _ =  zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "proteins"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()

####################################################################################################

class Collab(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = GINDataset(name='COLLAB', raw_dir=self.path, self_loop=False)
        graph, _ =  zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "collab"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']
        #print("Label shape:", self.labels.shape)
        #print("Example labels:", max(self.labels))
        #print(self.graph.ndata.keys())

        self.generate_train_test_masks()
        #print("Number of selected nodes:", self.train_mask.sum().item())
        #print("Node features shape:", self.features.shape)
        #print("Node labels shape:", self.labels.shape)
        #print("Node features example:", self.features[0])
        #print("Node labels example:", self.labels[0])

class IMDB(Dataset):
    def __init__(self, api_type, path):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = GINDataset(name='IMDB-BINARY', raw_dir=self.path, self_loop=False)
        graph, _ =  zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "imdb"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()

####################################################################################################

class Computer(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        data = AmazonCoBuyComputerDataset(raw_dir=self.path)
        self.graph = dgl.add_self_loop(data[0])
        self.dataset_name = "computer"
        self.train_ratio = 0.8

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])
        #print("Label shape:", self.labels.shape)
        #print("Example labels:", self.labels[3])

        self.generate_train_test_masks()


class Photo(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        data = AmazonCoBuyPhotoDataset(raw_dir=self.path)
        self.graph = dgl.add_self_loop(data[0])
        self.dataset_name = "photo"
        self.train_ratio = 0.8

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])
        #print("Label shape:", self.labels.shape)
        #print("Example labels:", self.labels[3])

        self.generate_train_test_masks()
    

class YelpData(Dataset):
    def __init__(self, api_type='dgl', path='./downloads/'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = dgl.data.YelpDataset(raw_dir=self.path)
        self.graph = dataset[0]
        self.dataset_name = "yelp"

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        dataset = Yelp(root=self.path)
        self.data = dataset[0]
        self.dataset_name = "yelp"

        self.node_number = self.data.num_nodes
        self.feature_number = self.data.x.shape[1]
        self.label_number = len(torch.unique(self.data.y))

        self.features = self.data.x
        self.labels = self.data.y

        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask

####################################################################################################





















