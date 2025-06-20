import torch as th
th.manual_seed(0)
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

 
    def inference(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

    def extract_embedding(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
            x = y
            # return the embedding after the first layer;
            break
        return y



class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'gcn'))
        # for i in range(1, n_layers - 1):
        #     self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'gcn'))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'gcn'))
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

    def extract_embedding(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
            x = y
            # return the embedding after the first layer;
            break
        return y



class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout,num_heads=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads, allow_zero_in_degree=True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(n_hidden*num_heads, n_hidden, num_heads, allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(n_hidden*num_heads, n_classes, 1, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h).flatten(1)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        logits = h
        return logits

    def inference(self, g, x, device):
      
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden*self.num_heads if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h).flatten(1)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

    def extract_embedding(self, g, x,  device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden*self.num_heads if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h).flatten(1)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
            x = y
            # return the embedding after the first layer;
            break
        return y

class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = 2
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        linear = nn.Linear(in_feats, n_hidden)
        self.layers.append(dglnn.GINConv(linear, 'sum'))
        for i in range(1, n_layers-1):
            linear = nn.Linear(n_hidden, n_hidden)
            self.layers.append(dglnn.GINConv(linear, 'sum'))
        linear = nn.Linear(n_hidden, n_classes)
        self.layers.append(dglnn.GINConv(linear, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y
        
    def extract_embedding(self, g, x,  device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
            x = y
            # return the embedding after the first layer;
            break
        return y