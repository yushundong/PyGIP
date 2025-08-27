import torch
import torch.nn.functional as F
import dgl
import numpy as np
from dgl.nn.pytorch import GraphConv
from .defense_base import BaseDefense

# A standard, simple Graph Convolutional Network (GCN) model with two layers.
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# we implement the graph pruning defense ferom this class
class GraphPruningDefense(BaseDefense):
    def __init__(self, dataset, pruning_ratio=0.1, pruning_strategy='random', **kwargs):  #initialize the defense with key params
        super().__init__(dataset, **kwargs)
        self.pruning_ratio = pruning_ratio
        self.pruning_strategy = pruning_strategy # Stores the pruning strategy as 'random', 'degree_low', 'degree_high'

    def _prune_graph(self, original_graph):
        """Prunes the graph based on the selected strategy."""
        num_edges_to_remove = int(original_graph.number_of_edges() * self.pruning_ratio) #calculate the number of edges to remove
        if num_edges_to_remove == 0:
            return original_graph

        if self.pruning_strategy == 'random':
            print(f"(Strategy: Random)...", end="")
            all_edge_ids = np.arange(original_graph.number_of_edges())
            edges_to_remove_ids = np.random.choice(all_edge_ids, num_edges_to_remove, replace=False)  # Randomly choose a subset of edge IDs to remove.

        elif self.pruning_strategy in ['degree_low', 'degree_high']:
            print(f"(Strategy: {self.pruning_strategy})...", end="")
            degrees = original_graph.in_degrees().float() # # Get the number of connections for every node.
            u, v = original_graph.edges()
            edge_scores = degrees[u] + degrees[v]
            
            is_largest = self.pruning_strategy == 'degree_high'  # Determine whether to remove edges with the highest or lowest scores.
            _, edges_to_remove_ids = torch.topk(edge_scores, num_edges_to_remove, largest=is_largest)
            
        else:
            raise ValueError(f"Unknown pruning strategy: {self.pruning_strategy}")
            
        return dgl.remove_edges(original_graph, edges_to_remove_ids)

    def _train_gcn_model(self, graph, epochs=100, lr=0.01):  # handles training the GCN model
        """Private helper to train a SimpleGCN model on a given graph."""
        graph = graph.to(self.device)
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        test_mask = graph.ndata['test_mask']
        
        model = SimpleGCN(self.num_features, 16, self.num_classes).to(self.device)  # initialize the GCN model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            logits = model(graph, features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])   # calculate the loss on the training set
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            pred = logits.argmax(1) #  Get the final predictions by choosing the class with the highest score.
            accuracy = (pred[test_mask] == labels[test_mask]).float().mean()  # calculate the accuracy on the test set
        return accuracy.item()

    def _train_defense_model(self):  # it is for the pruning defense
        """Prunes the graph using the chosen strategy and then trains a model."""
        print(f"   - Training defense model (Ratio: {self.pruning_ratio*100:.1f}%) ", end="")
        
        pruned_graph = self._prune_graph(self.graph_data)  # first we prune the graph using the chosen strategy
        pruned_graph = dgl.add_self_loop(pruned_graph) #  then we add self-loops to prevent disconnected nodes
        
        accuracy = self._train_gcn_model(pruned_graph) # then we train a GCN model on the pruned graph
        print(f" Test Accuracy: {accuracy*100:.2f}%")
        return accuracy

    def defend(self): # this is the main public method
        """Main public method to execute the defense evaluation."""
        defended_acc = self._train_defense_model()  # it executes the pruning defense
        
        # it returns a dictionary with the defended accuracy and the pruning ratio
        return {
            'defended_accuracy': defended_acc,
            'pruning_ratio': self.pruning_ratio
        }