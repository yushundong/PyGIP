import torch
import numpy as np

def add_edge(edge_index, i, j):
    edge_index = torch.cat([edge_index, torch.tensor([[i, j], [j, i]])], dim=1)
    return edge_index

def generate_key_input(base_graph, n_random_nodes=5):
    key_graph = base_graph.clone()
    n_nodes = key_graph.num_nodes

    random_nodes = np.random.choice(n_nodes, n_random_nodes, replace=False)
    
    for i in random_nodes:
        for j in random_nodes:
            if i != j and np.random.rand() > 0.5:
                key_graph.edge_index = add_edge(key_graph.edge_index, i, j)
    
    key_graph.x[random_nodes] = torch.rand((n_random_nodes, key_graph.x.shape[1]))
    return key_graph
