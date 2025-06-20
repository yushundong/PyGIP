import numpy as np
import torch as th
import dgl
from tqdm import tqdm
import networkx as nx

from graphgallery.datasets import NPZDataset, KarateClub, Reddit

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def load_graphgallery_data(dataset):
    
    if dataset in ['deezer', 'lastfm']:
        data = KarateClub(dataset)
    elif dataset in ['reddit']:
        data = Reddit(dataset)
    else:
        data = NPZDataset(dataset, verbose=False)
    graph = data.graph
    nx_g = nx.from_scipy_sparse_array(graph.adj_matrix)
    for node_id, node_data in nx_g.nodes(data=True):
        node_data["features"] = graph.feat[node_id].astype(np.float32)
        if dataset in ['blogcatalog', 'flickr']:
            node_data["labels"] = graph.y[node_id].astype(np.longlong) - 1
        else:
            node_data["labels"] = graph.y[node_id].astype(np.longlong)
    dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    dgl_graph = dgl.to_simple(dgl_graph, copy_ndata=True)
    dgl_graph = dgl.to_bidirected(dgl_graph, copy_ndata=True)
    
    print(nx.density(dgl_graph.to_networkx()))
    print("Classes:%d" % (graph.num_classes))
    print("Feature dim: %d" % (dgl_graph.ndata['features'].shape[1]))
    print(f"Graph has {dgl_graph.number_of_nodes()} nodes, {dgl_graph.number_of_edges()} edges.")
    
    return dgl_graph, graph.num_classes


def node_sample(g, prop=0.5):
    '''
    sample target/shadow graph (1:1) 
    '''
    node_number = len(g.nodes())
    node_index_list = np.arange(node_number) 
    np.random.shuffle(node_index_list)
    split_length = int(node_number * prop)

    train_index = np.sort(node_index_list[:split_length])
    test_index = np.sort(node_index_list[split_length: ])

    return train_index, test_index


def remove_neighbor_edge_by_prop(g, prop=0.2):
    """
    Remove all edges from a graph, only save self connection 
    """
    real_pairs = []
    test_pairs = []
    start_ids, end_ids = g.edges()
    
    for i in tqdm(range(len(start_ids))):
        if start_ids[i] < end_ids[i]:
            real_pairs.append([i, start_ids[i].item(), end_ids[i].item()])
    
    delete_edge_num = int(len(real_pairs) * prop)
    print("Real Pairs Number (no self-loop & reverse edge): %d" % (len(real_pairs)))
    print("Delete real pairs number (no self-loop & reverse edge): %d" % (delete_edge_num))
    np.random
    delete_ids_1d = np.random.choice(len(real_pairs), delete_edge_num, replace=False)
    delete_eids = []
    for i in delete_ids_1d:
        eid, start_id, end_id = real_pairs[i]
        eid_2 = g.edge_ids(end_id, start_id)
        delete_eids += [eid, eid_2]
        test_pairs.append((start_id, end_id))

    print("All edge numbers: %d" % (len(start_ids)))
    g = dgl.remove_edges(g, th.tensor(delete_eids))
    
    print("Delete %d edges" % (len(delete_eids)))
    print("Lefted %d edges" % (len(g.edges()[0])))
    return g, test_pairs


def split_target_shadow(g):

    target_index, shadow_index = node_sample(g, 0.5)

    target_g = g.subgraph(target_index)
    shadow_g = g.subgraph(shadow_index)

    return target_g, shadow_g


def split_target_shadow_by_prop(args, g):

    target_index, shadow_index = node_sample(g, 0.5)

    target_g = g.subgraph(target_index)
    shadow_g = g.subgraph(shadow_index)
    shadow_index_prop, _ = node_sample(shadow_g, args.prop*0.01)
    shadow_g = shadow_g.subgraph(shadow_index_prop)

    return target_g, shadow_g


def split_train_test(g):

    train_index, test_index = node_sample(g, 0.8)

    train_g = g.subgraph(train_index)
    test_g = g.subgraph(test_index)

    return train_g, test_g


