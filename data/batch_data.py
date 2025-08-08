import dgl
import networkx as nx
from utils.query_model import query_trained_model
from data.generate_xy import generate_attack_xy_node,generate_attack_xy_graph,generate_attack_xy_node_graph, generate_attack_xy, generate_attack_xy_plus, generate_attack_xy_all, generate_attack_xy_plus2


def get_batch(args, batch_pairs, g, k, i, label, mode):
   
    query_graph_batch, index_mapping_dict_batch = get_khop_query_graph_batch(g, batch_pairs, k=k)
    index_update_batch = [node for _, nodes in index_mapping_dict_batch.items() for node in nodes]
    posteriors_dict_batch = query_trained_model(args, index_update_batch, query_graph_batch, mode)

    print('Finish generating posteriors and mapping dict...')
    return posteriors_dict_batch, index_mapping_dict_batch


def get_batch_posteriors(args, batch_pairs, g, k, i, label, mode):
    posteriors_dict_batch, index_mapping_dict_batch = get_batch(args, batch_pairs, g, k, i, label, mode)
    batch_features, batch_labels, batch_stat_dict = generate_attack_xy(args, batch_pairs, posteriors_dict_batch, label, index_mapping_dict_batch)  
    return batch_features, batch_labels, batch_stat_dict


def get_batch_posteriors_node(args, batch_pairs, g, k, i, label, mode):
    posteriors_dict_batch, index_mapping_dict_batch = get_batch(args, batch_pairs, g, k, i, label, mode)
    batch_node_features, batch_posteriors_features, batch_labels, batch_stat_dict = generate_attack_xy_plus(args, g, batch_pairs, posteriors_dict_batch, label, index_mapping_dict_batch)  
    return batch_node_features, batch_posteriors_features, batch_labels, batch_stat_dict


def get_batch_posteriors_graph(args, batch_pairs, g, k, i, label, mode):
    posteriors_dict_batch, index_mapping_dict_batch = get_batch(args, batch_pairs, g, k, i, label, mode)
    batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict = generate_attack_xy_plus2(args, g, batch_pairs, posteriors_dict_batch, label, mode, index_mapping_dict_batch)  
    return batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict


def get_batch_posteriors_node_graph(args, batch_pairs, g, k, i, label, mode):
    posteriors_dict_batch, index_mapping_dict_batch = get_batch(args, batch_pairs, g, k, i, label, mode)
    batch_node_features, batch_posteriors_features, batch_graph_features,  batch_labels, batch_stat_dict = generate_attack_xy_all(args, g, batch_pairs, posteriors_dict_batch, label, mode, index_mapping_dict_batch)  
    return batch_node_features, batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict

def get_batch_node_only(args, batch_pairs, g,  i, label, mode):
    
    batch_node_features, batch_labels, batch_stat_dict = generate_attack_xy_node(args, g, batch_pairs, label)
    return batch_node_features, batch_labels, batch_stat_dict

def get_batch_graph_only(args, batch_pairs, g,  i, label, mode):
    # Skip posteriors_dict and mapping: Not needed for graph-only
    batch_graph_features, batch_labels, batch_stat_dict = generate_attack_xy_graph(args, g, batch_pairs, label, mode)
    return batch_graph_features, batch_labels, batch_stat_dict

def get_batch_node_graph_only(args, batch_pairs, g,  i, label, mode):
    # Skip posteriors_dict and mapping: Not needed for node+graph only
    batch_node_features, batch_graph_features, batch_labels, batch_stat_dict = generate_attack_xy_node_graph(args, g, batch_pairs, label, mode)
    return batch_node_features, batch_graph_features, batch_labels, batch_stat_dict



def get_khop_query_graph_batch(g, pairs, k=2):

    nx_g = dgl.to_networkx(g, node_attrs=["features"]) 

    subgraph_list = []
    index_mapping_dict = {}
    bias = 0
    
    for pair in pairs:
        start_node = pair[0]
        end_node = pair[1]
        nx_g.remove_edges_from([(start_node, end_node), (end_node, start_node)])
        node_index = []
        for node in (start_node, end_node):
            node_neighbor = list(nx.ego.ego_graph(nx_g, n=node, radius=k).nodes())
            node_neighbor_num = len(node_neighbor)
            node_new_index = node_neighbor.index(node)
            subgraph_k_hop = g.subgraph(node_neighbor)        
            subgraph_list.append(subgraph_k_hop)
            node_index.append(node_new_index + bias)
            bias += node_neighbor_num
        nx_g.add_edges_from([(start_node, end_node), (end_node, start_node)])
        index_mapping_dict[(start_node, end_node)] = (node_index[0], node_index[1])
    update_query_graph = dgl.batch([row for row in subgraph_list])
    
    print("Get k-hop query graph")
    return update_query_graph, index_mapping_dict