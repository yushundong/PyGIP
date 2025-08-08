import networkx as nx
import dgl

def get_jaccard(nx_g, pairs):
    jaccard_dict = {}
    jaccard_tuple = nx.jaccard_coefficient(nx_g, pairs)
    for u, v, p in jaccard_tuple:
        jaccard_dict[(u, v)] = round(p,3)
    return jaccard_dict


def get_attach(nx_g, pairs):
    attach_dict = {}
    jaccard_tuple = nx.preferential_attachment(nx_g, pairs)
    for u, v, p in jaccard_tuple:
        attach_dict[(u, v)] = round(p,3)
    return attach_dict

def get_common_neighbors(nx_g, pairs):
    neihbors_dict = {}
    for start_id, end_id in pairs:
        neihbors_dict[(start_id, end_id)] = len(list(nx.common_neighbors(nx_g, start_id, end_id)))
    return neihbors_dict


def get_features(args, g, pairs, label, mode):
    k = 1
    
    jaccard_dict = {}
    attach_dict = {}
    neihbors_dict = {} 
    nx_g = nx.Graph(dgl.to_networkx(g, node_attrs=["features"]))
    for pair in pairs:
        start_subgraph_nodes = list(nx.ego.ego_graph(nx_g, n=pair[0], radius=k).nodes())
        end_subgraph_nodes = list(nx.ego.ego_graph(nx_g, n=pair[1], radius=k).nodes())
        subgraph_nodes = start_subgraph_nodes + end_subgraph_nodes
        subgraph = nx_g.subgraph(subgraph_nodes).copy()
        start_id = pair[0]
        end_id = pair[1]
        if label == 1:
            subgraph.remove_edge(start_id, end_id)
        jaccard_tuple = nx.jaccard_coefficient(subgraph, [(start_id, end_id)])
        for _, _, p in jaccard_tuple:
            jaccard_dict[pair] = round(p,3)
        attach_tuple = nx.preferential_attachment(subgraph, [(start_id, end_id)])
        for _, _, p in attach_tuple:
            attach_dict[pair] = round(p,3)
        neihbors_dict[pair] = len(list(nx.common_neighbors(subgraph, start_id, end_id)))
    print("Finish Generating trad_feature_dict...")
    return jaccard_dict, attach_dict, neihbors_dict

