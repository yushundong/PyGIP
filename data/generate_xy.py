from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from scipy.spatial import distance

from utils.tradition_metrics import get_features
np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def entropy(P):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    P = P + epsilon
    entropy_value = -np.sum(P * np.log(P))
    return np.array([entropy_value])


def js_divergence(a, b):
    return distance.jensenshannon(a, b, 2.0)


def cosine_sim(a, b):
    return 1 - distance.cosine(a, b)


def correlation_dist(a, b):
    return distance.correlation(a, b)


def pair_wise(a, b, edge_feature):
    if edge_feature == 'simple':
        return np.concatenate([a, b])
    elif edge_feature == 'add':
        return a + b
    elif edge_feature == 'hadamard':
        return a * b
    elif edge_feature == 'average':
        return (a + b) / 2
    elif edge_feature == 'l1':
        return abs(a - b)
    elif edge_feature == 'l2':
        return abs((a - b) * (a - b))
    elif edge_feature == 'all':
        hadamard = a * b
        average = (a + b) / 2
        weighted_l1 = abs(a - b)
        weighted_l2 = abs((a - b) * (a - b))
        return np.concatenate([hadamard, average, weighted_l1, weighted_l2])


def sim_metrics(a, b):
    a_entropy = entropy(a)
    b_entropy = entropy(b)
    entr_feature = pair_wise(a_entropy, b_entropy, 'all')
    sim_feature = np.array([js_divergence(a, b), cosine_sim(a,b), correlation_dist(a, b)])
    return np.concatenate([entr_feature, sim_feature])

def generate_attack_xy_node(args, g, pairs, label):
    features = g.ndata['features']
    node_features = []
    stat_dict = {}
    labels = [label] * len(pairs)

    for (start_id, end_id) in (pairs):
        start_feature = features[start_id].cpu().numpy()
        end_feature = features[end_id].cpu().numpy()

        if args.diff:
            start_entropy = entropy(start_feature)
            end_entropy = entropy(end_feature)
            node_feature = sim_metrics(start_entropy, end_entropy)
            node_feature[np.isnan(node_feature)] = 0
        else:
            node_feature = pair_wise(start_feature, end_feature, 'hadamard')

        node_features.append(node_feature)
        stat_dict[(start_id, end_id)] = {
            'node_ids': (start_id, end_id),
            f'{args.node_topology}_node_feature': node_feature,
            'label': label
        }

    print("Node features and labels of %d pairs have been generated" % len(labels))
    return node_features, labels, stat_dict

def generate_attack_xy_graph(args, g, pairs, label,mode):
    graph_features = []
    stat_dict = {}
    jaccard_dict, attach_dict, neighbor_dict = get_features(args, g, pairs, label,mode)
    labels = [label] * len(pairs)

    for (start_id, end_id) in (pairs):
        graph_feature = [
            jaccard_dict[(start_id, end_id)],
            attach_dict[(start_id, end_id)],
            neighbor_dict[(start_id, end_id)]
        ]
        graph_features.append(graph_feature)

        stat_dict[(start_id, end_id)] = {
            'node_ids': (start_id, end_id),
            f'{args.node_topology}_graph_feature': graph_feature,
            'label': label
        }

    print("Graph features and labels of %d pairs have been generated" % len(labels))
    return graph_features, labels, stat_dict

def generate_attack_xy_node_graph(args, g, pairs, label,mode):
    features = g.ndata['features']
    node_features = []
    graph_features = []
    stat_dict = {}

    jaccard_dict, attach_dict, neighbor_dict = get_features(args, g, pairs, label)
    labels = [label] * len(pairs)

    for (start_id, end_id) in tqdm(pairs):
        # Node feature
        start_feature = features[start_id].cpu().numpy()
        end_feature = features[end_id].cpu().numpy()

        if args.diff:
            start_entropy = entropy(start_feature)
            end_entropy = entropy(end_feature)
            node_feature = sim_metrics(start_entropy, end_entropy)
            node_feature[np.isnan(node_feature)] = 0
        else:
            node_feature = pair_wise(start_feature, end_feature, 'hadamard')
        node_features.append(node_feature)

        # Graph feature
        graph_feature = [
            jaccard_dict[(start_id, end_id)],
            attach_dict[(start_id, end_id)],
            neighbor_dict[(start_id, end_id)]
        ]
        graph_features.append(graph_feature)

        # Stat dict
        stat_dict[(start_id, end_id)] = {
            'node_ids': (start_id, end_id),
            'start_node_feature': start_feature,
            'end_node_feature': end_feature,
            'node_feature': node_feature,
            'graph_feature': graph_feature,
            'label': label
        }

    print("Node + graph features and labels of %d pairs have been generated" % len(labels))
    return node_features, graph_features, labels, stat_dict




def generate_attack_xy(args, pairs, posteriors_dict, label, index_mapping_dict=None):
    features = []
    stat_dict = {}
    for (start_id, end_id) in pairs:
        if index_mapping_dict:
            new_start_id, new_end_id = index_mapping_dict[(start_id, end_id)]
            start_posterior = F.softmax(posteriors_dict[new_start_id], dim=0)
            end_posterior = F.softmax(posteriors_dict[new_end_id], dim=0)
        else:
            start_posterior = F.softmax(posteriors_dict[start_id], dim=0)
            end_posterior = F.softmax(posteriors_dict[end_id], dim=0)
        if args.label_only:
            start_posterior, end_posterior = start_posterior.numpy(), end_posterior.numpy()
            label_dim = len(start_posterior)
            start_label = np.eye(label_dim)[np.argmax(start_posterior)]
            end_label = np.eye(label_dim)[np.argmax(end_posterior)]
            feature = pair_wise(start_label, end_label, 'add')
        elif args.diff:
            start_posterior, end_posterior = start_posterior.numpy(), end_posterior.numpy()
            feature = sim_metrics(start_posterior, end_posterior)
            feature[np.isnan(feature)] = 0
        elif args.soft_prob:
            start_posterior = F.softmax(start_posterior/args.T, dim=0).numpy()
            end_posterior = F.softmax(end_posterior/args.T, dim=0).numpy()
            feature = pair_wise(start_posterior, end_posterior, args.edge_feature)
        else:
            start_posterior, end_posterior = start_posterior.numpy(), end_posterior.numpy()
            feature = pair_wise(start_posterior, end_posterior, args.edge_feature)
            stat_dict[(start_id, end_id)] = {'node_ids':(start_id, end_id), f'{args.node_topology}_start_posterior': start_posterior, f'{args.node_topology}_end_posterior': end_posterior, f'{args.node_topology}_posterior_feature': feature, 'label': label}
        features.append(feature)
    print(start_posterior)
    labels = [label] * len(features)

    print("features and labels of %d pairs have been generated" % len(labels))
    
    return features, labels, stat_dict


def generate_attack_xy_plus(args, g, pairs, posteriors_dict, label, index_mapping_dict=None):
    features = g.ndata['features']
    node_features = []
    posterior_features = []
    stat_dict = {}
    labels = [label] * len(pairs)
    for (start_id, end_id) in tqdm(pairs):
        if index_mapping_dict:
            new_start_id, new_end_id = index_mapping_dict[(start_id, end_id)]
            start_posterior = F.softmax(posteriors_dict[new_start_id], dim=0 ).numpy()
            end_posterior = F.softmax(posteriors_dict[new_end_id], dim=0 ).numpy()
        else:
            start_posterior = F.softmax(posteriors_dict[start_id], dim=0).numpy()
            end_posterior = F.softmax(posteriors_dict[end_id], dim=0).numpy()
        if args.label_only:
            label_dim = len(start_posterior)
            start_label = np.eye(label_dim)[np.argmax(start_posterior)]
            end_label = np.eye(label_dim)[np.argmax(end_posterior)]
            posterior_feature = pair_wise(start_label, end_label, 'add')
        elif args.diff:
            start_entropy = entropy(start_posterior)
            end_entropy = entropy(end_posterior)
            posterior_feature = sim_metrics(start_entropy, end_entropy)
            posterior_feature[np.isnan(posterior_feature)] = 0
        else:
            posterior_feature = pair_wise(start_posterior, end_posterior, args.edge_feature)
        posterior_features.append(posterior_feature)

        start_feature = features[start_id].cpu().numpy()
        end_feature = features[end_id].cpu().numpy()
        if args.diff:
            start_entropy = entropy(start_feature)
            end_entropy = entropy(end_feature)
            node_feature = sim_metrics(start_entropy, end_entropy)
            node_feature[np.isnan(node_feature)] = 0
        else:
            node_feature = pair_wise(start_feature, end_feature, 'hadamard')
        node_features.append(node_feature)

        stat_dict[(start_id, end_id)] = {'node_ids':(start_id, end_id), f'{args.node_topology}_start_posterior': start_posterior, f'{args.node_topology}_end_posterior': end_posterior, f'{args.node_topology}_posterior_feature': posterior_feature, 'label': label}

    print("features and labels of %d pairs have been generated" % len(labels))
    
    return node_features, posterior_features, labels, stat_dict

def generate_attack_xy_plus2(args, g, pairs, posteriors_dict, label, mode, index_mapping_dict=None):
    posterior_features = []
    graph_features = []
    stat_dict = {}
    jaccard_dict, attach_dict, neighbor_dict  = get_features(args, g, pairs, label, mode)
    labels = [label] * len(pairs)
    for (start_id, end_id) in tqdm(pairs):
        if index_mapping_dict:
            new_start_id, new_end_id = index_mapping_dict[(start_id, end_id)]
            start_posterior = F.softmax(posteriors_dict[new_start_id], dim=0 ).numpy()
            end_posterior = F.softmax(posteriors_dict[new_end_id], dim=0 ).numpy()
        else:
            start_posterior = F.softmax(posteriors_dict[start_id], dim=0).numpy()
            end_posterior = F.softmax(posteriors_dict[end_id], dim=0).numpy()
        
        if args.label_only:
            label_dim = len(start_posterior)
            start_label = np.eye(label_dim)[np.argmax(start_posterior)]
            end_label = np.eye(label_dim)[np.argmax(end_posterior)]
            posterior_feature = pair_wise(start_label, end_label, 'add')
        elif args.diff:
            start_entropy = entropy(start_posterior)
            end_entropy = entropy(end_posterior)
            posterior_feature = sim_metrics(start_entropy, end_entropy)
            posterior_feature[np.isnan(posterior_feature)] = 0
        else:
            posterior_feature = pair_wise(start_posterior, end_posterior, args.edge_feature)
        posterior_features.append(posterior_feature)


        graph_feature = [jaccard_dict[(start_id, end_id)], attach_dict[(start_id, end_id)], neighbor_dict[(start_id, end_id)]]
        graph_features.append(graph_feature)
       
        stat_dict[(start_id, end_id)] = {'node_ids':(start_id, end_id), f'{args.node_topology}_start_posterior': start_posterior, f'{args.node_topology}_end_posterior': end_posterior, f'{args.node_topology}_posterior_feature': posterior_feature, 'label': label}

    print("features and labels of %d pairs have been generated" % len(labels))
    
    return posterior_features, graph_features, labels, stat_dict


def generate_attack_xy_all(args, g, pairs, posteriors_dict, label, mode, index_mapping_dict=None):
    features = g.ndata['features']
    node_features = []
    posterior_features = []
    graph_features = []
    stat_dict = {}
    jaccard_dict, attach_dict, neighbor_dict  = get_features(args, g, pairs, label, mode)
    labels = [label] * len(pairs)
    for (start_id, end_id) in tqdm(pairs):
        if index_mapping_dict:
            new_start_id, new_end_id = index_mapping_dict[(start_id, end_id)]
            start_posterior = F.softmax(posteriors_dict[new_start_id], dim=0 ).numpy()
            end_posterior = F.softmax(posteriors_dict[new_end_id], dim=0 ).numpy()
        else:
            start_posterior = F.softmax(posteriors_dict[start_id], dim=0).numpy()
            end_posterior = F.softmax(posteriors_dict[end_id], dim=0).numpy()
        
        if args.label_only:
            label_dim = len(start_posterior)
            start_label = np.eye(label_dim)[np.argmax(start_posterior)]
            end_label = np.eye(label_dim)[np.argmax(end_posterior)]
            posterior_feature = pair_wise(start_label, end_label, 'add')
        elif args.diff:
            start_entropy = entropy(start_posterior)
            end_entropy = entropy(end_posterior)
            posterior_feature = sim_metrics(start_entropy, end_entropy)
            posterior_feature[np.isnan(posterior_feature)] = 0
        else:
            posterior_feature = pair_wise(start_posterior, end_posterior, args.edge_feature)

        posterior_features.append(posterior_feature)

        start_feature = features[start_id].cpu().numpy()
        end_feature = features[end_id].cpu().numpy()
        if args.diff:
            start_entropy = entropy(start_feature)
            end_entropy = entropy(end_feature)
            node_feature = sim_metrics(start_entropy, end_entropy)
            node_feature[np.isnan(node_feature)] = 0
        else:
            node_feature = pair_wise(start_feature, end_feature, 'hadamard')
        node_features.append(node_feature)

        graph_feature = [jaccard_dict[(start_id, end_id)], attach_dict[(start_id, end_id)], neighbor_dict[(start_id, end_id)]]
        graph_features.append(graph_feature)
       
        stat_dict[(start_id, end_id)] = {'node_ids':(start_id, end_id), f'{args.node_topology}_start_posterior': start_posterior, f'{args.node_topology}_end_posterior': end_posterior, f'{args.node_topology}_posterior_feature': posterior_feature, 'start_node_feature': start_feature, 'end_node_feature': end_feature, 'node_feature':node_feature, 'graph_feature': graph_feature, 'label': label}

    print("features and labels of %d pairs have been generated" % len(labels))
    
    return node_features, posterior_features, graph_features, labels, stat_dict

