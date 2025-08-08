
import numpy as np
import torch as th
import dgl
import os
from tqdm import tqdm
from multiprocessing import Pool
import sys

from utils.query_model import query_trained_model
from data.batch_data import get_batch_posteriors, get_batch_posteriors_node, get_batch_posteriors_graph, get_batch_posteriors_node_graph,get_batch_node_only,get_batch_graph_only,get_batch_node_graph_only
from data.generate_xy import generate_attack_xy, generate_attack_xy_plus,generate_attack_xy_node,generate_attack_xy_graph,generate_attack_xy_node_graph

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def generate_pairs(g, train_index):
    start_ids, end_ids = g.edges()
    postive_pairs = []
    negative_pairs = []
    for i in tqdm(range(len(start_ids))):
        if start_ids[i] < end_ids[i]:
            postive_pairs.append((start_ids[i].item(), end_ids[i].item()))

    num_pos_pairs = len(postive_pairs)
    print("There are %d edges in the training graph!" % (num_pos_pairs))
    while True:
        a, b = np.random.choice(list(train_index), 2, replace=False)
        random_pair = (a, b) if a < b else (b, a)
        if random_pair not in postive_pairs:
            negative_pairs.append(random_pair)
        if len(negative_pairs) == num_pos_pairs:
            break
    print("Finish Generating Pairs!")
    return postive_pairs, negative_pairs


def make_dirs():
    os.makedirs('./data/pairs/', exist_ok=True)
    os.makedirs('./data/posteriors/', exist_ok=True)
    os.makedirs('./data/mapping/', exist_ok=True)


def remove_neighbor_edge(g):
    """
    Remove all edges from a graph, only save self connection 
    """
    start_ids, end_ids = g.edges()
    delete_eid = []
    for i in tqdm(range(len(start_ids))):
        if start_ids[i] != end_ids[i]:
            delete_eid.append(i)
    g = dgl.remove_edges(g, th.tensor(delete_eid))

    return g


def normalized(feature, scaler, mode):
    if mode == 'shadow':
        feature_scaled = scaler.fit_transform(feature)
        return feature_scaled, scaler
    else:
        feature_scaled = scaler.transform(feature)
        return feature_scaled, scaler
    
    
def inductive_split_posteriors(args, train_g, test_g):
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0



    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        model = args.target_model if mode == 'target' else args.shadow_model
        index = np.arange(len(g.nodes()))
        stat_dict = {} 

        positive_pairs, negative_pairs = generate_pairs(g, index)
        print(f"Finish Generating Pairs...")
        
        if args.node_topology == '0-hop':
            zero_hop_g = remove_neighbor_edge(g)
        
            posteriors_dict = query_trained_model(args, index, zero_hop_g, mode)
            print("Finish Generating Posteriors Dict...") 

            positive_features, positive_labels, positive_stat_dict = generate_attack_xy(args, positive_pairs, posteriors_dict, 1)
            negative_features, negative_labels, negative_stat_dict = generate_attack_xy(args, negative_pairs, posteriors_dict, 0)
            
            stat_dict = {**positive_stat_dict, **negative_stat_dict}
            features = positive_features + negative_features
            labels = positive_labels + negative_labels
            
        elif args.node_topology == '1-hop' or args.node_topology == '2-hop':
            k = 1 if args.node_topology == '1-hop' else 2
            features = []
            labels = []
            flag = 1
            for pairs in (positive_pairs, negative_pairs):
                label = flag
                flag -= 1
                batch_size = 4096
                num_batch = len(pairs) // batch_size
                pool = Pool(12)
                results = []
                for i in tqdm(range(num_batch+1)):
                    if i == num_batch:
                        batch_pairs = pairs[i*batch_size:]
                    else:
                        batch_pairs = pairs[i*batch_size:(i+1)*batch_size]
                    batch_result = pool.apply_async(get_batch_posteriors, args=(args, batch_pairs, g, k, i, label, mode))
                    results.append(batch_result)
                pool.close()
                pool.join()
                for batch_result in results:
                    batch_result = batch_result.get()
                    features.extend(batch_result[0])
                    labels.extend(batch_result[1])
                    stat_dict.update(batch_result[2])
        
        features = np.array(features).astype(np.float32)
        features = th.from_numpy(features)            
        indices = th.from_numpy(np.array(positive_pairs+negative_pairs))
        labels = th.tensor(labels)
               
        dataset = th.utils.data.TensorDataset(indices, features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        stat_dicts.append(stat_dict)
        dataloaders.append(dataloader)
        count += 1
    feature_dim = features[0].shape[0]

    return dataloaders[0], dataloaders[1], feature_dim, stat_dicts[1]   
def inductive_split_node(args, train_g, test_g): 
    
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0

    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        index = np.arange(len(g.nodes()))
        stat_dict = {}

        positive_pairs, negative_pairs = generate_pairs(g, index)
        print("Finish Generating Pairs....")

        positive_node_features, positive_labels, positive_stat_dict = generate_attack_xy_node(args, g, positive_pairs, 1)
        negative_node_features, negative_labels, negative_stat_dict = generate_attack_xy_node(args, g, negative_pairs, 0)

        stat_dict = {**positive_stat_dict, **negative_stat_dict}
        node_features = positive_node_features + negative_node_features
        labels = positive_labels + negative_labels

        node_features = np.array(node_features).astype(np.float32)
        node_features = th.from_numpy(node_features)

        indices = th.from_numpy(np.array(positive_pairs + negative_pairs))
        labels = th.tensor(labels)

        dataset = th.utils.data.TensorDataset(indices, node_features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)

        count += 1

    feature_dim = node_features[0].shape[0]
    return dataloaders[0], dataloaders[1], feature_dim, stat_dicts[1]

def inductive_split_graph(args, train_g, test_g):
    
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0
    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        index = np.arange(len(g.nodes()))
        stat_dict = {}

        positive_pairs, negative_pairs = generate_pairs(g, index)
        print("Finish Generating Pairs....")

        graph_features = []
        labels = []
        flag = 1
        for pairs in (positive_pairs, negative_pairs):
            label = flag
            flag -= 1
            batch_size = 4096
            num_batch = len(pairs) // batch_size
            pool = Pool(12)
            results = []
            for i in tqdm(range(num_batch+1)):
                if i == num_batch:
                    batch_pairs = pairs[i*batch_size:]
                else:
                    batch_pairs = pairs[i*batch_size:(i+1)*batch_size]
                batch_result = pool.apply_async(get_batch_graph_only, args=(args, batch_pairs, g, i, label, mode))
                results.append(batch_result)
            pool.close()
            pool.join()
            for batch_result in results:
                batch_result = batch_result.get()
                graph_features.extend(batch_result[0])
                labels.extend(batch_result[1])
                stat_dict.update(batch_result[2])
                    
        graph_features = np.array(graph_features).astype(np.float32)
        graph_features = th.from_numpy(graph_features)

        indices = th.from_numpy(np.array(positive_pairs + negative_pairs))
        labels = th.tensor(labels)
        print(graph_features.shape)
        
        dataset = th.utils.data.TensorDataset(indices, graph_features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)

        count += 1

    feature_dim = graph_features[0].shape[0]
    return dataloaders[0], dataloaders[1], feature_dim, stat_dicts[1]

def inductive_split_node_graph(args, train_g, test_g):
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0
    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        all_index = np.arange(len(g.nodes()))
        stat_dict = {}
        positive_pairs, negative_pairs = generate_pairs(g, all_index)
        print("Finish Generating Pairs....")
        
        node_features = []
        graph_features = []
        labels = []
        flag = 1
        for pairs in (positive_pairs, negative_pairs):
            label = flag
            flag -= 1
            batch_size = 4096
            num_batch = len(pairs) // batch_size
            pool = Pool(12)
            results = []
            for i in tqdm(range(num_batch+1)):
                if i == num_batch:
                    batch_pairs = pairs[i*batch_size:]
                else:
                    batch_pairs = pairs[i*batch_size:(i+1)*batch_size]

                batch_result = pool.apply_async(get_batch_node_graph_only, args=(args, batch_pairs, g, i, mode,label))
                results.append(batch_result)
            pool.close()
            pool.join()

            for batch_result in results:
                batch_result = batch_result.get()
                node_features.extend(batch_result[0])
                graph_features.extend(batch_result[1])
                labels.extend(batch_result[2])
                stat_dict.update(batch_result[3])
        
        node_features = np.array(node_features).astype(np.float32)
        node_features = th.from_numpy(node_features)

        graph_features = np.array(graph_features).astype(np.float32)
        graph_features = th.from_numpy(graph_features)

        indices = th.from_numpy(np.array(positive_pairs + negative_pairs))
        labels = th.tensor(labels)

        count += 1

        dataset = th.utils.data.TensorDataset(indices, node_features, graph_features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)

    feature_dim = node_features[0].shape[0] + graph_features[0].shape[0]
    return dataloaders[0], dataloaders[1], feature_dim, stat_dicts[1]









def inductive_split_posteriors(args, train_g, test_g):
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0


    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        model = args.target_model if mode == 'target' else args.shadow_model
        index = np.arange(len(g.nodes()))
        stat_dict = {} 

        positive_pairs, negative_pairs = generate_pairs(g, index)
        print(f"Finish Generating Pairs...")
        
        if args.node_topology == '0-hop':
            zero_hop_g = remove_neighbor_edge(g)
        
            posteriors_dict = query_trained_model(args, index, zero_hop_g, mode)
            print("Finish Generating Posteriors Dict...") 

            positive_features, positive_labels, positive_stat_dict = generate_attack_xy(args, positive_pairs, posteriors_dict, 1)
            negative_features, negative_labels, negative_stat_dict = generate_attack_xy(args, negative_pairs, posteriors_dict, 0)
            
            stat_dict = {**positive_stat_dict, **negative_stat_dict}
            features = positive_features + negative_features
            labels = positive_labels + negative_labels
            
        elif args.node_topology == '1-hop' or args.node_topology == '2-hop':
            k = 1 if args.node_topology == '1-hop' else 2
            features = []
            labels = []
            flag = 1
            for pairs in (positive_pairs, negative_pairs):
                label = flag
                flag -= 1
                batch_size = 4096
                num_batch = len(pairs) // batch_size
                pool = Pool(12)
                results = []
                for i in tqdm(range(num_batch+1)):
                    if i == num_batch:
                        batch_pairs = pairs[i*batch_size:]
                    else:
                        batch_pairs = pairs[i*batch_size:(i+1)*batch_size]
                    batch_result = pool.apply_async(get_batch_posteriors, args=(args, batch_pairs, g, k, i, label, mode))
                    results.append(batch_result)
                pool.close()
                pool.join()
                for batch_result in results:
                    batch_result = batch_result.get()
                    features.extend(batch_result[0])
                    labels.extend(batch_result[1])
                    stat_dict.update(batch_result[2])
        
        features = np.array(features).astype(np.float32)
        features = th.from_numpy(features)            
        indices = th.from_numpy(np.array(positive_pairs+negative_pairs))
        labels = th.tensor(labels)
               
        dataset = th.utils.data.TensorDataset(indices, features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        stat_dicts.append(stat_dict)
        dataloaders.append(dataloader)
        count += 1
    feature_dim = features[0].shape[0]

    return dataloaders[0], dataloaders[1], feature_dim, stat_dicts[1]


def inductive_split_plus(args, train_g, test_g):
 
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0
    
    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        index = np.arange(len(g.nodes()))
        stat_dict = {}

        positive_pairs, negative_pairs = generate_pairs(g, index)
        print("Finish Generating Pairs....")

        if args.node_topology == '0-hop':
            zero_hop_g = remove_neighbor_edge(g)
            posteriors_dict = query_trained_model(args, index, zero_hop_g, mode)      
            print("Finish Generating Posteriors Dict...")
            
            positive_node_features, positive_posterior_features, positive_labels, positive_stat_dict = generate_attack_xy_plus(args, g, positive_pairs, posteriors_dict, 1)
            negative_node_features, negative_posterior_features, negative_labels, negative_stat_dict = generate_attack_xy_plus(args, g, negative_pairs, posteriors_dict, 0)

            stat_dict = {**positive_stat_dict, **negative_stat_dict}

            node_features = positive_node_features + negative_node_features
            posterior_features = positive_posterior_features + negative_posterior_features
            labels = positive_labels + negative_labels
        elif args.node_topology == '1-hop' or args.node_topology == '2-hop':
            k = 1 if args.node_topology == '1-hop' else 2
            node_features = []
            posterior_features = []
            labels = []
            flag = 1
            for pairs in (positive_pairs, negative_pairs):
                label = flag
                flag -= 1
                batch_size = 4096
                num_batch = len(pairs) // batch_size
                pool = Pool(12)
                results = []
                for i in tqdm(range(num_batch+1)):
                    if i == num_batch:
                        batch_pairs = pairs[i*batch_size:]
                    else:
                        batch_pairs = pairs[i*batch_size:(i+1)*batch_size]
                    batch_result = pool.apply_async(get_batch_posteriors_node, args=(args, batch_pairs, g, k, i, label, mode))
                    results.append(batch_result)
                pool.close()
                pool.join()
                for batch_result in results:
                    batch_result = batch_result.get()
                    node_features.extend(batch_result[0])
                    posterior_features.extend(batch_result[1])
                    labels.extend(batch_result[2])
                    stat_dict.update(batch_result[3])
                    
        node_features = np.array(node_features).astype(np.float32)
        node_features = th.from_numpy(node_features)

        posterior_features = np.array(posterior_features).astype(np.float32)
        posterior_features = th.from_numpy(posterior_features)

        indices = th.from_numpy(np.array(positive_pairs+negative_pairs))
        labels = th.tensor(labels)

        dataset = th.utils.data.TensorDataset(indices, node_features, posterior_features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)

        count += 1

    
    posterior_feature_dim = posterior_features[0].shape[0]

    return dataloaders[0], dataloaders[1], posterior_feature_dim, stat_dicts[1]

def inductive_split_plus2(args, train_g, test_g):
    # train_g is the shadow graph
    # test_g is the target graph
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0
    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        index = np.arange(len(g.nodes()))
        stat_dict = {}

        positive_pairs, negative_pairs = generate_pairs(g, index)
        print("Finish Generating Pairs....")

        if args.node_topology == '0-hop':
            print("Wrong Action")
            sys.exit(0)
        
        k = 1 if args.node_topology == '1-hop' else 2
        graph_features = []
        posterior_features = []
        labels = []
        flag = 1
        for pairs in (positive_pairs, negative_pairs):
            label = flag
            flag -= 1
            batch_size = 4096
            num_batch = len(pairs) // batch_size
            pool = Pool(12)
            results = []
            for i in tqdm(range(num_batch+1)):
                if i == num_batch:
                    batch_pairs = pairs[i*batch_size:]
                else:
                    batch_pairs = pairs[i*batch_size:(i+1)*batch_size]
                batch_result = pool.apply_async(get_batch_posteriors_graph, args=(args, batch_pairs, g, k, i, label, mode))
                results.append(batch_result)
            pool.close()
            pool.join()
            for batch_result in results:
                batch_result = batch_result.get()
                posterior_features.extend(batch_result[0])
                graph_features.extend(batch_result[1])
                labels.extend(batch_result[2])
                stat_dict.update(batch_result[3])
                    
        graph_features = np.array(graph_features).astype(np.float32)
        graph_features = th.from_numpy(graph_features)

        posterior_features = np.array(posterior_features).astype(np.float32)
        posterior_features = th.from_numpy(posterior_features)

        indices = th.from_numpy(np.array(positive_pairs+negative_pairs))
        labels = th.tensor(labels)
        print(graph_features.shape)
        
        dataset = th.utils.data.TensorDataset(indices, graph_features, posterior_features, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)

        count += 1
    
    posterior_feature_dim = posterior_features[0].shape[0]

    return dataloaders[0], dataloaders[1], posterior_feature_dim, stat_dicts[1]


def inductive_split_all(args, train_g, test_g):
    make_dirs()
    dataloaders = []
    stat_dicts = []
    count = 0
    for g in (train_g, test_g):
        if args.prop:
            mode = 'shadow'+ str(args.prop) if count == 0 else 'target'
        else:
            mode = 'shadow' if count == 0 else 'target'
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        all_index = np.arange(len(g.nodes()))
        stat_dict = {}
        positive_pairs, negative_pairs = generate_pairs(g, all_index)
        print("Finish Generating Pairs....")
        
        if args.node_topology == '0-hop':
            print("wrong action")
            sys.exit(0)

        k = 1 if args.node_topology == '1-hop' else 2
        node_features = []
        posterior_features = []
        graph_features = []
        labels = []
        flag = 1
        for pairs in (positive_pairs, negative_pairs):
            label = flag
            flag -= 1
            batch_size = 4096
            num_batch = len(pairs) // batch_size
            pool = Pool(12)
            results = []
            for i in tqdm(range(num_batch+1)):
                if i == num_batch:
                    batch_pairs = pairs[i*batch_size:]
                else:
                    batch_pairs = pairs[i*batch_size:(i+1)*batch_size]
                    
                batch_result = pool.apply_async(get_batch_posteriors_node_graph, args=(args, batch_pairs, g, k, i, label, mode))
                results.append(batch_result)
            pool.close()
            pool.join()

            for batch_result in results:
                batch_result = batch_result.get()
                node_features.extend(batch_result[0])
                posterior_features.extend(batch_result[1])
                graph_features.extend(batch_result[2])
                labels.extend(batch_result[3])
                stat_dict.update(batch_result[4])
        
        node_features = np.array(node_features).astype(np.float32)
        node_features = th.from_numpy(node_features)            
            
        graph_features = np.array(graph_features).astype(np.float32)
        graph_features = th.from_numpy(graph_features)

        posterior_features = np.array(posterior_features).astype(np.float32)
        posterior_features = th.from_numpy(posterior_features)

        indices = th.from_numpy(np.array(positive_pairs+negative_pairs))
        labels = th.tensor(labels)

        count += 1

        dataset = th.utils.data.TensorDataset(indices, node_features, posterior_features, graph_features, labels)
    
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)

    posterior_feature_dim = posterior_features[0].shape[0]

    return dataloaders[0], dataloaders[1], posterior_feature_dim,  stat_dicts[1]
