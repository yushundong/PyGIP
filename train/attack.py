import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
th.manual_seed(0)

from model.mlp import MLP_ATTACK, MLP_ATTACK_PLUS, MLP_ATTACK_PLUS2, MLP_ATTACK_ALL,Baseline0_MLP,Baseline1_MLP,Baseline2_MLP

def _weights_init_normal(m):

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        m.bias.data.fill_(0)
        
        
def save_attack_model(args, model):
    if not os.path.exists(args.attack_model_save_path):
            os.makedirs(args.attack_model_save_path)
    if args.prop:
        save_name = os.path.join(args.attack_model_save_path + f'attack_model_{args.dataset}_{args.target_model}_{args.shadow_model}_{args.prop}_{args.node_topology}_{args.feature}_{args.edge_feature}.pth')
    elif args.diff:
        save_name = os.path.join(args.attack_model_save_path + f'diff_attack_model_{args.target_dataset}_{args.shadow_dataset}_{args.target_model}_{args.shadow_model}_{args.node_topology}_{args.feature}_{args.edge_feature}.pth')
    else:
        save_name = os.path.join(args.attack_model_save_path + f'attack_model_{args.dataset}_{args.target_model}_{args.shadow_model}_{args.node_topology}_{args.feature}_{args.edge_feature}.pth')
    th.save(model.state_dict(), save_name)
    print("Finish training, save model to %s" % (save_name))


def load_attack_model(model, model_path, device):
    print("load model from: ", model_path)
    state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def test_one_feature(args, epoch, model, test_dataloader, stat_dict=None):
    device = args.device
    test_acc = 0.0
    correct = 0
    total = 0
    scores = []
    targets = []
    if not stat_dict:
        stat_dict = {} 
    model.eval()
    
    with th.no_grad():
        for indice, feature, label in test_dataloader:
            indice, feature, label = indice.to(device), feature.to(device), label.to(device)

            outputs = model(feature)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if epoch == args.num_epochs - 1 and not args.diff and not args.label_only and (args.mlp_layers == 3) and not args.soft_prob:
                for i, posterior in zip(indice, posteriors):  
                    stat_dict[tuple(i.cpu().numpy())][f'{args.method}_attack_posterior'] =  posterior.cpu().numpy()

            targets.extend(label.cpu().numpy().tolist())
            scores.extend([i.cpu().numpy()[1] for i in posteriors])     
        targets = np.array(targets)
        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        test_acc = correct / total
        test_auc = roc_auc_score(targets, scores)
        print('Test Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (100. * test_acc, correct, total, test_auc))

    return test_acc, test_auc, stat_dict


def test_two_features(args, epoch, model, test_dataloader, stat_dict=None):
    device = args.device
    test_acc = 0.0
    correct = 0
    total = 0
    scores = []
    targets = []
    stat_dict = {} if stat_dict is None else stat_dict
    model.eval()
    
    with th.no_grad():
        for indice, feature1, feature2, label in test_dataloader:
            indice, feature1, feature2, label = indice.to(device), feature1.to(device), feature2.to(device), label.to(device)

            outputs = model(feature1, feature2)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if epoch == args.num_epochs - 1 and not args.diff:
                for i, posterior in zip(indice, posteriors):
                    stat_dict[tuple(i.cpu().numpy())][f'{args.method}_attack_posterior'] =  posterior.cpu().numpy()
            targets.extend(label.cpu().numpy().tolist())
            scores.extend([i.cpu().numpy()[1] for i in posteriors])     

        test_acc = correct / total
        test_auc = roc_auc_score(targets, scores)
        print('Test Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (100. * test_acc, correct, total, test_auc))
    return test_acc, test_auc, stat_dict


def test_three_features(args, epoch, model, test_dataloader, stat_dict=None):
    device = args.device
    test_acc = 0.0
    correct = 0
    total = 0
    scores = []
    targets = []
    model.eval()
    
    with th.no_grad():
        for indice, feature1, feature2, feature3, label in test_dataloader:
            indice, feature1, feature2, feature3, label = indice.to(device), feature1.to(device), feature2.to(device), feature3.to(device), label.to(device)

            outputs = model(feature1, feature2, feature3)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if epoch == args.num_epochs - 1 and not args.diff:
                for i, posterior in zip(indice, posteriors):
                    stat_dict[tuple(i.cpu().numpy())][f'{args.method}_attack_posterior'] =  posterior.cpu().numpy()
            targets.extend(label.cpu().numpy().tolist())
            scores.extend([i.cpu().numpy()[1] for i in posteriors])         
        test_acc = correct / total
        test_auc = roc_auc_score(targets, scores)
        print('Test Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (100. * test_acc, correct, total, test_auc))
        
    return test_acc, test_auc, stat_dict

def test_gorn_feature(args, epoch, model, test_dataloader, stat_dict=None):
    device = args.device
    test_acc = 0.0
    correct = 0
    total = 0
    scores = []
    targets = []
    stat_dict = {} if stat_dict is None else stat_dict
    model.eval()
    
    with th.no_grad():
        for indice, feature, label in test_dataloader:
            indice = indice.to(device)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            targets.extend(label.cpu().numpy().tolist())
            scores.extend([i.cpu().numpy()[1] for i in posteriors])

        targets = np.array(targets)
        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        test_acc = correct / total
        test_auc = roc_auc_score(targets, scores)
        print('Test Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (100. * test_acc, correct, total, test_auc))

    return test_acc, test_auc, stat_dict



def test_graph_node_features(args, epoch, model, test_dataloader, stat_dict=None):
    device = args.device
    test_acc = 0.0
    correct = 0
    total = 0
    scores = []
    targets = []
    stat_dict = {} if stat_dict is None else stat_dict
    model.eval()
    
    with th.no_grad():
        for indice, feature1, feature2, label in test_dataloader:
            indice = indice.to(device)
            feature1 = feature1.to(device)
            feature2 = feature2.to(device)
            label = label.to(device)

            outputs = model(feature1, feature2)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            targets.extend(label.cpu().numpy().tolist())
            scores.extend([i.cpu().numpy()[1] for i in posteriors])

        test_acc = correct / total
        test_auc = roc_auc_score(targets, scores)
        print('Test Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (100. * test_acc, correct, total, test_auc))

    return test_acc, test_auc, stat_dict




def run_attack(args, in_dim, train_dataloader, test_dataloader, stat_dict):
    epoch = args.num_epochs
    device = args.device
    model = MLP_ATTACK(in_dim)
    model = model.to(args.device)
    model.apply(_weights_init_normal)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    
    if args.optim == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)
    scheduler3 = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
         
        correct = 0
        total = 0
        targets = []
        scores = []
        model.train()
        for _, feature, label in train_dataloader:
            optimizer.zero_grad()
            feature, label = feature.to(device), label.to(device)
            outputs = model(feature)
            posteriors = F.softmax(outputs, dim=1)
            
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors])  
        if args.scheduler:
            scheduler3.step()
            print(scheduler3.get_last_lr())
        train_acc = correct / total
        targets = np.array(targets)
        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        train_auc = roc_auc_score(targets, scores)
        print('[Epoch %d] Train Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (e, 100. * train_acc, correct, total, train_auc))
        
        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_one_feature(args, e, model, test_dataloader, stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_one_feature(args, e, model, test_dataloader)

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict


def run_attack_two_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict):
    epoch = args.num_epochs
    device = args.device

    if (args.feature == 'posteriors_graph') or (args.feature == 'label_graph') :
        model = MLP_ATTACK_PLUS2(args.graph_feature_dim, posterior_feature_dim)
    elif (args.feature == 'posteriors_node') or (args.feature == 'label_node'):
        model = MLP_ATTACK_PLUS(args.node_feature_dim, posterior_feature_dim)
    model = model.to(args.device)
    model.apply(_weights_init_normal)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    if args.optim == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)
    
    scheduler3 = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
         
        correct = 0
        total = 0
        targets = []
        scores = []
        model.train()
        for _, origin_feature, posterior_feature, label in train_dataloader:
            optimizer.zero_grad()
            origin_feature, posterior_feature, label = origin_feature.to(device), posterior_feature.to(device), label.to(device)
            outputs = model(origin_feature, posterior_feature)
            posteriors = F.softmax(outputs, dim=1)
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors])  
        if args.scheduler:
            scheduler3.step()
            print(scheduler3.get_last_lr())

        train_acc = correct / total
        train_auc = roc_auc_score(targets, scores)
        print('[Epoch %d] Train Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (e, 100. * train_acc, correct, total, train_auc))

        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_two_features(args, e, model, test_dataloader, stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_two_features(args, e, model, test_dataloader)

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict


def run_attack_three_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict):
    epoch = args.num_epochs
    device = args.device
    model = MLP_ATTACK_ALL(args.node_feature_dim, posterior_feature_dim, args.graph_feature_dim)
    model = model.to(args.device)
    model.apply(_weights_init_normal)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    if args.optim == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)

    scheduler3 = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
        targets = []
        scores = [] 
        correct = 0
        total = 0
        model.train()
        for _, node_feature, posterior_feature, graph_feature, label in train_dataloader:
            optimizer.zero_grad()
            node_feature, posterior_feature, graph_feature, label = node_feature.to(device), posterior_feature.to(device), graph_feature.to(device), label.to(device)

            outputs = model(node_feature, posterior_feature, graph_feature)
            posteriors = F.softmax(outputs, dim=1)
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors]) 
        if args.scheduler:
            scheduler3.step()
            print(scheduler3.get_last_lr())
        train_acc = correct / total
        train_auc = roc_auc_score(targets, scores)
        print('[Epoch %d] Train Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (e, 100. * train_acc, correct, total, train_auc))

        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_three_features(args, e, model, test_dataloader, stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_three_features(args, e, model, test_dataloader)

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict



def run_b0_attack(args, train_dataloader, test_dataloader, stat_dict):
    epoch = args.num_epochs
    device = args.device
    model = Baseline0_MLP(args.node_feature_dim)
    model = model.to(args.device)
    model.apply(_weights_init_normal)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    
    if args.optim == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)
    scheduler3 = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
         
        correct = 0
        total = 0
        targets = []
        scores = []
        model.train()
          
        for _, node_feature, label in train_dataloader:
            optimizer.zero_grad()
            node_feature, label = node_feature.to(device), label.to(device)
            outputs = model(node_feature)
            posteriors = F.softmax(outputs, dim=1)
            
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors])  
        if args.scheduler:
            scheduler3.step()
            print(scheduler3.get_last_lr())
        train_acc = correct / total
        targets = np.array(targets)
        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        train_auc = roc_auc_score(targets, scores)
        print('[Epoch %d] Train Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (e, 100. * train_acc, correct, total, train_auc))
        
        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_gorn_feature(args, e, model, test_dataloader, stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_gorn_feature(args, e, model, test_dataloader)

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict

def run_b1_attack(args, train_dataloader, test_dataloader, stat_dict):
    epoch = args.num_epochs
    device = args.device
    model = Baseline1_MLP(args.graph_feature_dim)
    model = model.to(args.device)
    model.apply(_weights_init_normal)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    
    if args.optim == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)
    scheduler3 = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
         
        correct = 0
        total = 0
        targets = []
        scores = []
        model.train()
       
        for _, graph_feature, label in train_dataloader:
            optimizer.zero_grad()
            graph_feature, label = graph_feature.to(device), label.to(device)
            outputs = model(graph_feature)
            posteriors = F.softmax(outputs, dim=1)
            
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors])  
        if args.scheduler:
            scheduler3.step()
            print(scheduler3.get_last_lr())
        train_acc = correct / total
        targets = np.array(targets)
        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        train_auc = roc_auc_score(targets, scores)
        print('[Epoch %d] Train Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (e, 100. * train_acc, correct, total, train_auc))
        
        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_gorn_feature(args, e, model, test_dataloader, stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_gorn_feature(args, e, model, test_dataloader)

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict





def run_b2_attack(args, train_dataloader, test_dataloader, stat_dict):
    epoch = args.num_epochs
    device = args.device
    model = Baseline2_MLP(args.node_feature_dim, args.graph_feature_dim)
    model = model.to(args.device)
    model.apply(_weights_init_normal)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    if args.optim == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr)

    scheduler3 = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
        targets = []
        scores = [] 
        correct = 0
        total = 0
        model.train()
        for _, node_feature, graph_feature, label in train_dataloader:
            optimizer.zero_grad()
            node_feature, graph_feature, label = node_feature.to(device),  graph_feature.to(device), label.to(device)

            outputs = model(node_feature, graph_feature)
            posteriors = F.softmax(outputs, dim=1)
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors]) 
        if args.scheduler:
            scheduler3.step()
            print(scheduler3.get_last_lr())
        train_acc = correct / total
        train_auc = roc_auc_score(targets, scores)
        print('[Epoch %d] Train Acc: %.3f%% (%d/%d) AUC Score: %.3f' % (e, 100. * train_acc, correct, total, train_auc))

        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_graph_node_features(args, e, model, test_dataloader, stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_graph_node_features(args, e, model, test_dataloader)

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict