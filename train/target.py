import os
import dgl
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
th.manual_seed(1)

from utils.load_model import get_gnn_model
from utils.metrics__ import compute_acc, evaluate

def run_gnn(args, data):
    train_g, test_g = data
        
    train_nid = th.tensor(range(0, len(train_g.nodes())))
    test_nid = th.tensor(range(0, len(test_g.nodes())))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    
    # Define model and optimizer
    model = get_gnn_model(args)
    print(model)
    model = model.to(args.device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        tic_step = time.time()
        for step, (_, seeds, blocks) in enumerate(dataloader):
            blocks = [block.int().to(args.device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels'].to(device=args.device, dtype=th.long)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            batch_pred = F.softmax(batch_pred, dim=1)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))
            tic_step = time.time()

        toc = time.time()
        print('Epoch %d, Time(s):%.4f'%(epoch, toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            train_acc, _ = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, args.device)
            print('Train Acc {:.4f}'.format(train_acc))

            test_acc, _ = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.device)
            print('Test Acc: {:.4f}'.format(test_acc))
    
    if args.prop and args.mode == 'shadow':
        saving_path = os.path.join(args.model_save_path, '%s_%s_%s_%s%d.pth'%(args.setting, args.dataset, args.model, args.mode, args.prop))
    else:
        saving_path = os.path.join(args.model_save_path, '%s_%s_%s_%s.pth'%(args.setting, args.dataset, args.model, args.mode))
    print("Finish training, save model to %s"%(saving_path))
    th.save(model.state_dict(), saving_path)

    #finish training
    train_acc, _ = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid,  args.device)
    print('Final Train Acc {:.4f}'.format(train_acc))

    test_acc, _ = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.device)
    print('Final Test Acc {:.4f}'.format(test_acc))

    return train_acc, test_acc    
