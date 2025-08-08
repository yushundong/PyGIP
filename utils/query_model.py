import os
import torch as th

from utils.load_model import get_gnn_model, load_trained_gnn_model

def query_trained_model(args, train_index, g, mode):
    '''
    query trained model using 0-hop training graph nodes
    '''
    if args.diff:
        args.in_feats = args.target_in_feats if mode == 'target' else args.shadow_in_feats
        args.n_classes= args.target_classes if mode == 'target' else args.shadow_classes
    args.model = args.target_model if mode == 'target' else args.shadow_model
    model = get_gnn_model(args).to(args.device)
    model_save_path = os.path.join(args.model_save_path, '%s_%s_%s_%s.pth' % (args.setting, args.dataset, args.model, mode))
    print(args.model_save_path)    
    print(f'Load {mode} model from: {model_save_path}')
    model = load_trained_gnn_model(model, model_save_path, args.device)

    model.eval()
    with th.no_grad():
        train_pred = model.inference(g, g.ndata['features'], args.device)
    res_dict = {}
    for i in range(len(train_index)):
        res_dict[train_index[i]] = train_pred[train_index[i]]
    print("Finish Querying %s Model!" % (mode))
    return res_dict