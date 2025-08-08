import torch as th
from scipy.special import softmax


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), softmax(pred[val_nid].detach().cpu().numpy(), axis=1)

