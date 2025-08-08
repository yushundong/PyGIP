import torch
import torch.nn.functional as F

def soft_nearest_neighbor_loss(embeddings, labels, temperature=1.0):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    exp_dist = torch.exp(-pairwise_dist / temperature)
    same_class = (exp_dist * mask.float()).sum(1)
    all_class = exp_dist.sum(1)
    
    loss = -torch.log((same_class + 1e-8) / (all_class + 1e-8)).mean()
    return loss
