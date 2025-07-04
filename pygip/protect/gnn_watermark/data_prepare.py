from torch_geometric.datasets import TUDataset

print(" Loading dataset...")
datasets = {
    'enzymes': TUDataset(root='data/', name='ENZYMES'),
    'msrc': TUDataset(root='data/', name='MSRC_9')
}
print(f" Dataset loaded: ENZYMES={len(datasets['enzymes'])}, MSRC_9={len(datasets['msrc'])}")
