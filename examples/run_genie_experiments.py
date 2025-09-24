from datasets import Cora, PubMed
from models.attack import ModelExtractionAttack0 as MEA

dataset = Cora(api_type='dgl')
print(dataset)

mea = MEA(dataset, attack_node_fraction=0.1)
mea.attack()
