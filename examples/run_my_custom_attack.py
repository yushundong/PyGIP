from datasets import Cora
from models.attack import MyCustomAttack

dataset = Cora(api_type="pyg", path="./data")
attack = MyCustomAttack(dataset, attack_node_fraction=0.3)
results = attack.attack()
print("Final Results:", results)
