from datasets import Cora
from models.defense.my_custom_defense import MyCustomDefense

dataset = Cora(api_type="pyg", path="./data")
defense = MyCustomDefense(dataset, defense_node_fraction=0.3)
results = defense.defend()
print("Final Defense Results:", results)
