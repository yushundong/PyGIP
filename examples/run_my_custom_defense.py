from datasets import Cora
from models.defense.my_custom_defense import MyCustomDefense

if __name__ == "__main__":
    dataset = Cora(api_type="pyg", path="data/Cora")

    defense = MyCustomDefense(dataset, defense_node_fraction=0.3)
    defense.run()


    results = defense.defend()
    print("Defense results:", results)

