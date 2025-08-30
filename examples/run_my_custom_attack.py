from models.attack.my_custom_attack import MyCustomAttack
from datasets import Cora


def main():
    dataset = Cora(api_type="pyg", path="./data")  # loads Cora 
    attack = MyCustomAttack(dataset, attack_node_fraction=0.2)
    results = attack.attack()
    print("Results:", results)

if __name__ == "__main__":
    main()
