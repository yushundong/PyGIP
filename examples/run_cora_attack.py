from datasets import Cora
from models.attack import ModelExtractionAttack0

def main():
    dataset = Cora()
    attack = ModelExtractionAttack0(dataset, 0.25)
    results = attack.attack()
    print("=== Attack finished on Cora dataset ===")
    print("Results:", results)

if __name__ == "__main__":
    main()

