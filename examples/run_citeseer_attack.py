from datasets import CiteSeer
from models.attack import ModelExtractionAttack0

def main():
    dataset = CiteSeer()
    attack = ModelExtractionAttack0(dataset, 0.25)
    results = attack.attack()
    print("=== Attack finished on Citeseer dataset ===")
    print("Results:", results)

if __name__ == "__main__":
    main()