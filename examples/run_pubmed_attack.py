from datasets import PubMed
from models.attack import ModelExtractionAttack0

def main():
    dataset = PubMed()
    attack = ModelExtractionAttack0(dataset, 0.25)
    results = attack.attack()
    print("=== Attack finished on PuubMed dataset ===")
    print("Results:", results)

if __name__ == "__main__":
    main()