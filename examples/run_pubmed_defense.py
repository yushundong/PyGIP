from datasets import PubMed
from models.defense import RandomWM

def main():
    dataset = PubMed()
    defense = RandomWM(dataset, 0.25)
    results = defense.defend()
    print("=== Defense finished on PubMed dataset ===")
    print("Results:", results)

if __name__ == "__main__":
    main()