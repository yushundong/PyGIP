from datasets import CiteSeer
from models.defense import RandomWM

def main():
    dataset = CiteSeer()
    defense = RandomWM(dataset, 0.25)
    results = defense.defend()
    print("=== Defense finished on Citeseer dataset ===")
    print("Results:", results)

if __name__ == "__main__":
    main()