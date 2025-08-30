from datasets import Cora
from models.defense import RandomWM

def main():
    dataset = Cora()
    defense = RandomWM(dataset, 0.25)
    results = defense.defend()
    print("=== Defense finished on Cora dataset ===")
    print("Results:", results)

if __name__ == "__main__":
    main()

