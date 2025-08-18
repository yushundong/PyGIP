from datasets import Cora
from models.defense.RandomWM import RandomWM

def main():
    dataset = Cora()
    mead = RandomWM(dataset, 0.25)  # sampling ratio
    print("Initialized defense; starting defend()...")
    mead.defend()
    print("Defense finished.")

if __name__ == "__main__":
    main()
