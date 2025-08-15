# examples/defense_cora.py
from datasets import Cora
# Your probe showed the module name is capitalized:
from models.defense.RandomWM import RandomWM  # module: RandomWM, class: RandomWM

def main():
    dataset = Cora()
    mead = RandomWM(dataset, 0.25)  # sampling ratio per README
    print("Initialized defense; starting defend()...")
    mead.defend()
    print("Defense finished.")

if __name__ == "__main__":
    main()
