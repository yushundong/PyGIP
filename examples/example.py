# examples/example.py
from datasets import Cora
from models.attack import ModelExtractionAttack6 as MEA

def main():
    dataset = Cora()
    mea = MEA(dataset, 0.25)
    print("Running Attack-6 on Cora...")
    res = mea.attack()
    print("Results:", res)

if __name__ == "__main__":
    main()
