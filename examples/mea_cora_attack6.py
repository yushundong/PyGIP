from datasets import Cora
from models.attack import ModelExtractionAttack6 as MEA

def main():
    dataset = Cora()
    mea = MEA(dataset, 0.25)
    print("Initialized MEA-6; starting attack...")
    res = mea.attack()
    print("Attack-6 finished. Results:", res)

if __name__ == "__main__":
    main()
