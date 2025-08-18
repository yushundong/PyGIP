from datasets import Cora

MEA = None
try:
    from models.attack import ModelExtractionAttack0 as MEA
except Exception:
    try:
        from models.attack import ModelExtractionAttack as MEA
    except Exception:
        # Direct import from the submodule if not re-exported
        from models.attack.mea import ModelExtractionAttack0 as MEA

def main():
    dataset = Cora()
    mea = MEA(dataset, 0.25)   # sampling ratio
    print("Initialized MEA; starting attack...")
    mea.attack()
    print("Attack finished.")

if __name__ == "__main__":
    main()
