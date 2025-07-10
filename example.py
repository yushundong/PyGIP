# from datasets import Cora
# from models.attack.mea import ModelExtractionAttack0

# dataset = Cora()

# >>>>>>>>>> test MEA

# mea = ModelExtractionAttack0(dataset, 0.25)
# mea.attack()


# >>>>>>>>>> test ImperceptibleWM
# from models.defense.OwnerWatermarkingDefense import OwnerWatermarkingDefense
# from utils.dglTopyg import dgl_to_pyg_data
#
# pyg_data = dgl_to_pyg_data(dataset.graph)
#
# defense = OwnerWatermarkingDefense(dataset)
# metrics = defense.defend(pyg_data)
#
# print(metrics)


# >>>>>>>>>> test ImperceptibleWM2
# from models.defense.ImperceptibleOwnerUniqueWatermark import WatermarkByBilevelOptimization
#
# defense = WatermarkByBilevelOptimization(dataset)
# defense.defend()


# >>>>>>>>>> test SurviveWM2
# from models.defense.SurviveWM2 import OptimizedWatermarkDefense
# from datasets import ENZYMES

# dataset = ENZYMES()
# defense = OptimizedWatermarkDefense(dataset, 0.25)
# defense.defend()


# >>>>>>>>>> test Grove
from models.attack.grove_attack import GroveAttack
from models.defense.grove_defense import GroveDefense
from datasets import ACM, CoauthorCS, PubMed, CitationFullDBLP, Citeseer, AmazonPhoto

# Load dataset
print("Loading dataset...")

citeseer = Citeseer(api_type='torch_geometric', path='./downloads/')

print(f"Citeseer dataset loaded: {citeseer.node_number} nodes, {citeseer.feature_number} features, {citeseer.label_number} classes")

# Basic Grove attack example
print("\n=== Basic Grove Attack Example ===")
attack = GroveAttack(
    dataset=citeseer,
    attack_node_fraction=0.4,
    attack_type="type_i",
    surrogate_architecture="gat",
    recovery_from="embedding",
    structure="original",
    num_epochs=200,
    learning_rate=0.001
)

results = attack.attack(attack_method="simple")
print(f"Attack Results:")
print(f"  Fidelity: {results['fidelity']:.4f}")
print(f"  Target Accuracy: {results['target_accuracy']:.4f}")
print(f"  Surrogate Accuracy: {results['surrogate_accuracy']:.4f}")
print(f"  Accuracy Gap: {results['accuracy_gap']:.4f}")

# Advanced attack example
print("\n=== Advanced Grove Attack Example ===")
advanced_attack = GroveAttack(
    dataset=citeseer,
    attack_node_fraction=0.4,
    attack_type="type_ii",
    surrogate_architecture="gin",
    recovery_from="prediction",
    structure="idgl",
    num_epochs=200,
    learning_rate=0.001
)

advanced_results = advanced_attack.attack(attack_method="fine_tuning")
print(f"Advanced Attack Results:")
print(f"  Fidelity: {advanced_results['fidelity']:.4f}")
print(f"  Surrogate Accuracy: {advanced_results['surrogate_accuracy']:.4f}")

# Grove defense example
print("\n=== Grove Defense Example ===")
defense = GroveDefense(citeseer, attack_node_fraction=0.4)
defense_results = defense.defend()
print(f"Defense Results: {defense_results}")

# >>>>>>>>>> test Grove complete example
