from datasets import Cora
from models.defense import GNNFingers

# Load the Cora dataset
dataset = Cora()

# Initialize the attack with a sampling ratio of 0.25
mead = GNNFingers(dataset, attack_node_fraction=0.25)

# Execute the attack
mead.defend()