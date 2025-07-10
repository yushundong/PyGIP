# Grove Documentation

This document provides a comprehensive guide to using the Grove model stealing attack framework.

## Quick Start

```python
from models.attack.grove_attack import GroveAttack
from models.defense.grove_defense import GroveDefense
from datasets import CoauthorCS, PubMed, Citeseer

# Load dataset
dataset = Citeseer(api_type='torch_geometric', path='./downloads/')

# Basic attack
attack = GroveAttack(dataset, attack_node_fraction=0.4)
results = attack.attack()
print(f"Attack Fidelity: {results['fidelity']:.4f}")
```

## Attack Types

### Type I Attack (Original Structure)
Uses the original graph structure for model stealing.

```python
attack = GroveAttack(
    dataset=dataset,
    attack_node_fraction=0.4,
    attack_type="type_i",          # Use original structure
    surrogate_architecture="gat",   # GAT surrogate model
    recovery_from="embedding"       # Recover embeddings
)
```

### Type II Attack (Reconstructed Structure)
Reconstructs graph structure using various methods.

```python
attack = GroveAttack(
    dataset=dataset,
    attack_node_fraction=0.4,
    attack_type="type_ii",         # Reconstruct structure
    structure="idgl",              # IDGL reconstruction
    surrogate_architecture="gin",   # GIN surrogate model
    recovery_from="prediction"      # Recover predictions
)
```

## Configuration Options

### Surrogate Architectures
- `"gat"` - Graph Attention Network
- `"gin"` - Graph Isomorphism Network  
- `"graphsage"` - GraphSAGE

### Recovery Methods
- `"embedding"` - Recover node embeddings
- `"prediction"` - Recover model predictions

### Structure Reconstruction (Type II only)
- `"original"` - Use original structure
- `"idgl"` - IDGL-based reconstruction
- `"knn"` - K-nearest neighbors
- `"random"` - Random graph generation

## Attack Methods

### Simple Attack
```python
results = attack.attack(attack_method="simple")
```

### Advanced Attacks
```python
# Fine-tuning attack
results = attack.attack(attack_method="fine_tuning")

# Double extraction attack
results = attack.attack(attack_method="double_extraction")

# Distribution shift attack
results = attack.attack(attack_method="distribution_shift")

# Pruned attack
results = attack.attack(attack_method="pruned")
```

## Defense

```python
from models.defense.grove_defense import GroveDefense

# Initialize defense
defense = GroveDefense(dataset, attack_node_fraction=0.4)

# Apply defense
defense_results = defense.defend()
print(f"Defense effectiveness: {defense_results}")
```

## Supported Datasets

```python
from datasets import (
    Citeseer,           # Citeseer dataset with auto-download
    CoauthorCS,    # Coauthor CS dataset
    AmazonPhoto,   # Amazon Photo dataset
    CitationFullDBLP, # Citation Full DBLP dataset
    PubMed,        # PubMed dataset
    Citeseer      # Citeseer dataset
)
```

## Complete Example

For a more comprehensive example demonstrating various attack and defense scenarios, refer to the `examples/grove_example.py` file.

## Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `attack_type` | Attack type | "type_i" | "type_i", "type_ii" |
| `surrogate_architecture` | Surrogate model | "gat" | "gat", "gin", "graphsage" |
| `recovery_from` | Recovery target | "embedding" | "embedding", "prediction" |
| `structure` | Graph structure | "original" | "original", "idgl", "knn", "random" |
| `num_epochs` | Training epochs | 200 | 50-500 |
| `learning_rate` | Learning rate | 0.001 | 0.0001-0.01 |


