# PyGIP

PyGIP is a Python library designed for experimenting with graph-based model extraction attacks and defenses. It provides
a modular framework to implement and test attack and defense strategies on graph datasets.

## Installation

To get started with PyGIP, set up your environment by installing the required dependencies:

```bash
pip install -r reqs.txt
```

Ensure you have Python installed (version 3.8 or higher recommended) along with the necessary libraries listed
in `reqs.txt`.

Specifically, using following command to install `dgl 2.2.1` and ensure your `pytorch==2.3.0`.

```shell
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```

cuda

```shell
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
```

## Quick Start

Here’s a simple example to launch a Model Extraction Attack using PyGIP:

```python
from datasets import Cora
from models.attack import ModelExtractionAttack0

# Load the Cora dataset
dataset = Cora()

# Initialize the attack with a sampling ratio of 0.25
mea = ModelExtractionAttack0(dataset, 0.25)

# Execute the attack
mea.attack()
```

This code loads the Cora dataset, initializes a basic model extraction attack (`ModelExtractionAttack0`), and runs the
attack with a specified sampling ratio.

And a simple example to run a Defense method against Model Extraction Attack:

```python
from datasets import Cora
from models.defense import RandomWM

# Load the Cora dataset
dataset = Cora()

# Initialize the attack with a sampling ratio of 0.25
mead = RandomWM(dataset, 0.25)

# Execute the attack
mead.defend()
```

which runs the Random Watermarking Graph to defend against MEA.

If you want to use cuda, please set environment variable:

```shell
export PYGIP_DEVICE=cuda:0
```

## Implementation & Contributors Guideline

Refer to [Implementation Guideline](.github/IMPLEMENTATION.md)

Refer to [Contributors Guideline](.github/CONTRIBUTING.md)

## License

MIT License

## Contact

For questions or contributions, please contact blshen@fsu.edu.

---

## ️ GNN Watermark Defense

This module implements the watermarking method proposed in:

**Making Watermark Survive Model Extraction Attacks in Graph Neural Networks**  
*Wang, Shi, Xu, Sun, and Tang. NeurIPS 2023.*

This implementation is part of our internal reproduction effort, based on the original paper shared by the authors.

###  Integration

- All files are located in:  
  `pygip/models/defense/gnn_watermark/`

- The module is implemented as a subclass of `DefenseBase`, encapsulating both training and watermark verification steps.

- Entry point script:  
  `pygip/runners/run_watermark.py`

###  Run the Experiment

To run the full training and verification process:

```bash
PYTHONPATH=. python -m pygip.runners.run_watermark

