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

## Quick Start

Here’s a simple example to launch a model extraction attack using PyGIP:

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
Here’s an expanded and detailed version of the "Contribute to Code" section for your README.md, incorporating the
specifics of `BaseAttack` and `Dataset` you provided. This version is thorough, clear, and tailored for contributors:

## Implementation

PyGIP is built to be modular and extensible, allowing contributors to implement their own attack and defense strategies.
Below, we detail how to extend the framework by implementing custom attack and defense classes, with a focus on how to
leverage the provided dataset structure.

### Implementing Attack

To create a custom attack, you need to extend the abstract base class `BaseAttack`. Here’s the structure
of `BaseAttack`:

```python
class BaseAttack(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
        """Base class for all attack implementations."""
        self.dataset = dataset
        self.graph = dataset.graph  # Access the DGL-based graph directly
        # Additional initialization can go here

    @abstractmethod
    def attack(self):
        raise NotImplementedError

    def _train_target_model(self):
        raise NotImplementedError

    def _train_attack_model(self):
        raise NotImplementedError

    def _load_model(self, model_path):
        raise NotImplementedError
```

To implement your own attack:

1. **Inherit from `BaseAttack`**:
   Create a new class that inherits from `BaseAttack`. You’ll need to provide the following required parameters in the
   constructor:

- `dataset`: An instance of the `Dataset` class (see below for details).
- `attack_node_fraction`: A float between 0 and 1 representing the fraction of nodes to attack.
- `model_path` (optional): A string specifying the path to a pre-trained model (defaults to `None`).

You need to implement following methods:

- `attack()`: Add main attack logic here. If multiple attack types are supported, define the attack type as an optional
  argument to this function.  
  For each specific attack type, implement a corresponding helper function such as `_attack_type1()`
  or `_attack_type2()`,  
  and call the appropriate helper inside `attack()` based on the given method name.
- `_load_model()`: Load victim model.
- `_train_target_model()`: Train victim model.
- `_train_attack_model()`: Train attack model.
- `_helper_func()`(optional): Add your helper functions based on your needs, but keep the methods private.

2. **Implement the `attack()` Method**:
   Override the abstract `attack()` method with your attack logic, and return a dict of results. For example:

```python
class MyCustomAttack(BaseAttack):
    def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
        super().__init__(dataset, attack_node_fraction, model_path)
        # Additional initialization if needed

    def attack(self):
        # Example: Access the graph and perform an attack
        print(f"Attacking {self.attack_node_fraction * 100}% of nodes")
        num_nodes = self.graph.num_nodes()
        print(f"Graph has {num_nodes} nodes")
        # Add your attack logic here
        return {
            'metric1': 'metric1 here',
            'metric2': 'metric2 here'
        }

    def _load_model(self):
        # add your logic here
        pass

    def _train_target_model(self):
        # add your logic here
        pass

    def _train_attack_model(self):
        # add your logic here
        pass
```

### Implementing Defense

To create a custom defense, you need to extend the abstract base class `BaseDefense`. Here’s the structure
of `BaseDefense`:

```python
class BaseDefense(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float):
        """Base class for all defense implementations."""
        # add initialization here

    @abstractmethod
    def defend(self):
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError

    def _train_target_model(self):
        raise NotImplementedError

    def _train_defense_model(self):
        raise NotImplementedError

    def _train_surrogate_model(self):
        raise NotImplementedError
```

To implement your own defense:

1. **Inherit from `BaseDefense`**:
   Create a new class that inherits from `BaseDefense`. You’ll need to provide the following required parameters in the
   constructor:

- `dataset`: An instance of the `Dataset` class (see below for details).
- `attack_node_fraction`: A float between 0 and 1 representing the fraction of nodes to attack.
- `model_path` (optional): A string specifying the path to a pre-trained model (defaults to `None`).

You need to implement following methods:

- `defense()`: Add main defense logic here. If multiple defense types are supported, define the defense type as an
  optional argument to this function.  
  For each specific defense type, implement a corresponding helper function such as `_defense_type1()`
  or `_defense_type2()`,  
  and call the appropriate helper inside `defense()` based on the given method name.
- `_load_model()`: Load victim model.
- `_train_target_model()`: Train victim model.
- `_train_defense_model()`: Train defense model.
- `_train_surrogate_model()`: Train attack model.
- `_helper_func()`(optional): Add your helper functions based on your needs, but keep the methods private.


2. **Implement the `defense()` Method**:
   Override the abstract `defense()` method with your defense logic, and return a dict of results. For example:

```python
class MyCustomDefense(BaseDefense):
    def defend(self):
        # Step 1: Train target model
        target_model = self._train_target_model()
        # Step 2: Attack target model
        attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3)
        attack.attack(target_model)
        # Step 3: Train defense model
        defense_model = self._train_defense_model()
        # Step 4: Test defense against attack
        attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3)
        attack.attack(defense_model)
        # Print performance metrics

    def _load_model(self):
        # add your logic here
        pass

    def _train_target_model(self):
        # add your logic here
        pass

    def _train_defense_model(self):
        # add your logic here
        pass

    def _train_surrogate_model(self):
        # add your logic here
        pass
```

### Understanding the Dataset Class

The `Dataset` class standardizes the data format across PyGIP. Here’s its structure:

```python
class Dataset(object):
    def __init__(self, api_type='pyg', path='./downloads/'):
        self.api_type = api_type  # Set to 'pyg' for torch_geometric-based graphs
        self.path = path  # Directory for dataset storage
        self.dataset_name = ""  # Name of the dataset (e.g., "Cora")

        # Graph properties
        self.node_number = 0  # Number of nodes
        self.feature_number = 0  # Number of features per node
        self.label_number = 0  # Number of label classes

        # Core data
        self.graph = None  # PyG graph object
        self.features = None  # Node features
        self.labels = None  # Node labels

        # Data splits
        self.train_mask = None  # Boolean mask for training nodes
        self.val_mask = None  # Boolean mask for validation nodes
        self.test_mask = None  # Boolean mask for test nodes
```

- **Importance**: We are currently using the default api_type='pyg' to load the data. It is important to note that when
  api_type='pyg', `self.graph` should be an instance of `torch_geometric.data.Data`. In your implementation, make sure to
  use our defined Dataset class to build your code.
- Additional attributes like `self.dataset.features` (node features), `self.dataset.labels` (node labels),
  and `self.dataset.train_mask` (training split) are also available if your logic requires them.

### Miscellaneous Tips

- **Reference Implementation**: The `ModelExtractionAttack0` class is a fully implemented attack example. Study it for
  inspiration or as a template.
- **Flexibility**: Add as many helper functions as needed within your class to keep your code clean and modular.
- **Backbone Models**: We provide several basic backbone models like `GCN, GraphSAGE`. You can use or add more
  at `from models.nn import GraphSAGE`.

By following these guidelines, you can seamlessly integrate your custom attack or defense strategies into PyGIP. Happy
coding!

## Internal Code Submission Guideline

For internal team members with write access to the repository:

1. Always Use Feature/Fix Branches

- Never commit directly to the main or develop branch.
- Create a new branch for each feature, bug fix.

```shell
git checkout -b feat/your-feature-name
```

```shell
git checkout -b fix/your-fix-name
```

2. Keep Commits Clean & Meaningful

- feat: add data loader for graph dataset
- fix: resolve crash on edge cases

Use clear commit messages following the format:

```shell
<type>: <summary>
```

3. Test Before Pushing

- Test your implementation in `example.py`, and compare the performance with the results in original paper.

4. Push to Internal Branch

- Always run `git pull origin pygip-release` before pushing your changes
- Submit a pull request targeting the `pygip-release` branch
- Write a brief summary describing the features you’ve added, how to run your method, and how to evaluate its
  performance

Push to the remote feature branch.

```shell
git push origin feat/your-feature-name
```

## External Pull Request Guideline

Refer to [guidline](.github/CONTRIBUTING.md)

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

