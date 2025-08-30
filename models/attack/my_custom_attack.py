from models.attack.base import BaseAttack

class MyCustomAttack(BaseAttack):
    supported_api_types = {"pyg"}   # you are using PyTorch Geometric
    supported_datasets = {"Cora"}   # or leave {} if you want it generic

    def __init__(self, dataset, attack_node_fraction=0.1, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        print(f"[MyCustomAttack] Running attack on {self.dataset.dataset_name}")
        print(f"[MyCustomAttack] Attacking {self.attack_node_fraction * 100}% of nodes")

        num_nodes = self.num_nodes
        num_features = self.num_features
        num_classes = self.num_classes

        print(f"Graph has {num_nodes} nodes, {num_features} features, {num_classes} classes")

        # === Dummy attack logic (replace with your real one) ===
        results = {
            "num_nodes_attacked": int(self.attack_node_fraction * num_nodes),
            "success_rate": 0.75,   # fake result for now
        }
        print(f"[MyCustomAttack] Results: {results}")
        return results

    def _load_model(self):
        print("[MyCustomAttack] Loading target model...")
        # TODO: add logic
        return None

    def _train_target_model(self):
        print("[MyCustomAttack] Training target model...")
        # TODO: add logic
        return None

    def _train_attack_model(self):
        print("[MyCustomAttack] Training attack model...")
        # TODO: add logic
        return None
