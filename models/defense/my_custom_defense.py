from typing import Dict, Any
from models.defense.base import BaseDefense
from models.attack.my_custom_attack import MyCustomAttack


class MyCustomDefense(BaseDefense): 
    supported_api_types = {"pyg"}       # can also add "dgl"
    supported_datasets = {"Cora"}       # you can add more later

    def __init__(self, dataset, defense_node_fraction: float, model_path: str = None):
        super().__init__(dataset, defense_node_fraction)
        self.model_path = model_path
        print("✅ MyCustomDefense initialized")

    def defend(self) -> Dict[str, Any]:
        print("🔒 Running defense...")

        # Step 1: Fake target model training
        print("📘 Training target model...")
        target_model = "dummy_target_model"

        # Step 2: Run attack
        print("⚔️ Running attack...")
        attack = MyCustomAttack(self.dataset, self.attack_node_fraction)
        attack_results_before = {"attack_success_rate": 0.75}  # dummy result
        print("⚔️ Attack results before defense:", attack_results_before)

        # Step 3: Fake defense training
        print("🛡️ Training defense model...")
        defense_model = "dummy_defense_model"

        # Step 4: Return results
        defense_results = {"status": "done", "defense_model": defense_model}
        print("🛡️ Defense results:", defense_results)

        return defense_results

    def run(self) -> Dict[str, Any]:
        """
        Entry point so you can call defense.run() in your example script.
        """
        return self.defend()


