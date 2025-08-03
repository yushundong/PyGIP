from .base import BaseDefense
import torch
import torch.nn.functional as F
from torch.optim import Adam
from models.nn import GCN
import numpy as np
import random
from collections import Counter
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
import networkx as nx
import copy
import torch.optim as optim
import dgl
from itertools import combinations
from tqdm import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QueryBasedVerificationDefense(BaseDefense):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    def defend(self, num_trials=10, k=5, attack_type='mettack', knowledge='full', mode='transductive', verbose=True, **kwargs):
        """
        Main defense routine. Generates fingerprints, runs attacks, and verifies integrity.
        Returns a dict with per-trial and average metrics.
        """
        trial_results = []
        for trial in range(num_trials):
            if verbose:
                print(f"\n=== Trial {trial+1}/{num_trials} ===")

            # Step 1: Train target model
            model_clean = self._train_target_model()
            acc_clean = self._evaluate_accuracy(model_clean, self.dataset)

            # Step 2: Fingerprint it
            fingerprints = self._generate_fingerprints(model_clean, mode=mode, knowledge=knowledge, k=k, **kwargs)

            # Step 3: Attack the model
            poisoned_model, attack_info = self._run_attack(model_clean, attack_type=attack_type, knowledge=knowledge, **kwargs)
            poisoned_dataset = copy.deepcopy(self.dataset)
            if 'graph' in attack_info:
                poisoned_dataset.graph = attack_info['graph']
            acc_poisoned = self._evaluate_accuracy(poisoned_model, poisoned_dataset)


            # Step 4: Detect fingerprint flips
            flipped_info = self._evaluate_fingerprints(poisoned_model, fingerprints)

            flip_rate = flipped_info['flip_rate']
            acc_drop = acc_clean - acc_poisoned

            if verbose:
                print(f"Clean Accuracy:    {acc_clean:.4f}")
                print(f"Poisoned Accuracy: {acc_poisoned:.4f}")
                print(f"Accuracy Drop:     {acc_drop:.4f}")
                print(f"Flip Rate:         {flip_rate:.4f}")

            trial_results.append({
                'flip_rate': flip_rate,
                'accuracy_drop': acc_drop,
            })

        # Compute averages
        avg_flip_rate = sum(r['flip_rate'] for r in trial_results) / num_trials
        avg_acc_drop = sum(r['accuracy_drop'] for r in trial_results) / num_trials

        print(f"Clean Graph NumEdges:    {self.dataset.graph.num_edges()}")
        print(f"Poisoned Graph NumEdges: {poisoned_model.graph.num_edges() if hasattr(poisoned_model, 'graph') else 'N/A'}")


        return {
            'trial_results': trial_results,
            'average_flip_rate': avg_flip_rate,
            'average_accuracy_drop': avg_acc_drop,
        }



    def _train_target_model(self, epochs=200):
        """
        Trains target GCN model according to protocol in
        Wu et al. (2023), Section 6.1 for graph node classification.

        Returns
        -------
        model : torch.nn.Module
            The trained GCN model.
        """
        model = GCN(
        feature_number=self.dataset.feature_number,
        label_number=self.dataset.label_number
        ).to(device)
        print(f"Training target model on device: {device} ...")

        optimizer = Adam(model.parameters(), lr=0.02)
        loss_fn = torch.nn.NLLLoss()

        features = self.dataset.features.to(device)
        labels = self.dataset.labels.to(device)
        train_mask = self.dataset.train_mask.to(device)
        val_mask = getattr(self.dataset, "val_mask", None)
        if val_mask is None:
            val_mask = self.dataset.test_mask
        val_mask = val_mask.to(device)

        for epoch in range(epochs):
            model.train()
            logits = model(self.dataset.graph.to(device), features)
            log_probs = F.log_softmax(logits, dim=1)
            loss = loss_fn(log_probs[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(self.dataset.graph.to(device), features)
                    val_log_probs = F.log_softmax(val_logits, dim=1)
                    val_pred = val_log_probs[val_mask].max(1)[1]
                    val_acc = (val_pred == labels[val_mask]).float().mean().item()
                    print(f"Epoch {epoch+1}: Loss={loss.item():.4f} | Val Acc={val_acc:.4f}")

        return model

    def _load_model(self, model_path):
        model = GCN(
            in_feats=self.dataset.feature_number, 
            hidden_feats=16, 
            out_feats=self.dataset.label_number
        )
        model.load_state_dict(torch.load(model_path))
        return model


    def _generate_fingerprints(self, model, mode='transductive', knowledge='full', k=5, **kwargs):
        """
        Wrapper for fingerprint generation based on mode and knowledge level.
        Returns:
            List of fingerprints
        """
        if mode == 'transductive':
            generator = TransductiveFingerprintGenerator(
                model=model,
                dataset=self.dataset,
                candidate_fraction=kwargs.get('candidate_fraction', 1.0),
                random_seed=kwargs.get('random_seed', None),
                device=self.device,
                randomize=kwargs.get('randomize', True),
            )
            fingerprints = generator.generate_fingerprints(k=k, method=knowledge)

            unified_fingerprints = [(self.dataset.graph, node_id, label) for (node_id, label) in fingerprints]

        elif mode == 'inductive':
            generator = InductiveFingerprintGenerator(
                model=model,
                shadow_graph=self.dataset.graph,
                knowledge=knowledge,
                candidate_fraction=kwargs.get('candidate_fraction', 0.3),
                num_fingerprints=k,
                randomize=kwargs.get('randomize', True),
                random_seed=kwargs.get('random_seed', None),
                device=self.device,
                perturb_fingerprints=kwargs.get('perturb_fingerprints', False),
                perturb_budget=kwargs.get('perturb_budget', 5),
            )
            fingerprints = generator.generate_fingerprints(method=knowledge)
            unified_fingerprints = fingerprints

        else:
            raise ValueError("Unknown fingerprinting mode. Use 'transductive' or 'inductive'.")

        return unified_fingerprints

    def _evaluate_fingerprints(self, model, fingerprints):
        """
        Checks if fingerprinted nodes have changed labels under the given model.
        
        Args:
            model: The model to evaluate.
            fingerprints: List of (graph, node_id, label) tuples.

        Returns:
            results: {
                'flipped': List[Tuple[node_id, old_label, new_label]],
                'flip_rate': float
            }
        """
        model.eval()
        flipped = []
        
        with torch.no_grad():
            for graph, node_id, expected_label in fingerprints:
                x = graph.ndata['feat'] if hasattr(graph, 'ndata') else graph.x
                logits = model(graph.to(self.device), x.to(self.device))
                pred = logits[node_id].argmax().item()
                if pred != expected_label:
                    flipped.append((node_id, expected_label, pred))

        return {
            'flipped': flipped,
            'flip_rate': len(flipped) / len(fingerprints) if fingerprints else 0.0
        }


    def _run_attack(self, model, attack_type='mettack', knowledge='full', **kwargs):
        """
        Run the specified attack on the model.
        Returns:
            poisoned_model: torch.nn.Module
            metadata: dict with info about the attack
        """
        if attack_type == 'bitflip':
            attacker = BitFlipAttack(model=model, attack_type=kwargs.get('bitflip_type', 'random'), bit=kwargs.get('bit', 0))
            info = attacker.apply()
            return model, {'type': 'bitflip', 'info': info}

        elif attack_type == 'random':
            perturbed_graph = self._random_edge_addition_poisoning(
                perturb_frac=kwargs.get('perturb_frac', 0.01),
                random_seed=kwargs.get('random_seed', None),
            )
            poisoned_model = self._retrain_poisoned_model(
                poisoned_graph=perturbed_graph,
                epochs=kwargs.get('epochs', 200),
            )
            return poisoned_model, {'type': 'random_poison', 'graph': perturbed_graph}

        elif attack_type == 'mettack':
            helper = MettackHelper(
                graph=self.dataset.graph,
                features=self.dataset.features,
                labels=self.dataset.labels,
                train_mask=self.dataset.train_mask,
                val_mask=getattr(self.dataset, 'val_mask', None),
                test_mask=self.dataset.test_mask,
                n_perturbations=kwargs.get('n_perturbations', 5),
                device=self.device,
                max_perturbations=kwargs.get('max_perturbations', 50),
                surrogate_epochs=kwargs.get('surrogate_epochs', 30),
                candidate_sample_size=kwargs.get('candidate_sample_size', 20),
            )
            poisoned_graph, attack_metrics = helper.run()
            poisoned_model = self._retrain_poisoned_model(
                poisoned_graph=poisoned_graph,
                epochs=kwargs.get('epochs', 200),
            )
            return poisoned_model, {'type': 'mettack', 'metrics': attack_metrics, 'graph': poisoned_graph}

        else:
            raise ValueError(f"Unsupported attack_type: {attack_type}")


    def _random_edge_addition_poisoning(dataset, perturb_frac, random_seed=None):
        """
        Returns a new DGLGraph with random edges added.

        Args:
            dataset: Dataset object (with .graph as DGLGraph)
            perturb_frac: Fraction of edges to add (e.g., 0.01 = 1%)
            random_seed: Optional integer for reproducibility

        Returns:
            poisoned_graph: DGLGraph (deepcopy of original with new edges)
        """

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        orig_graph = dataset.graph
        poisoned_graph = copy.deepcopy(orig_graph)
        num_nodes = poisoned_graph.num_nodes()
        num_edges_to_add = int(perturb_frac * orig_graph.num_edges())

        existing_edges = set(zip(
            orig_graph.edges()[0].tolist(),
            orig_graph.edges()[1].tolist()
        ))

        candidate_pairs = [
            (i, j)
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j and (i, j) not in existing_edges
        ]

        if len(candidate_pairs) < num_edges_to_add:
            raise ValueError("Perturbation budget too large: not enough candidate edges.")

        new_edges = random.sample(candidate_pairs, num_edges_to_add)
        src, dst = zip(*new_edges)
        poisoned_graph.add_edges(src, dst)

        return poisoned_graph

    def _retrain_poisoned_model(self, poisoned_graph, epochs=200):
        """
        Retrain target GCN using the poisoned graph structure.

        Args:
            dataset: Original Dataset object (provides features, labels, masks)
            poisoned_graph: DGLGraph (with new random edges added)
            defense_class: The defense class to use for model training (e.g., QueryBasedVerificationDefense)
            device: 'cpu' or 'cuda'

        Returns:
            model: Trained GCN model
        """
        dataset_poisoned = copy.deepcopy(self.dataset)
        dataset_poisoned.graph = poisoned_graph

        defense = QueryBasedVerificationDefense(dataset=dataset_poisoned, attack_node_fraction=0.1)
        model = defense._train_target_model(epochs=epochs)
        return model


    def _evaluate_accuracy(self, model, dataset):
        """
        Evaluates test accuracy of the given model on the dataset.

        Args:
            model: Trained GCN model
            dataset: Dataset object (provides features, labels, test_mask, graph)
            device: 'cpu' or 'cuda'

        Returns:
            accuracy: float (test accuracy, 0-1)
        """
        model.eval()
        features = dataset.features.to(device)
        labels = dataset.labels.to(device)
        test_mask = dataset.test_mask

        with torch.no_grad():
            logits = model(dataset.graph.to(device), features)
            pred = logits.argmax(dim=1)
            correct = (pred[test_mask] == labels[test_mask]).float()
            accuracy = correct.sum().item() / test_mask.sum().item()
        return accuracy

    def run_full_pipeline(self, attack_type='random', mode='transductive', knowledge='full', k=5, trials=1, **kwargs):
        """
        Runs the full fingerprinting + attack + evaluation pipeline.
        
        Parameters:
            attack_type: 'random', 'bitflip', or 'mettack'
            mode: 'transductive' or 'inductive'
            knowledge: 'full' or 'limited'
            k: number of fingerprints
            trials: number of repeated trials
            kwargs: extra params for attack or fingerprinting

        Prints per-trial results and summary statistics.
        """
        flip_rates = []
        acc_drops = []

        for trial in range(trials):
            print(f"\n=== Trial {trial+1}/{trials} ===")

            model_clean = self._train_target_model()
            acc_clean = self._evaluate_accuracy(model_clean, self.dataset)
            print(f"Clean model accuracy: {acc_clean:.4f}")

            fingerprints = self._generate_fingerprints(model_clean, mode=mode, knowledge=knowledge, k=k, **kwargs)

            model_poisoned, attack_meta = self._run_attack(model_clean, attack_type=attack_type, knowledge=knowledge, **kwargs)
            acc_poisoned = self._evaluate_accuracy(model_poisoned, self.dataset)
            print(f"Poisoned model accuracy: {acc_poisoned:.4f}")

            eval_result = self._evaluate_fingerprints(model_poisoned, fingerprints)
            flip_rate = eval_result['flip_rate']
            print(f"Fingerprint flip rate: {flip_rate:.4f}")
            for (nid, old, new) in eval_result['flipped']:
                print(f"  Node {nid}: {old} → {new}")

            flip_rates.append(flip_rate)
            acc_drops.append(acc_clean - acc_poisoned)

        print("\n=== Summary ===")
        print(f"Avg Accuracy Drop: {np.mean(acc_drops):.4f}")
        print(f"Avg Fingerprint Flip Rate: {np.mean(flip_rates):.4f}")




class TransductiveFingerprintGenerator:
    def __init__(self, model, dataset, candidate_fraction=1.0, random_seed=None, device='cpu', randomize=True):
        self.model = model.to(device)
        self.dataset = dataset
        self.candidate_fraction = candidate_fraction
        self.random_seed = random_seed
        self.device = device
        self.randomize = randomize

    def get_candidate_nodes(self):
        """
        Step 1: Randomly sample a subset of nodes as candidates (for robustness).
        Step 2: Return that set for scoring.
        """
        all_nodes = torch.arange(self.dataset.graph.num_nodes())
        num_candidates = max(1, int(len(all_nodes) * self.candidate_fraction))

        if self.randomize and self.candidate_fraction < 1.0:
            generator = torch.Generator(device=self.device)
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed)
            idx = torch.randperm(len(all_nodes), generator=generator)[:num_candidates]
            candidates = all_nodes[idx]
            print(f"[DEBUG] Trial {self.random_seed}: Sampled candidates = {candidates.tolist()[:5]}")
        else:
            candidates = all_nodes

        return candidates



    def compute_fingerprint_scores_full(self, candidate_nodes):
        self.model.eval()
        scores = []
        logits = self.model(self.dataset.graph.to(self.device), self.dataset.features.to(self.device))

        for node in candidate_nodes:
            self.model.zero_grad()
            logit = logits[node]
            label = logit.argmax().item()
            loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([label], device=self.device))
            loss.backward(retain_graph=True)
            grad_norm = sum((p.grad ** 2).sum().item() for p in self.model.parameters() if p.grad is not None)
            scores.append(grad_norm)

        scores_tensor = torch.tensor(scores, device=self.device)
        print(f"[FULL] Fingerprint scores: mean={scores_tensor.mean():.4f}, std={scores_tensor.std():.4f}, max={scores_tensor.max():.4f}, min={scores_tensor.min():.4f}")
        return scores_tensor


    def compute_fingerprint_scores_limited(self, candidate_nodes):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.dataset.graph.to(self.device), self.dataset.features.to(self.device))
            probs = F.softmax(logits, dim=1)
            labels = probs.argmax(dim=1)
            scores = 1.0 - probs[candidate_nodes, labels[candidate_nodes]]

        print(f"[LIMITED] Fingerprint scores: mean={scores.mean():.4f}, std={scores.std():.4f}, max={scores.max():.4f}, min={scores.min():.4f}")
        return scores


    def select_top_fingerprints(self, scores, candidate_nodes, k, method='full'):
        """
        Selects top-k fingerprint nodes after filtering out extreme score outliers.
        """
        q = 0.99 if method == 'full' else 1.0  
        threshold = torch.quantile(scores, q)
        mask = scores <= threshold

        filtered_scores = scores[mask]
        filtered_candidates = candidate_nodes[mask]

        if filtered_scores.size(0) < k:
            print(f"[WARN] Only {filtered_scores.size(0)} candidates left after filtering, reducing k to fit.")
            k = filtered_scores.size(0)

        topk = torch.topk(filtered_scores, k)
        selected_nodes = filtered_candidates[topk.indices]
        selected_scores = topk.values

        return selected_nodes, selected_scores


    def generate_fingerprints(self, k=5, method='full'):
        candidate_nodes = self.get_candidate_nodes().to(self.device)

        with torch.no_grad():
            logits = self.model(self.dataset.graph.to(self.device), self.dataset.features.to(self.device))
            labels = logits.argmax(dim=1)

        if method == 'full':
            scores = self.compute_fingerprint_scores_full(candidate_nodes)
        elif method == 'limited':
            scores = self.compute_fingerprint_scores_limited(candidate_nodes)
        else:
            raise ValueError("method must be 'full' or 'limited'")

        class_to_candidates = {}
        for i, node in enumerate(candidate_nodes):
            cls = int(labels[node])
            if cls not in class_to_candidates:
                class_to_candidates[cls] = []
            class_to_candidates[cls].append((node.item(), scores[i].item()))

        rng = random.Random(self.random_seed)

        class_list = list(class_to_candidates.keys())
        rng.shuffle(class_list)

        fingerprints = []
        for cls in class_list:

            class_nodes = sorted(class_to_candidates[cls], key=lambda x: x[1], reverse=True)
            top_node = class_nodes[0][0]
            fingerprints.append((top_node, cls))
            if len(fingerprints) >= k:
                break

        if len(fingerprints) < k:

            fingerprint_nodes, _ = self.select_top_fingerprints(scores, candidate_nodes, k, method=method)
            fingerprints = [(int(n), int(labels[n])) for n in fingerprint_nodes]


        labels_only = [label for (_, label) in fingerprints]
        nodes_only = [node for (node, _) in fingerprints]

        print(f"[{method.upper()}] Fingerprint label distribution: {Counter(labels_only)}")
        print(f"[{method.upper()}] Fingerprint node IDs: {nodes_only}")

        return fingerprints



class InductiveFingerprintGenerator:
    """
    Implements inductive fingerprint generation for both Full ('full') and Limited ('limited')
    knowledge settings, as described in Wu et al. (2023) Sections 4.2, 4.2.2, and 5.2.
    Supports randomized candidate selection for robustness against adaptive attackers.
    """

    def __init__(self, model, shadow_graph, knowledge='limited',
                 candidate_fraction=0.3, num_fingerprints=5,
                 randomize=True, random_seed=None, device='cpu', 
                 perturb_fingerprints=False, perturb_budget=5):
        """
        Args:
            model: GNN model to be fingerprinted.
            shadow_graph: PyG/DGL graph object for querying (shadow/inference graph).
            knowledge: 'full' for gradient-based (requires model weights), 'limited' for output-based.
            candidate_fraction: Fraction of nodes considered as candidates for fingerprinting.
            num_fingerprints: Number of fingerprint nodes to select.
            randomize: Whether to randomly sample candidate nodes (default True).
            random_seed: Optional seed for reproducibility.
            device: Torch device string (e.g., 'cpu' or 'cuda').
            perturb_fingerprints: Whether to greedily perturb fingerprint nodes' features/edges to increase sensitivity.
            perturb_budget: Max number of perturbation steps per fingerprint node (default 5).
       
        """
        self.model = model.to(device)
        self.shadow_graph = shadow_graph
        self.knowledge = knowledge
        self.candidate_fraction = candidate_fraction
        self.num_fingerprints = num_fingerprints
        self.randomize = randomize
        self.random_seed = random_seed
        self.device = device
        self.perturb_fingerprints = perturb_fingerprints
        self.perturb_budget = perturb_budget

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)

    def get_candidate_nodes(self):
        """
        Step 1: Randomly sample a subset of nodes as candidates (for robustness).
        Step 2: Score and select top-k from this set.
        """
        all_nodes = torch.arange(self.shadow_graph.num_nodes())
        num_candidates = max(1, int(len(all_nodes) * self.candidate_fraction))

        if self.randomize and self.candidate_fraction < 1.0:
            generator = torch.Generator(device=self.device)
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed)
            idx = torch.randperm(len(all_nodes), generator=generator)[:num_candidates]
            candidates = all_nodes[idx]
            print(f"[DEBUG] Trial {self.random_seed}: Sampled candidates = {candidates.tolist()[:5]}")
        else:
            candidates = all_nodes

        return candidates


    def compute_fingerprint_score(self, node_idx):
        """
        Computes the fingerprint score for a given node according to knowledge mode.
        Returns: float: Sensitivity score for the node.
        """
        features = self.shadow_graph.ndata['feat'] if hasattr(self.shadow_graph, 'ndata') else self.shadow_graph.x
        features = features.to(self.device)
        self.model.eval()

        if self.knowledge == 'limited':
            with torch.no_grad():
                logits = self.model(self.shadow_graph.to(self.device), features)
                probs = torch.softmax(logits[node_idx], dim=0)
                pred_class = probs.argmax().item()
                score = 1 - probs[pred_class].item()
            return score

        elif self.knowledge == 'full':
            # Full knowledge: compute gradient norm wrt input features of the node
            features.requires_grad_(True)
            logits = self.model(self.shadow_graph.to(self.device), features)
            pred = logits[node_idx]
            label = pred.argmax().item()

            self.model.zero_grad()
            loss = torch.nn.functional.nll_loss(
                torch.log_softmax(pred.unsqueeze(0), dim=1),
                torch.tensor([label], device=self.device)
            )
            loss.backward(retain_graph=True)
            # For simplicity, we use grad wrt features (could be extended to model params)
            grad = features.grad[node_idx]
            grad_norm_sq = (grad ** 2).sum().item()
            features.requires_grad_(False)
            features.grad = None  # Clean up
            return grad_norm_sq

        else:
            raise ValueError("knowledge must be 'limited' or 'full'")


    def generate_fingerprint_nodes(self):
        """
        Step 3: Identifies and returns the top-k (num_fingerprints) nodes with the highest
        fingerprint scores from the candidate set. (Section 4.2.2)

        Returns:
            List[int]: Indices of selected fingerprint nodes.
        """
        candidates = self.get_candidate_nodes()
        scores = []
        for idx in candidates:
            score = self.compute_fingerprint_score(idx)
            scores.append((score, int(idx)))
        # Sort candidates by score, descending
        scores.sort(reverse=True)
        selected = [idx for (_, idx) in scores[:self.num_fingerprints]]
        return selected

    def save_fingerprint_tuples(self, node_indices):
        """
        Step 4: Creates the final fingerprint set, storing the expected label for each
        selected fingerprint node. Tuples (graph, node_id, label) will be used
        during online verification.

        Args:
            node_indices: List[int] of selected fingerprint node indices.

        Returns:
            List[Tuple[graph, node_id, label]]: The fingerprints for online checking.
        """
        self.model.eval()
        with torch.no_grad():
            features = self.shadow_graph.ndata['feat'] if hasattr(self.shadow_graph, 'ndata') else self.shadow_graph.x
            logits = self.model(self.shadow_graph.to(self.device), features.to(self.device))
            labels = logits.argmax(dim=1).cpu().numpy()
            fingerprints = [(self.shadow_graph, int(idx), int(labels[idx])) for idx in node_indices]
        return fingerprints

    def generate_fingerprints(self, method='full'):
            """
            Generate inductive fingerprints for model watermarking.

            Parameters:
                method (str): 'full' for gradient-based or 'limited' for output-based

            Returns:
                List of fingerprints
            """
            if method == 'full':
                return self._generate_full()
            elif method == 'limited':
                return self._generate_limited()
            else:
                raise ValueError(f"Invalid fingerprinting method: '{method}'")
            
    def _generate_full(self):
        """
        Implements full knowledge fingerprint generation (gradient-based).
        Based on Section 4.2.1 and 5.2 of Wu et al. (2023).
        """
        self.knowledge = 'full'
        print("[Fingerprint] Generating FULL knowledge fingerprints...")
        fingerprint_nodes = self.generate_fingerprint_nodes()

        if self.perturb_fingerprints:
            print("[Fingerprint] Applying greedy feature perturbation (FULL)...")
            self.greedy_perturb_fingerprints(fingerprint_nodes)

        return self.save_fingerprint_tuples(fingerprint_nodes)

    def _generate_limited(self):
        """
        Implements limited knowledge fingerprint generation (output-based).
        Based on Section 4.2.2 and 5.2 of Wu et al. (2023).
        """
        self.knowledge = 'limited'
        print("[Fingerprint] Generating LIMITED knowledge fingerprints...")
        fingerprint_nodes = self.generate_fingerprint_nodes()

        if self.perturb_fingerprints:
            print("[Fingerprint] Applying greedy feature perturbation (LIMITED)...")
            self.greedy_perturb_fingerprints(fingerprint_nodes)

        return self.save_fingerprint_tuples(fingerprint_nodes)


    def greedy_perturb_fingerprints(self, node_indices):
        """
        Greedily perturbs each fingerprint node's features (not edges) to increase its
        fingerprint score, without changing the predicted label.

        - For each node, for each feature dimension:
            - Add or subtract a small epsilon.
            - Accept change if predicted label stays the same and fingerprint score increases.
            - Stop after perturb_budget attempts or no improvement.

        Returns:
            List[int]: Indices of perturbed fingerprint nodes (features in shadow_graph are updated in-place).
        """
        epsilon = 0.01  # Perturbation magnitude; you may want to tune this
        features = self.shadow_graph.ndata['feat'] if hasattr(self.shadow_graph, 'ndata') else self.shadow_graph.x
        features = features.clone().detach().to(self.device)
        self.shadow_graph = self.shadow_graph.to(self.device)

        for idx in node_indices:
            num_tries = 0
            improved = True
            while num_tries < self.perturb_budget and improved:
                improved = False
                current_score = self.compute_fingerprint_score(idx)
                # Get current prediction
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(self.shadow_graph, features)
                    pred_label = logits[idx].argmax().item()
                original_features = features[idx].clone()
                for dim in range(features.shape[1]):
                    for direction in [+1, -1]:
                        features[idx][dim] += direction * epsilon
                        # Get new prediction and score
                        self.model.eval()
                        with torch.no_grad():
                            logits_new = self.model(self.shadow_graph, features)
                            new_pred_label = logits_new[idx].argmax().item()
                        new_score = self.compute_fingerprint_score(idx)
                        # Accept if label unchanged and score increased
                        if new_pred_label == pred_label and new_score > current_score:
                            current_score = new_score
                            improved = True
                            num_tries += 1
                        else:
                            features[idx][dim] = original_features[dim]  # Revert
                        if num_tries >= self.perturb_budget:
                            break
                    if num_tries >= self.perturb_budget:
                        break
        # Optionally, update self.shadow_graph features (depends on your data structure)
        if hasattr(self.shadow_graph, 'ndata'):
            self.shadow_graph.ndata['feat'] = features
        else:
            self.shadow_graph.x = features
        return node_indices


class BitFlipAttack:
    def __init__(self, model, attack_type='random', bit=0):
        self.model = model
        self.attack_type = attack_type
        self.bit = bit
        
    def _get_target_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad and p.numel() > 0]
        if self.attack_type == 'random':
            return params
        elif self.attack_type == 'BFA-F':
            return [params[0]]
        elif self.attack_type == 'BFA-L':
            return [params[-1]]
        else:
            raise ValueError(f"Unknown attack_type {self.attack_type}")
        
    def _true_bit_flip(self, tensor, index=None, bit=0):
        a = tensor.detach().cpu().numpy().copy()
        flat = a.ravel()
        if index is None:
            index = np.random.randint(0, flat.size)
        old_val = flat[index]
        int_view = np.frombuffer(flat[index].tobytes(), dtype=np.uint32)[0]
        int_view ^= (1 << bit)
        new_val = np.frombuffer(np.uint32(int_view).tobytes(), dtype=np.float32)[0]
        flat[index] = new_val
        a = flat.reshape(a.shape)
        tensor.data = torch.from_numpy(a).to(tensor.device)
        return old_val, new_val, index
    
    def apply(self):
        params = self._get_target_params()
        with torch.no_grad():
            layer_idx = random.randrange(len(params))
            param = params[layer_idx]
            idx = random.randrange(param.numel())
            old_val, new_val, actual_idx = self._true_bit_flip(param, index=idx, bit=self.bit)
        return {
            'layer': layer_idx,
            'param_idx': actual_idx,
            'old_val': old_val,
            'new_val': new_val,
            'bit': self.bit,
            'attack_type': self.attack_type
        }
    

class MettackHelper:
    def __init__(self, graph, features, labels, train_mask, val_mask, test_mask,
                 n_perturbations=5, device='cpu', max_perturbations=50,
                 surrogate_epochs=30, candidate_sample_size=20):
        # Add self-loops to the original graph to prevent zero in-degree issues
        self.graph = dgl.add_self_loop(graph).to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.train_mask = train_mask.to(device)
        self.surrogate_epochs = surrogate_epochs
        self.candidate_sample_size = candidate_sample_size
        # Handle case where val_mask might be None
        if val_mask is not None:
            self.val_mask = val_mask.to(device)
        else:
            # Create a validation mask from a subset of training data
            self.val_mask = self._create_val_mask_from_train(train_mask).to(device)
            
        self.test_mask = test_mask.to(device)
        
        # Cap the number of perturbations to a reasonable limit
        self.n_perturbations = min(n_perturbations, max_perturbations)
        self.device = device

        # Surrogate GCN, matches the victim model structure from the paper (Sec. 6.1)
        in_feats = features.shape[1]
        n_classes = int(labels.max().item()) + 1
        self.surrogate = GCN(in_feats, n_classes).to(device)

        # For reproducibility (optional)
        torch.manual_seed(42)
        np.random.seed(42)

        # Track current edge modifications if desired
        self.modified_edges = set()
        
        # Store original adjacency for candidate generation (without self-loops for edge candidates)
        original_graph_no_self_loop = dgl.remove_self_loop(graph)
        self.original_edges = set(zip(original_graph_no_self_loop.edges()[0].cpu().numpy(), 
                                    original_graph_no_self_loop.edges()[1].cpu().numpy()))
        
        # Pre-compute candidate edges for efficiency
        self.candidate_edges = self._get_candidate_edges()

    def _create_val_mask_from_train(self, train_mask):
        """
        Create a validation mask by taking a subset of training nodes.
        This is needed when the dataset doesn't provide a validation mask.
        """
        train_indices = torch.where(train_mask)[0]
        n_val = min(500, len(train_indices) // 4)  # Use 25% of training data or 500, whichever is smaller
        
        # Randomly select validation indices from training indices
        perm = torch.randperm(len(train_indices))
        val_indices = train_indices[perm[:n_val]]
        
        # Create validation mask
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        val_mask[val_indices] = True
        
        # Update training mask to exclude validation nodes
        self.train_mask = train_mask.clone()
        self.train_mask[val_indices] = False
        
        return val_mask

    def run(self):
        """
        Main entrypoint to run the Mettack algorithm.
        Returns:
            poisoned_graph (DGLGraph): The perturbed graph with edges changed.
            metrics (dict): Metrics for before/after attack, for evaluation.
        """
        print("Starting Mettack attack...")
        
        # 1. Train surrogate GCN on the clean graph
        print("Training surrogate model...")
        self._train_surrogate()

        # 2. Run bi-level optimization to find edge perturbations
        print("Applying structure attack...")
        poisoned_graph = self._apply_structure_attack()

        # 3. (Optional) Retrain model on poisoned_graph and collect metrics
        print("Evaluating attack results...")
        metrics = self._evaluate(poisoned_graph)

        return poisoned_graph, metrics

    def _train_surrogate(self):
        """
        Trains a surrogate GCN on the clean graph.
        (Matches Wu et al., Section 6.1)
        """
        optimizer = optim.Adam(self.surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        self.surrogate.train()
        
        # Standard GCN training loop
        for epoch in range(self.surrogate_epochs):
            optimizer.zero_grad()
            logits = self.surrogate(self.graph, self.features)
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                self.surrogate.eval()
                with torch.no_grad():
                    val_logits = self.surrogate(self.graph, self.features)
                    val_acc = self._compute_accuracy(val_logits[self.val_mask], 
                                                   self.labels[self.val_mask])
                    print(f"Surrogate epoch {epoch}: Val Acc = {val_acc:.4f}")
                self.surrogate.train()

    def _apply_structure_attack(self):
        """
        Runs the Mettack structure perturbation loop (bi-level optimization).
        - At each step, modify the adjacency matrix (add/remove an edge).
        - Select the perturbation that maximizes surrogate model loss on the validation nodes.
        - Repeat up to n_perturbations times.
        Returns a new DGLGraph with edges modified.
        (See Appendix A.2 in Wu et al.)
        """
        current_graph = copy.deepcopy(self.graph)
        perturbed_edges = set()
        
        for step in range(self.n_perturbations):
            print(f"Perturbation step {step + 1}/{self.n_perturbations}")
            
            best_edge = None
            best_loss = -float('inf')
            best_action = None  # 'add' or 'remove'
            
            # Sample candidate edges for efficiency (reduced for speed)
            candidate_sample = np.random.choice(len(self.candidate_edges), 
                                            min(self.candidate_sample_size, len(self.candidate_edges)),
                                            replace=False)

            
            for idx in tqdm(candidate_sample, desc="Evaluating candidates"):
                edge = self.candidate_edges[idx]
                
                # Skip if already perturbed
                if edge in perturbed_edges or (edge[1], edge[0]) in perturbed_edges:
                    continue
                
                # Try both add and remove operations
                for action in ['add', 'remove']:
                    if action == 'add' and edge in self.original_edges:
                        continue
                    if action == 'remove' and edge not in self.original_edges:
                        continue
                    
                    # Create temporary graph with this perturbation
                    temp_graph = self._apply_single_perturbation(current_graph, edge, action)
                    
                    # Evaluate attack loss on this perturbed graph
                    attack_loss = self._compute_attack_loss(temp_graph)
                    
                    if attack_loss > best_loss:
                        best_loss = attack_loss
                        best_edge = edge
                        best_action = action
            
            # Apply the best perturbation
            if best_edge is not None:
                current_graph = self._apply_single_perturbation(current_graph, best_edge, best_action)
                perturbed_edges.add(best_edge)
                self.modified_edges.add((best_edge, best_action))
                print(f"Applied {best_action} edge {best_edge} with loss increase: {best_loss:.4f}")
            else:
                print("No beneficial perturbation found, stopping early.")
                break
        
        return current_graph

    def _get_candidate_edges(self):
        """
        Generate candidate edges for perturbation.
        Includes both existing edges (for removal) and non-existing edges (for addition).
        """
        n_nodes = self.graph.num_nodes()
        
        # Get all possible edges (excluding self-loops for undirected graphs)
        all_possible_edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):  # Assume undirected graph
                all_possible_edges.append((i, j))
        
        # Convert to set for faster lookup
        return all_possible_edges[:min(10000, len(all_possible_edges))]  # Limit for efficiency

    def _apply_single_perturbation(self, graph, edge, action):
        """
        Apply a single edge perturbation (add or remove) to the graph.
        """
        temp_graph = copy.deepcopy(graph)
        
        if action == 'add':
            # Add edge in both directions for undirected graph
            temp_graph.add_edges([edge[0], edge[1]], [edge[1], edge[0]])
        elif action == 'remove':
            # Find and remove the edge
            src, dst = temp_graph.edges()
            edge_ids = []
            
            for i, (s, d) in enumerate(zip(src.cpu().numpy(), dst.cpu().numpy())):
                if (s == edge[0] and d == edge[1]) or (s == edge[1] and d == edge[0]):
                    edge_ids.append(i)
            
            if edge_ids:
                temp_graph.remove_edges(edge_ids)
        
        # Add self-loops to handle zero in-degree nodes
        temp_graph = dgl.add_self_loop(temp_graph)
        
        return temp_graph

    def _compute_attack_loss(self, perturbed_graph):
        """
        Compute the attack loss on a perturbed graph.
        This measures how much the surrogate model's performance degrades.
        Uses proper bi-level optimization as in the original Mettack paper.
        """
        # Create a temporary surrogate model copy
        temp_surrogate = copy.deepcopy(self.surrogate)
        temp_surrogate.train()
        
        # Fine-tune on perturbed graph for a few steps (bi-level optimization)
        optimizer = optim.Adam(temp_surrogate.parameters(), lr=0.01)
        
        for _ in range(5):  # Reduced from 10 for efficiency but still doing proper retraining
            optimizer.zero_grad()
            logits = temp_surrogate(perturbed_graph, self.features)
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set - higher loss means better attack
        temp_surrogate.eval()
        with torch.no_grad():
            val_logits = temp_surrogate(perturbed_graph, self.features)
            val_loss = F.cross_entropy(val_logits[self.val_mask], self.labels[self.val_mask])
        
        return val_loss.item()

    def _evaluate(self, poisoned_graph):
        """
        Evaluates GCN accuracy before/after poisoning, etc.
        """
        metrics = {}
        
        # Evaluate surrogate on clean graph
        self.surrogate.eval()
        with torch.no_grad():
            clean_logits = self.surrogate(self.graph, self.features)
            clean_acc = self._compute_accuracy(clean_logits[self.test_mask], 
                                             self.labels[self.test_mask])
            metrics['clean_test_acc'] = clean_acc
        
        # Train new model on poisoned graph
        poisoned_model = GCN(self.features.shape[1], 
                           int(self.labels.max().item()) + 1).to(self.device)
        optimizer = optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)
        
        poisoned_model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            logits = poisoned_model(poisoned_graph, self.features)
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            optimizer.step()
        
        # Evaluate poisoned model
        poisoned_model.eval()
        with torch.no_grad():
            poisoned_logits = poisoned_model(poisoned_graph, self.features)
            poisoned_acc = self._compute_accuracy(poisoned_logits[self.test_mask], 
                                                self.labels[self.test_mask])
            metrics['poisoned_test_acc'] = poisoned_acc
        
        metrics['accuracy_drop'] = clean_acc - poisoned_acc
        metrics['num_perturbations'] = len(self.modified_edges)
        
        
        return metrics

    def _compute_accuracy(self, logits, labels):
        """Helper function to compute accuracy."""
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        return correct / len(labels)
