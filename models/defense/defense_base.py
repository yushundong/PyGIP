import torch
from abc import ABC, abstractmethod
import os

def get_device():
    """Helper function to get the appropriate device."""
    if os.getenv('PYGIP_DEVICE') and torch.cuda.is_available():
        return torch.device(os.getenv('PYGIP_DEVICE'))
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class BaseDefense(ABC): # this classs serves as the base for all defenses
    """
    Abstract base class for all defense implementations, as per PyGIP guidelines.
    """
    def __init__(self, dataset, attack_node_fraction=None, device=None):
        self.device = device if device else get_device() # it automatically determine and set the computing device (e.g., 'cuda' or 'cpu').
        print(f"INFO: Defense class initialized on device: '{self.device}'")

        # graph data from the dataset wrapper
        self.dataset = dataset
        self.graph_data = dataset.graph_data
        
        # meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        #  Store parameters relevant to attacks.
        self.attack_node_fraction = attack_node_fraction

    @abstractmethod
    def defend(self):
        """
        Main defense logic must be implemented by subclasses.
        """
        pass