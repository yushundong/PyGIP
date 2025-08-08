from .RandomWM import RandomWM

# New GNNFingers defense import - use lazy import to avoid circular dependency
# from .gnn_fingers_defense import GNNFingersDefense

__all__ = [
    'RandomWM',
    'GNNFingersDefense'
]

DEFENSE_REGISTRY = {
    'random_watermark': RandomWM,
    'randomwm': RandomWM,
    'gnn_fingers': 'GNNFingersDefense',  # Use string for lazy loading
    'fingerprinting': 'GNNFingersDefense',  # Use string for lazy loading
}


def get_defense(defense_name: str):
    """
    Factory function to get defense by name.
    
    Args:
        defense_name: Name of the defense mechanism
        
    Returns:
        Defense class
        
    Raises:
        ValueError: If defense is not found
    """
    if defense_name.lower() in DEFENSE_REGISTRY:
        defense_class = DEFENSE_REGISTRY[defense_name.lower()]
        if isinstance(defense_class, str):
            # Lazy import for GNNFingersDefense to avoid circular dependency
            from .gnn_fingers_defense import GNNFingersDefense
            return GNNFingersDefense
        return defense_class
    else:
        available = list(DEFENSE_REGISTRY.keys())
        raise ValueError(f"Defense '{defense_name}' not found. Available defenses: {available}")