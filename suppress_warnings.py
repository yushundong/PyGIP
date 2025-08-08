#!/usr/bin/env python3
"""
Script to suppress PyTorch Geometric CUDA warnings for CPU-only installations.
Run this before importing torch_geometric to suppress the warnings.
"""

import warnings
import os

# Suppress specific PyTorch Geometric CUDA warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*torch-cluster.*")
warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
warnings.filterwarnings("ignore", message=".*torch-sparse.*")

# Set environment variable to suppress warnings
os.environ['PYTORCH_GEOMETRIC_SUPPRESS_WARNINGS'] = '1'

print("PyTorch Geometric CUDA warnings suppressed.")
print("You can now import torch_geometric without warnings.")
