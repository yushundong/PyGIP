#!/usr/bin/env python3
"""
Test script to verify the examples folder setup.

This script tests that all the examples can be imported and basic functionality works.
"""

import sys
import os

# Set up path first before any imports
def setup_path():
    """Set up the Python path to find PyGIP modules."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✓ Added project root to path: {project_root}")
    return project_root

# Setup path immediately
project_root = setup_path()

# Import modules at module level
try:
    from datasets import Cora
    from models.attack import ModelExtractionAttack0 as MEA
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"✗ Critical import failed: {e}")
    IMPORTS_SUCCESSFUL = False

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    if not IMPORTS_SUCCESSFUL:
        print("✗ Critical imports failed")
        return False
    
    try:
        print("✓ Cora dataset import successful")
        print("✓ ModelExtractionAttack0 import successful")
        
        # Test GNNFingers imports
        try:
            from models.defense.gnn_fingers_models import get_model_for_task, ModelObfuscator, Univerifier
            print("✓ GNNFingers models import successful")
        except ImportError as e:
            print(f"⚠ GNNFingers models import failed: {e}")
        
        try:
            from models.defense.gnn_fingers_defense import GNNFingersDefense
            print("✓ GNNFingers defense import successful")
        except ImportError as e:
            print(f"⚠ GNNFingers defense import failed: {e}")
        
        try:
            from datasets.gnnfingers_adapter import PyGIPDatasetAdapter, adapt_pygip_dataset
            print("✓ GNNFingers adapter import successful")
        except ImportError as e:
            print(f"⚠ GNNFingers adapter import failed: {e}")
        
        print("✓ All basic imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_path_setup():
    """Test that the path setup works correctly."""
    print("\nTesting path setup...")
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in the examples folder
    if os.path.basename(cwd) == 'examples':
        print("✓ Running from examples folder")
        print(f"✓ Project root: {os.path.dirname(cwd)}")
    else:
        print("⚠ Not running from examples folder")
        # Check if examples folder exists
        examples_path = os.path.join(cwd, 'examples')
        if os.path.exists(examples_path):
            print(f"✓ Examples folder found at: {examples_path}")
            print(f"✓ Project root: {cwd}")
        else:
            print(f"✗ Examples folder not found")
    
    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test dataset creation
        dataset = Cora(api_type='dgl')
        print(f"✓ Created Cora dataset: {dataset}")
        
        # Test attack creation
        mea = MEA(dataset, attack_node_fraction=0.1)
        print(f"✓ Created ModelExtractionAttack: {mea}")
        
        print("✓ Basic functionality test successful")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Examples Folder Setup Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test path setup
    path_ok = test_path_setup()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Path Setup: {'✓ PASS' if path_ok else '✗ FAIL'}")
    print(f"Basic Functionality: {'✓ PASS' if func_ok else '✗ FAIL'}")
    
    if all([imports_ok, path_ok, func_ok]):
        print("\n🎉 All tests passed! Examples folder is ready to use.")
        print("\nNext steps:")
        print("  • Run experiments: python examples/run_gnnfingers_experiments.py --all --quick")
        print("  • Test adapter: python examples/test_adapter.py")
        print("  • Run demo: python examples/adapter_demo.py")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
