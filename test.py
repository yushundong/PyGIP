import sys
import os

def main():
    """Redirect users to the examples folder."""
    print("=" * 80)
    print("PyGIP Test Suite - Moved to Examples Folder")
    print("=" * 80)
    print()
    print("This test.py file has been reorganized for better project structure.")
    print("All experiment scripts are now located in the examples/ folder.")
    print()
    print("📁 Available Scripts:")
    print("  • examples/run_gnnfingers_experiments.py - Main experiment runner")
    print("  • examples/adapter_demo.py - GNNFingers adapter demonstration")
    print("  • examples/test_adapter.py - Adapter functionality testing")
    print("  • examples/gnn_fingers_example.py - Basic GNNFingers example")
    print()
    print("🚀 Quick Start:")
    print("  # Run all experiments in quick mode")
    print("  python examples/run_gnnfingers_experiments.py --all --quick")
    print()
    print("  # Run all experiments in full mode")
    print("  python examples/run_gnnfingers_experiments.py --all --full")
    print()
    print("  # Run specific task")
    print("  python examples/run_gnnfingers_experiments.py --task node_classification --dataset Cora --model GCN --quick")
    print()
    print("📖 For detailed documentation:")
    print("  examples/README.md")
    print()
    print("=" * 80)
    print("Redirecting to examples folder...")
    print("=" * 80)
    
    # Check if examples folder exists
    examples_path = os.path.join(os.path.dirname(__file__), 'examples')
    if os.path.exists(examples_path):
        print(f"Examples folder found at: {examples_path}")
        print("Please use the scripts in the examples folder for all experiments.")
    else:
        print("ERROR: Examples folder not found!")
        print("Please ensure the examples folder exists in the project root.")
    
    print()
    print("For help with specific commands:")
    print("  python examples/run_gnnfingers_experiments.py --help")


if __name__ == "__main__":
    main()
