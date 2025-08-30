#!/bin/bash

echo "Running all PyGIP scripts..."

# Ensure the script runs in its own directory
cd "$(dirname "$0")"

# Run each Python script
python3 run_cora_attack.py
python3 run_cora_defense.py
python3 run_citeseer_attack.py
python3 run_citeseer_defense.py
python3 run_pubmed_attack.py
python3 run_pubmed_defense.py
python3 run_my_custom_attack.py
python3 run_my_custom_defense.py

echo "Done!"
