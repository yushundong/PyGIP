#!/bin/bash
echo "Running all PyGIP scripts..."

python -m PyGIP.examples.run_cora_attack
python -m PyGIP.examples.run_cora_defense
python -m PyGIP.examples.run_citeseer_attack
python -m PyGIP.examples.run_citeseer_defense
python -m PyGIP.examples.run_pubmed_attack
python -m PyGIP.examples.run_pubmed_defense
python -m PyGIP.examples.run_my_custom_attack
python -m PyGIP.examples.run_my_custom_defense

echo "Done!"
