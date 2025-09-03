@echo off
echo Running all PyGIP scripts...

:: Go into the PyGIP folder
cd /d "C:\Users\hp\PYGIP\PyGIP"

python -m examples.run_cora_attack
python -m examples.run_cora_defense
python -m examples.run_citeseer_attack
python -m examples.run_citeseer_defense
python -m examples.run_pubmed_attack
python -m examples.run_pubmed_defense
python -m examples.run_my_custom_attack
python -m examples.run_my_custom_defense

echo Done!
pause



