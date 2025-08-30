@echo off
echo Running all PyGIP scripts...

REM Ensure we are in the examples directory
cd /d %~dp0

REM Run each Python script with .py extension
python run_cora_attack.py
python run_cora_defense.py
python run_citeseer_attack.py
python run_citeseer_defense.py
python run_pubmed_attack.py
python run_pubmed_defense.py
python run_my_custom_attack.py
python run_my_custom_defense.py

echo Done!
pause


