```bash
conda env create -f environment.yml -n gnnip
conda activate gnnip
pip install dgl -f https://data.dgl.ai/wheels/repo.html #due to dgl issues, unfortunately we have to install this dgl 2.2.1 manually.

# Under the GNNIP directory
export PYTHONPATH=`pwd`

# Quick testing
python3 example/example.py

```

TODO
