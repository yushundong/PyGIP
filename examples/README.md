#Fingerprinting Graph Neural Networks

Steps

1. Create virtual env. Activate it.
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Create folders

   ```bash
   mkdir -p  data models fingerprints plots
   ```

### GNN task types: Node Classification, Link Prediction, Graph Classification

For Node Classification (NC): \
 &emsp;Folder name: node_class/ \
 &emsp;Filename Suffix: \*\_nc.\*

For Link Prediction (LP): \
 &emsp;Folder name: link_pred/ \
 &emsp;Filename Suffix: \*\_lp.\*

For Graph Classification (GC): \
 &emsp;Folder name: graph_class/ \
 &emsp;Filename Suffix: \*\_gc.\*

&emsp;Example: \
 &emsp;`bash 
    python node_class/train_nc.py ` \
 &emsp;`bash 
    python link_pred/train_lp.py ` \
 &emsp;`bash 
    python graph_class/train_gc.py `

### For node classification task on Cora dataset (GCN arch)

```bash
python node_class/train_nc.py
```

```bash
python node_class/fine_tune_pirate_nc.py
```

```bash
python node_class/distill_students_nc.py
```

```bash
python node_class/train_unrelated_nc.py
```

```bash
python node_class/fingerprint_generator_nc.py
```

```bash
python node_class/generate_univerifier_dataset_nc.py
```

```bash
python train_univerifier.py --dataset fingerprints/univerifier_dataset_nc.pt --fingerprints_path fingerprints/fingerprints_nc.pt --out fingerprints/univerifier_nc.pt
```

```bash
python node_class/eval_verifier_nc.py
```

Follow similar approach as Node Classification for Link Prediction on Citeseer dataset (GCN arch) and Graph Classification on ENZYMES dataset (Graphsage arch).

Change argument paths for LP and GC for training univerifier

```bash
python train_univerifier.py --dataset fingerprints/univerifier_dataset_lp.pt --fingerprints_path fingerprints/fingerprints_lp.pt --out fingerprints/univerifier_lp.pt
```

```bash
python train_univerifier.py --dataset fingerprints/univerifier_dataset_gc.pt --fingerprints_path fingerprints/fingerprints_gc.pt --out fingerprints/univerifier_gc.pt
```

To evaluate suspect GNNs for NC tasks
 ```bash
 python node_class/make_suspect_neg_nc.py
 ```
 ```bash
 python node_class/score_suspect_nc.py  --suspect_pt models/suspects/neg_nc_seed9999.pt  --suspect_meta models/suspects/neg_nc_seed9999.json
 ```

