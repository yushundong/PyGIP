```bash
conda env create -f environment.yml -n gnnip
conda activate gnnip
pip install dgl -f https://data.dgl.ai/wheels/repo.html #due to dgl issues, unfortunately we have to install this dgl 2.2.1 manually.

# Under the GNNIP directory
export PYTHONPATH=`pwd`

# Quick testing
python3 example/example.py

```
## Attack0-Watermark

### 1. Attack0-Watermark on Cora

Run

```
python Defense.py
```

Follow the instructions to enter 1 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
1
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```
We present the sample log as follows

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:07<00:00,  6.53it/s]
Marked Acc: 0.7890
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 145.31it/s]
Final results
Non-Marked Acc: 0.1400, Marked Acc: 0.4980, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 61.88it/s]
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:05<00:00, 38.38it/s]
========================Final results:=========================================
Fidelity: 0.8598356694055099, Accuracy: 0.7878202029966167
Watermark Graph - Accuracy: 0.24
```

### 2. Attack0-Watermark on Citeseer

Run

```
python Defense.py
```

Follow the instructions to enter 1 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
1
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```
We present the sample log as follows

```
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:09<00:00,  5.03it/s]
Marked Acc: 0.7080
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 96.68it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4860, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:11<00:00, 18.11it/s]
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:17<00:00, 11.31it/s]
========================Final results:=========================================
Fidelity: 0.8484011054086064, Accuracy: 0.7090406632451638
Watermark Graph - Accuracy: 0.06
```

### 3. Attack0-Watermark on PubMed

Run

```
python Defense.py
```

Follow the instructions to enter 1 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
1
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```
We present the sample log as follows

```
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.49it/s]
Marked Acc: 0.7800
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 136.56it/s]
Final results
Non-Marked Acc: 0.2400, Marked Acc: 0.5640, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:07<00:00, 26.09it/s]
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:14<00:00, 13.71it/s]
========================Final results:=========================================
Fidelity: 0.9202077431539188, Accuracy: 0.7921219479293133
Watermark Graph - Accuracy: 0.32
```

## Attack1-Watermark

### 1. Attack1-Watermark on Cora

Run

```
python Defense.py
```

Follow the instructions to enter 2 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
2
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```
We present the sample log as follows

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:07<00:00,  7.13it/s]
Marked Acc: 0.7740
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 150.18it/s]
Final results
Non-Marked Acc: 0.2000, Marked Acc: 0.5910, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 56.25it/s]
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
Net_shadow(
  (layer1): GraphConv(in=1433, out=16, normalization=both, activation=None)
  (layer2): GraphConv(in=16, out=7, normalization=both, activation=None)
)
===================Model Extracting================================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:02<00:00, 66.87it/s]
Fidelity: 0.7750242954324587, Accuracy: 0.7769679300291545
Watermark Graph - Accuracy: 0.12
```

### 2. Attack1-Watermark on Citeseer

Run

```
python Defense.py
```

Follow the instructions to enter 2 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
2
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```
We present the sample log as follows

```
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:09<00:00,  5.20it/s]
Marked Acc: 0.7180
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 89.53it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4220, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:11<00:00, 18.16it/s]
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
Net_shadow(
  (layer1): GraphConv(in=3703, out=16, normalization=both, activation=None)
  (layer2): GraphConv(in=16, out=6, normalization=both, activation=None)
)
===================Model Extracting================================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:07<00:00, 26.10it/s]
Fidelity: 0.6955547254389242, Accuracy: 0.6936869630183041
Watermark Graph - Accuracy: 0.18
```

### 3. Attack1-Watermark on PubMed

Run

```
python Defense.py
```

Follow the instructions to enter 2 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
2
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```
We present the sample log as follows

```
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.43it/s]
Marked Acc: 0.7660
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 196.00it/s]
Final results
Non-Marked Acc: 0.3000, Marked Acc: 0.7190, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:07<00:00, 25.77it/s]
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
Net_shadow(
  (layer1): GraphConv(in=500, out=16, normalization=both, activation=None)
  (layer2): GraphConv(in=16, out=3, normalization=both, activation=None)
)
===================Model Extracting================================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:07<00:00, 25.27it/s]
Fidelity: 0.813971783710075, Accuracy: 0.8138668904389783
Watermark Graph - Accuracy: 0.44
```

## Attack2-Watermark

### 1. Attack2-Watermark on Cora

Run

```
python Defense.py
```

Follow the instructions to enter 3 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
3
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```
We present the sample log as follows

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:07<00:00,  6.48it/s]
Marked Acc: 0.7900
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 164.84it/s]
Final results
Non-Marked Acc: 0.1400, Marked Acc: 0.3080, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 58.55it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:09<00:00, 20.58it/s]
Fidelity: 0.793, Accuracy: 0.758
Watermark Graph - Accuracy: 0.1
```

### 2. Attack2-Watermark on Citeseer

Run

```
python Defense.py
```

Follow the instructions to enter 3 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
3
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```
We present the sample log as follows

```
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:09<00:00,  5.34it/s]
Marked Acc: 0.6940
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 107.36it/s]
Final results
Non-Marked Acc: 0.1200, Marked Acc: 0.3230, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:11<00:00, 17.95it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:15<00:00, 13.08it/s]
Fidelity: 0.698, Accuracy: 0.587
Watermark Graph - Accuracy: 0.22
```

### 3. Attack3-Watermark on PubMed

Run

```
python Defense.py
```

Follow the instructions to enter 3 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
3
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```
We present the sample log as follows

```
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  8.83it/s]
Marked Acc: 0.7730
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 203.23it/s]
Final results
Non-Marked Acc: 0.3600, Marked Acc: 0.7300, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:07<00:00, 26.79it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [3:03:21<00:00, 55.01s/it]
Fidelity: 0.887, Accuracy: 0.782
Watermark Graph - Accuracy: 0.34
```

## Attack3-Watermark

### 1. Attack3-Watermark on Cora

Run

```
python Defense.py
```

Follow the instructions to enter 4 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
4
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```
We present the sample log as follows

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:08<00:00,  5.83it/s]
Marked Acc: 0.7840
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 137.76it/s]
Final results
Non-Marked Acc: 0.1600, Marked Acc: 0.4780, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 51.53it/s]
generated_train_mask 1989
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
  0%|                                                                                                             | 0/300 [00:00<?, ?it/s]/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/gnn_mea.py:766: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /opt/conda/conda-bld/pytorch_1711403233856/work/aten/src/ATen/native/IndexingUtils.h:27.)
  loss_a = F.nll_loss(logp_a[generated_train_mask],
/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/gnn_mea.py:767: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /opt/conda/conda-bld/pytorch_1711403233856/work/aten/src/ATen/native/IndexingUtils.h:27.)
  generated_labels[generated_train_mask])
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /opt/conda/conda-bld/pytorch_1711403233856/work/aten/src/ATen/native/IndexingUtils.h:27.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:05<00:00, 51.62it/s]
Fidelity: 0.7689274447949527, Accuracy: 0.8280757097791798
Watermark Graph - Accuracy: 0.14
```

### 2. Attack3-Watermark on Citeseer

Run

```
python Defense.py
```

Follow the instructions to enter 4 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
4
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```
We present the sample log as follows

```
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:10<00:00,  4.67it/s]
Marked Acc: 0.7000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 95.75it/s]
Final results
Non-Marked Acc: 0.1400, Marked Acc: 0.3450, Watermark Acc: 0.9600
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:11<00:00, 17.85it/s]
generated_train_mask 1846
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
Traceback (most recent call last):
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 444, in <module>
    defense.watermark_attack(Citeseer(), attack_name, dataset_name)
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 268, in watermark_attack
    attack.attack()
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/gnn_mea.py", line 704, in attack
    generated_train_mask[i] = 0
IndexError: index 1846 is out of bounds for dimension 0 with size 1846
```

### 3. Attack3-Watermark on PubMed

Run

```
python Defense.py
```

Follow the instructions to enter 4 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
4
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```
We present the sample log as follows

```
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.03it/s]
Marked Acc: 0.7750
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 202.34it/s]
Final results
Non-Marked Acc: 0.4200, Marked Acc: 0.6750, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:07<00:00, 26.74it/s]
Traceback (most recent call last):
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 447, in <module>
    defense.watermark_attack(PubMed(), attack_name, dataset_name)
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 302, in watermark_attack
    attack.attack()
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/gnn_mea.py", line 618, in attack
    fileObject = open('./gnnip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './gnnip/data/attack3_shadow_graph/pubmed/target_graph_index.txt'
```

## Attack4-Watermark

### 1. Attack4-Watermark on Cora

Run

```
python Defense.py
```

Follow the instructions to enter 5 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
5
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```
We present the sample log as follows

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:08<00:00,  5.98it/s]
Marked Acc: 0.7890
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 166.12it/s]
Final results
Non-Marked Acc: 0.2000, Marked Acc: 0.3170, Watermark Acc: 1.0000
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
1408 1408
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:05<00:00, 59.47it/s]
Fidelity: 0.010830324909747292, Accuracy: 0.15433212996389892
Watermark Graph - Accuracy: 0.14
```

### 2. Attack4-Watermark on Citeseer

Run

```
python Defense.py
```

Follow the instructions to enter 5 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
5
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```
We present the sample log as follows

```
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:10<00:00,  4.58it/s]
Marked Acc: 0.7070
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 97.85it/s]
Final results
Non-Marked Acc: 0.2400, Marked Acc: 0.1940, Watermark Acc: 1.0000
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
2325 2325
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:10<00:00, 29.91it/s]
Fidelity: 0.08691358024691358, Accuracy: 0.12493827160493827
Watermark Graph - Accuracy: 0.2
```

### 3. Attack4-Watermark on PubMed

Run

```
python Defense.py
```

Follow the instructions to enter 5 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
5
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```
We present the sample log as follows

```
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.02it/s]
Marked Acc: 0.7730
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 182.80it/s]
Final results
Non-Marked Acc: 0.3600, Marked Acc: 0.5980, Watermark Acc: 1.0000
Traceback (most recent call last):
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 447, in <module>
    defense.watermark_attack(PubMed(), attack_name, dataset_name)
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 307, in watermark_attack
    attack.attack()
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/gnn_mea.py", line 800, in attack
    fileObject = open('./gnnip/data/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './gnnip/data/pubmed/target_graph_index.txt'
```

## Attack5-Watermark

### 1. Attack5-Watermark on Cora

Run

```
python Defense.py
```

Follow the instructions to enter 6 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
6
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```
We present the sample log as follows

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:08<00:00,  6.11it/s]
Marked Acc: 0.7770
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 131.62it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4630, Watermark Acc: 1.0000
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:04<00:00, 61.87it/s]
Fidelity: 0.20126353790613719, Accuracy: 0.13357400722021662
Watermark Graph - Accuracy: 0.18
```

### 2. Attack5-Watermark on Citeseer

Run

```
python Defense.py
```

Follow the instructions to enter 6 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
6
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```
We present the sample log as follows

```
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:11<00:00,  4.40it/s]
Marked Acc: 0.6990
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 106.87it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4000, Watermark Acc: 1.0000
/home/syx/.conda/envs/gnnip2/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:10<00:00, 28.05it/s]
Fidelity: 0.15604938271604937, Accuracy: 0.19358024691358025
Watermark Graph - Accuracy: 0.14
```

### 3. Attack5-Watermark on PubMed

Run

```
python Defense.py
```

Follow the instructions to enter 6 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
6
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```
We present the sample log as follows

```
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.95it/s]
Marked Acc: 0.7730
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 176.38it/s]
Final results
Non-Marked Acc: 0.3200, Marked Acc: 0.7020, Watermark Acc: 1.0000
Traceback (most recent call last):
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 447, in <module>
    defense.watermark_attack(PubMed(), attack_name, dataset_name)
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/Defense.py", line 313, in watermark_attack
    attack.attack()
  File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/core_algo/gnn_mea.py", line 997, in attack
    fileObject = open('./gnnip/data/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './gnnip/data/pubmed/target_graph_index.txt'
```
