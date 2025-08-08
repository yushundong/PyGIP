# 📘 Reproduction of "Making Watermark Survive Model Extraction Attacks in GNNs" (NeurIPS 2023)

This repository reproduces the experiments from the paper:
> *Making Watermark Survive Model Extraction Attacks in Graph Neural Networks* (Wang et al., 2023)

---

## 🔧 Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Experiment Execution Guide

| Experiment | Script | Description | Output |
|-----------|--------|-------------|--------|
| M1 (SNNL) Watermarked Model | `experiment.py` | Trains model with SNNL on ENZYMES | `watermarked_model_m1.pth`, key files |
| M0 (Strawman) Model | `m0_baseline.py` | Baseline model with no SNNL | `watermarked_model_m0.pth` |
| Watermark Verification | `verifywatermark.py` | Tests accuracy of watermark on either M0 or M1 | Printed accuracy |
| Query Attack | `attacks/query_attack.py` | Simulates query-based mimic model | Logs + accuracy |
| Distill Attack | `attacks/distill_attack.py` | Knowledge distillation mimic model | Logs + accuracy |
| Fine-tune Attack | `attacks/finetune_attack.py` | Attacker retrains model on new data | Logs + accuracy |

To switch between verifying M0 or M1, change the `use_model = "M1"` line in `verifywatermark.py`.

---

## 📊 Results Reproduced

See `results/` folder for:

- `m0_m1_comparison.csv`: Main table in paper
- `m1_enzymes_accuracy.txt`: Training + verification log

Sample table:

| Method | No Attack | Query | Distill | Fine-tune |
|--------|-----------|--------|---------|-----------|
| M0     | 94.3%     | 31.2%  | 27.1%   | 42.1%     |
| M1     | 98.1%     | 82.3%  | 75.6%   | 79.3%     |

---

## 📌 Citation

```
@inproceedings{wang2023watermark,
  title={Making Watermark Survive Model Extraction Attacks in Graph Neural Networks},
  author={Wang, Mengnan and Jin, Xiaojun and others},
  booktitle={NeurIPS},
  year={2023}
}
```
