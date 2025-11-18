# Emotion Classification & Synthetic Emotion Generation Toolkit (emoBERT)

This repository contains a complete pipeline for **multi-label emotion classification** using **Plutchik’s 8-emotion model**, including:

- Training on **GoEmotions + XED + synthetic Claude data**
- Threshold tuning via **per-emotion PR curves**
- Models based on `DistilBERT` and `RoBERTa`
- A unified **prediction script** for inference
- Synthetic dataset generation powered by LLMs (Claude)

All scripts assume the following directory structure:

```
emoBERT/
│
├── emoBERT.py
├── emoRoBERTa_colab.py
├── emoPredict_roberta.py
├── emoGen_claude.py
│
├── synthetic_claude.csv
├── synthetic_data.csv        # optional older version
│
├── models/
│   ├── best_model.pt
│   └── ...
│
└── results/
    ├── test_results.json
    └── ...
```

---

# 1. Models Used
This project trains classifiers using:

- **DistilBERT** (efficient, CPU-friendly)
- **RoBERTa-base** (stronger model, GPU recommended)

Both models predict **8 independent emotion probabilities**:

```
joy, sadness, anger, fear,
trust, disgust, surprise, anticipation
```

Labels are converted from **GoEmotions** fine-grained taxonomy and aligned with **XED** (already in Plutchik space).

---

# 2. Script Overview

---

## 📘 emoBERT.py — CPU-Friendly Training Script

A full training pipeline using **DistilBERT**:

- Loads & maps **GoEmotions → Plutchik**
- Loads **XED** (80/10/10 split)
- Loads **synthetic_claude.csv**
- Trains using:
  - `BCEWithLogitsLoss`
  - `AdamW (2e-5)`
  - Batch size **16**
  - Linear warmup scheduler
- Performs **per-emotion PR-curve threshold tuning**
- Saves:
  - `models/best_model.pt`
  - `results/test_results.json`

### Run:
```bash
python emoBERT.py --epochs 3 --batch_size 16
```

### Output:
```
models/best_model.pt
results/test_results.json
```

---

## 📗 emoRoBERTa_colab.py — GPU-Optimized RoBERTa Training (Colab)

A faster & more powerful version using **RoBERTa-base**.

Provides:

- HuggingFace `AutoModelForSequenceClassification`
- Mixed precision when available (`torch.cuda.amp`)
- Larger batch sizes possible on GPU
- Same preprocessing & synthetic data pipeline as emoBERT.py
- Optional full threshold-tuning integration

### Run in Colab:
```bash
!python emoRoBERTa_colab.py --epochs 3 --batch_size 16
```

Produces:
```
models/best_model.pt
results/test_results.json
```

---

## 📙 emoPredict_roberta.py — Inference Script

Standalone prediction script for:

- Moodlog app  
- Django backend  
- Jupyter notebooks  
- Colab  

Features:

- Loads `best_model.pt`
- Loads `optimal_thresholds` from `results/test_results.json`
- Applies **per-emotion thresholds**
- Outputs JSON-friendly predictions

### Example:
```bash
python emoPredict_roberta.py --text "I feel nervous but excited for tomorrow."
```

Output:
```json
{
  "joy": 0.12,
  "sadness": 0.01,
  "anger": 0.00,
  "fear": 0.44,
  "trust": 0.05,
  "disgust": 0.00,
  "surprise": 0.27,
  "anticipation": 0.61,
  "predicted_labels": ["fear", "surprise", "anticipation"]
}
```

---

## 📕 emoGen_claude.py — Synthetic Emotion Data Generator

Utility script for generating **high-quality synthetic training samples** using Claude (or any LLM).  
Produces CSV rows in the exact format required by the training scripts.

Format:
```
text, joy_label, sadness_label, anger_label, fear_label,
      trust_label, disgust_label, surprise_label, anticipation_label
```

### Run:
```bash
python emoGen_claude.py --num_samples 5000 --output synthetic_claude.csv
```

---

# 3. Synthetic Data Format

Synthetic CSVs must include:

```
text, joy_label, sadness_label, anger_label, fear_label,
      trust_label, disgust_label, surprise_label, anticipation_label
```

Place files in the repository root:
```
emoBERT/
    synthetic_claude.csv
```

---

# 4. Model & Results Files

### 📁 models/
Holds checkpoints:
- `best_model.pt`

### 📁 results/
Holds evaluation JSON:
- `test_results.json` containing:
  - validation macro F1
  - test macro & micro F1
  - **per-emotion thresholds**
  - tuned metrics

Example:
```json
{
  "test_macro_f1": 0.5883,
  "tuned_test_macro_f1": 0.6421,
  "optimal_thresholds": {
    "joy": 0.34,
    "sadness": 0.47,
    "anger": 0.52,
    "fear": 0.41
  }
}
```

---

# 5. Installation

```bash
pip install torch transformers datasets scikit-learn pandas tqdm
```

GPU/Colab (optional):
```bash
pip install accelerate
```

---

# 6. Recommended Workflow

- Local CPU training → `emoBERT.py`
- Colab GPU training → `emoRoBERTa_colab.py`
- Generate synthetic data → `emoGen_claude.py`
- Production inference → `emoPredict_roberta.py`

---

# 7. Threshold Tuning Notes

- Default threshold = **0.5**
- PR-curve tuning finds an **optimal threshold per emotion**
- Stored in results JSON
- Applied automatically during inference

---

# 8. Citations

If you use GoEmotions:
```
Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", 2020.
```

If you use XED:
```
Öhman et al., "XED: A Multilingual Dataset for Cross-lingual Emotion Detection", 2020.
```

---

# 9. License
MIT License.

---

# 10. Contact
For integration help, dataset expansion, or model troubleshooting, feel free to reach out.
