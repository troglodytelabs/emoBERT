# emoBERT - Multi-Label Emotion Classification Toolkit

A complete pipeline for **multi-label emotion classification** using **Plutchik's 8-emotion model**, designed to power affective computing applications.

## Purpose & Vision

emoBERT is an **affective computing** project designed to power [moodlog.io](https://github.com/troglodytelabs/moodlog.io) - a mood tracking application that helps users understand their emotions over time. The emoBERT model will:

1. **Analyze journal entries** - Process short text entries from users logging their daily moods
2. **Predict base emotions** - Identify Plutchik's 8 emotions present in the text
3. **Suggest dyadic relationships** - Combine emotions into more interpretable concepts (e.g., joy + trust = love)
4. **Track emotional patterns** - Help users understand mood trends and emotional health over time

## Core Emotions (Plutchik)

```
joy, sadness, anger, fear, trust, disgust, surprise, anticipation
```

The emotion order is critical - all label vectors, model outputs, and thresholds follow this exact sequence.

## Features

- Training on **GoEmotions + XED + synthetic Claude data**
- Threshold tuning via **per-emotion PR curves**
- Models based on `DistilBERT` and `RoBERTa`
- Unified **prediction scripts** for inference with dyad detection
- Synthetic dataset generation powered by Claude API

## Repository Structure

```
emoBERT/
├── src/
│   ├── training/
│   │   ├── emoBERT.py           # CPU training script (DistilBERT)
│   │   ├── emoBERT_colab.py     # GPU training script (DistilBERT, Colab)
│   │   └── emoRoBERTa_colab.py  # GPU training script (RoBERTa, Colab) - RECOMMENDED
│   ├── inference/
│   │   ├── emoPredict.py        # Inference with DistilBERT + dyad detection
│   │   └── emoPredict_roberta.py # Inference with RoBERTa - RECOMMENDED
│   └── data_generation/
│       ├── emoGen.py            # Synthetic data generation (templates/backtrans)
│       ├── emoGen_claude.py     # Synthetic data generation using Claude API
│       └── emoGen_claude_v2.py  # Alternate synthetic data generator (async)
├── data/
│   └── synthetic/
│       ├── synthetic_claude.csv        # Generated synthetic training data
│       └── synthetic_claude_merged.csv # Combined synthetic datasets
├── models/                    # Model checkpoints and results
│   ├── best_model.pt          # Best trained model checkpoint
│   └── test_results.json      # Training metrics and optimal thresholds
├── README.md
└── CLAUDE.md                  # Detailed AI assistant guide
```

---

## Performance Benchmarks

Current best results (RoBERTa with tuned thresholds):
- **Test Macro F1: 0.6277**
- **Test Micro F1: 0.6838**
- Best emotion: joy (0.83 F1)
- Hardest emotion: fear (0.54 F1)

**Goal: All emotions >67% F1**

---

## Models Used

This project trains classifiers using:

- **DistilBERT** (efficient, CPU-friendly)
- **RoBERTa-base** (stronger model, GPU recommended) - **RECOMMENDED**

Both models predict **8 independent emotion probabilities** using:
- `RobertaModel → dropout(0.15) → Linear(768 → 8) → sigmoid`

Labels are converted from **GoEmotions** fine-grained taxonomy and aligned with **XED** (already in Plutchik space).

---

## Quick Start

### Installation

```bash
pip install torch transformers datasets scikit-learn pandas tqdm

# For Colab GPU
pip install accelerate

# For synthetic data generation
pip install anthropic
```

### Training (RoBERTa - Recommended)

```bash
python src/training/emoRoBERTa_colab.py
```

Default configuration:
```python
config = {
    "epochs": 12,
    "batch_size": 64,
    "learning_rate": 2e-5,
    "dropout": 0.15,
    "warmup_ratio": 0.1,
    "weight_decay": 0.02,
    "label_smoothing": 0.1,
    "layerwise_lr_decay": 0.85,
    "max_length": 128,
    "patience": 5,  # Early stopping
}
```

### Inference

**Single text prediction**:
```bash
python src/inference/emoPredict_roberta.py --text "I feel nervous but excited for tomorrow."
```

**Interactive mode**:
```bash
python src/inference/emoPredict_roberta.py --interactive
```

**Batch processing**:
```bash
python src/inference/emoPredict_roberta.py --file input.txt --verbose
```

Output example:
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

### Generate Synthetic Data

```bash
# Generate for all rare emotions
python src/data_generation/emoGen_claude.py --all_rare --count 10000 --api_key sk-ant-...

# Generate for specific emotion
python src/data_generation/emoGen_claude.py --emotion fear --count 15000
```

Required API key: Set `ANTHROPIC_API_KEY` env var or use `--api_key` flag.

---

## Synthetic Data Format

Synthetic CSVs must include:

```csv
text,joy_label,sadness_label,anger_label,fear_label,trust_label,disgust_label,surprise_label,anticipation_label
"Example text",1,0,0,0,1,0,0,0
```

Place files in `data/synthetic/` directory.

---

## Threshold Tuning

- Default threshold = **0.5**
- PR-curve tuning finds an **optimal threshold per emotion**
- Optimal thresholds stored in `models/test_results.json`
- Thresholds typically range from 0.53 to 0.79 depending on emotion
- Applied automatically during inference

---

## Roadmap / Next Steps

1. **Dyad interpretation layer** - Build on `emoPredict.py`'s dyad detection to provide human-readable emotional insights
2. **Django integration** - Integrate the best-performing RoBERTa model into the moodlog.io Django backend
3. **API endpoint** - Create prediction endpoint that accepts journal text and returns emotions + dyads
4. **Mood visualization** - Display emotional patterns and trends in the moodlog.io frontend

---

## Citations

If you use GoEmotions:
```
Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", 2020.
```

If you use XED:
```
Öhman et al., "XED: A Multilingual Dataset for Cross-lingual Emotion Detection", 2020.
```

---

## License

MIT License.

---

## Contributing

For detailed technical information, code conventions, and development workflows, see [CLAUDE.md](CLAUDE.md).

For integration help, dataset expansion, or model troubleshooting, feel free to reach out or open an issue.
