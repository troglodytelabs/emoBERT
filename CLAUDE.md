# CLAUDE.md - AI Assistant Guide for emoBERT

## Project Overview

emoBERT is a multi-label emotion classification toolkit based on **Plutchik's 8-emotion model**. It provides training pipelines, inference scripts, and synthetic data generation for classifying text into emotional categories.

### Purpose & Vision

This is an **affective computing** project designed to power [moodlog.io](https://github.com/troglodytelabs/moodlog.io) - a mood tracking application that helps users understand their emotions over time. The emoBERT model will:

1. **Analyze journal entries** - Process short text entries from users logging their daily moods
2. **Predict base emotions** - Identify Plutchik's 8 emotions present in the text
3. **Suggest dyadic relationships** - Combine emotions into more interpretable/meaningful concepts (e.g., joy + trust = love)
4. **Track emotional patterns** - Help users understand mood trends and emotional health over time

### Roadmap / Next Steps

1. **Dyad interpretation layer** - Build on `emoPredict.py`'s dyad detection to provide human-readable emotional insights
2. **Django integration** - Integrate the best-performing RoBERTa model into the moodlog.io Django backend
3. **API endpoint** - Create prediction endpoint that accepts journal text and returns emotions + dyads
4. **Mood visualization** - Display emotional patterns and trends in the moodlog.io frontend

### Core Emotions (Plutchik)
```
joy, sadness, anger, fear, trust, disgust, surprise, anticipation
```

The emotion order is critical - all label vectors, model outputs, and thresholds follow this exact sequence.

## Repository Structure

```
emoBERT/
├── emoBERT.py              # CPU training script (DistilBERT)
├── emoBERT_colab.py         # GPU training script (DistilBERT, Colab)
├── emoRoBERTa_colab.py      # GPU training script (RoBERTa, Colab) - RECOMMENDED
├── emoPredict.py            # Inference with DistilBERT + dyad detection
├── emoPredict_roberta.py    # Inference with RoBERTa - RECOMMENDED
├── emoGen_claude.py         # Synthetic data generation using Claude API
├── emoGen_claude_v2.py      # Alternate synthetic data generator
├── synthetic_claude.csv     # Generated synthetic training data
├── synthetic_claude_merged.csv  # Combined synthetic datasets
├── models/                  # Model checkpoints and results
│   ├── best_model.pt        # Best trained model checkpoint
│   └── test_results.json    # Training metrics and optimal thresholds
└── README.md
```

## Key Technical Details

### Model Architecture

**DistilBERT-based** (`emoBERT.py`, `emoPredict.py`):
```python
DistilBertModel → dropout(0.3) → Linear(768 → 8) → sigmoid
```

**RoBERTa-based** (`emoRoBERTa_colab.py`, `emoPredict_roberta.py`):
```python
RobertaModel → dropout(0.15) → Linear(768 → 8) → sigmoid
```

RoBERTa is recommended for better performance (12 layers vs DistilBERT's 6).

### Training Data Sources

1. **GoEmotions** (HuggingFace) - Fine-grained emotions mapped to Plutchik
2. **XED** (Extended Emotion Dataset) - Already in Plutchik format
3. **Synthetic Claude data** - LLM-generated training samples

### GoEmotions to Plutchik Mapping

The mapping is defined in all training scripts. Key examples:
```python
goemotions_to_plutchik = {
    "admiration": ["joy", "trust"],
    "excitement": ["joy", "anticipation", "surprise"],
    "nervousness": ["fear", "anticipation"],
    "remorse": ["sadness", "disgust"],
    "neutral": [],  # No emotion
    # ... etc
}
```

### Loss Functions and Techniques

- **BCEWithLogitsLoss** - Standard multi-label classification loss
- **Focal Loss** - Available for class imbalance (use_focal_loss=True)
- **Label Smoothing** - Prevents overconfidence (default 0.1)
- **Class Weighting** - Automatic based on class distribution
- **Layer-wise LR Decay** - Lower layers learn slower (decay_rate=0.85)

### Threshold Tuning

Models use **per-emotion PR-curve threshold tuning** for optimal predictions:
- Default threshold: 0.5
- Optimal thresholds are computed on validation set and stored in `test_results.json`
- Thresholds typically range from 0.53 to 0.79 depending on emotion

## Development Workflows

### Training a New Model

**Local CPU (DistilBERT)**:
```bash
python emoBERT.py --epochs 3 --batch_size 16 --learning_rate 2e-5
```

**Google Colab GPU (RoBERTa - Recommended)**:
```bash
python emoRoBERTa_colab.py
```

Default configuration in `emoRoBERTa_colab.py`:
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

### Running Inference

**Single text prediction**:
```bash
python emoPredict_roberta.py --text "I'm excited about the future!"
```

**Interactive mode**:
```bash
python emoPredict_roberta.py --interactive
```

**Batch processing**:
```bash
python emoPredict_roberta.py --file input.txt --verbose
```

### Generating Synthetic Data

```bash
# Generate for all rare emotions
python emoGen_claude.py --all_rare --count 10000 --api_key sk-ant-...

# Generate for specific emotion
python emoGen_claude.py --emotion fear --count 15000

# Generate for all 8 emotions
python emoGen_claude.py --all_emotions --count 15000
```

Required API key: Set `ANTHROPIC_API_KEY` env var or use `--api_key` flag.

## Code Conventions

### CSV Data Format

Synthetic data CSVs must follow this exact format:
```csv
text,joy_label,sadness_label,anger_label,fear_label,trust_label,disgust_label,surprise_label,anticipation_label
"Example text",1,0,0,0,1,0,0,0
```

### Model Checkpoint Format

Saved in `models/best_model.pt`:
```python
checkpoint = {
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,  # RoBERTa only
    "macro_f1": float,
    "micro_f1": float,
    "per_emotion_f1": list,
    "min_f1": float,
    "train_losses": list,
    "val_losses": list,
    "config": dict,
}
```

### Results Format

Saved in `models/test_results.json`:
```json
{
  "config": {...},
  "model_type": "roberta-base",
  "best_val_macro_f1": 0.6157,
  "test_macro_f1_default": 0.6078,
  "test_macro_f1_tuned": 0.6277,
  "per_emotion_f1_default": {...},
  "per_emotion_f1_tuned": {...},
  "optimal_thresholds": [0.66, 0.70, 0.71, 0.78, 0.73, 0.55, 0.53, 0.79]
}
```

### Naming Conventions

- Scripts: `emo<Function>[_variant].py` (e.g., `emoPredict_roberta.py`)
- Model files: `best_model.pt`
- Results: `test_results.json`, `test_results_v<N>.json`
- Synthetic data: `synthetic_claude.csv`, `synthetic_claude_merged.csv`

## Important Patterns

### Dataset Loading Pattern

All training scripts follow this pattern:
```python
# 1. Load GoEmotions and map to Plutchik
go_train_texts, go_train_labels = convert_goemotions_split(goemotions, "train")

# 2. Load XED (already Plutchik format, 80/10/10 split)
xed_train, xed_val, xed_test = load_xed()

# 3. Load synthetic data (training only)
synthetic_texts, synthetic_labels = load_synthetic_data("synthetic_claude.csv")

# 4. Combine all datasets
train_texts = go_train + xed_train + synthetic  # Synthetic only in train!
```

### Inference Pattern

```python
# Load model and thresholds
model, thresholds = load_model(model_path, device)

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Predict
with torch.no_grad():
    logits, probs = model(input_ids, attention_mask)

# Apply per-emotion thresholds
predictions = (probs >= thresholds).astype(int)
```

### Plutchik Dyads (Advanced Feature)

The `emoPredict.py` script includes dyad detection (emotion combinations):
- **Primary dyads**: Adjacent emotions (e.g., joy + trust = love)
- **Secondary dyads**: Skip one emotion (e.g., joy + fear = guilt)
- **Tertiary dyads**: Skip two emotions (e.g., joy + surprise = delight)
- **Opposition dyads**: Opposites (e.g., joy + sadness = bittersweetness)

## Performance Benchmarks

Current best results (RoBERTa with tuned thresholds):
- Test Macro F1: 0.6277
- Test Micro F1: 0.6838
- Best emotion: joy (0.83 F1)
- Hardest emotion: fear (0.54 F1)

Goal: All emotions >67% F1

## Common Tasks for AI Assistants

### Adding New Training Data
1. Format data as CSV with required columns
2. Place in repository root
3. Update `load_synthetic_data()` function with new path
4. Synthetic data is only added to training split (not val/test)

### Modifying Model Architecture
- Model class is `PlutchikEmotionClassifier` in each script
- Changes must be synchronized across training and inference scripts
- Architecture: encoder → dropout → linear classifier

### Adjusting Thresholds
- Thresholds are computed via PR-curve during training
- Manual thresholds can be set in `EMOTION_THRESHOLDS` dict in `emoPredict.py`
- Lower thresholds for rare/underdetected emotions (fear, anticipation)

### Debugging Training Issues
- Check class distribution in training data
- Monitor per-emotion F1 during validation
- Consider increasing epochs or adjusting learning rate
- Early stopping patience is 5 epochs by default

## Dependencies

```bash
pip install torch transformers datasets scikit-learn pandas tqdm

# For Colab GPU
pip install accelerate

# For synthetic data generation
pip install anthropic
```

## Environment Notes

- GPU strongly recommended for RoBERTa training
- CPU training viable with DistilBERT (slower)
- Default max sequence length: 128 tokens
- Batch size: 64 (GPU) or 16 (CPU)

## File-Specific Notes

### emoBERT.py (Lines 383-395)
Contains hardcoded absolute path for synthetic data that should be changed for different environments.

### emoPredict.py (Lines 385-398)
Contains adaptive thresholds that can be manually tuned for better detection of specific emotions.

### emoRoBERTa_colab.py (Lines 57-71)
Main configuration dict that controls all training hyperparameters.

## Testing Changes

After making changes:
1. Run training script on small dataset (`--max_samples 1000`)
2. Check validation metrics improve
3. Test inference on sample texts
4. Verify output format matches expected structure
