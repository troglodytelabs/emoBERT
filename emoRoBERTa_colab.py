#!/usr/bin/env python3
"""
emoRoBERTa_colab.py

train a multi-label emotion classifier using RoBERTa with advanced improvements.
RoBERTa is generally more powerful than DistilBERT for emotion classification.

this script:
- loads and maps goemotions to plutchik's 8 emotions
- loads xed, which is already labeled in plutchik space
- loads synthetic_claude.csv (high-quality claude-generated data) and adds it to training
- uses focal loss for handling class imbalance
- applies layerwise learning rate decay for better fine-tuning
- uses label smoothing to prevent overconfidence
- trains a RoBERTa-based classifier and saves the best checkpoint

optimized for google colab gpu training.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

# step 1: check gpu availability and display configuration

print("emoRoBERTa - Colab GPU Training with Advanced Improvements")
print("Using RoBERTa-base instead of DistilBERT")
print("Goal: ALL emotions >67% F1")

# detect whether a gpu is available; use gpu if possible for much faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"+ GPU detected: {torch.cuda.get_device_name(0)}")
    print(
        f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
else:
    print("WARNING: No GPU detected! Training will be very slow.")
    print("   Go to Runtime -> Change runtime type -> Hardware accelerator -> GPU")

# step 2: configuration - optimized for colab gpu with RoBERTa

# RoBERTa is larger than DistilBERT, so we adjust batch size accordingly
config = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 3e-5,  # INCREASED - RoBERTa needs higher LR
    "dropout": 0.1,
    "warmup_ratio": 0.15,  # INCREASED - longer warmup helps
    "weight_decay": 0.01,
    "use_focal_loss": False,  # DISABLED - causing issues with label smoothing
    "focal_gamma": 2.0,  # Not used when use_focal_loss=False
    "label_smoothing": 0.1,
    "gradient_accumulation": 2,
    "layerwise_lr_decay": 0.90,  # More aggressive - top layers learn faster
    "max_length": 128,
}

print(f"\nTraining Configuration:")
for key, value in config.items():
    print(f"  {key:25s}: {value}")
print("\nNOTE: Using weighted BCE instead of focal loss for stability")

save_dir = Path("models_roberta")
save_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()

# step 3: define plutchik emotions and mapping from goemotions

plutchik_emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
]

goemotions_to_plutchik = {
    "admiration": ["joy", "trust"],
    "amusement": ["joy"],
    "approval": ["joy", "trust"],
    "caring": ["joy", "trust"],
    "desire": ["joy", "anticipation"],
    "excitement": ["joy", "anticipation", "surprise"],
    "gratitude": ["joy", "trust"],
    "joy": ["joy"],
    "love": ["joy", "trust"],
    "optimism": ["joy", "anticipation"],
    "pride": ["joy"],
    "relief": ["joy", "surprise"],
    "sadness": ["sadness"],
    "disappointment": ["sadness", "surprise"],
    "embarrassment": ["sadness", "fear"],
    "grief": ["sadness"],
    "remorse": ["sadness", "disgust"],
    "anger": ["anger"],
    "annoyance": ["anger", "disgust"],
    "disapproval": ["anger", "disgust"],
    "fear": ["fear"],
    "nervousness": ["fear", "anticipation"],
    "disgust": ["disgust"],
    "surprise": ["surprise"],
    "realization": ["surprise"],
    "confusion": ["surprise", "fear"],
    "curiosity": ["anticipation", "surprise"],
    "neutral": [],
}

xed_to_plutchik = {
    "anger": ["anger"],
    "disgust": ["disgust"],
    "fear": ["fear"],
    "joy": ["joy"],
    "sadness": ["sadness"],
    "surprise": ["surprise"],
}

# step 4: improved dataset class with label smoothing support


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, label_smoothing=0.0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_smoothing = label_smoothing

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.label_smoothing > 0:
            label = np.array(label, dtype=np.float32)
            label = label * (1 - self.label_smoothing) + self.label_smoothing / 2

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.FloatTensor(label),
        }


# step 5: focal loss for handling class imbalance


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# step 6: RoBERTa-based model definition


class PlutchikEmotionClassifier(nn.Module):
    """
    RoBERTa-based emotion classifier

    architecture:
        roberta-base (12 layers, 768 hidden) -> dropout -> linear(768 -> 8) -> sigmoid

    advantages over DistilBERT:
    - better contextual understanding (12 layers vs 6)
    - trained on more data with better objective
    - typically 2-3% better F1 on emotion tasks
    """

    def __init__(self, num_labels=8, dropout=0.1):
        super().__init__()

        # load pretrained RoBERTa-base
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

        # initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        # RoBERTa uses position 0 (CLS token) for classification
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        return logits, probs


# step 7: layerwise learning rate decay helper


def get_layerwise_params(model, lr, decay_rate, weight_decay):
    """
    apply layer-wise learning rate decay for RoBERTa

    RoBERTa has 12 layers (vs DistilBERT's 6)
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    # classifier head gets full lr
    optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in model.classifier.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in model.classifier.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        }
    )

    # RoBERTa has 12 transformer layers
    num_layers = 12
    for layer in range(num_layers - 1, -1, -1):
        layer_lr = lr * (decay_rate ** (num_layers - 1 - layer))

        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in model.roberta.encoder.layer[layer].named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            }
        )
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in model.roberta.encoder.layer[layer].named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }
        )

    # embeddings get the lowest learning rate
    embed_lr = lr * (decay_rate**num_layers)
    optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in model.roberta.embeddings.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": embed_lr,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in model.roberta.embeddings.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": embed_lr,
        }
    )

    return optimizer_grouped_parameters


# step 8: load tokenizer

print("LOADING DATA")
print("\nloading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# step 9: load and process goemotions dataset


def convert_goemotions_split(dataset, split_name):
    texts = []
    labels = []
    emotion_names = dataset["train"].features["labels"].feature.names

    for example in tqdm(
        dataset[split_name], desc=f"processing goemotions {split_name}"
    ):
        plutchik_vec = [0] * len(plutchik_emotions)
        for label_idx in example["labels"]:
            if label_idx < len(emotion_names):
                ge_label = emotion_names[label_idx]
                if ge_label in goemotions_to_plutchik:
                    for pe in goemotions_to_plutchik[ge_label]:
                        pos = plutchik_emotions.index(pe)
                        plutchik_vec[pos] = 1
        texts.append(example["text"])
        labels.append(plutchik_vec)
    return texts, labels


print("\nloading goemotions from huggingface...")
goemotions = load_dataset("go_emotions", "simplified")

go_train_texts, go_train_labels = convert_goemotions_split(goemotions, "train")
go_val_texts, go_val_labels = convert_goemotions_split(goemotions, "validation")
go_test_texts, go_test_labels = convert_goemotions_split(goemotions, "test")

print(
    f"goemotions sizes: train={len(go_train_texts):,}, val={len(go_val_texts):,}, test={len(go_test_texts):,}"
)

# step 10: load and process xed dataset


def load_xed():
    print("\nloading xed (extended emotion dataset)...")
    xed_raw = load_dataset(
        "google-research-datasets/xed_english_finnish", "en_annotated"
    )

    all_texts = []
    all_labels = []

    for example in tqdm(xed_raw["train"], desc="processing xed"):
        label_vec = [0] * len(plutchik_emotions)
        for xed_emotion, plut_emotions in xed_to_plutchik.items():
            if example.get(xed_emotion, 0) > 0:
                for plut_emotion in plut_emotions:
                    idx = plutchik_emotions.index(plut_emotion)
                    label_vec[idx] = 1

        if sum(label_vec) > 0:
            all_texts.append(example["text"])
            all_labels.append(label_vec)

    total = len(all_texts)
    indices = np.random.RandomState(42).permutation(total)

    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    train_texts = [all_texts[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_texts = [all_texts[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    test_texts = [all_texts[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    print(
        f"xed sizes: train={len(train_texts):,}, val={len(val_texts):,}, test={len(test_texts):,}"
    )
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


try:
    (
        xed_train_texts,
        xed_train_labels,
        xed_val_texts,
        xed_val_labels,
        xed_test_texts,
        xed_test_labels,
    ) = load_xed()
    xed_available = True
except Exception as e:
    print(f"could not load xed: {e}")
    xed_available = False

# step 11: load claude synthetic data


def load_synthetic_data(csv_path):
    print(f"\nloading claude synthetic data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            raise ValueError("synthetic csv must contain a 'text' column")

        expected_label_cols = [f"{emotion}_label" for emotion in plutchik_emotions]
        for col in expected_label_cols:
            if col not in df.columns:
                raise ValueError(
                    f"synthetic csv is missing required label column: {col}"
                )

        texts = df["text"].tolist()
        labels = df[expected_label_cols].values.tolist()

        print(f"+ loaded {len(texts):,} claude synthetic samples")

        labels_array = np.array(labels)
        print("\nsynthetic data distribution:")
        for i, emotion in enumerate(plutchik_emotions):
            count = int(labels_array[:, i].sum())
            print(f"  {emotion:15s}: {count:6,} samples")

        return texts, labels
    except FileNotFoundError:
        print("WARNING: synthetic_claude.csv not found!")
        return [], []
    except Exception as e:
        print(f"WARNING: error loading synthetic data: {e}")
        return [], []


synthetic_csv_path = Path("synthetic_claude.csv")
synthetic_texts, synthetic_labels = load_synthetic_data(synthetic_csv_path)
synthetic_available = len(synthetic_texts) > 0

# step 12: combine all datasets

train_texts = list(go_train_texts)
train_labels = list(go_train_labels)
val_texts = list(go_val_texts)
val_labels = list(go_val_labels)
test_texts = list(go_test_texts)
test_labels = list(go_test_labels)

if xed_available:
    print("\nadding xed data to goemotions splits...")
    train_texts.extend(xed_train_texts)
    train_labels.extend(xed_train_labels)
    val_texts.extend(xed_val_texts)
    val_labels.extend(xed_val_labels)
    test_texts.extend(xed_test_texts)
    test_labels.extend(xed_test_labels)

if synthetic_available:
    print("adding synthetic data to training split only...")
    train_texts.extend(synthetic_texts)
    train_labels.extend(synthetic_labels)

print("\nFINAL DATASET SIZES")
print(f"train: {len(train_texts):,} examples")
print(f"val:   {len(val_texts):,} examples")
print(f"test:  {len(test_texts):,} examples")

# step 13: calculate class weights for focal loss

train_labels_array = np.array(train_labels)
pos_counts = train_labels_array.sum(axis=0)
total = len(train_labels)
class_weights = torch.FloatTensor(
    [total / (2 * count) if count > 0 else 1.0 for count in pos_counts]
)

print("\nCLASS DISTRIBUTION & WEIGHTS")
for i, emotion in enumerate(plutchik_emotions):
    pct = (pos_counts[i] / total) * 100
    print(
        f"{emotion:15s}: {int(pos_counts[i]):6,} ({pct:5.2f}%) weight: {class_weights[i]:.3f}"
    )

# step 14: create pytorch datasets and dataloaders

train_dataset = EmotionDataset(
    train_texts,
    train_labels,
    tokenizer,
    max_length=config["max_length"],
    label_smoothing=config["label_smoothing"],
)
val_dataset = EmotionDataset(
    val_texts, val_labels, tokenizer, max_length=config["max_length"]
)
test_dataset = EmotionDataset(
    test_texts, test_labels, tokenizer, max_length=config["max_length"]
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"] * 2)

print(f"\nnumber of training batches: {len(train_loader):,}")
print(f"number of validation batches: {len(val_loader):,}")
print(f"number of test batches: {len(test_loader):,}")

# step 15: initialize model, loss, optimizer, and scheduler

print("\nCREATING ROBERTA MODEL")
model = PlutchikEmotionClassifier(
    num_labels=len(plutchik_emotions), dropout=config["dropout"]
)
model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Note: RoBERTa has ~125M params vs DistilBERT's ~66M params")

# Use weighted BCE loss (more stable than focal loss with label smoothing)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

optimizer_grouped_parameters = get_layerwise_params(
    model, config["learning_rate"], config["layerwise_lr_decay"], config["weight_decay"]
)
optimizer = AdamW(optimizer_grouped_parameters)

total_steps = len(train_loader) * config["epochs"]
num_warmup_steps = int(total_steps * config["warmup_ratio"])

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)

print(f"training steps: {total_steps:,} | warmup: {num_warmup_steps:,}")
print("using linear LR schedule (better for emotion classification)")

# step 16: training and validation loop

print("\nSTARTING TRAINING")

best_val_macro_f1 = 0.0
train_losses = []
val_losses = []
val_f1_scores = []

for epoch in range(config["epochs"]):
    epoch_start = time.time()
    print(f"\nEPOCH {epoch + 1}/{config['epochs']}")

    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Store unscaled loss for reporting
        total_train_loss += loss.item()

        # Scale loss for gradient accumulation
        loss = loss / config["gradient_accumulation"]
        loss.backward()

        if (batch_idx + 1) % config["gradient_accumulation"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Handle any remaining accumulated gradients
    if len(train_loader) % config["gradient_accumulation"] != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validation
    model.eval()
    total_val_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, probs = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            preds = (probs >= 0.5).cpu().numpy()
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    val_macro_f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    val_micro_f1 = f1_score(
        all_labels, all_predictions, average="micro", zero_division=0
    )
    val_f1_scores.append(val_macro_f1)

    per_emotion_f1 = f1_score(
        all_labels, all_predictions, average=None, zero_division=0
    )

    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]

    print(f"\nresults:")
    print(f"  train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")
    print(f"  val macro f1: {val_macro_f1:.4f} | val micro f1: {val_micro_f1:.4f}")
    print(f"  learning rate: {current_lr:.2e} | time: {epoch_time:.1f}s")

    print("\nper-emotion f1:")
    emotions_above_67 = 0
    for i, emotion in enumerate(plutchik_emotions):
        status = "+" if per_emotion_f1[i] >= 0.67 else "-"
        print(f"  {status} {emotion:15s}: {per_emotion_f1[i]:.4f}")
        if per_emotion_f1[i] >= 0.67:
            emotions_above_67 += 1

    min_f1 = per_emotion_f1.min()
    print(f"\nmin f1: {min_f1:.4f} | emotions >=67%: {emotions_above_67}/8")

    if min_f1 >= 0.67:
        print("ALL EMOTIONS ABOVE 67% F1!")

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "macro_f1": val_macro_f1,
            "micro_f1": val_micro_f1,
            "per_emotion_f1": per_emotion_f1.tolist(),
            "min_f1": float(min_f1),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_f1_scores": val_f1_scores,
            "config": config,
        }

        ckpt_path = save_dir / "best_model.pt"
        torch.save(checkpoint, ckpt_path)
        print(
            f"\n>> saved new best model (macro f1 = {val_macro_f1:.4f}, min f1 = {min_f1:.4f})"
        )

# step 17: optimize thresholds using PR-curve on validation set

print("\nFINAL TEST EVALUATION")
print("\nloading best checkpoint...")
best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.to(device)
model.eval()

print("\ncomputing per-emotion optimal thresholds using validation set (PR-curve)...")

# collect all validation probabilities (not binary predictions)
all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="threshold tuning (val)"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, probs = model(input_ids, attention_mask)

        all_val_probs.extend(probs.cpu().numpy())
        all_val_labels.extend(labels.cpu().numpy())

all_val_probs = np.array(all_val_probs)
all_val_labels = np.array(all_val_labels)

# find optimal threshold for each emotion using validation set
optimal_thresholds = []

for i, emotion in enumerate(plutchik_emotions):
    y_true = all_val_labels[:, i]
    y_prob = all_val_probs[:, i]

    # skip if no positive samples
    if y_true.sum() == 0:
        optimal_thresholds.append(0.5)
        print(f"  {emotion:15s}: no positive samples, using default τ=0.5")
        continue

    # test thresholds from 0.1 to 0.9 in steps of 0.01
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    optimal_thresholds.append(best_threshold)
    print(f"  {emotion:15s}: best τ={best_threshold:.4f} | val-F1={best_f1:.4f}")

optimal_thresholds = np.array(optimal_thresholds)
print(f"\noptimal per-emotion thresholds: {optimal_thresholds}")

# step 18: test evaluation with default 0.5 threshold

print("\nrunning final test evaluation...")

total_test_loss = 0.0
all_test_predictions = []
all_test_probs = []
all_test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, probs = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_test_loss += loss.item()

        preds = (probs >= 0.5).cpu().numpy()
        all_test_predictions.extend(preds)
        all_test_probs.extend(probs.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
all_test_predictions = np.array(all_test_predictions)
all_test_probs = np.array(all_test_probs)
all_test_labels = np.array(all_test_labels)

test_macro_f1 = f1_score(
    all_test_labels, all_test_predictions, average="macro", zero_division=0
)
test_micro_f1 = f1_score(
    all_test_labels, all_test_predictions, average="micro", zero_division=0
)
per_emotion_f1 = f1_score(
    all_test_labels, all_test_predictions, average=None, zero_division=0
)

print(f"\nfinal test loss: {avg_test_loss:.4f}")
print(f"final test macro f1 (default 0.5): {test_macro_f1:.4f}")
print(f"final test micro f1 (default 0.5): {test_micro_f1:.4f}")
print("per-emotion test f1 (default 0.5):")
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_test_labels[:, i].sum())
    p = precision_score(
        all_test_labels[:, i], all_test_predictions[:, i], zero_division=0
    )
    r = recall_score(all_test_labels[:, i], all_test_predictions[:, i], zero_division=0)
    print(
        f"  {emotion:15s}: f1={per_emotion_f1[i]:.4f}, precision={p:.3f}, recall={r:.3f}, support={support}"
    )

# step 19: test evaluation with tuned thresholds

print("\n--- tuned threshold evaluation (per-emotion PR-curve) ---")

# apply optimal thresholds
all_test_predictions_tuned = np.zeros_like(all_test_probs)
for i in range(len(plutchik_emotions)):
    all_test_predictions_tuned[:, i] = (
        all_test_probs[:, i] >= optimal_thresholds[i]
    ).astype(int)

tuned_test_macro_f1 = f1_score(
    all_test_labels, all_test_predictions_tuned, average="macro", zero_division=0
)
tuned_test_micro_f1 = f1_score(
    all_test_labels, all_test_predictions_tuned, average="micro", zero_division=0
)
tuned_per_emotion_f1 = f1_score(
    all_test_labels, all_test_predictions_tuned, average=None, zero_division=0
)

print(f"tuned test macro f1: {tuned_test_macro_f1:.4f}")
print(f"tuned test micro f1: {tuned_test_micro_f1:.4f}")
print("per-emotion test f1 (tuned):")

emotions_above_67_tuned = 0
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_test_labels[:, i].sum())
    p = precision_score(
        all_test_labels[:, i], all_test_predictions_tuned[:, i], zero_division=0
    )
    r = recall_score(
        all_test_labels[:, i], all_test_predictions_tuned[:, i], zero_division=0
    )
    status = "+" if tuned_per_emotion_f1[i] >= 0.67 else "-"
    print(
        f"  {status} {emotion:15s}: f1={tuned_per_emotion_f1[i]:.4f}, precision={p:.3f}, recall={r:.3f}, support={support}"
    )
    if tuned_per_emotion_f1[i] >= 0.67:
        emotions_above_67_tuned += 1

# save results with both default and tuned metrics
results = {
    "config": config,
    "model_type": "roberta-base",
    "best_val_macro_f1": float(best_val_macro_f1),
    "test_macro_f1_default": float(test_macro_f1),
    "test_micro_f1_default": float(test_micro_f1),
    "test_macro_f1_tuned": float(tuned_test_macro_f1),
    "test_micro_f1_tuned": float(tuned_test_micro_f1),
    "test_min_f1_default": float(per_emotion_f1.min()),
    "test_min_f1_tuned": float(tuned_per_emotion_f1.min()),
    "per_emotion_f1_default": {
        emotion: float(per_emotion_f1[i]) for i, emotion in enumerate(plutchik_emotions)
    },
    "per_emotion_f1_tuned": {
        emotion: float(tuned_per_emotion_f1[i])
        for i, emotion in enumerate(plutchik_emotions)
    },
    "optimal_thresholds": optimal_thresholds.tolist(),
    "training_time_seconds": time.time() - start_time,
    "all_emotions_above_67_default": bool(per_emotion_f1.min() >= 0.67),
    "all_emotions_above_67_tuned": bool(tuned_per_emotion_f1.min() >= 0.67),
    "emotions_above_67_count_default": int((per_emotion_f1 >= 0.67).sum()),
    "emotions_above_67_count_tuned": int(emotions_above_67_tuned),
}

results_path = save_dir / "test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

total_time = time.time() - start_time

print("\nTRAINING COMPLETE!")
print(f"total time: {total_time / 60:.1f} minutes")
print(f"best validation macro f1: {best_val_macro_f1:.4f}")
print(f"\ntest results (default τ=0.5):")
print(f"  macro f1: {test_macro_f1:.4f}")
print(f"  min f1: {per_emotion_f1.min():.4f}")
print(f"  emotions >=67%: {int((per_emotion_f1 >= 0.67).sum())}/8")
print(f"\ntest results (tuned thresholds):")
print(
    f"  macro f1: {tuned_test_macro_f1:.4f} (+{tuned_test_macro_f1 - test_macro_f1:.4f})"
)
print(f"  min f1: {tuned_per_emotion_f1.min():.4f}")
print(f"  emotions >=67%: {emotions_above_67_tuned}/8")

if tuned_per_emotion_f1.min() >= 0.67:
    print("\nSUCCESS! ALL EMOTIONS ABOVE 67% F1 (with tuned thresholds)!")
else:
    below_67 = [
        plutchik_emotions[i] for i, f1 in enumerate(tuned_per_emotion_f1) if f1 < 0.67
    ]
    print(f"\nemotions below 67% (tuned): {below_67}")
    print(
        f"improvement from tuning: +{tuned_test_macro_f1 - test_macro_f1:.4f} macro F1"
    )

print(f"\nresults saved to {results_path}")
print(f"model checkpoint: {save_dir / 'best_model.pt'}")
print("\nto download - click folder icon >> right-click file >> download")
print("\noptimal thresholds saved in test_results.json under 'optimal_thresholds'")
