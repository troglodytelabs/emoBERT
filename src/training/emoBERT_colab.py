#!/usr/bin/env python3
"""
emoBERT_collab.py

colab-ready version of emoBERT that mirrors the original cpu training behavior.

train a multi-label emotion classifier using distilbert.
this script:
- loads and maps goemotions to plutchik's 8 emotions
- loads xed, which is already labeled in plutchik space
- loads synthetic_claude.csv (high-quality synthetic data) and adds it to training
- trains a distilbert-based classifier and saves the best checkpoint
- performs per-emotion PR-curve threshold tuning on the validation set
- evaluates test performance for both default (0.5) and tuned thresholds

differences from the original cpu script:
- save_dir default is "../../models" (relative to script location)
- synthetic csv is read from "../../data/synthetic/synthetic_claude.csv"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertModel, get_linear_schedule_with_warmup

# step 1: argument parsing and basic setup

parser = argparse.ArgumentParser(
    description="train distilbert on goemotions / xed / synthetic data (colab gpu)"
)

parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--save_dir", type=str, default="../../models")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

print(f"\ntraining on {device}")
print(
    f"epochs = {args.epochs}, batch_size = {args.batch_size}, "
    f"lr = {args.learning_rate}, dropout = {args.dropout}"
)

# step 2: define plutchik emotions and mapping from goemotions

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

# step 3: dataset class used by pytorch dataloaders


class EmotionDataset(Dataset):
    """
    this dataset holds raw text strings and their corresponding plutchik label vectors.
    it tokenizes the text on-the-fly into input ids and attention masks using the
    same tokenizer as the distilbert model.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        texts: list of strings (input sentences or comments)
        labels: list of lists with length 8 (plutchik one-hot or multi-hot vectors)
        tokenizer: huggingface tokenizer compatible with distilbert
        max_length: maximum number of tokens per sequence (longer texts are truncated)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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


# step 4: load tokenizer

print("\nloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# helper to convert a goemotions split into plutchik vectors
def convert_goemotions_split(dataset, split_name, max_samples=None):
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

        if max_samples is not None and len(texts) >= max_samples:
            break

    return texts, labels


# step 5: load goemotions dataset and convert to plutchik space

print("\nloading goemotions from huggingface...")
goemotions = load_dataset("go_emotions", "simplified")

go_train_texts, go_train_labels = convert_goemotions_split(
    goemotions, "train", max_samples=args.max_samples
)

val_cap = args.max_samples // 4 if args.max_samples is not None else None
go_val_texts, go_val_labels = convert_goemotions_split(
    goemotions, "validation", max_samples=val_cap
)
go_test_texts, go_test_labels = convert_goemotions_split(
    goemotions, "test", max_samples=val_cap
)

print(
    f"goemotions sizes: train={len(go_train_texts)}, "
    f"val={len(go_val_texts)}, test={len(go_test_texts)}"
)

# step 6: load xed dataset and convert to plutchik space


def load_xed(max_samples=None):
    """
    loads the xed (extended emotion dataset) english annotations directly from
    the public github tsv and converts each example into a plutchik label vector.
    """

    print("\nloading xed (extended emotion dataset)...")

    xed_raw = load_dataset(
        "csv",
        data_files={
            "train": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/AnnotatedData/en-annotated.tsv"
        },
        delimiter="\t",
        column_names=["sentence", "labels"],
    )

    all_texts = []
    all_labels = []

    for example in tqdm(xed_raw["train"], desc="processing xed"):
        label_vec = [0] * len(plutchik_emotions)
        label_str = example["labels"].strip().lower()
        if label_str in plutchik_emotions:
            idx = plutchik_emotions.index(label_str)
            label_vec[idx] = 1

        all_texts.append(example["sentence"])
        all_labels.append(label_vec)

        if max_samples is not None and len(all_texts) >= max_samples:
            break

    total = len(all_texts)
    if total == 0:
        raise RuntimeError("xed dataset appears to be empty or could not be loaded")

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
        f"xed sizes: train={len(train_texts)}, "
        f"val={len(val_texts)}, test={len(test_texts)}"
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
    ) = load_xed(max_samples=args.max_samples)
    xed_available = True
except Exception as e:
    print(f"could not load xed, continuing with goemotions only. error was: {e}")
    xed_available = False

# step 7: load synthetic data from csv and convert to plutchik labels


def load_synthetic_data(csv_path):
    """
    loads synthetic training data from a csv file.
    the csv is expected to have a column named 'text' plus 8 label columns:
      joy_label, sadness_label, anger_label, fear_label,
      trust_label, disgust_label, surprise_label, anticipation_label
    """
    print(f"\nloading synthetic data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("synthetic csv must contain a 'text' column")

    expected_label_cols = [f"{emotion}_label" for emotion in plutchik_emotions]
    for col in expected_label_cols:
        if col not in df.columns:
            raise ValueError(f"synthetic csv is missing required label column: {col}")

    texts = df["text"].tolist()
    labels = df[expected_label_cols].values.tolist()

    print(f"synthetic samples loaded: {len(texts)}")
    return texts, labels


script_dir = Path(__file__).parent
synthetic_csv_path = script_dir / "../../data/synthetic/synthetic_claude.csv"

if synthetic_csv_path.exists():
    synthetic_texts, synthetic_labels = load_synthetic_data(synthetic_csv_path)
    synthetic_available = True
else:
    print(
        "\nno synthetic_claude.csv found in working directory, skipping synthetic data"
    )
    synthetic_texts, synthetic_labels = [], []
    synthetic_available = False

# step 8: combine goemotions, xed, and synthetic data into final splits

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

print(f"\nfinal dataset sizes (after combining):")
print(f"  train: {len(train_texts)} examples")
print(f"  val:   {len(val_texts)} examples")
print(f"  test:  {len(test_texts)} examples")

# step 9: create pytorch dataloaders for training, validation, and test

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
)

print(f"\nnumber of training batches: {len(train_loader)}")
print(f"number of validation batches: {len(val_loader)}")
print(f"number of test batches: {len(test_loader)}")

# step 10: define the model: distilbert + dropout + linear classifier


class PlutchikEmotionClassifier(nn.Module):
    """
    distilbert + dropout + linear layer → 8 logits (plutchik emotions)
    """

    def __init__(self, num_labels=8, dropout=0.3):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        return logits, probs


# step 11: set up loss function, optimizer, and learning rate scheduler

print("\ninitializing model...")
model = PlutchikEmotionClassifier(
    num_labels=len(plutchik_emotions), dropout=args.dropout
)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

total_steps = len(train_loader) * args.epochs
warmup_steps = total_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"total training steps: {total_steps}, warmup steps: {warmup_steps}")

best_val_macro_f1 = 0.0
train_losses = []
val_losses = []
val_f1_scores = []

start_time = time.time()

for epoch in range(args.epochs):
    print(f"\nstarting epoch {epoch + 1} / {args.epochs}")

    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc="training", ncols=80):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"epoch {epoch + 1} training loss: {avg_train_loss:.4f}")

    model.eval()
    total_val_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating", ncols=80):
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

    print(f"epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")
    print(
        f"epoch {epoch + 1} macro f1: {val_macro_f1:.4f}, micro f1: {val_micro_f1:.4f}"
    )
    print("per-emotion f1 scores:")
    for i, emotion in enumerate(plutchik_emotions):
        print(f"  {emotion:12s}: {per_emotion_f1[i]:.3f}")

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "macro_f1": val_macro_f1,
            "micro_f1": val_micro_f1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_f1_scores": val_f1_scores,
        }

        ckpt_path = save_dir / "best_model.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"saved new best model to {ckpt_path} (macro f1 = {val_macro_f1:.4f})")

# step 12: load best checkpoint and compute PR-curve thresholds on validation set

print("\nloading best checkpoint for final evaluation...")
best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.to(device)
model.eval()

print("\ncomputing per-emotion optimal thresholds using validation set (PR-curve)...")
val_probs = []
val_true = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="threshold tuning (val)", ncols=80):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        _, probs = model(input_ids, attention_mask)

        val_probs.extend(probs.cpu().numpy())
        val_true.extend(labels)

val_probs = np.array(val_probs)
val_true = np.array(val_true)

optimal_thresholds = []
val_best_f1 = []

for e_idx, emotion in enumerate(plutchik_emotions):
    y_true = val_true[:, e_idx]
    y_prob = val_probs[:, e_idx]

    if y_true.sum() == 0:
        optimal_thresholds.append(0.5)
        val_best_f1.append(0.0)
        print(f"  {emotion:12s}: no positives in val → τ=0.5")
        continue

    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    if best_idx >= len(thresh):
        best_thr = 0.5
    else:
        best_thr = thresh[best_idx]

    optimal_thresholds.append(best_thr)
    val_best_f1.append(f1_scores[best_idx])

    print(f"  {emotion:12s}: best τ={best_thr:.4f} | val-F1={f1_scores[best_idx]:.4f}")

optimal_thresholds = np.array(optimal_thresholds)
print("\noptimal per-emotion thresholds:", optimal_thresholds)

# step 13: final test evaluation (default 0.5 and tuned thresholds)

print("\nrunning final test evaluation...")

total_test_loss = 0.0
all_test_labels = []
all_test_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="testing", ncols=80):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, probs = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_test_loss += loss.item()

        all_test_probs.extend(probs.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
all_test_probs = np.array(all_test_probs)
all_test_labels = np.array(all_test_labels)

# default threshold 0.5
test_preds_default = (all_test_probs >= 0.5).astype(int)
test_macro_f1 = f1_score(
    all_test_labels, test_preds_default, average="macro", zero_division=0
)
test_micro_f1 = f1_score(
    all_test_labels, test_preds_default, average="micro", zero_division=0
)
per_emotion_f1_default = f1_score(
    all_test_labels, test_preds_default, average=None, zero_division=0
)

print(f"\nfinal test loss: {avg_test_loss:.4f}")
print(f"final test macro f1 (default 0.5): {test_macro_f1:.4f}")
print(f"final test micro f1 (default 0.5): {test_micro_f1:.4f}")
print("per-emotion test f1 (default 0.5):")
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_test_labels[:, i].sum())
    p = precision_score(
        all_test_labels[:, i], test_preds_default[:, i], zero_division=0
    )
    r = recall_score(all_test_labels[:, i], test_preds_default[:, i], zero_division=0)
    print(
        f"  {emotion:12s}: f1={per_emotion_f1_default[i]:.4f}, "
        f"precision={p:.3f}, recall={r:.3f}, support={support}"
    )

# tuned thresholds
thr = optimal_thresholds.reshape(1, -1)
test_preds_tuned = (all_test_probs >= thr).astype(int)

tuned_test_macro_f1 = f1_score(
    all_test_labels, test_preds_tuned, average="macro", zero_division=0
)
tuned_test_micro_f1 = f1_score(
    all_test_labels, test_preds_tuned, average="micro", zero_division=0
)
per_emotion_f1_tuned = f1_score(
    all_test_labels, test_preds_tuned, average=None, zero_division=0
)

print("\n--- tuned threshold evaluation (per-emotion PR-curve) ---")
print(f"tuned test macro f1: {tuned_test_macro_f1:.4f}")
print(f"tuned test micro f1: {tuned_test_micro_f1:.4f}")
print("per-emotion test f1 (tuned):")
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_test_labels[:, i].sum())
    p = precision_score(all_test_labels[:, i], test_preds_tuned[:, i], zero_division=0)
    r = recall_score(all_test_labels[:, i], test_preds_tuned[:, i], zero_division=0)
    print(
        f"  {emotion:12s}: f1={per_emotion_f1_tuned[i]:.4f}, "
        f"precision={p:.3f}, recall={r:.3f}, support={support}"
    )

# step 14: save summary of training and test results

results = {
    "config": {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "max_samples": args.max_samples,
        "device": str(device),
    },
    "best_val_macro_f1": float(best_val_macro_f1),
    # default 0.5 thresholds
    "test_macro_f1": float(test_macro_f1),
    "test_micro_f1": float(test_micro_f1),
    "per_emotion_f1_default": {
        emotion: float(per_emotion_f1_default[i])
        for i, emotion in enumerate(plutchik_emotions)
    },
    # tuned thresholds
    "tuned_test_macro_f1": float(tuned_test_macro_f1),
    "tuned_test_micro_f1": float(tuned_test_micro_f1),
    "per_emotion_f1_tuned": {
        emotion: float(per_emotion_f1_tuned[i])
        for i, emotion in enumerate(plutchik_emotions)
    },
    "optimal_thresholds": {
        emotion: float(optimal_thresholds[i])
        for i, emotion in enumerate(plutchik_emotions)
    },
    "training_time_seconds": time.time() - start_time,
}

results_path = save_dir / "test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\ntraining complete. results saved to {results_path}")
print(f"best validation macro f1: {best_val_macro_f1:.4f}")
print(f"test macro f1 (default 0.5): {test_macro_f1:.4f}")
print(f"test macro f1 (tuned): {tuned_test_macro_f1:.4f}")
print(f"model checkpoint: {save_dir / 'best_model.pt'}")
