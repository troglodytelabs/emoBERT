#!/usr/bin/env python3
"""
emoBERT.py

train a multi-label emotion classifier using distilbert.
this script:
- loads and maps goemotions to plutchik's 8 emotions
- loads xed, which is already labeled in plutchik space
- optionally loads synthetic_data.csv and adds it to the training set
- trains a distilbert-based classifier and saves the best checkpoint
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
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertModel, get_linear_schedule_with_warmup

# step 1: argument parsing and basic setup

# create an argument parser so you can change basic training settings
# from the command line without editing the file
parser = argparse.ArgumentParser(
    description="train distilbert on goemotions / xed / synthetic data"
)

# number of passes over the training data
parser.add_argument("--epochs", type=int, default=3)

# how many examples to process in one batch
parser.add_argument("--batch_size", type=int, default=16)

# learning rate for adamw optimizer (small value because transformers are sensitive)
parser.add_argument("--learning_rate", type=float, default=2e-5)

# dropout rate applied to the pooled distilbert embedding before the classifier layer
parser.add_argument("--dropout", type=float, default=0.3)

# optional cap on number of samples from goemotions/xed (useful for quick experiments)
parser.add_argument("--max_samples", type=int, default=None)

# directory where model checkpoints and results will be saved
parser.add_argument("--save_dir", type=str, default="../../models")

args = parser.parse_args()

# detect whether a gpu is available; use gpu if possible for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ensure that the save directory exists
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

print(f"\ntraining on {device}")
print(
    f"epochs = {args.epochs}, batch_size = {args.batch_size}, lr = {args.learning_rate}, dropout = {args.dropout}"
)

# step 2: define plutchik emotions and mapping from goemotions

# this fixed list defines the order of the 8 plutchik emotions used everywhere
# the order is important because label vectors and output logits all follow this order
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

# this dictionary maps each goemotions label into one or more plutchik emotions
# many goemotions labels are more fine-grained than plutchik, so they may map to multiple plutchik entries
# for example:
# - "love" is treated as a mix of joy and trust
# - "nervousness" is treated as fear plus anticipation
# - "excitement" is treated as joy, anticipation, and surprise
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
    # neutral means no emotion is active; we leave the plutchik vector all zeros
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
        # number of examples in dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # fetch raw text and label for a given index
        text = str(self.texts[idx])
        label = self.labels[idx]

        # tokenize the text into input ids and attention mask
        # padding='max_length' ensures all sequences in the batch have the same length
        # truncation=True ensures texts longer than max_length are cut off
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # we flatten because tokenizer returns tensors with an extra batch dimension (1, seq_len)
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            # convert label list (python list of ints) to float tensor for bcewithlogitsloss
            "labels": torch.FloatTensor(label),
        }


# step 4: load tokenizer

# the tokenizer converts raw text strings into integer token ids and attention masks
# we use the same base model name as the distilbert model to ensure compatibility
print("\nloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# helper to convert a goemotions split into plutchik vectors
def convert_goemotions_split(dataset, split_name, max_samples=None):
    """
    converts one split of the goemotions dataset (train / validation / test)
    into a list of texts and corresponding plutchik multi-label vectors.

    dataset: the loaded goemotions dataset from huggingface datasets
    split_name: 'train', 'validation', or 'test'
    max_samples: optional cap on number of examples used from this split

    returns:
        texts: list of strings
        labels: list of length-8 lists (multi-hot plutchik label vectors)
    """
    texts = []
    labels = []

    # goemotions encodes labels as integer indices; we need to map them to names first
    emotion_names = dataset["train"].features["labels"].feature.names

    for example in tqdm(
        dataset[split_name], desc=f"processing goemotions {split_name}"
    ):
        # start with a zero vector for all 8 plutchik emotions
        plutchik_vec = [0] * len(plutchik_emotions)

        # each example can have multiple goemotions labels
        for label_idx in example["labels"]:
            if label_idx < len(emotion_names):
                ge_label = emotion_names[label_idx]
                if ge_label in goemotions_to_plutchik:
                    # for each plutchik emotion mapped from this goemotions label,
                    # set the corresponding position in the plutchik vector to 1
                    for pe in goemotions_to_plutchik[ge_label]:
                        pos = plutchik_emotions.index(pe)
                        plutchik_vec[pos] = 1

        texts.append(example["text"])
        labels.append(plutchik_vec)

        # if a maximum number of samples is specified, stop once we reach that limit
        if max_samples is not None and len(texts) >= max_samples:
            break

    return texts, labels


# step 5: load goemotions dataset and convert to plutchik space

print("\nloading goemotions from huggingface...")
goemotions = load_dataset("go_emotions", "simplified")

# train / validation / test splits with optional downsampling controlled by max_samples
go_train_texts, go_train_labels = convert_goemotions_split(
    goemotions, "train", max_samples=args.max_samples
)

# for validation and test, if max_samples is set, we usually only need a fraction of that
val_cap = args.max_samples // 4 if args.max_samples is not None else None
go_val_texts, go_val_labels = convert_goemotions_split(
    goemotions, "validation", max_samples=val_cap
)
go_test_texts, go_test_labels = convert_goemotions_split(
    goemotions, "test", max_samples=val_cap
)

print(
    f"goemotions sizes: train={len(go_train_texts)}, val={len(go_val_texts)}, test={len(go_test_texts)}"
)

# step 6: load xed dataset and convert to plutchik space


def load_xed(max_samples=None):
    """
    loads the xed (extended emotion dataset) english annotations directly from
    the public github tsv and converts each example into a plutchik label vector.

    xed labels are already given as plutchik emotion names (or neutral),
    so the conversion is simpler than for goemotions.
    this function also creates its own train/val/test split with an 80/10/10 ratio.

    returns:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    print("\nloading xed (extended emotion dataset)...")

    # load raw tsv as a csv dataset using huggingface datasets
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

        # labels are strings like "anger", "joy", or "neutral"
        label_str = example["labels"].strip().lower()
        if label_str in plutchik_emotions:
            idx = plutchik_emotions.index(label_str)
            label_vec[idx] = 1
        # if label_str is "neutral" or unknown, the vector stays all zeros

        all_texts.append(example["sentence"])
        all_labels.append(label_vec)

        if max_samples is not None and len(all_texts) >= max_samples:
            break

    total = len(all_texts)
    if total == 0:
        raise RuntimeError("xed dataset appears to be empty or could not be loaded")

    # create a reproducible permutation of indices so splits are deterministic
    indices = np.random.RandomState(42).permutation(total)

    # define split sizes: 80% train, 10% val, 10% test
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    # construct split lists by indexing into all_texts and all_labels
    train_texts = [all_texts[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]

    val_texts = [all_texts[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    test_texts = [all_texts[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    print(
        f"xed sizes: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}"
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# try loading xed; if it fails, we proceed with only goemotions
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

    these columns must align with the order in plutchik_emotions.
    the function returns two lists: synthetic_texts and synthetic_labels.
    """
    print(f"\nloading synthetic data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("synthetic_data.csv must contain a 'text' column")

    # ensure all expected label columns are present
    expected_label_cols = [f"{emotion}_label" for emotion in plutchik_emotions]
    for col in expected_label_cols:
        if col not in df.columns:
            raise ValueError(
                f"synthetic_data.csv is missing required label column: {col}"
            )

    texts = df["text"].tolist()

    # extract label values in the strict plutchik order
    labels = df[expected_label_cols].values.tolist()

    print(f"synthetic samples loaded: {len(texts)}")
    return texts, labels


# path relative to this script's location
script_dir = Path(__file__).parent
synthetic_csv_path = script_dir / "../../data/synthetic/synthetic_claude.csv"

if synthetic_csv_path.exists():
    synthetic_texts, synthetic_labels = load_synthetic_data(synthetic_csv_path)
    synthetic_available = True
else:
    print(
        "\nno synthetic_data.csv found at the configured path, skipping synthetic data"
    )
    synthetic_texts, synthetic_labels = [], []
    synthetic_available = False

# step 8: combine goemotions, xed, and synthetic data into final splits

# start by using goemotions as the base dataset for each split
train_texts = list(go_train_texts)
train_labels = list(go_train_labels)

val_texts = list(go_val_texts)
val_labels = list(go_val_labels)

test_texts = list(go_test_texts)
test_labels = list(go_test_labels)

# if xed loaded successfully, extend each split with xed samples
if xed_available:
    print("\nadding xed data to goemotions splits...")
    train_texts.extend(xed_train_texts)
    train_labels.extend(xed_train_labels)

    val_texts.extend(xed_val_texts)
    val_labels.extend(xed_val_labels)

    test_texts.extend(xed_test_texts)
    test_labels.extend(xed_test_labels)

# synthetic data is used only for training; we do not contaminate validation or test splits
# this makes evaluation more honest because synthetic patterns do not appear in held-out sets
if synthetic_available:
    print("adding synthetic data to training split only...")
    train_texts.extend(synthetic_texts)
    train_labels.extend(synthetic_labels)

print(f"\nfinal dataset sizes (after combining):")
print(f"  train: {len(train_texts)} examples")
print(f"  val:   {len(val_texts)} examples")
print(f"  test:  {len(test_texts)} examples")

# step 9: create pytorch dataloaders for training, validation, and test

# wrap raw lists in our custom EmotionDataset, which handles tokenization and tensor conversion
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

# dataloaders handle batching and optional shuffling
# we shuffle the training set each epoch, but validation/test remain in fixed order
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
    this model wraps a pretrained distilbert backbone and adds a simple linear layer
    that predicts 8 independent emotion probabilities (one per plutchik emotion).

    architecture:
        distilbert (pretrained) → dropout → linear(768 → 8) → sigmoid
    """

    def __init__(self, num_labels=8, dropout=0.3):
        super().__init__()

        # load pretrained distilbert used as the sentence encoder
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # dropout helps reduce overfitting by randomly zeroing some features during training
        self.dropout = nn.Dropout(dropout)

        # linear classifier maps the 768-dimensional distilbert pooled embedding
        # (corresponding to the first token) into 8 logits (one per emotion)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

        # initialize classifier weights with xavier uniform for more stable training
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: tensor of shape (batch_size, seq_len) with token ids
        attention_mask: tensor of shape (batch_size, seq_len) with 1 for real tokens and 0 for padding

        returns:
            logits: raw scores before sigmoid (batch_size, num_labels)
            probs: sigmoid(logits), values in [0,1] interpreted as emotion probabilities
        """
        # run distilbert encoder to get hidden states for each token
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # distilbert does not have a separate pooler, so we use the hidden state
        # of the first token (position 0) as a fixed-size sentence embedding
        pooled = outputs.last_hidden_state[:, 0]

        # apply dropout to the pooled embedding
        pooled = self.dropout(pooled)

        # project to logits for each emotion
        logits = self.classifier(pooled)

        # apply sigmoid to convert logits to independent probabilities in [0,1]
        probs = torch.sigmoid(logits)

        return logits, probs


# step 11: set up loss function, optimizer, and learning rate scheduler

# create model instance and move it to the chosen device (gpu or cpu)
print("\ninitializing model...")
model = PlutchikEmotionClassifier(
    num_labels=len(plutchik_emotions), dropout=args.dropout
)
model.to(device)

# binary cross entropy with logits is standard for multi-label classification
# it combines a sigmoid layer and the binary cross entropy loss in a stable way
criterion = nn.BCEWithLogitsLoss()

# adamw is a variant of adam that works well with transformers and uses decoupled weight decay
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

# total number of training steps is number of batches per epoch times number of epochs
total_steps = len(train_loader) * args.epochs

# warmup steps define how many updates are used to gradually ramp up the learning rate from 0
warmup_steps = total_steps // 10

# linear learning rate schedule with warmup increases lr for warmup_steps
# and then linearly decays it back to zero by the end of training
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"total training steps: {total_steps}, warmup steps: {warmup_steps}")

# training and validation loop

best_val_macro_f1 = 0.0
train_losses = []
val_losses = []
val_f1_scores = []

start_time = time.time()

for epoch in range(args.epochs):
    print(f"\nstarting epoch {epoch + 1} / {args.epochs}")

    # put model in training mode so dropout and other training-specific behaviors are enabled
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc="training", ncols=80):
        # move batch tensors onto the same device as the model
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # clear any existing gradients from the previous step
        optimizer.zero_grad()

        # forward pass: obtain logits and probabilities from the model
        logits, _ = model(input_ids, attention_mask)

        # compute loss by comparing logits against ground-truth labels
        loss = criterion(logits, labels)

        # backward pass: compute gradients of loss with respect to model parameters
        loss.backward()

        # clip gradients to prevent exploding gradients, which can destabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # take an optimization step using the computed gradients
        optimizer.step()

        # update learning rate according to the scheduler
        scheduler.step()

        total_train_loss += loss.item()

    # compute average training loss for this epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"epoch {epoch + 1} training loss: {avg_train_loss:.4f}")

    # switch model to evaluation mode to disable dropout etc. during validation
    model.eval()
    total_val_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating", ncols=80):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward pass without gradient tracking
            logits, probs = model(input_ids, attention_mask)

            # compute validation loss
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            # convert probabilities to binary predictions using a fixed threshold of 0.5
            preds = (probs >= 0.5).cpu().numpy()

            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # macro f1 treats all classes equally by averaging f1 across emotions
    val_macro_f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )

    # micro f1 aggregates contributions of all classes to compute global precision and recall
    val_micro_f1 = f1_score(
        all_labels, all_predictions, average="micro", zero_division=0
    )

    val_f1_scores.append(val_macro_f1)

    # compute per-emotion f1 scores to see which emotions the model is learning well
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

    # if this epoch achieves the best macro f1 so far, save the model checkpoint
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

# step 12: final test evaluation using the best checkpoint

print("\nloading best checkpoint for final test evaluation...")
best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.to(device)
model.eval()

total_test_loss = 0.0
all_test_predictions = []
all_test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="testing", ncols=80):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, probs = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_test_loss += loss.item()

        preds = (probs >= 0.5).cpu().numpy()
        all_test_predictions.extend(preds)
        all_test_labels.extend(labels.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
all_test_predictions = np.array(all_test_predictions)
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
print(f"final test macro f1: {test_macro_f1:.4f}")
print(f"final test micro f1: {test_micro_f1:.4f}")
print("final per-emotion test f1 scores:")
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_test_labels[:, i].sum())
    p = precision_score(
        all_test_labels[:, i], all_test_predictions[:, i], zero_division=0
    )
    r = recall_score(all_test_labels[:, i], all_test_predictions[:, i], zero_division=0)
    print(
        f"  {emotion:12s}: f1={per_emotion_f1[i]:.4f}, precision={p:.3f}, recall={r:.3f}, support={support}"
    )

# save summary of training and test results to a json file for later inspection
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
    "test_macro_f1": float(test_macro_f1),
    "test_micro_f1": float(test_micro_f1),
    "per_emotion_f1": {
        emotion: float(per_emotion_f1[i]) for i, emotion in enumerate(plutchik_emotions)
    },
    "training_time_seconds": time.time() - start_time,
}

results_path = save_dir / "test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\ntraining complete. results saved to {results_path}")
print(f"best validation macro f1: {best_val_macro_f1:.4f}")
print(f"test macro f1: {test_macro_f1:.4f}")
print(f"model checkpoint: {save_dir / 'best_model.pt'}")
