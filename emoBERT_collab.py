#!/usr/bin/env python3
"""
emoBERT_colab.py

train a multi-label emotion classifier using distilbert with advanced improvements.
this script:
- loads and maps goemotions to plutchik's 8 emotions
- loads xed, which is already labeled in plutchik space
- loads synthetic_claude.csv (high-quality claude-generated data) and adds it to training
- uses focal loss for handling class imbalance
- applies layerwise learning rate decay for better fine-tuning
- uses label smoothing to prevent overconfidence
- trains a distilbert-based classifier and saves the best checkpoint

optimized for google colab gpu training.
"""

import argparse
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
from transformers import AutoTokenizer, DistilBertModel, get_cosine_schedule_with_warmup

# step 1: check gpu availability and display configuration


print("emoBERT - Colab GPU Training with Advanced Improvements")
print("Goal: ALL emotions >67% F1")


# detect whether a gpu is available; use gpu if possible for much faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"+ GPU detected: {torch.cuda.get_device_name(0)}")
    print(
        f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
else:
    print("WARNING: WARNING: No GPU detected! Training will be very slow.")
    print("   Go to Runtime >> Change runtime type >> Hardware accelerator >> GPU")

# step 2: configuration - optimized for colab gpu

# these settings are tuned for gpu training with the full dataset plus synthetic data
config = {
    # number of passes over the training data
    "epochs": 10,
    # how many examples to process in one batch (larger for gpu)
    "batch_size": 64,
    # learning rate for adamw optimizer (small value because transformers are sensitive)
    "learning_rate": 2e-5,
    # dropout rate applied to pooled distilbert embedding (lower to preserve rare signals)
    "dropout": 0.1,
    # fraction of training steps used for learning rate warmup
    "warmup_ratio": 0.1,
    # weight decay for adamw optimizer (l2 regularization)
    "weight_decay": 0.01,
    # focal loss gamma parameter (higher = focus more on hard examples)
    "focal_gamma": 2.0,
    # label smoothing epsilon (prevents overconfidence)
    "label_smoothing": 0.1,
    # number of gradient accumulation steps (simulate larger batch)
    "gradient_accumulation": 1,
    # layerwise learning rate decay factor (deeper layers get lower lr)
    "layerwise_lr_decay": 0.95,
    # maximum sequence length in tokens
    "max_length": 128,
}

print(f"\nTraining Configuration:")
for key, value in config.items():
    print(f"  {key:25s}: {value}")


# ensure that the save directory exists
save_dir = Path("models_colab")
save_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()

# step 3: define plutchik emotions and mapping from goemotions

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

# xed dataset uses plutchik emotions directly (simpler mapping)
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
    """
    this dataset holds raw text strings and their corresponding plutchik label vectors.
    it tokenizes the text on-the-fly into input ids and attention masks using the
    same tokenizer as the distilbert model.

    now includes label smoothing to prevent overconfidence:
    - positive labels (1) become (1 - epsilon)
    - negative labels (0) become epsilon / 2
    """

    def __init__(self, texts, labels, tokenizer, max_length=128, label_smoothing=0.0):
        """
        texts: list of strings (input sentences or comments)
        labels: list of lists with length 8 (plutchik one-hot or multi-hot vectors)
        tokenizer: huggingface tokenizer compatible with distilbert
        max_length: maximum number of tokens per sequence (longer texts are truncated)
        label_smoothing: epsilon value for label smoothing (0 = no smoothing)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_smoothing = label_smoothing

    def __len__(self):
        # number of examples in dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # fetch raw text and label for a given index
        text = str(self.texts[idx])
        label = self.labels[idx]

        # apply label smoothing if enabled
        # this prevents the model from becoming overconfident
        if self.label_smoothing > 0:
            label = np.array(label, dtype=np.float32)
            # positive labels: 1 >> 1 - epsilon
            # negative labels: 0 >> epsilon / 2
            label = label * (1 - self.label_smoothing) + self.label_smoothing / 2

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
            # convert label to float tensor for bcewithlogitsloss (or focal loss)
            "labels": torch.FloatTensor(label),
        }


# step 5: focal loss for handling class imbalance


class FocalLoss(nn.Module):
    """
    focal loss focuses training on hard-to-classify examples by down-weighting
    easy examples. this is especially useful for imbalanced multi-label classification
    where some emotions are much rarer than others.

    formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma controls how much to down-weight easy examples:
    - gamma = 0: equivalent to standard cross entropy
    - gamma = 2: typical value, focuses on hard examples
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        alpha: class weights (tensor of shape [num_classes])
        gamma: focusing parameter (higher = more focus on hard examples)
        reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw model outputs before sigmoid (batch_size, num_labels)
        targets: ground truth labels (batch_size, num_labels)
        """
        # compute standard binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # get predicted probabilities
        probs = torch.sigmoid(logits)

        # compute p_t: probability of the true class
        # if target = 1, use prob; if target = 0, use (1 - prob)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # apply focal term: (1 - p_t)^gamma
        # easy examples (high p_t) get down-weighted
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        # apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # reduce to scalar
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# step 6: model definition with improved initialization


class PlutchikEmotionClassifier(nn.Module):
    """
    this model wraps a pretrained distilbert backbone and adds a simple linear layer
    that predicts 8 independent emotion probabilities (one per plutchik emotion).

    architecture:
        distilbert (pretrained) >> dropout >> linear(768 >> 8) >> sigmoid

    improvements over baseline:
    - xavier initialization for classifier layer
    - configurable dropout rate
    """

    def __init__(self, num_labels=8, dropout=0.1):
        super().__init__()

        # load pretrained distilbert used as the sentence encoder
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # dropout helps reduce overfitting by randomly zeroing some features during training
        self.dropout = nn.Dropout(dropout)

        # linear classifier maps the 768-dimensional distilbert pooled embedding
        # (corresponding to the first token) into 8 logits (one per emotion)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

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
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

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


# step 7: layerwise learning rate decay helper


def get_layerwise_params(model, lr, decay_rate, weight_decay):
    """
    apply layer-wise learning rate decay to the model parameters.

    the intuition is that earlier layers (closer to input) should be updated more
    conservatively since they capture general linguistic features, while later layers
    (closer to output) can be updated more aggressively since they're more task-specific.

    parameter groups:
    - classifier head: full learning rate
    - distilbert layer 5 (top): lr * decay^0
    - distilbert layer 4: lr * decay^1
    - ...
    - distilbert layer 0 (bottom): lr * decay^5
    - embeddings: lr * decay^6
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

    # distilbert has 6 transformer layers
    num_layers = 6
    for layer in range(num_layers - 1, -1, -1):
        # deeper layers get exponentially decayed learning rates
        layer_lr = lr * (decay_rate ** (num_layers - 1 - layer))

        # parameters with weight decay
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in model.distilbert.transformer.layer[
                        layer
                    ].named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            }
        )
        # parameters without weight decay (biases, layer norms)
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in model.distilbert.transformer.layer[
                        layer
                    ].named_parameters()
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
                for n, p in model.distilbert.embeddings.named_parameters()
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
                for n, p in model.distilbert.embeddings.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": embed_lr,
        }
    )

    return optimizer_grouped_parameters


# step 8: load tokenizer

print("\n" + "=" * 80)
print("LOADING DATA")


# the tokenizer converts raw text strings into integer token ids and attention masks
# we use the same base model name as the distilbert model to ensure compatibility
print("\nloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# step 9: load and process goemotions dataset


# helper to convert a goemotions split into plutchik vectors
def convert_goemotions_split(dataset, split_name):
    """
    converts one split of the goemotions dataset (train / validation / test)
    into a list of texts and corresponding plutchik multi-label vectors.

    dataset: the loaded goemotions dataset from huggingface datasets
    split_name: 'train', 'validation', or 'test'

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

    return texts, labels


print("\nloading goemotions from huggingface...")
goemotions = load_dataset("go_emotions", "simplified")

# train / validation / test splits
go_train_texts, go_train_labels = convert_goemotions_split(goemotions, "train")
go_val_texts, go_val_labels = convert_goemotions_split(goemotions, "validation")
go_test_texts, go_test_labels = convert_goemotions_split(goemotions, "test")

print(
    f"goemotions sizes: train={len(go_train_texts):,}, val={len(go_val_texts):,}, test={len(go_test_texts):,}"
)

# step 10: load and process xed dataset


def load_xed():
    """
    loads the xed (extended emotion dataset) and converts to plutchik space.
    xed labels are already plutchik emotions, making conversion simpler.
    creates 80/10/10 train/val/test split.

    returns:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    print("\nloading xed (extended emotion dataset)...")

    # load the english annotations directly
    xed_raw = load_dataset(
        "google-research-datasets/xed_english_finnish", "en_annotated"
    )

    all_texts = []
    all_labels = []

    for example in tqdm(xed_raw["train"], desc="processing xed"):
        label_vec = [0] * len(plutchik_emotions)

        # xed provides emotion scores; we check which emotions are present
        for xed_emotion, plut_emotions in xed_to_plutchik.items():
            if example.get(xed_emotion, 0) > 0:
                for plut_emotion in plut_emotions:
                    idx = plutchik_emotions.index(plut_emotion)
                    label_vec[idx] = 1

        # only include examples with at least one emotion
        if sum(label_vec) > 0:
            all_texts.append(example["text"])
            all_labels.append(label_vec)

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
        f"xed sizes: train={len(train_texts):,}, val={len(val_texts):,}, test={len(test_texts):,}"
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
    ) = load_xed()
    xed_available = True
except Exception as e:
    print(f"could not load xed, continuing with goemotions only. error was: {e}")
    xed_available = False

# step 11: load claude synthetic data


def load_synthetic_data(csv_path):
    """
    loads high-quality synthetic training data generated by claude.
    the csv has a 'text' column plus 8 label columns aligned with plutchik_emotions.

    returns:
        texts: list of strings
        labels: list of length-8 lists (plutchik label vectors)
    """
    print(f"\nloading claude synthetic data from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)

        if "text" not in df.columns:
            raise ValueError("synthetic csv must contain a 'text' column")

        # ensure all expected label columns are present
        expected_label_cols = [f"{emotion}_label" for emotion in plutchik_emotions]
        for col in expected_label_cols:
            if col not in df.columns:
                raise ValueError(
                    f"synthetic csv is missing required label column: {col}"
                )

        texts = df["text"].tolist()
        labels = df[expected_label_cols].values.tolist()

        print(f"+ loaded {len(texts):,} claude synthetic samples")

        # show distribution
        labels_array = np.array(labels)
        print("\nsynthetic data distribution:")
        for i, emotion in enumerate(plutchik_emotions):
            count = int(labels_array[:, i].sum())
            print(f"  {emotion:15s}: {count:6,} samples")

        return texts, labels

    except FileNotFoundError:
        print("WARNING: WARNING: synthetic_claude.csv not found!")
        print("   upload the file or training will proceed without synthetic data")
        return [], []
    except Exception as e:
        print(f"WARNING: error loading synthetic data: {e}")
        return [], []


# this path works in colab after you upload the file
synthetic_csv_path = Path("synthetic_claude.csv")
synthetic_texts, synthetic_labels = load_synthetic_data(synthetic_csv_path)
synthetic_available = len(synthetic_texts) > 0

# step 12: combine all datasets

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

print(f"\n{'=' * 80}")
print("FINAL DATASET SIZES")
print(f"train: {len(train_texts):,} examples")
print(f"val:   {len(val_texts):,} examples")
print(f"test:  {len(test_texts):,} examples")

# step 13: calculate class weights for focal loss

# class weights help the model focus on rare emotions by giving them higher importance
train_labels_array = np.array(train_labels)
pos_counts = train_labels_array.sum(axis=0)
total = len(train_labels)

# weight = total / (2 * positive_count) is a common heuristic
class_weights = torch.FloatTensor(
    [total / (2 * count) if count > 0 else 1.0 for count in pos_counts]
)

print(f"\n{'=' * 80}")
print("CLASS DISTRIBUTION & WEIGHTS")
for i, emotion in enumerate(plutchik_emotions):
    pct = (pos_counts[i] / total) * 100
    print(
        f"{emotion:15s}: {int(pos_counts[i]):6,} ({pct:5.2f}%) weight: {class_weights[i]:.3f}"
    )

# step 14: create pytorch datasets and dataloaders

# wrap raw lists in our custom EmotionDataset, which handles tokenization and tensor conversion
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

# dataloaders handle batching and optional shuffling
# we shuffle the training set each epoch, but validation/test remain in fixed order
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"] * 2,  # larger batch for inference
)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"] * 2)

print(f"\nnumber of training batches: {len(train_loader):,}")
print(f"number of validation batches: {len(val_loader):,}")
print(f"number of test batches: {len(test_loader):,}")

# step 15: initialize model, loss, optimizer, and scheduler

print(f"\n{'=' * 80}")
print("CREATING MODEL")

# create model instance and move it to the chosen device (gpu or cpu)
model = PlutchikEmotionClassifier(
    num_labels=len(plutchik_emotions), dropout=config["dropout"]
)
model.to(device)

# focal loss with class weights for handling imbalanced data
criterion = FocalLoss(alpha=class_weights.to(device), gamma=config["focal_gamma"])

# apply layerwise learning rate decay for better fine-tuning
optimizer_grouped_parameters = get_layerwise_params(
    model, config["learning_rate"], config["layerwise_lr_decay"], config["weight_decay"]
)

# adamw is a variant of adam that works well with transformers
optimizer = AdamW(optimizer_grouped_parameters)

# total number of training steps is number of batches per epoch times number of epochs
total_steps = len(train_loader) * config["epochs"]

# warmup steps define how many updates are used to gradually ramp up the learning rate from 0
num_warmup_steps = int(total_steps * config["warmup_ratio"])

# cosine learning rate schedule with warmup
# lr increases linearly during warmup, then follows a cosine decay to zero
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)

print(f"training steps: {total_steps:,} | warmup: {num_warmup_steps:,}")

# step 16: training and validation loop

print(f"\n{'=' * 80}")
print("STARTING TRAINING")

best_val_macro_f1 = 0.0
train_losses = []
val_losses = []
val_f1_scores = []

for epoch in range(config["epochs"]):
    epoch_start = time.time()

    print(f"\n{'=' * 80}")
    print(f"EPOCH {epoch + 1}/{config['epochs']}")
    print(f"{'=' * 80}")

    # put model in training mode so dropout and other training-specific behaviors are enabled
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc="training"):
        # move batch tensors onto the same device as the model
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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

        # clear gradients for next iteration
        optimizer.zero_grad()

        total_train_loss += loss.item()

    # compute average training loss for this epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # switch model to evaluation mode to disable dropout etc. during validation
    model.eval()
    total_val_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating"):
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

    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]

    print(f"\nresults:")
    print(f"  train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")
    print(f"  val macro f1: {val_macro_f1:.4f} | val micro f1: {val_micro_f1:.4f}")
    print(f"  learning rate: {current_lr:.2e} | time: {epoch_time:.1f}s")

    print("\nper-emotion f1:")
    emotions_above_67 = 0
    for i, emotion in enumerate(plutchik_emotions):
        status = "✓" if per_emotion_f1[i] >= 0.67 else "✗"
        print(f"  {status} {emotion:15s}: {per_emotion_f1[i]:.4f}")
        if per_emotion_f1[i] >= 0.67:
            emotions_above_67 += 1

    min_f1 = per_emotion_f1.min()
    print(f"\nmin f1: {min_f1:.4f} | emotions ≥67%: {emotions_above_67}/8")

    if min_f1 >= 0.67:
        print("ALL EMOTIONS ABOVE 67% F1!")

    # if this epoch achieves the best macro f1 so far, save the model checkpoint
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
            f"\n>> saved new best model to {ckpt_path} (macro f1 = {val_macro_f1:.4f}, min f1 = {min_f1:.4f})"
        )

# step 17: final test evaluation using the best checkpoint

print(f"\n{'=' * 80}")
print("FINAL TEST EVALUATION")

print("\nloading best checkpoint for final test evaluation...")
best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.to(device)
model.eval()

total_test_loss = 0.0
all_test_predictions = []
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

print(f"\ntest results:")
print(f"  macro f1: {test_macro_f1:.4f} | micro f1: {test_micro_f1:.4f}")
print(f"  min f1: {per_emotion_f1.min():.4f}")

print("\nper-emotion test f1:")
emotions_above_67 = 0
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_test_labels[:, i].sum())
    p = precision_score(
        all_test_labels[:, i], all_test_predictions[:, i], zero_division=0
    )
    r = recall_score(all_test_labels[:, i], all_test_predictions[:, i], zero_division=0)
    status = "✓" if per_emotion_f1[i] >= 0.67 else "✗"
    print(
        f"  {status} {emotion:15s}: f1={per_emotion_f1[i]:.4f}, precision={p:.3f}, recall={r:.3f}, support={support:,}"
    )
    if per_emotion_f1[i] >= 0.67:
        emotions_above_67 += 1

# save summary of training and test results to a json file for later inspection
results = {
    "config": config,
    "best_val_macro_f1": float(best_val_macro_f1),
    "test_macro_f1": float(test_macro_f1),
    "test_micro_f1": float(test_micro_f1),
    "test_min_f1": float(per_emotion_f1.min()),
    "per_emotion_f1": {
        emotion: float(per_emotion_f1[i]) for i, emotion in enumerate(plutchik_emotions)
    },
    "training_time_seconds": time.time() - start_time,
    "all_emotions_above_67": bool(per_emotion_f1.min() >= 0.67),
    "emotions_above_67_count": int(emotions_above_67),
}

results_path = save_dir / "test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

total_time = time.time() - start_time

print(f"\n{'=' * 80}")
print("TRAINING COMPLETE!")
print(f"total time: {total_time / 60:.1f} minutes")
print(f"best validation macro f1: {best_val_macro_f1:.4f}")
print(f"test macro f1: {test_macro_f1:.4f}")
print(f"test min f1: {per_emotion_f1.min():.4f}")
print(f"emotions ≥67%: {emotions_above_67}/8")

if per_emotion_f1.min() >= 0.67:
    print("\nSUCCESS! ALL EMOTIONS ABOVE 67% F1!")
else:
    below_67 = [
        plutchik_emotions[i] for i, f1 in enumerate(per_emotion_f1) if f1 < 0.67
    ]
    print(f"\nWARNING: emotions below 67%: {below_67}")
    print("   consider: longer training, more synthetic data, or adaptive thresholds")

print(f"\nresults saved to {results_path}")
print(f"model checkpoint: {save_dir / 'best_model.pt'}")
print("\nto download: click folder icon >> right-click file >> download")
