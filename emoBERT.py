#!/usr/bin/env python3
"""
emoBERT.py - simple linear script for multi-label emotion classification
trains distilbert on goemotions + xed to predict plutchik's 8 emotions
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import DistilBertModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
import json
import time
import argparse

# parse command line arguments for training configuration
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--max_samples', type=int, default=None)
parser.add_argument('--save_dir', type=str, default='models')
args = parser.parse_args()

# set up device (gpu if available, otherwise cpu) and create save directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
start_time = time.time()

print(f"\ntraining on {device} | epochs: {args.epochs} | batch: {args.batch_size} | lr: {args.learning_rate}")

# plutchik's 8 basic emotions - this is what we're trying to predict
plutchik_emotions = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']

# goemotions has 27 fine-grained emotions that we need to map to plutchik's 8
# many goemotions map to multiple plutchik emotions (multi-label)
# for example: "love" involves both joy and trust
#              "excitement" involves joy, anticipation, and surprise
#              "nervousness" involves fear and anticipation (anxious waiting)
goemotions_to_plutchik = {
    'admiration': ['joy', 'trust'],         # admiring someone = positive feeling + trust
    'amusement': ['joy'],                   # being entertained = joy
    'approval': ['joy', 'trust'],           # approving = positive + trust in correctness
    'caring': ['joy', 'trust'],             # caring for someone = warmth + trust
    'desire': ['joy', 'anticipation'],      # wanting something = positive + looking forward
    'excitement': ['joy', 'anticipation', 'surprise'],  # high arousal positive emotion
    'gratitude': ['joy', 'trust'],          # thankfulness = joy + trust in giver
    'joy': ['joy'],                         # pure joy
    'love': ['joy', 'trust'],               # love = joy + deep trust
    'optimism': ['joy', 'anticipation'],    # hopefulness = joy + positive expectation
    'pride': ['joy'],                       # pride = positive self-regard
    'relief': ['joy', 'surprise'],          # relief = joy + unexpected resolution
    'sadness': ['sadness'],                 # pure sadness
    'disappointment': ['sadness', 'surprise'],  # unmet expectations = sad + surprised
    'embarrassment': ['sadness', 'fear'],   # social shame = sadness + fear of judgment
    'grief': ['sadness'],                   # deep sadness
    'remorse': ['sadness', 'disgust'],      # regret = sadness + self-disgust
    'anger': ['anger'],                     # pure anger
    'annoyance': ['anger', 'disgust'],      # irritation = anger + disgust
    'disapproval': ['anger', 'disgust'],    # rejection = anger + disgust
    'fear': ['fear'],                       # pure fear
    'nervousness': ['fear', 'anticipation'], # anxious = fear + worried anticipation
    'disgust': ['disgust'],                 # pure disgust
    'surprise': ['surprise'],               # pure surprise
    'realization': ['surprise'],            # sudden understanding = surprise
    'confusion': ['surprise', 'fear'],      # confusion = surprised + uncertain/fearful
    'curiosity': ['anticipation', 'surprise'], # curiosity = looking forward + openness to surprise
    'neutral': []                           # neutral = no emotions activated
}

# dataset class that handles tokenization and returns tensors for training
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # tokenize the text: convert words to token ids, add special tokens ([CLS], [SEP])
        # pad/truncate to max_length, create attention mask to ignore padding
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

print("\nloading tokenizer...")
# load distilbert tokenizer - converts text to token ids that the model understands
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

print("loading goemotions dataset...")
# load goemotions from huggingface - has ~58k reddit comments labeled with 27 emotions
goemotions = load_dataset("go_emotions", "simplified")
emotion_names = goemotions['train'].features['labels'].feature.names

# lists to store processed data - we'll convert goemotions labels to plutchik vectors
goemotions_train_texts = []
goemotions_train_labels = []
goemotions_val_texts = []
goemotions_val_labels = []
goemotions_test_texts = []
goemotions_test_labels = []

# process train split
# for each example, convert its goemotions labels to a plutchik binary vector
# the vector has 8 positions (one per plutchik emotion), each is 0 or 1
# multi-label means multiple positions can be 1 (e.g., "love" → [1,0,0,0,1,0,0,0] for joy+trust)
for example in tqdm(goemotions['train'], desc="processing goemotions train"):
    # start with all zeros [0,0,0,0,0,0,0,0] for the 8 plutchik emotions
    plutchik_vector = [0] * 8

    # look at each emotion label this example has
    for label_idx in example['labels']:
        if label_idx < len(emotion_names):
            ge_emotion = emotion_names[label_idx]  # get emotion name like "love" or "joy"

            # look up which plutchik emotion(s) this maps to
            if ge_emotion in goemotions_to_plutchik:
                # set each mapped plutchik emotion to 1 in the vector
                for plut_emotion in goemotions_to_plutchik[ge_emotion]:
                    idx = plutchik_emotions.index(plut_emotion)
                    plutchik_vector[idx] = 1  # activate this emotion

    goemotions_train_texts.append(example['text'])
    goemotions_train_labels.append(plutchik_vector)

    # stop if we hit max_samples limit (for quick testing)
    if args.max_samples and len(goemotions_train_texts) >= args.max_samples:
        break

# process validation split - same logic as train
for example in tqdm(goemotions['validation'], desc="processing goemotions val"):
    plutchik_vector = [0] * 8
    for label_idx in example['labels']:
        if label_idx < len(emotion_names):
            ge_emotion = emotion_names[label_idx]
            if ge_emotion in goemotions_to_plutchik:
                for plut_emotion in goemotions_to_plutchik[ge_emotion]:
                    idx = plutchik_emotions.index(plut_emotion)
                    plutchik_vector[idx] = 1
    goemotions_val_texts.append(example['text'])
    goemotions_val_labels.append(plutchik_vector)
    if args.max_samples and len(goemotions_val_texts) >= args.max_samples // 4:
        break

# process test split - same logic as train
for example in tqdm(goemotions['test'], desc="processing goemotions test"):
    plutchik_vector = [0] * 8
    for label_idx in example['labels']:
        if label_idx < len(emotion_names):
            ge_emotion = emotion_names[label_idx]
            if ge_emotion in goemotions_to_plutchik:
                for plut_emotion in goemotions_to_plutchik[ge_emotion]:
                    idx = plutchik_emotions.index(plut_emotion)
                    plutchik_vector[idx] = 1
    goemotions_test_texts.append(example['text'])
    goemotions_test_labels.append(plutchik_vector)
    if args.max_samples and len(goemotions_test_texts) >= args.max_samples // 4:
        break

# calculate multi-label statistics to see how many samples have multiple emotions
label_array = np.array(goemotions_train_labels)
avg_labels = label_array.sum(axis=1).mean()  # average number of emotions per sample
multi_label_count = (label_array.sum(axis=1) > 1).sum()  # how many have 2+ emotions
multi_label_pct = multi_label_count / len(label_array) * 100

print(f"goemotions: {len(goemotions_train_texts):,} train, {len(goemotions_val_texts):,} val, {len(goemotions_test_texts):,} test")
print(f"  avg emotions per sample: {avg_labels:.2f} | multi-label: {multi_label_pct:.1f}%")

print("\nloading xed dataset...")
# xed (extended emotion dataset) uses plutchik's 8 emotions directly!
# it's from helsinki-nlp and has sentences labeled with plutchik emotions
try:
    # load from raw tsv file since dataset script is deprecated
    xed = load_dataset(
        "csv",
        data_files={"train": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/AnnotatedData/en-annotated.tsv"},
        delimiter="\t",
        column_names=["sentence", "labels"]
    )

    # process xed: convert each sentence and its plutchik label to a binary vector
    xed_texts = []
    xed_labels = []

    for example in tqdm(xed['train'], desc="processing xed"):
        plutchik_vector = [0] * 8

        # xed label is already a plutchik emotion name like "anger", "joy", "trust"
        label_str = example['labels'].strip().lower()

        # set the corresponding position to 1 in the vector
        if label_str in plutchik_emotions:
            idx = plutchik_emotions.index(label_str)
            plutchik_vector[idx] = 1
        # if it's "neutral", leave all zeros

        xed_texts.append(example['sentence'])
        xed_labels.append(plutchik_vector)

        if args.max_samples and len(xed_texts) >= args.max_samples:
            break

    # xed doesn't have predefined splits, so we create our own (80/10/10)
    total = len(xed_texts)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    # shuffle with fixed seed for reproducibility
    indices = np.random.RandomState(42).permutation(total)

    # split by indices
    xed_train_texts = [xed_texts[i] for i in indices[:train_size]]
    xed_train_labels = [xed_labels[i] for i in indices[:train_size]]
    xed_val_texts = [xed_texts[i] for i in indices[train_size:train_size+val_size]]
    xed_val_labels = [xed_labels[i] for i in indices[train_size:train_size+val_size]]
    xed_test_texts = [xed_texts[i] for i in indices[train_size+val_size:]]
    xed_test_labels = [xed_labels[i] for i in indices[train_size+val_size:]]

    print(f"xed: {len(xed_train_texts):,} train, {len(xed_val_texts):,} val, {len(xed_test_texts):,} test")
    xed_available = True

except Exception as e:
    print(f"could not load xed, continuing with goemotions only")
    xed_available = False

# combine goemotions and xed datasets for more training data
if xed_available:
    train_texts = goemotions_train_texts + xed_train_texts
    train_labels = goemotions_train_labels + xed_train_labels
    val_texts = goemotions_val_texts + xed_val_texts
    val_labels = goemotions_val_labels + xed_val_labels
    test_texts = goemotions_test_texts + xed_test_texts
    test_labels = goemotions_test_labels + xed_test_labels
    print(f"combined: {len(train_texts):,} train, {len(val_texts):,} val, {len(test_texts):,} test")
else:
    train_texts = goemotions_train_texts
    train_labels = goemotions_train_labels
    val_texts = goemotions_val_texts
    val_labels = goemotions_val_labels
    test_texts = goemotions_test_texts
    test_labels = goemotions_test_labels

# create pytorch datasets that handle tokenization and batching
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

# create dataloaders that iterate over batches during training
# shuffle train data each epoch, don't shuffle val/test
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

print(f"\ncreated dataloaders with {len(train_loader)} train batches")

# define the model architecture
class PlutchikEmotionClassifier(nn.Module):
    def __init__(self, num_labels=8, dropout=0.3):
        super().__init__()
        # load pretrained distilbert - a smaller/faster version of bert
        # it's already learned general language understanding from massive text corpus
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # dropout randomly zeros some neurons during training to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        # classifier layer: transforms distilbert output (768 dims) to 8 emotion predictions
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

        # initialize classifier weights with xavier uniform for stable training
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        # run distilbert on the input tokens
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # use the [CLS] token representation (first token) as sentence encoding
        pooled_output = outputs.last_hidden_state[:, 0]
        # apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        # get logits (raw scores) for each of 8 emotions
        logits = self.classifier(pooled_output)
        # apply sigmoid to get independent probabilities (0-1) for each emotion
        # sigmoid treats each emotion independently (vs softmax which makes them sum to 1)
        probs = torch.sigmoid(logits)
        return logits, probs

print("\ninitializing model...")
# create the model and move it to gpu/cpu
model = PlutchikEmotionClassifier(num_labels=8, dropout=args.dropout)
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"model has {num_params:,} parameters")

# binary cross entropy loss: measures error for multi-label classification
# it compares predicted probabilities [0.2, 0.8, 0.1, ...] to true labels [0, 1, 0, ...]
criterion = nn.BCEWithLogitsLoss()

# adamw optimizer: adaptive learning rate algorithm that works well with transformers
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

# learning rate scheduler: gradually reduces learning rate during training
# starts with warmup (slow ramp up) then linear decay
total_steps = len(train_loader) * args.epochs
warmup_steps = total_steps // 10  # 10% of training is warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"training for {total_steps:,} steps ({warmup_steps:,} warmup)\n")

# track metrics across epochs
best_f1 = 0.0
train_losses = []
val_losses = []
val_f1_scores = []

# main training loop
for epoch in range(args.epochs):
    epoch_start = time.time()
    print(f"epoch {epoch + 1}/{args.epochs}")

    # training phase: update model weights
    model.train()  # set model to training mode (enables dropout)
    total_train_loss = 0

    for batch in tqdm(train_loader, desc='training', ncols=80):
        # get batch data and move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # reset gradients from previous step
        optimizer.zero_grad()

        # forward pass: run model to get predictions
        logits, _ = model(input_ids, attention_mask)

        # calculate loss: how different are predictions from true labels?
        loss = criterion(logits, labels)

        # backward pass: compute gradients of loss with respect to parameters
        loss.backward()

        # clip gradients to prevent exploding gradients (common with transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update model parameters based on gradients
        optimizer.step()
        # update learning rate according to schedule
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validation phase: evaluate model without updating weights
    model.eval()  # set to eval mode (disables dropout)
    total_val_loss = 0
    all_predictions = []
    all_labels = []

    # torch.no_grad() disables gradient calculation to save memory
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='validating', ncols=80):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # get predictions
            logits, probs = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            # convert probabilities to binary predictions using 0.5 threshold
            # if prob >= 0.5, predict emotion is present (1), else absent (0)
            predictions = (probs >= 0.5).cpu().numpy()

            # collect all predictions and labels for metric calculation
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # calculate f1 scores
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # macro f1: average f1 across all 8 emotions (treats each emotion equally)
    # good for imbalanced classes where some emotions are rare
    val_macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    # micro f1: calculate f1 globally across all predictions (favors common emotions)
    val_micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    val_f1_scores.append(val_macro_f1)

    # get f1 for each individual emotion
    per_emotion_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

    epoch_time = time.time() - epoch_start

    # print epoch results
    print(f"  train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")
    print(f"  val_macro_f1: {val_macro_f1:.4f} {'✓' if val_macro_f1 >= 0.70 else ''} | val_micro_f1: {val_micro_f1:.4f} | time: {epoch_time:.1f}s")

    # show per-emotion f1 scores to see which emotions are learned well
    print("  per-emotion f1:", end=" ")
    for i, emotion in enumerate(plutchik_emotions):
        status = "✓" if per_emotion_f1[i] >= 0.70 else ""
        print(f"{emotion}:{per_emotion_f1[i]:.3f}{status}", end=" ")
    print()

    # save model if it's the best so far (highest validation f1)
    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'macro_f1': val_macro_f1,
            'micro_f1': val_micro_f1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores
        }
        torch.save(checkpoint, save_dir / 'best_model.pt')
        print(f"  → saved new best model (macro_f1: {best_f1:.4f})")
    print()

# test the best model on held-out test set
print("\ntesting best model...")
# load the best checkpoint
checkpoint = torch.load(save_dir / 'best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# run evaluation on test set (same as validation but with test data)
model.eval()
total_test_loss = 0
all_predictions = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='testing', ncols=80):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits, probs = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_test_loss += loss.item()
        predictions = (probs >= 0.5).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# calculate final test metrics
test_macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
test_micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
per_emotion_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

print(f"\ntest results:")
print(f"  loss: {avg_test_loss:.4f} | macro_f1: {test_macro_f1:.4f} {'✓' if test_macro_f1 >= 0.70 else '✗'} | micro_f1: {test_micro_f1:.4f}")
print("\nper-emotion test f1 (with precision, recall, support):")
for i, emotion in enumerate(plutchik_emotions):
    support = int(all_labels[:, i].sum())  # how many test samples have this emotion
    precision = precision_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
    recall = recall_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
    status = "✓" if per_emotion_f1[i] >= 0.70 else " "
    print(f"  {status} {emotion:15s}: f1={per_emotion_f1[i]:.4f} p={precision:.3f} r={recall:.3f} n={support:,}")

# save results to json file
results = {
    'config': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_samples': args.max_samples,
        'device': str(device)
    },
    'best_val_macro_f1': float(best_f1),
    'test_macro_f1': float(test_macro_f1),
    'test_micro_f1': float(test_micro_f1),
    'per_emotion_f1': {emotion: float(per_emotion_f1[i]) for i, emotion in enumerate(plutchik_emotions)},
    'training_time_seconds': time.time() - start_time
}

with open(save_dir / 'test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

total_time = time.time() - start_time
print(f"\ntraining complete! total time: {total_time/60:.1f} min")
print(f"best val f1: {best_f1:.4f} | test f1: {test_macro_f1:.4f}")
print(f"model: {save_dir / 'best_model.pt'} | results: {save_dir / 'test_results.json'}")

# test inference on example sentences to see model predictions
print("\ntesting inference on examples...")
examples = [
    "I'm so happy and excited about the future!",
    "This makes me really angry and frustrated.",
    "I feel sad and nostalgic thinking about the past.",
    "I'm nervous but also hopeful about what's coming.",
    "That's absolutely disgusting and revolting.",
    "I'm scared of what might happen next.",
    "Wow, I can't believe this is happening!",
    "I have complete trust and confidence in you.",
    "I'm happy but also sad it's ending.",
    "I'm excited but terrified at the same time.",
]

model.eval()

for text in examples:
    # tokenize the example text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # get predictions
    with torch.no_grad():
        _, probs = model(input_ids, attention_mask)
        probs = probs.squeeze().cpu().numpy()

    print(f"\n'{text}'")

    # sort emotions by probability and show only significant ones (>0.1)
    emotion_probs = [(plutchik_emotions[i], probs[i]) for i in range(8)]
    emotion_probs.sort(key=lambda x: x[1], reverse=True)

    detected = 0
    for emotion, prob in emotion_probs:
        if prob > 0.1:  # only show emotions with >10% probability
            bar = '█' * int(prob * 40)  # visual bar for probability
            pred = '✓' if prob >= 0.5 else ' '  # checkmark if predicted (>=0.5)
            print(f"  {pred} {emotion:15s} {prob:.3f} {bar}")
            if prob >= 0.5:
                detected += 1

    # highlight if multiple emotions were detected (multi-label)
    if detected > 1:
        print(f"  → multi-label: {detected} emotions")

print("\ndone!")
