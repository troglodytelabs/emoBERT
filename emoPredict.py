#!/usr/bin/env python3
"""
emoPredict.py - predict emotions and detect plutchik dyads
uses the trained distilbert model to analyze text and find emotional dyads
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, AutoTokenizer
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm import tqdm

# plutchik's 8 basic emotions
plutchik_emotions = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']

# plutchik's dyads - combinations of emotions that create complex feelings
# primary dyads (adjacent on the wheel)
primary_dyads = {
    ('joy', 'trust'): 'love',
    ('trust', 'fear'): 'submission',
    ('fear', 'surprise'): 'awe',
    ('surprise', 'sadness'): 'disapproval',
    ('sadness', 'disgust'): 'remorse',
    ('disgust', 'anger'): 'contempt',
    ('anger', 'anticipation'): 'aggressiveness',
    ('anticipation', 'joy'): 'optimism',
}

# secondary dyads (skip one emotion)
secondary_dyads = {
    ('joy', 'fear'): 'guilt',
    ('trust', 'surprise'): 'curiosity',
    ('fear', 'sadness'): 'despair',
    ('surprise', 'disgust'): 'unbelief',
    ('sadness', 'anger'): 'envy',
    ('disgust', 'anticipation'): 'cynicism',
    ('anger', 'joy'): 'pride',
    ('anticipation', 'trust'): 'fatalism',
}

# tertiary dyads (skip two emotions)
tertiary_dyads = {
    ('joy', 'surprise'): 'delight',
    ('trust', 'sadness'): 'sentimentality',
    ('fear', 'disgust'): 'shame',
    ('surprise', 'anger'): 'outrage',
    ('sadness', 'anticipation'): 'pessimism',
    ('disgust', 'joy'): 'morbidness',
    ('anger', 'trust'): 'dominance',
    ('anticipation', 'fear'): 'anxiety',
}

# oppositions (opposite on the wheel - conflicting emotions)
opposition_dyads = {
    ('joy', 'sadness'): 'bittersweetness',
    ('trust', 'disgust'): 'ambivalence',
    ('fear', 'anger'): 'conflict',
    ('surprise', 'anticipation'): 'confusion',
}

# combine all dyads for lookup
all_dyads = {}
for dyad_type, dyads in [
    ('primary', primary_dyads),
    ('secondary', secondary_dyads),
    ('tertiary', tertiary_dyads),
    ('opposition', opposition_dyads)
]:
    for (e1, e2), name in dyads.items():
        # store both orderings (joy+trust and trust+joy both map to love)
        all_dyads[(e1, e2)] = {'name': name, 'type': dyad_type}
        all_dyads[(e2, e1)] = {'name': name, 'type': dyad_type}

# model architecture (same as training)
class PlutchikEmotionClassifier(nn.Module):
    def __init__(self, num_labels=8, dropout=0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.sigmoid(logits)
        return logits, probs

def load_model(model_path, device='cpu'):
    """load the trained model from checkpoint"""
    print(f"loading model from {model_path}...")

    # create model
    model = PlutchikEmotionClassifier(num_labels=8)

    # load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  validation macro f1: {checkpoint['macro_f1']:.4f}")

    return model

def predict_emotions(model, tokenizer, text, device='cpu', threshold=0.5):
    """
    predict plutchik emotions for a text
    returns dict with emotion probabilities and binary predictions
    """
    # tokenize text
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

    # convert to dict
    emotion_probs = {emotion: float(probs[i]) for i, emotion in enumerate(plutchik_emotions)}

    # binary predictions using threshold
    emotion_binary = {emotion: 1 if prob >= threshold else 0 for emotion, prob in emotion_probs.items()}

    return {
        'probabilities': emotion_probs,
        'predictions': emotion_binary,
        'predicted_emotions': [e for e, v in emotion_binary.items() if v == 1]
    }

def detect_dyads(emotion_probs, threshold=0.3, method='product'):
    """
    detect plutchik dyads from emotion probabilities

    method can be:
    - 'product': e1 * e2 (both must be present)
    - 'minimum': min(e1, e2) (limited by weaker emotion)
    - 'average': (e1 + e2) / 2 (simple average)
    """
    detected_dyads = {
        'primary': [],
        'secondary': [],
        'tertiary': [],
        'opposition': []
    }

    # check all dyad types
    for dyad_type, dyads in [
        ('primary', primary_dyads),
        ('secondary', secondary_dyads),
        ('tertiary', tertiary_dyads),
        ('opposition', opposition_dyads)
    ]:
        for (e1, e2), dyad_name in dyads.items():
            # get probabilities for both emotions
            prob1 = emotion_probs.get(e1, 0)
            prob2 = emotion_probs.get(e2, 0)

            # calculate dyad strength
            if method == 'product':
                strength = prob1 * prob2
            elif method == 'minimum':
                strength = min(prob1, prob2)
            elif method == 'average':
                strength = (prob1 + prob2) / 2
            else:
                strength = prob1 * prob2

            # add if above threshold
            if strength >= threshold:
                detected_dyads[dyad_type].append({
                    'name': dyad_name,
                    'emotions': (e1, e2),
                    'strength': strength,
                    'prob1': prob1,
                    'prob2': prob2
                })

    # sort each type by strength
    for dyad_type in detected_dyads:
        detected_dyads[dyad_type].sort(key=lambda x: x['strength'], reverse=True)

    return detected_dyads

def get_dominant_dyads(emotion_probs, top_n=3, threshold=0.3):
    """get top n strongest dyads regardless of type"""
    all_detected = detect_dyads(emotion_probs, threshold=threshold)

    # flatten all dyads into single list
    flat_dyads = []
    for dyad_type, dyads in all_detected.items():
        for dyad in dyads:
            dyad['type'] = dyad_type
            flat_dyads.append(dyad)

    # sort by strength and return top n
    flat_dyads.sort(key=lambda x: x['strength'], reverse=True)
    return flat_dyads[:top_n]

def analyze_text(model, tokenizer, text, device='cpu', emotion_threshold=0.5, dyad_threshold=0.3):
    """
    complete analysis: predict emotions and detect dyads
    returns dict with all results
    """
    # predict emotions
    emotion_results = predict_emotions(model, tokenizer, text, device, emotion_threshold)

    # detect dyads
    dyads = detect_dyads(emotion_results['probabilities'], threshold=dyad_threshold)
    dominant_dyads = get_dominant_dyads(emotion_results['probabilities'], top_n=3, threshold=dyad_threshold)

    return {
        'text': text,
        'emotions': emotion_results,
        'dyads': dyads,
        'dominant_dyads': dominant_dyads
    }

def batch_predict(model, tokenizer, texts, device='cpu', batch_size=32,
                 emotion_threshold=0.5, dyad_threshold=0.3):
    """
    predict emotions and dyads for multiple texts efficiently
    returns list of results dicts
    """
    results = []

    # process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="processing texts"):
        batch_texts = texts[i:i+batch_size]

        # tokenize batch
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # predict
        with torch.no_grad():
            _, probs = model(input_ids, attention_mask)
            probs = probs.cpu().numpy()

        # process each result in batch
        for j, text in enumerate(batch_texts):
            # extract probabilities for this text
            text_probs = {emotion: float(probs[j][k]) for k, emotion in enumerate(plutchik_emotions)}

            # get predictions
            predictions = {e: 1 if p >= emotion_threshold else 0 for e, p in text_probs.items()}
            predicted_emotions = [e for e, v in predictions.items() if v == 1]

            # detect dyads
            dyads = detect_dyads(text_probs, threshold=dyad_threshold)
            dominant_dyads = get_dominant_dyads(text_probs, top_n=3, threshold=dyad_threshold)

            results.append({
                'text': text,
                'emotion_probs': text_probs,
                'predicted_emotions': predicted_emotions,
                'dyads': dyads,
                'dominant_dyads': dominant_dyads
            })

    return results

def save_results_csv(results, output_path):
    """save results to csv for analysis"""
    rows = []

    for result in results:
        row = {
            'text': result['text'],
            # emotion probabilities
            **{f'{e}_prob': result['emotion_probs'][e] for e in plutchik_emotions},
            # predicted emotions (binary)
            **{f'{e}_pred': 1 if e in result['predicted_emotions'] else 0 for e in plutchik_emotions},
            # dominant dyads
            'dyad_1': result['dominant_dyads'][0]['name'] if len(result['dominant_dyads']) > 0 else '',
            'dyad_1_strength': result['dominant_dyads'][0]['strength'] if len(result['dominant_dyads']) > 0 else 0,
            'dyad_1_type': result['dominant_dyads'][0]['type'] if len(result['dominant_dyads']) > 0 else '',
            'dyad_2': result['dominant_dyads'][1]['name'] if len(result['dominant_dyads']) > 1 else '',
            'dyad_2_strength': result['dominant_dyads'][1]['strength'] if len(result['dominant_dyads']) > 1 else 0,
            'dyad_3': result['dominant_dyads'][2]['name'] if len(result['dominant_dyads']) > 2 else '',
            'dyad_3_strength': result['dominant_dyads'][2]['strength'] if len(result['dominant_dyads']) > 2 else 0,
            # counts
            'num_emotions': len(result['predicted_emotions']),
            'num_dyads': sum(len(result['dyads'][t]) for t in result['dyads'])
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"saved results to {output_path}")
    return df

# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict emotions and detect dyads')
    parser.add_argument('--model', type=str, default='/Users/devindyson/Desktop/troglodytelabs/moodlog/emoBERT/models/best_model.pt', help='path to trained model')
    parser.add_argument('--input', type=str, help='input text file or csv')
    parser.add_argument('--output', type=str, default='predictions.csv', help='output csv file')
    parser.add_argument('--text', type=str, help='single text to analyze')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emotion_threshold', type=float, default=0.5)
    parser.add_argument('--dyad_threshold', type=float, default=0.3)
    args = parser.parse_args()

    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    model = load_model(args.model, device)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # single text analysis
    if args.text:
        print(f"\nanalyzing text: '{args.text}'")
        result = analyze_text(model, tokenizer, args.text, device,
                            args.emotion_threshold, args.dyad_threshold)

        print("\nemotion probabilities:")
        for emotion, prob in sorted(result['emotions']['probabilities'].items(),
                                   key=lambda x: x[1], reverse=True):
            if prob > 0.1:
                bar = '█' * int(prob * 40)
                pred = '✓' if prob >= args.emotion_threshold else ' '
                print(f"  {pred} {emotion:15s} {prob:.3f} {bar}")

        print("\ndominant dyads:")
        for i, dyad in enumerate(result['dominant_dyads'], 1):
            print(f"  {i}. {dyad['name']:20s} [{dyad['type']:10s}] strength: {dyad['strength']:.3f}")
            print(f"     {dyad['emotions'][0]} ({dyad['prob1']:.3f}) + {dyad['emotions'][1]} ({dyad['prob2']:.3f})")

    # batch processing from file
    elif args.input:
        print(f"\nprocessing file: {args.input}")

        # read input file
        if args.input.endswith('.csv'):
            # assume first column is text
            df = pd.read_csv(args.input)
            texts = df.iloc[:, 0].tolist()
        else:
            # read as text file (one text per line)
            with open(args.input, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]

        print(f"found {len(texts)} texts to process")

        # batch predict
        results = batch_predict(model, tokenizer, texts, device,
                              args.batch_size, args.emotion_threshold, args.dyad_threshold)

        # save results
        df_results = save_results_csv(results, args.output)

        print(f"\nsummary statistics:")
        print(f"  avg emotions per text: {df_results['num_emotions'].mean():.2f}")
        print(f"  avg dyads per text: {df_results['num_dyads'].mean():.2f}")
        print(f"  texts with multi-label: {(df_results['num_emotions'] > 1).sum()} ({(df_results['num_emotions'] > 1).sum() / len(df_results) * 100:.1f}%)")

        # most common dyads
        dyad_counts = {}
        for col in ['dyad_1', 'dyad_2', 'dyad_3']:
            for dyad in df_results[col]:
                if dyad:
                    dyad_counts[dyad] = dyad_counts.get(dyad, 0) + 1

        print("\nmost common dyads:")
        for dyad, count in sorted(dyad_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dyad:20s}: {count:4d} ({count/len(df_results)*100:.1f}%)")

    # interactive mode
    else:
        print("\ninteractive mode - enter text to analyze (ctrl+c to exit)")
        while True:
            try:
                text = input("\ntext: ")
                if not text.strip():
                    continue

                result = analyze_text(model, tokenizer, text, device,
                                    args.emotion_threshold, args.dyad_threshold)

                # show emotions
                print("emotions:", end=" ")
                for emotion in result['emotions']['predicted_emotions']:
                    prob = result['emotions']['probabilities'][emotion]
                    print(f"{emotion}({prob:.2f})", end=" ")
                print()

                # show dyads
                if result['dominant_dyads']:
                    print("dyads:", end=" ")
                    for dyad in result['dominant_dyads'][:3]:
                        print(f"{dyad['name']}({dyad['strength']:.2f})", end=" ")
                    print()
                else:
                    print("dyads: none detected")

            except KeyboardInterrupt:
                print("\nexiting...")
                break
