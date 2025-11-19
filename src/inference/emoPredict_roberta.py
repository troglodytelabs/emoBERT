#!/usr/bin/env python3
"""
emoPredict_roberta.py - Emotion prediction using trained RoBERTa model

This script loads a trained RoBERTa-based emotion classifier and predicts
emotions for input text.

Usage:
    python emoPredict_roberta.py --text "I feel so happy today!"
    python emoPredict_roberta.py --interactive
    python emoPredict_roberta.py --file input.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

# Plutchik's 8 basic emotions
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


class PlutchikEmotionClassifier(nn.Module):
    """RoBERTa-based multi-label emotion classifier for Plutchik's 8 emotions."""

    def __init__(self, num_labels=8, dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        return logits, probs


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")

    # Initialize model architecture
    model = PlutchikEmotionClassifier()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load optimal thresholds if available
    thresholds = checkpoint.get("optimal_thresholds", None)
    if thresholds is None:
        # Try loading from test_results.json
        results_path = Path(model_path).parent / "test_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
                thresholds = results.get("optimal_thresholds", None)

    if thresholds is not None:
        thresholds = np.array(thresholds)
        print(f"Loaded optimal thresholds: {thresholds}")
    else:
        thresholds = np.array([0.5] * 8)
        print("Using default threshold: 0.5")

    # Print model info
    if "macro_f1" in checkpoint:
        print(f"Model macro F1: {checkpoint['macro_f1']:.4f}")
    if "epoch" in checkpoint:
        print(f"Model from epoch: {checkpoint['epoch'] + 1}")

    return model, thresholds


def predict_emotions(text, model, tokenizer, thresholds, device, top_k=None):
    """Predict emotions for a single text."""

    # Tokenize
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        logits, probs = model(input_ids, attention_mask)

    probs = probs.cpu().numpy()[0]

    # Apply thresholds
    predictions = (probs >= thresholds).astype(int)

    # Create results
    results = []
    for i, emotion in enumerate(plutchik_emotions):
        results.append(
            {
                "emotion": emotion,
                "probability": float(probs[i]),
                "predicted": bool(predictions[i]),
                "threshold": float(thresholds[i]),
            }
        )

    # Sort by probability
    results.sort(key=lambda x: x["probability"], reverse=True)

    if top_k:
        results = results[:top_k]

    return results


def format_results(results, verbose=False):
    """Format prediction results for display."""

    # Get predicted emotions
    predicted = [r for r in results if r["predicted"]]

    output = []

    if predicted:
        emotions_str = ", ".join([r["emotion"] for r in predicted])
        output.append(f"Predicted emotions: {emotions_str}")
    else:
        # Show top emotion even if below threshold
        top = results[0]
        output.append(f"Top emotion: {top['emotion']} ({top['probability']:.1%})")
        output.append("(below threshold)")

    if verbose:
        output.append("\nAll emotions:")
        for r in results:
            marker = "✓" if r["predicted"] else " "
            output.append(
                f"  {marker} {r['emotion']:15s}: {r['probability']:5.1%} "
                f"(threshold: {r['threshold']:.2f})"
            )

    return "\n".join(output)


def interactive_mode(model, tokenizer, thresholds, device):
    """Run interactive prediction mode."""

    print("\n" + "=" * 60)
    print("INTERACTIVE EMOTION PREDICTION")
    print("=" * 60)
    print("Enter text to analyze emotions.")
    print("Commands: 'quit' to exit, 'verbose' to toggle details")
    print("=" * 60 + "\n")

    verbose = False

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not text:
            continue

        if text.lower() == "quit":
            print("Goodbye!")
            break

        if text.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        # Predict
        results = predict_emotions(text, model, tokenizer, thresholds, device)
        print(format_results(results, verbose=verbose))
        print()


def batch_predict(file_path, model, tokenizer, thresholds, device):
    """Predict emotions for texts in a file (one per line)."""

    print(f"\nProcessing file: {file_path}")

    with open(file_path) as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Found {len(texts)} texts\n")

    for i, text in enumerate(texts, 1):
        print(f"[{i}] {text[:80]}{'...' if len(text) > 80 else ''}")
        results = predict_emotions(text, model, tokenizer, thresholds, device)

        predicted = [r["emotion"] for r in results if r["predicted"]]
        if predicted:
            print(f"    → {', '.join(predicted)}")
        else:
            top = results[0]
            print(f"    → {top['emotion']} ({top['probability']:.1%})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotions using trained RoBERTa model"
    )

    script_dir = Path(__file__).parent
    default_model = str(script_dir / "../../models/best_model.pt")

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument(
        "--file", type=str, help="File with texts to analyze (one per line)"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed probabilities"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}")
        print("\nLooking for model in common locations...")

        # Try common locations
        script_dir = Path(__file__).parent
        common_paths = [
            str(script_dir / "../../models/best_model.pt"),
            "models/best_model.pt",
            "best_model.pt",
        ]

        for path in common_paths:
            if Path(path).exists():
                print(f"  Found: {path}")
                args.model = path
                break
        else:
            print("  No model found!")
            print("\nPlease specify model path with --model")
            sys.exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, thresholds = load_model(args.model, device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Ready!\n")

    # Run appropriate mode
    if args.interactive:
        interactive_mode(model, tokenizer, thresholds, device)

    elif args.file:
        batch_predict(args.file, model, tokenizer, thresholds, device)

    elif args.text:
        results = predict_emotions(args.text, model, tokenizer, thresholds, device)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Text: {args.text}\n")
            print(format_results(results, verbose=args.verbose))

    else:
        # Default to interactive mode
        interactive_mode(model, tokenizer, thresholds, device)


if __name__ == "__main__":
    main()
