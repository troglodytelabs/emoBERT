#!/usr/bin/env python3
"""
emoGen_claude_v2.py

Generate MORE synthetic data for specific emotions (sadness, anger)
and merge with existing synthetic_claude.csv

Usage:
    python emoGen_claude_v2.py --emotions sadness anger --samples 10000
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import anthropic
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--emotions",
    nargs="+",
    required=True,
    help="Emotions to generate (e.g., sadness anger)",
)
parser.add_argument(
    "--samples", type=int, default=10000, help="Total samples to generate per emotion"
)
parser.add_argument(
    "--existing-csv",
    type=str,
    default="synthetic_claude.csv",
    help="Path to existing synthetic data CSV",
)
parser.add_argument(
    "--output-csv",
    type=str,
    default="synthetic_claude_merged.csv",
    help="Path to save merged CSV",
)
parser.add_argument("--batch-size", type=int, default=10, help="Samples per API call")
parser.add_argument(
    "--max-concurrent", type=int, default=5, help="Max concurrent API requests"
)
args = parser.parse_args()

# Validate emotions
valid_emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
]
for emotion in args.emotions:
    if emotion not in valid_emotions:
        raise ValueError(f"Invalid emotion: {emotion}. Must be one of {valid_emotions}")

print(f"Generating {args.samples} samples for: {', '.join(args.emotions)}")
print(f"Batch size: {args.batch_size}, Max concurrent: {args.max_concurrent}")

# Initialize Anthropic client
client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Plutchik emotion definitions
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


async def generate_samples_for_emotion(emotion: str, num_samples: int) -> list:
    """Generate synthetic samples for a single emotion"""

    samples = []
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size

    emotion_descriptions = {
        "joy": "happiness, contentment, satisfaction, pleasure, delight, cheerfulness",
        "sadness": "sorrow, grief, disappointment, loneliness, melancholy, heartache",
        "anger": "frustration, irritation, rage, resentment, indignation, fury",
        "fear": "anxiety, worry, apprehension, terror, dread, nervousness",
        "trust": "confidence, faith, reliance, security, acceptance, belief",
        "disgust": "revulsion, contempt, aversion, loathing, distaste, repugnance",
        "surprise": "astonishment, amazement, shock, wonder, disbelief, bewilderment",
        "anticipation": "expectation, hope, excitement for future, eagerness, looking forward",
    }

    async def generate_batch(batch_num: int):
        prompt = f"""You are generating training data for an emotion classification AI model.

TASK: Generate {args.batch_size} diverse, realistic text samples that express the emotion: {emotion.upper()}

EMOTION DEFINITION:
{emotion_descriptions[emotion]}

REQUIREMENTS:
1. Each sample should be 10-40 words long
2. Use VARIED vocabulary - avoid repeating the same words
3. Mix different sentence structures (statements, questions, exclamations)
4. Include diverse contexts:
   - Personal situations (relationships, health, achievements)
   - Social situations (interactions, events, community)
   - Work/professional contexts
   - Family and friends
   - Abstract thoughts and reflections
5. Vary the intensity (mild to extreme expressions of {emotion})
6. Make them sound natural and conversational
7. Include both direct expressions ("I feel {emotion}") and indirect ("This situation makes me...")
8. Avoid clichés and overly dramatic language
9. Each sample should be UNIQUE - no repetitive patterns

IMPORTANT GUIDELINES:
- Make each sample distinctly different from others
- Use natural, everyday language
- Include specific details and scenarios
- Vary the pronoun usage (I, we, you, they)
- Include different tenses (present, past, future)
- Mix statement types (declarative, interrogative, exclamatory)

Generate exactly {args.batch_size} samples.
Return ONLY a JSON array of strings. No explanations, no markdown, just the array.
Format: ["sample 1", "sample 2", ...]"""

        try:
            message = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=1.0,  # Higher temperature for diversity
                messages=[{"role": "user", "content": prompt}],
            )

            text = message.content[0].text.strip()

            # Remove markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                text = text.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            parsed_samples = json.loads(text)

            # Validate samples (10-40 words)
            valid_samples = [
                s
                for s in parsed_samples
                if isinstance(s, str) and 5 < len(s.split()) < 50
            ]

            # Create label vector for this emotion
            label_vector = [1 if e == emotion else 0 for e in plutchik_emotions]

            batch_samples = []
            for sample_text in valid_samples[: args.batch_size]:
                sample = {"text": sample_text}
                for i, e in enumerate(plutchik_emotions):
                    sample[f"{e}_label"] = label_vector[i]
                batch_samples.append(sample)

            return batch_samples

        except json.JSONDecodeError as e:
            print(f"\nJSON error in batch {batch_num}: {e}")
            return []
        except Exception as e:
            print(f"\nError in batch {batch_num}: {e}")
            return []

    # Generate batches with concurrency limit
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def limited_generate(batch_num):
        async with semaphore:
            return await generate_batch(batch_num)

    tasks = [limited_generate(i) for i in range(num_batches)]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Generating {emotion}")

    for batch_samples in results:
        samples.extend(batch_samples)

    return samples[:num_samples]


async def main():
    print("\n" + "=" * 80)
    print("GENERATING ADDITIONAL SYNTHETIC DATA")
    print("=" * 80)

    # Generate new samples for specified emotions
    all_new_samples = []

    for emotion in args.emotions:
        print(f"\nGenerating {args.samples} samples for {emotion}...")
        samples = await generate_samples_for_emotion(emotion, args.samples)
        all_new_samples.extend(samples)
        print(f"✓ Generated {len(samples)} samples for {emotion}")

    # Convert to DataFrame
    new_df = pd.DataFrame(all_new_samples)
    print(f"\n✓ Generated {len(new_df):,} total new samples")

    # Load existing CSV
    print(f"\nLoading existing data from {args.existing_csv}...")
    existing_df = pd.read_csv(args.existing_csv)
    print(f"✓ Loaded {len(existing_df):,} existing samples")

    # Show existing distribution
    print("\nExisting data distribution:")
    for emotion in plutchik_emotions:
        count = existing_df[f"{emotion}_label"].sum()
        print(f"  {emotion:15s}: {int(count):6,} samples")

    # Merge DataFrames
    print("\nMerging datasets...")
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"✓ Merged dataset has {len(merged_df):,} total samples")

    # Show new distribution
    print("\nMerged data distribution:")
    for emotion in plutchik_emotions:
        count = merged_df[f"{emotion}_label"].sum()
        print(f"  {emotion:15s}: {int(count):6,} samples")

    # Save merged CSV
    print(f"\nSaving to {args.output_csv}...")
    merged_df.to_csv(args.output_csv, index=False)
    print(f"✓ Saved {len(merged_df):,} samples to {args.output_csv}")

    # Calculate cost
    total_samples = len(new_df)
    estimated_tokens = total_samples * 150  # rough estimate
    cost = (estimated_tokens / 1_000_000) * 3.0  # $3 per million tokens

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"New samples generated: {len(new_df):,}")
    print(f"Total samples: {len(merged_df):,}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"\nMerged file: {args.output_csv}")
    print("\nTo use in training, replace synthetic_claude.csv with the merged file:")
    print(f"  mv {args.output_csv} synthetic_claude.csv")


if __name__ == "__main__":
    asyncio.run(main())
