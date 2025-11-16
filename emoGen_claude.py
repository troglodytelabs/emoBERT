#!/usr/bin/env python3
"""
emoGen_claude.py - high-quality synthetic data generation using Claude
generates diverse, natural-sounding emotion samples via Anthropic API
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed!")
    print("Install with: pip install anthropic")
    exit(1)

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

# High-quality seed examples for Claude to learn from
SEED_EXAMPLES = {
    "joy": [
        "I'm so happy and excited!",
        "This is the best day ever!",
        "I feel wonderful and delighted!",
        "I'm thrilled about this opportunity!",
        "This brings me so much joy!",
        "My heart is full of happiness right now!",
        "I can't stop smiling about this!",
        "Everything feels perfect today!",
    ],
    "sadness": [
        "I'm feeling so sad and hopeless.",
        "This is heartbreaking and devastating.",
        "I'm overwhelmed with grief.",
        "I feel miserable and alone.",
        "This makes me deeply melancholy.",
        "My heart aches with sorrow.",
        "I can't stop crying about this.",
        "The pain is unbearable right now.",
    ],
    "anger": [
        "I'm furious about this injustice!",
        "This makes me so angry and frustrated!",
        "I'm outraged by this behavior!",
        "I'm mad and fed up with this!",
        "This is infuriating and unacceptable!",
        "I can't believe they would do this!",
        "This disrespect makes my blood boil!",
        "I'm seething with rage right now!",
    ],
    "fear": [
        "I'm terrified of what might happen.",
        "I'm scared and anxious about this.",
        "This frightens me deeply.",
        "I'm worried and afraid.",
        "The uncertainty is terrifying.",
        "I'm panicking about the future.",
        "This makes me incredibly nervous.",
        "I'm dreading what comes next.",
    ],
    "trust": [
        "I have complete faith in you.",
        "I trust you completely.",
        "I believe in your abilities.",
        "You're reliable and dependable.",
        "I have confidence in this plan.",
        "I know I can count on you.",
        "You've never let me down.",
        "I feel safe and secure with you.",
    ],
    "disgust": [
        "That's absolutely revolting and disgusting.",
        "I'm sickened by this behavior.",
        "This is morally repugnant.",
        "That's vile and offensive.",
        "I find this deeply distasteful.",
        "This makes me nauseous.",
        "I'm appalled by what I'm seeing.",
        "This is abhorrent and loathsome.",
    ],
    "surprise": [
        "Wow! I can't believe this!",
        "This is completely unexpected!",
        "I'm shocked and amazed!",
        "What a surprising turn of events!",
        "I'm stunned by this revelation!",
        "I never saw this coming!",
        "This caught me completely off guard!",
        "I'm flabbergasted by this news!",
    ],
    "anticipation": [
        "I can't wait for tomorrow!",
        "I'm looking forward to this!",
        "I'm eager to get started!",
        "I'm excited about what's coming!",
        "I'm hopeful about the future!",
        "I'm preparing for something great!",
        "The anticipation is killing me!",
        "I'm ready for what comes next!",
    ],
}

# Detailed emotion descriptions for Claude
EMOTION_DESCRIPTIONS = {
    "joy": """Joy is a feeling of great pleasure, happiness, and delight. It includes:
- Happiness, cheerfulness, contentment
- Excitement, elation, euphoria
- Gratitude, appreciation, thankfulness
- Amusement, playfulness, fun
- Pride in achievements
- Love and affection combined with happiness""",
    "sadness": """Sadness is an emotional pain associated with feelings of loss, disappointment, or grief. It includes:
- Unhappiness, sorrow, melancholy
- Loneliness, isolation
- Disappointment, dejection
- Grief, heartbreak, despair
- Nostalgia, wistfulness
- Feeling down, blue, or depressed""",
    "anger": """Anger is a strong feeling of displeasure and hostility. It includes:
- Frustration, irritation, annoyance
- Rage, fury, outrage
- Resentment, bitterness
- Indignation at injustice
- Exasperation, aggravation
- Feeling mad, furious, or livid""",
    "fear": """Fear is an unpleasant emotion caused by the threat of danger, pain, or harm. It includes:
- Anxiety, worry, nervousness
- Terror, dread, panic
- Apprehension, unease
- Phobias and specific fears
- Concern, distress
- Feeling scared, frightened, or terrified""",
    "trust": """Trust is a firm belief in the reliability, truth, or ability of someone or something. It includes:
- Confidence, faith, assurance
- Reliability, dependability
- Security, safety
- Belief in others
- Loyalty, faithfulness
- Feeling secure and confident""",
    "disgust": """Disgust is a strong feeling of revulsion or profound disapproval. It includes:
- Revulsion, repulsion, nausea
- Contempt, disdain, scorn
- Moral outrage at behavior
- Distaste, aversion
- Feeling sickened or appalled
- Finding something repugnant or loathsome""",
    "surprise": """Surprise is a feeling of mild astonishment or shock caused by something unexpected. It includes:
- Astonishment, amazement, wonder
- Shock, disbelief
- Being caught off guard
- Revelation, discovery
- Unexpectedness
- Feeling stunned, shocked, or amazed""",
    "anticipation": """Anticipation is excitement and anxiety about something that is going to happen. It includes:
- Expectation, hope, eagerness
- Looking forward to something
- Preparation, readiness
- Excitement about the future
- Planning, forethought
- Feeling ready or expectant""",
}


def generate_samples_with_claude(emotion, count, api_key, batch_size=100):
    """
    Generate high-quality synthetic samples using Claude

    Args:
        emotion: Which Plutchik emotion to generate
        count: How many samples to generate
        api_key: Anthropic API key
        batch_size: Samples per API call (max 100 for quality)

    Returns:
        List of generated text samples
    """
    client = anthropic.Anthropic(api_key=api_key)

    seed_examples = "\n".join([f"- {ex}" for ex in SEED_EXAMPLES[emotion]])
    description = EMOTION_DESCRIPTIONS[emotion]

    all_samples = []
    batches_needed = (count + batch_size - 1) // batch_size

    print(f"\nGenerating {count} samples for {emotion.upper()}")
    print(f"Using {batches_needed} API calls ({batch_size} samples each)")

    for batch_num in tqdm(range(batches_needed), desc=f"  Generating {emotion}"):
        # Calculate how many samples in this batch
        samples_in_batch = min(batch_size, count - len(all_samples))

        prompt = f"""You are generating training data for an emotion classification AI model.

TASK: Generate {samples_in_batch} diverse, realistic text samples that express the emotion: {emotion.upper()}

EMOTION DEFINITION:
{description}

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

SEED EXAMPLES (for reference, but create NEW diverse samples):
{seed_examples}

IMPORTANT GUIDELINES:
- Make each sample distinctly different from others
- Use natural, everyday language
- Include specific details and scenarios
- Vary the pronoun usage (I, we, you, they)
- Include different tenses (present, past, future)
- Mix statement types (declarative, interrogative, exclamatory)

Generate exactly {samples_in_batch} samples.
Return ONLY a JSON array of strings. No explanations, no markdown, just the array.
Format: ["sample 1", "sample 2", ...]"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=1.0,  # Higher temperature for more diversity
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = (
                    "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                )
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )

            samples = json.loads(response_text)

            # Validate samples
            valid_samples = [
                s for s in samples if isinstance(s, str) and 5 < len(s.split()) < 50
            ]

            if len(valid_samples) < samples_in_batch * 0.8:
                print(
                    f"\n  Warning: Only got {len(valid_samples)}/{samples_in_batch} valid samples in batch {batch_num + 1}"
                )

            all_samples.extend(valid_samples)

        except json.JSONDecodeError as e:
            print(f"\n  Error parsing JSON in batch {batch_num + 1}: {e}")
            print(f"  Response was: {response_text[:200]}...")
            continue
        except Exception as e:
            print(f"\n  Error in batch {batch_num + 1}: {e}")
            continue

    # Remove duplicates
    unique_samples = list(set(all_samples))

    print(f"  ✓ Generated {len(all_samples)} samples ({len(unique_samples)} unique)")

    # If we don't have enough, warn user
    if len(unique_samples) < count * 0.9:
        print(f"  ⚠️  Warning: Only got {len(unique_samples)}/{count} requested samples")

    return unique_samples[:count]


def save_to_csv(all_samples_dict, output_path):
    """Save all generated samples to CSV"""

    # Flatten all samples
    texts = []
    labels = []

    for emotion, samples in all_samples_dict.items():
        emotion_idx = plutchik_emotions.index(emotion)

        for sample in samples:
            texts.append(sample)

            # Create one-hot label
            label = [0] * 8
            label[emotion_idx] = 1
            labels.append(label)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "text": texts,
            **{
                f"{emotion}_label": [l[i] for l in labels]
                for i, emotion in enumerate(plutchik_emotions)
            },
        }
    )

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(texts)} total samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality synthetic emotion data using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all rare emotions (40,000 total samples)
  python emoGen_claude.py --all_rare --count 10000 --api_key sk-ant-...

  # Generate for specific emotion
  python emoGen_claude.py --emotion fear --count 15000 --api_key sk-ant-...

  # Use environment variable for API key
  export ANTHROPIC_API_KEY=sk-ant-...
  python emoGen_claude.py --all_rare --count 10000

  # Generate for all 8 emotions
  python emoGen_claude.py --all_emotions --count 15000 --api_key sk-ant-...
        """,
    )

    parser.add_argument(
        "--emotion",
        type=str,
        choices=plutchik_emotions,
        help="Single emotion to generate data for",
    )
    parser.add_argument(
        "--all_rare",
        action="store_true",
        help="Generate for rare emotions: fear, disgust, surprise, anticipation",
    )
    parser.add_argument(
        "--all_emotions",
        action="store_true",
        help="Generate for ALL 8 Plutchik emotions",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of samples per emotion (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_claude.csv",
        help="Output CSV file (default: synthetic_claude.csv)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Samples per API call (default: 100, max: 200)",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("❌ ERROR: No API key provided!")
        print("\nProvide API key via:")
        print("  1. --api_key argument: python emoGen_claude.py --api_key sk-ant-...")
        print("  2. Environment variable: export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nGet your API key at: https://console.anthropic.com/")
        exit(1)

    # Validate API key format
    if not api_key.startswith("sk-ant-"):
        print("❌ ERROR: Invalid API key format!")
        print("API key should start with 'sk-ant-'")
        exit(1)

    # Determine which emotions to generate
    if args.all_emotions:
        emotions_to_generate = plutchik_emotions
    elif args.all_rare:
        emotions_to_generate = ["fear", "disgust", "surprise", "anticipation"]
    elif args.emotion:
        emotions_to_generate = [args.emotion]
    else:
        print("❌ ERROR: Must specify --emotion, --all_rare, or --all_emotions")
        parser.print_help()
        exit(1)

    # Estimate cost
    total_samples = len(emotions_to_generate) * args.count
    batches = len(emotions_to_generate) * (
        (args.count + args.batch_size - 1) // args.batch_size
    )
    estimated_cost = batches * 0.015  # ~$0.015 per API call

    print("=" * 80)
    print("CLAUDE-POWERED SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"Emotions: {', '.join(emotions_to_generate)}")
    print(f"Samples per emotion: {args.count:,}")
    print(f"Total samples: {total_samples:,}")
    print(f"API calls needed: ~{batches}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print("=" * 80)

    # Confirm
    response = input("\nProceed with generation? (y/n): ")
    if response.lower() != "y":
        print("Cancelled.")
        exit(0)

    # Generate samples
    all_samples = {}

    for emotion in emotions_to_generate:
        samples = generate_samples_with_claude(
            emotion=emotion,
            count=args.count,
            api_key=api_key,
            batch_size=args.batch_size,
        )
        all_samples[emotion] = samples

    # Save to CSV
    save_to_csv(all_samples, args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for emotion in emotions_to_generate:
        print(f"{emotion:15s}: {len(all_samples[emotion]):6,} samples")
    print(f"{'TOTAL':15s}: {sum(len(s) for s in all_samples.values()):6,} samples")
    print("=" * 80)

    # Show sample outputs
    print("\nSAMPLE OUTPUTS (first 3 from each emotion):")
    print("=" * 80)
    for emotion in emotions_to_generate:
        print(f"\n{emotion.upper()}:")
        for i, sample in enumerate(all_samples[emotion][:3], 1):
            print(f"  {i}. {sample}")

    print(f"\n✅ SUCCESS! Data saved to: {args.output}")
    print(
        f"Use this file with: python emoBERT_cloud.py --use_synthetic --synthetic_data_path {args.output}"
    )


if __name__ == "__main__":
    main()
