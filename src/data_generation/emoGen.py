#!/usr/bin/env python3
"""
emoGen.py - create synthetic training data for rare emotions
uses back-translation, templates, and LLM generation to balance dataset
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Try importing optional dependencies
try:
    from transformers import MarianMTModel, MarianTokenizer
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("Warning: transformers not available, back-translation disabled")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not available, LLM generation disabled")

plutchik_emotions = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']

# High-quality seed examples for each emotion
SEED_EXAMPLES = {
    'joy': [
        "I'm so happy and excited!",
        "This is the best day ever!",
        "I feel wonderful and delighted!",
        "I'm thrilled about this opportunity!",
        "This brings me so much joy!",
    ],
    'sadness': [
        "I'm feeling so sad and hopeless.",
        "This is heartbreaking and devastating.",
        "I'm overwhelmed with grief.",
        "I feel miserable and alone.",
        "This makes me deeply melancholy.",
    ],
    'anger': [
        "I'm furious about this injustice!",
        "This makes me so angry and frustrated!",
        "I'm outraged by this behavior!",
        "I'm mad and fed up with this!",
        "This is infuriating and unacceptable!",
    ],
    'fear': [
        "I'm terrified of what might happen.",
        "I'm scared and anxious about this.",
        "This frightens me deeply.",
        "I'm worried and afraid.",
        "The uncertainty is terrifying.",
    ],
    'trust': [
        "I have complete faith in you.",
        "I trust you completely.",
        "I believe in your abilities.",
        "You're reliable and dependable.",
        "I have confidence in this plan.",
    ],
    'disgust': [
        "That's absolutely revolting and disgusting.",
        "I'm sickened by this behavior.",
        "This is morally repugnant.",
        "That's vile and offensive.",
        "I find this deeply distasteful.",
    ],
    'surprise': [
        "Wow! I can't believe this!",
        "This is completely unexpected!",
        "I'm shocked and amazed!",
        "What a surprising turn of events!",
        "I'm stunned by this revelation!",
    ],
    'anticipation': [
        "I can't wait for tomorrow!",
        "I'm looking forward to this!",
        "I'm eager to get started!",
        "I'm excited about what's coming!",
        "I'm hopeful about the future!",
    ]
}

# Comprehensive templates for each emotion
EMOTION_TEMPLATES = {
    'fear': {
        'templates': [
            "I'm {adjective} about {situation}",
            "{situation} makes me feel {adjective}",
            "I feel {adjective} when I think about {situation}",
            "The thought of {situation} {verb} me",
            "I'm experiencing {noun} about {situation}",
            "I'm {adjective} that {situation} might happen",
            "{situation} fills me with {noun}",
            "I can't stop being {adjective} about {situation}",
        ],
        'adjectives': ['terrified', 'scared', 'frightened', 'afraid', 'anxious', 'worried',
                      'nervous', 'fearful', 'alarmed', 'panicked', 'uneasy', 'apprehensive'],
        'situations': ['what might happen', 'the future', 'losing someone I love',
                      'being alone', 'making mistakes', 'failing', 'the unknown',
                      'change', 'confrontation', 'rejection', 'uncertainty', 'the dark'],
        'verbs': ['terrifies', 'scares', 'frightens', 'worries', 'alarms', 'unnerves'],
        'nouns': ['fear', 'anxiety', 'terror', 'dread', 'panic', 'apprehension'],
    },
    'disgust': {
        'templates': [
            "That's absolutely {adjective}",
            "I'm {adjective} by {situation}",
            "This is {adjective} and {adjective}",
            "I find {situation} {adjective}",
            "{situation} is completely {adjective}",
            "I'm feeling {noun} toward {situation}",
            "The {situation} {verb} me",
        ],
        'adjectives': ['disgusting', 'revolting', 'repulsive', 'gross', 'sickening',
                      'nauseating', 'vile', 'repugnant', 'offensive', 'distasteful',
                      'abhorrent', 'loathsome'],
        'situations': ['this behavior', 'that idea', 'their actions', 'this situation',
                      'what happened', 'that smell', 'those comments', 'this treatment'],
        'verbs': ['disgusts', 'revolts', 'repulses', 'sickens', 'nauseates'],
        'nouns': ['disgust', 'revulsion', 'repulsion', 'contempt', 'disdain'],
    },
    'anticipation': {
        'templates': [
            "I'm {adjective} about {situation}",
            "I can't wait for {situation}",
            "I'm {adjective} to {action}",
            "{situation} is {adjective}",
            "I'm filled with {noun} about {situation}",
            "I'm {adjective} for {situation}",
            "The {noun} of {situation} is exciting",
        ],
        'adjectives': ['excited', 'eager', 'looking forward', 'hopeful', 'expectant',
                      'anticipating', 'ready', 'prepared', 'impatient'],
        'situations': ['tomorrow', 'the future', 'what comes next', 'the weekend',
                      'the results', 'this opportunity', 'new beginnings', 'change'],
        'actions': ['get started', 'see what happens', 'begin', 'move forward',
                   'take the next step', 'discover more'],
        'nouns': ['anticipation', 'expectation', 'hope', 'excitement', 'eagerness'],
    },
    'surprise': {
        'templates': [
            "I'm {adjective} by {situation}!",
            "This is {adjective}!",
            "I can't believe {situation}!",
            "{situation} is completely {adjective}!",
            "What a {adjective} {noun}!",
            "I'm {adjective} that {situation}",
            "The {noun} of {situation} is amazing",
        ],
        'adjectives': ['shocked', 'surprised', 'amazed', 'astonished', 'stunned',
                      'startled', 'astounded', 'flabbergasted', 'unexpected'],
        'situations': ['this news', 'what happened', 'the outcome', 'this turn of events',
                      'the results', 'this revelation', 'what I learned'],
        'nouns': ['surprise', 'shock', 'revelation', 'discovery', 'turn of events'],
    },
    'sadness': {
        'templates': [
            "I'm feeling so {adjective}",
            "This makes me {adjective}",
            "I'm {adjective} about {situation}",
            "{situation} fills me with {noun}",
            "I'm overwhelmed by {noun}",
            "The {noun} is unbearable",
            "I can't stop feeling {adjective} about {situation}",
        ],
        'adjectives': ['sad', 'depressed', 'miserable', 'heartbroken', 'devastated',
                      'melancholy', 'sorrowful', 'mournful', 'dejected', 'despondent'],
        'situations': ['the loss', 'what happened', 'this situation', 'the ending',
                      'being apart', 'the memories', 'letting go'],
        'nouns': ['sadness', 'grief', 'sorrow', 'despair', 'heartbreak', 'melancholy'],
    },
    'anger': {
        'templates': [
            "I'm {adjective} about {situation}",
            "This makes me so {adjective}!",
            "I'm {adjective} and {adjective}!",
            "{situation} is {adjective}!",
            "I'm filled with {noun}",
            "The {situation} {verb} me",
            "I can't stand {situation}",
        ],
        'adjectives': ['angry', 'furious', 'enraged', 'mad', 'frustrated', 'irritated',
                      'outraged', 'infuriated', 'livid', 'irate'],
        'situations': ['this injustice', 'the disrespect', 'this treatment', 'their behavior',
                      'the situation', 'what they said', 'being ignored'],
        'verbs': ['angers', 'infuriates', 'enrages', 'frustrates', 'irritates'],
        'nouns': ['anger', 'rage', 'fury', 'frustration', 'outrage', 'irritation'],
    },
    'joy': {
        'templates': [
            "I'm so {adjective}!",
            "This is {adjective}!",
            "I feel {adjective} and {adjective}!",
            "{situation} brings me {noun}",
            "I'm filled with {noun}",
            "This is the most {adjective} {noun}!",
        ],
        'adjectives': ['happy', 'joyful', 'delighted', 'thrilled', 'ecstatic',
                      'elated', 'cheerful', 'pleased', 'overjoyed', 'wonderful'],
        'situations': ['this moment', 'the news', 'this opportunity', 'today'],
        'nouns': ['joy', 'happiness', 'delight', 'pleasure', 'bliss', 'contentment'],
    },
    'trust': {
        'templates': [
            "I {verb} you {adverb}",
            "You're so {adjective}",
            "I have {noun} in {situation}",
            "I {verb} in {situation}",
            "You've proven to be {adjective}",
        ],
        'verbs': ['trust', 'believe', 'rely on', 'have faith in', 'depend on'],
        'adjectives': ['trustworthy', 'reliable', 'dependable', 'honest', 'faithful'],
        'adverbs': ['completely', 'entirely', 'fully', 'absolutely', 'totally'],
        'nouns': ['trust', 'faith', 'confidence', 'belief'],
        'situations': ['you', 'this', 'your judgment', 'the process', 'your abilities'],
    },
}

class SyntheticDataGenerator:
    def __init__(self):
        self.translation_models = {}

    def generate_templates(self, emotion, n=100):
        """Generate samples using templates"""
        if emotion not in EMOTION_TEMPLATES:
            return []

        config = EMOTION_TEMPLATES[emotion]
        templates = config['templates']
        samples = []

        for _ in range(n):
            template = np.random.choice(templates)

            # Fill in template with random choices
            kwargs = {}
            for key in ['adjectives', 'situations', 'verbs', 'nouns', 'actions', 'adverbs']:
                if key in config:
                    # Remove 's' to get singular key name
                    key_name = key[:-1] if key.endswith('s') else key
                    if f'{{{key_name}}}' in template:
                        kwargs[key_name] = np.random.choice(config[key])

            try:
                text = template.format(**kwargs)
                samples.append(text)
            except KeyError:
                continue

        return samples

    def back_translate(self, text, pivot_lang='fr'):
        """Translate text to another language and back"""
        if not TRANSLATION_AVAILABLE:
            return None

        try:
            # Load models if not cached
            if pivot_lang not in self.translation_models:
                fwd_name = f'Helsinki-NLP/opus-mt-en-{pivot_lang}'
                back_name = f'Helsinki-NLP/opus-mt-{pivot_lang}-en'

                print(f"  Loading translation models for {pivot_lang}...", end='', flush=True)
                self.translation_models[pivot_lang] = {
                    'fwd_tokenizer': MarianTokenizer.from_pretrained(fwd_name),
                    'fwd_model': MarianMTModel.from_pretrained(fwd_name),
                    'back_tokenizer': MarianTokenizer.from_pretrained(back_name),
                    'back_model': MarianMTModel.from_pretrained(back_name),
                }
                print(" done!")

            models = self.translation_models[pivot_lang]

            # Translate forward
            inputs = models['fwd_tokenizer'](text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            translated = models['fwd_model'].generate(**inputs, max_length=128)
            pivot_text = models['fwd_tokenizer'].decode(translated[0], skip_special_tokens=True)

            # Translate back
            inputs = models['back_tokenizer'](pivot_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            back = models['back_model'].generate(**inputs, max_length=128)
            result = models['back_tokenizer'].decode(back[0], skip_special_tokens=True)

            # Only return if result is different and valid
            if result and result != text and len(result.strip()) > 5:
                return result
            return None

        except Exception as e:
            print(f"\n  Back-translation error for {pivot_lang}: {str(e)[:50]}...")
            return None

    def generate_back_translations(self, texts, n_per_text=3):
        """Generate variations using back-translation"""
        if not TRANSLATION_AVAILABLE:
            print("  Back-translation not available (install: pip install transformers sentencepiece)")
            return []

        # Use only most reliable language pairs
        pivot_langs = ['fr', 'de', 'es']  # French, German, Spanish work best
        samples = []

        print(f"  Attempting back-translation through {pivot_langs[:n_per_text]}...")

        for text in tqdm(texts, desc="  Back-translating", disable=len(texts) < 10):
            for lang in pivot_langs[:n_per_text]:
                try:
                    translated = self.back_translate(text, lang)
                    if translated and translated != text:
                        samples.append(translated)
                except Exception as e:
                    continue  # Skip failed translations

        print(f"  Successfully generated {len(samples)} back-translations")
        return samples

    def generate_with_llm(self, emotion, n=100, api_key=None):
        """Generate samples using Claude"""
        if not ANTHROPIC_AVAILABLE or not api_key:
            return []

        client = anthropic.Anthropic(api_key=api_key)

        seed_examples = "\n".join([f"- {ex}" for ex in SEED_EXAMPLES[emotion]])

        prompt = f"""Generate {n} diverse, realistic text samples that express {emotion}.

Requirements:
- Each should be 10-30 words long
- Use varied vocabulary and sentence structures
- Mix direct and indirect expressions of the emotion
- Include different contexts (personal, social, work, relationships, etc.)
- Make them sound natural and conversational
- Avoid repetitive patterns

Seed examples for {emotion}:
{seed_examples}

Generate exactly {n} new samples. Return ONLY a JSON array of strings, no other text.
Example format: ["sample 1", "sample 2", ...]"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            response_text = response.content[0].text
            # Remove markdown code blocks if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            samples = json.loads(response_text)

            return samples[:n]  # Ensure we don't get more than requested
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return []

def augment_emotion_data(emotion, target_count=15000, strategy='hybrid', api_key=None):
    """
    Augment data for a specific emotion to reach target count

    strategy:
        'templates': Only use templates (fastest, good quality)
        'backtrans': Only use back-translation (slow, high quality)
        'llm': Only use LLM (requires API, highest quality)
        'hybrid': Mix of all (recommended)
    """
    generator = SyntheticDataGenerator()

    # Start with seed examples
    current_samples = SEED_EXAMPLES[emotion].copy()
    current_count = len(current_samples)
    needed = target_count - current_count

    print(f"\nGenerating {needed} synthetic samples for {emotion}")
    print(f"Strategy: {strategy}")

    generated = []

    if strategy == 'templates' or strategy == 'hybrid':
        # Generate template-based samples
        n_templates = needed if strategy == 'templates' else int(needed * 0.6)  # Increased from 0.5
        print(f"Generating {n_templates} template-based samples...")
        template_samples = generator.generate_templates(emotion, n_templates)
        generated.extend(template_samples)
        print(f"  ✓ Generated {len(template_samples)} template samples")

    if strategy == 'backtrans' or strategy == 'hybrid':
        # Generate back-translation samples
        if strategy == 'backtrans':
            n_backtrans = needed
        else:
            # For hybrid, use remaining count
            remaining = target_count - current_count - len(generated)
            n_backtrans = max(0, int(remaining * 0.4))  # Use 40% of remaining

        if n_backtrans > 0:
            print(f"Generating up to {n_backtrans} back-translated samples...")
            # Use both seed examples and generated templates as source
            source_texts = current_samples + generated[:min(20, len(generated))]
            n_per_seed = max(1, (n_backtrans // len(source_texts)) + 1)
            bt_samples = generator.generate_back_translations(source_texts, n_per_seed)
            generated.extend(bt_samples[:n_backtrans])
            print(f"  ✓ Generated {len(bt_samples[:n_backtrans])} back-translation samples")

    if strategy == 'llm' or strategy == 'hybrid':
        # Generate LLM samples
        if strategy == 'llm':
            n_llm = needed
        else:
            # For hybrid, use remaining count
            remaining = target_count - current_count - len(generated)
            n_llm = max(0, int(remaining))

        if n_llm > 0 and api_key:
            print(f"Generating {n_llm} LLM-based samples...")
            llm_samples = generator.generate_with_llm(emotion, n_llm, api_key)
            generated.extend(llm_samples)
            print(f"  ✓ Generated {len(llm_samples)} LLM samples")
        elif n_llm > 0 and not api_key:
            print(f"  ⊘ Skipping {n_llm} LLM samples (no API key provided)")
            # Fill remaining with more templates
            print(f"  → Generating {n_llm} additional template samples instead...")
            extra_templates = generator.generate_templates(emotion, n_llm)
            generated.extend(extra_templates)
            print(f"  ✓ Generated {len(extra_templates)} extra template samples")

    # Combine all samples
    all_samples = current_samples + generated

    # Remove exact duplicates
    seen = set()
    unique_samples = []
    for sample in all_samples:
        sample_lower = sample.lower().strip()
        if sample_lower not in seen:
            seen.add(sample_lower)
            unique_samples.append(sample)

    print(f"\nTotal unique samples for {emotion}: {len(unique_samples)}")

    # If we don't have enough, generate more templates to fill
    if len(unique_samples) < target_count:
        shortfall = target_count - len(unique_samples)
        print(f"  → Need {shortfall} more samples, generating additional templates...")
        extra = generator.generate_templates(emotion, shortfall)
        unique_samples.extend(extra)
        print(f"  ✓ Added {len(extra)} more samples")

    # Take exactly target_count samples
    final_samples = unique_samples[:target_count]

    # Create labels (one-hot for this emotion)
    labels = []
    for _ in final_samples:
        label = [0] * 8
        emotion_idx = plutchik_emotions.index(emotion)
        label[emotion_idx] = 1
        labels.append(label)

    print(f"Final count for {emotion}: {len(final_samples)}")

    return final_samples, labels

def save_synthetic_data(texts, labels, output_path):
    """Save synthetic data to CSV"""
    df = pd.DataFrame({
        'text': texts,
        **{f'{emotion}_label': [l[i] for l in labels]
           for i, emotion in enumerate(plutchik_emotions)}
    })

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(texts)} samples to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--emotion', type=str, choices=plutchik_emotions,
                       help='Emotion to generate data for')
    parser.add_argument('--count', type=int, default=5000,
                       help='Number of samples to generate')
    parser.add_argument('--strategy', type=str,
                       choices=['templates', 'backtrans', 'llm', 'hybrid'],
                       default='hybrid', help='Generation strategy')
    script_dir = Path(__file__).parent
    default_output = str(script_dir / "../../data/synthetic/synthetic_data.csv")
    parser.add_argument('--output', type=str, default=default_output,
                       help='Output CSV file')
    parser.add_argument('--api_key', type=str, help='Anthropic API key for LLM generation')
    parser.add_argument('--all_rare', action='store_true',
                       help='Generate for all rare emotions (fear, disgust, surprise, anticipation)')

    args = parser.parse_args()

    if args.all_rare:
        # Generate for all rare emotions
        rare_emotions = ['fear', 'disgust', 'surprise', 'anticipation']
        all_texts = []
        all_labels = []

        for emotion in rare_emotions:
            texts, labels = augment_emotion_data(
                emotion,
                target_count=args.count,
                strategy=args.strategy,
                api_key=args.api_key
            )
            all_texts.extend(texts)
            all_labels.extend(labels)

        save_synthetic_data(all_texts, all_labels, args.output)

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for emotion in rare_emotions:
            count = sum(1 for l in all_labels if l[plutchik_emotions.index(emotion)] == 1)
            print(f"{emotion:15s}: {count:6,} samples")
        print(f"{'TOTAL':15s}: {len(all_texts):6,} samples")

    elif args.emotion:
        # Generate for single emotion
        texts, labels = augment_emotion_data(
            args.emotion,
            target_count=args.count,
            strategy=args.strategy,
            api_key=args.api_key
        )
        save_synthetic_data(texts, labels, args.output)
    else:
        print("Please specify --emotion or use --all_rare")
        print("\nExamples:")
        print("  python emoGen.py --emotion fear --count 5000")
        print("  python emoGen.py --all_rare --count 10000 --strategy templates")
