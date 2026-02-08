#!/usr/bin/env python3
"""
Prepare SmolLM2 Training Dataset for Josi

Combines gold training examples and applies quality filters:
- Trims verbose responses to target max word count
- Removes duplicates
- Adds system prompt for coaching personality
- Outputs a single clean JSONL file ready for finetune_smollm2.py

Usage:
    python prepare_training_data.py
    python prepare_training_data.py --max-words 80 --output ./data/smollm2_train.jsonl
    python prepare_training_data.py --stats  # Show stats without writing
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter


# Josi system prompt — must match finetune_smollm2.py
JOSI_SYSTEM_PROMPT = (
    "You are Josi, a friendly and knowledgeable sports coaching assistant for MiValta. "
    "You communicate training decisions made by the coaching engine. "
    "Rules: Keep responses under 80 words. Be warm and conversational. "
    "Use simple language, not textbook explanations. Ask follow-up questions. "
    "Never invent training rules — only explain what the engine decided. "
    "Never mention algorithms, GATC, Viterbi, ACWR, or internal systems."
)

# Forbidden words — responses containing these get flagged
FORBIDDEN_WORDS = [
    "gatc", "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "acute:chronic", "acute chronic", "load ratio",
    "monotony index", "training monotony", "strain index",
    "exponentially weighted", "ewma", "tss", "ctl", "atl", "tsb",
    "impulse-response", "banister", "fitness-fatigue",
]

DATA_DIR = Path(__file__).parent.parent / "data"


def load_jsonl(path: str) -> list:
    """Load a JSONL file."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_messages(example: dict) -> tuple:
    """Extract user and assistant messages from various formats."""
    if "messages" in example:
        user_msg = ""
        assistant_msg = ""
        for msg in example["messages"]:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]
        return user_msg, assistant_msg
    elif "instruction" in example and "response" in example:
        return example["instruction"], example["response"]
    elif "user" in example and "assistant" in example:
        return example["user"], example["assistant"]
    return "", ""


def check_forbidden(text: str) -> list:
    """Check for forbidden words."""
    text_lower = text.lower()
    return [w for w in FORBIDDEN_WORDS if w in text_lower]


def trim_response(text: str, max_words: int) -> str:
    """Trim response to max words, ending at a sentence boundary."""
    words = text.split()
    if len(words) <= max_words:
        return text

    # Try to end at a sentence boundary
    trimmed = " ".join(words[:max_words])

    # Find last sentence-ending punctuation
    for end_char in [".", "!", "?"]:
        last_idx = trimmed.rfind(end_char)
        if last_idx > len(trimmed) * 0.5:  # Don't cut more than half
            return trimmed[:last_idx + 1]

    # No good sentence boundary — just cut and add period
    return trimmed.rstrip(",;:— ") + "."


def prepare_dataset(
    max_words: int = 80,
    min_words: int = 15,
    include_system: bool = True,
    stats_only: bool = False,
) -> list:
    """Prepare clean training dataset from gold examples."""

    # Files to include, in priority order
    source_files = [
        # Well-sized gold files (already concise)
        ("gold_examples/gold_readiness.jsonl", "readiness"),
        ("gold_examples/gold_smart_questions.jsonl", "smart_questions"),
        ("gold_examples/gold_user_intent.jsonl", "user_intent"),
        ("gold_examples/gold_gatc_explanations.jsonl", "gatc_explanations"),
        ("gold_examples/gold_education_basics.jsonl", "education_basics"),
        ("gold_examples/gold_sport_science.jsonl", "sport_science"),
        ("gold_examples/gold_onboarding.jsonl", "onboarding"),

        # Longer gold files (will be trimmed)
        ("gold_examples/gold_recovery.jsonl", "recovery"),
        ("gold_examples/gold_motivation.jsonl", "motivation"),
        ("gold_examples/gold_zones.jsonl", "zones"),
        ("gold_examples/gold_balance.jsonl", "balance"),
        ("gold_examples/gold_illness.jsonl", "illness"),
        ("gold_examples/gold_beginners.jsonl", "beginners"),
        ("gold_examples/gold_masters.jsonl", "masters"),
        ("gold_examples/gold_seniors.jsonl", "seniors"),

        # Philosophy (enhanced version is already concise)
        ("philosophy_enhanced.jsonl", "philosophy"),
    ]

    all_examples = []
    seen_prompts = set()  # Dedup by user prompt
    stats = Counter()

    for filename, source_label in source_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        raw = load_jsonl(str(filepath))
        file_kept = 0
        file_trimmed = 0
        file_skipped = 0
        file_forbidden = 0

        for example in raw:
            user_msg, assistant_msg = extract_messages(example)

            if not user_msg or not assistant_msg:
                file_skipped += 1
                continue

            # Dedup
            prompt_key = user_msg.strip().lower()
            if prompt_key in seen_prompts:
                file_skipped += 1
                stats["duplicates"] += 1
                continue
            seen_prompts.add(prompt_key)

            # Check forbidden words
            forbidden = check_forbidden(assistant_msg)
            if forbidden:
                file_forbidden += 1
                stats["forbidden"] += 1
                continue

            # Check min length
            word_count = len(assistant_msg.split())
            if word_count < min_words:
                file_skipped += 1
                stats["too_short"] += 1
                continue

            # Trim if too long
            original_words = word_count
            if word_count > max_words:
                assistant_msg = trim_response(assistant_msg, max_words)
                file_trimmed += 1
                stats["trimmed"] += 1

            # Build clean example
            messages = []
            if include_system:
                messages.append({"role": "system", "content": JOSI_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

            all_examples.append({
                "messages": messages,
                "source": source_label,
            })
            file_kept += 1

        total = file_kept + file_skipped + file_forbidden
        print(f"  {source_label:<20} {file_kept:>4} kept, {file_trimmed:>3} trimmed, "
              f"{file_skipped:>3} skipped, {file_forbidden:>2} forbidden  (of {total})")

    # Summary
    print(f"\n  Total: {len(all_examples)} training examples")
    print(f"  Duplicates removed: {stats['duplicates']}")
    print(f"  Forbidden word removed: {stats['forbidden']}")
    print(f"  Too short removed: {stats['too_short']}")
    print(f"  Trimmed to {max_words}w: {stats['trimmed']}")

    # Word count distribution of final dataset
    word_counts = [len(ex["messages"][-1]["content"].split()) for ex in all_examples]
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        word_counts_sorted = sorted(word_counts)
        median_words = word_counts_sorted[len(word_counts_sorted) // 2]
        print(f"\n  Final avg response: {avg_words:.0f} words")
        print(f"  Final median response: {median_words} words")
        print(f"  Range: {min(word_counts)}-{max(word_counts)} words")

        # Distribution by source
        source_counts = Counter(ex["source"] for ex in all_examples)
        print(f"\n  By source:")
        for source, count in source_counts.most_common():
            print(f"    {source:<20} {count:>4}")

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clean training data for SmolLM2 Josi fine-tuning"
    )
    parser.add_argument(
        "--max-words", type=int, default=80,
        help="Maximum words per response (longer get trimmed)",
    )
    parser.add_argument(
        "--min-words", type=int, default=15,
        help="Minimum words per response (shorter get dropped)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--no-system", action="store_true",
        help="Don't include system prompt in training examples",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show stats only, don't write output",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Preparing SmolLM2 Training Data for Josi")
    print(f"  Max words: {args.max_words}  |  Min words: {args.min_words}")
    print(f"{'=' * 60}\n")

    examples = prepare_dataset(
        max_words=args.max_words,
        min_words=args.min_words,
        include_system=not args.no_system,
        stats_only=args.stats,
    )

    if args.stats:
        print("\n  Stats only mode — no file written.")
        return

    # Write output
    output_path = args.output or str(DATA_DIR / "smollm2_train.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Written: {output_path} ({size_kb:.0f} KB)")
    print(f"\n  Next: python finetune_smollm2.py train --model 1.7b --train_data {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
