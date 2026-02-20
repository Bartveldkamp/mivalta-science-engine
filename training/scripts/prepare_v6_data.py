#!/usr/bin/env python3
"""
MiValta Josi v6 — Unified Data Preparation

Merges interpreter + coach training data into a single unified dataset.
Optionally replaces system prompts with v6 versions.

The unified dataset trains ONE Qwen3-4B model to handle both modes.
The model learns which mode to use based on the system prompt.

Input:
  - train_interpreter.jsonl      (~1450 examples, GATCRequest JSON)
  - train_explainer_sequential.jsonl  (~812 examples, coaching text)
  - val_interpreter.jsonl        (~149 examples)
  - val_explainer_sequential.jsonl    (~92 examples)

Output:
  - train_v6_unified.jsonl       (merged + shuffled)
  - val_v6_unified.jsonl         (merged + shuffled)

Usage:
    python prepare_v6_data.py
    python prepare_v6_data.py --update-prompts   # Replace system prompts with v6 versions
    python prepare_v6_data.py --seed 42          # Reproducible shuffle
"""

import argparse
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"

# Input files
INTERPRETER_TRAIN = DATA_DIR / "train_interpreter.jsonl"
INTERPRETER_VAL = DATA_DIR / "val_interpreter.jsonl"
COACH_TRAIN = DATA_DIR / "train_explainer_sequential.jsonl"
COACH_VAL = DATA_DIR / "val_explainer_sequential.jsonl"

# Output files
UNIFIED_TRAIN = DATA_DIR / "train_v6_unified.jsonl"
UNIFIED_VAL = DATA_DIR / "val_v6_unified.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(data: list[dict], path: Path):
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_v6_prompt(mode: str) -> str | None:
    """Load v6 system prompt if available."""
    if mode == "interpreter":
        path = PROMPTS_DIR / "josi_v6_interpreter.txt"
    else:
        path = PROMPTS_DIR / "josi_v6_coach.txt"

    if path.exists():
        return path.read_text().strip()
    return None


def detect_mode(example: dict) -> str:
    """Detect whether an example is interpreter or coach mode."""
    messages = example.get("messages", [])
    for msg in messages:
        if msg["role"] == "system":
            content = msg["content"]
            if "Interpreter" in content and "GATCRequest" in content:
                return "interpreter"
            if "coaching assistant" in content.lower():
                return "coach"
    # Check assistant response: if it starts with {, likely interpreter
    for msg in messages:
        if msg["role"] == "assistant":
            if msg["content"].strip().startswith("{"):
                return "interpreter"
            return "coach"
    return "unknown"


def update_system_prompt(example: dict, new_prompt: str) -> dict:
    """Replace the system prompt in an example."""
    messages = example["messages"]
    updated = []
    for msg in messages:
        if msg["role"] == "system":
            updated.append({"role": "system", "content": new_prompt})
        else:
            updated.append(msg)
    return {"messages": updated}


def main():
    parser = argparse.ArgumentParser(description="Prepare unified v6 training data")
    parser.add_argument("--update-prompts", action="store_true",
                        help="Replace system prompts with v6 versions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load v6 prompts if updating
    interpreter_prompt = None
    coach_prompt = None
    if args.update_prompts:
        interpreter_prompt = load_v6_prompt("interpreter")
        coach_prompt = load_v6_prompt("coach")
        if interpreter_prompt:
            print(f"  Loaded v6 interpreter prompt ({len(interpreter_prompt)} chars)")
        else:
            print(f"  WARNING: v6 interpreter prompt not found, keeping original")
        if coach_prompt:
            print(f"  Loaded v6 coach prompt ({len(coach_prompt)} chars)")
        else:
            print(f"  WARNING: v6 coach prompt not found, keeping original")

    # Process train and val sets
    for split_name, interp_path, coach_path, output_path in [
        ("train", INTERPRETER_TRAIN, COACH_TRAIN, UNIFIED_TRAIN),
        ("val", INTERPRETER_VAL, COACH_VAL, UNIFIED_VAL),
    ]:
        print(f"\n{'='*50}")
        print(f"  Processing {split_name} split")
        print(f"{'='*50}")

        # Load interpreter data
        if interp_path.exists():
            interp_data = load_jsonl(interp_path)
            print(f"  Interpreter: {len(interp_data)} examples from {interp_path.name}")
        else:
            interp_data = []
            print(f"  WARNING: {interp_path.name} not found")

        # Load coach data
        if coach_path.exists():
            coach_data = load_jsonl(coach_path)
            print(f"  Coach: {len(coach_data)} examples from {coach_path.name}")
        else:
            coach_data = []
            print(f"  WARNING: {coach_path.name} not found")

        # Optionally update system prompts
        if args.update_prompts:
            if interpreter_prompt:
                interp_data = [update_system_prompt(ex, interpreter_prompt) for ex in interp_data]
                print(f"  Updated interpreter prompts to v6")
            if coach_prompt:
                coach_data = [update_system_prompt(ex, coach_prompt) for ex in coach_data]
                print(f"  Updated coach prompts to v6")

        # Merge and shuffle
        unified = interp_data + coach_data
        random.shuffle(unified)

        # Verify mode distribution
        modes = {}
        for ex in unified:
            mode = detect_mode(ex)
            modes[mode] = modes.get(mode, 0) + 1

        print(f"  Unified: {len(unified)} total examples")
        for mode, count in sorted(modes.items()):
            pct = count / len(unified) * 100
            print(f"    {mode}: {count} ({pct:.0f}%)")

        # Save
        save_jsonl(unified, output_path)
        print(f"  Saved to: {output_path}")

    print(f"\n{'='*50}")
    print(f"  Done! Unified data ready for training.")
    print(f"{'='*50}")
    print(f"\n  Next step:")
    print(f"    python finetune_qwen3.py train --mode unified")


if __name__ == "__main__":
    main()
