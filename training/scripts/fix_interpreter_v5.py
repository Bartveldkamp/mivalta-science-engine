#!/usr/bin/env python3
"""
Fix interpreter training data — v5 quality cleanup.

Addresses two issues found during review:

  1. System prompt drift: Training examples contain an older version of
     the system prompt that lacks TIME EXTRACTION, MEMORY, and
     ILLNESS vs MEDICAL sections. Updates all examples to use the
     current prompt from training/prompts/interpreter_system.txt.

  2. Zone-change mislabels: 15 examples where "change my zone" or
     "change the zones" are labeled create_workout but should be
     replan (the user wants to modify an existing session's intensity).

Usage:
    python fix_interpreter_v5.py
    # Reads:  training/data/train_interpreter.jsonl
    # Writes: training/data/train_interpreter.jsonl (in-place backup at .bak)
    # Also fixes: training/data/val_interpreter.jsonl
"""

import json
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"

TRAIN_FILE = DATA_DIR / "train_interpreter.jsonl"
VAL_FILE = DATA_DIR / "val_interpreter.jsonl"
PROMPT_FILE = PROMPTS_DIR / "interpreter_system.txt"


def load_current_prompt() -> str:
    """Load the current system prompt from the prompts directory."""
    return PROMPT_FILE.read_text().strip()


def is_zone_change_message(user_msg: str) -> bool:
    """Detect messages that ask to change/switch zones (should be replan)."""
    # Extract just the user part (before CONTEXT)
    msg = user_msg.split("\nCONTEXT:")[0].strip().lower()
    return bool(re.search(r'\b(?:change|switch|modify|adjust)\b.{0,15}\bzone', msg))


def fix_zone_change_label(response: dict, user_msg: str) -> dict:
    """Relabel zone-change create_workout → replan with reduce_intensity."""
    if response.get("action") != "create_workout":
        return response

    if not is_zone_change_message(user_msg):
        return response

    # Extract sport from the original response (keep it)
    sport = response.get("sport")

    fixed = {
        "action": "replan",
        "replan_type": "reduce_intensity",
        "free_text": response.get("free_text", ""),
    }
    if sport:
        fixed["sport"] = sport

    return fixed


def process_file(filepath: Path, current_prompt: str) -> dict:
    """Process a single JSONL file."""
    if not filepath.exists():
        print(f"  Skipping {filepath.name} (not found)")
        return {"total": 0, "prompts_updated": 0, "labels_fixed": 0}

    examples = []
    with open(filepath) as f:
        for line in f:
            examples.append(json.loads(line))

    stats = {"total": len(examples), "prompts_updated": 0, "labels_fixed": 0}

    for ex in examples:
        msgs = ex["messages"]

        # Fix 1: Update system prompt
        if msgs[0]["role"] == "system":
            old_prompt = msgs[0]["content"]
            if old_prompt != current_prompt:
                msgs[0]["content"] = current_prompt
                stats["prompts_updated"] += 1

        # Fix 2: Relabel zone-change examples
        user_msg = msgs[1]["content"]
        old_response = json.loads(msgs[-1]["content"])
        new_response = fix_zone_change_label(old_response, user_msg)

        if new_response != old_response:
            msgs[-1]["content"] = json.dumps(new_response, ensure_ascii=False)
            stats["labels_fixed"] += 1

    # Backup and write
    backup = filepath.with_suffix(".jsonl.bak")
    shutil.copy2(filepath, backup)
    print(f"  Backup: {backup.name}")

    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return stats


def verify_fixes(filepath: Path, current_prompt: str):
    """Verify all prompts are updated and no zone-change mislabels remain."""
    old_prompts = 0
    zone_mislabels = 0

    with open(filepath) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            msgs = ex["messages"]

            if msgs[0]["content"] != current_prompt:
                old_prompts += 1

            user_msg = msgs[1]["content"]
            resp = json.loads(msgs[-1]["content"])
            if resp.get("action") == "create_workout" and is_zone_change_message(user_msg):
                zone_mislabels += 1

    return old_prompts, zone_mislabels


def main():
    print("=" * 60)
    print("  Interpreter Training Data Fix — v5")
    print("=" * 60)

    current_prompt = load_current_prompt()
    print(f"\nCurrent prompt length: {len(current_prompt)} chars")

    for filepath in [TRAIN_FILE, VAL_FILE]:
        print(f"\nProcessing {filepath.name}...")
        stats = process_file(filepath, current_prompt)
        print(f"  Total examples:    {stats['total']}")
        print(f"  Prompts updated:   {stats['prompts_updated']}")
        print(f"  Labels fixed:      {stats['labels_fixed']}")

        # Verify
        old_p, zone_m = verify_fixes(filepath, current_prompt)
        print(f"  Remaining old prompts:     {old_p}")
        print(f"  Remaining zone mislabels:  {zone_m}")

    print(f"\n{'=' * 60}")
    print("  Done. Backups saved as .jsonl.bak")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
