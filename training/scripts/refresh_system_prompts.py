#!/usr/bin/env python3
"""
Refresh system prompts in existing training JSONL files.

Replaces the system prompt in every example with the current version
from training/prompts/, so existing data stays in sync after prompt updates.

Usage:
    python refresh_system_prompts.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def refresh(jsonl_path: Path, prompt_file: str):
    prompt = (PROMPTS_DIR / prompt_file).read_text().strip()

    with open(jsonl_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    updated = 0
    for ex in examples:
        msgs = ex.get("messages", [])
        if msgs and msgs[0]["role"] == "system":
            if msgs[0]["content"] != prompt:
                msgs[0]["content"] = prompt
                updated += 1

    with open(jsonl_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  {jsonl_path.name}: {updated}/{len(examples)} prompts updated")


def main():
    print("\nRefreshing system prompts in training data...\n")

    for prefix in ["train", "val"]:
        interp = DATA_DIR / f"{prefix}_interpreter.jsonl"
        expl = DATA_DIR / f"{prefix}_explainer.jsonl"

        if interp.exists():
            refresh(interp, "interpreter_system.txt")
        if expl.exists():
            refresh(expl, "explainer_system.txt")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
