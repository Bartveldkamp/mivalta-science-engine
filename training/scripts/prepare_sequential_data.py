#!/usr/bin/env python3
"""
Prepare sequential explainer training data for Qwen2.5 dual-model architecture.

Takes existing interpreter and explainer training data, and creates new
explainer training files where each example includes the interpreter's
GATCRequest output as context. Also filters to only include actions that
need the explainer (explain, answer_question).

Sequential architecture:
  User message → INTERPRETER → GATCRequest JSON → ROUTER (code)
    → explain/answer_question → EXPLAINER (with interpreter context) → coaching text
    → create_workout/replan   → skip explainer, return JSON only
    → clarify                 → skip explainer, use interpreter's clarify_message

Usage:
    python prepare_sequential_data.py

    # Custom paths
    python prepare_sequential_data.py \
        --interpreter_train data/train_interpreter.jsonl \
        --explainer_train data/train_explainer.jsonl \
        --output_train data/train_explainer_sequential.jsonl
"""

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"

# Actions that need the explainer in sequential architecture
ACTIONS_NEEDING_EXPLAINER = {"explain", "answer_question"}


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_user_message(messages: list[dict]) -> str | None:
    """Extract the user message content for matching."""
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return None


def build_interpreter_index(interpreter_data: list[dict]) -> dict[str, str]:
    """Build a lookup from user message → interpreter JSON output."""
    index = {}
    for ex in interpreter_data:
        user_msg = extract_user_message(ex["messages"])
        if user_msg:
            assistant_content = ex["messages"][-1]["content"]
            index[user_msg] = assistant_content
    return index


def process_split(
    interp_path: str,
    expl_path: str,
    out_path: str,
    explainer_system: str,
    split_name: str,
) -> dict:
    """Process one data split (train or val)."""

    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")

    # Load data
    interp_data = load_jsonl(interp_path)
    expl_data = load_jsonl(expl_path)
    print(f"  Interpreter examples: {len(interp_data)}")
    print(f"  Explainer examples:   {len(expl_data)}")

    # Build interpreter lookup
    interp_index = build_interpreter_index(interp_data)
    print(f"  Unique interpreter user messages: {len(interp_index)}")

    # Process explainer examples
    sequential_examples = []
    skipped_no_match = 0
    skipped_action = 0
    skipped_parse = 0
    action_counts = {}

    for ex in expl_data:
        user_msg = extract_user_message(ex["messages"])
        if user_msg is None:
            skipped_no_match += 1
            continue

        # Look up interpreter output for this user message
        interp_output = interp_index.get(user_msg)
        if interp_output is None:
            skipped_no_match += 1
            continue

        # Parse the interpreter JSON to check the action
        try:
            interp_json = json.loads(interp_output)
            action = interp_json.get("action", "")
        except json.JSONDecodeError:
            skipped_parse += 1
            continue

        # Track action distribution
        action_counts[action] = action_counts.get(action, 0) + 1

        # Filter: only keep actions that need the explainer
        if action not in ACTIONS_NEEDING_EXPLAINER:
            skipped_action += 1
            continue

        # Build sequential example: inject interpreter output into user message
        new_user_content = f"{user_msg}\n\n[INTERPRETER]\n{interp_output}"

        new_messages = [
            {"role": "system", "content": explainer_system},
            {"role": "user", "content": new_user_content},
            {"role": "assistant", "content": ex["messages"][-1]["content"]},
        ]

        sequential_examples.append({"messages": new_messages})

    # Report
    print(f"\n  Action distribution in matched explainer data:")
    for action, count in sorted(action_counts.items()):
        marker = " <-- KEPT" if action in ACTIONS_NEEDING_EXPLAINER else " <-- skipped"
        print(f"    {action}: {count}{marker}")

    print(f"\n  Results:")
    print(f"    Sequential examples created: {len(sequential_examples)}")
    print(f"    Skipped (no interpreter match): {skipped_no_match}")
    print(f"    Skipped (action doesn't need explainer): {skipped_action}")
    if skipped_parse:
        print(f"    Skipped (JSON parse error): {skipped_parse}")

    # Show a sample
    if sequential_examples:
        sample = sequential_examples[0]
        user_content = sample["messages"][1]["content"]
        assistant_content = sample["messages"][2]["content"]
        print(f"\n  Sample user message (last 200 chars):")
        print(f"    ...{user_content[-200:]}")
        print(f"  Sample assistant response:")
        print(f"    {assistant_content[:150]}...")

    # Write output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in sequential_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\n  Written to: {out_path}")

    return {
        "total": len(sequential_examples),
        "skipped_no_match": skipped_no_match,
        "skipped_action": skipped_action,
        "action_counts": action_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sequential explainer training data for Qwen2.5 dual-model architecture"
    )
    parser.add_argument(
        "--interpreter_train",
        default=str(DATA_DIR / "train_interpreter.jsonl"),
        help="Path to interpreter training data",
    )
    parser.add_argument(
        "--explainer_train",
        default=str(DATA_DIR / "train_explainer.jsonl"),
        help="Path to explainer training data",
    )
    parser.add_argument(
        "--output_train",
        default=str(DATA_DIR / "train_explainer_sequential.jsonl"),
        help="Output path for sequential explainer training data",
    )
    parser.add_argument(
        "--interpreter_val",
        default=str(DATA_DIR / "val_interpreter.jsonl"),
        help="Path to interpreter validation data",
    )
    parser.add_argument(
        "--explainer_val",
        default=str(DATA_DIR / "val_explainer.jsonl"),
        help="Path to explainer validation data",
    )
    parser.add_argument(
        "--output_val",
        default=str(DATA_DIR / "val_explainer_sequential.jsonl"),
        help="Output path for sequential explainer validation data",
    )

    args = parser.parse_args()

    # Load system prompt
    prompt_path = PROMPTS_DIR / "explainer_sequential_system.txt"
    if not prompt_path.exists():
        prompt_path = PROMPTS_DIR / "explainer_system.txt"
        print(f"Note: Using {prompt_path.name} (explainer_sequential_system.txt not found)")
    explainer_system = prompt_path.read_text().strip()
    print(f"Loaded system prompt from: {prompt_path}")

    # Process train split
    train_stats = process_split(
        args.interpreter_train,
        args.explainer_train,
        args.output_train,
        explainer_system,
        "train",
    )

    # Process val split
    val_stats = process_split(
        args.interpreter_val,
        args.explainer_val,
        args.output_val,
        explainer_system,
        "val",
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {train_stats['total']} sequential examples")
    print(f"  Val:   {val_stats['total']} sequential examples")
    print(f"\nSequential explainer data ready for training.")
    print(f"Next step:")
    print(f"  python finetune_qwen25.py train --mode explainer")


if __name__ == "__main__":
    main()
