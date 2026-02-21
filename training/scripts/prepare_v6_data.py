#!/usr/bin/env python3
"""
MiValta Josi v6 — Unified Data Preparation

Merges interpreter + coach training data into a single unified dataset.
Injects [KNOWLEDGE] blocks into coach examples so the model learns to
ground responses in coaching science shipped with the model.

The unified dataset trains ONE Qwen3 model to handle both modes.
The model learns which mode to use based on the system prompt.

Input:
  - train_interpreter.jsonl      (~1450 examples, GATCRequest JSON)
  - train_explainer_sequential.jsonl  (~812 examples, coaching text)
  - val_interpreter.jsonl        (~149 examples)
  - val_explainer_sequential.jsonl    (~92 examples)
  - knowledge/generated/knowledge.json  (114 coaching knowledge cards)

Output:
  - train_v6_unified.jsonl       (merged + shuffled, coach examples with [KNOWLEDGE])
  - val_v6_unified.jsonl         (merged + shuffled, coach examples with [KNOWLEDGE])

Usage:
    python prepare_v6_data.py --update-prompts --inject-knowledge
    python prepare_v6_data.py --update-prompts   # Without knowledge injection
    python prepare_v6_data.py --seed 42           # Reproducible shuffle
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
KNOWLEDGE_JSON = PROJECT_ROOT / "knowledge" / "generated" / "knowledge.json"

# Add shared/ to path for knowledge selector
sys.path.insert(0, str(PROJECT_ROOT))

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


def extract_user_message_parts(user_content: str) -> dict:
    """Parse a coach user message into its component parts.

    Returns dict with keys: athlete_text, context_block, history_block,
    interpreter_block, and their raw positions.
    """
    parts = {
        "athlete_text": "",
        "context_block": "",
        "history_block": "",
        "interpreter_block": "",
    }

    # Split on known block markers
    # The message format is: athlete_text \n\n CONTEXT: ... \n\n [HISTORY:] ... \n\n [INTERPRETER] ...
    interpreter_match = re.search(r'\[INTERPRETER\]\s*', user_content)
    if interpreter_match:
        parts["interpreter_block"] = user_content[interpreter_match.start():]
        before_interp = user_content[:interpreter_match.start()].rstrip()
    else:
        before_interp = user_content

    # Find HISTORY block (optional)
    history_match = re.search(r'HISTORY:\s*\n', before_interp)
    if history_match:
        # History goes from HISTORY: to end of before_interp section
        parts["history_block"] = before_interp[history_match.start():].rstrip()
        before_history = before_interp[:history_match.start()].rstrip()
    else:
        before_history = before_interp

    # Find CONTEXT block
    context_match = re.search(r'CONTEXT:\s*\n', before_history)
    if context_match:
        parts["context_block"] = before_history[context_match.start():].rstrip()
        parts["athlete_text"] = before_history[:context_match.start()].rstrip()
    else:
        parts["athlete_text"] = before_history.rstrip()

    return parts


def extract_interpreter_json(interpreter_block: str) -> dict | None:
    """Extract the JSON from an [INTERPRETER] block."""
    # Remove the [INTERPRETER] marker
    json_text = re.sub(r'\[INTERPRETER\]\s*', '', interpreter_block).strip()
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def extract_sport_from_context(context_block: str) -> str | None:
    """Extract sport from CONTEXT block."""
    sport_map = {
        "running": "run", "run": "run", "hardlopen": "run",
        "cycling": "bike", "bike": "bike", "fietsen": "bike",
        "skiing": "ski", "ski": "ski",
        "skating": "skate", "skate": "skate",
        "strength": "strength",
    }
    match = re.search(r'Sport:\s*(\w+)', context_block, re.IGNORECASE)
    if match:
        sport_raw = match.group(1).lower()
        return sport_map.get(sport_raw, sport_raw)
    return None


def inject_knowledge_block(example: dict, selector) -> dict:
    """Inject a [KNOWLEDGE] block into a coach example's user message.

    The knowledge block is placed BEFORE the [INTERPRETER] block so the
    model sees relevant coaching context before generating its response.
    """
    messages = example["messages"]
    updated = []

    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            parts = extract_user_message_parts(content)

            # Extract action and sport for selector
            interp_data = extract_interpreter_json(parts["interpreter_block"])
            action = interp_data.get("action") if interp_data else None
            sport = extract_sport_from_context(parts["context_block"])

            # Select relevant knowledge cards
            athlete_text = parts["athlete_text"]
            cards = selector.select(
                user_message=athlete_text,
                action=action,
                sport=sport,
                max_cards=2,  # 2 cards for training to keep context manageable
            )

            if cards:
                knowledge_block = selector.format_knowledge_block(cards)

                # Rebuild the user message with [KNOWLEDGE] before [INTERPRETER]
                rebuilt = parts["athlete_text"]
                if parts["context_block"]:
                    rebuilt += "\n\n" + parts["context_block"]
                if parts["history_block"]:
                    rebuilt += "\n\n" + parts["history_block"]
                rebuilt += "\n\n" + knowledge_block
                if parts["interpreter_block"]:
                    rebuilt += "\n\n" + parts["interpreter_block"]

                updated.append({"role": "user", "content": rebuilt})
            else:
                updated.append(msg)
        else:
            updated.append(msg)

    return {"messages": updated}


def main():
    parser = argparse.ArgumentParser(description="Prepare unified v6 training data")
    parser.add_argument("--update-prompts", action="store_true",
                        help="Replace system prompts with v6 versions")
    parser.add_argument("--inject-knowledge", action="store_true",
                        help="Inject [KNOWLEDGE] blocks into coach examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load knowledge selector if injecting
    selector = None
    if args.inject_knowledge:
        if KNOWLEDGE_JSON.exists():
            from shared.knowledge_selector import KnowledgeSelector
            selector = KnowledgeSelector.from_json(str(KNOWLEDGE_JSON))
            print(f"  Loaded knowledge: {len(selector.entries)} entries from {KNOWLEDGE_JSON.name}")
        else:
            print(f"  WARNING: {KNOWLEDGE_JSON} not found, skipping knowledge injection")
            print(f"  Run: python knowledge/scripts/export_knowledge_json.py")
            args.inject_knowledge = False

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

        # Inject knowledge into coach examples
        knowledge_injected = 0
        if args.inject_knowledge and selector:
            enriched = []
            for ex in coach_data:
                enriched_ex = inject_knowledge_block(ex, selector)
                # Check if knowledge was actually injected
                for msg in enriched_ex["messages"]:
                    if msg["role"] == "user" and "[KNOWLEDGE]" in msg["content"]:
                        knowledge_injected += 1
                        break
                enriched.append(enriched_ex)
            coach_data = enriched
            print(f"  Knowledge injected: {knowledge_injected}/{len(coach_data)} coach examples")

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
