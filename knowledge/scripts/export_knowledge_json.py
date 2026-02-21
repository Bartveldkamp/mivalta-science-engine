#!/usr/bin/env python3
"""
Export coaching knowledge from context.py to knowledge.json.

The resulting JSON file ships alongside the GGUF model so the app
can inject relevant knowledge cards into prompts at inference time.

Usage:
    python export_knowledge_json.py
    python export_knowledge_json.py --output ../generated/knowledge.json
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = SCRIPT_DIR.parent
GENERATED_DIR = KNOWLEDGE_DIR / "generated"

# Add generated dir so we can import context.py
sys.path.insert(0, str(GENERATED_DIR))

# Topic metadata for knowledge selector: card_prefix -> {sport, topics, keywords}
CARD_METADATA = {
    "cycling_v4": {
        "sport": "bike",
        "topics": ["cycling", "power", "ftp", "indoor"],
        "keywords": ["bike", "cycling", "power", "ftp", "watts", "watt", "indoor",
                      "trainer", "turbo", "pedal", "cadence", "ride", "fiets"],
    },
    "running_v4": {
        "sport": "run",
        "topics": ["running", "pace", "long_run", "strides"],
        "keywords": ["run", "running", "pace", "marathon", "half", "10k", "5k",
                      "strides", "long run", "hardlopen", "tempo"],
    },
    "recovery_v4": {
        "topics": ["recovery", "rest", "sleep", "illness", "injury"],
        "keywords": ["recovery", "rest", "sleep", "tired", "fatigue", "sick",
                      "illness", "injury", "pain", "sore", "overtraining",
                      "herstel", "moe", "rust"],
    },
    "energy_zones_v4": {
        "topics": ["zones", "intensity", "heart_rate", "effort"],
        "keywords": ["zone", "zones", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
                      "z8", "intensity", "effort", "heart rate", "easy", "hard",
                      "threshold", "tempo", "interval", "warmup", "cooldown"],
    },
    "zone_anchoring_v4": {
        "topics": ["zones", "threshold", "testing", "heart_rate", "power"],
        "keywords": ["zone", "anchor", "threshold", "test", "ftp", "ltlt",
                      "heart rate", "power", "talk test", "hrmax"],
    },
    "ntiz_dose_v4": {
        "topics": ["training_dose", "volume", "time_in_zone"],
        "keywords": ["dose", "volume", "ntiz", "time in zone", "how much",
                      "how long", "minutes", "hours", "beginner"],
    },
    "ntiz_master_v4": {
        "topics": ["training_dose", "volume", "distribution"],
        "keywords": ["dose", "distribution", "volume", "percentage", "how much",
                      "ntiz", "deload"],
    },
    "session_dose_v4": {
        "topics": ["session", "workout_structure", "recovery"],
        "keywords": ["session", "workout", "clean", "wallet", "shadow",
                      "recovery", "meso"],
    },
    "session_structure_variants_v4": {
        "topics": ["workout_structure", "intervals", "session"],
        "keywords": ["interval", "structure", "workout", "session", "30/30",
                      "tabata", "threshold", "vo2", "tempo", "sprint",
                      "warmup", "cooldown", "reps", "sets", "rest"],
    },
    "session_templates_v4": {
        "topics": ["warmup", "session"],
        "keywords": ["warmup", "warm up", "scaling", "short", "time"],
    },
    "periodization": {
        "topics": ["periodization", "planning", "phases", "taper", "race"],
        "keywords": ["phase", "base", "build", "peak", "taper", "race",
                      "plan", "meso", "mesocycle", "block", "season"],
    },
    "performance_v4": {
        "topics": ["performance", "goals", "improvement", "progress"],
        "keywords": ["improve", "progress", "goal", "target", "faster",
                      "better", "performance", "rate", "realistic", "age",
                      "frequency", "how often", "taper"],
    },
    "threshold_estimation_v4": {
        "topics": ["threshold", "testing", "zones"],
        "keywords": ["threshold", "test", "estimate", "ftp", "pace",
                      "protocol", "20 minute", "time trial"],
    },
    "goal_modifiers_v4": {
        "topics": ["goals", "race", "weight_loss"],
        "keywords": ["goal", "race", "weight", "loss", "5k", "10k",
                      "marathon", "triathlon", "fitness", "choose"],
    },
    "micro_training_commute_v4": {
        "topics": ["micro_training", "commute", "time_efficient"],
        "keywords": ["commute", "short", "time", "busy", "micro", "hiit",
                      "snack", "stairs", "quick", "minutes", "10 min",
                      "15 min", "20 min"],
    },
    "motivation_v4": {
        "topics": ["motivation", "mental", "mindset"],
        "keywords": ["motivation", "motivated", "setback", "quit", "tired",
                      "bored", "discipline", "mental", "give up", "unmotivated"],
    },
    "beginner_v4": {
        "topics": ["beginner", "getting_started"],
        "keywords": ["beginner", "start", "new", "first", "never", "begin",
                      "getting started", "newbie"],
    },
    "balance_v4": {
        "topics": ["life_balance", "time_management"],
        "keywords": ["busy", "time", "balance", "work", "life", "stress",
                      "family", "schedule"],
    },
    "masters_v4": {
        "topics": ["masters", "age", "older"],
        "keywords": ["masters", "older", "age", "40", "50", "60",
                      "aging", "mature"],
    },
    "seniors_v4": {
        "topics": ["seniors", "elderly", "age"],
        "keywords": ["senior", "elderly", "70", "80", "90", "old",
                      "late", "never too late", "walking"],
    },
    "josi_personas_v1": {
        "topics": ["persona", "coaching_style"],
        "keywords": ["persona", "style", "tone", "direct", "encouraging",
                      "technical", "balanced", "dutch"],
    },
    "planner_policy_v4": {
        "topics": ["planning", "validation"],
        "keywords": ["plan", "validate", "policy"],
    },
}


def build_knowledge_entry(key: str, content: str) -> dict:
    """Build a single knowledge entry with metadata."""
    parts = key.split("__", 1)
    card = parts[0]
    section = parts[1] if len(parts) > 1 else "main"

    meta = CARD_METADATA.get(card, {})

    entry = {
        "id": key,
        "card": card,
        "section": section,
        "sport": meta.get("sport"),
        "topics": meta.get("topics", []),
        "keywords": meta.get("keywords", []),
        "content": content.strip(),
    }

    # Remove None values
    return {k: v for k, v in entry.items() if v is not None}


def main():
    parser = argparse.ArgumentParser(description="Export knowledge to JSON")
    parser.add_argument("--output", type=str,
                        default=str(GENERATED_DIR / "knowledge.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    from context import Context
    ctx = Context()
    keys = ctx.list_contexts()

    entries = []
    for key in sorted(keys):
        content = ctx._contexts[key]
        entries.append(build_knowledge_entry(key, content))

    output = {
        "version": "v6",
        "description": "MiValta coaching knowledge — ships with model for on-device inference",
        "total_entries": len(entries),
        "entries": entries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_kb = output_path.stat().st_size / 1024
    print(f"Exported {len(entries)} knowledge entries to {output_path}")
    print(f"File size: {size_kb:.1f} KB")

    # Summary by card
    cards = {}
    for e in entries:
        card = e["card"]
        cards[card] = cards.get(card, 0) + 1
    for card, count in sorted(cards.items()):
        print(f"  {card}: {count} entries")


if __name__ == "__main__":
    main()
