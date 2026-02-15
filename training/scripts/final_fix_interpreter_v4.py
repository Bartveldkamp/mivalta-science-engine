#!/usr/bin/env python3
"""
Final targeted fixes for 2 remaining interpreter failures:

1. "I have an hour for strength training" → missed time_available_min: 60
   Root cause: "an hour" isn't matched by the model (only "N min" patterns)
   Fix: Add examples with "an hour", "half an hour", "a couple hours" etc.

2. "I want a workout" (no context) → model guesses sport instead of clarify
   Root cause: model has many create_workout examples that infer sport from context
   Fix: Add more no-context clarify examples, reinforce the pattern

Usage:
    python final_fix_interpreter_v4.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def load_system_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


# Fix 1: "an hour" / "half an hour" / "a couple hours" duration extraction
HOUR_DURATION_EXAMPLES = [
    {"user": "I have an hour for strength training\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "strength", "time_available_min": 60, "free_text": "I have an hour for strength training", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I've got an hour free\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "I've got an hour free", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I have about an hour\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "I have about an hour", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "An hour for running please\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "An hour for running please", "constraints": {"fatigue_hint": "ok"}}},
    {"user": "I have an hour and a half for cycling\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 90, "free_text": "I have an hour and a half for cycling", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Half an hour before work\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 30, "free_text": "Half an hour before work", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Two hours for a long ride\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 120, "free_text": "Two hours for a long ride", "goal": "endurance", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I have about an hour for a run today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "I have about an hour for a run today", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Got an hour, let's do something hard\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "Got an hour, let's do something hard", "goal": "vo2", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I can do an hour of strength work\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "strength", "time_available_min": 60, "free_text": "I can do an hour of strength work", "constraints": {"fatigue_hint": "fresh"}}},
]

# Fix 2: More "no sport, no context" → clarify examples
# The key: these have NO CONTEXT block at all, or no Sport in context
NO_CONTEXT_CLARIFY = [
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train? Running, cycling, or something else?", "free_text": "I want a workout"}},
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport do you want a workout for?", "free_text": "I want a workout"}},
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What type of workout? Running, cycling, strength, or something else?", "free_text": "I want a workout"}},
    {"user": "Make me a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Make me a workout"}},
    {"user": "Workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength, or something else?", "free_text": "Workout"}},
    {"user": "Build me a session",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train?", "free_text": "Build me a session"}},
    {"user": "I need a workout today",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train today?", "free_text": "I need a workout today"}},
    {"user": "Give me today's workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, or something else?", "free_text": "Give me today's workout"}},
    {"user": "I want to exercise today",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What type of exercise? Running, cycling, strength, or something else?", "free_text": "I want to exercise today"}},
    {"user": "Can you create a workout for me?",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Can you create a workout for me?"}},
]


def main():
    prompt = load_system_prompt("interpreter_system.txt")
    train_path = DATA_DIR / "train_interpreter.jsonl"

    all_new = HOUR_DURATION_EXAMPLES + NO_CONTEXT_CLARIFY

    print(f"\n  Final Interpreter Fix")
    print(f"  Adding {len(HOUR_DURATION_EXAMPLES)} 'hour' duration examples")
    print(f"  Adding {len(NO_CONTEXT_CLARIFY)} no-context clarify examples")

    with open(train_path, "a") as f:
        for item in all_new:
            ex = {
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": json.dumps(item["response"], ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Count total
    with open(train_path) as f:
        total = sum(1 for _ in f)

    print(f"  Total training examples: {total}")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
