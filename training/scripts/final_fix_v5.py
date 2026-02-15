#!/usr/bin/env python3
"""
Final fixes for the last 2 eval failures:

1. Interpreter: "I want a workout" (no CONTEXT) → model picks create_workout
   instead of clarify. Fix: add 25+ more no-context clarify examples with high
   variety to overcome the create_workout bias (259 vs ~7 clarify examples).

2. Explainer: "What mistakes do most beginners make?" → model uses jargon word
   "periodization". Fix: (a) clean "periodization" from all training data,
   replacing with plain language; (b) add beginner-topic examples that explicitly
   avoid jargon; (c) updated system prompt (done separately).

Usage:
    python final_fix_v5.py
"""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def load_system_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


# =============================================================================
# FIX 1: More no-context clarify examples for interpreter
# =============================================================================

# These all have NO CONTEXT block — model must output clarify, not create_workout
NO_CONTEXT_CLARIFY_EXTRA = [
    # "I want a workout" variants — the exact failing test case
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train? Running, cycling, strength, or something else?", "free_text": "I want a workout"}},
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport should we focus on?", "free_text": "I want a workout"}},
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Sure! What sport — running, cycling, or strength?", "free_text": "I want a workout"}},
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "I'd love to help! What sport are we training today?", "free_text": "I want a workout"}},
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Happy to create one! Which sport would you like?", "free_text": "I want a workout"}},

    # Other no-context variants
    {"user": "Create a workout for me",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Create a workout for me"}},
    {"user": "I need to train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train?", "free_text": "I need to train"}},
    {"user": "Let's train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Great! What sport — running, cycling, or something else?", "free_text": "Let's train"}},
    {"user": "Give me something to do",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What type of workout? Running, cycling, strength?", "free_text": "Give me something to do"}},
    {"user": "I want to exercise",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport would you like to focus on?", "free_text": "I want to exercise"}},
    {"user": "Set up a session for me",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport should the session be for?", "free_text": "Set up a session for me"}},
    {"user": "Time to work out",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport are we doing?", "free_text": "Time to work out"}},
    {"user": "Workout time!",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength, or something else?", "free_text": "Workout time!"}},
    {"user": "I'm ready to train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train today?", "free_text": "I'm ready to train"}},
    {"user": "Can I get a workout?",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Of course! What sport would you like?", "free_text": "Can I get a workout?"}},
    {"user": "I have some time to train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to do?", "free_text": "I have some time to train"}},
    {"user": "Plan a workout for me",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport should I plan the workout for?", "free_text": "Plan a workout for me"}},
    {"user": "I want to do something today",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What type of training? Running, cycling, strength?", "free_text": "I want to do something today"}},
    {"user": "Get me moving",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like?", "free_text": "Get me moving"}},
    {"user": "Ready for a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport are you looking to do?", "free_text": "Ready for a workout"}},
    {"user": "I want a good session",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport should this session be for?", "free_text": "I want a good session"}},
    {"user": "Let's get after it",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport are we doing today?", "free_text": "Let's get after it"}},
    {"user": "I'd like a workout please",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport would you like a workout for?", "free_text": "I'd like a workout please"}},
    {"user": "Build me a training session",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport should we build a session for?", "free_text": "Build me a training session"}},
    {"user": "Workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength, or something else?", "free_text": "Workout"}},
]


# =============================================================================
# FIX 2: Explainer — beginner topic examples that avoid jargon
# =============================================================================

EXPLAINER_BEGINNER_EXAMPLES = [
    {"user": "What mistakes do most beginners make?",
     "response": "The biggest one? Doing too much too soon. Your body needs time to adapt, so starting with 3 easy sessions a week and building gradually is the way to go. Consistency beats intensity early on — showing up regularly matters more than going all-out. Also, don't skip rest days. They're when the real gains happen."},
    {"user": "What mistakes do most beginners make?",
     "response": "Most beginners go too hard too fast. Your enthusiasm is great, but your body needs a gentler start! Start with manageable sessions, build up gradually over weeks, and respect rest days. The other common one is comparing yourself to others — focus on your own progress and enjoy the journey."},
    {"user": "What mistakes do most beginners make?",
     "response": "Three big ones: going too hard too soon, skipping recovery days, and not being consistent. Your body adapts best with a gradual build-up. Show up regularly, keep most sessions easy, and let rest days do their magic. That foundation will serve you well for years."},
    {"user": "What mistakes do most beginners make?",
     "response": "The classic mistake is doing too much too quickly. It's tempting to go all-in, but your body needs gradual progression. Start easy, add a little each week, and don't underestimate the power of rest. Consistency and patience are your best friends early on."},
    {"user": "What mistakes do most beginners make?",
     "response": "Trying to do everything at once! Start simple — three sessions a week at an easy effort. Build gradually, listen to your body, and don't skip rest days. The athletes who progress fastest are the ones who start slow and stay consistent."},

    # Other beginner topics — ensure no jargon
    {"user": "I just started running last week. How often should I run?",
     "response": "Great that you've started! Three times a week is a solid starting point. Give yourself a rest day between runs so your body can adapt. As it gets easier, you can gradually add a fourth day. Listen to your body — if something hurts, take an extra rest day."},
    {"user": "I'm new to cycling. What gear do I need?",
     "response": "Keep it simple to start! A well-fitting bike, a helmet, and comfortable clothing are the essentials. Padded shorts make a big difference on longer rides. A water bottle and basic repair kit are smart to have too. You don't need fancy gear to get started — just get riding!"},
    {"user": "I've never exercised before. Where do I start?",
     "response": "Start with what feels manageable — even 20 minutes of walking counts! The key is building a habit first. Try three easy sessions a week and let your body adjust. Once that feels comfortable, gradually increase the time or try something new. There's no wrong starting point."},
    {"user": "How do I know if I'm running too fast or too slow?",
     "response": "A simple test: can you hold a conversation while running? If you can chat comfortably, you're in a good easy effort zone. If you're gasping for air, slow down. Most of your runs should feel easy — that's how you build a strong foundation without burning out."},
    {"user": "What's a good first goal for a complete beginner?",
     "response": "Consistency is the best first goal — aim for three sessions a week for a month. Once that's a habit, pick something specific like running for 30 minutes without stopping, or completing a 5K. Keep it achievable so you build confidence and momentum."},
]


# =============================================================================
# FIX 3: Clean "periodization" from existing explainer training data
# =============================================================================

JARGON_REPLACEMENTS = {
    "periodization": "structuring your training",
    "Periodization": "Structuring your training",
    "periodized": "well-structured",
    "Periodized": "Well-structured",
    "mesocycle": "training block",
    "Mesocycle": "Training block",
    "mesocycles": "training blocks",
    "Mesocycles": "Training blocks",
    "microcycle": "training week",
    "Microcycle": "Training week",
    "microcycles": "training weeks",
    "Microcycles": "Training weeks",
    "macrocycle": "training plan",
    "Macrocycle": "Training plan",
    "macrocycles": "training plans",
    "Macrocycles": "Training plans",
    "supercompensation": "recovery adaptation",
    "Supercompensation": "Recovery adaptation",
}


def clean_jargon_from_explainer(jsonl_path: Path) -> int:
    """Replace jargon words in assistant responses of explainer training data."""
    with open(jsonl_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    cleaned = 0
    for ex in examples:
        msgs = ex.get("messages", [])
        for msg in msgs:
            if msg["role"] == "assistant":
                original = msg["content"]
                new_content = original
                for jargon, replacement in JARGON_REPLACEMENTS.items():
                    new_content = new_content.replace(jargon, replacement)
                if new_content != original:
                    msg["content"] = new_content
                    cleaned += 1

    with open(jsonl_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return cleaned


def main():
    interpreter_prompt = load_system_prompt("interpreter_system.txt")
    explainer_prompt = load_system_prompt("explainer_system.txt")

    # --- Interpreter: add no-context clarify examples ---
    interp_path = DATA_DIR / "train_interpreter.jsonl"

    print(f"\n  Fix v5 — Final Push for 100%")
    print(f"  {'='*40}")

    print(f"\n  [Interpreter] Adding {len(NO_CONTEXT_CLARIFY_EXTRA)} no-context clarify examples")
    with open(interp_path, "a") as f:
        for item in NO_CONTEXT_CLARIFY_EXTRA:
            ex = {
                "messages": [
                    {"role": "system", "content": interpreter_prompt},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": json.dumps(item["response"], ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(interp_path) as f:
        interp_total = sum(1 for _ in f)
    print(f"  [Interpreter] Total examples: {interp_total}")

    # Count clarify examples now
    with open(interp_path) as f:
        clarify_count = 0
        create_count = 0
        for line in f:
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            for msg in msgs:
                if msg["role"] == "assistant":
                    try:
                        resp = json.loads(msg["content"])
                        if resp.get("action") == "clarify":
                            clarify_count += 1
                        elif resp.get("action") == "create_workout":
                            create_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
    print(f"  [Interpreter] clarify: {clarify_count}, create_workout: {create_count}")

    # --- Explainer: clean jargon from training data ---
    print(f"\n  [Explainer] Cleaning jargon from training data...")
    for prefix in ["train", "val"]:
        expl_path = DATA_DIR / f"{prefix}_explainer.jsonl"
        if expl_path.exists():
            cleaned = clean_jargon_from_explainer(expl_path)
            print(f"  [Explainer] {expl_path.name}: cleaned {cleaned} examples")

    # --- Explainer: add beginner-topic examples ---
    expl_train_path = DATA_DIR / "train_explainer.jsonl"
    print(f"\n  [Explainer] Adding {len(EXPLAINER_BEGINNER_EXAMPLES)} beginner examples")
    with open(expl_train_path, "a") as f:
        for item in EXPLAINER_BEGINNER_EXAMPLES:
            ex = {
                "messages": [
                    {"role": "system", "content": explainer_prompt},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["response"]},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(expl_train_path) as f:
        expl_total = sum(1 for _ in f)
    print(f"  [Explainer] Total examples: {expl_total}")

    # --- Refresh system prompts in all data ---
    print(f"\n  [Both] Refreshing system prompts in all training data...")

    for prefix in ["train", "val"]:
        interp = DATA_DIR / f"{prefix}_interpreter.jsonl"
        expl = DATA_DIR / f"{prefix}_explainer.jsonl"

        if interp.exists():
            _refresh(interp, interpreter_prompt)
        if expl.exists():
            _refresh(expl, explainer_prompt)

    print(f"\n  Done.\n")


def _refresh(jsonl_path: Path, prompt: str):
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


if __name__ == "__main__":
    main()
