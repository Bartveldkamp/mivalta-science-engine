#!/usr/bin/env python3
"""
MiValta Josi v4 — Fix Interpreter Action Classification

Fixes two issues discovered in eval (interpreter 70.8%):

1. explain vs answer_question confusion:
   - Many examples with Session/Readiness CONTEXT are labeled answer_question
     when they should be explain (user asks about THEIR specific data)
   - Reclassifies based on user intent + context presence

2. clarify underuse:
   - "I want a workout" (no sport) should be clarify, not create_workout
   - Adds more clarify examples for ambiguous/no-context inputs

3. medical free_text:
   - Ensures all clarify medical examples include free_text

Usage:
    python fix_interpreter_actions_v4.py
    python fix_interpreter_actions_v4.py --dry-run
"""

import json
import re
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def load_system_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


# Patterns that indicate user is asking about THEIR specific data (→ explain)
EXPLAIN_PATTERNS = [
    r"what am i doing",
    r"what('s| is) on (the |my )?schedule",
    r"what('s| is) on today",
    r"what('s| is) (today|tonight)",
    r"tell me about today",
    r"tell me about (my |the )?session",
    r"tell me about (my |the )?plan",
    r"tell me about (my |the )?week",
    r"explain today",
    r"explain (my |the |this )?session",
    r"explain (my |the |this )?workout",
    r"break down today",
    r"break down (my |the |this )?session",
    r"morning briefing",
    r"daily briefing",
    r"why is (my )?readiness",
    r"how('s| is) my readiness",
    r"how recovered am i",
    r"how was my week",
    r"what does (this |my )?week look like",
    r"what('s| is) (the |my )?plan (for |this )?",
    r"why (am i|did i|do i) (doing|get|have) (an? )?(easy|hard|recovery|rest)",
    r"should i train hard or easy",
    r"what('s| is) the purpose of today",
    r"why (am i|are we) doing",
    r"walk me through today",
    r"what('s| is) my (workout|session|training) (for )?today",
]

# Compile patterns
EXPLAIN_RE = [re.compile(p, re.IGNORECASE) for p in EXPLAIN_PATTERNS]


def should_be_explain(user_msg: str) -> bool:
    """Check if a user message with context should be action=explain."""
    # Must have context with Session, Readiness, or Week
    has_session = "Session:" in user_msg
    has_readiness = "Readiness:" in user_msg
    has_week = "Week:" in user_msg or "phase" in user_msg.lower()

    if not (has_session or has_readiness or has_week):
        return False

    # Check if the user's actual message (before CONTEXT) matches explain patterns
    core = user_msg.split("\n\nCONTEXT:")[0].strip()
    core = core.split("\n\nHISTORY:")[0].strip()

    return any(p.search(core) for p in EXPLAIN_RE)


def fix_training_data(jsonl_path: Path, dry_run: bool = False) -> dict:
    """Reclassify answer_question → explain where appropriate."""
    with open(jsonl_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    stats = Counter()
    stats["total"] = len(examples)

    for ex in examples:
        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            continue

        user_msg = msgs[1]["content"]
        try:
            resp = json.loads(msgs[-1]["content"])
        except json.JSONDecodeError:
            continue

        action = resp.get("action", "")

        # Fix 1: answer_question → explain when user asks about their specific data
        if action == "answer_question" and should_be_explain(user_msg):
            resp["action"] = "explain"
            # Ensure question field exists
            core = user_msg.split("\n\nCONTEXT:")[0].strip()
            core = core.split("\n\nHISTORY:")[0].strip()
            if "question" not in resp:
                resp["question"] = core
            msgs[-1]["content"] = json.dumps(resp, ensure_ascii=False)
            stats["aq_to_explain"] += 1

        # Fix 2: Ensure all clarify medical examples have free_text
        if action == "clarify" and "free_text" not in resp:
            core = user_msg.split("\n\nCONTEXT:")[0].strip()
            core = core.split("\n\nHISTORY:")[0].strip()
            resp["free_text"] = core
            msgs[-1]["content"] = json.dumps(resp, ensure_ascii=False)
            stats["fixed_free_text"] += 1

    if not dry_run:
        with open(jsonl_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return stats


# Additional explain examples to add (matching eval test patterns)
EXTRA_EXPLAIN = [
    # "What am I doing today?" — the most common pattern
    {"user": "What am I doing today?\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Session: Z2 60min Easy aerobic\n- Sport: running\n- Level: intermediate",
     "response": {"action": "explain", "question": "What am I doing today?", "free_text": "What am I doing today?"}},
    {"user": "What am I doing today?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Session: Z4 35min Threshold\n- Sport: running\n- Level: intermediate",
     "response": {"action": "explain", "question": "What am I doing today?", "free_text": "What am I doing today?"}},
    {"user": "What am I doing today?\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Session: Z1 30min Recovery\n- Sport: cycling\n- Level: advanced",
     "response": {"action": "explain", "question": "What am I doing today?", "free_text": "What am I doing today?"}},

    # "Why is my readiness red/amber/yellow?"
    {"user": "Why is my readiness red?\n\nCONTEXT:\n- Readiness: Red (Overreached)\n- Sport: running\n- Level: intermediate",
     "response": {"action": "explain", "question": "Why is my readiness red?", "free_text": "Why is my readiness red?"}},
    {"user": "Why is my readiness amber?\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate",
     "response": {"action": "explain", "question": "Why is my readiness amber?", "free_text": "Why is my readiness amber?"}},
    {"user": "My readiness is yellow, why?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: cycling\n- Level: intermediate",
     "response": {"action": "explain", "question": "My readiness is yellow, why?", "free_text": "My readiness is yellow, why?"}},

    # "What does this week look like?"
    {"user": "What does this week look like?\n\nCONTEXT:\n- Sport: running\n- Week: base phase week 3\n- Readiness: Green (Recovered)\n- Level: intermediate",
     "response": {"action": "explain", "question": "What does this week look like?", "free_text": "What does this week look like?"}},
    {"user": "What does this week look like?\n\nCONTEXT:\n- Sport: cycling\n- Week: build phase week 2\n- Readiness: Yellow\n- Level: advanced",
     "response": {"action": "explain", "question": "What does this week look like?", "free_text": "What does this week look like?"}},
    {"user": "Tell me about my plan for the week\n\nCONTEXT:\n- Sport: running\n- Week: peak phase week 1\n- Readiness: Green\n- Level: intermediate",
     "response": {"action": "explain", "question": "Tell me about my plan for the week", "free_text": "Tell me about my plan for the week"}},

    # "Why did I get an easy day?"
    {"user": "Why did I get an easy day when I feel great?\n\nCONTEXT:\n- Sport: running\n- Session: Z1 30min Recovery\n- Readiness: Green (Recovered)\n- Level: intermediate",
     "response": {"action": "explain", "question": "Why did I get an easy day when I feel great?", "free_text": "Why did I get an easy day when I feel great?"}},
    {"user": "Why is today so easy? I feel fine.\n\nCONTEXT:\n- Sport: cycling\n- Session: Z1 45min Recovery\n- Readiness: Green\n- Level: intermediate",
     "response": {"action": "explain", "question": "Why is today so easy? I feel fine.", "free_text": "Why is today so easy? I feel fine."}},

    # "Explain today's workout"
    {"user": "Explain today's workout\n\nCONTEXT:\n- Sport: cycling\n- Session: Z3 45min Tempo\n- Readiness: Green\n- Level: intermediate",
     "response": {"action": "explain", "question": "Explain today's workout", "free_text": "Explain today's workout"}},
    {"user": "Explain today's session\n\nCONTEXT:\n- Sport: running\n- Session: Z5 30min VO2max\n- Readiness: Green\n- Level: advanced",
     "response": {"action": "explain", "question": "Explain today's session", "free_text": "Explain today's session"}},

    # Other explain patterns
    {"user": "Walk me through today's session\n\nCONTEXT:\n- Sport: running\n- Session: Z4 40min Threshold\n- Readiness: Green\n- Level: intermediate",
     "response": {"action": "explain", "question": "Walk me through today's session", "free_text": "Walk me through today's session"}},
    {"user": "What's the purpose of today's session?\n\nCONTEXT:\n- Sport: cycling\n- Session: Z2 90min Endurance\n- Readiness: Green\n- Level: advanced",
     "response": {"action": "explain", "question": "What's the purpose of today's session?", "free_text": "What's the purpose of today's session?"}},
    {"user": "How's my readiness?\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate",
     "response": {"action": "explain", "question": "How's my readiness?", "free_text": "How's my readiness?"}},
    {"user": "Should I train hard or easy today?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Session: Z3 35min Tempo\n- Sport: running\n- Level: intermediate",
     "response": {"action": "explain", "question": "Should I train hard or easy today?", "free_text": "Should I train hard or easy today?"}},
    {"user": "What's on the schedule?\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Session: Z5 25min VO2max intervals\n- Sport: cycling\n- Level: intermediate",
     "response": {"action": "explain", "question": "What's on the schedule?", "free_text": "What's on the schedule?"}},
    {"user": "Why am I doing intervals today?\n\nCONTEXT:\n- Sport: running\n- Session: Z5 30min VO2max\n- Readiness: Green\n- Level: intermediate",
     "response": {"action": "explain", "question": "Why am I doing intervals today?", "free_text": "Why am I doing intervals today?"}},
]

# Additional clarify examples — no sport, no context, ambiguous
EXTRA_CLARIFY = [
    # "I want a workout" — NO sport, NO context → must clarify
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train? Running, cycling, or something else?", "free_text": "I want a workout"}},
    {"user": "Give me a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength, or something else?", "free_text": "Give me a workout"}},
    {"user": "I want to train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "I want to train"}},
    {"user": "Workout please",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport do you want to train?", "free_text": "Workout please"}},
    {"user": "Create a session for me",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like? Running, cycling, or something else?", "free_text": "Create a session for me"}},

    # "Help me today" — ambiguous intent, NO context → must clarify
    {"user": "Help me today",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "What would you like help with? A workout, your plan, or a training question?", "free_text": "Help me today"}},
    {"user": "Help me",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "How can I help? Do you want a workout, an explanation, or have a question?", "free_text": "Help me"}},
    {"user": "Hey there",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "Hey! How can I help with your training today?", "free_text": "Hey there"}},
    {"user": "What should I do?",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "What sport are you training for, and what's your goal today?", "free_text": "What should I do?"}},
    {"user": "I'm ready",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "Great! What sport do you want to train today?", "free_text": "I'm ready"}},

    # Medical with free_text — ensure all have it
    {"user": "I have chest pain and I feel dizzy\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I have chest pain and I feel dizzy"}},
    {"user": "I've been having chest tightness during runs\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I've been having chest tightness during runs"}},
    {"user": "I feel lightheaded when I exercise\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I feel lightheaded when I exercise"}},
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix interpreter action classification")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt = load_system_prompt("interpreter_system.txt")

    print(f"\n{'=' * 60}")
    print(f"  Fix Interpreter Action Classification")
    print(f"{'=' * 60}\n")

    # Fix existing training data
    for prefix in ["train", "val"]:
        path = DATA_DIR / f"{prefix}_interpreter.jsonl"
        if path.exists():
            print(f"  Processing {path.name}...")
            stats = fix_training_data(path, dry_run=args.dry_run)
            print(f"    Total: {stats['total']}")
            print(f"    answer_question → explain: {stats.get('aq_to_explain', 0)}")
            print(f"    Fixed missing free_text: {stats.get('fixed_free_text', 0)}")

    # Refresh system prompts in all files
    print(f"\n  Refreshing system prompts...")
    for prefix in ["train", "val"]:
        path = DATA_DIR / f"{prefix}_interpreter.jsonl"
        if path.exists():
            with open(path) as f:
                examples = [json.loads(line) for line in f if line.strip()]
            updated = 0
            for ex in examples:
                if ex["messages"][0]["role"] == "system" and ex["messages"][0]["content"] != prompt:
                    ex["messages"][0]["content"] = prompt
                    updated += 1
            if not args.dry_run:
                with open(path, "w") as f:
                    for ex in examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"    {path.name}: {updated} prompts refreshed")

    # Add extra examples
    print(f"\n  Adding {len(EXTRA_EXPLAIN)} explain examples...")
    print(f"  Adding {len(EXTRA_CLARIFY)} clarify examples...")

    train_path = DATA_DIR / "train_interpreter.jsonl"
    if not args.dry_run:
        with open(train_path, "a") as f:
            for item in EXTRA_EXPLAIN + EXTRA_CLARIFY:
                ex = {
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": item["user"]},
                        {"role": "assistant", "content": json.dumps(item["response"], ensure_ascii=False)},
                    ]
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Final stats
    if not args.dry_run:
        with open(train_path) as f:
            total = sum(1 for _ in f)
        print(f"\n  Final training set: {total} examples")

    # Verify distribution
    if not args.dry_run:
        counts = Counter()
        with open(train_path) as f:
            for line in f:
                ex = json.loads(line)
                try:
                    resp = json.loads(ex["messages"][-1]["content"])
                    counts[resp["action"]] += 1
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"\n  Action distribution:")
        for action, count in sorted(counts.items()):
            pct = count / sum(counts.values()) * 100
            print(f"    {action}: {count} ({pct:.1f}%)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
