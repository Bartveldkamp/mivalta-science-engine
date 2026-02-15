#!/usr/bin/env python3
"""
MiValta Josi v4 — Training Data Converter (LLMIntent → Dual-Mode)

Converts the existing train_v3.jsonl (single-mode LLMIntent JSON responses)
into two clean datasets for the dual-mode architecture:

  1. train_interpreter.jsonl — LLM outputs small GATCRequest JSON
  2. train_explainer.jsonl   — LLM outputs plain coaching text

The old LLMIntent format crammed structured routing + coaching text into one
JSON blob, which a 2B model couldn't reliably produce within 150 tokens.
The new architecture splits these responsibilities:

  - Interpreter: detect intent, extract parameters → small JSON
  - Explainer:   explain GATC results, answer questions → plain text

Usage:
    python convert_training_data_v4.py
    python convert_training_data_v4.py --input data/train_v3.jsonl --stats
    python convert_training_data_v4.py --input data/val_v3.jsonl --output-prefix data/val
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter


# ============================================================================
# MAPPING: Old LLMIntent → New GATCRequest
# ============================================================================

# Map old intent + response_type + tool_call → new action
def map_to_action(intent: str, response_type: str, tool_call: dict | None) -> str:
    """Map old LLMIntent fields to new GATCRequest action."""

    # Tool calls map directly
    if tool_call is not None:
        tool = tool_call.get("tool", "")
        if tool in ("create_today_workout", "create_plan"):
            return "create_workout"
        if tool in ("replan",):
            return "replan"
        if tool in ("explain_workout", "get_user_status", "get_recent_workouts"):
            return "explain"
        if tool in ("log_workout",):
            return "answer_question"  # logging is an action, but we treat it as Q&A for now

    # Blocked / decline — the user tried to do something the validator should reject.
    # Map to what they were *trying* to do so GATC's validator can give a proper error.
    if intent == "blocked":
        if response_type == "Decline":
            # User tried to prescribe/modify training → validator will reject
            return "create_workout"
        return "answer_question"

    # Medical red flag → clarify (stop training, see a doctor)
    if intent == "medical_red_flag":
        return "clarify"

    # Replan intent
    if intent == "replan":
        return "replan"

    # Map by response type
    type_map = {
        "ExplainWorkout": "explain",
        "ExplainZone": "answer_question",
        "DailyBrief": "explain",
        "ReadinessSummary": "explain",
        "WeeklyReview": "explain",
        "QuestionAnswer": "answer_question",
        "Encouragement": "answer_question",
        "SafetyWarning": "answer_question",
        "Decline": "answer_question",
    }
    return type_map.get(response_type, "answer_question")


def extract_sport_from_context(user_msg: str) -> str | None:
    """Try to extract sport from the user message / context block."""
    lower = user_msg.lower()

    # Look for explicit "Sport:" in context block
    sport_match = re.search(r'sport:\s*(\w+)', lower)
    if sport_match:
        sport_text = sport_match.group(1)
        sport_map = {
            "running": "run", "run": "run", "cycling": "bike",
            "bike": "bike", "biking": "bike", "swimming": "other",
            "triathlon": "other", "skiing": "ski", "ski": "ski",
            "skating": "skate", "skate": "skate", "strength": "strength",
            "gym": "strength", "weight": "strength",
        }
        return sport_map.get(sport_text, "other")

    # Infer from keywords in the message
    if any(w in lower for w in ["run", "running", "jog", "marathon", "5k", "10k"]):
        return "run"
    if any(w in lower for w in ["bike", "cycling", "ride", "pedal", "cycling"]):
        return "bike"
    if any(w in lower for w in ["ski", "skiing", "cross-country"]):
        return "ski"
    if any(w in lower for w in ["skate", "skating", "ice"]):
        return "skate"
    if any(w in lower for w in ["gym", "strength", "weight", "lift"]):
        return "strength"

    return None


def extract_duration_from_context(user_msg: str) -> int | None:
    """Try to extract duration in minutes from user message / context."""
    lower = user_msg.lower()

    # Look for explicit duration patterns: "45 minutes", "30 min"
    m = re.search(r'(\d+)\s*(?:min|minutes|mins|minute)', lower)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 300:
            return val

    # Look for hour patterns: "an hour", "one hour", "1 hour", "2 hours"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)', lower)
    if m:
        val = int(float(m.group(1)) * 60)
        if 10 <= val <= 300:
            return val

    # "an hour" / "one hour" / "half an hour"
    if re.search(r'\bhalf\s+an?\s+hour\b', lower):
        return 30
    if re.search(r'\ban?\s+hour\b', lower) or re.search(r'\bone\s+hour\b', lower):
        return 60

    # Look for "Z2 60min" style in context
    m = re.search(r'[ZzRr]\d\s+(\d+)\s*min', user_msg)
    if m:
        return int(m.group(1))

    return None


def extract_fatigue_hint(user_msg: str) -> str | None:
    """Infer fatigue level from user language."""
    lower = user_msg.lower()

    if any(w in lower for w in ["exhausted", "terrible", "awful", "drained", "very tired"]):
        return "very_tired"
    if any(w in lower for w in ["tired", "fatigue", "exhaustion", "sleepy", "low energy"]):
        return "tired"
    if any(w in lower for w in ["great", "fresh", "energized", "amazing", "fantastic"]):
        return "fresh"
    if any(w in lower for w in ["ok", "okay", "fine", "decent", "alright", "not bad"]):
        return "ok"

    # Check readiness context
    readiness_match = re.search(r'readiness:\s*(\w+)', lower)
    if readiness_match:
        readiness = readiness_match.group(1)
        readiness_map = {
            "green": "fresh", "recovered": "fresh",
            "yellow": "ok", "amber": "ok",
            "orange": "tired",
            "red": "very_tired", "overreached": "very_tired",
        }
        return readiness_map.get(readiness)

    return None


def extract_replan_type(old_data: dict, user_msg: str) -> str | None:
    """Extract replan type from old LLMIntent or user message."""
    # From old replan_request
    replan = old_data.get("replan_request")
    if replan and isinstance(replan, dict):
        return replan.get("type")

    # Infer from user message
    lower = user_msg.lower()
    if any(w in lower for w in ["skip", "can't today", "cancel today"]):
        return "skip_today"
    if any(w in lower for w in ["swap", "switch"]):
        return "swap_days"
    if any(w in lower for w in ["reschedule", "move", "different day"]):
        return "reschedule"
    if any(w in lower for w in ["easier", "reduce", "less intense", "lighter"]):
        return "reduce_intensity"
    if any(w in lower for w in ["sick", "ill", "cold", "flu"]):
        return "illness"
    if any(w in lower for w in ["travel", "trip", "vacation", "holiday"]):
        return "travel"
    if any(w in lower for w in ["new goal", "change goal", "different goal"]):
        return "goal_change"

    return "skip_today"  # default


def extract_goal_from_context(user_msg: str) -> str | None:
    """Try to infer training goal from context."""
    lower = user_msg.lower()

    if any(w in lower for w in ["recovery", "easy", "rest", "zone 1", "z1", "zone 2", "z2"]):
        return "recovery"
    if any(w in lower for w in ["threshold", "tempo", "zone 4", "z4"]):
        return "threshold"
    if any(w in lower for w in ["vo2", "interval", "zone 5", "z5", "hard"]):
        return "vo2"
    if any(w in lower for w in ["endurance", "base", "aerobic", "long"]):
        return "endurance"
    if any(w in lower for w in ["race", "competition", "event"]):
        return "race_prep"
    if any(w in lower for w in ["strength", "gym", "weights"]):
        return "strength"

    return None


def build_gatc_request(old_data: dict, user_msg: str) -> dict:
    """Build a GATCRequest from old LLMIntent data + user message."""
    action = map_to_action(
        old_data.get("intent", "general"),
        old_data.get("response_type", "QuestionAnswer"),
        old_data.get("tool_call"),
    )

    request = {
        "action": action,
        "free_text": user_msg.split("\n\nCONTEXT:")[0].strip(),  # strip context block
    }

    # Handle medical red flag → clarify with safety message
    if old_data.get("intent") == "medical_red_flag":
        request["action"] = "clarify"
        request["missing"] = ["medical_clearance"]
        request["clarify_message"] = "Please stop training and consult a medical professional immediately."
        return request

    # Add fields based on action
    if action == "create_workout":
        sport = extract_sport_from_context(user_msg)
        if sport:
            request["sport"] = sport

        duration = extract_duration_from_context(user_msg)
        if duration:
            request["time_available_min"] = duration

        goal = extract_goal_from_context(user_msg)
        if goal:
            request["goal"] = goal

        fatigue = extract_fatigue_hint(user_msg)
        if fatigue:
            request["constraints"] = {"fatigue_hint": fatigue}

    elif action == "replan":
        request["replan_type"] = extract_replan_type(old_data, user_msg)
        sport = extract_sport_from_context(user_msg)
        if sport:
            request["sport"] = sport

    elif action in ("explain", "answer_question"):
        # Extract the core question (strip CONTEXT and HISTORY blocks)
        core = user_msg.split("\n\nCONTEXT:")[0].strip()
        core = core.split("\n\nHISTORY:")[0].strip()
        if core:
            request["question"] = core

    return request


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

def load_system_prompt(filename: str) -> str:
    """Load a system prompt from the prompts directory."""
    path = PROMPTS_DIR / filename
    return path.read_text().strip()


# ============================================================================
# CONVERSION
# ============================================================================

def convert_example_interpreter(messages: list[dict]) -> dict | None:
    """Convert a training example to Interpreter format (user → GATCRequest JSON)."""
    user_msg = ""
    assistant_content = ""

    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    if not user_msg or not assistant_content:
        return None

    # Parse old LLMIntent JSON from assistant content
    try:
        old_data = json.loads(assistant_content)
    except json.JSONDecodeError:
        return None

    # Build new GATCRequest
    request = build_gatc_request(old_data, user_msg)

    # Build new training example
    system_prompt = load_system_prompt("interpreter_system.txt")
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(request, ensure_ascii=False)},
        ]
    }


def convert_example_explainer(messages: list[dict]) -> dict | None:
    """Convert a training example to Explainer format (context → plain text)."""
    user_msg = ""
    assistant_content = ""

    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    if not user_msg or not assistant_content:
        return None

    # Parse old LLMIntent JSON to extract the coaching text
    try:
        old_data = json.loads(assistant_content)
    except json.JSONDecodeError:
        # If it's already plain text, use it directly
        if len(assistant_content.split()) >= 10:
            coaching_text = assistant_content
        else:
            return None
    else:
        coaching_text = old_data.get("message", "")

    if not coaching_text or len(coaching_text.strip()) < 10:
        return None

    # Clean the coaching text
    coaching_text = coaching_text.strip()

    # Quality filters
    word_count = len(coaching_text.split())
    if word_count < 8:
        return None
    if word_count > 150:
        # Trim at sentence boundary
        words = coaching_text.split()
        trimmed = " ".join(words[:100])
        for end_char in [".", "!", "?"]:
            last_idx = trimmed.rfind(end_char)
            if last_idx > len(trimmed) * 0.5:
                trimmed = trimmed[:last_idx + 1]
                break
        coaching_text = trimmed

    # Check for forbidden words (these should never be in training data)
    forbidden = [
        "gatc", "algorithm", "viterbi", "hmm", "hidden markov",
        "acwr", "acute:chronic", "ewma", "tss", "ctl", "atl", "tsb",
    ]
    lower = coaching_text.lower()
    if any(w in lower for w in forbidden):
        return None

    # Build new training example
    system_prompt = load_system_prompt("explainer_system.txt")
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": coaching_text},
        ]
    }


def convert_file(
    input_path: str,
    output_prefix: str,
    stats_only: bool = False,
) -> dict:
    """Convert a training data file into interpreter + explainer datasets."""

    with open(input_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    print(f"  Loaded {len(raw)} examples from {input_path}")

    interpreter_examples = []
    explainer_examples = []
    stats = Counter()

    for ex in raw:
        messages = ex.get("messages", [])

        # Interpreter conversion
        interp = convert_example_interpreter(messages)
        if interp:
            interpreter_examples.append(interp)
            stats["interpreter_ok"] += 1

            # Track action distribution
            try:
                req = json.loads(interp["messages"][-1]["content"])
                stats[f"action:{req['action']}"] += 1
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            stats["interpreter_skip"] += 1

        # Explainer conversion
        expl = convert_example_explainer(messages)
        if expl:
            explainer_examples.append(expl)
            stats["explainer_ok"] += 1

            # Track word count
            words = len(expl["messages"][-1]["content"].split())
            stats["explainer_total_words"] += words
        else:
            stats["explainer_skip"] += 1

    # Report
    print(f"\n  Interpreter: {stats['interpreter_ok']} examples ({stats['interpreter_skip']} skipped)")
    print(f"  Explainer:   {stats['explainer_ok']} examples ({stats['explainer_skip']} skipped)")

    if stats["explainer_ok"] > 0:
        avg_words = stats["explainer_total_words"] / stats["explainer_ok"]
        print(f"  Explainer avg words: {avg_words:.0f}")

    print(f"\n  Action distribution (Interpreter):")
    for key in sorted(stats.keys()):
        if key.startswith("action:"):
            print(f"    {key}: {stats[key]}")

    if stats_only:
        return stats

    # Write files
    interp_path = f"{output_prefix}_interpreter.jsonl"
    expl_path = f"{output_prefix}_explainer.jsonl"

    with open(interp_path, "w") as f:
        for ex in interpreter_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n  Written: {interp_path} ({len(interpreter_examples)} examples)")

    with open(expl_path, "w") as f:
        for ex in explainer_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Written: {expl_path} ({len(explainer_examples)} examples)")

    # Write a combined sample for quick inspection
    sample_path = f"{output_prefix}_samples.json"
    samples = {
        "interpreter_samples": [
            {
                "user": ex["messages"][1]["content"][:100],
                "gatc_request": json.loads(ex["messages"][-1]["content"]),
            }
            for ex in interpreter_examples[:5]
        ],
        "explainer_samples": [
            {
                "user": ex["messages"][1]["content"][:100],
                "coaching_text": ex["messages"][-1]["content"][:200],
            }
            for ex in explainer_examples[:5]
        ],
    }
    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  Written: {sample_path} (quick inspection)")

    return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLMIntent training data to dual-mode (Interpreter + Explainer)"
    )
    parser.add_argument(
        "--input", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "train_v3.jsonl"),
        help="Input JSONL file (LLMIntent format)",
    )
    parser.add_argument(
        "--val-input", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "val_v3.jsonl"),
        help="Validation JSONL file (LLMIntent format)",
    )
    parser.add_argument(
        "--output-prefix", type=str, default=None,
        help="Output prefix (default: data/train)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show stats only, don't write files",
    )

    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"

    print(f"\n{'=' * 60}")
    print(f"  Converting Training Data: LLMIntent → Dual-Mode")
    print(f"  Interpreter (GATCRequest JSON) + Explainer (plain text)")
    print(f"{'=' * 60}\n")

    # Convert training data
    train_prefix = args.output_prefix or str(data_dir / "train")
    print(f"--- Training Data ---")
    convert_file(args.input, train_prefix, stats_only=args.stats)

    # Convert validation data if it exists
    val_input = args.val_input
    if Path(val_input).exists():
        val_prefix = str(data_dir / "val")
        print(f"\n--- Validation Data ---")
        convert_file(val_input, val_prefix, stats_only=args.stats)

    print(f"\n{'=' * 60}")
    print(f"  Conversion complete!")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. Inspect samples: cat {data_dir}/train_samples.json | python -m json.tool")
    print(f"    2. Train interpreter: python finetune_gemma3n.py train --train_data {data_dir}/train_interpreter.jsonl")
    print(f"    3. Train explainer:   python finetune_gemma3n.py train --train_data {data_dir}/train_explainer.jsonl")
    print(f"    4. Evaluate:          python evaluate_gemma3n_v4.py --model <gguf_path>")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
