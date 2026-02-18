#!/usr/bin/env python3
"""
Fix explainer training data — v5 quality cleanup.

Addresses three issues found during review:

  1. "Engine" references (205 examples, 18%): Responses that reference
     "the engine", "the planning engine", "the coaching engine" break
     Josi's coach persona. Rewritten to sound like a coach explaining
     decisions, not a system deferring to an algorithm.

  2. Banned jargon (80 examples, 7%): Responses containing terms
     explicitly forbidden by the system prompt (lactate threshold,
     vo2max, anaerobic capacity, etc.). Replaced with plain-language
     equivalents per the system prompt's guidance.

  3. Short/robotic deflections: Terse "falls outside my scope" or
     "not my role" responses replaced with empathetic coach language.

Usage:
    python fix_explainer_v5.py
    # Reads:  training/data/train_explainer.jsonl
    # Writes: training/data/train_explainer.jsonl (in-place backup at .bak)
    # Also fixes: training/data/val_explainer.jsonl
"""

import json
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

TRAIN_FILE = DATA_DIR / "train_explainer.jsonl"
VAL_FILE = DATA_DIR / "val_explainer.jsonl"


# ============================================================================
# 1. ENGINE REFERENCE REPLACEMENTS
# ============================================================================

# Full-response replacements for common robotic patterns
ENGINE_FULL_REPLACEMENTS = {
    # The most common offender (appears ~40 times)
    "Modifying training parameters falls outside my scope. The engine uses your biometric data and training history to optimize load. Would you like me to elaborate on today's prescription?":
        "I can't change the plan on the fly, but I can help you understand why it looks the way it does. Your training is built around where your body is right now — that's how we keep you progressing safely.",

    "No. Plan is set by the engine. Ask me why, not to change it.":
        "I can't change the plan directly, but I can definitely explain why it's set up this way. What part are you curious about?",

    "I can't modify your training plan — that's the engine's job based on your physiology.":
        "I can't change the plan directly, but it's built around your current fitness and recovery. I'm happy to explain the thinking behind it.",
}

# Substring replacements for engine references embedded in otherwise-good responses
ENGINE_SUBSTRING_REPLACEMENTS = [
    # "The engine" variants
    (r"[Tt]he (?:planning |coaching )?engine (?:uses|considers|takes into account|factors in|will recalculate|has|adjusts)",
     "Your training plan takes into account"),
    (r"[Tt]he engine",
     "your training plan"),
    # "set by the engine" / "decided by the engine"
    (r"(?:set|decided|determined|calculated|optimized) by the engine",
     "built around your current fitness"),
    # "the engine's job" / "the engine's role"
    (r"the engine's (?:job|role|responsibility)",
     "how the planning works"),
    # "engine will" / "engine should"
    (r"engine will (?:recalculate|adjust|update|modify)",
     "your plan will adjust"),
    # "outside my scope" / "not my role" (robotic deflections)
    (r"(?:falls |is )?outside my (?:scope|role|capabilities)",
     "not something I can change directly"),
    (r"(?:Modifying|Changing|Adjusting) training parameters (?:falls |is )?(?:outside my scope|not my role|not something I do)",
     "I can't change the plan on the fly"),
    # "Initiating X replan request"
    (r"Initiating \w+ replan request\.",
     "I'll get that adjustment started."),
    (r"The planning engine will recalculate based on current parameters and readiness state \(\w+\)\.",
     "Your plan will adjust based on how you're doing right now."),
]


# ============================================================================
# 2. JARGON REPLACEMENTS
# ============================================================================

JARGON_REPLACEMENTS = [
    # Must be ordered longest-first to avoid partial matches
    (r"\blactate threshold\b", "your threshold"),
    (r"\banaerobic capacity\b", "short-burst power"),
    (r"\bAnaerobic Capacity\b", "Short-Burst Power"),
    (r"\bglycolytic capacity\b", "high-intensity endurance"),
    (r"\banaerobic power\b", "explosive effort"),
    (r"\bvo2max\b", "your fitness ceiling", re.IGNORECASE),
    (r"\bVO2max\b", "your fitness ceiling"),
    (r"\bperiodization\b", "structuring your training", re.IGNORECASE),
    (r"\bftp\b", "your threshold power", re.IGNORECASE),
    (r"\bFTP\b", "your threshold power"),
    (r"\bthreshold power\b", "your sustained-effort ceiling"),
    (r"\bsupercompensation\b", "the recovery bounce-back"),
    (r"\bmesocycle\b", "training block"),
    (r"\bmicrocycle\b", "training week"),
    (r"\bmacrocycle\b", "training season"),
]


# ============================================================================
# Processing
# ============================================================================

def fix_response(response: str) -> str:
    """Apply all fixes to a single explainer response."""
    fixed = response

    # 1. Full-response replacements (exact match)
    if fixed in ENGINE_FULL_REPLACEMENTS:
        return ENGINE_FULL_REPLACEMENTS[fixed]

    # 2. Engine substring replacements
    for pattern, replacement in ENGINE_SUBSTRING_REPLACEMENTS:
        fixed = re.sub(pattern, replacement, fixed)

    # 3. Jargon replacements
    for entry in JARGON_REPLACEMENTS:
        if len(entry) == 3:
            pattern, replacement, flags = entry
            fixed = re.sub(pattern, replacement, fixed, flags=flags)
        else:
            pattern, replacement = entry
            fixed = re.sub(pattern, replacement, fixed)

    return fixed


def process_file(filepath: Path) -> dict:
    """Process a single JSONL file, fixing all responses."""
    if not filepath.exists():
        print(f"  Skipping {filepath.name} (not found)")
        return {"total": 0, "engine_fixed": 0, "jargon_fixed": 0}

    # Read all examples
    examples = []
    with open(filepath) as f:
        for line in f:
            examples.append(json.loads(line))

    stats = {"total": len(examples), "engine_fixed": 0, "jargon_fixed": 0}

    # Fix each example
    for ex in examples:
        original = ex["messages"][-1]["content"]
        fixed = fix_response(original)

        if fixed != original:
            # Categorize the fix
            if "engine" in original.lower() or "outside my scope" in original.lower():
                stats["engine_fixed"] += 1
            else:
                stats["jargon_fixed"] += 1
            ex["messages"][-1]["content"] = fixed

    # Backup and write
    backup = filepath.with_suffix(".jsonl.bak")
    shutil.copy2(filepath, backup)
    print(f"  Backup: {backup.name}")

    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return stats


def verify_fixes(filepath: Path):
    """Verify no engine references or jargon remain."""
    remaining_engine = 0
    remaining_jargon = 0
    samples = []

    with open(filepath) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            resp = ex["messages"][-1]["content"]
            lower = resp.lower()

            if "engine" in lower:
                remaining_engine += 1
                if len(samples) < 3:
                    samples.append((i, "engine", resp[:80]))

            for jw in ["lactate threshold", "anaerobic capacity", "vo2max",
                        "periodization", "ftp", "supercompensation",
                        "mesocycle", "microcycle", "macrocycle"]:
                if jw in lower:
                    remaining_jargon += 1
                    if len(samples) < 3:
                        samples.append((i, jw, resp[:80]))
                    break

    return remaining_engine, remaining_jargon, samples


def main():
    print("=" * 60)
    print("  Explainer Training Data Fix — v5")
    print("=" * 60)

    for filepath in [TRAIN_FILE, VAL_FILE]:
        print(f"\nProcessing {filepath.name}...")
        stats = process_file(filepath)
        print(f"  Total examples:  {stats['total']}")
        print(f"  Engine fixed:    {stats['engine_fixed']}")
        print(f"  Jargon fixed:    {stats['jargon_fixed']}")

        # Verify
        re_engine, re_jargon, samples = verify_fixes(filepath)
        print(f"  Remaining engine refs: {re_engine}")
        print(f"  Remaining jargon:      {re_jargon}")
        if samples:
            print(f"  Samples of remaining issues:")
            for idx, issue, text in samples:
                print(f"    Line {idx} [{issue}]: {text}")

    print(f"\n{'=' * 60}")
    print("  Done. Backups saved as .jsonl.bak")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
