#!/usr/bin/env python3
"""
Feedback Loop: Extract failures from evaluation → generate training fixes.

This script takes the JSON output from evaluate_smollm2.py and:
1. Extracts all failed prompts
2. Shows what went wrong (forbidden words, jargon, tone, pushback)
3. Generates corrected training examples in JSONL format
4. Merges fixes into the training dataset for retraining

Smart training strategy:
  evaluate → find failures → write correct responses → retrain → repeat

Usage:
    # Step 1: Run evaluation with --output to save results
    python evaluate_smollm2.py --hf-model ./models/josi-smollm2-merged-v2 \
        --prompts-file data/test_prompts_1000.json \
        --output data/eval_results.json

    # Step 2: Extract failures and generate fix templates
    python feedback_loop.py extract data/eval_results.json

    # Step 3: Review and edit the generated fixes (data/fixes_to_review.jsonl)
    #         Then approve them:
    python feedback_loop.py approve data/fixes_to_review.jsonl

    # Step 4: Merge approved fixes into training data
    python feedback_loop.py merge

    # Step 5: Retrain with the improved dataset
    python finetune_smollm2.py train --model 1.7b --train_data data/smollm2_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

JOSI_SYSTEM_PROMPT = (
    "You are Josi, a friendly and knowledgeable sports coaching assistant for MiValta. "
    "You communicate training decisions made by the coaching engine. "
    "Rules: Keep responses under 80 words. Be warm and conversational. "
    "Use simple language, not textbook explanations. Ask follow-up questions. "
    "Never invent training rules — only explain what the engine decided. "
    "Never mention algorithms, GATC, Viterbi, ACWR, or internal systems."
)

FORBIDDEN_WORDS = [
    "gatc", "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "acute:chronic", "acute chronic", "load ratio",
    "monotony index", "training monotony", "strain index",
    "exponentially weighted", "ewma", "tss", "ctl", "atl", "tsb",
    "impulse-response", "banister", "fitness-fatigue",
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "lactate threshold",
    "ftp", "threshold power", "anaerobic capacity",
]


def cmd_extract(eval_json_path: str):
    """Extract failures from evaluation results and generate fix templates."""
    with open(eval_json_path) as f:
        report = json.load(f)

    results = report.get("results", [])
    failures = [r for r in results if not r["passed"]]

    if not failures:
        print("No failures found! Model is clean.")
        return

    print(f"\nFound {len(failures)} failures out of {len(results)} prompts")
    print(f"Pass rate: {report['pass_rate']:.1f}%\n")

    # Group failures by reason
    by_reason = {}
    for r in failures:
        for reason in r["failure_reasons"]:
            key = reason.split(":")[0].strip()
            by_reason.setdefault(key, []).append(r)

    print("Failure breakdown:")
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"  {reason}: {len(items)}")

    # Generate fix templates
    fixes_path = DATA_DIR / "fixes_to_review.jsonl"
    count = 0

    with open(fixes_path, "w") as f:
        for r in failures:
            # Create a training example template with the correct format
            fix = {
                "messages": [
                    {"role": "system", "content": JOSI_SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": _generate_fix_response(r)},
                ],
                "_meta": {
                    "original_response": r["response"],
                    "failure_reasons": r["failure_reasons"],
                    "category": r["category"],
                    "action": "REVIEW",  # Change to APPROVE when ready
                },
            }
            f.write(json.dumps(fix, ensure_ascii=False) + "\n")
            count += 1

    print(f"\nGenerated {count} fix templates → {fixes_path}")
    print(f"\nNext steps:")
    print(f"  1. Open {fixes_path}")
    print(f"  2. Review each fix — edit the assistant response to be correct")
    print(f"  3. Change '_meta.action' from 'REVIEW' to 'APPROVE' for good ones")
    print(f"  4. Run: python feedback_loop.py approve {fixes_path}")


def _generate_fix_response(result: dict) -> str:
    """Generate a corrected response template based on failure reasons."""
    prompt = result["prompt"]
    original = result["response"]
    reasons = result["failure_reasons"]
    category = result["category"]

    # Start with a cleaned version of the original
    response = original

    # If forbidden words, strip them
    for reason in reasons:
        if reason.startswith("Forbidden"):
            words = reason.split(": ", 1)[1].split(", ")
            for w in words:
                response = response.replace(w, "[REMOVED]")

    # If too verbose, truncate at sentence boundary
    if any("verbose" in r.lower() for r in reasons):
        sentences = response.split(". ")
        short = []
        word_count = 0
        for s in sentences:
            word_count += len(s.split())
            if word_count > 70:
                break
            short.append(s)
        response = ". ".join(short)
        if not response.endswith(".") and not response.endswith("?"):
            response += "."

    # If no pushback on unrealistic goal, add it
    if any("pushback" in r.lower() for r in reasons):
        response = (
            f"[EDIT THIS] That's a really ambitious goal, and I want to be honest with you — "
            f"it may be too much too soon. {response}"
        )

    # Mark for review
    if "[REMOVED]" in response or "[EDIT THIS]" in response:
        response = f"[NEEDS EDITING] {response}"
    else:
        response = f"[REVIEW] {response}"

    return response


def cmd_approve(fixes_path: str):
    """Filter approved fixes and add to training data."""
    approved = []
    reviewed = 0
    skipped = 0

    with open(fixes_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            reviewed += 1
            meta = item.get("_meta", {})

            if meta.get("action") == "APPROVE":
                # Remove meta, keep only the training example
                clean = {"messages": item["messages"]}
                # Remove system message (prepare_training_data.py adds it)
                clean["messages"] = [
                    m for m in clean["messages"] if m["role"] != "system"
                ]
                approved.append(clean)
            else:
                skipped += 1

    if not approved:
        print(f"No approved fixes found in {fixes_path}")
        print(f"Mark fixes with '_meta.action': 'APPROVE' to include them")
        return

    # Write approved fixes
    approved_path = DATA_DIR / "gold_examples" / "gold_feedback_fixes.jsonl"
    with open(approved_path, "w") as f:
        for item in approved:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Approved: {len(approved)}/{reviewed} fixes")
    print(f"Skipped: {skipped}")
    print(f"Written: {approved_path}")
    print(f"\nNext: python feedback_loop.py merge")


def cmd_merge():
    """Regenerate training dataset with approved fixes included."""
    fixes_path = DATA_DIR / "gold_examples" / "gold_feedback_fixes.jsonl"

    if not fixes_path.exists():
        print(f"No fixes found at {fixes_path}")
        print(f"Run 'feedback_loop.py approve' first")
        return

    fix_count = sum(1 for line in open(fixes_path) if line.strip())
    print(f"Found {fix_count} approved fixes in {fixes_path}")

    # Check if prepare_training_data.py includes this file
    prep_script = Path(__file__).parent / "prepare_training_data.py"
    prep_content = prep_script.read_text()

    if "gold_feedback_fixes.jsonl" not in prep_content:
        print(f"\nWARNING: gold_feedback_fixes.jsonl is not in prepare_training_data.py")
        print(f"Add this line to the source_files list:")
        print(f'        ("gold_examples/gold_feedback_fixes.jsonl", "feedback_fixes"),')
        return

    print(f"\nRegenerating training data...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "prepare_training_data.py")],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return

    print("Training data regenerated with fixes included!")
    print("\nNext: retrain the model:")
    print("  python finetune_smollm2.py train --model 1.7b --train_data data/smollm2_train.jsonl")


def cmd_auto(eval_json_path: str):
    """Auto-approve responses that only failed on pushback (tone issues).

    For forbidden word / jargon failures, those need manual editing.
    """
    with open(eval_json_path) as f:
        report = json.load(f)

    results = report.get("results", [])
    failures = [r for r in results if not r["passed"]]

    auto_approved = 0
    needs_review = 0
    fixes = []

    for r in failures:
        reasons = r["failure_reasons"]

        # Check if it's ONLY a pushback issue (tone, not content)
        only_pushback = all("pushback" in reason.lower() for reason in reasons)

        if only_pushback:
            # Auto-generate a response with pushback added
            fix = {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": _auto_fix_pushback(r["prompt"])},
                ],
            }
            fixes.append(fix)
            auto_approved += 1
        else:
            needs_review += 1

    if fixes:
        auto_path = DATA_DIR / "gold_examples" / "gold_feedback_fixes.jsonl"
        # Append to existing fixes
        mode = "a" if auto_path.exists() else "w"
        with open(auto_path, mode) as f:
            for fix in fixes:
                f.write(json.dumps(fix, ensure_ascii=False) + "\n")

        print(f"Auto-approved {auto_approved} pushback fixes → {auto_path}")

    if needs_review:
        print(f"{needs_review} failures need manual review (forbidden words/jargon/tone)")
        print(f"Run: python feedback_loop.py extract {eval_json_path}")

    if not fixes and not needs_review:
        print("No failures to fix!")


def _auto_fix_pushback(prompt: str) -> str:
    """Generate a response that includes pushback for unrealistic goals."""
    # These are template responses that push back appropriately
    templates = [
        "I appreciate the ambition, but I want to be honest — that timeline is too aggressive and risks injury. Let's set a more realistic target that still challenges you. What if we aimed for a longer timeframe and built up safely? I'd rather see you succeed at a sustainable pace than get hurt rushing.",
        "That's a big goal, and I love the enthusiasm! But rushing it could lead to injury or burnout. Your body needs time to adapt. Let's find a challenging but achievable timeline that keeps you healthy and motivated. What does your training look like right now?",
        "I hear you, and that's an ambitious target. But being real with you — pushing that hard, that fast, significantly increases your injury risk. How about we adjust the timeline to something your body can handle? You'll still get there, just more safely.",
    ]
    import random
    return random.choice(templates)


def main():
    parser = argparse.ArgumentParser(description="Josi training feedback loop")
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract", help="Extract failures → fix templates")
    p_extract.add_argument("eval_json", help="Path to evaluation JSON output")

    p_approve = sub.add_parser("approve", help="Filter approved fixes")
    p_approve.add_argument("fixes_file", help="Path to fixes JSONL file")

    p_merge = sub.add_parser("merge", help="Regenerate training data with fixes")

    p_auto = sub.add_parser("auto", help="Auto-approve safe fixes (pushback only)")
    p_auto.add_argument("eval_json", help="Path to evaluation JSON output")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args.eval_json)
    elif args.command == "approve":
        cmd_approve(args.fixes_file)
    elif args.command == "merge":
        cmd_merge()
    elif args.command == "auto":
        cmd_auto(args.eval_json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
