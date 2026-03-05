#!/usr/bin/env python3
"""
Grounding Discipline Validator

Validates that coach training examples stay grounded in CONTEXT data.
The LLM (Josi) is the messenger — GATC + Viterbi provide the numbers.
Coach responses must NEVER invent zones, paces, power, HR, or durations
that aren't present in the CONTEXT or [KNOWLEDGE] block.

Usage:
    # Validate a single JSONL file
    python validate_grounding.py training/data/claude_generated_coach.jsonl

    # Validate all training data
    python validate_grounding.py --all

    # Strict mode (fail on any invented number)
    python validate_grounding.py --strict training/data/claude_generated_coach.jsonl

    # Show details for each violation
    python validate_grounding.py -v training/data/claude_generated_coach.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"


# ---------------------------------------------------------------------------
# Grounding rules
# ---------------------------------------------------------------------------

# Patterns that indicate the coach is inventing specific numbers
INVENTED_NUMBER_PATTERNS = [
    # Specific HR values (e.g., "aim for 145 bpm", "keep HR at 160")
    (r'\b(?:aim|keep|target|hold|stay|maintain)\b.{0,20}\b\d{2,3}\s*(?:bpm|beats|heartbeats|HR)',
     "Invented HR target"),
    (r'\bHR\s*(?:of|at|around|between|should be)\s*\d{2,3}',
     "Invented HR value"),
    (r'\b\d{2,3}\s*(?:to|-)\s*\d{2,3}\s*bpm',
     "Invented HR range"),

    # Specific pace values (e.g., "run at 5:30/km", "aim for 4:15 pace")
    (r'\b(?:run|aim|go|hold|target)\b.{0,20}\b\d{1}:\d{2}\s*/\s*(?:km|mi)',
     "Invented pace target"),
    (r'\b\d{1}:\d{2}\s*(?:per|/)\s*(?:km|mi|kilometer|mile)',
     "Invented pace value"),
    (r'\bpace\s*(?:of|at|around)\s*\d{1}:\d{2}',
     "Invented pace value"),

    # Specific power values (e.g., "ride at 250W", "target 200 watts")
    (r'\b(?:ride|aim|hold|target|do)\b.{0,20}\b\d{2,3}\s*(?:W|watts|watt)\b',
     "Invented power target"),
    (r'\b\d{2,3}\s*(?:to|-)\s*\d{2,3}\s*(?:W|watts|watt)\b',
     "Invented power range"),

    # Specific distance prescriptions (e.g., "run 8km", "do 5 miles")
    (r'\b(?:run|do|complete)\s+\d+\s*(?:km|mi|kilometers|miles)\b',
     "Invented distance prescription"),

    # Specific rep schemes not in context (e.g., "do 5x1000m")
    (r'\bdo\s+\d+\s*x\s*\d+',
     "Invented rep scheme"),

    # Specific percentage prescriptions (e.g., "at 80% max HR")
    (r'\bat\s+\d{2,3}\s*%\s*(?:max|of|FTP|threshold)',
     "Invented percentage target"),

    # Specific weekly volume prescriptions (e.g., "run 50km this week")
    (r'\b(?:run|ride|do|aim for)\s+\d{2,3}\s*(?:km|mi|hours?)\s*(?:this|per|a)\s*week',
     "Invented weekly volume"),
]

# Patterns that indicate the coach is prescribing (not explaining)
PRESCRIPTION_PATTERNS = [
    (r'\byou\s+should\s+(?:run|ride|do|train|go)\s+(?:at|for)\s+\d',
     "Prescribing with specific numbers"),
    (r'\b(?:I recommend|I suggest|try doing|you need to do)\s+\d+\s*x',
     "Prescribing workout structure"),
    (r'\b(?:your workout|your session)\s+(?:should be|is|will be)\s*:?\s*\d+\s*x',
     "Prescribing workout details"),
    (r'\bhere\'?s?\s+(?:your|the|a)\s+(?:workout|plan)\s*:',
     "Prescribing a workout"),
]

# Patterns indicating proper grounding (good behaviors to track)
GROUNDING_PATTERNS = [
    (r'\bcheck\s+(?:your|the)\s+(?:app|zone|profile|settings|overview)',
     "Defers to app"),
    (r'\b(?:the\s+)?engine\s+(?:will|has|knows|designed|set)',
     "Credits engine"),
    (r'\byour\s+(?:personal|specific|individual)\s+(?:zones?|targets?|numbers?)',
     "Acknowledges personal data"),
    (r'\b(?:zone\s+overview|zone\s+settings|profile|weekly\s+overview)',
     "References app feature"),
    (r'\b(?:depends on|based on|specific to)\s+your',
     "Acknowledges individuality"),
]

# System leak patterns (coach mentioning internals)
SYSTEM_LEAK_PATTERNS = [
    (r'\b(?:GATC|Viterbi|HMM|ACWR|EWMA|TSS|CTL|ATL|TSB)\b',
     "System name leak"),
    (r'\b(?:knowledge\s+card|knowledge\s+block|interpreter|GATCRequest)\b',
     "Internal reference leak"),
    (r'\b(?:the\s+algorithm|my\s+programming|the\s+system\s+(?:says|decided|classified))',
     "System mechanism leak"),
]


def extract_context_numbers(user_content: str) -> set[str]:
    """Extract all numbers/metrics that appear in the CONTEXT and KNOWLEDGE blocks."""
    numbers = set()

    # Extract zone references from context
    zone_matches = re.findall(r'Z\d', user_content)
    numbers.update(zone_matches)

    # Extract duration from context (e.g., "60min", "45min")
    duration_matches = re.findall(r'(\d+)\s*min', user_content)
    numbers.update(duration_matches)

    # Extract HR values from context
    hr_matches = re.findall(r'(\d{2,3})\s*(?:bpm|beats)', user_content)
    numbers.update(hr_matches)

    # Extract pace values from context
    pace_matches = re.findall(r'(\d:\d{2})\s*/\s*(?:km|mi)', user_content)
    numbers.update(pace_matches)

    # Extract percentages from context
    pct_matches = re.findall(r'(\d{2,3})%', user_content)
    numbers.update(pct_matches)

    return numbers


def validate_coach_grounding(
    user_content: str,
    coach_response: str,
    strict: bool = False,
) -> list[dict]:
    """
    Validate that a coach response stays grounded in the provided context.

    Returns a list of violations, each with:
        - type: "invented_number" | "prescription" | "system_leak"
        - pattern: what was matched
        - match: the actual text matched
        - severity: "error" | "warning"
    """
    violations = []
    response_lower = coach_response.lower()

    # Check for invented numbers
    context_numbers = extract_context_numbers(user_content)
    for pattern, description in INVENTED_NUMBER_PATTERNS:
        matches = re.finditer(pattern, coach_response, re.IGNORECASE)
        for match in matches:
            matched_text = match.group()
            # Check if the number appears in context (allowed)
            numbers_in_match = re.findall(r'\d+', matched_text)
            all_in_context = all(n in context_numbers for n in numbers_in_match)

            if not all_in_context:
                violations.append({
                    "type": "invented_number",
                    "pattern": description,
                    "match": matched_text,
                    "severity": "error",
                })

    # Check for prescription patterns
    for pattern, description in PRESCRIPTION_PATTERNS:
        matches = re.finditer(pattern, coach_response, re.IGNORECASE)
        for match in matches:
            violations.append({
                "type": "prescription",
                "pattern": description,
                "match": match.group(),
                "severity": "error" if strict else "warning",
            })

    # Check for system leaks
    for pattern, description in SYSTEM_LEAK_PATTERNS:
        matches = re.finditer(pattern, coach_response, re.IGNORECASE)
        for match in matches:
            violations.append({
                "type": "system_leak",
                "pattern": description,
                "match": match.group(),
                "severity": "error",
            })

    return violations


def score_grounding_quality(
    user_content: str,
    coach_response: str,
) -> dict:
    """
    Score how well-grounded a coach response is.

    Returns:
        - grounding_score: 0.0-1.0 (higher = better grounded)
        - positive_signals: list of good grounding behaviors found
        - violations: list of grounding violations
    """
    violations = validate_coach_grounding(user_content, coach_response, strict=True)

    positive_signals = []
    for pattern, description in GROUNDING_PATTERNS:
        if re.search(pattern, coach_response, re.IGNORECASE):
            positive_signals.append(description)

    # Score: start at 1.0, deduct for violations, bonus for positive signals
    score = 1.0
    for v in violations:
        if v["severity"] == "error":
            score -= 0.3
        else:
            score -= 0.1
    # Bonus for positive grounding signals (up to 0.2)
    score += min(len(positive_signals) * 0.05, 0.2)
    score = max(0.0, min(1.0, score))

    return {
        "grounding_score": round(score, 2),
        "positive_signals": positive_signals,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_jsonl_file(
    filepath: Path,
    strict: bool = False,
    verbose: bool = False,
) -> dict:
    """Validate all examples in a JSONL file."""
    stats = {
        "total": 0,
        "clean": 0,
        "with_violations": 0,
        "errors": 0,
        "warnings": 0,
        "violation_types": {},
        "avg_grounding_score": 0.0,
        "violations_detail": [],
    }

    scores = []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                stats["errors"] += 1
                continue

            messages = example.get("messages", [])
            if len(messages) < 2:
                continue

            # Find user and assistant messages
            user_content = ""
            coach_response = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_content = msg["content"]
                elif msg["role"] == "assistant":
                    coach_response = msg["content"]

            if not coach_response:
                continue

            stats["total"] += 1

            # Validate grounding
            result = score_grounding_quality(user_content, coach_response)
            scores.append(result["grounding_score"])

            violations = validate_coach_grounding(
                user_content, coach_response, strict=strict
            )

            if violations:
                stats["with_violations"] += 1
                for v in violations:
                    vtype = v["type"]
                    stats["violation_types"][vtype] = stats["violation_types"].get(vtype, 0) + 1
                    if v["severity"] == "error":
                        stats["errors"] += 1
                    else:
                        stats["warnings"] += 1

                if verbose:
                    stats["violations_detail"].append({
                        "line": line_num,
                        "athlete": user_content[:80] + "..." if len(user_content) > 80 else user_content,
                        "violations": violations,
                    })
            else:
                stats["clean"] += 1

    if scores:
        stats["avg_grounding_score"] = round(sum(scores) / len(scores), 3)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate grounding discipline in coach training examples"
    )
    parser.add_argument("file", nargs="?", type=Path,
                        help="JSONL file to validate")
    parser.add_argument("--all", action="store_true",
                        help="Validate all training data files")
    parser.add_argument("--strict", action="store_true",
                        help="Treat all violations as errors")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show details for each violation")
    parser.add_argument("--score-only", action="store_true",
                        help="Just show grounding scores, no details")

    args = parser.parse_args()

    if not args.file and not args.all:
        parser.print_help()
        print("\nQuick start:")
        print("  python validate_grounding.py training/data/claude_generated_coach.jsonl")
        print("  python validate_grounding.py --all")
        sys.exit(0)

    files = []
    if args.all:
        # Validate all JSONL files in data/
        files.extend(DATA_DIR.glob("*.jsonl"))
        files.extend((DATA_DIR / "gold_examples").glob("*.jsonl"))
    elif args.file:
        files.append(args.file)

    total_stats = {
        "files": 0, "total": 0, "clean": 0,
        "with_violations": 0, "errors": 0, "warnings": 0,
    }
    all_scores = []

    for filepath in sorted(files):
        if not filepath.exists():
            print(f"  SKIP: {filepath} (not found)")
            continue

        stats = validate_jsonl_file(filepath, strict=args.strict, verbose=args.verbose)

        if stats["total"] == 0:
            continue

        total_stats["files"] += 1
        total_stats["total"] += stats["total"]
        total_stats["clean"] += stats["clean"]
        total_stats["with_violations"] += stats["with_violations"]
        total_stats["errors"] += stats["errors"]
        total_stats["warnings"] += stats["warnings"]
        all_scores.append(stats["avg_grounding_score"])

        # Print per-file summary
        status = "PASS" if stats["with_violations"] == 0 else "WARN" if stats["errors"] == 0 else "FAIL"
        print(f"  [{status}] {filepath.name}: "
              f"{stats['clean']}/{stats['total']} clean  "
              f"score={stats['avg_grounding_score']}  "
              f"errors={stats['errors']} warnings={stats['warnings']}")

        if stats["violation_types"]:
            for vtype, count in sorted(stats["violation_types"].items()):
                print(f"         {vtype}: {count}")

        if args.verbose and stats["violations_detail"]:
            for detail in stats["violations_detail"]:
                print(f"         Line {detail['line']}: {detail['athlete']}")
                for v in detail["violations"]:
                    print(f"           [{v['severity'].upper()}] {v['pattern']}: \"{v['match']}\"")

    # Print summary
    if total_stats["files"] > 1:
        avg_score = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0
        print(f"\n{'='*60}")
        print(f"  GROUNDING VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Files: {total_stats['files']}")
        print(f"  Examples: {total_stats['total']}")
        print(f"  Clean: {total_stats['clean']} ({100*total_stats['clean']/max(total_stats['total'],1):.0f}%)")
        print(f"  With violations: {total_stats['with_violations']}")
        print(f"  Errors: {total_stats['errors']}")
        print(f"  Warnings: {total_stats['warnings']}")
        print(f"  Avg grounding score: {avg_score}")

    # Exit code
    if args.strict and total_stats["errors"] > 0:
        sys.exit(1)
    elif total_stats["with_violations"] > total_stats["total"] * 0.1:
        # Fail if >10% of examples have violations
        sys.exit(1)


if __name__ == "__main__":
    main()
