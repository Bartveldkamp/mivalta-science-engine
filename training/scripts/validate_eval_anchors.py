#!/usr/bin/env python3
"""
validate_eval_anchors.py — Verify training data covers all 42 eval cases.

For each eval case, checks that the training dataset contains at least one
example demonstrating the expected behavior (tier + intent + response_type).

For output-validation cases (I6 guardrails), verifies that NO training example
teaches the model to produce the banned output patterns.

Usage:
    python training/scripts/validate_eval_anchors.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "shared" / "eval_cases"
DATA_DIR = ROOT / "training" / "data"

TIER_KEYWORDS = {
    "monitor": "MODE: Monitor",
    "advisor": "MODE: Advisor",
    "coach": "MODE: Coach",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_cases() -> dict[str, list[dict]]:
    cases = {}
    for fp in sorted(EVAL_DIR.glob("*.json")):
        data = json.loads(fp.read_text())
        cases[data["category"]] = data["cases"]
    return cases


def parse_example(ex: dict) -> dict:
    """Extract tier, intent, response_type, tool, message from a training example."""
    msgs = ex["messages"]
    system_msg = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    user_msg = msgs[1]["content"] if len(msgs) > 1 and msgs[1]["role"] == "user" else ""
    assistant_msg = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""

    # Parse tier from system prompt
    tier = None
    for t, kw in TIER_KEYWORDS.items():
        if kw in system_msg:
            tier = t
            break

    # Parse LLMIntent JSON from assistant
    intent_data = {}
    try:
        intent_data = json.loads(assistant_msg)
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "tier": tier,
        "intent": intent_data.get("intent"),
        "response_type": intent_data.get("response_type"),
        "guardrail_triggered": intent_data.get("guardrail_triggered", False),
        "replan_request": intent_data.get("replan_request"),
        "tool_call": intent_data.get("tool_call"),
        "message": intent_data.get("message", ""),
        "user_msg": user_msg,
    }


def check_tier_compliance(cases: list[dict], examples: list[dict]) -> list[dict]:
    """Check training coverage for tier compliance eval cases."""
    results = []
    for case in cases:
        case_id = case["id"]
        tier = case["tier"]
        expected_intent = case.get("expected_intent")
        expected_rtype = case.get("expected_response_type")
        replan_must_be = case.get("replan_request_must_be")
        replan_type = case.get("replan_type")

        # Find matching examples
        matches = []
        for ex in examples:
            if ex["tier"] != tier:
                continue
            if expected_intent and ex["intent"] != expected_intent:
                continue
            if expected_rtype and ex["response_type"] != expected_rtype:
                continue
            if replan_must_be == "not null" and ex["replan_request"] is None:
                continue
            if replan_must_be is None and case.get("replan_request_must_be") is not None:
                if ex["replan_request"] is not None:
                    continue
            if replan_type:
                rr = ex["replan_request"]
                if not rr or rr.get("type") != replan_type:
                    continue
            matches.append(ex)

        results.append({
            "id": case_id,
            "category": "tier_compliance",
            "description": case.get("notes", ""),
            "covered": len(matches) > 0,
            "match_count": len(matches),
        })
    return results


def check_i6_guardrails(cases: list[dict], examples: list[dict]) -> list[dict]:
    """Check training coverage for I6 guardrail eval cases."""
    results = []

    # Banned tokens for output validation
    BANNED_TOKENS = [
        "transition matrix", "viterbi", "hmm", "hidden markov",
        "acwr", "ewma", "tss", "ctl", "atl", "tsb",
    ]

    for case in cases:
        case_id = case["id"]

        if case.get("output_validation"):
            # Output validation cases: verify no training example teaches bad patterns
            llm_output = case.get("llm_output", "").lower()
            expected_blocked = case.get("expected_blocked", True)
            block_reason = case.get("block_reason", "")

            if not expected_blocked:
                # I6-011: valid output, check we DO have examples like this
                matches = [
                    ex for ex in examples
                    if ex["tier"] == case["tier"]
                    and not ex["guardrail_triggered"]
                ]
                results.append({
                    "id": case_id,
                    "category": "i6_guardrails",
                    "description": case.get("notes", ""),
                    "type": "output_positive",
                    "covered": len(matches) > 0,
                    "match_count": len(matches),
                })
                continue

            # For blocked outputs, verify NO training example teaches this pattern
            violations = []
            for ex in examples:
                msg_lower = ex["message"].lower()

                if block_reason == "banned_token_internal_parameter":
                    for token in BANNED_TOKENS:
                        # Use word-boundary matching to avoid substring false positives
                        # e.g. "ctl" in "exactly" should NOT match
                        if re.search(r'\b' + re.escape(token) + r'\b', msg_lower):
                            violations.append({"token": token, "msg_snippet": msg_lower[:80]})
                elif block_reason == "prescription_in_output":
                    # Check if any non-guardrail example prescribes specific workouts
                    if not ex["guardrail_triggered"] and any(
                        p in msg_lower for p in ["5x400m", "4x800m", "3x1000m", "do 5 sets"]
                    ):
                        violations.append({"msg_snippet": msg_lower[:80]})
                elif block_reason == "modification_claim":
                    if not ex["guardrail_triggered"] and any(
                        p in msg_lower for p in ["i've changed your", "i changed your", "i modified"]
                    ):
                        violations.append({"msg_snippet": msg_lower[:80]})
                elif block_reason == "override_readiness":
                    if not ex["guardrail_triggered"] and any(
                        p in msg_lower for p in ["push through the fatigue", "push through fatigue", "ignore your fatigue"]
                    ):
                        violations.append({"msg_snippet": msg_lower[:80]})
                elif block_reason == "advisor_mode_prescription":
                    if ex["tier"] == "advisor" and not ex["guardrail_triggered"] and any(
                        p in msg_lower for p in ["you should do", "i recommend you", "try doing"]
                    ):
                        violations.append({"msg_snippet": msg_lower[:80]})
                elif block_reason == "monitor_mode_plan_discussion":
                    if ex["tier"] == "monitor" and not ex["guardrail_triggered"] and any(
                        p in msg_lower for p in ["planned session", "today's session", "your workout"]
                    ):
                        violations.append({"msg_snippet": msg_lower[:80]})

            results.append({
                "id": case_id,
                "category": "i6_guardrails",
                "description": case.get("notes", ""),
                "type": "output_negative",
                "covered": len(violations) == 0,
                "violation_count": len(violations),
                "violations": violations[:3] if violations else [],
            })
        else:
            # Input validation cases: check we have examples teaching the block
            tier = case["tier"]
            expected_intent = case.get("expected_intent", "blocked")

            matches = [
                ex for ex in examples
                if ex["tier"] == tier
                and ex["intent"] == expected_intent
                and ex["guardrail_triggered"]
            ]

            results.append({
                "id": case_id,
                "category": "i6_guardrails",
                "description": case.get("notes", ""),
                "type": "input",
                "covered": len(matches) > 0,
                "match_count": len(matches),
            })
    return results


def check_tool_dispatch(cases: list[dict], examples: list[dict]) -> list[dict]:
    """Check training coverage for tool dispatch eval cases."""
    results = []

    for case in cases:
        case_id = case["id"]
        tier = case["tier"]
        expected_tool = case.get("expected_tool")
        expected_rtype = case.get("expected_response_type")

        if expected_tool is None and expected_rtype:
            # No-tool cases (decline or pure QA)
            matches = [
                ex for ex in examples
                if ex["tier"] == tier
                and ex["response_type"] == expected_rtype
                and ex["tool_call"] is None
            ]
        elif expected_tool:
            # Tool dispatch cases
            matches = [
                ex for ex in examples
                if ex["tier"] == tier
                and ex["tool_call"] is not None
                and ex["tool_call"].get("tool") == expected_tool
            ]
        elif case.get("expected_clarification"):
            # Clarification cases — tool dispatch examples where model asks for more info
            matches = [
                ex for ex in examples
                if ex["tier"] == tier
                and ex["tool_call"] is not None
                and ex["tool_call"].get("tool") == "create_today_workout"
            ]
        else:
            matches = []

        results.append({
            "id": case_id,
            "category": "tool_dispatch",
            "description": case.get("notes", ""),
            "covered": len(matches) > 0,
            "match_count": len(matches) if "match_count" not in case else len(matches),
        })
    return results


def main():
    # Load data
    train = load_jsonl(DATA_DIR / "train_v3.jsonl")
    val = load_jsonl(DATA_DIR / "val_v3.jsonl")
    all_data = train + val
    print(f"Loaded {len(train)} train + {len(val)} val = {len(all_data)} total examples")

    # Parse all examples
    parsed = [parse_example(ex) for ex in all_data]
    print(f"Parsed {len(parsed)} examples")

    # Load eval cases
    eval_cases = load_eval_cases()
    total_cases = sum(len(c) for c in eval_cases.values())
    print(f"Loaded {total_cases} eval cases across {len(eval_cases)} categories\n")

    # Run checks
    all_results = []

    if "tier_compliance" in eval_cases:
        results = check_tier_compliance(eval_cases["tier_compliance"], parsed)
        all_results.extend(results)

    if "i6_guardrails" in eval_cases:
        results = check_i6_guardrails(eval_cases["i6_guardrails"], parsed)
        all_results.extend(results)

    if "tool_dispatch" in eval_cases:
        results = check_tool_dispatch(eval_cases["tool_dispatch"], parsed)
        all_results.extend(results)

    # Report
    covered = [r for r in all_results if r["covered"]]
    uncovered = [r for r in all_results if not r["covered"]]

    print("=" * 70)
    print(f"EVAL ANCHOR COVERAGE: {len(covered)}/{len(all_results)} cases covered")
    print("=" * 70)

    # Print by category
    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r["category"]].append(r)

    for cat, results in sorted(by_cat.items()):
        cat_covered = sum(1 for r in results if r["covered"])
        print(f"\n  {cat}: {cat_covered}/{len(results)}")
        for r in results:
            status = "PASS" if r["covered"] else "FAIL"
            detail = ""
            if "match_count" in r:
                detail = f" ({r['match_count']} matches)"
            elif "violation_count" in r:
                detail = f" ({r['violation_count']} violations)" if r["violation_count"] > 0 else ""
            print(f"    [{status}] {r['id']}: {r['description']}{detail}")

    if uncovered:
        print(f"\n{'=' * 70}")
        print(f"UNCOVERED CASES ({len(uncovered)}):")
        print("=" * 70)
        for r in uncovered:
            print(f"  {r['id']}: {r['description']}")
            if r.get("violations"):
                for v in r["violations"]:
                    print(f"    -> violation: {v}")

    # Write report JSON
    report = {
        "total_eval_cases": len(all_results),
        "covered": len(covered),
        "uncovered": len(uncovered),
        "coverage_pct": round(100 * len(covered) / len(all_results), 1),
        "results": all_results,
    }
    report_path = DATA_DIR / "eval_anchor_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Exit code
    if uncovered:
        print(f"\nWARNING: {len(uncovered)} eval cases not covered by training data")
        sys.exit(1)
    else:
        print("\nAll eval cases covered by training data!")
        sys.exit(0)


if __name__ == "__main__":
    main()
