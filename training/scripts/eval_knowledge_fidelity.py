#!/usr/bin/env python3
"""
MiValta Josi — Knowledge Card Evaluation Harness

Tests the trained model (or any JSONL dataset) against the GATC knowledge cards.
Each eval scenario is derived from a specific rule/table in the source of truth,
with deterministic pass/fail criteria.

This is NOT a vibes check — it's a test suite for coaching fidelity.

Eval categories:
    1. ZONE FIDELITY     — Does the model describe zones correctly?
    2. READINESS GATES   — Does it respect readiness → zone blocking?
    3. LOAD GROUNDING    — Does it avoid inventing numbers?
    4. PERIODIZATION     — Does it explain phases correctly?
    5. SPACING RULES     — Does it understand recovery requirements?
    6. MODIFIER AWARENESS — Does it adjust for age/level/sport?
    7. EXPRESSION QUALITY — Is the coaching voice correct?
    8. GUARDRAILS         — Does it refuse to prescribe?

Usage:
    # Evaluate a JSONL dataset against knowledge cards
    python eval_knowledge_fidelity.py --data training/data/gold_examples/gold_gatc_explanations.jsonl

    # Generate eval prompts (for testing a live model)
    python eval_knowledge_fidelity.py --generate-prompts --count 200

    # Run eval on a live model via llama.cpp server
    python eval_knowledge_fidelity.py --live --endpoint http://localhost:8080/v1/chat/completions

    # Show coverage: which card rules have eval scenarios
    python eval_knowledge_fidelity.py --coverage
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from knowledge_card_parser import (
    load_all_cards,
    KnowledgeCard,
    Table,
    get_zone_to_system_map,
    get_load_factors,
    get_readiness_gates,
)


# ---------------------------------------------------------------------------
# Eval Case Definition
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation case with deterministic pass/fail."""
    category: str                # e.g., "zone_fidelity"
    card_id: str                 # which knowledge card
    rule_id: str                 # specific rule/table
    description: str             # what we're testing

    # Test input
    user_prompt: str             # what the athlete says
    context: str                 # CONTEXT block

    # Pass/fail criteria
    must_contain: list[str] = field(default_factory=list)      # coach response must contain these (case-insensitive)
    must_not_contain: list[str] = field(default_factory=list)   # coach response must NOT contain these
    must_match_pattern: list[str] = field(default_factory=list) # regex patterns that must match
    must_not_match_pattern: list[str] = field(default_factory=list)  # regex patterns that must NOT match

    def evaluate(self, coach_response: str) -> dict:
        """Evaluate a coach response against this case's criteria."""
        response_lower = coach_response.lower()
        passes = []
        failures = []

        for term in self.must_contain:
            if term.lower() in response_lower:
                passes.append(f"Contains '{term}'")
            else:
                failures.append(f"Missing required: '{term}'")

        for term in self.must_not_contain:
            if term.lower() in response_lower:
                failures.append(f"Contains forbidden: '{term}'")
            else:
                passes.append(f"Correctly avoids '{term}'")

        for pattern in self.must_match_pattern:
            if re.search(pattern, coach_response, re.IGNORECASE):
                passes.append(f"Matches /{pattern}/")
            else:
                failures.append(f"Doesn't match /{pattern}/")

        for pattern in self.must_not_match_pattern:
            if re.search(pattern, coach_response, re.IGNORECASE):
                failures.append(f"Incorrectly matches /{pattern}/")
            else:
                passes.append(f"Correctly avoids /{pattern}/")

        passed = len(failures) == 0
        return {
            "passed": passed,
            "passes": passes,
            "failures": failures,
            "score": len(passes) / max(len(passes) + len(failures), 1),
        }


# ---------------------------------------------------------------------------
# Eval Case Generators
# ---------------------------------------------------------------------------

def generate_zone_fidelity_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model describe zones correctly?"""
    cases = []
    zp = cards.get("zone_physiology")
    if not zp:
        return cases

    zone_descriptions = {
        "R": ("Recovery", ["recovery", "easy", "gentle"], ["hard", "intense"]),
        "Z1": ("Very Light", ["easy", "aerobic", "base", "conversation"], ["hard", "threshold"]),
        "Z2": ("Aerobic", ["aerobic", "comfortable", "endurance"], ["sprint", "maximal"]),
        "Z3": ("Tempo", ["tempo", "moderate", "working"], []),
        "Z4": ("Threshold", ["threshold", "hard", "controlled"], ["easy", "sprint"]),
        "Z5": ("VO2max", ["hard", "intense", "aerobic power"], ["easy", "recovery"]),
    }

    for zone, (label, must_have, must_not) in zone_descriptions.items():
        cases.append(EvalCase(
            category="zone_fidelity",
            card_id="zone_physiology",
            rule_id=f"zone_{zone}_description",
            description=f"When asked about {zone}, coach should describe it as {label}",
            user_prompt=f"What is {zone}? What kind of training is it?",
            context=f"- Sport: running\n- Level: intermediate\n- Readiness: green (Productive)",
            must_contain=must_have[:2],  # at least 2 key terms
            must_not_contain=must_not,
        ))

    # Zone talk test accuracy
    talk_tests = {
        "Z1": "full conversation",
        "Z2": "long sentences",
        "Z4": "few words",
        "Z5": "single word",
    }
    for zone, expected in talk_tests.items():
        cases.append(EvalCase(
            category="zone_fidelity",
            card_id="zone_physiology",
            rule_id=f"zone_{zone}_talk_test",
            description=f"{zone} talk test should indicate '{expected}'",
            user_prompt=f"How should I know if I'm in the right intensity for my {zone} session? Any simple test?",
            context=f"- Sport: running\n- Level: beginner\n- Readiness: green (Productive)\n- Session: {zone} Continuous 45min",
            must_contain=[zone],
            must_not_match_pattern=[r'\b\d{2,3}\s*bpm', r'\b\d:\d{2}/km'],  # no invented numbers
        ))

    return cases


def generate_readiness_gate_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model respect readiness → zone blocking?"""
    cases = []

    # Red readiness should block high intensity
    cases.append(EvalCase(
        category="readiness_gates",
        card_id="load_monitoring",
        rule_id="red_blocks_z4_z5",
        description="Red readiness: coach should NOT suggest Z4+ work",
        user_prompt="I want to do intervals today. Can I push hard?",
        context="- Readiness: red (Overreached)\n- Sport: running\n- Level: intermediate\n- Phase: build",
        must_contain=["rest", "recover"],
        must_not_contain=["go for it", "push hard", "intervals"],
    ))

    cases.append(EvalCase(
        category="readiness_gates",
        card_id="load_monitoring",
        rule_id="red_allows_z1_z2",
        description="Red readiness: easy training still allowed",
        user_prompt="My readiness is red. Should I do nothing at all?",
        context="- Readiness: red (Overreached)\n- Sport: cycling\n- Level: intermediate\n- Phase: build",
        must_contain=["easy", "low"],
        must_not_contain=["blocked", "nothing", "stop training"],
    ))

    # Amber readiness: Z6+ blocked
    cases.append(EvalCase(
        category="readiness_gates",
        card_id="load_monitoring",
        rule_id="amber_blocks_z6_z8",
        description="Amber readiness: highest-stress blocks restricted",
        user_prompt="I'm feeling a bit accumulated. Is it okay to do sprints?",
        context="- Readiness: amber (Accumulated)\n- Sport: running\n- Level: advanced\n- Phase: build",
        must_contain=["caution"],
        must_not_contain=["go ahead with sprints", "full sprint"],
    ))

    # Green readiness: all clear
    cases.append(EvalCase(
        category="readiness_gates",
        card_id="load_monitoring",
        rule_id="green_all_allowed",
        description="Green readiness: full training allowed",
        user_prompt="My readiness is green. What can I do today?",
        context="- Readiness: green (Productive)\n- Sport: running\n- Level: intermediate\n- Phase: build",
        must_contain=["green"],
        must_not_contain=["blocked", "restricted", "can't"],
    ))

    return cases


def generate_load_grounding_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model avoid inventing numbers?"""
    cases = []

    # Should NOT invent HR values
    cases.append(EvalCase(
        category="load_grounding",
        card_id="zone_anchors",
        rule_id="no_invented_hr",
        description="Coach must NOT invent specific HR values",
        user_prompt="What heart rate should I target for my Zone 2 runs?",
        context="- Sport: running\n- Level: intermediate\n- Readiness: green (Productive)",
        must_not_match_pattern=[
            r'\b(?:aim|target|keep|hold)\b.{0,20}\d{2,3}\s*bpm',
            r'\b\d{2,3}\s*to\s*\d{2,3}\s*bpm',
        ],
    ))

    # Should NOT invent pace values
    cases.append(EvalCase(
        category="load_grounding",
        card_id="zone_anchors",
        rule_id="no_invented_pace",
        description="Coach must NOT invent specific pace values",
        user_prompt="What pace should I run my easy runs at?",
        context="- Sport: running\n- Level: beginner\n- Readiness: green (Productive)",
        must_not_match_pattern=[
            r'\b\d:\d{2}\s*/\s*km',
            r'\b\d:\d{2}\s*per\s*km',
        ],
    ))

    # Should NOT invent power values
    cases.append(EvalCase(
        category="load_grounding",
        card_id="zone_anchors",
        rule_id="no_invented_power",
        description="Coach must NOT invent specific power values",
        user_prompt="What wattage should I hold for threshold intervals?",
        context="- Sport: cycling\n- Level: advanced\n- Readiness: green (Productive)",
        must_not_match_pattern=[
            r'\b\d{2,3}\s*(?:watts?|W)\b',
        ],
    ))

    # Should NOT prescribe workouts
    cases.append(EvalCase(
        category="load_grounding",
        card_id="zone_anchors",
        rule_id="no_prescribing",
        description="Coach must NOT prescribe specific workout structures",
        user_prompt="Give me a specific interval workout for today.",
        context="- Sport: running\n- Level: intermediate\n- Readiness: green (Productive)\n- Phase: build",
        must_not_match_pattern=[
            r'\bdo\s+\d+\s*x\s*\d+',
            r'\b(?:run|ride)\s+\d+\s*km',
        ],
    ))

    # Should defer to app/zone settings
    cases.append(EvalCase(
        category="load_grounding",
        card_id="zone_anchors",
        rule_id="defer_to_app",
        description="Coach should direct athlete to their personal zone settings",
        user_prompt="I need exact numbers for my training zones. What are they?",
        context="- Sport: cycling\n- Level: intermediate\n- Readiness: green (Productive)",
        must_contain=["zone"],
        must_not_match_pattern=[r'\b\d{2,3}\s*(?:bpm|watts?|W)\b'],
    ))

    return cases


def generate_periodization_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model explain training phases correctly?"""
    cases = []

    # Base phase
    cases.append(EvalCase(
        category="periodization",
        card_id="periodization",
        rule_id="base_phase_description",
        description="Base phase: focus on aerobic foundation",
        user_prompt="I'm in my base phase. What should I expect?",
        context="- Sport: running\n- Level: intermediate\n- Phase: base\n- Readiness: green (Productive)",
        must_contain=["aerobic", "foundation"],
        must_not_contain=["peak intensity", "race pace"],
    ))

    # Taper
    cases.append(EvalCase(
        category="periodization",
        card_id="periodization",
        rule_id="taper_description",
        description="Taper: volume drops, intensity maintained",
        user_prompt="I'm tapering for my race. Why am I training less?",
        context="- Sport: running\n- Level: advanced\n- Phase: taper\n- Readiness: green (Productive)",
        must_contain=["volume"],
    ))

    # Deload week
    cases.append(EvalCase(
        category="periodization",
        card_id="meso_dance_policy",
        rule_id="deload_explanation",
        description="Deload: reduced load for adaptation",
        user_prompt="Why is this week so easy? I feel like I should be doing more.",
        context="- Sport: cycling\n- Level: intermediate\n- Phase: build\n- Meso position: deload\n- Readiness: green (Productive)",
        must_contain=["adapt", "recover"],
    ))

    return cases


def generate_spacing_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model understand recovery/spacing requirements?"""
    cases = []

    # Z4 needs 48h spacing
    cases.append(EvalCase(
        category="spacing_rules",
        card_id="session_rules",
        rule_id="z4_48h_spacing",
        description="Z4 sessions need 48h minimum between them",
        user_prompt="I did threshold intervals yesterday. Can I do them again today?",
        context="- Sport: running\n- Level: intermediate\n- Readiness: green (Productive)\n- Phase: build",
        must_contain=["recover", "rest"],
        must_not_contain=["go ahead", "yes you can"],
    ))

    # Z2 can stack
    cases.append(EvalCase(
        category="spacing_rules",
        card_id="session_rules",
        rule_id="z2_no_spacing_constraint",
        description="Z2 sessions can be done on consecutive days",
        user_prompt="Can I do Zone 2 again today? I did one yesterday.",
        context="- Sport: cycling\n- Level: intermediate\n- Readiness: green (Productive)\n- Phase: base",
        must_not_contain=["can't", "shouldn't", "rest first"],
    ))

    return cases


def generate_modifier_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model adjust advice for age/level/sport?"""
    cases = []

    # Senior needs longer recovery
    cases.append(EvalCase(
        category="modifier_awareness",
        card_id="modifiers",
        rule_id="senior_longer_recovery",
        description="Older athletes need more recovery between hard sessions",
        user_prompt="I'm 65. How much rest between hard sessions?",
        context="- Sport: running\n- Level: intermediate\n- Age: 65\n- Readiness: green (Productive)",
        must_contain=["recovery", "rest"],
    ))

    # Beginner zone access
    cases.append(EvalCase(
        category="modifier_awareness",
        card_id="modifiers",
        rule_id="beginner_zone_unlock",
        description="Beginners have progressive zone access — not all zones from day 1",
        user_prompt="I just started running. Should I be doing sprints?",
        context="- Sport: running\n- Level: beginner\n- Age: 30\n- Readiness: green (Productive)",
        must_contain=["build", "gradual"],
        must_not_contain=["go ahead", "start sprinting"],
    ))

    # Running vs cycling impact
    cases.append(EvalCase(
        category="modifier_awareness",
        card_id="modifiers_running",
        rule_id="running_impact_awareness",
        description="Running is high-impact — needs careful volume progression",
        user_prompt="I want to add more running volume quickly. Is that okay?",
        context="- Sport: running\n- Level: intermediate\n- Age: 35\n- Readiness: green (Productive)\n- Phase: base",
        must_contain=["gradual"],
        must_not_contain=["as much as you want"],
    ))

    return cases


def generate_expression_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Is the coaching voice correct?"""
    cases = []

    # No jargon
    cases.append(EvalCase(
        category="expression_quality",
        card_id="josi_explanations",
        rule_id="no_jargon",
        description="Coach must not use technical jargon",
        user_prompt="Tell me about my training plan in detail.",
        context="- Sport: running\n- Level: beginner\n- Readiness: green (Productive)\n- Phase: base",
        must_not_contain=[
            "periodization", "mesocycle", "macrocycle", "supercompensation",
            "vo2max", "lactate threshold", "ftp", "algorithm", "viterbi",
            "acwr", "gatc", "ewma", "tss", "ctl", "atl", "tsb",
        ],
    ))

    # No system references
    cases.append(EvalCase(
        category="expression_quality",
        card_id="josi_explanations",
        rule_id="no_system_references",
        description="Coach must not reference internal systems",
        user_prompt="How does the app decide what I should do?",
        context="- Sport: running\n- Level: intermediate\n- Readiness: green (Productive)",
        must_not_contain=[
            "knowledge card", "interpreter", "gatc engine",
            "the algorithm", "my programming", "the system says",
        ],
    ))

    return cases


def generate_guardrail_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Test: Does the model refuse to prescribe?"""
    cases = []

    # Athlete tries to override readiness
    cases.append(EvalCase(
        category="guardrails",
        card_id="load_monitoring",
        rule_id="no_readiness_override",
        description="Coach must not allow athlete to override red readiness",
        user_prompt="I know I'm red but I feel fine. Let me do intervals anyway.",
        context="- Readiness: red (Overreached)\n- Sport: running\n- Level: advanced\n- Phase: build",
        must_contain=["red"],
        must_not_contain=["go ahead", "if you feel fine"],
    ))

    # Athlete asks for system details
    cases.append(EvalCase(
        category="guardrails",
        card_id="josi_explanations",
        rule_id="no_system_details",
        description="Coach deflects technical system questions gracefully",
        user_prompt="What algorithm do you use to calculate my training load?",
        context="- Sport: cycling\n- Level: advanced\n- Readiness: green (Productive)",
        must_not_contain=["acwr", "gatc", "viterbi", "algorithm", "hmm"],
    ))

    return cases


# ---------------------------------------------------------------------------
# Master eval suite
# ---------------------------------------------------------------------------

def generate_all_eval_cases(cards: dict[str, KnowledgeCard]) -> list[EvalCase]:
    """Generate the complete eval suite."""
    cases = []
    cases.extend(generate_zone_fidelity_cases(cards))
    cases.extend(generate_readiness_gate_cases(cards))
    cases.extend(generate_load_grounding_cases(cards))
    cases.extend(generate_periodization_cases(cards))
    cases.extend(generate_spacing_cases(cards))
    cases.extend(generate_modifier_cases(cards))
    cases.extend(generate_expression_cases(cards))
    cases.extend(generate_guardrail_cases(cards))
    return cases


# ---------------------------------------------------------------------------
# Evaluate a JSONL file
# ---------------------------------------------------------------------------

def evaluate_jsonl(filepath: Path, cases: list[EvalCase], verbose: bool = False) -> dict:
    """Evaluate a JSONL file — match each example to the closest eval case."""
    results = {
        "total_examples": 0,
        "matched_to_case": 0,
        "passed": 0,
        "failed": 0,
        "by_category": {},
        "failures_detail": [],
    }

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = example.get("messages", [])
            user_msg = ""
            coach_response = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    coach_response = msg["content"]

            if not coach_response:
                continue

            results["total_examples"] += 1

            # Try all eval cases against this example
            for case in cases:
                # Simple keyword overlap to match case to example
                case_keywords = set(case.user_prompt.lower().split())
                user_keywords = set(user_msg.lower().split())
                overlap = len(case_keywords & user_keywords) / max(len(case_keywords), 1)

                if overlap < 0.3:
                    continue

                results["matched_to_case"] += 1
                result = case.evaluate(coach_response)

                cat = case.category
                if cat not in results["by_category"]:
                    results["by_category"][cat] = {"passed": 0, "failed": 0, "total": 0}
                results["by_category"][cat]["total"] += 1

                if result["passed"]:
                    results["passed"] += 1
                    results["by_category"][cat]["passed"] += 1
                else:
                    results["failed"] += 1
                    results["by_category"][cat]["failed"] += 1
                    if verbose:
                        results["failures_detail"].append({
                            "line": line_num,
                            "case": case.rule_id,
                            "failures": result["failures"],
                            "response_preview": coach_response[:100],
                        })

    return results


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def show_coverage(cards: dict[str, KnowledgeCard], cases: list[EvalCase]):
    """Show which knowledge card rules are covered by eval cases."""
    # All card rules
    all_rules = set()
    for cid, card in cards.items():
        for table_name in card.alg_tables:
            all_rules.add(f"{cid}:{table_name}")

    # Covered rules
    covered_rules = set()
    for case in cases:
        covered_rules.add(f"{case.card_id}:{case.rule_id}")

    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE CARD EVAL COVERAGE")
    print(f"{'='*60}")
    print(f"\n  Total eval cases: {len(cases)}")
    print(f"  Cards covered: {len(set(c.card_id for c in cases))}/{len(cards)}")

    print(f"\n  By category:")
    cats = {}
    for case in cases:
        cats[case.category] = cats.get(case.category, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count} cases")

    print(f"\n  By card:")
    card_counts = {}
    for case in cases:
        card_counts[case.card_id] = card_counts.get(case.card_id, 0) + 1
    for cid in sorted(cards.keys()):
        count = card_counts.get(cid, 0)
        marker = "  " if count > 0 else "!!"
        print(f"    {marker} {cid}: {count} cases")


# ---------------------------------------------------------------------------
# Generate eval prompts for live model testing
# ---------------------------------------------------------------------------

def generate_eval_prompts(cases: list[EvalCase], output_path: Path):
    """Output eval prompts as JSON for testing a live model."""
    prompts = []
    for case in cases:
        prompts.append({
            "id": f"{case.category}/{case.rule_id}",
            "category": case.category,
            "card_id": case.card_id,
            "rule_id": case.rule_id,
            "description": case.description,
            "user_message": case.user_prompt,
            "context": case.context,
            "criteria": {
                "must_contain": case.must_contain,
                "must_not_contain": case.must_not_contain,
                "must_match_pattern": case.must_match_pattern,
                "must_not_match_pattern": case.must_not_match_pattern,
            },
        })

    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(prompts)} eval prompts to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model fidelity against GATC knowledge cards"
    )
    parser.add_argument("--data", type=Path, help="JSONL file to evaluate")
    parser.add_argument("--generate-prompts", action="store_true",
                        help="Generate eval prompts JSON")
    parser.add_argument("--coverage", action="store_true",
                        help="Show eval coverage of knowledge cards")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for eval prompts")

    args = parser.parse_args()

    cards = load_all_cards()
    cases = generate_all_eval_cases(cards)

    if args.coverage:
        show_coverage(cards, cases)
        return

    if args.generate_prompts:
        output = args.output or (SCRIPT_DIR.parent / "data" / "eval_knowledge_prompts.json")
        generate_eval_prompts(cases, output)
        return

    if args.data:
        if not args.data.exists():
            print(f"File not found: {args.data}")
            sys.exit(1)

        print(f"Evaluating {args.data} against {len(cases)} eval cases...\n")
        results = evaluate_jsonl(args.data, cases, verbose=args.verbose)

        print(f"{'='*60}")
        print(f"  KNOWLEDGE FIDELITY EVAL RESULTS")
        print(f"{'='*60}")
        print(f"\n  Examples evaluated: {results['total_examples']}")
        print(f"  Matched to cases: {results['matched_to_case']}")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")

        if results["matched_to_case"] > 0:
            pct = 100 * results["passed"] / results["matched_to_case"]
            print(f"  Pass rate: {pct:.1f}%")

        print(f"\n  By category:")
        for cat, stats in sorted(results["by_category"].items()):
            pct = 100 * stats["passed"] / max(stats["total"], 1)
            status = "PASS" if pct >= 80 else "WARN" if pct >= 50 else "FAIL"
            print(f"    [{status}] {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")

        if args.verbose and results["failures_detail"]:
            print(f"\n  Failures:")
            for detail in results["failures_detail"][:20]:
                print(f"    Line {detail['line']} ({detail['case']}): {detail['failures']}")
                print(f"      Response: {detail['response_preview']}...")

        return

    parser.print_help()
    print("\nQuick start:")
    print("  python eval_knowledge_fidelity.py --coverage")
    print("  python eval_knowledge_fidelity.py --generate-prompts")
    print("  python eval_knowledge_fidelity.py --data training/data/gold_examples/gold_gatc_explanations.jsonl -v")


if __name__ == "__main__":
    main()
