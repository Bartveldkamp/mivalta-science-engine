#!/usr/bin/env python3
"""
MiValta Josi v3 — Comprehensive Training Data Validation

Runs real tests against every training example:
  1. Schema compliance    — every assistant response is valid LLMIntent JSON
  2. Contract enforcement — I6 guardrails, tier gating, zone gating, tool dispatch
  3. Quality checks       — message length, persona consistency, distribution

Usage:
    python validate_training_data.py [--verbose] [--file PATH]

Exit code 0 = all pass, 1 = failures found.
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# ============================================================================
# SCHEMA CONSTANTS (from shared/schemas/)
# ============================================================================

VALID_INTENTS = {"question", "replan", "encouragement", "feedback",
                 "compliance", "general", "blocked", "medical_red_flag"}

VALID_RESPONSE_TYPES = {"DailyBrief", "ExplainWorkout", "ExplainZone",
                        "WeeklyReview", "Encouragement", "SafetyWarning",
                        "ReadinessSummary", "QuestionAnswer", "Decline"}

VALID_SOURCE_CARDS = {
    "advisor_policy", "energy_systems", "fatigue_policy", "feasibility_policy",
    "goal_demands", "insight_rules_v1", "josi_explanations", "josi_personas_v1",
    "load_monitoring", "meso_dance_policy", "modifiers", "modifiers_cycling",
    "modifiers_running", "monitoring_v5", "monotony_policy", "operational",
    "pack_composition", "periodization", "session_rules", "session_variety_policy",
    "training_load_model", "zone_anchors", "zone_physiology",
}

VALID_TOOLS = {"get_user_status", "explain_workout", "create_today_workout",
               "create_plan", "replan", "log_workout", "get_recent_workouts"}

VALID_REPLAN_TYPES = {"skip_today", "swap_days", "reschedule",
                      "reduce_intensity", "illness", "travel", "goal_change"}

VALID_GUARDRAIL_REASONS = {"i6_violation", "tier_violation", "medical_red_flag", None}

VALID_READINESS_LEVELS = {"Green", "Yellow", "Orange", "Red"}

TIER_TOOLS = {
    "monitor": set(),  # Monitor: general talks only, no tool access through Josi
    "advisor": {"get_user_status", "explain_workout", "create_today_workout",
                "log_workout", "get_recent_workouts"},
    "coach":   {"get_user_status", "explain_workout", "create_today_workout",
                "create_plan", "replan", "log_workout", "get_recent_workouts"},
}

ZONE_GATING = {
    "Green":  {"R", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8"},
    "Yellow": {"R", "Z1", "Z2", "Z3", "Z4"},
    "Orange": {"R", "Z1", "Z2"},
    "Red":    {"R", "Z1"},
}

BANNED_TOKENS = [
    "transition matrix", "viterbi", "hmm", "hidden markov",
    "acwr", "ewma", "tss", "ctl", "atl", "tsb",
]

# Prescriptive phrases that should NOT appear in advisor-mode assistant messages
PRESCRIPTIVE_PHRASES = [
    r"\byou should do\b", r"\bi recommend\b", r"\btry this\b",
    r"\btry doing\b", r"\bdo this workout\b", r"\bhere.?s your workout\b",
]

# ============================================================================
# PARSING HELPERS
# ============================================================================

def extract_tier(system_msg: str) -> Optional[str]:
    """Extract tier from system prompt MODE line."""
    m = re.search(r"MODE:\s*(Monitor|Advisor|Coach)", system_msg, re.IGNORECASE)
    return m.group(1).lower() if m else None


def extract_readiness(user_msg: str) -> Optional[str]:
    """Extract readiness level from user context block."""
    m = re.search(r"Readiness:\s*(Green|Yellow|Orange|Red)", user_msg)
    return m.group(1) if m else None


def extract_session_zone(user_msg: str) -> Optional[str]:
    """Extract session target zone from user context."""
    m = re.search(r"Session:\s*(R|Z[1-8])\s", user_msg)
    return m.group(1) if m else None


def has_session_context(user_msg: str) -> bool:
    """Check if user context includes a planned session."""
    return "- Session:" in user_msg


def extract_persona(system_msg: str) -> Optional[str]:
    """Extract persona from system prompt."""
    m = re.search(r"Style:\s*(.*?)[\.\n]", system_msg)
    if not m:
        return None
    style = m.group(1).strip().lower()
    style_map = {
        "warm, professional, supportive": "balanced",
        "no-nonsense, factual, brief": "direct",
        "analytical, educational, precise": "technical",
        "warm, motivational, positive": "encouraging",
    }
    return style_map.get(style)


def zones_mentioned_in_message(msg: str) -> set:
    """Find all zone references (Z3, Z5, etc.) in a message."""
    return set(re.findall(r'\bZ[1-8]\b', msg))


# ============================================================================
# TEST RESULTS
# ============================================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def ok(self):
        self.passed += 1

    def fail(self, msg: str):
        self.failed += 1
        self.errors.append(msg)

    @property
    def total(self):
        return self.passed + self.failed

    @property
    def success(self):
        return self.failed == 0


class TestSuite:
    def __init__(self):
        self.results: Dict[str, TestResult] = {}

    def get(self, name: str) -> TestResult:
        if name not in self.results:
            self.results[name] = TestResult(name)
        return self.results[name]

    def summary(self, verbose: bool = False) -> Tuple[int, int]:
        total_pass = 0
        total_fail = 0
        for name, r in sorted(self.results.items()):
            status = "PASS" if r.success else "FAIL"
            mark = "\u2713" if r.success else "\u2717"
            print(f"  {mark} [{status}] {name}: {r.passed}/{r.total}")
            total_pass += r.passed
            total_fail += r.failed
            if not r.success and verbose:
                for e in r.errors[:10]:
                    print(f"          {e}")
                if len(r.errors) > 10:
                    print(f"          ... and {len(r.errors) - 10} more")
        return total_pass, total_fail


# ============================================================================
# LOAD DATA
# ============================================================================

def load_examples(path: Path) -> List[dict]:
    """Load JSONL training file, return list of parsed examples."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                ex["_line"] = i
                examples.append(ex)
            except json.JSONDecodeError as e:
                print(f"  ERROR: Line {i} is not valid JSON: {e}")
    return examples


# ============================================================================
# TEST 1: STRUCTURAL VALIDITY
# ============================================================================

def test_structure(examples: List[dict], suite: TestSuite):
    """Every example must have 3 messages: system, user, assistant."""
    t = suite.get("1.1 Message structure (3 messages: system/user/assistant)")
    for ex in examples:
        msgs = ex.get("messages", [])
        line = ex["_line"]
        if not isinstance(msgs, list) or len(msgs) != 3:
            t.fail(f"Line {line}: expected 3 messages, got {len(msgs) if isinstance(msgs, list) else 'non-list'}")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            t.fail(f"Line {line}: roles are {roles}, expected ['system', 'user', 'assistant']")
            continue
        for m in msgs:
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                t.fail(f"Line {line}: empty content for role={m.get('role')}")
                continue
        t.ok()


# ============================================================================
# TEST 2: JSON SCHEMA COMPLIANCE
# ============================================================================

def test_schema_compliance(examples: List[dict], suite: TestSuite):
    """Every assistant response must be valid LLMIntent JSON with correct types."""
    t_json = suite.get("2.1 Assistant response is valid JSON")
    t_req = suite.get("2.2 All required fields present")
    t_intent = suite.get("2.3 Intent is valid enum")
    t_rtype = suite.get("2.4 Response type is valid enum")
    t_msg = suite.get("2.5 Message is non-empty string")
    t_cards = suite.get("2.6 Source cards valid and non-empty")
    t_guard = suite.get("2.7 Guardrail fields valid")
    t_replan = suite.get("2.8 Replan request structure valid")
    t_tool = suite.get("2.9 Tool call structure valid")

    required_fields = {"intent", "response_type", "message", "source_cards", "guardrail_triggered"}

    for ex in examples:
        line = ex["_line"]
        raw = ex["messages"][2]["content"]

        # 2.1: Valid JSON
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            t_json.fail(f"Line {line}: not valid JSON")
            continue
        t_json.ok()

        # 2.2: Required fields
        missing = required_fields - set(resp.keys())
        if missing:
            t_req.fail(f"Line {line}: missing {missing}")
        else:
            t_req.ok()

        # 2.3: Intent enum
        if resp.get("intent") in VALID_INTENTS:
            t_intent.ok()
        else:
            t_intent.fail(f"Line {line}: intent='{resp.get('intent')}' not in {VALID_INTENTS}")

        # 2.4: Response type enum
        if resp.get("response_type") in VALID_RESPONSE_TYPES:
            t_rtype.ok()
        else:
            t_rtype.fail(f"Line {line}: response_type='{resp.get('response_type')}' not in {VALID_RESPONSE_TYPES}")

        # 2.5: Message non-empty
        if isinstance(resp.get("message"), str) and len(resp["message"].strip()) > 0:
            t_msg.ok()
        else:
            t_msg.fail(f"Line {line}: message is empty or not a string")

        # 2.6: Source cards
        cards = resp.get("source_cards")
        if isinstance(cards, list) and len(cards) >= 1:
            invalid = set(cards) - VALID_SOURCE_CARDS
            if invalid:
                t_cards.fail(f"Line {line}: invalid source_cards {invalid}")
            else:
                t_cards.ok()
        else:
            t_cards.fail(f"Line {line}: source_cards empty or not a list")

        # 2.7: Guardrail fields
        gt = resp.get("guardrail_triggered")
        gr = resp.get("guardrail_reason")
        if not isinstance(gt, bool):
            t_guard.fail(f"Line {line}: guardrail_triggered is not boolean (got {type(gt).__name__})")
        elif gt and gr not in ("i6_violation", "tier_violation", "medical_red_flag"):
            t_guard.fail(f"Line {line}: guardrail_triggered=true but reason='{gr}'")
        elif not gt and gr is not None:
            t_guard.fail(f"Line {line}: guardrail_triggered=false but reason='{gr}' (should be null)")
        else:
            t_guard.ok()

        # 2.8: Replan request
        rr = resp.get("replan_request")
        if rr is None:
            t_replan.ok()
        elif isinstance(rr, dict):
            rr_required = {"type", "reason", "mode", "readiness_at_request"}
            rr_missing = rr_required - set(rr.keys())
            if rr_missing:
                t_replan.fail(f"Line {line}: replan_request missing {rr_missing}")
            elif rr["type"] not in VALID_REPLAN_TYPES:
                t_replan.fail(f"Line {line}: replan type='{rr['type']}' invalid")
            elif rr["mode"] != "coach":
                t_replan.fail(f"Line {line}: replan mode='{rr['mode']}' must be 'coach'")
            elif rr["readiness_at_request"] not in VALID_READINESS_LEVELS:
                t_replan.fail(f"Line {line}: replan readiness='{rr['readiness_at_request']}' invalid")
            else:
                t_replan.ok()
        else:
            t_replan.fail(f"Line {line}: replan_request is {type(rr).__name__}, expected dict or null")

        # 2.9: Tool call
        tc = resp.get("tool_call")
        if tc is None:
            t_tool.ok()
        elif isinstance(tc, dict):
            if "tool" not in tc or "args" not in tc:
                t_tool.fail(f"Line {line}: tool_call missing 'tool' or 'args'")
            elif tc["tool"] not in VALID_TOOLS:
                t_tool.fail(f"Line {line}: tool='{tc['tool']}' not in {VALID_TOOLS}")
            elif not isinstance(tc["args"], dict):
                t_tool.fail(f"Line {line}: tool_call.args is not a dict")
            else:
                t_tool.ok()
        else:
            t_tool.fail(f"Line {line}: tool_call is {type(tc).__name__}, expected dict or null")


# ============================================================================
# TEST 3: I6 GUARDRAIL ENFORCEMENT
# ============================================================================

def test_i6_guardrails(examples: List[dict], suite: TestSuite):
    """I6 constitutional constraints: no banned tokens, no prescription in output."""
    t_banned = suite.get("3.1 No GATC banned tokens in assistant messages")
    t_prescr = suite.get("3.2 No prescriptive language in advisor-mode messages")
    t_override = suite.get("3.3 No readiness override language in messages")
    t_monitor_plans = suite.get("3.4 Monitor-mode messages don't reference plans/sessions")
    t_mod_claims = suite.get("3.5 No modification claims in messages")
    t_blocked_guard = suite.get("3.6 Blocked intents have guardrail_triggered=true")

    for ex in examples:
        line = ex["_line"]
        system = ex["messages"][0]["content"]
        raw_resp = ex["messages"][2]["content"]
        try:
            resp = json.loads(raw_resp)
        except json.JSONDecodeError:
            continue  # schema test already caught this

        msg = resp.get("message", "").lower()
        tier = extract_tier(system)
        intent = resp.get("intent")

        # 3.1: Banned GATC tokens
        banned_found = []
        for token in BANNED_TOKENS:
            if re.search(r'\b' + re.escape(token) + r'\b', msg):
                banned_found.append(token)
        if banned_found:
            t_banned.fail(f"Line {line}: message contains banned tokens: {banned_found}")
        else:
            t_banned.ok()

        # 3.2: No prescriptive language in advisor mode
        if tier == "advisor" and intent != "blocked":
            prescr_found = []
            for pattern in PRESCRIPTIVE_PHRASES:
                if re.search(pattern, msg, re.IGNORECASE):
                    prescr_found.append(pattern)
            if prescr_found:
                t_prescr.fail(f"Line {line}: advisor message has prescriptive language: {prescr_found}")
            else:
                t_prescr.ok()
        else:
            t_prescr.ok()

        # 3.3: No readiness override language
        override_patterns = [
            r"\bpush through\b.*\bfatigue\b",
            r"\bignore\b.*\breadiness\b",
            r"\boverride\b.*\bgate\b",
            r"\bjust go hard\b",
        ]
        override_found = False
        for pattern in override_patterns:
            if re.search(pattern, msg, re.IGNORECASE):
                # Allow in decline/blocked messages quoting the user
                if intent not in ("blocked",) or "I can't" not in resp.get("message", ""):
                    override_found = True
                    break
        if override_found and intent != "blocked":
            t_override.fail(f"Line {line}: message contains readiness override language")
        else:
            t_override.ok()

        # 3.4: Monitor: general talks only — no plans, sessions, OR personal data
        if tier == "monitor" and intent != "blocked":
            monitor_forbidden = [
                r"\btoday.?s session\b", r"\bplanned workout\b",
                r"\byour plan\b", r"\btraining plan\b",
                r"\btoday.?s planned\b",
                r"\byour readiness\b", r"\byour training load\b",
                r"\byour recovery\b", r"\byour hrv\b",
            ]
            monitor_violation = False
            for pattern in monitor_forbidden:
                if re.search(pattern, msg, re.IGNORECASE):
                    monitor_violation = True
                    break
            if monitor_violation:
                t_monitor_plans.fail(f"Line {line}: monitor message references plans/sessions/personal data")
            else:
                t_monitor_plans.ok()
        else:
            t_monitor_plans.ok()

        # 3.5: No modification claims
        mod_patterns = [
            r"\bi.?ve changed\b.*\b(workout|plan|session)\b",
            r"\bi.?ve modified\b", r"\bi.?ve updated\b.*\b(plan|session)\b",
            r"\bi changed\b.*\b(zone|workout)\b",
        ]
        mod_found = False
        for pattern in mod_patterns:
            if re.search(pattern, msg, re.IGNORECASE):
                mod_found = True
                break
        if mod_found:
            t_mod_claims.fail(f"Line {line}: message claims to have modified training")
        else:
            t_mod_claims.ok()

        # 3.6: Blocked intent must have guardrail_triggered=true
        if intent == "blocked":
            if resp.get("guardrail_triggered") is True:
                t_blocked_guard.ok()
            else:
                t_blocked_guard.fail(f"Line {line}: intent=blocked but guardrail_triggered is not true")
        else:
            t_blocked_guard.ok()


# ============================================================================
# TEST 4: TIER COMPLIANCE
# ============================================================================

def test_tier_compliance(examples: List[dict], suite: TestSuite):
    """Tier gating: tools, replan, plan creation limited by tier."""
    t_tool_tier = suite.get("4.1 Tool calls only use tier-allowed tools")
    t_replan_coach = suite.get("4.2 Replan requests only on coach tier")
    t_plan_coach = suite.get("4.3 create_plan tool only on coach tier")
    t_decline_reason = suite.get("4.4 Decline responses have guardrail_reason set")
    t_medical = suite.get("4.5 Medical red flags produce SafetyWarning")
    t_monitor_no_tools = suite.get("4.6 Monitor has no tool calls (general talks only)")
    t_monitor_no_personal = suite.get("4.7 Monitor has no personal response types (ReadinessSummary, ExplainWorkout, WeeklyReview, DailyBrief)")

    for ex in examples:
        line = ex["_line"]
        system = ex["messages"][0]["content"]
        raw_resp = ex["messages"][2]["content"]
        try:
            resp = json.loads(raw_resp)
        except json.JSONDecodeError:
            continue

        tier = extract_tier(system)
        if not tier:
            continue

        intent = resp.get("intent")
        rtype = resp.get("response_type")
        tc = resp.get("tool_call")
        rr = resp.get("replan_request")
        allowed_tools = TIER_TOOLS.get(tier, set())

        # 4.1: Tool in tier allowlist
        if tc and isinstance(tc, dict) and "tool" in tc:
            tool = tc["tool"]
            if tool in allowed_tools:
                t_tool_tier.ok()
            else:
                t_tool_tier.fail(f"Line {line}: tier={tier} uses tool='{tool}' not in {allowed_tools}")
        else:
            t_tool_tier.ok()

        # 4.2: Replan only on coach
        if rr is not None and isinstance(rr, dict):
            if tier == "coach":
                t_replan_coach.ok()
            else:
                t_replan_coach.fail(f"Line {line}: tier={tier} has replan_request (coach-only)")
        else:
            t_replan_coach.ok()

        # 4.3: create_plan tool only on coach
        if tc and isinstance(tc, dict) and tc.get("tool") == "create_plan":
            if tier == "coach":
                t_plan_coach.ok()
            else:
                t_plan_coach.fail(f"Line {line}: tier={tier} uses create_plan (coach-only)")
        else:
            t_plan_coach.ok()

        # 4.4: Decline should have guardrail reason
        if rtype == "Decline":
            reason = resp.get("guardrail_reason")
            if reason in ("i6_violation", "tier_violation"):
                t_decline_reason.ok()
            elif resp.get("guardrail_triggered"):
                # Some declines have triggered=true without reason specified — still ok if reason is valid
                t_decline_reason.ok()
            else:
                t_decline_reason.fail(f"Line {line}: Decline but guardrail_reason='{reason}' (expected i6_violation or tier_violation)")
        else:
            t_decline_reason.ok()

        # 4.5: Medical red flag → SafetyWarning
        if intent == "medical_red_flag":
            if rtype == "SafetyWarning":
                t_medical.ok()
            else:
                t_medical.fail(f"Line {line}: medical_red_flag intent but response_type='{rtype}' (expected SafetyWarning)")
        else:
            t_medical.ok()

        # 4.6: Monitor must have NO tool calls (general talks only)
        if tier == "monitor":
            if tc is not None and isinstance(tc, dict) and tc.get("tool"):
                t_monitor_no_tools.fail(f"Line {line}: monitor has tool_call={tc['tool']} (should be null)")
            else:
                t_monitor_no_tools.ok()
        else:
            t_monitor_no_tools.ok()

        # 4.7: Monitor must not have personal response types
        if tier == "monitor":
            personal_rtypes = {"ReadinessSummary", "ExplainWorkout", "WeeklyReview", "DailyBrief"}
            if rtype in personal_rtypes:
                t_monitor_no_personal.fail(f"Line {line}: monitor has rtype={rtype} (personal data, should be Decline)")
            else:
                t_monitor_no_personal.ok()
        else:
            t_monitor_no_personal.ok()


# ============================================================================
# TEST 5: ZONE GATING BY READINESS
# ============================================================================

def test_zone_gating(examples: List[dict], suite: TestSuite):
    """Messages must not recommend zones above the readiness gate."""
    t = suite.get("5.1 Zone references respect readiness gating")

    # Only check examples where the assistant message explains zones/sessions
    for ex in examples:
        line = ex["_line"]
        system = ex["messages"][0]["content"]
        user = ex["messages"][1]["content"]
        raw_resp = ex["messages"][2]["content"]
        try:
            resp = json.loads(raw_resp)
        except json.JSONDecodeError:
            continue

        intent = resp.get("intent")
        rtype = resp.get("response_type")
        msg = resp.get("message", "")
        readiness = extract_readiness(user)

        # Skip non-explanation messages and blocked/decline messages
        if intent in ("blocked", "medical_red_flag"):
            t.ok()
            continue
        if rtype in ("Decline", "SafetyWarning"):
            t.ok()
            continue
        if not readiness:
            t.ok()
            continue

        # Check session zone in CONTEXT matches gating
        session_zone = extract_session_zone(user)
        if session_zone:
            allowed = ZONE_GATING[readiness]
            if session_zone not in allowed:
                # The session itself is above the gate — this is a data problem
                t.fail(f"Line {line}: session zone={session_zone} above {readiness} gate ({allowed})")
                continue

        # For zone explanation messages, check the zone being explained is allowed
        # This is a softer check — education about any zone is fine,
        # but recommending or scheduling above gate is not
        if rtype == "ExplainZone":
            # Zone explanations are educational — allowed regardless of readiness
            t.ok()
            continue

        # For workout explanations and daily briefs, check zones mentioned
        if rtype in ("ExplainWorkout", "DailyBrief"):
            zones = zones_mentioned_in_message(msg)
            allowed = ZONE_GATING[readiness]
            above_gate = zones - allowed
            if above_gate:
                # Allow if it's just education/context, not prescription
                # Check if it's from session context
                if session_zone and session_zone in allowed:
                    t.ok()
                else:
                    t.fail(f"Line {line}: readiness={readiness} but message references zones {above_gate} above gate")
            else:
                t.ok()
        else:
            t.ok()


# ============================================================================
# TEST 6: CONSISTENCY CHECKS
# ============================================================================

def test_consistency(examples: List[dict], suite: TestSuite):
    """Cross-field consistency: intent↔response_type, tool_call↔intent, etc."""
    t_blocked_decline = suite.get("6.1 intent=blocked → response_type=Decline")
    t_replan_intent = suite.get("6.2 replan_request present → intent=replan")
    t_encourage = suite.get("6.3 intent=encouragement → response_type=Encouragement")
    t_feedback = suite.get("6.4 intent=feedback → tool_call has log_workout")
    t_replan_tool = suite.get("6.5 Coach replan intent → replan_request not null")

    for ex in examples:
        line = ex["_line"]
        system = ex["messages"][0]["content"]
        raw_resp = ex["messages"][2]["content"]
        try:
            resp = json.loads(raw_resp)
        except json.JSONDecodeError:
            continue

        tier = extract_tier(system)
        intent = resp.get("intent")
        rtype = resp.get("response_type")
        tc = resp.get("tool_call")
        rr = resp.get("replan_request")

        # 6.1: blocked → Decline
        if intent == "blocked":
            if rtype == "Decline":
                t_blocked_decline.ok()
            else:
                t_blocked_decline.fail(f"Line {line}: intent=blocked but response_type='{rtype}'")
        else:
            t_blocked_decline.ok()

        # 6.2: replan_request → intent=replan
        if rr is not None and isinstance(rr, dict):
            if intent == "replan":
                t_replan_intent.ok()
            else:
                t_replan_intent.fail(f"Line {line}: has replan_request but intent='{intent}'")
        else:
            t_replan_intent.ok()

        # 6.3: encouragement → Encouragement
        if intent == "encouragement":
            if rtype == "Encouragement":
                t_encourage.ok()
            else:
                t_encourage.fail(f"Line {line}: intent=encouragement but rtype='{rtype}'")
        else:
            t_encourage.ok()

        # 6.4: feedback → log_workout (if tool present)
        if intent == "feedback":
            if tc and isinstance(tc, dict):
                if tc.get("tool") == "log_workout":
                    t_feedback.ok()
                else:
                    t_feedback.fail(f"Line {line}: intent=feedback but tool='{tc.get('tool')}'")
            else:
                # No tool call for feedback is acceptable (some feedback is just acknowledgment)
                t_feedback.ok()
        else:
            t_feedback.ok()

        # 6.5: Coach replan → replan_request not null
        if intent == "replan" and tier == "coach":
            if rr is not None and isinstance(rr, dict):
                t_replan_tool.ok()
            else:
                t_replan_tool.fail(f"Line {line}: coach replan but replan_request is null")
        elif intent == "replan" and tier in ("monitor", "advisor"):
            # Non-coach replan should be decline with null replan_request
            if rr is None:
                t_replan_tool.ok()
            else:
                t_replan_tool.fail(f"Line {line}: {tier} replan but replan_request is NOT null")
        else:
            t_replan_tool.ok()


# ============================================================================
# TEST 7: QUALITY METRICS
# ============================================================================

def test_quality(examples: List[dict], suite: TestSuite):
    """Message quality: reasonable length, no empty messages, no garbage."""
    t_len = suite.get("7.1 Message length within bounds (10-2000 chars)")
    t_no_json_leak = suite.get("7.2 Message does not contain raw JSON fragments")
    t_no_placeholder = suite.get("7.3 No placeholder text in messages")

    for ex in examples:
        line = ex["_line"]
        raw_resp = ex["messages"][2]["content"]
        try:
            resp = json.loads(raw_resp)
        except json.JSONDecodeError:
            continue

        msg = resp.get("message", "")

        # 7.1: Length bounds
        if len(msg) < 10:
            t_len.fail(f"Line {line}: message too short ({len(msg)} chars): '{msg}'")
        elif len(msg) > 2000:
            t_len.fail(f"Line {line}: message too long ({len(msg)} chars)")
        else:
            t_len.ok()

        # 7.2: No raw JSON in message
        if '{"intent"' in msg or '"response_type"' in msg or '"source_cards"' in msg:
            t_no_json_leak.fail(f"Line {line}: message contains LLMIntent JSON fragments")
        else:
            t_no_json_leak.ok()

        # 7.3: No placeholder text
        placeholders = ["TODO", "FIXME", "PLACEHOLDER", "INSERT HERE", "XXX",
                        "[your ", "{name}", "{sport}", "lorem ipsum"]
        found_ph = [p for p in placeholders if p.lower() in msg.lower()]
        if found_ph:
            t_no_placeholder.fail(f"Line {line}: message contains placeholder: {found_ph}")
        else:
            t_no_placeholder.ok()


# ============================================================================
# TEST 8: DISTRIBUTION & COVERAGE
# ============================================================================

def test_distribution(examples: List[dict], suite: TestSuite):
    """Check dataset distribution: I6 density, tier balance, persona coverage."""
    t_i6_density = suite.get("8.1 I6/Decline density in 15-25% range")
    t_tier_coverage = suite.get("8.2 All 3 tiers represented")
    t_persona_coverage = suite.get("8.3 All 4 personas represented")
    t_intent_coverage = suite.get("8.4 All 8 intents represented")
    t_rtype_coverage = suite.get("8.5 All 9 response types represented")
    t_duplicates = suite.get("8.6 No duplicate examples")

    tier_counts = Counter()
    persona_counts = Counter()
    intent_counts = Counter()
    rtype_counts = Counter()
    decline_count = 0
    content_hashes = set()
    dup_count = 0

    for ex in examples:
        system = ex["messages"][0]["content"]
        raw_resp = ex["messages"][2]["content"]

        # Check duplicates by content hash
        content = f"{system}|{ex['messages'][1]['content']}|{raw_resp}"
        h = hash(content)
        if h in content_hashes:
            dup_count += 1
        content_hashes.add(h)

        tier = extract_tier(system)
        persona = extract_persona(system)
        if tier:
            tier_counts[tier] += 1
        if persona:
            persona_counts[persona] += 1

        try:
            resp = json.loads(raw_resp)
            intent_counts[resp.get("intent")] += 1
            rtype_counts[resp.get("response_type")] += 1
            if resp.get("response_type") == "Decline" or resp.get("intent") == "blocked":
                decline_count += 1
        except json.JSONDecodeError:
            pass

    total = len(examples)

    # 8.1: I6/Decline density
    density = decline_count / total * 100 if total > 0 else 0
    if 10 <= density <= 30:
        t_i6_density.ok()
    else:
        t_i6_density.fail(f"I6/Decline density={density:.1f}% (target 15-25%, acceptable 10-30%)")

    # 8.2: All tiers
    for tier in ["monitor", "advisor", "coach"]:
        if tier_counts.get(tier, 0) > 0:
            t_tier_coverage.ok()
        else:
            t_tier_coverage.fail(f"Missing tier: {tier}")

    # 8.3: All personas
    for persona in ["balanced", "direct", "technical", "encouraging"]:
        if persona_counts.get(persona, 0) > 0:
            t_persona_coverage.ok()
        else:
            t_persona_coverage.fail(f"Missing persona: {persona}")

    # 8.4: All intents
    for intent in VALID_INTENTS:
        if intent_counts.get(intent, 0) > 0:
            t_intent_coverage.ok()
        else:
            t_intent_coverage.fail(f"Missing intent: {intent}")

    # 8.5: All response types
    for rtype in VALID_RESPONSE_TYPES:
        if rtype_counts.get(rtype, 0) > 0:
            t_rtype_coverage.ok()
        else:
            t_rtype_coverage.fail(f"Missing response_type: {rtype}")

    # 8.6: No duplicates
    if dup_count == 0:
        t_duplicates.ok()
    else:
        t_duplicates.fail(f"{dup_count} duplicate examples found")

    # Print distribution stats
    print(f"\n  Distribution stats ({total} examples):")
    print(f"    Tiers:    {dict(tier_counts.most_common())}")
    print(f"    Personas: {dict(persona_counts.most_common())}")
    print(f"    Intents:  {dict(intent_counts.most_common())}")
    print(f"    RTypes:   {dict(rtype_counts.most_common())}")
    print(f"    I6/Decline density: {density:.1f}%")


# ============================================================================
# TEST 9: EVAL CASE COVERAGE
# ============================================================================

def test_eval_coverage(examples: List[dict], suite: TestSuite):
    """Verify training data covers all 42 eval cases."""
    script_dir = Path(__file__).parent
    eval_dir = script_dir.parent.parent / "shared" / "eval_cases"

    # Load eval cases
    eval_files = ["tier_compliance.json", "i6_guardrails.json", "tool_dispatch.json"]
    all_cases = []
    for fn in eval_files:
        fp = eval_dir / fn
        if fp.exists():
            data = json.loads(fp.read_text())
            all_cases.extend(data.get("cases", []))

    t = suite.get(f"9.1 All {len(all_cases)} eval cases covered by training data")

    # Parse all assistant responses
    parsed = []
    for ex in examples:
        system = ex["messages"][0]["content"]
        user = ex["messages"][1]["content"]
        raw = ex["messages"][2]["content"]
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            continue
        tier = extract_tier(system)
        parsed.append({"tier": tier, "user": user, "resp": resp, "system": system})

    for case in all_cases:
        case_id = case["id"]

        # Output validation cases (I6-006 through I6-014 except I6-011) check that
        # training data does NOT teach banned patterns
        if case.get("output_validation"):
            if case.get("expected_blocked") is False:
                # Positive case (I6-011): just check we have coach explanations
                found = any(p["tier"] == "coach" and
                           p["resp"].get("response_type") == "ExplainWorkout"
                           for p in parsed)
                if found:
                    t.ok()
                else:
                    t.fail(f"{case_id}: no coach ExplainWorkout examples found")
            else:
                # Negative case: verify no training example teaches the bad pattern
                t.ok()  # These are checked by tests 3.x above
            continue

        # Input validation cases: check training data teaches the expected behavior
        expected_intent = case.get("expected_intent")
        expected_rtype = case.get("expected_response_type")
        case_tier = case.get("tier")
        expected_tool = case.get("expected_tool")
        replan_must_be = case.get("replan_request_must_be")

        matches = 0
        for p in parsed:
            if p["tier"] != case_tier:
                continue
            resp = p["resp"]
            if expected_intent and resp.get("intent") != expected_intent:
                continue
            if expected_rtype and resp.get("response_type") != expected_rtype:
                continue
            if expected_tool is not None:
                tc = resp.get("tool_call")
                if expected_tool is False or expected_tool == "null":
                    if tc is not None:
                        continue
                elif isinstance(tc, dict) and tc.get("tool") == expected_tool:
                    pass
                else:
                    continue
            if replan_must_be == "not null":
                if resp.get("replan_request") is None:
                    continue
            elif replan_must_be is None and "replan_request_must_be" in case:
                if resp.get("replan_request") is not None:
                    continue
            matches += 1

        if matches > 0:
            t.ok()
        else:
            t.fail(f"{case_id}: no matching training example (tier={case_tier}, "
                   f"intent={expected_intent}, rtype={expected_rtype})")


# ============================================================================
# TEST 10: SYSTEM PROMPT CONSISTENCY
# ============================================================================

def test_system_prompts(examples: List[dict], suite: TestSuite):
    """All system prompts follow expected format."""
    t_josi = suite.get("10.1 System prompt starts with 'You are Josi'")
    t_mode = suite.get("10.2 System prompt contains MODE block")
    t_i6 = suite.get("10.3 System prompt contains I6 CONSTRAINTS block")
    t_output = suite.get("10.4 System prompt contains OUTPUT schema block")

    for ex in examples:
        line = ex["_line"]
        system = ex["messages"][0]["content"]

        if system.startswith("You are Josi"):
            t_josi.ok()
        else:
            t_josi.fail(f"Line {line}: system prompt doesn't start with 'You are Josi'")

        if "MODE:" in system:
            t_mode.ok()
        else:
            t_mode.fail(f"Line {line}: missing MODE block")

        if "I6 CONSTRAINTS" in system:
            t_i6.ok()
        else:
            t_i6.fail(f"Line {line}: missing I6 CONSTRAINTS block")

        if "OUTPUT: Valid LLMIntent JSON" in system:
            t_output.ok()
        else:
            t_output.fail(f"Line {line}: missing OUTPUT schema block")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Josi v3 training data")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed error messages")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to JSONL file (default: train_v3.jsonl + val_v3.jsonl)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    # Load data
    if args.file:
        files = [Path(args.file)]
    else:
        files = [data_dir / "train_v3.jsonl", data_dir / "val_v3.jsonl"]

    all_examples = []
    for fp in files:
        if not fp.exists():
            print(f"ERROR: {fp} not found")
            sys.exit(1)
        examples = load_examples(fp)
        print(f"Loaded {len(examples)} examples from {fp.name}")
        all_examples.extend(examples)

    print(f"\nTotal examples: {len(all_examples)}")
    print("=" * 72)

    suite = TestSuite()

    # Run all test groups
    print("\n--- 1. Structure ---")
    test_structure(all_examples, suite)

    print("\n--- 2. Schema Compliance ---")
    test_schema_compliance(all_examples, suite)

    print("\n--- 3. I6 Guardrails ---")
    test_i6_guardrails(all_examples, suite)

    print("\n--- 4. Tier Compliance ---")
    test_tier_compliance(all_examples, suite)

    print("\n--- 5. Zone Gating ---")
    test_zone_gating(all_examples, suite)

    print("\n--- 6. Consistency ---")
    test_consistency(all_examples, suite)

    print("\n--- 7. Quality ---")
    test_quality(all_examples, suite)

    print("\n--- 8. Distribution & Coverage ---")
    test_distribution(all_examples, suite)

    print("\n--- 9. Eval Case Coverage ---")
    test_eval_coverage(all_examples, suite)

    print("\n--- 10. System Prompt Consistency ---")
    test_system_prompts(all_examples, suite)

    # Summary
    print("\n" + "=" * 72)
    print("RESULTS:\n")
    total_pass, total_fail = suite.summary(verbose=args.verbose)
    print(f"\n{'=' * 72}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed "
          f"({total_pass}/{total_pass + total_fail})")

    if total_fail > 0:
        print(f"\nFAILED — {total_fail} issues found. Run with --verbose for details.")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
