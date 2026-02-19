"""
MiValta LLMIntent JSON Parser — Production Post-Processor

Extracts and validates the first valid LLMIntent JSON object from raw
model output. Handles common small-model artifacts:

  1. Multiple JSON objects concatenated (takes the first valid one)
  2. JavaScript-style string concatenation ("a" + "b")
  3. Trailing text after the JSON closing brace
  4. Missing closing braces (attempts repair)
  5. Single-quoted strings (converts to double-quoted)

This is the reference Python implementation. The same logic should be
ported to Kotlin (Android) / Swift (iOS) for on-device parsing.

Usage:
    from shared.llm_intent_parser import parse_llm_intent

    raw = model.generate(...)
    result = parse_llm_intent(raw)
    if result is not None:
        print(result["intent"], result["message"])
"""

import json
import re
from typing import Optional


# Required fields per LLMIntent schema
REQUIRED_FIELDS = {"intent", "response_type", "message", "source_cards", "guardrail_triggered"}

VALID_INTENTS = {
    "question", "replan", "encouragement", "feedback",
    "compliance", "general", "blocked", "medical_red_flag",
}

VALID_RESPONSE_TYPES = {
    "DailyBrief", "ExplainWorkout", "ExplainZone", "WeeklyReview",
    "Encouragement", "SafetyWarning", "ReadinessSummary",
    "QuestionAnswer", "Decline",
}


def _fix_string_concatenation(text: str) -> str:
    """Fix JavaScript-style string concatenation: "a" + "b" -> "ab"

    The 360M model sometimes produces:
        "message": "Z2 60min " + "Easy Aerobic" + " (base phase) "
    This merges them into a single string.
    """
    # Pattern: "..." + "..."  (with optional whitespace around +)
    pattern = r'"([^"]*?)"\s*\+\s*"([^"]*?)"'
    prev = None
    result = text
    # Repeat until no more matches (handles chains of 3+ concatenations)
    while result != prev:
        prev = result
        result = re.sub(pattern, r'"\1\2"', result)
    return result


def _fix_single_quotes(text: str) -> str:
    """Convert single-quoted JSON strings to double-quoted.

    Only converts when it looks like JSON structure (after { or , or :).
    """
    # Simple heuristic: replace '...' with "..." when preceded by : or , or [
    return re.sub(r"(?<=[:,\[\{])\s*'([^']*?)'", r' "\1"', text)


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first complete JSON object from text using brace matching.

    Handles nested objects and strings containing braces.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    i = start

    while i < len(text):
        ch = text[i]

        if escape:
            escape = False
            i += 1
            continue

        if ch == "\\":
            escape = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

        i += 1

    # If we ran out of text with unclosed braces, try adding closing braces
    if depth > 0:
        candidate = text[start:] + ("}" * depth)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def _validate_llm_intent(obj: dict) -> bool:
    """Check that the parsed object has the required LLMIntent fields.

    Lenient on source_cards: if the model produces a valid response but
    forgets source_cards (or leaves it empty), we backfill a default
    rather than rejecting the entire response.
    """
    if not isinstance(obj, dict):
        return False

    # Check required fields, but allow source_cards to be missing/empty
    # (we'll backfill in _set_defaults)
    required_minus_cards = REQUIRED_FIELDS - {"source_cards"}
    if not required_minus_cards.issubset(obj.keys()):
        return False
    if obj.get("intent") not in VALID_INTENTS:
        return False
    if obj.get("response_type") not in VALID_RESPONSE_TYPES:
        return False
    if not isinstance(obj.get("message"), str) or len(obj["message"]) == 0:
        return False
    if not isinstance(obj.get("guardrail_triggered"), bool):
        return False
    return True


_DEFAULT_SOURCE_CARD = "josi_explanations"


def _set_defaults(obj: dict) -> dict:
    """Ensure optional fields have defaults.

    Backfills source_cards with a safe default if the model omitted it,
    rather than rejecting an otherwise valid response.
    """
    obj.setdefault("guardrail_reason", None)
    obj.setdefault("replan_request", None)
    obj.setdefault("tool_call", None)
    # Backfill source_cards if missing or empty
    if not isinstance(obj.get("source_cards"), list) or len(obj.get("source_cards", [])) == 0:
        obj["source_cards"] = [_DEFAULT_SOURCE_CARD]
    return obj


def parse_llm_intent(raw: str) -> Optional[dict]:
    """Parse raw model output into a validated LLMIntent dict.

    Returns None if no valid LLMIntent JSON can be extracted.

    Args:
        raw: Raw string output from the model (may contain artifacts).

    Returns:
        A validated LLMIntent dict, or None if parsing fails.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Step 1: Try direct parse (fast path for clean output)
    try:
        obj = json.loads(text)
        if _validate_llm_intent(obj):
            return _set_defaults(obj)
    except json.JSONDecodeError:
        pass

    # Step 2: Fix common artifacts
    text = _fix_string_concatenation(text)
    text = _fix_single_quotes(text)

    # Step 3: Extract first JSON object via brace matching
    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if _validate_llm_intent(obj):
                return _set_defaults(obj)
        except json.JSONDecodeError:
            pass

        # Step 4: Try fixing concatenation on the extracted candidate
        fixed = _fix_string_concatenation(candidate)
        if fixed != candidate:
            try:
                obj = json.loads(fixed)
                if _validate_llm_intent(obj):
                    return _set_defaults(obj)
            except json.JSONDecodeError:
                pass

    return None


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import sys

    test_cases = [
        # Clean output
        '{"intent": "question", "response_type": "ExplainWorkout", "message": "Today is Z2.", "source_cards": ["zone_physiology"], "guardrail_triggered": false, "guardrail_reason": null, "replan_request": null, "tool_call": null}',

        # String concatenation
        '{"intent": "question", "response_type": "ExplainWorkout", "message": "Z2 60min " + "Easy Aerobic" + " (base phase) ", "source_cards": ["session_rules"], "guardrail_triggered": false, "guardrail_reason": null, "replan_request": null, "tool_call": null}',

        # Multiple JSON objects (take first)
        '{"intent": "question", "response_type": "ExplainWorkout", "message": "Z2 60min Easy aerobic.", "source_cards": ["session_rules"], "guardrail_triggered": false, "guardrail_reason": null, "replan_request": null, "tool_call": null},\n{"intent": "general", "response_type": "WeeklyReview", "message": "Extra stuff", "source_cards": ["zone_physiology"], "guardrail_triggered": false}',

        # Trailing text
        '{"intent": "blocked", "response_type": "Decline", "message": "Can not do that.", "source_cards": ["josi_explanations"], "guardrail_triggered": true, "guardrail_reason": "i6_violation", "replan_request": null, "tool_call": null}\nSome trailing text here',

        # Empty / garbage
        "",
        "not json at all",
        '{"intent": "invalid"}',
    ]

    passed = 0
    for i, tc in enumerate(test_cases):
        result = parse_llm_intent(tc)
        status = "OK" if (result is not None) == (i < 4) else "FAIL"
        if status == "OK":
            passed += 1
        print(f"  [{status}] Test {i+1}: {'parsed' if result else 'None'}")
        if result:
            print(f"         intent={result['intent']}, rtype={result['response_type']}")
            print(f"         msg={result['message'][:60]}...")

    print(f"\n{passed}/{len(test_cases)} passed")
