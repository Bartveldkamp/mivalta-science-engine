"""
MiValta GATCRequest Post-Processor — Deterministic Fixups

Applied after the interpreter model produces a GATCRequest JSON object.
Fixes known small-model failure modes that are hard to teach via fine-tuning:

  1. time_available_min extraction — model often omits the field even when
     the user mentions a duration in the message. Regex extraction from
     free_text is deterministic and reliable.

  2. Clarify enforcement — when action is create_workout but no sport is
     in the message AND no CONTEXT block provides one, switch to clarify.
     The 2B model struggles with this "no context = ask" pattern.

  3. Markdown fence stripping — base Gemma instruct tuning adds ```json
     fences that the LoRA can't fully suppress.

Usage:
    from shared.gatc_postprocessor import postprocess_gatc_request

    raw_json = model.generate(...)
    parsed = json.loads(raw_json)
    fixed = postprocess_gatc_request(parsed, user_message)
"""

import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Extract time_available_min from free_text
# ---------------------------------------------------------------------------

def _extract_duration(text: str) -> Optional[int]:
    """Extract duration in minutes from a natural-language string.

    Returns an int in [10, 300] or None.
    """
    lower = text.lower()

    # "45 minutes", "30 min", "20 mins"
    m = re.search(r'(\d+)\s*(?:min|minutes|mins|minute)\b', lower)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 300:
            return val

    # "1.5 hours", "2 hours", "1 hr"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b', lower)
    if m:
        val = int(float(m.group(1)) * 60)
        if 10 <= val <= 300:
            return val

    # "half an hour"
    if re.search(r'\bhalf\s+an?\s+hour\b', lower):
        return 30

    # "an hour", "one hour"
    if re.search(r'\ban?\s+hour\b', lower) or re.search(r'\bone\s+hour\b', lower):
        return 60

    return None


# ---------------------------------------------------------------------------
# 2. Detect missing sport / context for clarify enforcement
# ---------------------------------------------------------------------------

_HAS_CONTEXT_RE = re.compile(r'\nCONTEXT:', re.IGNORECASE)
_SPORT_IN_CONTEXT_RE = re.compile(r'sport:\s*\w+', re.IGNORECASE)


def _has_sport_context(user_message: str) -> bool:
    """Return True if the user message contains a CONTEXT block with Sport."""
    return bool(_SPORT_IN_CONTEXT_RE.search(user_message))


def _has_any_context(user_message: str) -> bool:
    """Return True if the user message contains any CONTEXT block."""
    return bool(_HAS_CONTEXT_RE.search(user_message))


# ---------------------------------------------------------------------------
# 3. Medical red-flag detection
# ---------------------------------------------------------------------------

# These indicate potential medical emergencies — NOT common illness.
# Common illness (cold, flu, sick, stomach bug) should stay as replan:illness.
_MEDICAL_RED_FLAGS = [
    r'\bchest\s+pain\b',
    r'\bheart\s+(?:pain|feels?\s+weird|racing|pounding)\b',
    r'\bdizz(?:y|iness)\b',
    r'\blight[- ]?headed\b',
    r'\bblacked?\s+out\b',
    r'\bfainted?\b',
    r'\blost\s+consciousness\b',
    r'\bcan\'?t\s+breathe\b',
    r'\bbreathing\s+(?:problem|difficult|trouble)\b',
    r'\bnumb(?:ness)?\b',
    r'\btingling\b',
    # Persistent pain / injury patterns (multi-day or specific body part pain)
    r'\b(?:hurt|pain)\w*\s+for\s+\d+\s+day',
    r'\bbeen\s+(?:hurt|pain)ing\b',
    r'\bsharp\s+pain\b',
    r'\bshooting\s+pain\b',
    # Chest/heart pain is ALWAYS a red flag (no duration required)
    r'\bpain\s+in\s+my\s+(?:chest|heart)\b',
    # Non-cardiac body part pain only qualifies with duration/persistence qualifier
    r'\b(?:knee|back|shin|shoulder|hip|ankle)\s+(?:has\s+been|been)\s+(?:hurt|pain)ing\b',
    r'\b(?:knee|back|shin|shoulder|hip|ankle)\b.{0,20}\bhurt(?:s|ing)\b.{0,20}\b\d+\s+day',
]

_MEDICAL_RED_FLAG_RES = [re.compile(p, re.IGNORECASE) for p in _MEDICAL_RED_FLAGS]


def _has_medical_red_flag(user_message: str) -> bool:
    """Return True if the user message contains medical red-flag symptoms."""
    return any(r.search(user_message) for r in _MEDICAL_RED_FLAG_RES)


# ---------------------------------------------------------------------------
# 4. Strip markdown fences
# ---------------------------------------------------------------------------

def strip_markdown_fences(raw: str) -> str:
    """Remove ```json / ``` wrappers from model output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


# ---------------------------------------------------------------------------
# 5. Parse raw model output into dict
# ---------------------------------------------------------------------------

def _repair_truncated_json(text: str) -> Optional[dict]:
    """Attempt to repair JSON truncated mid-value (e.g. by token limit).

    Common case: the model ran out of tokens while writing a long free_text
    string, so the JSON has an unclosed string/object. Strategy:
      1. Find complete key-value pairs already present
      2. Truncate at the last complete pair
      3. Close the object
    """
    # Find the opening brace
    start = text.find("{")
    if start < 0:
        return None

    fragment = text[start:]

    # Try progressively truncating from the end to find valid JSON
    # Look for the last complete "key": value pair
    # Strategy: find last comma that gives valid JSON when we close after it
    for i in range(len(fragment) - 1, 0, -1):
        ch = fragment[i]
        if ch in (',', '{'):
            candidate = fragment[:i] + ('' if ch == '{' else '') + '}'
            if ch == '{':
                candidate = fragment[:i + 1] + '}'
            else:
                candidate = fragment[:i] + '}'
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


def parse_gatc_response(raw: str) -> Optional[dict]:
    """Parse raw model output string into a GATCRequest dict.

    Handles markdown fences, noisy output, and truncated JSON.
    """
    cleaned = strip_markdown_fences(raw)

    # Fast path: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object
    start = cleaned.find("{")
    if start >= 0:
        end = cleaned.rfind("}")
        if end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    # Last resort: repair truncated JSON (model ran out of tokens)
    repaired = _repair_truncated_json(cleaned)
    if repaired is not None:
        return repaired

    return None


# ---------------------------------------------------------------------------
# Main post-processor
# ---------------------------------------------------------------------------

def postprocess_gatc_request(parsed: dict, user_message: str) -> dict:
    """Apply deterministic fixups to a parsed GATCRequest.

    Args:
        parsed: The parsed JSON dict from the interpreter model.
        user_message: The original user message (including CONTEXT block).

    Returns:
        A new dict with fixups applied. The original is not mutated.
    """
    result = dict(parsed)
    action = result.get("action", "")

    # --- Fix 1: Extract time_available_min when missing ---
    if action == "create_workout" and "time_available_min" not in result:
        # Try free_text first, then fall back to user message
        source = result.get("free_text", "") or user_message
        duration = _extract_duration(source)
        if duration is None:
            # Also check the full user message if free_text didn't have it
            duration = _extract_duration(user_message)
        if duration is not None:
            result["time_available_min"] = duration

    # --- Fix 2: Coerce time_available_min to int ---
    if "time_available_min" in result:
        try:
            result["time_available_min"] = int(result["time_available_min"])
        except (ValueError, TypeError):
            del result["time_available_min"]

    # --- Fix 3: Enforce clarify when create_workout has no sport context ---
    if action == "create_workout":
        sport = result.get("sport", "")
        has_sport_ctx = _has_sport_context(user_message)
        has_any_ctx = _has_any_context(user_message)

        # Case A: sport="other" or missing sport, and no CONTEXT with Sport → clarify
        # Case B: model hallucinated a sport but there's NO CONTEXT block at all
        #         AND user didn't mention the sport in their message
        needs_clarify = False
        if (sport == "other" or not sport) and not has_sport_ctx:
            needs_clarify = True
        elif sport and not has_any_ctx:
            # No CONTEXT block — did the user actually mention this sport?
            msg_lower = user_message.split("\n\nCONTEXT:")[0].strip().lower()
            _SPORT_KEYWORDS = {
                "run": ["run", "running", "jog", "jogging"],
                "bike": ["bike", "cycling", "cycle", "ride", "biking"],
                "ski": ["ski", "skiing"],
                "skate": ["skate", "skating"],
                "strength": ["strength", "gym", "weights", "lifting", "lift"],
            }
            sport_mentioned = False
            for keywords in _SPORT_KEYWORDS.values():
                if any(kw in msg_lower for kw in keywords):
                    sport_mentioned = True
                    break
            if not sport_mentioned:
                needs_clarify = True

        if needs_clarify:
            result["action"] = "clarify"
            result["missing"] = ["sport"]
            result["clarify_message"] = (
                "What sport would you like to train? "
                "Running, cycling, strength, or something else?"
            )
            # Remove create_workout-specific fields
            result.pop("sport", None)
            result.pop("time_available_min", None)
            result.pop("goal", None)
            result.pop("constraints", None)

    # --- Fix 4: Enforce clarify for ambiguous / ultra-short messages ---
    if action in ("explain", "answer_question"):
        msg = user_message.split("\n\nCONTEXT:")[0].strip()
        msg_lower = msg.lower()
        msg_words = msg_lower.split()

        # Ultra-short acknowledgements that are not real questions
        _ACK_WORDS = {
            "ok", "okay", "k", "sure", "yes", "yeah", "yep", "alright",
            "right", "cool", "fine", "thanks", "thank you", "cheers",
            "so", "so?", "and?", "then?", "hm", "hmm",
        }
        is_ack = msg_lower.rstrip("?!. ") in _ACK_WORDS

        # Short follow-ups like "why?", "how?", "what?" with no TOPIC context
        is_bare_question = (
            len(msg_words) <= 2
            and msg.rstrip().endswith("?")
            and "\nTOPIC:" not in user_message  # no topic hint injected
            and "\nHISTORY:" not in user_message  # no conversation history
        )

        # Ambiguous: short non-question without context
        has_ctx = _has_any_context(user_message)
        is_ambiguous = len(msg_words) <= 4 and "?" not in msg and not has_ctx

        if is_ack or is_bare_question or is_ambiguous:
            result["action"] = "clarify"
            result["missing"] = ["intent"]
            result["clarify_message"] = (
                "What would you like help with? "
                "A workout, explaining your plan, or a training question?"
            )
            result.pop("question", None)

    # --- Fix 5: Medical red-flag override ---
    # Persistent pain, cardiac symptoms, dizziness → clarify with medical safety
    # Distinct from common illness (cold, flu, sick) which stays as replan:illness
    if _has_medical_red_flag(user_message):
        result["action"] = "clarify"
        result["missing"] = ["medical_clearance"]
        result["clarify_message"] = (
            "Please stop training and consult a medical professional immediately."
        )
        # Remove fields that don't belong on a medical clarify
        result.pop("replan_type", None)
        result.pop("sport", None)
        result.pop("question", None)

    # --- Fix 6: Validate required fields per action ---
    action = result.get("action", "")  # re-read after possible overrides
    if action in ("answer_question", "explain"):
        question = result.get("question", "")
        if not question or not question.strip():
            # Model produced answer_question/explain with no question —
            # backfill from the user's actual message
            msg = user_message.split("\n\nCONTEXT:")[0].strip()
            if msg:
                result["question"] = msg

    # --- Fix 7: Ensure free_text is always present and valid ---
    msg = user_message.split("\n\nCONTEXT:")[0].strip()
    if msg:
        free = result.get("free_text", "")
        # Backfill if: missing, empty, is an action name, or suspiciously
        # short vs the actual message (model sometimes puts a single word)
        _ACTION_NAMES = {"answer_question", "clarify", "explain", "replan", "create_workout"}
        needs_fix = (
            not free
            or free.lower() in _ACTION_NAMES
            or (len(free.split()) <= 2 and len(msg.split()) > 3)
        )
        if needs_fix:
            result["free_text"] = msg

    return result
