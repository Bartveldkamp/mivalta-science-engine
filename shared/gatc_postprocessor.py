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
    r'\bpain\s+in\s+my\s+(?:chest|heart|shin|knee|back|shoulder|hip|ankle)\b',
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
# 4. Parse raw model output into dict
# ---------------------------------------------------------------------------

def parse_gatc_response(raw: str) -> Optional[dict]:
    """Parse raw model output string into a GATCRequest dict.

    Handles markdown fences and noisy output.
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

        # sport="other" or missing sport, and no CONTEXT with Sport → clarify
        if (sport == "other" or not sport) and not has_sport_ctx:
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

    # --- Fix 4: Enforce clarify for ambiguous intent without context ---
    if action in ("explain", "answer_question") and not _has_any_context(user_message):
        free = result.get("free_text", "").lower().strip()
        # Very short, ambiguous messages that aren't real questions
        ambiguous = len(free.split()) <= 4 and "?" not in free
        if ambiguous:
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

    # --- Fix 6: Ensure free_text is always present ---
    if "free_text" not in result or not result.get("free_text"):
        # Extract from user message (strip CONTEXT block)
        msg = user_message.split("\n\nCONTEXT:")[0].strip()
        if msg:
            result["free_text"] = msg

    return result
