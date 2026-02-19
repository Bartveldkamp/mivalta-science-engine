"""
MiValta Dialogue Governor — Answer-First, Minimal Questions

Post-processing layer that enforces Josi's dialogue rules on LLM output:

  1. Answer-first: The response must lead with substantive content before
     any question. Responses that open with a question are restructured.

  2. Max 1 follow-up question per turn: If the model produces multiple
     questions, only the most relevant one is kept.

  3. Question quality: Only asks when the answer genuinely depends on
     the athlete's input. Rhetorical, filler, and "offer to continue"
     questions are stripped.

  4. Zero-question preference: When the response has enough substantive
     content (3+ sentences of statements), trailing questions are stripped
     unless they score high enough to genuinely need athlete input.

This runs AFTER llm_intent_parser.py extracts valid JSON, operating on
the "message" field of the LLMIntent.

Usage:
    from shared.dialogue_governor import govern_dialogue

    intent = parse_llm_intent(raw)
    if intent is not None:
        intent = govern_dialogue(intent)
"""

import re
from typing import Optional


# Patterns that indicate a rhetorical or filler question (not genuinely
# seeking athlete input — these get stripped)
RHETORICAL_PATTERNS = [
    # Direct rhetorical closers
    r"^sound good\??$",
    r"^does that make sense\??$",
    r"^make sense\??$",
    r"^right\??$",
    r"^okay\??$",
    r"^got it\??$",
    r"^ready to go\??$",
    r"^ready\??$",
    r"^shall we\??$",
    r"^what do you think\??$",
    r"^how does that sound\??$",
    r"^fair enough\??$",
    r"^does that help\??$",
    r"^anything else\??$",
    r"^any questions\??$",
    r"^clear\??$",
    # "Would you like me to..." offer-to-continue patterns
    r"^would you like me to (?:elaborate|explain|go into|break that down|walk you through).*\??$",
    r"^want me to (?:elaborate|explain|go into|break that down|walk you through).*\??$",
    r"^would you like to (?:know|hear|learn) more.*\??$",
    r"^want to (?:know|hear|learn) more.*\??$",
    r"^shall i (?:elaborate|explain|go into).*\??$",
    r"^need me to (?:elaborate|explain|break that down).*\??$",
    # "Can I help with..." patterns
    r"^(?:can|may) i help (?:you )?with anything else\??$",
    r"^is there anything else.*\??$",
]

# Compiled for performance
_RHETORICAL_RE = [re.compile(p, re.IGNORECASE) for p in RHETORICAL_PATTERNS]

# Zone/abbreviation patterns that should NOT trigger sentence splits
# e.g. "Z2." should not be treated as end of sentence
_ABBREV_RE = re.compile(r'\bZ\d+\.')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving question marks.

    Handles zone abbreviations (Z2., Z3.) that should not cause splits.
    """
    # Temporarily protect zone abbreviations from splitting
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace('.', '\x00'), text.strip())
    # Split on sentence-ending punctuation followed by space
    parts = re.split(r'(?<=[.!?])\s+', protected)
    # Restore protected dots
    return [p.strip().replace('\x00', '.') for p in parts if p.strip()]


def _is_question(sentence: str) -> bool:
    """Check if a sentence is a question."""
    return sentence.rstrip().endswith("?")


def _is_rhetorical(sentence: str) -> bool:
    """Check if a question is rhetorical/filler (not seeking real input)."""
    cleaned = sentence.strip()
    return any(r.match(cleaned) for r in _RHETORICAL_RE)


def _score_question_relevance(question: str) -> int:
    """Score how relevant a follow-up question is (higher = better).

    Prefers questions that:
    - Ask about the athlete's state or feelings (high value for coaching)
    - Seek specific information needed for a decision
    - Are not yes/no questions (open-ended preferred)
    """
    score = 0
    q_lower = question.lower()

    # High-value: asks about athlete state
    if any(w in q_lower for w in ["how are you", "how do you feel", "how's your",
                                   "what's your", "how did", "how was"]):
        score += 3

    # High-value: seeks specific info
    if any(w in q_lower for w in ["what time", "which day", "what goal",
                                   "how many", "how long", "what kind"]):
        score += 3

    # Medium-value: open-ended
    if any(w_start in q_lower for w_start in ["what ", "how ", "why ", "when "]):
        score += 2

    # Lower-value: yes/no questions
    if any(q_lower.startswith(w) for w in ["do you", "are you", "can you",
                                            "would you", "is there", "did you",
                                            "have you", "should "]):
        score += 1

    # Penalty: rhetorical
    if _is_rhetorical(question):
        score -= 5

    return score


def enforce_answer_first(message: str) -> str:
    """Ensure the message leads with substantive content, not a question.

    If the first sentence is a question, move it after the first statement.
    """
    sentences = _split_sentences(message)
    if len(sentences) <= 1:
        return message

    # Check if the response opens with a question
    if not _is_question(sentences[0]):
        return message  # Already answer-first

    # Find the first non-question sentence
    first_statement_idx = None
    for i, s in enumerate(sentences):
        if not _is_question(s):
            first_statement_idx = i
            break

    if first_statement_idx is None:
        # All sentences are questions — keep as-is (edge case)
        return message

    # Restructure: move the first statement to the front
    opening_question = sentences[0]
    statement = sentences[first_statement_idx]

    # Build restructured: statement first, then remaining in order
    reordered = [statement]
    for i, s in enumerate(sentences):
        if i == 0 or i == first_statement_idx:
            continue
        reordered.append(s)
    reordered.append(opening_question)

    return " ".join(reordered)


def enforce_max_one_question(message: str) -> str:
    """Keep at most 1 follow-up question in the message.

    If multiple questions exist, keep the highest-scoring one and strip
    the rest. Rhetorical questions are always stripped.
    """
    sentences = _split_sentences(message)
    if not sentences:
        return message

    questions = []
    non_questions = []
    # Track original positions to preserve interleaving when possible
    tagged = []  # (sentence, is_question, is_rhetorical)

    for s in sentences:
        if _is_question(s):
            if _is_rhetorical(s):
                tagged.append((s, True, True))
            else:
                questions.append(s)
                tagged.append((s, True, False))
        else:
            non_questions.append(s)
            tagged.append((s, False, False))

    if len(questions) <= 1:
        # At most 1 real question — reconstruct without rhetorical ones,
        # preserving original order
        parts = [s for s, is_q, is_rhet in tagged if not is_rhet]
        return " ".join(parts) if parts else message

    # Multiple questions — keep only the best one
    scored = [(q, _score_question_relevance(q)) for q in questions]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_question = scored[0][0]

    # Reconstruct: keep original order, but only the best question survives
    parts = []
    for s, is_q, is_rhet in tagged:
        if is_rhet:
            continue
        if is_q and s != best_question:
            continue
        parts.append(s)
    return " ".join(parts) if parts else message


_MIN_QUESTION_SCORE = 3  # Minimum score for a question to survive zero-question preference


def enforce_zero_question_preference(message: str) -> str:
    """Strip trailing questions when the response is already substantive.

    The system prompt says "Most responses need zero questions." This rule
    enforces that preference: if the response has 3+ statement sentences,
    trailing questions are stripped unless they score high enough (genuinely
    need athlete input to proceed).
    """
    sentences = _split_sentences(message)
    if not sentences:
        return message

    non_questions = [s for s in sentences if not _is_question(s)]

    # Only apply zero-question preference when there's enough substance
    if len(non_questions) < 2:
        return message

    # Check if the last sentence is a question
    if not _is_question(sentences[-1]):
        return message

    # Score the trailing question — keep it only if it's high-value
    trailing_q = sentences[-1]
    score = _score_question_relevance(trailing_q)
    if score >= _MIN_QUESTION_SCORE:
        return message

    # Strip the low-value trailing question
    parts = [s for s in sentences if s != trailing_q]
    return " ".join(parts) if parts else message


def govern_dialogue(intent: dict) -> dict:
    """Apply dialogue governor rules to a validated LLMIntent.

    Modifies the 'message' field in-place and returns the intent.

    Rules applied:
      1. Answer-first: restructure if response opens with a question
      2. Max 1 follow-up: strip extra questions, keep most relevant
      3. Strip rhetorical filler questions
      4. Zero-question preference: strip low-value trailing questions
         when the response is already substantive

    Args:
        intent: A validated LLMIntent dict (from parse_llm_intent).

    Returns:
        The same dict with governed 'message' field.
    """
    if intent is None:
        return intent

    message = intent.get("message", "")
    if not message:
        return intent

    # Don't govern safety warnings or decline messages — they need to be direct
    response_type = intent.get("response_type", "")
    if response_type in ("SafetyWarning", "Decline"):
        return intent

    # Apply rules in order
    message = enforce_answer_first(message)
    message = enforce_max_one_question(message)
    message = enforce_zero_question_preference(message)

    intent["message"] = message
    return intent


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    test_cases = [
        # Good: answer-first with high-value question (should keep)
        (
            "Today you have a 60-minute Zone 2 run. Nice and easy, building your aerobic base. How are you feeling?",
            "Should keep as-is (answer-first, high-value question)",
        ),
        # Bad: opens with question
        (
            "How are you feeling today? You have a 60-minute Zone 2 session planned. It's a nice easy aerobic run.",
            "Should restructure — statement first",
        ),
        # Bad: multiple questions
        (
            "Your readiness is Yellow today. That means we should take it easier. How did you sleep? Are you feeling any muscle soreness? What's your energy level?",
            "Should keep only 1 question",
        ),
        # Bad: rhetorical filler
        (
            "You have a recovery day today. Light stretching or a short walk would be perfect. Sound good?",
            "Should strip rhetorical question",
        ),
        # Bad: "would you like me to elaborate" filler
        (
            "Your plan is set based on your current fitness. Would you like me to elaborate on today's prescription?",
            "Should strip offer-to-continue question",
        ),
        # Bad: trailing low-value yes/no with substantive content
        (
            "Your readiness is Yellow today. That means we should ease up. A lighter session will help you bounce back stronger. Do you want to proceed?",
            "Should strip low-value trailing question (substantive response)",
        ),
        # Edge: all questions
        (
            "How are you feeling? Did you sleep well? Any soreness?",
            "Should keep only 1 question",
        ),
        # Good: no questions
        (
            "Great session yesterday. Your body needs recovery now. Take it easy today.",
            "Should keep as-is",
        ),
        # Edge: zone abbreviation shouldn't split
        (
            "Today is a Z2. Nice and easy, just keep it conversational.",
            "Should keep as-is (Z2. not a sentence break)",
        ),
    ]

    for msg, description in test_cases:
        intent = {"message": msg, "response_type": "QuestionAnswer"}
        governed = govern_dialogue(intent)
        result = governed["message"]

        questions_before = msg.count("?")
        questions_after = result.count("?")

        print(f"\n  [{description}]")
        print(f"    Before ({questions_before}?): {msg[:80]}...")
        print(f"    After  ({questions_after}?): {result[:80]}...")
        status = "OK" if questions_after <= 1 else "FAIL"
        print(f"    Status: {status}")

    print()
