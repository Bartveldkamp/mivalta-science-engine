"""
MiValta Dialogue Governor — Answer-First, Max 1 Follow-Up

Post-processing layer that enforces Josi's dialogue rules on LLM output:

  1. Answer-first: The response must lead with substantive content before
     any question. Responses that open with a question are restructured.

  2. Max 1 follow-up question per turn: If the model produces multiple
     questions, only the most relevant one is kept.

  3. Question quality: Only asks when the answer genuinely depends on
     the athlete's input. Trivial or rhetorical questions are stripped.

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
]

# Compiled for performance
_RHETORICAL_RE = [re.compile(p, re.IGNORECASE) for p in RHETORICAL_PATTERNS]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving question marks."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


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

    for s in sentences:
        if _is_question(s):
            if _is_rhetorical(s):
                continue  # Always strip rhetorical
            questions.append(s)
        else:
            non_questions.append(s)

    if len(questions) <= 1:
        # At most 1 real question — reconstruct without rhetorical ones
        parts = non_questions + questions
        return " ".join(parts) if parts else message

    # Multiple questions — keep only the best one
    scored = [(q, _score_question_relevance(q)) for q in questions]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_question = scored[0][0]

    # Reconstruct: statements first, then the one question at the end
    parts = non_questions + [best_question]
    return " ".join(parts)


def govern_dialogue(intent: dict) -> dict:
    """Apply dialogue governor rules to a validated LLMIntent.

    Modifies the 'message' field in-place and returns the intent.

    Rules applied:
      1. Answer-first: restructure if response opens with a question
      2. Max 1 follow-up: strip extra questions, keep most relevant
      3. Strip rhetorical filler questions

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

    # Apply rules
    message = enforce_answer_first(message)
    message = enforce_max_one_question(message)

    intent["message"] = message
    return intent


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    test_cases = [
        # Good: answer-first with 1 question
        (
            "Today you have a 60-minute Zone 2 run. Nice and easy, building your aerobic base. How are you feeling?",
            "Should keep as-is (answer-first, 1 question)",
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
