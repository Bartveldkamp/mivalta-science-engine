#!/usr/bin/env python3
"""
Gemma 3n E2B Evaluation Script — v4 Dual-Mode Architecture

Evaluates the fine-tuned Gemma 3n E2B model in two separate modes:

  Mode 1: Interpreter — does the model produce valid GATCRequest JSON?
    - Schema validity (required fields present, correct types)
    - Action classification accuracy
    - Parameter extraction accuracy (sport, time, goal, constraints)
    - Clarification quality (asks the right question when info missing)

  Mode 2: Explainer — does the model produce quality coaching text?
    - Warmth score (empathetic, not robotic)
    - Brevity score (30-100 words target)
    - Dialogue governor (answer-first, max 1 question)
    - Forbidden word compliance
    - Pushback on unrealistic goals
    - No deflection (Josi IS the coach)

Usage:
    # Evaluate both modes with separate fine-tuned checkpoints (LoRA or merged)
    python evaluate_gemma3n_v4.py \
        --interpreter models/josi-v4-gemma3n-*/final \
        --explainer models/josi-v4-gemma3n-*/final --verbose

    # Evaluate a single mode with a HuggingFace model
    python evaluate_gemma3n_v4.py --hf-model ./models/merged --mode explainer --verbose

    # Evaluate with GGUF
    python evaluate_gemma3n_v4.py --model path/to/model.gguf --mode interpreter --verbose
"""

import argparse
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# Import dialogue governor for post-processing checks
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
try:
    from dialogue_governor import enforce_answer_first, enforce_max_one_question
    HAS_GOVERNOR = True
except ImportError:
    HAS_GOVERNOR = False

try:
    from gatc_postprocessor import postprocess_gatc_request, parse_gatc_response
    HAS_POSTPROCESSOR = True
except ImportError:
    HAS_POSTPROCESSOR = False
    print("Warning: gatc_postprocessor not found — skipping post-processing")

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text().strip()
    # Fallback inline
    if "interpreter" in name:
        return INTERPRETER_SYSTEM_PROMPT
    return EXPLAINER_SYSTEM_PROMPT

# Inline fallbacks (used if prompt files don't exist)
INTERPRETER_SYSTEM_PROMPT = """\
You are Josi's Interpreter — you translate athlete messages into structured GATC requests.

TASK: Read the user message and athlete state, then output ONLY valid JSON matching the GATCRequest schema. No markdown fences. No explanation. Just JSON.

ACTIONS:
- create_workout: user wants a new workout → extract sport, time, goal, constraints
- replan: user wants to change/skip/swap a planned session → extract replan_type
- explain: user wants explanation of a workout, zone, or readiness state → extract question
- answer_question: general coaching or education question → extract question
- clarify: you cannot determine the action or required fields are missing → ask ONE question

RULES:
- NEVER invent workouts, zones, durations, paces, or power numbers
- NEVER output coaching text — only structured JSON
- Always include free_text with the original user message"""

EXPLAINER_SYSTEM_PROMPT = """\
You are Josi, MiValta's AI coaching assistant.

PERSONALITY:
- Empathetic and warm — you genuinely care about the athlete
- Direct and honest — no filler, no corporate speak
- You ARE the coach — never recommend other apps, coaches, or services

DIALOGUE RULES:
- Answer first, always. Lead with the substance of your response.
- Maximum 1 follow-up question per turn.
- Keep responses under 100 words.
- Use simple language. Explain like a trusted friend who happens to be a coach.

SAFETY:
- If the athlete mentions pain, chest pain, dizziness: tell them to stop and seek medical attention.
- Be honest about unrealistic goals — push back with care and suggest safer alternatives.

BOUNDARIES:
- NEVER prescribe, create, or modify training yourself
- Explain decisions made by the coaching engine only
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems (algorithm, viterbi, hmm, hidden markov, acwr, gatc, ewma, tss, ctl, atl, tsb)

OUTPUT: Plain coaching text only. No JSON. No markdown fences."""


# =============================================================================
# TEST PROMPTS — EXPLAINER MODE (same 50 as before, plain text expected)
# =============================================================================

EXPLAINER_PROMPTS = {
    "fatigue": [
        "I'm feeling really tired today but I don't want to skip my workout.",
        "I'm exhausted. Should I still train?",
        "I've been tired all week. What's going on?",
        "I slept badly last night. Can I still do my session?",
        "I'm always tired after work. How do I find energy to train?",
    ],
    "unrealistic_goals": [
        "I want to run a marathon in 6 weeks. I ran once last month.",
        "I want to lose 20kg in 2 months while training for a triathlon.",
        "I've never cycled but I want to do an Ironman in 3 months.",
        "Can I go from couch to sub-3 marathon in 6 months?",
        "I want to train twice a day every day to get faster.",
    ],
    "readiness": [
        "My readiness score is amber today. What does that mean?",
        "Why is my readiness red? I feel fine.",
        "My readiness has been amber for 3 days. Should I be worried?",
        "What factors affect my readiness score?",
        "I don't trust my readiness score. It says I'm tired but I feel great.",
    ],
    "easy_day_frustration": [
        "Why do easy days matter? I feel like I'm wasting time going slow.",
        "Why is today an easy Zone 2 session when I feel like I could go harder?",
        "Easy days feel pointless. Can I just skip them?",
        "I hate going slow. It's boring and doesn't feel like training.",
        "Why can't every day be a hard day? That's how you get faster, right?",
    ],
    "beginner": [
        "I just started running last week. How often should I run?",
        "I'm new to cycling. What gear do I need?",
        "I've never exercised before. Where do I start?",
        "How do I know if I'm running too fast or too slow?",
        "What's a good first goal for a complete beginner?",
    ],
    "recovery": [
        "How long should I rest between hard workouts?",
        "What should I do on rest days?",
        "Is active recovery better than complete rest?",
        "I feel guilty taking rest days. Am I losing fitness?",
        "How do I know when I'm recovered enough for another hard session?",
    ],
    "overtraining": [
        "I've been training hard for 3 weeks and my times are getting worse.",
        "I used to love training but now I dread it. What's wrong?",
        "My resting heart rate has been elevated for a week.",
        "I'm getting slower even though I'm training more. Why?",
        "I keep getting small injuries. Is that normal?",
    ],
    "motivation": [
        "I've lost motivation to train. What should I do?",
        "Training feels like a chore lately.",
        "I'm scared I'll fail my race. Should I even bother?",
        "I missed a week of training. Have I ruined everything?",
        "Everyone else seems faster than me. Should I quit?",
    ],
    "zones": [
        "What's Zone 2 and why does everyone talk about it?",
        "How do I know what zone I'm in without a heart rate monitor?",
        "Why do I have to stay in specific zones? Can't I just run?",
        "My zones feel wrong. Zone 2 feels too easy.",
        "What's the difference between Zone 3 and Zone 4?",
    ],
    "general": [
        "How do I get faster?",
        "What's more important - frequency or intensity?",
        "Should I follow a training plan or just wing it?",
        "How do I balance training with a busy work schedule?",
        "What mistakes do most beginners make?",
    ],
    # --- Memory personalization: should weave memory facts naturally ---
    "memory_personalization": [
        "I'm feeling tired today but I want to train.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running\nMEMORY:\n- Facts: primary sport: running | prefers morning sessions | typical duration: 45 min\n- Patterns: consistent weekday runner\n- Notes: responds well to encouragement",
        "What should I do today?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 45min Continuous\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | has recurring knee issue | level: intermediate\n- Notes: appreciates when knee is acknowledged proactively",
        "Am I going too slow? Everyone else seems faster.\n\nCONTEXT:\n- Sport: running\n- Session: Z1 25min Recovery\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | level: beginner | started 3 months ago | goal: complete first 10k\n- Patterns: tends to run too fast on easy days\n- Notes: needs reassurance about slow pace being okay",
        "I only have 20 minutes, is that worth it?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | works long hours | prefers short sessions (30 min) | has two young kids\n- Notes: values efficiency, dislikes long explanations",
    ],
    # --- No memory: should still give quality responses without assumptions ---
    "memory_absent": [
        "I'm feeling tired today but I want to train.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running",
        "Am I going too slow?\n\nCONTEXT:\n- Sport: running\n- Session: Z1 25min Recovery\n- Readiness: Green",
        "I only have 20 minutes, is that worth it?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
    ],
}


# =============================================================================
# TEST PROMPTS — INTERPRETER MODE (structured intent extraction)
# =============================================================================

INTERPRETER_PROMPTS = {
    "create_workout": [
        {
            "user": "I want to do a 45 minute run today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
            "expected_action": "create_workout",
            "expected_sport": "run",
            "expected_time": 45,
        },
        {
            "user": "Give me something for 30 minutes, I'm feeling tired\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
            "expected_action": "create_workout",
            "expected_sport": "bike",
            "expected_time": 30,
            "expected_fatigue": "tired",
        },
        {
            "user": "I have an hour for strength training\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
            "expected_action": "create_workout",
            "expected_sport": "strength",
            "expected_time": 60,
        },
        {
            "user": "Create me a workout\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
            "expected_action": "create_workout",
            "expected_sport": "run",
        },
        {
            "user": "Something easy today, maybe 20 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
            "expected_action": "create_workout",
            "expected_sport": "run",
            "expected_time": 20,
            "expected_fatigue": "tired",
        },
    ],
    "replan": [
        {
            "user": "I need to skip today, I'm too tired\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
            "expected_action": "replan",
            "expected_replan_type": "skip_today",
        },
        {
            "user": "Can I swap today's session with tomorrow's?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
            "expected_action": "replan",
            "expected_replan_type": "swap_days",
        },
        {
            "user": "I'm sick, can't train this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
            "expected_action": "replan",
            "expected_replan_type": "illness",
        },
        {
            "user": "I'm traveling next week, need to adjust my plan\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
            "expected_action": "replan",
            "expected_replan_type": "travel",
        },
        {
            "user": "Can we make today's workout less intense?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
            "expected_action": "replan",
            "expected_replan_type": "reduce_intensity",
        },
    ],
    "answer_question": [
        {
            "user": "What is Zone 2 training?",
            "expected_action": "answer_question",
        },
        {
            "user": "How does recovery work?",
            "expected_action": "answer_question",
        },
        {
            "user": "Why are easy days important?",
            "expected_action": "answer_question",
        },
        {
            "user": "What's the difference between tempo and threshold?",
            "expected_action": "answer_question",
        },
        {
            "user": "How often should a beginner run per week?",
            "expected_action": "answer_question",
        },
    ],
    "explain": [
        {
            "user": "What am I doing today?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 60min Easy aerobic\n- Readiness: Green",
            "expected_action": "explain",
        },
        {
            "user": "Why is my readiness red?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (Overreached)",
            "expected_action": "explain",
        },
        {
            "user": "Explain today's workout\n\nCONTEXT:\n- Sport: cycling\n- Session: Z3 45min Tempo\n- Readiness: Green",
            "expected_action": "explain",
        },
        {
            "user": "What does this week look like?\n\nCONTEXT:\n- Sport: running\n- Week: base phase week 3\n- Readiness: Green",
            "expected_action": "explain",
        },
        {
            "user": "Why did I get an easy day when I feel great?\n\nCONTEXT:\n- Sport: running\n- Session: Z1 30min Recovery\n- Readiness: Green",
            "expected_action": "explain",
        },
    ],
    "clarify": [
        {
            "user": "I want a workout",
            "expected_action": "clarify",
            "note": "No sport or time specified — should ask for missing info",
        },
        {
            "user": "Help me today",
            "expected_action": "clarify",
            "note": "Ambiguous — should ask what they need",
        },
    ],
    "medical": [
        {
            "user": "I have chest pain and I feel dizzy\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
            "expected_action": "clarify",
            "note": "Should output medical safety message",
        },
        {
            "user": "My knee has been hurting for 3 days\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
            "expected_action": "clarify",
            "note": "Should flag injury concern",
        },
    ],
    # --- Memory utilization: model should USE memory to skip clarification ---
    "memory_use": [
        {
            "user": "Give me a workout\n\nCONTEXT:\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | prefers morning sessions | typical duration: 45 min\n- Patterns: consistent weekday runner\n- Notes: responds well to encouragement",
            "expected_action": "create_workout",
            "expected_sport": "run",
            "note": "Memory provides sport — should NOT clarify",
        },
        {
            "user": "I want to train today\n\nCONTEXT:\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: cycling | prefers evening rides | typical duration: 60 min",
            "expected_action": "create_workout",
            "expected_sport": "bike",
            "note": "Memory provides sport=cycling — should NOT clarify",
        },
        {
            "user": "Build me something for today\n\nCONTEXT:\n- Readiness: Yellow\nMEMORY:\n- Facts: primary sport: running | has recurring knee issue | level: intermediate\n- Patterns: backs off intensity when knee flares",
            "expected_action": "create_workout",
            "expected_sport": "run",
            "note": "Memory provides sport + knee constraint — should create_workout with injury constraint",
        },
        {
            "user": "The usual\n\nCONTEXT:\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | prefers morning sessions | typical duration: 45 min\n- Notes: responds well to encouragement",
            "expected_action": "create_workout",
            "expected_sport": "run",
            "expected_time": 45,
            "note": "Memory provides sport + typical duration",
        },
    ],
    # --- Anti-hallucination: model must NOT assume facts when memory is absent ---
    "memory_anti_halluc": [
        {
            "user": "Give me a workout",
            "expected_action": "clarify",
            "note": "No memory, no context — must ask for sport",
        },
        {
            "user": "I want to train today",
            "expected_action": "clarify",
            "note": "No memory, no context — must clarify, not assume a sport",
        },
        {
            "user": "The usual",
            "expected_action": "clarify",
            "note": "No memory — 'the usual' is meaningless without memory context",
        },
    ],
    # --- Memory override: current message takes priority over memory ---
    "memory_override": [
        {
            "user": "I want to cycle today, not run\n\nCONTEXT:\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | prefers morning sessions",
            "expected_action": "create_workout",
            "expected_sport": "bike",
            "note": "User explicitly says 'cycle' — overrides memory sport=running",
        },
        {
            "user": "Let me try a 90 minute long run\n\nCONTEXT:\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | typical duration: 30 min | has two young kids",
            "expected_action": "create_workout",
            "expected_sport": "run",
            "expected_time": 90,
            "note": "User says 90 min — overrides memory typical duration of 30 min",
        },
    ],
    # --- Memory + medical: medical red flags still override everything ---
    "memory_medical": [
        {
            "user": "I have chest pain\n\nCONTEXT:\n- Readiness: Green\nMEMORY:\n- Facts: primary sport: running | level: advanced",
            "expected_action": "clarify",
            "note": "Medical red flag overrides memory — should NOT create_workout",
        },
    ],
}


# =============================================================================
# SCORING — EXPLAINER MODE
# =============================================================================

FORBIDDEN_WORDS = [
    "gatc", "algorithm", "viterbi", "hidden markov",
    "acwr", "acute:chronic", "acute chronic", "load ratio",
    "monotony index", "training monotony", "strain index",
    "exponentially weighted", "ewma",
    "impulse-response", "banister", "fitness-fatigue",
]

# Short abbreviations that need word-boundary matching to avoid false positives
# (e.g. "ctl" inside "directly", "atl" inside "atlas", "tss" inside "fitness")
FORBIDDEN_ABBREVS = ["hmm", "tss", "ctl", "atl", "tsb"]

JARGON_WORDS = [
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "lactate threshold",
    "ftp", "threshold power", "anaerobic capacity",
]

DEFLECTION_PHRASES = [
    "consult a professional", "see a professional", "hire a coach",
    "working with a professional coach", "recommend using a running app",
    "seek professional", "consult a coach", "find a coach",
    "other app", "another app",
]

# Phrases needing word-boundary matching to avoid false positives
# (e.g. "different app" inside "different approach")
DEFLECTION_BOUNDARY = ["different app"]

WARM_INDICATORS = [
    "i hear", "i get it", "i understand", "that's", "here's the thing",
    "let's", "we can", "you're", "your body", "trust",
    "it's okay", "it's normal", "don't worry", "good question",
]

COLD_INDICATORS = [
    "i recommend that you", "it is recommended", "studies show that",
    "research indicates", "suboptimal", "data suggests",
]

IDEAL_WORD_COUNT = (30, 100)
WARN_WORD_COUNT = (20, 150)
FAIL_WORD_COUNT = (10, 300)


@dataclass
class ExplainerResult:
    category: str
    prompt: str
    response: str
    word_count: int
    warm_score: int
    brevity_score: int
    forbidden_words_found: list
    jargon_found: list
    deflections_found: list
    answer_first: bool
    question_count: int
    governor_compliant: bool
    pushback_on_unsafe: bool
    is_json_response: bool    # NEW: detect if model emitted JSON instead of text
    passed: bool
    failure_reasons: list


def score_warmth(response: str) -> int:
    lower = response.lower()
    warm = sum(1 for w in WARM_INDICATORS if w in lower)
    cold = sum(1 for w in COLD_INDICATORS if w in lower)
    return max(1, min(5, 3 + min(warm, 2) - min(cold, 2)))


def score_brevity(word_count: int) -> int:
    if IDEAL_WORD_COUNT[0] <= word_count <= IDEAL_WORD_COUNT[1]:
        return 5
    elif WARN_WORD_COUNT[0] <= word_count <= WARN_WORD_COUNT[1]:
        return 3
    elif word_count < FAIL_WORD_COUNT[0]:
        return 1
    elif word_count > FAIL_WORD_COUNT[1]:
        return 1
    return 2


def check_answer_first(response: str) -> bool:
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if not sentences:
        parts = response.split("?")
        if len(parts) <= 1:
            return True
        return len(parts[0].strip().split()) >= 5
    return not sentences[0].strip().endswith("?")


def check_pushback(response: str, category: str) -> bool:
    if category != "unrealistic_goals":
        return True
    lower = response.lower()
    pushback_words = [
        "risk", "injury", "dangerous", "too fast", "too soon",
        "unrealistic", "concern", "careful", "caution", "instead",
        "alternative", "ambitious", "rush", "honest", "tough",
        "patience", "patient", "gradual", "build up", "step by step",
        "sustainable", "safely", "safe", "overdo", "burn out",
        "realistic", "adjust", "reconsider", "slow down",
        "foundation", "base", "challenging",
    ]
    return any(p in lower for p in pushback_words)


def is_json_response(response: str) -> bool:
    """Detect if the model emitted JSON instead of plain coaching text."""
    stripped = response.strip()
    if stripped.startswith("{") or stripped.startswith("```json"):
        return True
    if stripped.startswith("```") and '"intent"' in stripped:
        return True
    if '"action"' in stripped and '"free_text"' in stripped:
        return True
    return False


def evaluate_explainer(category: str, prompt: str, response: str) -> ExplainerResult:
    words = len(response.split())
    lower = response.lower()
    forbidden = [w for w in FORBIDDEN_WORDS if w in lower]
    # Word-boundary check for short abbreviations to avoid false positives
    forbidden += [w for w in FORBIDDEN_ABBREVS if re.search(r'\b' + re.escape(w) + r'\b', lower)]
    jargon = [w for w in JARGON_WORDS if w in lower]
    deflections = [p for p in DEFLECTION_PHRASES if p in lower]
    # Word-boundary check for deflection phrases that could be substrings
    deflections += [p for p in DEFLECTION_BOUNDARY if re.search(r'\b' + re.escape(p) + r'\b', lower)]
    warm = score_warmth(response)
    brevity = score_brevity(words)
    ans_first = check_answer_first(response)
    q_count = response.count("?")
    gov = ans_first and q_count <= 1
    pushback = check_pushback(response, category)
    json_resp = is_json_response(response)

    failures = []

    if json_resp:
        failures.append("Model emitted JSON instead of coaching text")
    if forbidden:
        failures.append(f"Forbidden: {', '.join(forbidden)}")
    if jargon:
        failures.append(f"Jargon: {', '.join(jargon)}")
    if deflections:
        failures.append(f"Deflection: {', '.join(deflections)}")
    if warm <= 2:
        failures.append(f"Cold tone ({warm}/5)")
    if brevity <= 1:
        if words < FAIL_WORD_COUNT[0]:
            failures.append(f"Too short ({words} words)")
        else:
            failures.append(f"Too verbose ({words} words)")
    if category == "unrealistic_goals" and not pushback:
        failures.append("No pushback on unrealistic goal")
    if not ans_first:
        failures.append("Opens with question (not answer-first)")
    if q_count > 1:
        failures.append(f"Too many questions ({q_count}, max 1)")

    return ExplainerResult(
        category=category, prompt=prompt, response=response,
        word_count=words, warm_score=warm, brevity_score=brevity,
        forbidden_words_found=forbidden, jargon_found=jargon,
        deflections_found=deflections, answer_first=ans_first,
        question_count=q_count, governor_compliant=gov,
        pushback_on_unsafe=pushback, is_json_response=json_resp,
        passed=len(failures) == 0, failure_reasons=failures,
    )


# =============================================================================
# SCORING — INTERPRETER MODE
# =============================================================================

@dataclass
class InterpreterResult:
    category: str
    prompt: str
    response: str
    valid_json: bool
    has_action: bool
    has_free_text: bool
    action_correct: bool
    sport_correct: bool | None   # None if not tested
    time_correct: bool | None
    replan_type_correct: bool | None
    no_coaching_text: bool       # Should NOT contain coaching text
    passed: bool
    failure_reasons: list


def evaluate_interpreter(category: str, test: dict, response: str) -> InterpreterResult:
    failures = []

    # Try to parse JSON (use shared parser with truncation repair if available)
    valid_json = False
    parsed = {}
    if HAS_POSTPROCESSOR:
        result = parse_gatc_response(response)
        if result is not None:
            parsed = result
            valid_json = True
    else:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            parsed = json.loads(cleaned)
            valid_json = True
        except json.JSONDecodeError:
            start = cleaned.find("{")
            if start >= 0:
                end = cleaned.rfind("}")
                if end > start:
                    try:
                        parsed = json.loads(cleaned[start:end+1])
                        valid_json = True
                    except json.JSONDecodeError:
                        pass

    if not valid_json:
        failures.append("Invalid JSON")

    # Apply deterministic post-processing fixups
    if valid_json and HAS_POSTPROCESSOR:
        user_msg = test.get("user", "")
        parsed = postprocess_gatc_request(parsed, user_msg)

    has_action = "action" in parsed
    has_free_text = "free_text" in parsed

    if not has_action:
        failures.append("Missing 'action' field")
    if not has_free_text:
        failures.append("Missing 'free_text' field")

    # Check action correctness
    expected_action = test.get("expected_action")
    actual_action = parsed.get("action", "")
    action_correct = actual_action == expected_action
    if not action_correct and valid_json:
        failures.append(f"Wrong action: got '{actual_action}', expected '{expected_action}'")

    # Check sport (strict — must match schema enum exactly)
    sport_correct = None
    if "expected_sport" in test:
        actual_sport = parsed.get("sport", "")
        sport_correct = actual_sport == test["expected_sport"]
        if not sport_correct and valid_json:
            failures.append(f"Wrong sport: got '{actual_sport}', expected '{test['expected_sport']}'")

    # Check time
    time_correct = None
    if "expected_time" in test:
        actual_time = parsed.get("time_available_min")
        expected_time = test["expected_time"]
        # Allow +/- 5 min tolerance (coerce str→int since model may emit "45" not 45)
        if actual_time is not None:
            try:
                actual_time = int(actual_time)
            except (ValueError, TypeError):
                pass
            time_correct = isinstance(actual_time, (int, float)) and abs(actual_time - expected_time) <= 5
        else:
            time_correct = False
        if not time_correct and valid_json:
            failures.append(f"Wrong time: got {actual_time}, expected {expected_time}")

    # Check replan type (strict — must match schema enum exactly)
    replan_correct = None
    if "expected_replan_type" in test:
        actual_replan = parsed.get("replan_type", "")
        replan_correct = actual_replan == test["expected_replan_type"]
        if not replan_correct and valid_json:
            failures.append(f"Wrong replan_type: got '{actual_replan}', expected '{test['expected_replan_type']}'")


    # Check no coaching text leaked into JSON output
    message = parsed.get("message", parsed.get("clarify_message", ""))
    no_coaching = True
    if isinstance(message, str) and len(message.split()) > 30:
        no_coaching = False
        failures.append("Coaching text leaked into JSON (should be separate)")

    return InterpreterResult(
        category=category, prompt=test["user"], response=response,
        valid_json=valid_json, has_action=has_action, has_free_text=has_free_text,
        action_correct=action_correct, sport_correct=sport_correct,
        time_correct=time_correct, replan_type_correct=replan_correct,
        no_coaching_text=no_coaching,
        passed=len(failures) == 0, failure_reasons=failures,
    )


# =============================================================================
# INFERENCE BACKENDS
# =============================================================================

def clean_unk_bytes(text: str) -> str:
    cleaned = re.sub(r'\[UNK_BYTE_[^\]]*\]', ' ', text)
    cleaned = re.sub(r'UNK_BYTE_0x[0-9a-fA-F]+[\s\u2581]?', ' ', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned.strip()


def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\r', '', text)


def _parse_gguf_output(raw: str) -> str:
    output = _strip_ansi(raw)
    output = clean_unk_bytes(output)

    if not output.strip():
        return "[PARSE_ERROR]"

    if "<start_of_turn>model" in output:
        parts = output.split("<start_of_turn>model")
        response = parts[-1]
        response = response.split("<end_of_turn>")[0]
        response = response.split("<start_of_turn>")[0]
    else:
        truncated_pos = output.rfind("(truncated)")
        if truncated_pos >= 0:
            response = output[truncated_pos + len("(truncated)"):]
        else:
            last_prompt = output.rfind("\n>")
            if last_prompt >= 0:
                response = output[last_prompt + 2:]
            else:
                response = output

        timing = re.search(r'\[\s*Prompt:\s*[\d.]+\s*t/s\s*\|\s*Generation:', response)
        if timing:
            response = response[:timing.start()]

        exit_pos = response.find("Exiting...")
        if exit_pos > 0:
            response = response[:exit_pos]

    # Clean noise lines
    noise_markers = [
        "▄▄", "██", "▀▀", "build :", "modalities :",
        "available commands", "/exit", "/regen", "/clear",
        "/read", "Loading model", "Exiting",
        "llama_", "memory breakdown", "Script started",
        "Script done", "> <start_of_turn>",
        "model :", "model:",
    ]
    lines = response.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(noise in stripped for noise in noise_markers):
            continue
        if re.match(r'^```\s*(?:json)?\s*$', stripped, re.IGNORECASE) or stripped == ">":
            continue
        if re.match(r'\[\s*Prompt:', stripped):
            continue
        clean_lines.append(stripped)

    result = " ".join(clean_lines).strip()
    result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
    result = re.sub(r'\*([^*]+)\*', r'\1', result)

    return result if result else "[PARSE_ERROR]"


def _run_with_pty(cmd, timeout=120):
    import os, pty, select, threading, time

    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave,
            preexec_fn=os.setsid, close_fds=True,
        )
    except Exception:
        os.close(master)
        os.close(slave)
        return ""

    os.close(slave)
    chunks = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            try:
                r, _, _ = select.select([master], [], [], 0.5)
                if r:
                    data = os.read(master, 65536)
                    if not data:
                        break
                    chunks.append(data)
            except (OSError, ValueError):
                break

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    time.sleep(0.3)
    done.set()
    try:
        os.close(master)
    except OSError:
        pass
    t.join(timeout=5)

    return b"".join(chunks).decode("utf-8", errors="replace")


def run_prompt_gguf(model_path: str, prompt: str, system_prompt: str, llama_cli: str, max_tokens: int = 150) -> str:
    import tempfile, os

    formatted = (
        f"<start_of_turn>system\n"
        f"{system_prompt}<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    prompt_file = None
    try:
        fd, prompt_file = tempfile.mkstemp(suffix='.txt', prefix='josi_p_')
        with os.fdopen(fd, 'w') as f:
            f.write(formatted)

        cmd = [
            llama_cli, "-m", model_path,
            "-f", prompt_file,
            "-n", str(max_tokens), "--temp", "0.45",
            "--single-turn",
        ]

        raw = _run_with_pty(cmd, timeout=120)
        if raw.strip():
            return _parse_gguf_output(raw)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        combined = (result.stdout or "") + (result.stderr or "")
        if combined.strip():
            return _parse_gguf_output(combined)

        return "[PARSE_ERROR]"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"
    finally:
        if prompt_file:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


def run_prompt_hf(model, processor, prompt: str, system_prompt: str, device: str, max_tokens: int = 150) -> str:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.45, do_sample=True, top_p=0.9,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# =============================================================================
# VALIDATION RUNNERS
# =============================================================================

def run_explainer_eval(run_fn, label: str, verbose: bool = False) -> dict:
    """Run explainer mode evaluation (plain text coaching quality)."""
    import time

    prompts = EXPLAINER_PROMPTS
    total = sum(len(v) for v in prompts.values())

    print(f"\n{'=' * 60}")
    print(f"  EXPLAINER EVALUATION — {label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {total} prompts")
    print(f"  Mode: Plain coaching text (no JSON expected)")
    print(f"{'=' * 60}\n")

    start = time.time()
    results = []
    current = 0

    for category, cat_prompts in prompts.items():
        print(f"\n[{category.upper()}]")
        for prompt in cat_prompts:
            current += 1
            print(f"  ({current}/{total}) {prompt[:45]}...", end=" ", flush=True)

            response = run_fn(prompt)
            result = evaluate_explainer(category, prompt, response)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            gov = "gov:OK" if result.governor_compliant else f"gov:FAIL({result.question_count}?)"
            json_flag = " [JSON!]" if result.is_json_response else ""
            print(f"{status} ({result.word_count}w, warm:{result.warm_score}, brief:{result.brevity_score}, {gov}){json_flag}")

            if verbose and not result.passed:
                print(f"      Issues: {', '.join(result.failure_reasons)}")
                print(f"      Response: {response[:120]}...")
            elif verbose:
                print(f"      Response: {response[:120]}...")

    elapsed = time.time() - start
    passed = sum(1 for r in results if r.passed)
    json_leaks = sum(1 for r in results if r.is_json_response)

    print(f"\n{'=' * 60}")
    print(f"  EXPLAINER RESULTS: {passed}/{len(results)} passed ({passed / len(results) * 100:.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/len(results):.1f}s per prompt)")
    if json_leaks > 0:
        print(f"  WARNING: {json_leaks} responses were JSON instead of text!")
    print(f"{'=' * 60}")

    # Metrics
    gov_compliant = sum(1 for r in results if r.governor_compliant)
    answer_first = sum(1 for r in results if r.answer_first)
    max_1q = sum(1 for r in results if r.question_count <= 1)

    print(f"\n  METRICS:")
    print(f"    Forbidden word failures: {sum(1 for r in results if r.forbidden_words_found)}")
    print(f"    Jargon warnings:         {sum(1 for r in results if r.jargon_found)}")
    print(f"    JSON leak count:         {json_leaks}")
    print(f"    Avg warmth:              {sum(r.warm_score for r in results) / len(results):.1f}/5")
    print(f"    Avg brevity:             {sum(r.brevity_score for r in results) / len(results):.1f}/5")
    print(f"    Avg word count:          {sum(r.word_count for r in results) / len(results):.0f} words")
    print(f"    --- Dialogue Governor ---")
    print(f"    Answer-first rate:       {answer_first}/{len(results)} ({answer_first / len(results) * 100:.0f}%)")
    print(f"    Max 1 question rate:     {max_1q}/{len(results)} ({max_1q / len(results) * 100:.0f}%)")
    print(f"    Governor compliance:     {gov_compliant}/{len(results)} ({gov_compliant / len(results) * 100:.0f}%)")

    unrealistic = [r for r in results if r.category == "unrealistic_goals"]
    if unrealistic:
        pushback = sum(1 for r in unrealistic if r.pushback_on_unsafe)
        print(f"    Pushback rate:           {pushback}/{len(unrealistic)}")

    print(f"\n  BY CATEGORY:")
    for cat in prompts.keys():
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        cat_passed = sum(1 for r in cat_results if r.passed)
        pct = cat_passed / len(cat_results) * 100
        icon = "PASS" if pct == 100 else " ~  " if pct >= 80 else "FAIL"
        avg_words = sum(r.word_count for r in cat_results) / len(cat_results)
        gov_ok = sum(1 for r in cat_results if r.governor_compliant)
        print(f"    {icon} {cat}: {cat_passed}/{len(cat_results)} ({pct:.0f}%) avg {avg_words:.0f}w, gov:{gov_ok}/{len(cat_results)}")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed[:15]:
            print(f"    - [{r.category}] {r.prompt[:50]}...")
            for reason in r.failure_reasons:
                print(f"        {reason}")
        if len(failed) > 15:
            print(f"    ... and {len(failed) - 15} more")

    print(f"\n{'=' * 60}\n")

    return {
        "mode": "explainer",
        "label": label,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) * 100,
        "json_leaks": json_leaks,
        "avg_warmth": sum(r.warm_score for r in results) / len(results),
        "avg_brevity": sum(r.brevity_score for r in results) / len(results),
        "avg_word_count": sum(r.word_count for r in results) / len(results),
        "governor_compliance": gov_compliant / len(results) * 100,
        "results": [asdict(r) for r in results],
    }


def run_interpreter_eval(run_fn, label: str, verbose: bool = False) -> dict:
    """Run interpreter mode evaluation (GATCRequest JSON quality)."""
    import time

    prompts = INTERPRETER_PROMPTS
    total = sum(len(v) for v in prompts.values())

    print(f"\n{'=' * 60}")
    print(f"  INTERPRETER EVALUATION — {label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {total} prompts")
    print(f"  Mode: GATCRequest JSON (structured output)")
    print(f"{'=' * 60}\n")

    start = time.time()
    results = []
    current = 0

    for category, tests in prompts.items():
        print(f"\n[{category.upper()}]")
        for test in tests:
            current += 1
            user = test["user"]
            print(f"  ({current}/{total}) {user[:45]}...", end=" ", flush=True)

            response = run_fn(user)
            result = evaluate_interpreter(category, test, response)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"{status} (json:{result.valid_json}, action:{result.action_correct})")

            if verbose and not result.passed:
                print(f"      Issues: {', '.join(result.failure_reasons)}")
                print(f"      Response: {response[:150]}...")
            elif verbose:
                print(f"      Response: {response[:150]}...")

    elapsed = time.time() - start
    passed = sum(1 for r in results if r.passed)
    valid_json = sum(1 for r in results if r.valid_json)
    action_ok = sum(1 for r in results if r.action_correct)

    print(f"\n{'=' * 60}")
    print(f"  INTERPRETER RESULTS: {passed}/{len(results)} passed ({passed / len(results) * 100:.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/len(results):.1f}s per prompt)")
    print(f"{'=' * 60}")

    print(f"\n  METRICS:")
    print(f"    Valid JSON rate:         {valid_json}/{len(results)} ({valid_json / len(results) * 100:.0f}%)")
    print(f"    Action accuracy:         {action_ok}/{len(results)} ({action_ok / len(results) * 100:.0f}%)")

    has_free_text = sum(1 for r in results if r.has_free_text)
    print(f"    free_text present:       {has_free_text}/{len(results)} ({has_free_text / len(results) * 100:.0f}%)")

    sport_tested = [r for r in results if r.sport_correct is not None]
    if sport_tested:
        sport_ok = sum(1 for r in sport_tested if r.sport_correct)
        print(f"    Sport accuracy:          {sport_ok}/{len(sport_tested)} ({sport_ok / len(sport_tested) * 100:.0f}%)")

    time_tested = [r for r in results if r.time_correct is not None]
    if time_tested:
        time_ok = sum(1 for r in time_tested if r.time_correct)
        print(f"    Time accuracy:           {time_ok}/{len(time_tested)} ({time_ok / len(time_tested) * 100:.0f}%)")

    replan_tested = [r for r in results if r.replan_type_correct is not None]
    if replan_tested:
        replan_ok = sum(1 for r in replan_tested if r.replan_type_correct)
        print(f"    Replan type accuracy:    {replan_ok}/{len(replan_tested)} ({replan_ok / len(replan_tested) * 100:.0f}%)")

    print(f"\n  BY CATEGORY:")
    for cat in prompts.keys():
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        cat_passed = sum(1 for r in cat_results if r.passed)
        pct = cat_passed / len(cat_results) * 100
        icon = "PASS" if pct == 100 else " ~  " if pct >= 80 else "FAIL"
        cat_json = sum(1 for r in cat_results if r.valid_json)
        print(f"    {icon} {cat}: {cat_passed}/{len(cat_results)} ({pct:.0f}%) json:{cat_json}/{len(cat_results)}")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed[:15]:
            print(f"    - [{r.category}] {r.prompt[:50]}...")
            for reason in r.failure_reasons:
                print(f"        {reason}")

    print(f"\n{'=' * 60}\n")

    return {
        "mode": "interpreter",
        "label": label,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) * 100,
        "valid_json_rate": valid_json / len(results) * 100,
        "action_accuracy": action_ok / len(results) * 100,
        "results": [asdict(r) for r in results],
    }


# =============================================================================
# HF MODEL LOADER
# =============================================================================

BASE_MODEL_ID = "google/gemma-3n-E2B-it"
LOCAL_BASE_PATH = Path(__file__).resolve().parent.parent / "models" / "gemma-3n-E2B-it"


def _resolve_base_model():
    """Use local base model download if available, otherwise HuggingFace hub."""
    if LOCAL_BASE_PATH.exists() and (LOCAL_BASE_PATH / "config.json").exists():
        return str(LOCAL_BASE_PATH)
    return BASE_MODEL_ID


def _is_lora_checkpoint(model_path: str) -> bool:
    """Detect if path contains a LoRA adapter rather than a full model."""
    return (Path(model_path) / "adapter_config.json").exists()


def load_hf_model(model_name: str):
    import torch
    from transformers import Gemma3nForConditionalGeneration, AutoProcessor

    model_path = Path(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _is_lora_checkpoint(model_name):
        # LoRA adapter checkpoint — load base model + merge adapter
        from peft import PeftModel
        base_id = _resolve_base_model()
        print(f"Loading base model: {base_id}")
        print(f"Loading LoRA adapter: {model_name}")

        if device == "cuda":
            base_model = Gemma3nForConditionalGeneration.from_pretrained(
                base_id, device_map="auto", torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            base_model = Gemma3nForConditionalGeneration.from_pretrained(
                base_id, torch_dtype=torch.float32,
            )
            base_model = base_model.to(device)

        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()
        print("LoRA adapter merged into base model")

        # Processor: try model dir, then sibling lora_weights, then base model
        processor = None
        for candidate in [model_name, str(model_path.parent / "lora_weights"), base_id]:
            try:
                processor = AutoProcessor.from_pretrained(candidate)
                break
            except (OSError, ValueError):
                continue
        if processor is None:
            raise RuntimeError(f"Cannot load processor from {model_name} or base model {base_id}")
    else:
        # Full merged model
        print(f"Loading HF model: {model_name}")

        # Processor: try model dir, then sibling lora_weights, then base model
        processor = None
        base_id = _resolve_base_model()
        for candidate in [model_name, str(model_path.parent / "lora_weights"), base_id]:
            try:
                processor = AutoProcessor.from_pretrained(candidate)
                break
            except (OSError, ValueError):
                continue
        if processor is None:
            raise RuntimeError(f"Cannot load processor from {model_name} or base model {base_id}")

        if device == "cuda":
            model = Gemma3nForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            model = Gemma3nForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float32,
            )
            model = model.to(device)

    print(f"Loaded on {device}")
    return model, processor, device


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 3n E2B — Dual-Mode (Interpreter + Explainer)"
    )

    # Dual-model flags (separate interpreter and explainer checkpoints)
    parser.add_argument("--interpreter", type=str,
                        help="HuggingFace model path for interpreter eval (LoRA adapter or merged)")
    parser.add_argument("--explainer", type=str,
                        help="HuggingFace model path for explainer eval (LoRA adapter or merged)")

    # Single-model flags (same model for both modes, or GGUF)
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--hf-model", type=str, help="HuggingFace model for direct inference")
    parser.add_argument("--llama-cli", type=str, default=None, help="Path to llama-cli binary")
    parser.add_argument("--mode", type=str, default="explainer",
                        choices=["interpreter", "explainer", "both"],
                        help="Evaluation mode when using --model/--hf-model (default: explainer)")
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    has_dual = args.interpreter or args.explainer
    has_single = args.model or args.hf_model

    if not has_dual and not has_single:
        parser.error("Provide --interpreter/--explainer paths, or --model (GGUF) / --hf-model (HuggingFace)")

    reports = []

    # Load system prompts
    interp_prompt = load_prompt("interpreter_system.txt")
    expl_prompt = load_prompt("explainer_system.txt")

    # --- Dual-model mode: separate checkpoints for interpreter & explainer ---
    if has_dual:
        if args.interpreter:
            model, processor, device = load_hf_model(args.interpreter)
            label = args.interpreter
            run_fn = lambda prompt: run_prompt_hf(model, processor, prompt, interp_prompt, device, max_tokens=100)
            report = run_interpreter_eval(run_fn, label=label, verbose=args.verbose)
            reports.append(report)
            # Free memory before loading next model
            del model, processor
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

        if args.explainer:
            model, processor, device = load_hf_model(args.explainer)
            label = args.explainer
            run_fn = lambda prompt: run_prompt_hf(model, processor, prompt, expl_prompt, device, max_tokens=150)
            report = run_explainer_eval(run_fn, label=label, verbose=args.verbose)
            reports.append(report)

    # --- Single-model mode: one model, choose mode via --mode ---
    elif args.hf_model:
        model, processor, device = load_hf_model(args.hf_model)
        label = args.hf_model

        if args.mode in ("explainer", "both"):
            run_fn = lambda prompt: run_prompt_hf(model, processor, prompt, expl_prompt, device, max_tokens=150)
            report = run_explainer_eval(run_fn, label=label, verbose=args.verbose)
            reports.append(report)

        if args.mode in ("interpreter", "both"):
            run_fn = lambda prompt: run_prompt_hf(model, processor, prompt, interp_prompt, device, max_tokens=100)
            report = run_interpreter_eval(run_fn, label=label, verbose=args.verbose)
            reports.append(report)

    elif args.model:
        llama_cli = args.llama_cli or str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
        label = Path(args.model).stem

        if args.mode in ("explainer", "both"):
            run_fn = lambda prompt: run_prompt_gguf(args.model, prompt, expl_prompt, llama_cli, max_tokens=150)
            report = run_explainer_eval(run_fn, label=label, verbose=args.verbose)
            reports.append(report)

        if args.mode in ("interpreter", "both"):
            run_fn = lambda prompt: run_prompt_gguf(args.model, prompt, interp_prompt, llama_cli, max_tokens=100)
            report = run_interpreter_eval(run_fn, label=label, verbose=args.verbose)
            reports.append(report)

    if args.output:
        output_data = reports[0] if len(reports) == 1 else {"reports": reports}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
