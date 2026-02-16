#!/usr/bin/env python3
"""
MiValta Josi v4 — Memory-Enriched Training Data Augmentation

Generates training examples that teach the model to:
  1. USE memory facts to skip clarification (interpreter)
  2. USE memory for personalized responses (explainer)
  3. NOT hallucinate facts when memory is absent (anti-hallucination)
  4. IGNORE irrelevant memory facts
  5. PREFER current user message over contradictory memory

Target: ~2K examples total:
  - ~600 interpreter with memory (sport inference, preference use, constraint injection)
  - ~400 interpreter anti-hallucination (no memory → don't assume)
  - ~600 explainer with memory (personalized coaching)
  - ~400 explainer anti-hallucination (no memory → generic response)

Usage:
    python augment_memory_training_data.py
    python augment_memory_training_data.py --dry-run
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def load_system_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


# =============================================================================
# MEMORY BLOCK TEMPLATES — compact format matching token budget (~174 tokens)
# =============================================================================

MEMORY_BLOCKS = {
    "runner_morning": (
        "MEMORY:\n"
        "- Facts: primary sport: running | prefers morning sessions | "
        "typical duration: 45-60 min | trains 4x per week\n"
        "- Patterns: consistent weekday runner, skips weekends occasionally\n"
        "- Notes: responds well to encouragement"
    ),
    "runner_knee": (
        "MEMORY:\n"
        "- Facts: primary sport: running | has recurring knee issue (right) | "
        "avoids downhill running | level: intermediate\n"
        "- Patterns: backs off intensity when knee flares\n"
        "- Notes: appreciates when knee is acknowledged proactively"
    ),
    "cyclist_evening": (
        "MEMORY:\n"
        "- Facts: primary sport: cycling | prefers evening rides | "
        "typical duration: 60-90 min | works night shifts\n"
        "- Patterns: trains late afternoon, often tired from work\n"
        "- Notes: prefers data-driven explanations"
    ),
    "runner_beginner": (
        "MEMORY:\n"
        "- Facts: primary sport: running | level: beginner | "
        "started 3 months ago | goal: complete first 10k\n"
        "- Patterns: tends to run too fast on easy days\n"
        "- Notes: needs reassurance about slow pace being okay"
    ),
    "multi_sport": (
        "MEMORY:\n"
        "- Facts: does running and cycling | prefers running on weekdays | "
        "cycles on weekends | typical run: 45 min | typical ride: 90 min\n"
        "- Patterns: consistent but overloads in build weeks\n"
        "- Notes: competitive, responds to challenge-based framing"
    ),
    "runner_busy": (
        "MEMORY:\n"
        "- Facts: primary sport: running | works long hours | "
        "prefers short sessions (30 min) | has two young kids\n"
        "- Patterns: skips sessions when work is busy\n"
        "- Notes: values efficiency, dislikes long explanations"
    ),
    "strength_runner": (
        "MEMORY:\n"
        "- Facts: primary sport: running | does strength 2x per week | "
        "had back injury 6 months ago | avoids heavy deadlifts\n"
        "- Patterns: prioritizes running over strength when time is short\n"
        "- Notes: cautious about back exercises"
    ),
    "runner_advanced": (
        "MEMORY:\n"
        "- Facts: primary sport: running | level: advanced | "
        "marathon PR: 3:15 | trains 6x per week | prefers structured intervals\n"
        "- Patterns: ramps volume too fast after rest weeks\n"
        "- Notes: technically curious, likes hearing the why behind sessions"
    ),
}


def _mem(key: str) -> str:
    """Shorthand to get a memory block."""
    return MEMORY_BLOCKS[key]


# =============================================================================
# INTERPRETER: MEMORY-ENRICHED EXAMPLES
# Memory resolves ambiguity → skip clarification
# =============================================================================

INTERPRETER_MEMORY_EXAMPLES = [
    # --- Memory provides sport → skip clarify ---
    {"user": "Give me a workout\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Give me a workout", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "I want to train today\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "create_workout", "sport": "bike", "free_text": "I want to train today", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Build me something for today\n\nCONTEXT:\n- Readiness: Yellow\n" + _mem("runner_knee"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Build me something for today", "constraints": {"fatigue_hint": "ok", "injury": "knee"}}},

    {"user": "Workout please\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_beginner"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Workout please", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "I have time to train\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "I have time to train", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Let's do it\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "create_workout", "sport": "bike", "free_text": "Let's do it", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Something for this morning\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Something for this morning", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "I'm ready to go\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_advanced"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "I'm ready to go", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Train me\n\nCONTEXT:\n- Readiness: Yellow\n" + _mem("multi_sport"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Train me", "constraints": {"fatigue_hint": "ok"}}},

    {"user": "Get me moving\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_busy"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Get me moving", "constraints": {"fatigue_hint": "fresh"}}},

    # --- Memory provides typical duration → infer time ---
    {"user": "Normal session today\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "Normal session today", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Regular ride please\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "Regular ride please", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "A quick one today\n\nCONTEXT:\n- Readiness: Yellow\n" + _mem("runner_busy"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 30, "free_text": "A quick one today", "constraints": {"fatigue_hint": "ok"}}},

    {"user": "The usual\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "The usual", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Usual length ride\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "Usual length ride", "constraints": {"fatigue_hint": "fresh"}}},

    # --- Memory provides injury → inject constraint ---
    {"user": "I want to run today, 40 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n" + _mem("runner_knee"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 40, "free_text": "I want to run today, 40 minutes", "constraints": {"fatigue_hint": "fresh", "injury": "knee"}}},

    {"user": "Give me an hour of running\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n" + _mem("runner_knee"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "Give me an hour of running", "constraints": {"fatigue_hint": "fresh", "injury": "knee"}}},

    {"user": "45 minute run\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow\n" + _mem("strength_runner"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "45 minute run", "constraints": {"fatigue_hint": "ok", "injury": "back"}}},

    # --- Current message overrides memory ---
    {"user": "I want to cycle today, not run\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "bike", "free_text": "I want to cycle today, not run", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Let me try cycling for 30 minutes\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 30, "free_text": "Let me try cycling for 30 minutes", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "I want to do a 90 minute long run\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_busy"),
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 90, "free_text": "I want to do a 90 minute long run", "goal": "endurance", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Strength today, skip the run\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "strength", "free_text": "Strength today, skip the run", "constraints": {"fatigue_hint": "fresh"}}},

    # --- Memory with replan ---
    {"user": "I need to skip today\n\nCONTEXT:\n- Readiness: Red\n" + _mem("runner_morning"),
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "I need to skip today"}},

    {"user": "Can we move my session to tomorrow?\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "bike", "free_text": "Can we move my session to tomorrow?"}},

    {"user": "I'm traveling next week\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "I'm traveling next week"}},

    {"user": "Can we dial it back today?\n\nCONTEXT:\n- Readiness: Orange\n" + _mem("runner_advanced"),
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "Can we dial it back today?"}},

    # --- Memory with explain/answer_question ---
    {"user": "Why is today an easy day?\n\nCONTEXT:\n- Sport: running\n- Session: Z1 30min Recovery\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "explain", "question": "Why is today an easy day?", "free_text": "Why is today an easy day?"}},

    {"user": "What's happening this week?\n\nCONTEXT:\n- Sport: running\n- Week: build phase week 3\n- Readiness: Green\n" + _mem("runner_advanced"),
     "response": {"action": "explain", "question": "What's happening this week?", "free_text": "What's happening this week?"}},

    {"user": "How does recovery work?\n\nCONTEXT:\n- Readiness: Yellow\n" + _mem("runner_beginner"),
     "response": {"action": "answer_question", "question": "How does recovery work?", "free_text": "How does recovery work?"}},

    {"user": "What is Zone 2?\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "answer_question", "question": "What is Zone 2?", "free_text": "What is Zone 2?"}},

    # --- Multi-sport: memory resolves to weekday sport ---
    {"user": "Give me a workout for today\n\nCONTEXT:\n- Readiness: Green\n" + _mem("multi_sport"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Give me a workout for today", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Weekend workout please\n\nCONTEXT:\n- Readiness: Green\n" + _mem("multi_sport"),
     "response": {"action": "create_workout", "sport": "bike", "free_text": "Weekend workout please", "constraints": {"fatigue_hint": "fresh"}}},

    # --- Memory present but still need to clarify (ambiguous multi-sport) ---
    {"user": "Build me something different today\n\nCONTEXT:\n- Readiness: Green\n" + _mem("multi_sport"),
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Running or cycling today?", "free_text": "Build me something different today"}},

    # --- More sport inference from memory ---
    {"user": "Session please\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Session please", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Make me sweat\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "create_workout", "sport": "bike", "free_text": "Make me sweat", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "I need a good session\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_advanced"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "I need a good session", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "Can you set me up for today?\n\nCONTEXT:\n- Readiness: Yellow\n" + _mem("runner_beginner"),
     "response": {"action": "create_workout", "sport": "run", "free_text": "Can you set me up for today?", "constraints": {"fatigue_hint": "ok"}}},

    # --- Illness still works with memory ---
    {"user": "I'm sick\n\nCONTEXT:\n- Readiness: Red\n" + _mem("runner_morning"),
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I'm sick"}},

    {"user": "Got the flu, need to pause\n\nCONTEXT:\n- Readiness: Red\n" + _mem("cyclist_evening"),
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "Got the flu, need to pause"}},

    # --- Medical red flags still override memory ---
    {"user": "I have chest pain\n\nCONTEXT:\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I have chest pain"}},

    {"user": "I felt dizzy during my ride yesterday\n\nCONTEXT:\n- Readiness: Green\n" + _mem("cyclist_evening"),
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I felt dizzy during my ride yesterday"}},
]


# =============================================================================
# INTERPRETER: ANTI-HALLUCINATION EXAMPLES
# No memory present → model must NOT assume facts
# =============================================================================

INTERPRETER_NO_MEMORY_EXAMPLES = [
    # --- No memory, no context → must clarify ---
    {"user": "Give me a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train? Running, cycling, or something else?", "free_text": "Give me a workout"}},

    {"user": "I want to train today",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength, or something else?", "free_text": "I want to train today"}},

    {"user": "Build me something",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Build me something"}},

    {"user": "Workout please",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport would you like a workout for?", "free_text": "Workout please"}},

    {"user": "Let's do it",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "Let's go! What sport and what kind of session?", "free_text": "Let's do it"}},

    {"user": "I'm ready to go",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "Great! What sport do you want to train today?", "free_text": "I'm ready to go"}},

    {"user": "Session please",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport would you like a session for?", "free_text": "Session please"}},

    {"user": "Train me",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Train me"}},

    {"user": "The usual",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "What would you like today? Which sport and what kind of session?", "free_text": "The usual"}},

    {"user": "Something for this morning",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to do this morning?", "free_text": "Something for this morning"}},

    # --- No memory, but has context with sport → works fine ---
    {"user": "Give me a workout\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "free_text": "Give me a workout", "constraints": {"fatigue_hint": "fresh"}}},

    {"user": "I want to train today\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
     "response": {"action": "create_workout", "sport": "bike", "free_text": "I want to train today", "constraints": {"fatigue_hint": "ok"}}},

    # --- No memory, readiness only → still need sport ---
    {"user": "I have time to train\n\nCONTEXT:\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Great! Which sport — running, cycling, strength, or something else?", "free_text": "I have time to train"}},

    {"user": "Get me moving\n\nCONTEXT:\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport do you want to get moving with?", "free_text": "Get me moving"}},

    {"user": "Make me sweat\n\nCONTEXT:\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, or something else?", "free_text": "Make me sweat"}},

    # --- General questions still work without memory ---
    {"user": "What is Zone 2?",
     "response": {"action": "answer_question", "question": "What is Zone 2?", "free_text": "What is Zone 2?"}},

    {"user": "How does recovery work?",
     "response": {"action": "answer_question", "question": "How does recovery work?", "free_text": "How does recovery work?"}},

    {"user": "Should I stretch before running?",
     "response": {"action": "answer_question", "question": "Should I stretch before running?", "free_text": "Should I stretch before running?"}},

    # --- Illness without memory ---
    {"user": "I'm sick\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I'm sick"}},

    {"user": "I have the flu\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "I have the flu"}},
]


# =============================================================================
# EXPLAINER: MEMORY-ENRICHED EXAMPLES
# Personalized coaching using memory facts
# =============================================================================

EXPLAINER_MEMORY_EXAMPLES = [
    # --- Morning runner, personalized encouragement ---
    {"user": "I'm feeling tired today but I want to train.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running\n" + _mem("runner_morning"),
     "response": "Your body's telling you to take it easy today. Since mornings are your sweet spot, a short easy run first thing might actually help you feel better. Keep the pace conversational and cut it to 30 minutes instead of your usual 45. You'll bounce back faster by respecting the fatigue now."},

    {"user": "What should I do today?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 45min Continuous\n- Readiness: Green\n" + _mem("runner_morning"),
     "response": "Today's a solid aerobic builder — 45 minutes at an easy, conversational pace. Right in your comfort zone for a morning session. Just keep it relaxed, enjoy the run, and let your body do the adapting."},

    # --- Knee issue runner, proactive acknowledgment ---
    {"user": "I want to go hard today.\n\nCONTEXT:\n- Sport: running\n- Session: Z4 35min Threshold\n- Readiness: Green\n" + _mem("runner_knee"),
     "response": "Today's session has some solid intensity — 35 minutes with threshold work. Your body is showing green across the board, which is great. Just be mindful of your right knee, especially on any downhill sections. If it starts talking to you, ease off the pace rather than pushing through."},

    {"user": "Can I do intervals today?\n\nCONTEXT:\n- Sport: running\n- Session: Z5 30min VO2max intervals\n- Readiness: Green\n" + _mem("runner_knee"),
     "response": "You're cleared for intervals today — your body is recovered and ready. The plan has 30 minutes of high-intensity work. Flat terrain is your friend here given your knee. If you feel any discomfort, swap to a steady effort instead."},

    # --- Cyclist who works nights ---
    {"user": "I'm exhausted from work, but I still want to ride.\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow\n" + _mem("cyclist_evening"),
     "response": "After a night shift, your body's already running on less. A shorter, easy spin would be the smart move — maybe 40 minutes instead of your usual hour. Keep the legs turning without pushing the heart rate. You'll get more out of quality rest than a forced hard session right now."},

    {"user": "Why is my readiness yellow?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow (Accumulated)\n" + _mem("cyclist_evening"),
     "response": "Your body is carrying some accumulated fatigue — it's absorbing recent training load. Your shift pattern can amplify this since sleep quality matters a lot for recovery. The numbers say ease off today. A lighter session or full rest day will help your body catch up."},

    # --- Beginner runner, reassurance ---
    {"user": "Am I going too slow? Everyone else seems faster.\n\nCONTEXT:\n- Sport: running\n- Session: Z1 25min Recovery\n- Readiness: Green\n" + _mem("runner_beginner"),
     "response": "Not at all — slow is exactly right for where you are. Three months in, your body is still building its aerobic foundation. The pace that feels too easy now is actually doing the most important work. Stick with it, and you'll naturally get faster over the coming weeks without even trying."},

    {"user": "I ran faster than planned today. Is that okay?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 30min Continuous\n- Readiness: Green\n" + _mem("runner_beginner"),
     "response": "I get the temptation — it feels good to go fast! But easy days need to stay easy for your body to recover and adapt. When you push harder than planned on easy days, you end up too tired for the sessions that actually build speed. Try the talk test next time: if you can't chat comfortably, you're too fast."},

    # --- Busy parent, efficiency-focused ---
    {"user": "I only have 20 minutes, is that worth it?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n" + _mem("runner_busy"),
     "response": "Absolutely. 20 minutes of focused running is better than skipping entirely. With your schedule, consistency matters more than duration. A quick session keeps the habit alive and your body adapting. Every minute counts."},

    {"user": "I missed three sessions this week because of work.\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n" + _mem("runner_busy"),
     "response": "That happens, and it's okay. With your schedule, some weeks will look like this. Don't try to make up missed sessions — just pick up where the plan is now. Consistency over months matters far more than any single week. You're doing fine."},

    # --- Advanced runner, technical ---
    {"user": "Why am I doing easy runs this week? I feel great.\n\nCONTEXT:\n- Sport: running\n- Week: recovery week\n- Readiness: Green\n" + _mem("runner_advanced"),
     "response": "This is a planned recovery week — your body needs it even when you feel good. After your recent build, pulling back now lets your muscles fully repair and absorb the training. It's the recovery that turns hard work into fitness gains. You'll come out of this week stronger and ready to push again."},

    {"user": "My legs feel heavy but my times are fine.\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow\n" + _mem("runner_advanced"),
     "response": "Heavy legs with maintained times is a classic sign of accumulated fatigue — your fitness is high but recovery is lagging. Given your volume, this is expected. An extra easy day now prevents the slide into a real dip. Your body is asking for rest even if the stopwatch isn't."},

    # --- Multi-sport athlete ---
    {"user": "Should I ride or run today?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n" + _mem("multi_sport"),
     "response": "Since it's a weekday, a run fits your usual rhythm. If your legs are feeling fresh, go for it. Save the bike for the weekend when you typically have more time for a longer ride. Keeping that pattern works well for your recovery between the two."},

    # --- Strength runner, back-conscious ---
    {"user": "Can I do strength today instead of running?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 40min Continuous\n- Readiness: Green\n" + _mem("strength_runner"),
     "response": "You can, but keep in mind today's planned run is a key aerobic session. If you swap, try to fit the run in tomorrow. For strength today, focus on the movements that feel safe for your back — bodyweight and core work are great alternatives to heavy loading."},
]


# =============================================================================
# EXPLAINER: ANTI-HALLUCINATION EXAMPLES
# No memory → generic but warm responses, no assumed facts
# =============================================================================

EXPLAINER_NO_MEMORY_EXAMPLES = [
    {"user": "I'm feeling tired today but I want to train.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running\n- Level: intermediate",
     "response": "I hear you — feeling tired but still showing up says a lot about your commitment. On days like this, your body benefits most from keeping the intensity low. Trust the easy pace today and you'll come back stronger tomorrow."},

    {"user": "What should I do today?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 45min Continuous\n- Readiness: Green",
     "response": "Today's an easy aerobic run — 45 minutes at a comfortable, conversational pace. Nothing fancy, just steady effort. Let your body do the work without pushing. This kind of session builds your endurance foundation."},

    {"user": "I want to go hard today.\n\nCONTEXT:\n- Sport: running\n- Session: Z4 35min Threshold\n- Readiness: Green",
     "response": "Good news — today's session has real intensity. 35 minutes of threshold work. Your body is recovered and ready for it. Push the effort but listen to your body. If something doesn't feel right, back off."},

    {"user": "Am I going too slow?\n\nCONTEXT:\n- Sport: running\n- Session: Z1 25min Recovery\n- Readiness: Green",
     "response": "Not at all. Easy days are supposed to feel easy. This pace is doing important work for your body even though it doesn't feel like it. Recovery runs build your aerobic base and help you absorb harder sessions."},

    {"user": "I missed three sessions this week.\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": "That happens to everyone. Don't try to make up for missed sessions — just pick up where the plan is now. One off week doesn't undo your progress. Consistency over months is what matters."},

    {"user": "My legs feel heavy.\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": "Heavy legs usually mean your body is still processing recent training. This is normal, especially after a hard stretch. An easy day or rest day now will help your body catch up. Listen to what it's telling you."},

    {"user": "Why is my readiness yellow?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow (Accumulated)",
     "response": "Your body is carrying some accumulated fatigue from recent training. It's not a red flag — it just means your recovery hasn't fully caught up yet. Backing off the intensity today will help you absorb the training you've already done."},

    {"user": "I only have 20 minutes, is that worth it?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": "Absolutely. 20 minutes of focused running is better than skipping entirely. Consistency matters more than duration. A short session keeps the habit alive and your body adapting."},

    {"user": "Should I ride or run today?\n\nCONTEXT:\n- Readiness: Green",
     "response": "That depends on what your plan calls for and how your body feels. If your legs are fresh, either works. Pick whichever one you're more motivated for today — showing up matters more than which sport you choose."},

    {"user": "Can I do strength today instead of running?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 40min Continuous\n- Readiness: Green",
     "response": "You can, but today's planned run is a key aerobic session. If you swap, try to fit the run in tomorrow. If strength is what you need today, go for it — just keep the overall week balanced."},
]


# =============================================================================
# MAIN
# =============================================================================

def build_interpreter_example(system_prompt: str, user: str, response: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
        ]
    }


def build_explainer_example(system_prompt: str, user: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Augment training data with memory examples")
    parser.add_argument("--dry-run", action="store_true", help="Show stats, don't write")
    args = parser.parse_args()

    interp_prompt = load_system_prompt("interpreter_system.txt")
    expl_prompt = load_system_prompt("explainer_system.txt")

    # Build interpreter examples
    interp_memory = [build_interpreter_example(interp_prompt, e["user"], e["response"])
                     for e in INTERPRETER_MEMORY_EXAMPLES]
    interp_no_memory = [build_interpreter_example(interp_prompt, e["user"], e["response"])
                        for e in INTERPRETER_NO_MEMORY_EXAMPLES]

    # Build explainer examples
    expl_memory = [build_explainer_example(expl_prompt, e["user"], e["response"])
                   for e in EXPLAINER_MEMORY_EXAMPLES]
    expl_no_memory = [build_explainer_example(expl_prompt, e["user"], e["response"])
                      for e in EXPLAINER_NO_MEMORY_EXAMPLES]

    total_interp = len(interp_memory) + len(interp_no_memory)
    total_expl = len(expl_memory) + len(expl_no_memory)

    # Stats
    from collections import Counter
    action_counts = Counter()
    for e in INTERPRETER_MEMORY_EXAMPLES + INTERPRETER_NO_MEMORY_EXAMPLES:
        r = e["response"]
        action_counts[r["action"]] += 1
        if r["action"] == "replan":
            action_counts[f"  replan:{r['replan_type']}"] += 1

    print(f"\n{'=' * 60}")
    print(f"  Memory Training Data Augmentation")
    print(f"{'=' * 60}")
    print(f"\n  Interpreter: {total_interp} examples")
    print(f"    With memory:    {len(interp_memory)}")
    print(f"    Anti-halluc.:   {len(interp_no_memory)}")
    for action, count in sorted(action_counts.items()):
        print(f"      {action}: {count}")
    print(f"\n  Explainer: {total_expl} examples")
    print(f"    With memory:    {len(expl_memory)}")
    print(f"    Anti-halluc.:   {len(expl_no_memory)}")
    print(f"\n  Total: {total_interp + total_expl} examples")

    # Validate interpreter JSON
    errors = 0
    for item in INTERPRETER_MEMORY_EXAMPLES + INTERPRETER_NO_MEMORY_EXAMPLES:
        r = item["response"]
        if "free_text" not in r:
            print(f"    ERROR: missing free_text in {item['user'][:40]}...")
            errors += 1
        if r["action"] == "replan" and "replan_type" not in r:
            print(f"    ERROR: missing replan_type in {item['user'][:40]}...")
            errors += 1

    # Validate explainer question count
    for item in EXPLAINER_MEMORY_EXAMPLES + EXPLAINER_NO_MEMORY_EXAMPLES:
        q_count = item["response"].count("?")
        if q_count > 1:
            print(f"    WARNING: {q_count} questions in explainer: {item['user'][:40]}...")
            errors += 1

    if errors > 0:
        print(f"\n  {errors} validation errors found!")
        return

    print(f"\n  All examples validated OK")

    if args.dry_run:
        print(f"\n  Dry run — no files written")
        return

    # Load existing data and append
    interp_path = DATA_DIR / "train_interpreter.jsonl"
    expl_path = DATA_DIR / "train_explainer.jsonl"

    # Count existing
    with open(interp_path) as f:
        existing_interp = sum(1 for _ in f)
    with open(expl_path) as f:
        existing_expl = sum(1 for _ in f)

    # Append
    with open(interp_path, "a") as f:
        for ex in interp_memory + interp_no_memory:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(expl_path, "a") as f:
        for ex in expl_memory + expl_no_memory:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Interpreter: {existing_interp} → {existing_interp + total_interp}")
    print(f"  Explainer:   {existing_expl} → {existing_expl + total_expl}")
    print(f"\n  Files updated:")
    print(f"    {interp_path}")
    print(f"    {expl_path}")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
