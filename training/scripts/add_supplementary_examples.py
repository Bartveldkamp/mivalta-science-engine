#!/usr/bin/env python3
"""
MiValta Josi — Supplementary Training Examples (v6.1)

Adds training examples to address specific gaps identified during chat testing:

1. Follow-up questions about sessions → explain (not create_workout)
2. Intensity keywords ("intensive", "hard") → correct goal mapping
3. Vague workout requests → omit goal field (let gathering handle it)
4. Warmer coach responses for illness/replan
5. Session presentation follow-ups
6. Dutch equivalents for intensity and follow-ups

These examples are appended to the source training files:
  - train_interpreter.jsonl
  - train_explainer_sequential.jsonl

Then prepare_v6_data.py handles prompt updates + knowledge injection.

Usage:
    python add_supplementary_examples.py
    python add_supplementary_examples.py --dry-run   # Preview without writing
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

INTERPRETER_FILE = DATA_DIR / "train_interpreter.jsonl"
COACH_FILE = DATA_DIR / "train_explainer_sequential.jsonl"


def get_interpreter_system_prompt():
    """Read the system prompt from the first existing interpreter example."""
    with open(INTERPRETER_FILE) as f:
        first = json.loads(f.readline())
    return first["messages"][0]["content"]


def get_coach_system_prompt():
    """Read the system prompt from the first existing coach example."""
    with open(COACH_FILE) as f:
        first = json.loads(f.readline())
    return first["messages"][0]["content"]


def interp_example(system_prompt, user_content, assistant_json):
    """Build an interpreter training example."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ]
    }


def coach_example(system_prompt, user_content, coach_response):
    """Build a coach training example."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": coach_response},
        ]
    }


# ─── INTERPRETER EXAMPLES ────────────────────────────────────────────

def build_interpreter_examples(sys_prompt):
    examples = []

    # ── 1. Follow-up questions about a session → explain ──────────────

    examples.append(interp_example(sys_prompt,
        'How much rest in between?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 60min "5 x 3min Z5 / 3min Z1" (peak phase)\n\nHISTORY:\n  [user]: I want a hard 60 minute run\n  [assistant]: Here\'s your session: 5 x 3min hard efforts with 3min easy recovery between each.',
        {"action": "explain", "question": "How much rest in between?", "free_text": "How much rest in between?"}
    ))

    examples.append(interp_example(sys_prompt,
        'What pace should I run the intervals?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z4 75min "4 x 6min Z4 / 5min Z1" (build phase)\n\nHISTORY:\n  [user]: Give me a threshold workout, 75 minutes\n  [assistant]: Your session is 4 x 6min at threshold with 5min easy between each.',
        {"action": "explain", "question": "What pace should I run the intervals?", "free_text": "What pace should I run the intervals?"}
    ))

    examples.append(interp_example(sys_prompt,
        'Tell me more about this workout\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Session: Z5 60min "7 x 3min Z5 / 3min Z1" (peak phase)',
        {"action": "explain", "question": "Tell me more about this workout", "free_text": "Tell me more about this workout"}
    ))

    examples.append(interp_example(sys_prompt,
        'How hard should the efforts be?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 60min "5 x 3min Z5 / 3min Z1" (peak phase)',
        {"action": "explain", "question": "How hard should the efforts be?", "free_text": "How hard should the efforts be?"}
    ))

    examples.append(interp_example(sys_prompt,
        'Can you break the session down for me?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow (Accumulated)\n- Session: Z4 60min "3 x 6min Z4 / 5min Z1" (build phase)',
        {"action": "explain", "question": "Can you break the session down for me?", "free_text": "Can you break the session down for me?"}
    ))

    examples.append(interp_example(sys_prompt,
        'Should I warm up before the intervals?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 45min "4 x 3min Z5 / 3min Z1" (peak phase)',
        {"action": "explain", "question": "Should I warm up before the intervals?", "free_text": "Should I warm up before the intervals?"}
    ))

    # Dutch follow-ups
    examples.append(interp_example(sys_prompt,
        'Hoeveel rust ertussen?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 60min "5 x 3min Z5 / 3min Z1" (peak phase)\n\nHISTORY:\n  [user]: Ik wil een zware training van 60 minuten\n  [assistant]: Je sessie: 5 x 3min hard met 3min rustig ertussen.',
        {"action": "explain", "question": "Hoeveel rust ertussen?", "free_text": "Hoeveel rust ertussen?"}
    ))

    examples.append(interp_example(sys_prompt,
        'Welk tempo moet ik lopen?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z4 60min "4 x 6min Z4 / 5min Z1" (build phase)',
        {"action": "explain", "question": "Welk tempo moet ik lopen?", "free_text": "Welk tempo moet ik lopen?"}
    ))

    # ── 2. Intensity keywords → correct goal mapping ─────────────────

    examples.append(interp_example(sys_prompt,
        'I want to train intensive\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "bike", "goal": "threshold", "free_text": "I want to train intensive"}
    ))

    examples.append(interp_example(sys_prompt,
        'I want to train intensive for an hour\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "goal": "threshold", "time_available_min": 60, "free_text": "I want to train intensive for an hour"}
    ))

    examples.append(interp_example(sys_prompt,
        'Make it harder\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z2 60min "Continuous Z2 60min" (build phase)',
        {"action": "create_workout", "sport": "run", "goal": "vo2", "time_available_min": 60, "free_text": "Make it harder"}
    ))

    examples.append(interp_example(sys_prompt,
        'I want something more challenging\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "bike", "goal": "threshold", "free_text": "I want something more challenging"}
    ))

    examples.append(interp_example(sys_prompt,
        'Can we go harder today?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "goal": "vo2", "free_text": "Can we go harder today?"}
    ))

    examples.append(interp_example(sys_prompt,
        'I want intervals\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "goal": "vo2", "free_text": "I want intervals"}
    ))

    examples.append(interp_example(sys_prompt,
        'Give me a tempo run, 45 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "goal": "threshold", "time_available_min": 45, "free_text": "Give me a tempo run, 45 minutes"}
    ))

    # Dutch intensity
    examples.append(interp_example(sys_prompt,
        'Ik wil intensief trainen\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "goal": "threshold", "free_text": "Ik wil intensief trainen"}
    ))

    examples.append(interp_example(sys_prompt,
        'Maak het maar stevig vandaag\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "bike", "goal": "threshold", "free_text": "Maak het maar stevig vandaag"}
    ))

    examples.append(interp_example(sys_prompt,
        'Ik wil het zwaarder\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z2 45min "Continuous Z2 45min" (build phase)',
        {"action": "create_workout", "sport": "run", "goal": "vo2", "time_available_min": 45, "free_text": "Ik wil het zwaarder"}
    ))

    # ── 3. Vague requests → no goal field (gathering handles it) ─────

    examples.append(interp_example(sys_prompt,
        'I want to do a bike workout, I have 75 minutes\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "bike", "time_available_min": 75, "free_text": "I want to do a bike workout, I have 75 minutes"}
    ))

    examples.append(interp_example(sys_prompt,
        'Give me a running session\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "free_text": "Give me a running session"}
    ))

    examples.append(interp_example(sys_prompt,
        'I want to train today, 60 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "I want to train today, 60 minutes"}
    ))

    examples.append(interp_example(sys_prompt,
        'Ik heb een uur de tijd voor een fietsworkout\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "Ik heb een uur de tijd voor een fietsworkout"}
    ))

    examples.append(interp_example(sys_prompt,
        'Let me do a run\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: beginner',
        {"action": "create_workout", "sport": "run", "free_text": "Let me do a run"}
    ))

    examples.append(interp_example(sys_prompt,
        'I want to ride my bike for 90 minutes\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow (Accumulated)\n- Level: advanced',
        {"action": "create_workout", "sport": "bike", "time_available_min": 90, "free_text": "I want to ride my bike for 90 minutes"}
    ))

    examples.append(interp_example(sys_prompt,
        'Ik wil een looptraining doen\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "free_text": "Ik wil een looptraining doen"}
    ))

    # ── 4. Sick athlete asking to train → answer_question (not create_workout) ──

    examples.append(interp_example(sys_prompt,
        'Can I still do a light workout?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (IllnessRisk)\n\nHISTORY:\n  [user]: I\'m sick with a cold\n  [assistant]: Rest is the priority right now. Your body needs energy to fight this off.',
        {"action": "answer_question", "question": "Can I still do a light workout while sick?", "free_text": "Can I still do a light workout?"}
    ))

    examples.append(interp_example(sys_prompt,
        'Should I do a bike workout?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Red (IllnessRisk)\n\nHISTORY:\n  [user]: I have the flu\n  [assistant]: Take it easy. Rest and fluids are your training plan for now.',
        {"action": "answer_question", "question": "Should I do a bike workout while sick?", "free_text": "Should I do a bike workout?"}
    ))

    # ── 5. Conversational acknowledgements → answer_question ─────────

    examples.append(interp_example(sys_prompt,
        'Okay thanks\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n\nHISTORY:\n  [user]: What is Zone 2?\n  [assistant]: Zone 2 is your aerobic base zone. It builds endurance and fat burning.',
        {"action": "answer_question", "question": "Okay thanks", "free_text": "Okay thanks"}
    ))

    examples.append(interp_example(sys_prompt,
        'Cool, got it\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n\nHISTORY:\n  [user]: How long should I warm up?\n  [assistant]: A good warm-up is 10-15 minutes of easy spinning to get your legs ready.',
        {"action": "answer_question", "question": "Cool, got it", "free_text": "Cool, got it"}
    ))

    # ── 6. "Actually make it easier" → create_workout with reduced goal ──

    examples.append(interp_example(sys_prompt,
        'Actually, make it easier\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 60min "5 x 3min Z5 / 3min Z1" (peak phase)\n\nHISTORY:\n  [user]: Give me a hard run\n  [assistant]: Your session: 5 x 3min hard efforts with 3min easy between each.',
        {"action": "create_workout", "sport": "run", "goal": "endurance", "time_available_min": 60, "free_text": "Actually, make it easier"}
    ))

    examples.append(interp_example(sys_prompt,
        'That\'s too hard, can you dial it down?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Session: Z5 75min "7 x 3min Z5 / 3min Z1" (peak phase)\n\nHISTORY:\n  [user]: I want a hard ride, 75 minutes\n  [assistant]: Your session: 7 x 3min hard efforts with 3min easy between each.',
        {"action": "create_workout", "sport": "bike", "goal": "endurance", "time_available_min": 75, "free_text": "That's too hard, can you dial it down?"}
    ))

    # Duplicate important patterns with slight variations for model robustness
    examples.append(interp_example(sys_prompt,
        'How long are the rest periods?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Session: Z4 60min "3 x 6min Z4 / 5min Z1" (build phase)',
        {"action": "explain", "question": "How long are the rest periods?", "free_text": "How long are the rest periods?"}
    ))

    examples.append(interp_example(sys_prompt,
        'What does this session do for me?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z2 75min "Continuous Z2 75min" (build phase)',
        {"action": "explain", "question": "What does this session do for me?", "free_text": "What does this session do for me?"}
    ))

    examples.append(interp_example(sys_prompt,
        'I want to go all out today, 30 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: advanced',
        {"action": "create_workout", "sport": "run", "goal": "vo2", "time_available_min": 30, "free_text": "I want to go all out today, 30 minutes"}
    ))

    examples.append(interp_example(sys_prompt,
        'Geef me een rustige training\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate',
        {"action": "create_workout", "sport": "run", "goal": "recovery", "free_text": "Geef me een rustige training"}
    ))

    return examples


# ─── COACH EXAMPLES ───────────────────────────────────────────────────

def build_coach_examples(sys_prompt):
    examples = []

    # ── 1. Warmer illness responses ──────────────────────────────────

    examples.append(coach_example(sys_prompt,
        'I\'m sick, can\'t train this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (IllnessRisk)\n\n[INTERPRETER]\n{"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I\'m sick, can\'t train this week"}',
        "I hear you — being sick is frustrating, especially when you're in a good training rhythm. But your body needs all its energy to fight this off right now. Training while sick would only delay your recovery. Rest, stay hydrated, and sleep as much as you can. Most people bounce back within 3-5 days. We'll pick up right where you left off once you're feeling better!"
    ))

    examples.append(coach_example(sys_prompt,
        'I have the flu\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Red (IllnessRisk)\n\n[INTERPRETER]\n{"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "I have the flu"}',
        "That's no fun. The flu hits hard, so let your body focus on getting well. No training until you're symptom-free — pushing through would just make it worse and risk a longer setback. Drink plenty of fluids, get rest, and give yourself permission to do nothing for a few days. Your fitness won't disappear. When you feel better, we'll ease back in gently."
    ))

    examples.append(coach_example(sys_prompt,
        'Ik ben ziek\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (IllnessRisk)\n\n[INTERPRETER]\n{"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "Ik ben ziek"}',
        "Beterschap! Rust is nu het allerbelangrijkste. Je lichaam heeft alle energie nodig om te herstellen — trainen zou dat alleen maar vertragen. Drink genoeg, slaap lekker uit, en luister naar je lijf. Je fitheid verdwijnt echt niet in een paar dagen. Zodra je je weer goed voelt, bouwen we rustig weer op!"
    ))

    # ── 2. Sick athlete wanting to train → push back warmly ──────────

    examples.append(coach_example(sys_prompt,
        'Can I still do a light workout?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (IllnessRisk)\n\nHISTORY:\n  [user]: I\'m sick with a cold\n  [assistant]: Rest is the priority. Your body needs energy to fight this off.\n\n[INTERPRETER]\n{"action": "answer_question", "question": "Can I still do a light workout while sick?", "free_text": "Can I still do a light workout?"}',
        "I know it's tempting, but even a light workout when you're sick makes your immune system work harder. Your body can't fight the infection AND recover from exercise at the same time. A light walk is fine if you feel up to it, but anything more should wait until you're symptom-free for at least 24 hours. Trust me — the rest now means you come back stronger."
    ))

    # ── 3. Session follow-up questions ───────────────────────────────

    examples.append(coach_example(sys_prompt,
        'How much rest in between?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 60min "5 x 3min Z5 / 3min Z1" (peak phase)\n\nHISTORY:\n  [user]: I want a hard 60 minute run\n  [assistant]: Your session: 5 x 3min hard efforts with 3min easy recovery between each.\n\n[INTERPRETER]\n{"action": "explain", "question": "How much rest in between?", "free_text": "How much rest in between?"}',
        "You've got 3 minutes of easy jogging between each hard effort. That's enough time for your heart rate to drop and your breathing to settle before the next one. The rest should feel genuinely easy — just shuffle along. If you're still gasping when the next interval starts, slow down the recovery even more."
    ))

    examples.append(coach_example(sys_prompt,
        'What pace should I run the intervals?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z4 75min "4 x 6min Z4 / 5min Z1" (build phase)\n\n[INTERPRETER]\n{"action": "explain", "question": "What pace should I run the intervals?", "free_text": "What pace should I run the intervals?"}',
        "For these threshold efforts, aim for a pace that feels 'comfortably hard' — you can speak in short sentences but wouldn't want to hold a conversation. Think of it as your 1-hour race effort. It should feel controlled and sustainable for the full 6 minutes, not all-out. If you have a heart rate monitor, you're looking at roughly 85-90% of max."
    ))

    examples.append(coach_example(sys_prompt,
        'Hoeveel rust ertussen?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z5 60min "5 x 3min Z5 / 3min Z1" (peak phase)\n\nHISTORY:\n  [user]: Ik wil een zware training van 60 minuten\n  [assistant]: Je sessie: 5 x 3min hard met 3min rustig ertussen.\n\n[INTERPRETER]\n{"action": "explain", "question": "Hoeveel rust ertussen?", "free_text": "Hoeveel rust ertussen?"}',
        "Je hebt 3 minuten rustig joggen tussen elke harde inspanning. Dat is genoeg om je hartslag te laten zakken en op adem te komen. De rust moet echt makkelijk voelen — gewoon lekker soepel doordraven. Als je nog steeds hijgt wanneer het volgende interval begint, neem dan iets meer rust."
    ))

    # ── 4. Better session presentations ──────────────────────────────

    examples.append(coach_example(sys_prompt,
        'I want to train intensive for an hour\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Session: Z4 60min "3 x 6min Z4 / 5min Z1" (build phase)\n\n[INTERPRETER]\n{"action": "create_workout", "sport": "run", "goal": "threshold", "time_available_min": 60, "free_text": "I want to train intensive for an hour"}',
        "Great — here's your threshold session! Start with a 10-minute warm-up jog, then hit 3 x 6 minutes at a comfortably hard effort with 5 minutes easy between each. Finish with a 10-minute cool-down. The hard parts should feel like your 1-hour race pace — challenging but controlled. This builds your ability to hold a strong effort over longer distances."
    ))

    examples.append(coach_example(sys_prompt,
        'Give me a hard ride, 75 minutes\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Session: Z5 75min "7 x 3min Z5 / 3min Z1" (peak phase)\n\n[INTERPRETER]\n{"action": "create_workout", "sport": "bike", "goal": "vo2", "time_available_min": 75, "free_text": "Give me a hard ride, 75 minutes"}',
        "Here's your hard session! Warm up for 15 minutes of easy spinning, then 7 x 3 minutes all-out effort with 3 minutes of easy pedaling between each. Cool down for the remaining time. The hard intervals should feel like a real push — breathing hard, legs burning. This is about building your top-end power. Commit to each effort fully!"
    ))

    # ── 5. Replan responses with warmth ──────────────────────────────

    examples.append(coach_example(sys_prompt,
        'I need to skip today, I\'m too tired\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (Overreached)\n\n[INTERPRETER]\n{"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "I need to skip today, I\'m too tired"}',
        "Completely understand — listening to your body is one of the smartest training decisions you can make. Your readiness is low right now, and pushing through fatigue is how injuries happen. Take today off, sleep well, eat well, and let your body absorb the training you've already done. Tomorrow is a new day."
    ))

    examples.append(coach_example(sys_prompt,
        'Ik sla vandaag over, te moe\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow (Accumulated)\n\n[INTERPRETER]\n{"action": "replan", "replan_type": "skip_today", "sport": "bike", "free_text": "Ik sla vandaag over, te moe"}',
        "Goed dat je naar je lichaam luistert — dat is slim trainen. Je bent de laatste tijd flink bezig geweest en je lichaam heeft wat extra rust nodig. Neem vandaag vrij, eet goed, slaap lekker, en morgen kijken we verder. Rust is ook trainen!"
    ))

    # ── 6. Gathering follow-up — coach asks for intensity ────────────

    examples.append(coach_example(sys_prompt,
        'I want to do a bike workout, I have 75 minutes\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green (Recovered)\n- Level: intermediate\n\n[INTERPRETER]\n{"action": "clarify", "missing": ["goal"], "clarify_message": "What kind of workout are you looking for?", "free_text": "I want to do a bike workout, I have 75 minutes"}',
        "75 minutes on the bike — nice! Before I set up your session, how are you feeling today? Would you like something easy and steady, a session with some harder efforts mixed in, or a proper hard workout? On a scale of 0-10, how hard do you want to push today?"
    ))

    examples.append(coach_example(sys_prompt,
        'Ik heb een uur de tijd voor een looptraining\n\nCONTEXT:\n- Sport: running\n- Readiness: Green (Recovered)\n- Level: intermediate\n\n[INTERPRETER]\n{"action": "clarify", "missing": ["goal"], "clarify_message": "Wat voor training wil je?", "free_text": "Ik heb een uur de tijd voor een looptraining"}',
        "Een uur hardlopen, mooi! Hoe voel je je vandaag? Wil je lekker rustig op tempo lopen, of heb je zin in iets pittigers met intervallen? Op een schaal van 0-10, hoe hard wil je het maken?"
    ))

    return examples


def main():
    parser = argparse.ArgumentParser(description="Add supplementary training examples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print examples without writing to files")
    args = parser.parse_args()

    # Load system prompts from existing examples
    interp_sys = get_interpreter_system_prompt()
    coach_sys = get_coach_system_prompt()

    interp_examples = build_interpreter_examples(interp_sys)
    coach_examples = build_coach_examples(coach_sys)

    print(f"  Supplementary examples prepared:")
    print(f"    Interpreter: {len(interp_examples)} new examples")
    print(f"    Coach:       {len(coach_examples)} new examples")
    print()

    if args.dry_run:
        print("  === DRY RUN — not writing to files ===\n")
        for i, ex in enumerate(interp_examples, 1):
            user = ex["messages"][1]["content"].split("\n")[0]
            assistant = json.loads(ex["messages"][2]["content"])
            print(f"  INTERP {i}: {user[:60]}")
            print(f"    → action={assistant.get('action')}, goal={assistant.get('goal', '(none)')}")
            print()
        for i, ex in enumerate(coach_examples, 1):
            user = ex["messages"][1]["content"].split("\n")[0]
            resp = ex["messages"][2]["content"][:80]
            print(f"  COACH {i}: {user[:60]}")
            print(f"    → {resp}...")
            print()
        return

    # Count existing examples
    with open(INTERPRETER_FILE) as f:
        existing_interp = sum(1 for _ in f)
    with open(COACH_FILE) as f:
        existing_coach = sum(1 for _ in f)

    # Append to source files
    with open(INTERPRETER_FILE, "a") as f:
        for ex in interp_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(COACH_FILE, "a") as f:
        for ex in coach_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Interpreter: {existing_interp} → {existing_interp + len(interp_examples)} examples")
    print(f"  Coach:       {existing_coach} → {existing_coach + len(coach_examples)} examples")
    print()
    print(f"  Next steps:")
    print(f"    1. Rebuild unified data:")
    print(f"       python prepare_v6_data.py --update-prompts --inject-knowledge")
    print(f"    2. Retrain:")
    print(f"       python finetune_qwen3.py train --mode unified")


if __name__ == "__main__":
    main()
