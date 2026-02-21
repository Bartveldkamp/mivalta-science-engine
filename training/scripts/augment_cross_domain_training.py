#!/usr/bin/env python3
"""
MiValta — Cross-Domain Training Data Augmentation

Generates training examples that teach the interpreter and explainer to
use cross-domain state (injuries, readiness, load, memory, preferences)
for smarter coaching decisions.

These examples teach the models what the deterministic layer already does
in code — so the LoRA weights internalize the patterns and produce better
output even before the post-processor runs.

Usage:
    python augment_cross_domain_training.py

    Produces:
      training/data/train_interpreter_v6.jsonl   (original + augmented)
      training/data/val_interpreter_v6.jsonl     (original)
      training/data/train_explainer_v6.jsonl     (original + augmented)
      training/data/val_explainer_v6.jsonl       (original)
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def load_system_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text().strip()


# =========================================================================
# INTERPRETER — Cross-domain aware training examples
# =========================================================================
# These teach the interpreter to:
# 1. Use STATE to override athlete requests when body says no
# 2. Use known_sport from STATE to skip clarification
# 3. Factor active injuries into constraints
# 4. Emit chain-of-thought reasoning line before JSON

INTERP_SYSTEM = load_system_prompt("interpreter_system.txt")

INTERPRETER_AUGMENTATION = [
    # --- READINESS OVERRIDE: Orange + wants intensity ---
    {
        "user": 'I want some hard intervals today\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"readiness_level":"Orange","readiness_trend":"declining","training_load":{"sessions_this_week":4,"trend":"increasing"}}\n</STATE>',
        "assistant": '> Orange readiness + declining + wants intensity. Override goal to recovery.\n{"action":"create_workout","sport":"run","goal":"recovery","constraints":{"fatigue_hint":"tired","readiness_override":"Orange declining — forced recovery"},"free_text":"I want some hard intervals today"}'
    },
    {
        "user": 'Give me a VO2 max session\n\nCONTEXT:\n- Readiness: Red (Overreached)\n- Sport: running\n- Level: advanced\n\n<STATE>\n{"readiness_level":"Red","readiness_trend":"declining"}\n</STATE>',
        "assistant": '> Red readiness. No intensity work. Force recovery.\n{"action":"create_workout","sport":"run","goal":"recovery","constraints":{"readiness_override":"Red — body needs rest, not VO2"},"free_text":"Give me a VO2 max session"}'
    },
    {
        "user": 'Push me hard today coach\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: cycling\n- Level: intermediate\n\n<STATE>\n{"readiness_level":"Orange","readiness_trend":"declining","training_load":{"sessions_this_week":5,"trend":"increasing"}}\n</STATE>',
        "assistant": '> Orange + declining + 5 sessions + increasing load = overreaching risk. Force recovery.\n{"action":"create_workout","sport":"bike","goal":"recovery","constraints":{"readiness_override":"Orange declining with high load — recovery day"},"free_text":"Push me hard today coach"}'
    },
    {
        "user": 'Threshold run today\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"readiness_level":"Yellow","readiness_trend":"declining"}\n</STATE>',
        "assistant": '> Yellow + declining. Wants threshold. Dial back to endurance.\n{"action":"create_workout","sport":"run","goal":"endurance","constraints":{"readiness_override":"Yellow declining — moderate is better today"},"free_text":"Threshold run today"}'
    },

    # --- INJURY AWARENESS ---
    {
        "user": 'Give me a 45-minute run\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"active_injuries":[{"area":"knee","severity":6}]}\n</STATE>',
        "assistant": '> Wants run but has active knee injury at 6/10. Add injury constraint.\n{"action":"create_workout","sport":"run","time_available_min":45,"goal":"recovery","constraints":{"injury":"knee 6/10 — avoid aggravation"},"free_text":"Give me a 45-minute run"}'
    },
    {
        "user": 'I want a strength session\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: strength\n- Level: intermediate\n\n<STATE>\n{"active_injuries":[{"area":"shoulder","severity":7}]}\n</STATE>',
        "assistant": '> Wants strength but has shoulder injury at 7/10. Too severe for strength.\n{"action":"create_workout","sport":"strength","goal":"recovery","constraints":{"injury":"shoulder 7/10 — forced recovery, no lifting"},"free_text":"I want a strength session"}'
    },
    {
        "user": 'Give me some intervals\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"active_injuries":[{"area":"shin","severity":4}]}\n</STATE>',
        "assistant": '> Wants intervals but has shin issue at 4/10. Mild enough to continue with caution.\n{"action":"create_workout","sport":"run","goal":"threshold","constraints":{"injury":"shin 4/10 — proceed with caution, stop if worsens"},"free_text":"Give me some intervals"}'
    },

    # --- KNOWN SPORT FROM STATE (skip clarification) ---
    {
        "user": 'Give me a workout\n\n<STATE>\n{"known_sport":"run"}\n</STATE>',
        "assistant": '> No sport in message but STATE has known_sport=run. Use it.\n{"action":"create_workout","sport":"run","goal":"endurance","free_text":"Give me a workout"}'
    },
    {
        "user": 'I want to train\n\n<STATE>\n{"known_sport":"bike"}\n</STATE>',
        "assistant": '> No sport specified but STATE says they cycle. Use known_sport.\n{"action":"create_workout","sport":"bike","goal":"endurance","free_text":"I want to train"}'
    },
    {
        "user": 'Make me something for 30 minutes\n\n<STATE>\n{"known_sport":"run","recent_fatigue":"tired"}\n</STATE>',
        "assistant": '> No sport but STATE.known_sport=run. Recent fatigue=tired. Easy session.\n{"action":"create_workout","sport":"run","time_available_min":30,"goal":"recovery","constraints":{"fatigue_hint":"tired"},"free_text":"Make me something for 30 minutes"}'
    },

    # --- HIGH LOAD AWARENESS ---
    {
        "user": 'Give me a 90-minute ride\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: cycling\n- Level: advanced\n\n<STATE>\n{"training_load":{"sessions_this_week":6,"trend":"increasing"}}\n</STATE>',
        "assistant": '> 6 sessions + increasing load. 90min ride will add too much volume. Note high load.\n{"action":"create_workout","sport":"bike","time_available_min":90,"goal":"endurance","constraints":{"load_warning":"6 sessions this week with increasing load — keep easy"},"free_text":"Give me a 90-minute ride"}'
    },
    {
        "user": 'Another session today\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"training_load":{"sessions_this_week":5,"trend":"increasing"},"readiness_level":"Green"}\n</STATE>',
        "assistant": '> 5 sessions + increasing. Careful about adding volume.\n{"action":"create_workout","sport":"run","goal":"endurance","constraints":{"load_warning":"5 sessions this week — keep it moderate"},"free_text":"Another session today"}'
    },

    # --- COMEBACK AFTER BREAK ---
    {
        "user": 'Give me a hard threshold run, 60 minutes\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"training_load":{"sessions_this_week":0,"trend":"stable"}}\n</STATE>',
        "assistant": '> Zero sessions this week. Wants threshold after break. Dial back to endurance.\n{"action":"create_workout","sport":"run","time_available_min":60,"goal":"endurance","constraints":{"comeback":"no recent sessions — start moderate"},"free_text":"Give me a hard threshold run, 60 minutes"}'
    },

    # --- FATIGUE DETECTION ---
    {
        "user": 'I\'m exhausted but I don\'t want to skip\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"readiness_level":"Yellow","readiness_trend":"declining"}\n</STATE>',
        "assistant": '> Exhausted + Yellow + declining. Discuss whether to train, not just build a workout.\n{"action":"explain","question":"Should I train when exhausted?","constraints":{"fatigue_hint":"very_tired"},"free_text":"I\'m exhausted but I don\'t want to skip"}'
    },
    {
        "user": 'I feel completely wiped out today\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"readiness_level":"Orange","readiness_trend":"declining","training_load":{"sessions_this_week":5,"trend":"increasing"}}\n</STATE>',
        "assistant": '> Wiped out + Orange + declining + 5 sessions increasing. Clearly needs rest.\n{"action":"explain","question":"Training when fatigued and overreaching","constraints":{"fatigue_hint":"very_tired"},"free_text":"I feel completely wiped out today"}'
    },

    # --- MEMORY USAGE ---
    {
        "user": 'Give me a workout\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Level: intermediate\n\nMEMORY:\n- Facts: primary sport: running | prefers morning sessions\n- Patterns: typical duration: 45 min',
        "assistant": '> No sport in message but MEMORY says runner, 45 min typical. Fill from memory.\n{"action":"create_workout","sport":"run","time_available_min":45,"goal":"endurance","free_text":"Give me a workout"}'
    },
    {
        "user": 'Something easy for today\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Level: intermediate\n\nMEMORY:\n- Facts: primary sport: cycling | has knee issue',
        "assistant": '> Memory says cyclist with knee issue. Easy session = recovery.\n{"action":"create_workout","sport":"bike","goal":"recovery","constraints":{"injury":"knee (from memory)"},"free_text":"Something easy for today"}'
    },

    # --- PATTERN AWARENESS ---
    {
        "user": 'Give me a long run, 90 minutes\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"known_patterns":["long_run causes knee_pain"]}\n</STATE>',
        "assistant": '> STATE has pattern: long runs cause knee pain. Warn via constraints.\n{"action":"create_workout","sport":"run","time_available_min":90,"goal":"endurance","constraints":{"pattern_warning":"long runs have caused knee pain before — extra warm-up recommended"},"free_text":"Give me a long run, 90 minutes"}'
    },

    # --- GREEN + GOOD = JUST DO IT ---
    {
        "user": 'Give me a 50-minute endurance run\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate',
        "assistant": '> Green, no injuries, clear request. Straightforward.\n{"action":"create_workout","sport":"run","time_available_min":50,"goal":"endurance","free_text":"Give me a 50-minute endurance run"}'
    },
    {
        "user": 'I want intervals today, 45 minutes\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: advanced\n\n<STATE>\n{"readiness_level":"Green","readiness_trend":"improving","training_load":{"sessions_this_week":3,"trend":"stable"}}\n</STATE>',
        "assistant": '> Green + improving + moderate load. Perfect for intensity.\n{"action":"create_workout","sport":"run","time_available_min":45,"goal":"threshold","free_text":"I want intervals today, 45 minutes"}'
    },

    # --- SLEEP / NUTRITION TOPICS WITH STATE ---
    {
        "user": 'I haven\'t been sleeping well, any tips?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: running\n- Level: intermediate\n\n<STATE>\n{"training_load":{"sessions_this_week":5,"trend":"increasing"}}\n</STATE>',
        "assistant": '> Sleep question + high training load. Contextual — load may be affecting sleep.\n{"action":"answer_question","question":"Sleep tips when training load is high","free_text":"I haven\'t been sleeping well, any tips?"}'
    },

    # --- TRIAGE STILL WORKS ---
    {
        "user": 'My knee hurts when I run\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate',
        "assistant": '> Pain during running. Injury triage.\n{"action":"injury_triage","pain_area":"knee","trigger":"running","free_text":"My knee hurts when I run"}'
    },
    {
        "user": 'I have a sharp pain in my back\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate',
        "assistant": '> Sharp pain = injury triage.\n{"action":"injury_triage","pain_area":"back","free_text":"I have a sharp pain in my back"}'
    },

    # --- REPLAN STILL WORKS ---
    {
        "user": 'I need to skip today, work meeting\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate',
        "assistant": '> Wants to skip. Replan.\n{"action":"replan","replan_type":"skip_today","free_text":"I need to skip today, work meeting"}'
    },
    {
        "user": 'I have the flu\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate',
        "assistant": '> Sick. Replan for illness, NOT medical emergency.\n{"action":"replan","replan_type":"illness","free_text":"I have the flu"}'
    },

    # --- EXPLAIN / ANSWER ---
    {
        "user": 'What is Zone 2 and why does everyone talk about it?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: cycling\n- Level: beginner',
        "assistant": '> General knowledge question about zones.\n{"action":"answer_question","question":"What is Zone 2?","free_text":"What is Zone 2 and why does everyone talk about it?"}'
    },
    {
        "user": 'What am I doing today?\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n- Session: Z2 60min \'Easy aerobic\' (base phase)',
        "assistant": '> Asking about their specific session. Explain.\n{"action":"explain","question":"What is today\'s session?","free_text":"What am I doing today?"}'
    },

    # --- MULTIPLE STATE SIGNALS ---
    {
        "user": 'Build me a hard cycling session, 60 minutes\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: cycling\n- Level: intermediate\n\n<STATE>\n{"readiness_level":"Orange","readiness_trend":"declining","active_injuries":[{"area":"hip","severity":4}],"training_load":{"sessions_this_week":4,"trend":"increasing"}}\n</STATE>',
        "assistant": '> Orange + declining + hip injury + high load. Multiple red flags. Force recovery.\n{"action":"create_workout","sport":"bike","time_available_min":60,"goal":"recovery","constraints":{"readiness_override":"Orange declining","injury":"hip 4/10","load_warning":"4 sessions increasing"},"free_text":"Build me a hard cycling session, 60 minutes"}'
    },

    # --- READINESS GREEN — DON'T OVER-RESTRICT ---
    {
        "user": 'Threshold intervals, 40 minutes\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: advanced\n\n<STATE>\n{"readiness_level":"Green","readiness_trend":"stable","training_load":{"sessions_this_week":3,"trend":"stable"}}\n</STATE>',
        "assistant": '> Green + stable + moderate load. Good to go for intensity.\n{"action":"create_workout","sport":"run","time_available_min":40,"goal":"threshold","free_text":"Threshold intervals, 40 minutes"}'
    },
]


# =========================================================================
# EXPLAINER — Cross-domain coaching playbook examples
# =========================================================================

EXPL_SYSTEM = load_system_prompt("explainer_sequential_system.txt")

EXPLAINER_AUGMENTATION = [
    # --- INJURY + WORKOUT ---
    {
        "user": 'Give me a 45-minute run\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"create_workout","sport":"run","goal":"recovery","known_injuries":["knee (6/10)"]}\nENGINE_PAYLOAD: {"action":"create_workout","sport":"run","goal":"recovery","constraints":{"injury":"knee 6/10"},"cross_domain":["With your knee at 6/10, run will likely aggravate it. Let\'s swap to cycling or swimming."],"message":"Building a 45min run session focused on recovery."}',
        "assistant": "With your knee still at 6/10, a full run isn't the move right now. Let's keep this to an easy 45-minute spin on the bike instead — same aerobic benefit, no impact on the knee. When it settles below a 3, we'll get you running again."
    },

    # --- READINESS OVERRIDE ---
    {
        "user": 'I want some hard intervals today\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"create_workout","sport":"run","goal":"recovery","readiness":"Orange","readiness_trend":"declining","constraints":{"readiness_override":"Orange declining"}}\nENGINE_PAYLOAD: {"action":"create_workout","sport":"run","goal":"recovery","cross_domain":["Your body is at Orange readiness and trending declining. High-intensity work right now risks overtraining. Let\'s swap to easy."],"message":"Building a run session focused on recovery."}',
        "assistant": "I know you want to push it today, but your body is telling a different story — you've been trending down and you're at Orange. An easy session today means you can actually hit those intervals hard when you're recovered. Trust the process."
    },

    # --- OVERREACHING ---
    {
        "user": 'Should I do another session today?\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"explain","readiness":"Orange","readiness_trend":"declining","load_trend":"increasing","sessions_this_week":6}\nENGINE_PAYLOAD: {"action":"explain","cross_domain":["Your training load is climbing but your body\'s response is heading the other way. That\'s a classic overreaching signal. A recovery week now will make you stronger next week."],"message":"Answer the athlete\'s question."}',
        "assistant": "Honestly? Not today. Your load has been climbing all week and your body's response is heading the other direction — that's textbook overreaching. A rest day now isn't losing fitness, it's protecting it. You'll come back stronger."
    },

    # --- HIGH LOAD WEEK ---
    {
        "user": 'Give me a 90-minute bike ride\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: cycling\n- Level: advanced\n\n[INTERPRETER]: {"action":"create_workout","sport":"bike","goal":"endurance","sessions_this_week":6,"load_trend":"increasing"}\nENGINE_PAYLOAD: {"action":"create_workout","sport":"bike","goal":"endurance","time_min":90,"cross_domain":["That\'s session #7 this week and your load has been climbing. Adding more volume risks diminishing returns. A recovery day today means better performance tomorrow."],"message":"Building a 90min bike session."}',
        "assistant": "That's a solid week already — 6 sessions in the bank. Your body can handle the ride, but after this much volume, the gains per session start dropping. If you're going to ride, keep it genuinely easy — zone 1-2, no pushing. Or honestly, a rest day might give you more than the 7th session would."
    },

    # --- COMEBACK ---
    {
        "user": 'I haven\'t trained in 2 weeks. Give me a hard threshold run.\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"create_workout","sport":"run","goal":"endurance","constraints":{"comeback":"no recent sessions"}}\nENGINE_PAYLOAD: {"action":"create_workout","sport":"run","goal":"endurance","cross_domain":["Welcome back! Since you\'ve been away, let\'s start with something moderate. Your cardiovascular system bounces back fast, but tendons and joints need a gentler ramp."],"message":"Building a run session focused on endurance."}',
        "assistant": "Welcome back! Your engine will come back fast — cardiovascular fitness is surprisingly resilient. But tendons and joints don't bounce back as quickly. Let's start with an easy to moderate run this week. Give it 7-10 days of building back and you'll be ready to hammer those thresholds again."
    },

    # --- CONSISTENT PROGRESS ---
    {
        "user": 'Give me a 50-minute endurance run\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"create_workout","sport":"run","goal":"endurance","readiness":"Green","readiness_trend":"improving","sessions_this_week":3,"load_trend":"increasing"}\nENGINE_PAYLOAD: {"action":"create_workout","sport":"run","goal":"endurance","time_min":50,"cross_domain":["Your body is responding well — load is building and readiness is holding. You\'re in a good spot to push a bit."],"message":"Building a 50min run session focused on endurance."}',
        "assistant": "You're in a really good rhythm right now — three sessions this week and your body is responding well. Here's your 50-minute endurance run. Keep it conversational pace and enjoy the miles. You've earned this good patch."
    },

    # --- FATIGUE + DELOAD ---
    {
        "user": 'I\'m exhausted, I can barely get out of bed\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"explain","readiness":"Orange","readiness_trend":"declining","constraints":{"fatigue_hint":"very_tired"}}\nENGINE_PAYLOAD: {"action":"explain","cross_domain":["3 days of declining readiness plus how you feel means your body needs a lighter week. Your plan will adjust."],"message":"Answer the athlete\'s question about training when exhausted."}',
        "assistant": "Your body is giving you a clear signal — three days of declining readiness plus feeling this drained means you need to back off. This isn't weakness, it's data. Take today completely off, and let's plan a lighter week. The fitness you've built isn't going anywhere."
    },

    # --- SLEEP + HIGH LOAD ---
    {
        "user": 'I haven\'t been sleeping well this week, any tips?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"answer_question","sessions_this_week":5,"load_trend":"increasing"}\nENGINE_PAYLOAD: {"action":"answer_question","question":"Sleep tips","cross_domain":["Poor sleep and increasing training load don\'t mix well. High training stress can actually disrupt sleep. Consider keeping today easy."],"message":"Answer about sleep."}',
        "assistant": "Sleep and training are deeply connected — when training load climbs like yours has this week, it can actually make sleep worse. The fix is counterintuitive: back off the intensity for a couple days. Keep today easy, prioritize getting to bed early, and you should see both sleep and performance bounce back."
    },

    # --- REPEATED INJURY ---
    {
        "user": 'My knee is acting up again\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"injury_triage","known_injuries":["knee (5/10) x3"]}\nENGINE_PAYLOAD: {"action":"injury_triage","cross_domain":["This is the 3rd time your knee has flared up. That kind of pattern usually means something structural needs attention. It\'s worth seeing a physio."],"message":"Assess knee pain."}',
        "assistant": "I hear you, and I want to be straight with you — this is the third time your knee has flared up. Once is bad luck, twice is a pattern, three times means something underneath needs attention. I really think it's worth seeing a physio. In the meantime, where exactly is the pain and how bad is it right now?"
    },

    # --- NUTRITION + POOR READINESS ---
    {
        "user": 'What should I eat after training?\n\nCONTEXT:\n- Readiness: Orange (Accumulated)\n- Sport: running\n- Level: intermediate\n\n[INTERPRETER]: {"action":"answer_question","readiness":"Orange"}\nENGINE_PAYLOAD: {"action":"answer_question","topic":"nutrition","cross_domain":["When you\'re this fatigued, recovery nutrition matters more. Focus on protein within 30 minutes after training and extra carbs tonight."],"message":"Answer about post-training nutrition."}',
        "assistant": "Good question, and especially important right now since your body is working hard to recover. Focus on protein within 30 minutes after your session — something like a shake or eggs. Then tonight, don't be shy with the carbs. When you're this fatigued, nutrition is one of the biggest levers you have."
    },

    # --- CLEAN WORKOUT (no issues) ---
    {
        "user": 'Give me a 40-minute threshold run\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: advanced\n\n[INTERPRETER]: {"action":"create_workout","sport":"run","goal":"threshold","readiness":"Green","readiness_trend":"stable"}\nENGINE_PAYLOAD: {"action":"create_workout","sport":"run","goal":"threshold","time_min":40,"message":"Building a 40min run session focused on threshold."}',
        "assistant": "Here's your 40-minute threshold session. Warm up for 10 minutes easy, then settle into that comfortably hard effort — you should be able to speak in short sentences but not hold a conversation. Cool down with 5-10 minutes easy to finish. Enjoy the work."
    },

    # --- GENERAL QUESTION ---
    {
        "user": 'What is Zone 2 and why does everyone talk about it?\n\nCONTEXT:\n- Readiness: Yellow (Productive)\n- Sport: cycling\n- Level: beginner\n\n[INTERPRETER]: {"action":"answer_question","question":"What is Zone 2?"}\nENGINE_PAYLOAD: {"action":"answer_question","question":"What is Zone 2?","message":"Explain Zone 2 to the athlete."}',
        "assistant": "Zone 2 is that easy effort where you can comfortably hold a conversation — you're working, but you're not straining. Everyone talks about it because it builds your aerobic engine without beating you up. Think of it as your fitness foundation. The more time you spend here, the faster you get everywhere else."
    },

    # --- MOTIVATION ---
    {
        "user": 'I missed a week of training. Have I ruined everything?\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: beginner\n\n[INTERPRETER]: {"action":"answer_question","question":"Impact of missing a week"}\nENGINE_PAYLOAD: {"action":"answer_question","question":"Impact of missing training","message":"Reassure about missed training."}',
        "assistant": "Not even close. A week off barely dents your fitness — research shows it takes 2-3 weeks before meaningful losses start. If anything, your body probably needed the break. Jump back in where you left off, maybe ease in for a session or two, and you'll be right back on track."
    },
]


def build_interp_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": INTERP_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_expl_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": EXPL_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main():
    print("=" * 60)
    print("  MiValta — Cross-Domain Training Data Augmentation")
    print("=" * 60)

    # Load existing data
    interp_train_path = DATA_DIR / "train_interpreter.jsonl"
    interp_val_path = DATA_DIR / "val_interpreter.jsonl"
    expl_train_path = DATA_DIR / "train_explainer_sequential.jsonl"
    expl_val_path = DATA_DIR / "val_explainer_sequential.jsonl"

    with open(interp_train_path) as f:
        interp_train = [json.loads(l) for l in f if l.strip()]
    with open(interp_val_path) as f:
        interp_val = [json.loads(l) for l in f if l.strip()]
    with open(expl_train_path) as f:
        expl_train = [json.loads(l) for l in f if l.strip()]
    with open(expl_val_path) as f:
        expl_val = [json.loads(l) for l in f if l.strip()]

    print(f"\n  Existing data:")
    print(f"    Interpreter: {len(interp_train)} train, {len(interp_val)} val")
    print(f"    Explainer:   {len(expl_train)} train, {len(expl_val)} val")

    # Build augmented examples
    new_interp = [build_interp_example(e["user"], e["assistant"])
                  for e in INTERPRETER_AUGMENTATION]
    new_expl = [build_expl_example(e["user"], e["assistant"])
                for e in EXPLAINER_AUGMENTATION]

    print(f"\n  New cross-domain examples:")
    print(f"    Interpreter: +{len(new_interp)} examples")
    print(f"    Explainer:   +{len(new_expl)} examples")

    # Combine
    interp_combined = interp_train + new_interp
    expl_combined = expl_train + new_expl

    print(f"\n  Combined totals:")
    print(f"    Interpreter: {len(interp_combined)} train, {len(interp_val)} val")
    print(f"    Explainer:   {len(expl_combined)} train, {len(expl_val)} val")

    # Write v6 files
    out_interp_train = DATA_DIR / "train_interpreter_v6.jsonl"
    out_interp_val = DATA_DIR / "val_interpreter_v6.jsonl"
    out_expl_train = DATA_DIR / "train_explainer_v6.jsonl"
    out_expl_val = DATA_DIR / "val_explainer_v6.jsonl"

    for path, data in [
        (out_interp_train, interp_combined),
        (out_interp_val, interp_val),
        (out_expl_train, expl_combined),
        (out_expl_val, expl_val),
    ]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(data)} examples → {path.name}")

    print(f"\n  Done! New training files ready.")
    print(f"\n  To train:")
    print(f"    python finetune_qwen25.py train --mode interpreter \\")
    print(f"      --train {out_interp_train} --val {out_interp_val}")
    print(f"    python finetune_qwen25.py train --mode explainer \\")
    print(f"      --train {out_expl_train} --val {out_expl_val}")


if __name__ == "__main__":
    main()
