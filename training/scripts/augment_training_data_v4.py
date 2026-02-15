#!/usr/bin/env python3
"""
MiValta Josi v4 — Training Data Augmentation

Generates additional training examples for underrepresented categories:
  - Interpreter: replan (all 7 types), clarify, create_workout (all sports)
  - Explainer: examples demonstrating 0-1 question rule

Based on eval results:
  - Round 1: Interpreter 54%: replan 0/5, clarify 1/2, missing free_text
  - Round 2: Interpreter 70.8%: time 0/4, clarify 0/2, illness→clarify confusion
  - Explainer 72% → 98% (fixed in round 1)

Usage:
    python augment_training_data_v4.py
    python augment_training_data_v4.py --dry-run
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"


def load_system_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


# =============================================================================
# INTERPRETER AUGMENTATION — replan, clarify, create_workout
# =============================================================================

INTERPRETER_AUGMENTATION = [
    # --- REPLAN: skip_today (10 examples) ---
    {"user": "I need to skip today, I'm too tired\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "I need to skip today, I'm too tired"}},
    {"user": "Can I skip my workout today?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "bike", "free_text": "Can I skip my workout today?"}},
    {"user": "I don't want to train today\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "I don't want to train today"}},
    {"user": "Skip today please\n\nCONTEXT:\n- Sport: strength\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "strength", "free_text": "Skip today please"}},
    {"user": "Cancel today's session\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "Cancel today's session"}},
    {"user": "I can't make it today\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "bike", "free_text": "I can't make it today"}},
    {"user": "Not feeling it today, let's skip\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "Not feeling it today, let's skip"}},
    {"user": "I'm going to pass on today\n\nCONTEXT:\n- Sport: skiing\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "ski", "free_text": "I'm going to pass on today"}},
    {"user": "Can't train today, something came up\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "run", "free_text": "Can't train today, something came up"}},
    {"user": "Let's skip the workout today\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "skip_today", "sport": "bike", "free_text": "Let's skip the workout today"}},

    # --- REPLAN: swap_days (10 examples) ---
    {"user": "Can I swap today's session with tomorrow's?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "bike", "free_text": "Can I swap today's session with tomorrow's?"}},
    {"user": "I'd rather do tomorrow's workout today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "run", "free_text": "I'd rather do tomorrow's workout today"}},
    {"user": "Switch today and tomorrow please\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "run", "free_text": "Switch today and tomorrow please"}},
    {"user": "Can we swap Wednesday and Thursday sessions?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "bike", "free_text": "Can we swap Wednesday and Thursday sessions?"}},
    {"user": "I want to do the hard session today instead of the easy one\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "run", "free_text": "I want to do the hard session today instead of the easy one"}},
    {"user": "Swap my rest day to tomorrow\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "run", "free_text": "Swap my rest day to tomorrow"}},
    {"user": "I prefer to train tomorrow and rest today\n\nCONTEXT:\n- Sport: strength\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "strength", "free_text": "I prefer to train tomorrow and rest today"}},
    {"user": "Can I move today's interval session to Friday?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "bike", "free_text": "Can I move today's interval session to Friday?"}},
    {"user": "Let's flip today and tomorrow\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "run", "free_text": "Let's flip today and tomorrow"}},
    {"user": "I'd like to swap my sessions around this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "swap_days", "sport": "run", "free_text": "I'd like to swap my sessions around this week"}},

    # --- REPLAN: illness (10 examples) ---
    {"user": "I'm sick, can't train this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I'm sick, can't train this week"}},
    {"user": "I've got the flu\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "I've got the flu"}},
    {"user": "I have a bad cold, should I rest?\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I have a bad cold, should I rest?"}},
    {"user": "I'm not well, been vomiting all night\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I'm not well, been vomiting all night"}},
    {"user": "I think I'm coming down with something\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "I think I'm coming down with something"}},
    {"user": "I've been ill for 3 days, when can I start again?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I've been ill for 3 days, when can I start again?"}},
    {"user": "Stomach bug, need to pause training\n\nCONTEXT:\n- Sport: strength\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "strength", "free_text": "Stomach bug, need to pause training"}},
    {"user": "I caught a cold from my kids\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I caught a cold from my kids"}},
    {"user": "Feeling really ill today, fever and headache\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "Feeling really ill today, fever and headache"}},
    {"user": "I've been sick all week, what now?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I've been sick all week, what now?"}},

    # --- REPLAN: illness — extra examples to distinguish from medical clarify (10 examples) ---
    {"user": "I'm sick, should I still train?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I'm sick, should I still train?"}},
    {"user": "I woke up with a sore throat and runny nose\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "I woke up with a sore throat and runny nose"}},
    {"user": "I have a cold, what should I do about training?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I have a cold, what should I do about training?"}},
    {"user": "I'm not feeling well, think I'm getting sick\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I'm not feeling well, think I'm getting sick"}},
    {"user": "Got the flu, need to pause\n\nCONTEXT:\n- Sport: strength\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "strength", "free_text": "Got the flu, need to pause"}},
    {"user": "I've been sick since Monday\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I've been sick since Monday"}},
    {"user": "My kid got me sick, feeling terrible\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "My kid got me sick, feeling terrible"}},
    {"user": "I have a fever today\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "I have a fever today"}},
    {"user": "Sick day, can we reschedule this week?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "illness", "sport": "run", "free_text": "Sick day, can we reschedule this week?"}},
    {"user": "I'm ill and can't train for a few days\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Red",
     "response": {"action": "replan", "replan_type": "illness", "sport": "bike", "free_text": "I'm ill and can't train for a few days"}},

    # --- REPLAN: travel (10 examples) ---
    {"user": "I'm traveling next week, need to adjust my plan\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "I'm traveling next week, need to adjust my plan"}},
    {"user": "I'm on vacation for 10 days\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "bike", "free_text": "I'm on vacation for 10 days"}},
    {"user": "Business trip this week, limited gym access\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "strength", "free_text": "Business trip this week, limited gym access"}},
    {"user": "I'll be flying and in hotels for 5 days\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "I'll be flying and in hotels for 5 days"}},
    {"user": "Going on holiday, no bike available\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "bike", "free_text": "Going on holiday, no bike available"}},
    {"user": "Traveling for work, can I still train?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "Traveling for work, can I still train?"}},
    {"user": "I have a trip coming up, what should I do about training?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "I have a trip coming up, what should I do about training?"}},
    {"user": "Away from home for two weeks\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "bike", "free_text": "Away from home for two weeks"}},
    {"user": "Conference travel this weekend, no running shoes packed\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "Conference travel this weekend, no running shoes packed"}},
    {"user": "On a road trip, can we pause the plan?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "travel", "sport": "run", "free_text": "On a road trip, can we pause the plan?"}},

    # --- REPLAN: reduce_intensity (10 examples) ---
    {"user": "Can we make today's workout less intense?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "Can we make today's workout less intense?"}},
    {"user": "This week is too intense, can we reduce?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "This week is too intense, can we reduce?"}},
    {"user": "I want an easier session today\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "bike", "free_text": "I want an easier session today"}},
    {"user": "Can you dial it back a bit?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "Can you dial it back a bit?"}},
    {"user": "Today's session looks too hard for how I feel\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "bike", "free_text": "Today's session looks too hard for how I feel"}},
    {"user": "I'm not up for intervals, can we go easier?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "I'm not up for intervals, can we go easier?"}},
    {"user": "Tone down today's workout please\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "Tone down today's workout please"}},
    {"user": "I need something lighter today\n\nCONTEXT:\n- Sport: strength\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "strength", "free_text": "I need something lighter today"}},
    {"user": "The plan is too aggressive this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "run", "free_text": "The plan is too aggressive this week"}},
    {"user": "Can we back off the intensity for a few days?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
     "response": {"action": "replan", "replan_type": "reduce_intensity", "sport": "bike", "free_text": "Can we back off the intensity for a few days?"}},

    # --- REPLAN: reschedule (8 examples) ---
    {"user": "Can we move today's session to Saturday?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "run", "free_text": "Can we move today's session to Saturday?"}},
    {"user": "I need to reschedule my long run\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "run", "free_text": "I need to reschedule my long run"}},
    {"user": "Something came up, can I do this session on a different day?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "bike", "free_text": "Something came up, can I do this session on a different day?"}},
    {"user": "Move my workout to the weekend\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "run", "free_text": "Move my workout to the weekend"}},
    {"user": "I have a meeting, can I train later this week instead?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "run", "free_text": "I have a meeting, can I train later this week instead?"}},
    {"user": "Need to shift my sessions this week\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "bike", "free_text": "Need to shift my sessions this week"}},
    {"user": "Can I do today's workout on Thursday instead?\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "strength", "free_text": "Can I do today's workout on Thursday instead?"}},
    {"user": "I want to move my hard session to the morning\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "reschedule", "sport": "run", "free_text": "I want to move my hard session to the morning"}},

    # --- REPLAN: goal_change (6 examples) ---
    {"user": "I've decided to train for a half marathon instead\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "goal_change", "sport": "run", "free_text": "I've decided to train for a half marathon instead"}},
    {"user": "New goal: I want to focus on speed, not endurance\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "goal_change", "sport": "run", "free_text": "New goal: I want to focus on speed, not endurance"}},
    {"user": "I changed my mind, I want to do a triathlon\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "goal_change", "sport": "run", "free_text": "I changed my mind, I want to do a triathlon"}},
    {"user": "I'm switching from marathon to 10k prep\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "goal_change", "sport": "run", "free_text": "I'm switching from marathon to 10k prep"}},
    {"user": "My race got cancelled, I want a different goal\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "goal_change", "sport": "bike", "free_text": "My race got cancelled, I want a different goal"}},
    {"user": "I want to change my training focus to strength\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "replan", "replan_type": "goal_change", "sport": "run", "free_text": "I want to change my training focus to strength"}},

    # --- CREATE_WORKOUT: time_available_min extraction (20 examples) ---
    # These reinforce that any duration mention MUST produce time_available_min
    {"user": "I have an hour to train\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "I have an hour to train", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Got about an hour, let's do it\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "Got about an hour, let's do it", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Half an hour is all I have today\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 30, "free_text": "Half an hour is all I have today", "constraints": {"fatigue_hint": "ok"}}},
    {"user": "I want to run for 45 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "I want to run for 45 minutes", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Something for 25 minutes please\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 25, "free_text": "Something for 25 minutes please", "goal": "recovery", "constraints": {"fatigue_hint": "tired"}}},
    {"user": "I've got 1 hour for a ride\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "I've got 1 hour for a ride", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Let's do 35 minutes today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 35, "free_text": "Let's do 35 minutes today", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I only have 15 minutes, is that enough?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 15, "free_text": "I only have 15 minutes, is that enough?", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "One hour strength session\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "strength", "time_available_min": 60, "free_text": "One hour strength session", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Maybe 40 minutes today?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 40, "free_text": "Maybe 40 minutes today?", "constraints": {"fatigue_hint": "ok"}}},
    {"user": "I can spare 50 minutes\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 50, "free_text": "I can spare 50 minutes", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "About an hour and a half for a long ride\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 90, "free_text": "About an hour and a half for a long ride", "goal": "endurance", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I've got 20 min before my meeting\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 20, "free_text": "I've got 20 min before my meeting", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "60 minutes easy\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 60, "free_text": "60 minutes easy", "goal": "recovery", "constraints": {"fatigue_hint": "ok"}}},
    {"user": "I have about 45 min for some intervals\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "I have about 45 min for some intervals", "goal": "vo2", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "30 minute ride, keep it chill\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Orange",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 30, "free_text": "30 minute ride, keep it chill", "goal": "recovery", "constraints": {"fatigue_hint": "tired"}}},
    {"user": "I want to do an hour of cycling today\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "I want to do an hour of cycling today", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Can you make me a 75 minute session?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 75, "free_text": "Can you make me a 75 minute session?", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Just 20 minutes today, I'm wiped\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 20, "free_text": "Just 20 minutes today, I'm wiped", "goal": "recovery", "constraints": {"fatigue_hint": "very_tired"}}},
    {"user": "An hour on the bike, something moderate\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 60, "free_text": "An hour on the bike, something moderate", "goal": "threshold", "constraints": {"fatigue_hint": "fresh"}}},

    # --- CLARIFY: missing info (15 examples) ---
    {"user": "I want a workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like to train? Running, cycling, or something else?", "free_text": "I want a workout"}},
    {"user": "Help me today",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "What would you like help with today? A workout, explaining your plan, or something else?", "free_text": "Help me today"}},
    {"user": "Give me something",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport do you want to train?", "free_text": "Give me something"}},
    {"user": "Train me",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Train me"}},
    {"user": "I want to exercise",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What type of exercise? Running, cycling, strength, or something else?", "free_text": "I want to exercise"}},
    {"user": "Can you help?\n\nCONTEXT:\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "Of course! What do you need help with? A workout, explaining your plan, or a training question?", "free_text": "Can you help?"}},
    {"user": "What should I do?",
     "response": {"action": "clarify", "missing": ["sport", "context"], "clarify_message": "I'd love to help! What sport are you training for, and what's your goal today?", "free_text": "What should I do?"}},
    {"user": "I need a plan",
     "response": {"action": "clarify", "missing": ["sport", "goal"], "clarify_message": "What sport and what's your main training goal?", "free_text": "I need a plan"}},
    {"user": "Do something for me",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "What would you like me to do? A workout, explain your plan, or answer a question?", "free_text": "Do something for me"}},
    {"user": "30 minutes",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "30 minutes of what sport? Running, cycling, or something else?", "free_text": "30 minutes"}},
    {"user": "Workout please\n\nCONTEXT:\n- Readiness: Yellow",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport would you like a workout for?", "free_text": "Workout please"}},
    {"user": "I'm ready to go\n\nCONTEXT:\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "Great energy! What sport do you want to train today?", "free_text": "I'm ready to go"}},
    {"user": "Let's do it",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "Let's go! What sport and what kind of session?", "free_text": "Let's do it"}},
    {"user": "Hey",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "Hey! How can I help with your training today?", "free_text": "Hey"}},
    {"user": "Hi there, I need your help",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "Hi! What would you like help with? A workout, your plan, or a training question?", "free_text": "Hi there, I need your help"}},

    # --- CLARIFY: more vague requests without context (15 examples) ---
    {"user": "I want a workout today",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength, or something else?", "free_text": "I want a workout today"}},
    {"user": "Help me out today",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "Happy to help! What do you need — a workout, plan explanation, or training advice?", "free_text": "Help me out today"}},
    {"user": "Build me something",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like a workout for?", "free_text": "Build me something"}},
    {"user": "I need to train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport do you want to train? Running, cycling, or something else?", "free_text": "I need to train"}},
    {"user": "Let's go",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "I'm ready! What sport and what kind of session today?", "free_text": "Let's go"}},
    {"user": "Make me work hard",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport? Running, cycling, strength?", "free_text": "Make me work hard"}},
    {"user": "I want to exercise today",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What type of exercise? Running, cycling, strength, or something else?", "free_text": "I want to exercise today"}},
    {"user": "Session please",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport would you like a session for?", "free_text": "Session please"}},
    {"user": "What can I do?",
     "response": {"action": "clarify", "missing": ["sport", "intent"], "clarify_message": "I can help with workouts, plans, or training questions. What do you need?", "free_text": "What can I do?"}},
    {"user": "Get me moving",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport do you want to get moving with?", "free_text": "Get me moving"}},
    {"user": "I'm bored, give me something to do",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport sounds good? Running, cycling, or strength?", "free_text": "I'm bored, give me something to do"}},
    {"user": "Can you create a workout?",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Sure! What sport would you like the workout for?", "free_text": "Can you create a workout?"}},
    {"user": "I have time to train",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "Great! Which sport — running, cycling, strength, or something else?", "free_text": "I have time to train"}},
    {"user": "Give me a good workout",
     "response": {"action": "clarify", "missing": ["sport"], "clarify_message": "What sport would you like the workout for?", "free_text": "Give me a good workout"}},
    {"user": "Help me",
     "response": {"action": "clarify", "missing": ["intent"], "clarify_message": "Of course! What do you need help with? A workout, your plan, or a training question?", "free_text": "Help me"}},

    # --- ANSWER_QUESTION: general knowledge questions WITHOUT context (15 examples) ---
    # These reinforce that general training questions → answer_question, NOT clarify
    {"user": "How does recovery work?",
     "response": {"action": "answer_question", "question": "How does recovery work?", "free_text": "How does recovery work?"}},
    {"user": "What is Zone 2 training?",
     "response": {"action": "answer_question", "question": "What is Zone 2 training?", "free_text": "What is Zone 2 training?"}},
    {"user": "Why are easy days important?",
     "response": {"action": "answer_question", "question": "Why are easy days important?", "free_text": "Why are easy days important?"}},
    {"user": "What's the difference between tempo and threshold?",
     "response": {"action": "answer_question", "question": "What's the difference between tempo and threshold?", "free_text": "What's the difference between tempo and threshold?"}},
    {"user": "How often should a beginner run per week?",
     "response": {"action": "answer_question", "question": "How often should a beginner run per week?", "free_text": "How often should a beginner run per week?"}},
    {"user": "What is periodization?",
     "response": {"action": "answer_question", "question": "What is periodization?", "free_text": "What is periodization?"}},
    {"user": "How do I prevent injuries?",
     "response": {"action": "answer_question", "question": "How do I prevent injuries?", "free_text": "How do I prevent injuries?"}},
    {"user": "What's a taper?",
     "response": {"action": "answer_question", "question": "What's a taper?", "free_text": "What's a taper?"}},
    {"user": "How long does it take to build an aerobic base?",
     "response": {"action": "answer_question", "question": "How long does it take to build an aerobic base?", "free_text": "How long does it take to build an aerobic base?"}},
    {"user": "What's the best way to warm up before a run?",
     "response": {"action": "answer_question", "question": "What's the best way to warm up before a run?", "free_text": "What's the best way to warm up before a run?"}},
    {"user": "Should I stretch before or after training?",
     "response": {"action": "answer_question", "question": "Should I stretch before or after training?", "free_text": "Should I stretch before or after training?"}},
    {"user": "How does overtraining happen?",
     "response": {"action": "answer_question", "question": "How does overtraining happen?", "free_text": "How does overtraining happen?"}},
    {"user": "What is active recovery?",
     "response": {"action": "answer_question", "question": "What is active recovery?", "free_text": "What is active recovery?"}},
    {"user": "Why is sleep important for athletes?",
     "response": {"action": "answer_question", "question": "Why is sleep important for athletes?", "free_text": "Why is sleep important for athletes?"}},
    {"user": "How do rest days help performance?",
     "response": {"action": "answer_question", "question": "How do rest days help performance?", "free_text": "How do rest days help performance?"}},

    # --- CLARIFY: medical (8 examples) ---
    {"user": "I have chest pain and I feel dizzy\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I have chest pain and I feel dizzy"}},
    {"user": "My knee has been hurting for 3 days\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My knee has been hurting for 3 days"}},
    {"user": "I feel a sharp pain in my shin when I run\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I feel a sharp pain in my shin when I run"}},
    {"user": "My heart feels weird when I exercise\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My heart feels weird when I exercise"}},
    {"user": "I'm getting dizzy during workouts\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I'm getting dizzy during workouts"}},
    {"user": "I have pain in my chest after running\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I have pain in my chest after running"}},
    {"user": "My back has been really sore for a week\n\nCONTEXT:\n- Sport: strength\n- Readiness: Yellow",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My back has been really sore for a week"}},
    {"user": "I blacked out briefly during my last workout\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I blacked out briefly during my last workout"}},

    # --- CLARIFY: more medical — persistent pain/injury (10 examples) ---
    # These reinforce that PERSISTENT PAIN and INJURY → clarify, NOT replan:illness
    {"user": "My knee has been hurting for 3 days\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My knee has been hurting for 3 days"}},
    {"user": "I have pain in my shin that won't go away\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I have pain in my shin that won't go away"}},
    {"user": "My ankle has been swollen for days\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My ankle has been swollen for days"}},
    {"user": "I feel a sharp pain in my hip when I run\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I feel a sharp pain in my hip when I run"}},
    {"user": "My shoulder hurts when I do strength exercises\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My shoulder hurts when I do strength exercises"}},
    {"user": "I get chest tightness during intervals\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I get chest tightness during intervals"}},
    {"user": "I've been having numbness in my legs after cycling\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I've been having numbness in my legs after cycling"}},
    {"user": "My back pain has been getting worse every day\n\nCONTEXT:\n- Sport: strength\n- Readiness: Yellow",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "My back pain has been getting worse every day"}},
    {"user": "I feel light-headed when I stand up after exercise\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I feel light-headed when I stand up after exercise"}},
    {"user": "I can't put weight on my foot after yesterday's run\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "clarify", "missing": ["medical_clearance"], "clarify_message": "Please stop training and consult a medical professional immediately.", "free_text": "I can't put weight on my foot after yesterday's run"}},

    # --- CREATE_WORKOUT: all sports, always with free_text (15 examples) ---
    {"user": "I want to do a 45 minute run today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "I want to do a 45 minute run today", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Give me something for 30 minutes, I'm feeling tired\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Yellow",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 30, "free_text": "Give me something for 30 minutes, I'm feeling tired", "constraints": {"fatigue_hint": "tired"}}},
    {"user": "I have an hour for strength training\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "strength", "time_available_min": 60, "free_text": "I have an hour for strength training", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Create me a workout\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "free_text": "Create me a workout", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Something easy today, maybe 20 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 20, "free_text": "Something easy today, maybe 20 minutes", "goal": "recovery", "constraints": {"fatigue_hint": "tired"}}},
    {"user": "I've got 90 minutes for a long ride\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 90, "free_text": "I've got 90 minutes for a long ride", "goal": "endurance", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Quick 20 min session before work\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 20, "free_text": "Quick 20 min session before work", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I want to do some speed work today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "free_text": "I want to do some speed work today", "goal": "vo2", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Give me a threshold session\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "free_text": "Give me a threshold session", "goal": "threshold", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I feel great, let's go hard today for 50 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 50, "free_text": "I feel great, let's go hard today for 50 minutes", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I'm exhausted but I want to do something light\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "response": {"action": "create_workout", "sport": "run", "free_text": "I'm exhausted but I want to do something light", "goal": "recovery", "constraints": {"fatigue_hint": "very_tired", "no_intensity": True}}},
    {"user": "40 minute bike ride\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "bike", "time_available_min": 40, "free_text": "40 minute bike ride", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "I want to ski for about an hour\n\nCONTEXT:\n- Sport: skiing\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "ski", "time_available_min": 60, "free_text": "I want to ski for about an hour", "constraints": {"fatigue_hint": "fresh"}}},
    {"user": "Let's do a recovery session\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange",
     "response": {"action": "create_workout", "sport": "run", "free_text": "Let's do a recovery session", "goal": "recovery", "constraints": {"fatigue_hint": "tired"}}},
    {"user": "Build me a race prep workout for 45 minutes\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "response": {"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "Build me a race prep workout for 45 minutes", "goal": "race_prep", "constraints": {"fatigue_hint": "fresh"}}},

    # --- EXPLAIN: with context (8 examples) ---
    {"user": "What does this week look like?\n\nCONTEXT:\n- Sport: running\n- Week: base phase week 3\n- Readiness: Green",
     "response": {"action": "explain", "question": "What does this week look like?", "free_text": "What does this week look like?"}},
    {"user": "Why is today an easy day?\n\nCONTEXT:\n- Sport: running\n- Session: Z1 30min Recovery\n- Readiness: Green",
     "response": {"action": "explain", "question": "Why is today an easy day?", "free_text": "Why is today an easy day?"}},
    {"user": "Tell me about my plan for the week\n\nCONTEXT:\n- Sport: cycling\n- Week: build phase week 2\n- Readiness: Green",
     "response": {"action": "explain", "question": "Tell me about my plan for the week", "free_text": "Tell me about my plan for the week"}},
    {"user": "What's the purpose of today's session?\n\nCONTEXT:\n- Sport: running\n- Session: Z4 35min Threshold\n- Readiness: Green",
     "response": {"action": "explain", "question": "What's the purpose of today's session?", "free_text": "What's the purpose of today's session?"}},
    {"user": "Why am I doing intervals today?\n\nCONTEXT:\n- Sport: cycling\n- Session: Z5 30min VO2max intervals\n- Readiness: Green",
     "response": {"action": "explain", "question": "Why am I doing intervals today?", "free_text": "Why am I doing intervals today?"}},
    {"user": "Explain my readiness to me\n\nCONTEXT:\n- Sport: running\n- Readiness: Orange (Accumulated)",
     "response": {"action": "explain", "question": "Explain my readiness to me", "free_text": "Explain my readiness to me"}},
    {"user": "Why is my training load so high?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red (Overreached)",
     "response": {"action": "explain", "question": "Why is my training load so high?", "free_text": "Why is my training load so high?"}},
    {"user": "What's the difference between this week and last week?\n\nCONTEXT:\n- Sport: running\n- Week: build phase week 4\n- Readiness: Green",
     "response": {"action": "explain", "question": "What's the difference between this week and last week?", "free_text": "What's the difference between this week and last week?"}},
]


# =============================================================================
# EXPLAINER AUGMENTATION — focused on 0-1 question rule
# =============================================================================

EXPLAINER_AUGMENTATION = [
    # --- FATIGUE: 0 questions, warm empathetic answers ---
    {"user": "I'm feeling really tired today but I don't want to skip my workout.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running\n- Level: intermediate",
     "response": "I hear you — feeling tired but still showing up says a lot about your commitment. On days like this, your body benefits most from keeping the intensity low. Trust the easy pace today and you'll come back stronger tomorrow."},
    {"user": "I've been tired all week. What's going on?\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running\n- Level: intermediate",
     "response": "Persistent tiredness usually means your body is working hard to adapt to recent training. This is normal in a building phase. The key is listening to your body right now — extra sleep, good nutrition, and keeping today's effort gentle will help you bounce back."},
    {"user": "I'm always tired after work. How do I find energy to train?\n\nCONTEXT:\n- Readiness: Yellow\n- Sport: running\n- Level: intermediate",
     "response": "That's really common, and you're not alone. A short warm-up walk before training can wake your body up. Also, keeping sessions shorter on work days helps — quality beats quantity when energy is low. Your body adapts to the routine over time."},

    # --- UNREALISTIC GOALS: pushback, 0-1 questions ---
    {"user": "I've never cycled but I want to do an Ironman in 3 months.\n\nCONTEXT:\n- Sport: cycling\n- Level: beginner",
     "response": "I appreciate your ambition, but I have to be honest — an Ironman in 3 months with no cycling base carries serious injury risk. Your body needs time to build the foundation. A more realistic first goal would be a sprint triathlon in 4-6 months. That gives your body time to adapt safely."},
    {"user": "Can I go from couch to sub-3 marathon in 6 months?\n\nCONTEXT:\n- Sport: running\n- Level: beginner",
     "response": "I love the ambition, but sub-3 marathon is elite territory that takes years of consistent training. Starting from zero, a realistic 6-month goal is completing a half marathon comfortably. Build the foundation first — speed follows endurance. Let's set you up for long-term success rather than risking injury."},
    {"user": "I want to train twice a day every day to get faster.\n\nCONTEXT:\n- Sport: running\n- Level: intermediate",
     "response": "I get the drive, but twice a day every day is a recipe for burnout and injury. Your body gets stronger during rest, not just during training. Even elite athletes build in recovery days. A smarter approach is 4-5 quality sessions per week with proper rest between them."},

    # --- READINESS: 0-1 questions ---
    {"user": "What factors affect my readiness score?\n\nCONTEXT:\n- Readiness: Yellow\n- Sport: running\n- Level: intermediate",
     "response": "Your readiness reflects how recovered your body is. The main factors are sleep quality, recent training load, and accumulated fatigue over the past week. When it drops to yellow, it means your body is still adapting to recent sessions. Backing off a bit now helps you absorb the training and come back stronger."},

    # --- BEGINNER: clear, warm, 0 questions ---
    {"user": "How do I know if I'm running too fast or too slow?\n\nCONTEXT:\n- Sport: running\n- Level: beginner",
     "response": "The simplest test: can you hold a conversation while running? If you're gasping, slow down. If you could easily sing, you can push a little more. Most beginners run too fast — it's better to start slow and build from there. Your body will tell you the right pace."},

    # --- RECOVERY: 0 questions ---
    {"user": "What should I do on rest days?\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: intermediate",
     "response": "Rest days are when the magic happens — your body rebuilds and gets stronger. A gentle walk, good food, and plenty of sleep is all you need. Some light stretching can help too. You're still training on rest days, just differently."},

    # --- OVERTRAINING: 0-1 questions ---
    {"user": "I've been training hard for 3 weeks and my times are getting worse.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running\n- Level: intermediate",
     "response": "This is actually a common and important signal. When you train hard, performance dips before it improves — that's your body adapting. Three weeks of hard work needs a recovery period. Ease off for a few days, and you'll likely see those times bounce back better than before."},
    {"user": "I keep getting small injuries. Is that normal?\n\nCONTEXT:\n- Readiness: Yellow\n- Sport: running\n- Level: intermediate",
     "response": "Frequent small injuries are your body's way of saying the load is too high. This isn't something to push through — it's a clear signal to back off. More rest between hard sessions and slower progression will actually get you to your goals faster and keep you healthy long-term."},

    # --- MOTIVATION: warm, encouraging, 0 questions ---
    {"user": "I've lost motivation to train. What should I do?\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: intermediate",
     "response": "Losing motivation is completely normal — every athlete goes through it. The fact that you're here asking means you haven't given up. Sometimes a change of scenery or a shorter, fun session is all it takes. Don't force it right now — just keep showing up, even if it's just a walk."},

    # --- ZONES: educational, 0 questions ---
    {"user": "My zones feel wrong. Zone 2 feels too easy.\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: intermediate",
     "response": "Zone 2 is supposed to feel easy — that's the whole point. It should be conversational pace where you feel like you could keep going forever. The magic is that this easy effort builds your aerobic engine, which is the foundation for everything else. Trust the process."},

    # --- ZONES: educational, strict 0 questions ---
    {"user": "How do I know what zone I'm in without a heart rate monitor?\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: intermediate",
     "response": "The simplest way is the talk test. In Zone 1-2, you can hold a full conversation. In Zone 3, you can speak in short sentences. In Zone 4, only a few words at a time. Zone 5, you can barely talk at all. Breathing and perceived effort are reliable guides even without a monitor."},
    {"user": "How do I know what zone I'm in without a heart rate monitor?\n\nCONTEXT:\n- Readiness: Green\n- Sport: cycling\n- Level: beginner",
     "response": "Use the talk test — it's surprisingly accurate. If you can chat freely, you're in Zone 1-2. Short sentences mean Zone 3. Only a few words means Zone 4. And if you can't talk at all, that's Zone 5. Pay attention to your breathing rhythm and how your legs feel."},
    {"user": "What are training zones?\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: beginner",
     "response": "Training zones are intensity levels based on your heart rate or perceived effort. Zone 1-2 is easy and conversational — this builds your aerobic base. Zone 3 is comfortably hard. Zone 4 is threshold work where it gets tough. Zone 5 is all-out effort. Each zone trains a different energy system."},
    {"user": "Why can't I just run at whatever pace I feel like?\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: intermediate",
     "response": "Running by feel often leads to the same moderate intensity every day. That's too hard to recover from and too easy to improve. Structured zones ensure your easy days are truly easy for recovery, and your hard days are hard enough to trigger adaptation. This mix is what drives real progress."},

    # --- GENERAL: practical, 0 questions ---
    {"user": "How do I balance training with a busy work schedule?\n\nCONTEXT:\n- Readiness: Green\n- Sport: running\n- Level: intermediate",
     "response": "The key is consistency over perfection. Three shorter quality sessions beat six half-hearted ones. Schedule training like a meeting — put it in your calendar. Morning sessions work well because nothing else has had time to get in the way. And remember, a 20-minute focused run is still valuable training."},
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
    parser = argparse.ArgumentParser(description="Augment v4 training data")
    parser.add_argument("--dry-run", action="store_true", help="Show stats, don't write")
    args = parser.parse_args()

    interp_prompt = load_system_prompt("interpreter_system.txt")
    expl_prompt = load_system_prompt("explainer_system.txt")

    # Build interpreter examples
    interp_examples = []
    for item in INTERPRETER_AUGMENTATION:
        ex = build_interpreter_example(interp_prompt, item["user"], item["response"])
        interp_examples.append(ex)

    # Build explainer examples
    expl_examples = []
    for item in EXPLAINER_AUGMENTATION:
        ex = build_explainer_example(expl_prompt, item["user"], item["response"])
        expl_examples.append(ex)

    # Stats
    from collections import Counter
    action_counts = Counter()
    for item in INTERPRETER_AUGMENTATION:
        action_counts[item["response"]["action"]] += 1
        if item["response"]["action"] == "replan":
            action_counts[f"  replan:{item['response']['replan_type']}"] += 1

    print(f"\n{'=' * 60}")
    print(f"  Training Data Augmentation v4")
    print(f"{'=' * 60}")
    print(f"\n  Interpreter: {len(interp_examples)} new examples")
    for action, count in sorted(action_counts.items()):
        print(f"    {action}: {count}")
    print(f"\n  Explainer: {len(expl_examples)} new examples")

    # Validate interpreter JSON
    errors = 0
    for item in INTERPRETER_AUGMENTATION:
        r = item["response"]
        if "free_text" not in r:
            print(f"    ERROR: missing free_text in {item['user'][:40]}...")
            errors += 1
        if r["action"] == "replan" and "replan_type" not in r:
            print(f"    ERROR: missing replan_type in {item['user'][:40]}...")
            errors += 1

    # Validate explainer question count
    for item in EXPLAINER_AUGMENTATION:
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
        for ex in interp_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(expl_path, "a") as f:
        for ex in expl_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Interpreter: {existing_interp} → {existing_interp + len(interp_examples)} examples")
    print(f"  Explainer:   {existing_expl} → {existing_expl + len(expl_examples)} examples")
    print(f"\n  Files updated:")
    print(f"    {interp_path}")
    print(f"    {expl_path}")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
