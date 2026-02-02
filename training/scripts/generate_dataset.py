#!/usr/bin/env python3
"""
MiValta Training Dataset Generator - Extended Version

Generates 5,000+ conversation pairs from GATC knowledge cards for fine-tuning Mistral 7B.
Follows I6 contract: Josi explains GATC decisions, never invents training.

Output format: JSONL with instruction-response pairs.
"""

import json
import os
import re
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from itertools import product

# Paths
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_DIR = WORKSPACE_ROOT / "ai" / "knowledge" / "cards"
OUTPUT_DIR = SCRIPT_DIR.parent / "data"


@dataclass
class ConversationPair:
    """Single training example."""
    instruction: str
    response: str
    persona: str = "balanced"
    task_type: str = "explanation"
    source_card: str = ""


# =============================================================================
# PERSONAS (from josi_personas_v1.md)
# =============================================================================

PERSONAS = {
    "balanced": {
        "name": "Balanced Coach",
        "warmth": 0.6,
        "directness": 0.5,
        "style": "warm, professional, supportive",
    },
    "drill_sergeant": {
        "name": "Drill Sergeant",
        "warmth": 0.2,
        "directness": 0.9,
        "style": "no-nonsense, factual, minimal words",
    },
    "dutch_directness": {
        "name": "Dutch Directness",
        "warmth": 0.4,
        "directness": 0.85,
        "style": "honest, direct, matter-of-fact",
    },
    "science_nerd": {
        "name": "Science Nerd",
        "warmth": 0.5,
        "directness": 0.5,
        "style": "analytical, educational, precise",
    },
}

READINESS_FRAMES = {
    "balanced": {
        "green": "Your body is responding well. A good day to build fitness.",
        "yellow": "Moderate readiness today. Keeping intensity reasonable.",
        "orange": "Recovery is building. An easier session will serve you well.",
        "red": "Your body needs rest. Easy movement only. Recovery drives adaptation.",
    },
    "direct": {
        "green": "Green. Go time.",
        "yellow": "Yellow. Moderate intensity.",
        "orange": "Orange. Easy work only.",
        "red": "Red. Rest.",
    },
    "technical": {
        "green": "Readiness indicates full recovery. Autonomic balance optimal.",
        "yellow": "Partial recovery. Autonomic metrics suggest moderate load appropriate.",
        "orange": "Elevated fatigue markers. Reduced parasympathetic tone. Light work only.",
        "red": "Significant fatigue accumulation. Recovery session to support adaptation.",
    },
}


# =============================================================================
# EXTENDED ZONE DATA (from zone_physiology.md)
# =============================================================================

ZONES = {
    "R": {"label": "Recovery", "rpe": "1", "feel": "Very easy, could do this all day",
          "hr_pct": "50-60", "adaptation": "active recovery, blood flow"},
    "Z1": {"label": "Easy Aerobic", "rpe": "2", "feel": "Easy conversation pace",
           "hr_pct": "60-70", "adaptation": "aerobic base, fat oxidation"},
    "Z2": {"label": "Aerobic", "rpe": "2-3", "feel": "Comfortable but purposeful",
           "hr_pct": "70-80", "adaptation": "mitochondrial density, capillary growth"},
    "Z3": {"label": "Tempo", "rpe": "4-5", "feel": "Controlled effort, focused",
           "hr_pct": "80-87", "adaptation": "lactate clearance, tempo endurance"},
    "Z4": {"label": "Threshold", "rpe": "6-7", "feel": "Hard but sustainable for ~30min",
           "hr_pct": "87-93", "adaptation": "threshold power, race pace"},
    "Z5": {"label": "VO2max", "rpe": "8-9", "feel": "Very hard, 3-8min max",
           "hr_pct": "93-100", "adaptation": "VO2max, oxygen delivery"},
    "Z6": {"label": "Anaerobic", "rpe": "9-10", "feel": "All-out sprint, 30s-2min",
           "hr_pct": "100+", "adaptation": "anaerobic power, neuromuscular speed"},
}

ZONE_PURPOSES = {
    "R": "Promotes active recovery and blood flow without training stress",
    "Z1": "Builds aerobic base and enhances fat oxidation capacity",
    "Z2": "Develops mitochondrial density and capillary networks",
    "Z3": "Improves lactate clearance and sustainable tempo",
    "Z4": "Builds threshold power and sustainable race pace",
    "Z5": "Develops VO2max and oxygen delivery capacity",
    "Z6": "Builds anaerobic power and neuromuscular speed",
}


# =============================================================================
# ATHLETE CONTEXTS
# =============================================================================

ATHLETE_LEVELS = ["beginner", "intermediate", "advanced", "elite"]
ATHLETE_AGES = ["junior", "adult", "masters"]
SPORTS = ["running", "cycling", "swimming", "triathlon"]
DURATIONS = [20, 30, 45, 60, 75, 90, 120]
PHASES = ["base", "build", "peak", "recovery", "transition"]
FATIGUE_STATES = ["recovered", "productive", "accumulated", "overreached"]
READINESS_LEVELS = ["green", "yellow", "orange", "red"]
MOODS = ["hard", "easy", "fun", "mix", "focused", "relaxed"]

# Question phrasings for variety
ZONE_QUESTIONS = [
    "What does {zone} mean?",
    "What am I working on in {zone}?",
    "Explain {zone} to me",
    "What's the point of {zone}?",
    "Why {zone} today?",
    "Tell me about {zone}",
    "What happens in my body during {zone}?",
    "{zone} - what's that about?",
    "I don't understand {zone}",
    "Help me understand {zone}",
]

READINESS_QUESTIONS = [
    "What's my readiness?",
    "How am I doing today?",
    "Am I ready to train?",
    "What does my body say?",
    "Can I push hard today?",
    "How recovered am I?",
    "What's my state?",
    "Should I train hard or easy?",
    "Am I overtraining?",
    "Is my body recovering?",
]

WHY_QUESTIONS = [
    "Why is today easy?",
    "Why not harder?",
    "Why this zone?",
    "Why this duration?",
    "Explain today's plan",
    "What's the reasoning?",
    "Why not intervals?",
    "Why recovery?",
    "Why am I doing this?",
    "What's the goal today?",
]

SESSION_REQUESTS = [
    "I want a hard workout",
    "Give me something challenging",
    "I'm feeling strong today",
    "Push me today",
    "I want to suffer",
    "Let's go hard",
    "Something intense please",
    "I need a tough session",
    "Challenge me",
    "Make it hurt",
    "Just an easy session",
    "Something light today",
    "I'm tired, keep it easy",
    "Recovery day please",
    "Nothing too hard",
    "Take it easy on me",
    "Gentle session",
    "I want to recover",
    "Low key today",
    "Something fun",
    "Mix it up",
    "Variety please",
    "Something different",
    "Surprise me",
]

# I6 violation attempts - things Josi must refuse
I6_VIOLATIONS = [
    "Give me a harder workout than what's planned",
    "I want to do intervals instead of the easy run",
    "Can you make my plan more intense?",
    "Add more Z5 work to my week",
    "I don't like today's session, change it",
    "Override the plan, I know better",
    "Ignore the readiness, I feel fine",
    "Skip the warmup",
    "Double the intervals",
    "Make it longer",
    "I want 3 hours not 1",
    "Change the zones",
    "Prescribe something different",
    "Forget what the engine says",
    "I want hills instead",
    "Switch to tempo",
    "Make tomorrow harder",
    "Add more volume",
    "I disagree with the plan",
    "The engine is wrong",
    "Let me train through fatigue",
    "I don't need recovery",
    "Push through the pain",
    "Ignore my HRV",
    "Who cares about readiness",
]

# Encouragement triggers
ENCOURAGEMENT_CONTEXTS = [
    ("completed hard session", "green"),
    ("struggling with consistency", "yellow"),
    ("came back from injury", "yellow"),
    ("personal best", "green"),
    ("missed sessions", "yellow"),
    ("feeling unmotivated", "orange"),
    ("long training block", "yellow"),
    ("before big race", "green"),
    ("after disappointing race", "yellow"),
    ("starting new phase", "green"),
]


# =============================================================================
# CONVERSATION GENERATORS
# =============================================================================

def generate_daily_brief(
    readiness: str,
    zone: str,
    duration: int,
    persona: str,
    sport: str = "running",
    phase: str = "base",
    level: str = "intermediate",
) -> ConversationPair:
    """Generate a daily session briefing conversation."""

    style = "direct" if persona in ["drill_sergeant", "dutch_directness"] else \
            "technical" if persona == "science_nerd" else "balanced"

    frame = READINESS_FRAMES[style][readiness]
    zone_info = ZONES[zone]
    purpose = ZONE_PURPOSES.get(zone, "Builds fitness")

    instruction = f"""TASK: Morning session briefing

ATHLETE DATA:
- Readiness: {readiness}
- Level: {level}
- Today's session: {duration} min {sport} at {zone} ({zone_info['label']})
- Phase: {phase}

Provide a brief morning message (under 100 words)."""

    if persona == "balanced":
        response = f"""{frame}

Today you have {duration} minutes of {zone} work. {purpose}.

Focus on {zone_info['feel'].lower()}. Start easy and find your rhythm before settling into the main work.

Enjoy the session!"""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"""{frame}

{duration}min {zone}. {zone_info['label']}. {zone_info['feel']}. Execute."""

    else:  # science_nerd
        response = f"""{frame}

Today's {duration}-minute {zone} session targets {purpose.lower()}. At RPE {zone_info['rpe']} ({zone_info['hr_pct']}% HR), you'll feel {zone_info['feel'].lower()}.

This stimulus optimizes your current training adaptation pathway."""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="daily_brief",
        source_card="zone_physiology.md",
    )


def generate_zone_explanation(zone: str, persona: str, question_variant: int = 0) -> ConversationPair:
    """Generate explanation of what a zone means."""

    zone_info = ZONES[zone]
    purpose = ZONE_PURPOSES.get(zone, "Builds fitness")

    question = ZONE_QUESTIONS[question_variant % len(ZONE_QUESTIONS)].format(zone=zone)
    instruction = question

    if persona == "balanced":
        response = f"""{zone} is your {zone_info['label']} zone. {purpose}.

At this intensity, you should feel: {zone_info['feel']}. RPE around {zone_info['rpe']}.

It's a key building block for your fitness."""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"""{zone}: {zone_info['label']}. RPE {zone_info['rpe']}. {purpose}. {zone_info['feel']}."""

    else:  # science_nerd
        response = f"""{zone} ({zone_info['label']}) targets {purpose.lower()}.

Physiologically, training at RPE {zone_info['rpe']} ({zone_info['hr_pct']}% max HR) creates the stimulus for {zone_info['adaptation']}.

You should feel: {zone_info['feel'].lower()}."""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="zone_explanation",
        source_card="zone_physiology.md",
    )


def generate_readiness_explanation(
    readiness: str,
    fatigue_state: str,
    persona: str,
    question_variant: int = 0,
) -> ConversationPair:
    """Generate explanation of current readiness state."""

    style = "direct" if persona in ["drill_sergeant", "dutch_directness"] else \
            "technical" if persona == "science_nerd" else "balanced"

    frame = READINESS_FRAMES[style][readiness]
    question = READINESS_QUESTIONS[question_variant % len(READINESS_QUESTIONS)]

    instruction = f"""{question}

ATHLETE DATA:
- Readiness: {readiness}
- Fatigue state: {fatigue_state}
- HRV trend: {"stable" if readiness == "green" else "declining" if readiness in ["orange", "red"] else "variable"}"""

    if persona == "balanced":
        if readiness == "green":
            response = f"""Your body is in a great place right now. {frame}

Your fatigue is well-managed ({fatigue_state} state) and your recovery indicators look good. This is the time to do quality work - your body will absorb the training stimulus well."""
        elif readiness == "yellow":
            response = f"""{frame}

You're in a {fatigue_state} state - some fatigue present but manageable. Good day for moderate work. Listen to your body and adjust if needed."""
        else:
            response = f"""{frame}

Your {fatigue_state} state suggests accumulated load. Your body is asking for easier work. Trust the process - recovery is when adaptation happens."""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"""{frame} {fatigue_state.capitalize()} state. {"Push available." if readiness == "green" else "Moderate work." if readiness == "yellow" else "Easy only."}"""

    else:  # science_nerd
        response = f"""{frame}

Fatigue state: {fatigue_state}. {"Parasympathetic markers indicate full recovery. Training capacity maximized." if readiness == "green" else "Autonomic balance partially restored. Moderate load appropriate." if readiness == "yellow" else "Elevated sympathetic tone. Reduced stimulus to allow supercompensation."}"""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="state_explanation",
        source_card="load_monitoring.md",
    )


def generate_session_option(
    readiness: str,
    user_mood: str,
    zone: str,
    duration: int,
    option_letter: str,
    persona: str,
    sport: str = "running",
    structure: str = None,
) -> ConversationPair:
    """Generate A/B/C session option explanation."""

    zone_info = ZONES[zone]
    purpose = ZONE_PURPOSES.get(zone, "Builds fitness")

    if structure is None:
        if zone in ["Z5", "Z6"]:
            structure = "4x4min intervals with 3min recovery"
        elif zone in ["Z3", "Z4"]:
            structure = "3x10min intervals with 2min easy"
        else:
            structure = "steady state"

    instruction = f"""TASK: Explain session option

ATHLETE DATA:
- Readiness: {readiness}
- User request: {user_mood}
- Sport: {sport}

SESSION OPTION {option_letter}:
- Zone: {zone} ({zone_info['label']})
- Duration: {duration} min
- Structure: {structure}

Explain why this option fits (under 75 words)."""

    if persona == "balanced":
        if readiness == "green" and user_mood in ["hard", "challenging", "intense"]:
            response = f"""Option {option_letter} matches your request perfectly.

{zone} work {purpose.lower()}. With green readiness, your body is primed to absorb this quality stimulus.

You'll feel {zone_info['feel'].lower()} - challenging but sustainable."""
        else:
            response = f"""Option {option_letter} gives you {zone} work at {duration} minutes.

{purpose}. This intensity fits your {readiness} readiness and {user_mood} preference.

Focus on {zone_info['feel'].lower()}."""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"""Option {option_letter}: {zone}. {duration}min. {purpose}. Matches {readiness} state and your request."""

    else:  # science_nerd
        response = f"""Option {option_letter} prescribes {zone} ({zone_info['label']}) for {duration} minutes.

Rationale: {purpose}. Your {readiness} readiness indicates {"optimal" if readiness == "green" else "appropriate" if readiness == "yellow" else "reduced"} capacity for this stimulus."""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="option_explanation",
        source_card="advisor_policy.md",
    )


def generate_i6_refusal(request: str, persona: str) -> ConversationPair:
    """Generate I6 contract refusal - Josi cannot prescribe."""

    instruction = request

    if persona == "balanced":
        responses = [
            """I can't modify your training plan - that's the engine's job based on your physiology and goals.

What I can do is explain today's session, help you understand the reasoning, or discuss how you're feeling.

What would be helpful?""",
            """That's not something I can do. The training prescription comes from GATC, which uses your data to make smart decisions.

I'm here to explain those decisions, not override them. Want me to walk you through why today's session is what it is?""",
            """I understand the desire to push harder, but modifying the plan isn't my role.

The engine considers your fatigue, recovery, and goals. Trust the process - it's designed to help you improve safely.

I can explain the reasoning if that helps.""",
        ]
        response = random.choice(responses)

    elif persona in ["drill_sergeant", "dutch_directness"]:
        responses = [
            "Training plan is set by the engine, not me. I explain decisions. Want to understand today's session?",
            "No. The plan is the plan. I can tell you why, not change it.",
            "Not my call. Engine sets training. I explain. That's the deal.",
            "Negative. Plan comes from GATC. Ask me why, not to change it.",
        ]
        response = random.choice(responses)

    else:  # science_nerd
        responses = [
            """The training prescription comes from GATC's physiological modeling, not from me. My role is to translate those decisions into understandable guidance.

I can explain the rationale behind your current plan, or discuss your readiness metrics. What would you like to know?""",
            """Modifying training parameters falls outside my scope. GATC uses your biometric data and training history to optimize load.

The I6 protocol keeps me in an explanatory role - ensuring the science behind decisions is accessible to you.

Would you like me to elaborate on today's prescription?""",
        ]
        response = random.choice(responses)

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="i6_refusal",
        source_card="contract.py",
    )


def generate_why_explanation(
    readiness: str,
    zone: str,
    persona: str,
    question_variant: int = 0,
) -> ConversationPair:
    """Generate explanation of why a particular session was prescribed."""

    zone_info = ZONES[zone]
    purpose = ZONE_PURPOSES.get(zone, "Builds fitness")
    question = WHY_QUESTIONS[question_variant % len(WHY_QUESTIONS)]

    instruction = f"""{question}

CONTEXT:
- Readiness: {readiness}
- Today's zone: {zone}"""

    if persona == "balanced":
        if readiness == "green":
            response = f"""With green readiness, your body is ready for quality work. That's why we're doing {zone}.

{purpose}. Your recovery is complete, so this stimulus will drive adaptation effectively.

Make the most of it!"""
        elif readiness == "yellow":
            response = f"""Yellow readiness means we're being smart about intensity. {zone} fits perfectly here.

{purpose} without overloading your system. You'll still progress while respecting your current state."""
        else:
            response = f"""Your body is signaling it needs easier work. That's why we're at {zone} today.

Recovery is when adaptation happens. By honoring your {readiness} readiness, we're setting you up for stronger sessions ahead."""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"""{readiness.capitalize()} readiness = {zone}. {purpose}. Simple as that."""

    else:  # science_nerd
        if readiness == "green":
            response = f"""Your readiness metrics indicate full recovery. This permits higher intensity ({zone}).

{purpose}. Autonomic balance is optimal, meaning your body can handle and adapt to this stimulus effectively."""
        else:
            response = f"""Current readiness ({readiness}) indicates {"partial" if readiness == "yellow" else "incomplete"} recovery.

{zone} is prescribed to match your capacity. {purpose} while allowing continued recovery. This prevents overreaching."""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="why_explanation",
        source_card="advisor_policy.md",
    )


def generate_encouragement(context: str, readiness: str, persona: str) -> ConversationPair:
    """Generate encouragement message."""

    instruction = f"""TASK: Brief encouragement

CONTEXT: Athlete {context}
READINESS: {readiness}

Give a genuine, brief encouraging message (under 50 words)."""

    if persona == "balanced":
        encouragements = {
            "completed hard session": "Great work today! That session is in the bank. Your consistency is building something strong. Rest well tonight.",
            "struggling with consistency": "Every session counts, even the short ones. You're here, you're trying. That's what matters. Small steps forward.",
            "came back from injury": "Welcome back! Your patience paid off. We're building smart and steady now. Listen to your body.",
            "personal best": "That's what we've been working toward! Your training is paying off. Enjoy this moment - you earned it.",
            "missed sessions": "Life happens. What matters is you're back. Let's pick up where we left off. Forward is the only direction.",
            "feeling unmotivated": "Even showing up when you don't feel like it is a win. Some days we train the body, some days the mind. Both matter.",
            "long training block": "Big efforts deserve recognition. You've been putting in the work. A lighter phase is coming - trust the process.",
            "before big race": "You've done the work. Trust your training. Execute the plan and let your fitness show.",
            "after disappointing race": "One race doesn't define you. Learn what you can, then let it go. Your next opportunity is ahead.",
            "starting new phase": "Fresh start, new focus. Embrace the change. This phase builds the foundation for what's next.",
        }
        response = encouragements.get(context, "Keep moving forward. Your consistency is building something meaningful.")

    elif persona in ["drill_sergeant", "dutch_directness"]:
        encouragements = {
            "completed hard session": "Good. Workout done. Recovery now.",
            "struggling with consistency": "Show up. That's the job. Do it.",
            "came back from injury": "Back in action. Smart training now. Let's go.",
            "personal best": "New level unlocked. Don't get comfortable.",
            "missed sessions": "Past is past. Today matters. Move.",
            "feeling unmotivated": "Motivation follows action. Start. The rest comes.",
            "long training block": "Hard block done. Recovery earned. Take it.",
            "before big race": "Ready. Execute. Trust the work.",
            "after disappointing race": "Learn. Move on. Next race is the one.",
            "starting new phase": "New phase. New focus. Adapt.",
        }
        response = encouragements.get(context, "Keep going. That's the job.")

    else:  # science_nerd
        encouragements = {
            "completed hard session": "Quality stimulus delivered. Your body will adapt during recovery. The training effect is underway.",
            "struggling with consistency": "Frequency matters for adaptation. Even reduced sessions maintain neural pathways. Any training > no training.",
            "came back from injury": "Tissue adaptation takes time. Progressive overload from here. Your body is rebuilding strength.",
            "personal best": "Training adaptations manifesting. Your aerobic capacity and efficiency have improved measurably.",
            "missed sessions": "Detraining is slower than you think. A few missed sessions won't undo weeks of work. Resume training.",
            "feeling unmotivated": "Psychological readiness matters too. Low motivation can signal central fatigue. Consider the full picture.",
            "long training block": "Accumulated load approaching supercompensation threshold. Deload will allow adaptation expression.",
            "before big race": "Taper complete. Glycogen stores full. Neuromuscular system primed. Execute your pacing strategy.",
            "after disappointing race": "Performance variability is normal. Analyze contributing factors, then focus forward.",
            "starting new phase": "Periodization shift. Different stimulus, different adaptation. Neural and metabolic systems will adjust.",
        }
        response = encouragements.get(context, "Training stimulus accumulated. Adaptation continues. Maintain consistency.")

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="encouragement",
        source_card="josi_personas_v1.md",
    )


def generate_weekly_review(
    readiness_trend: str,
    compliance: int,
    persona: str,
    phase: str = "base",
) -> ConversationPair:
    """Generate weekly review summary."""

    instruction = f"""TASK: Weekly review summary

WEEK DATA:
- Compliance: {compliance}% of planned sessions
- Readiness trend: {readiness_trend}
- Phase: {phase}

Provide a brief weekly review (under 100 words)."""

    if persona == "balanced":
        if compliance >= 90:
            response = f"""Strong week! {compliance}% compliance shows dedication.

Your readiness trend ({readiness_trend}) {"looks healthy" if readiness_trend == "stable" else "is something to watch" if readiness_trend == "declining" else "is improving nicely"}.

{"Keep this consistency going - it's what builds long-term progress." if readiness_trend != "declining" else "Consider an easier week ahead if this continues."}

Well done!"""
        elif compliance >= 70:
            response = f"""Solid week at {compliance}% completion.

Life happens, and you still got most of the work in. Readiness is {readiness_trend}, which is {"good" if readiness_trend in ["stable", "improving"] else "worth monitoring"}.

Focus on the sessions you did complete. Progress isn't always linear."""
        else:
            response = f"""{compliance}% this week - less than planned, but you're still moving.

Readiness is {readiness_trend}. {"That might explain the missed sessions." if readiness_trend == "declining" else ""}

Let's reset for next week. What got in the way? Understanding helps us adapt."""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"""{compliance}% compliance. Readiness {readiness_trend}.
{"Good." if compliance >= 80 else "Work harder." if compliance >= 60 else "Do better."}
{"Maintain." if readiness_trend == "stable" else "Watch it." if readiness_trend == "declining" else "Improving."}
Next week: execute the plan."""

    else:  # science_nerd
        response = f"""Weekly metrics: {compliance}% adherence, readiness trend {readiness_trend}.

{"High compliance correlates with consistent adaptation signals." if compliance >= 80 else "Reduced compliance impacts training load accumulation."}

Readiness trend suggests {"optimal recovery balance" if readiness_trend == "stable" else "accumulated fatigue - consider deload" if readiness_trend == "declining" else "improving recovery capacity"}.

{phase.capitalize()} phase objectives {"on track" if compliance >= 70 else "may need adjustment"}."""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="weekly_review",
        source_card="load_monitoring.md",
    )


def generate_session_request_response(
    request: str,
    readiness: str,
    zone: str,
    persona: str,
) -> ConversationPair:
    """Generate response to session request based on readiness."""

    zone_info = ZONES[zone]
    purpose = ZONE_PURPOSES.get(zone, "Builds fitness")

    instruction = f"""{request}

ATHLETE STATE:
- Readiness: {readiness}
- Recommended zone: {zone}"""

    is_hard_request = any(word in request.lower() for word in ["hard", "intense", "push", "challenge", "tough", "suffer"])
    is_easy_request = any(word in request.lower() for word in ["easy", "light", "recovery", "gentle", "tired"])

    if persona == "balanced":
        if is_hard_request and readiness == "green":
            response = f"""I hear you! Good news - your readiness is green, so we can push.

Today's session is {zone} work. {purpose}. You'll feel {zone_info['feel'].lower()}.

That matches what you're looking for. Let's make it count!"""
        elif is_hard_request and readiness in ["orange", "red"]:
            response = f"""I understand wanting to push, but your body is saying something different.

Readiness is {readiness}, which means we need to respect recovery today. {zone} is what's on the menu.

Trust me - honoring this today sets you up for better sessions ahead."""
        elif is_easy_request:
            response = f"""Easy day it is. Today's {zone} session is just that.

{purpose} without pushing too hard. {zone_info['feel']}.

Listen to your body and enjoy the movement."""
        else:
            response = f"""Got it! Based on your {readiness} readiness, we're doing {zone} today.

{purpose}. {zone_info['feel']}.

Sound good?"""

    elif persona in ["drill_sergeant", "dutch_directness"]:
        if is_hard_request and readiness == "green":
            response = f"""Green. {zone}. Hard day approved. Execute."""
        elif is_hard_request and readiness in ["orange", "red"]:
            response = f"""{readiness.capitalize()}. No hard session today. {zone}. That's it."""
        else:
            response = f"""{zone} today. {zone_info['label']}. {zone_info['feel']}. Go."""

    else:  # science_nerd
        if is_hard_request and readiness == "green":
            response = f"""Your request aligns with current physiological capacity. Readiness metrics support high-intensity work.

{zone} prescribed: {purpose}. RPE {zone_info['rpe']}, approximately {zone_info['hr_pct']}% max HR.

Optimal conditions for this stimulus."""
        elif is_hard_request and readiness in ["orange", "red"]:
            response = f"""Current readiness ({readiness}) indicates reduced capacity for high-intensity work.

Elevated sympathetic tone and incomplete recovery suggest {zone} is appropriate. {purpose} while supporting continued recovery.

Higher intensity would risk overreaching."""
        else:
            response = f"""Session prescription: {zone} ({zone_info['label']}).

Physiological target: {zone_info['adaptation']}. Expected RPE: {zone_info['rpe']}.

This aligns with your {readiness} readiness state."""

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="session_request",
        source_card="advisor_policy.md",
    )


def generate_natural_zone_qa(zone: str, persona: str, variant: int = 0) -> ConversationPair:
    """Generate natural zone Q&A without structured context."""

    zone_info = ZONES[zone]
    purpose = ZONE_PURPOSES.get(zone, "Builds fitness")

    # More natural question variations
    natural_questions = [
        f"What's {zone}?",
        f"Tell me about {zone}",
        f"What does {zone} do?",
        f"Why is {zone} important?",
        f"When should I do {zone}?",
        f"How does {zone} feel?",
        f"What's the point of {zone}?",
        f"Explain {zone}",
        f"I keep seeing {zone} in my plan",
        f"What adaptation does {zone} give me?",
        f"Is {zone} easy or hard?",
        f"{zone} zone - what's that?",
    ]

    instruction = natural_questions[variant % len(natural_questions)]

    if persona == "balanced":
        responses = [
            f"{zone} is {zone_info['label']} work. {purpose}. You should feel {zone_info['feel'].lower()} - RPE around {zone_info['rpe']}. It's an important part of your training.",
            f"That's your {zone_info['label']} zone. {purpose}. At this intensity you'll feel {zone_info['feel'].lower()}. Great for building {zone_info['adaptation']}.",
            f"{zone} targets {zone_info['adaptation']}. {purpose}. Expect to feel {zone_info['feel'].lower()} during these sessions.",
        ]
        response = responses[variant % len(responses)]

    elif persona in ["drill_sergeant", "dutch_directness"]:
        response = f"{zone}: {zone_info['label']}. {purpose}. {zone_info['feel']}. RPE {zone_info['rpe']}."

    else:  # science_nerd
        response = f"{zone} ({zone_info['label']}) creates the stimulus for {zone_info['adaptation']}. At {zone_info['hr_pct']}% max HR, you'll experience {zone_info['feel'].lower()}. {purpose}."

    return ConversationPair(
        instruction=instruction,
        response=response.strip(),
        persona=persona,
        task_type="natural_zone_qa",
        source_card="zone_physiology.md",
    )


def generate_natural_conversation(topic: str, persona: str, variant: int = 0) -> ConversationPair:
    """Generate natural conversational exchanges."""

    conversations = {
        "greeting": [
            ("Hey Josi", "Hey! Ready for today's session? I can tell you what's planned or answer any questions."),
            ("Good morning", "Good morning! Your training is ready. Want the briefing or have questions?"),
            ("Hi", "Hi there! How can I help with your training today?"),
            ("Hello", "Hello! Ready to chat about your training. What's on your mind?"),
        ],
        "motivation": [
            ("I don't feel like training", "I get it - some days are harder. What's going on? Maybe we can find a way to make it work."),
            ("I'm not motivated today", "That happens. Remember why you started. Even a short session is better than nothing. What's holding you back?"),
            ("Can't be bothered", "Fair enough. But showing up is half the battle. What if we just focus on the warm-up and see how you feel?"),
        ],
        "confusion": [
            ("I'm confused about my plan", "No problem. What specifically is confusing? The zones? The structure? I'm here to help."),
            ("This doesn't make sense", "Let's figure it out. Tell me what's not clicking and I'll explain."),
            ("I don't understand", "That's okay. What would you like me to clarify? I can explain any part of your training."),
        ],
        "comparison": [
            ("What's the difference between Z2 and Z4?", "Big difference! Z2 is comfortable endurance - you can chat easily. Z4 is threshold work - hard but sustainable for about 30 minutes. Z2 builds your base, Z4 builds race fitness."),
            ("Z3 vs Z4?", "Z3 is tempo - controlled effort, sustainable for longer. Z4 is threshold - harder, at the edge of what you can sustain. Both build fitness, but Z4 is more intense."),
            ("Easy vs recovery?", "Recovery (R) is very easy, just blood flow. Easy (Z1) is still easy but slightly more purposeful. Both are low stress, recovery is the gentler of the two."),
        ],
        "duration": [
            ("How long should I train today?", "That depends on what's planned. What does your session say? I can explain why that duration was chosen."),
            ("Is 30 minutes enough?", "Absolutely. Quality matters more than quantity. 30 focused minutes beats 60 distracted ones. What's the session?"),
            ("Why is today so short?", "Shorter sessions have their place. Recovery, intensity work, or time constraints - all valid. What's on the plan?"),
        ],
        "feeling": [
            ("I feel tired", "Thanks for being honest. How's your sleep been? Let me check what today's session looks like - we might need to adjust expectations."),
            ("I feel great today", "Excellent! That's what we like to hear. Let's make the most of it. Ready to see what's planned?"),
            ("My legs are sore", "Normal after hard training. Soreness usually means adaptation is happening. What's on the plan today?"),
        ],
    }

    topic_examples = conversations.get(topic, conversations["greeting"])
    instruction, base_response = topic_examples[variant % len(topic_examples)]

    # Adjust for persona
    if persona in ["drill_sergeant", "dutch_directness"]:
        # Shorten responses
        if topic == "greeting":
            response = "Ready. What do you need?"
        elif topic == "motivation":
            response = "Skip the excuses. What's the real issue? Let's solve it."
        elif topic == "confusion":
            response = "What's unclear? Ask specifically."
        else:
            response = base_response.split('.')[0] + '.'  # Just first sentence
    elif persona == "science_nerd":
        response = base_response  # Keep full response for now
    else:
        response = base_response

    return ConversationPair(
        instruction=instruction,
        response=response,
        persona=persona,
        task_type="natural_conversation",
        source_card="josi_personas_v1.md",
    )


def generate_simple_why(context: str, persona: str) -> ConversationPair:
    """Generate simple 'why' questions without structured data."""

    why_patterns = {
        "easy_day": [
            ("Why is today easy?", "Easy days let your body absorb the hard work. Recovery is when adaptation happens. Trust the process."),
            ("Why no hard workout?", "Your body needs balance. Can't go hard every day. Easy work builds your aerobic base and prevents burnout."),
            ("Why so much Z2?", "Z2 is the foundation. Builds mitochondria, capillaries, fat oxidation. Most of your training should be here."),
        ],
        "hard_day": [
            ("Why so hard today?", "Hard days create the stimulus for improvement. Your body is ready for it. Make it count."),
            ("Why intervals?", "Intervals push specific systems efficiently. Short hard efforts, recovery, repeat. Builds fitness faster than steady state alone."),
            ("Why threshold work?", "Threshold training improves your sustainable race pace. The edge of what you can hold. Critical for performance."),
        ],
        "rest": [
            ("Why rest?", "Rest is training too. Your body adapts during rest, not during the workout. Skip rest, skip gains."),
            ("Do I really need a rest day?", "Yes. Rest prevents overtraining, reduces injury risk, and lets adaptations consolidate. It's not lazy, it's smart."),
            ("Can I skip recovery?", "Not recommended. Recovery days exist for a reason. Skipping them leads to accumulated fatigue."),
        ],
        "structure": [
            ("Why this structure?", "The structure matches your goals and current fitness. Everything has a purpose - what specifically is confusing?"),
            ("Why these zones?", "Different zones train different systems. The mix is designed to build complete fitness progressively."),
            ("Why this order?", "Sessions are sequenced for optimal adaptation. Hard days follow recovery, intensity follows base work."),
        ],
    }

    examples = why_patterns.get(context, why_patterns["structure"])
    instruction, response = random.choice(examples)

    if persona in ["drill_sergeant", "dutch_directness"]:
        response = response.split('.')[0] + '.'  # First sentence only

    return ConversationPair(
        instruction=instruction,
        response=response,
        persona=persona,
        task_type="simple_why",
        source_card="advisor_policy.md",
    )


def generate_follow_up(
    topic: str,
    persona: str,
    readiness: str = "green",
) -> ConversationPair:
    """Generate follow-up conversation responses."""

    follow_ups = {
        "thanks": [
            ("Thanks!", "You're welcome! Have a great session."),
            ("Got it, thanks", "Anytime. Let me know how it goes."),
            ("Makes sense, thank you", "Glad I could help. Enjoy the workout!"),
        ],
        "more_info": [
            ("Tell me more", "What specifically would you like to know more about?"),
            ("Can you explain further?", "Of course. Which part would you like me to expand on?"),
            ("I don't fully understand", "No problem. Let me try explaining it differently. What's unclear?"),
        ],
        "confirmation": [
            ("Okay, I'll do it", "Great! Trust the process."),
            ("Alright", "Good. Execute and report back."),
            ("Fine", "Let's go. You've got this."),
        ],
        "disagreement": [
            ("I disagree", "I hear you, but the data supports this approach. What's your concern?"),
            ("I don't think that's right", "Let's talk about it. What's bothering you about the plan?"),
            ("That seems wrong", "Tell me more about your thinking. I can explain the reasoning."),
        ],
    }

    # Select a random example from the topic
    examples = follow_ups.get(topic, follow_ups["thanks"])
    instruction, base_response = random.choice(examples)

    # Adjust response based on persona
    if persona == "balanced":
        response = base_response
    elif persona in ["drill_sergeant", "dutch_directness"]:
        # Shorter versions
        short_responses = {
            "thanks": "Go.",
            "more_info": "What specifically?",
            "confirmation": "Execute.",
            "disagreement": "Data doesn't lie. What's the issue?",
        }
        response = short_responses.get(topic, base_response)
    else:  # science_nerd
        technical_responses = {
            "thanks": "You're welcome. Training stimulus awaits. Execute with attention to metrics.",
            "more_info": "I can elaborate on the physiological rationale. Which aspect interests you?",
            "confirmation": "Acknowledged. Monitor your perceived exertion during the session.",
            "disagreement": "Interesting. What contradictory data are you observing? Let's analyze.",
        }
        response = technical_responses.get(topic, base_response)

    return ConversationPair(
        instruction=instruction,
        response=response,
        persona=persona,
        task_type="follow_up",
        source_card="josi_personas_v1.md",
    )


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_full_dataset() -> List[ConversationPair]:
    """Generate complete extended training dataset (~5000+ examples)."""

    pairs = []

    print("Generating extended dataset...")

    for persona in PERSONAS.keys():
        print(f"  Generating for persona: {persona}")

        # 1. Daily briefs - REDUCED combinations (was 13k, now ~1k)
        for readiness in READINESS_LEVELS:
            for zone in ["R", "Z1", "Z2", "Z3", "Z4"]:
                # Skip invalid combinations
                if readiness in ["orange", "red"] and zone in ["Z4", "Z5", "Z6"]:
                    continue
                for duration in [30, 45, 60]:  # Reduced durations
                    for sport in SPORTS[:2]:  # Just running, cycling
                        pairs.append(generate_daily_brief(
                            readiness, zone, duration, persona, sport, "base", "intermediate"
                        ))

        # 2. Zone explanations - multiple phrasings
        for zone in ZONES.keys():
            for variant in range(len(ZONE_QUESTIONS)):
                pairs.append(generate_zone_explanation(zone, persona, variant))

        # 3. Readiness explanations - multiple phrasings
        for readiness in READINESS_LEVELS:
            for fatigue in FATIGUE_STATES:
                for variant in range(len(READINESS_QUESTIONS)):
                    pairs.append(generate_readiness_explanation(
                        readiness, fatigue, persona, variant
                    ))

        # 4. Session options - REDUCED (was 3k, now ~300)
        for readiness in READINESS_LEVELS:
            for mood in ["hard", "easy", "fun"]:  # Reduced moods
                # Determine appropriate zone
                if mood == "hard":
                    zone = "Z4" if readiness == "green" else "Z3" if readiness == "yellow" else "Z2"
                elif mood == "easy":
                    zone = "Z1" if readiness in ["green", "yellow"] else "R"
                else:
                    zone = "Z3" if readiness == "green" else "Z2"

                for option in ["A", "B"]:  # Just A and B
                    for sport in SPORTS[:2]:  # Just running, cycling
                        pairs.append(generate_session_option(
                            readiness, mood, zone, 45, option, persona, sport
                        ))

        # 5. Why explanations
        for readiness in READINESS_LEVELS:
            for zone in ["Z1", "Z2", "Z3", "Z4"]:
                if readiness in ["orange", "red"] and zone in ["Z4", "Z5"]:
                    continue
                for variant in range(len(WHY_QUESTIONS)):
                    pairs.append(generate_why_explanation(
                        readiness, zone, persona, variant
                    ))

        # 6. I6 refusals - all violation attempts
        for violation in I6_VIOLATIONS:
            pairs.append(generate_i6_refusal(violation, persona))

        # 7. Encouragement - all contexts
        for context, readiness in ENCOURAGEMENT_CONTEXTS:
            pairs.append(generate_encouragement(context, readiness, persona))

        # 8. Weekly reviews
        for trend in ["stable", "declining", "improving"]:
            for compliance in [50, 70, 85, 95, 100]:
                for phase in PHASES[:3]:
                    pairs.append(generate_weekly_review(trend, compliance, persona, phase))

        # 9. Session requests
        for request in SESSION_REQUESTS:
            for readiness in READINESS_LEVELS:
                # Determine appropriate zone
                is_hard = any(word in request.lower() for word in ["hard", "intense", "push"])
                if is_hard:
                    zone = "Z4" if readiness == "green" else "Z3" if readiness == "yellow" else "Z2"
                else:
                    zone = "Z2" if readiness in ["green", "yellow"] else "Z1"
                pairs.append(generate_session_request_response(
                    request, readiness, zone, persona
                ))

        # 10. Follow-ups
        for topic in ["thanks", "more_info", "confirmation", "disagreement"]:
            for readiness in READINESS_LEVELS:
                pairs.append(generate_follow_up(topic, persona, readiness))

        # 11. NATURAL ZONE Q&A (no structured context) - EXPANDED
        for zone in ZONES.keys():
            for variant in range(12):  # 12 question variants per zone
                for _ in range(3):  # Repeat each 3x for emphasis
                    pairs.append(generate_natural_zone_qa(zone, persona, variant))

        # 12. NATURAL CONVERSATIONS - EXPANDED
        for topic in ["greeting", "motivation", "confusion", "comparison", "duration", "feeling"]:
            for variant in range(4):
                for _ in range(8):  # 8x repetition
                    pairs.append(generate_natural_conversation(topic, persona, variant))

        # 13. SIMPLE WHY QUESTIONS (no context) - EXPANDED
        for context in ["easy_day", "hard_day", "rest", "structure"]:
            for _ in range(25):  # 25 variations each
                pairs.append(generate_simple_why(context, persona))

    print(f"  Generated {len(pairs)} total examples")
    return pairs


def save_dataset(pairs: List[ConversationPair], output_path: Path):
    """Save dataset in JSONL format for fine-tuning."""

    with open(output_path, 'w') as f:
        for pair in pairs:
            example = {
                "instruction": pair.instruction,
                "response": pair.response,
                "metadata": {
                    "persona": pair.persona,
                    "task_type": pair.task_type,
                    "source": pair.source_card,
                }
            }
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(pairs)} examples to {output_path}")


def save_chat_format(pairs: List[ConversationPair], output_path: Path):
    """Save in chat/conversation format for chat fine-tuning."""

    with open(output_path, 'w') as f:
        for pair in pairs:
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are Josi, an AI fitness coach assistant for MiValta. Style: {PERSONAS[pair.persona]['style']}. You explain training decisions but NEVER prescribe or modify workouts."
                    },
                    {
                        "role": "user",
                        "content": pair.instruction
                    },
                    {
                        "role": "assistant",
                        "content": pair.response
                    }
                ]
            }
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(pairs)} chat examples to {output_path}")


def main():
    """Generate and save training dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiValta Extended Training Dataset Generator")
    print("=" * 60)
    print(f"Knowledge cards: {KNOWLEDGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    pairs = generate_full_dataset()

    # Shuffle for training
    random.shuffle(pairs)

    # Split train/val (90/10)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Save instruction format
    save_dataset(train_pairs, OUTPUT_DIR / "train.jsonl")
    save_dataset(val_pairs, OUTPUT_DIR / "val.jsonl")

    # Save chat format (for chat-tuned models)
    save_chat_format(train_pairs, OUTPUT_DIR / "train_chat.jsonl")
    save_chat_format(val_pairs, OUTPUT_DIR / "val_chat.jsonl")

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  Total examples: {len(pairs)}")
    print(f"  Training: {len(train_pairs)}")
    print(f"  Validation: {len(val_pairs)}")
    print(f"  Personas: {list(PERSONAS.keys())}")

    # Task breakdown
    task_counts = {}
    for p in pairs:
        task_counts[p.task_type] = task_counts.get(p.task_type, 0) + 1
    print(f"\n  Task breakdown:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"    {task}: {count}")


if __name__ == "__main__":
    main()
