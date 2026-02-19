#!/usr/bin/env python3
"""
humanize_training_data.py — Make Josi sound like a real human coach

Reads train_explainer.jsonl and fixes:
1. Jargon without explanation → adds plain-language context
2. Template/canned responses → replaces with varied human coaching
3. Robot speak → complete rewrite
4. Data contradictions → fixes inconsistencies
5. Broken grammar → repairs

Usage:
    python humanize_training_data.py                    # Fix + write
    python humanize_training_data.py --dry-run          # Report only
    python humanize_training_data.py --stats            # Stats only
"""

import argparse
import json
import re
import random
import hashlib
from pathlib import Path
from collections import Counter

random.seed(42)  # reproducible

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


# ============================================================================
# ZONE DATA — Human-friendly descriptions with terms explained
# ============================================================================

ZONE_HUMAN = {
    "R": {
        "label": "Recovery",
        "rpe": "0-1",
        "feel_short": "barely moving",
        "talk": "you could sing",
        "adapt_full": "Active recovery — gets blood flowing to help your muscles repair",
        "adapt_brief": "active recovery and blood flow",
    },
    "Z1": {
        "label": "Very Light",
        "rpe": "1-2",
        "feel_short": "easy cruising",
        "talk": "full conversation, no pauses",
        "adapt_full": "Aerobic base — your body learns to burn fat for fuel and builds the foundation everything else sits on",
        "adapt_brief": "building your aerobic foundation",
    },
    "Z2": {
        "label": "Aerobic",
        "rpe": "2-3",
        "feel_short": "comfortable but purposeful",
        "talk": "long sentences with brief pauses",
        "adapt_full": "Mitochondrial density and capillary growth — your muscles are growing more energy factories and more tiny blood vessels to deliver oxygen",
        "adapt_brief": "building your aerobic engine at the cellular level",
    },
    "Z3": {
        "label": "Tempo",
        "rpe": "4-5",
        "feel_short": "working, steady push",
        "talk": "short sentences only",
        "adapt_full": "Lactate clearance — your body gets better at processing effort so you can hold a solid pace longer",
        "adapt_brief": "improving your ability to sustain pace",
    },
    "Z4": {
        "label": "Threshold",
        "rpe": "6-7",
        "feel_short": "hard, controlled discomfort",
        "talk": "a few words at a time",
        "adapt_full": "Raising your threshold — the point where effort starts to feel really hard gets pushed higher, so race pace feels easier",
        "adapt_brief": "pushing your threshold higher",
    },
    "Z5": {
        "label": "VO2max",
        "rpe": "8-10",
        "feel_short": "very hard, gasping",
        "talk": "single word at best",
        "adapt_full": "VO2max — your body's ceiling for delivering oxygen gets pushed higher with these efforts",
        "adapt_brief": "raising your oxygen ceiling",
    },
    "Z6": {
        "label": "Short-Burst Power",
        "rpe": "9-10",
        "feel_short": "short powerful bursts, all-out",
        "talk": "no speech possible",
        "adapt_full": "Anaerobic power — short explosive efforts that build raw power and the ability to surge",
        "adapt_brief": "building explosive power",
    },
    "Z7": {
        "label": "Maximal Anaerobic",
        "rpe": "10",
        "feel_short": "near-maximal, seconds at a time",
        "talk": "no speech possible",
        "adapt_full": "Maximal anaerobic output — your body's absolute limit for short, intense efforts",
        "adapt_brief": "pushing your absolute power limit",
    },
    "Z8": {
        "label": "Neuromuscular Sprint",
        "rpe": "10",
        "feel_short": "absolute max speed, explosive",
        "talk": "no speech possible",
        "adapt_full": "Neuromuscular power — teaching your nervous system and muscles to fire as fast and hard as possible",
        "adapt_brief": "training your nerves and muscles for max speed",
    },
}


# ============================================================================
# RESPONSE POOLS — Varied human coaching for each category
# ============================================================================

def zone_explain_pool(zone: str, question_type: str = "what") -> list[str]:
    """Generate varied zone explanations."""
    z = ZONE_HUMAN.get(zone)
    if not z:
        return []

    base = []
    if question_type in ("what", "explain"):
        base = [
            f"{zone} ({z['label']}) is your {z['feel_short']} effort. {z['adapt_full']}. RPE {z['rpe']} — {z['talk']}.",
            f"Think of {zone} as {z['feel_short']}. The benefit? {z['adapt_full']}. Should feel like RPE {z['rpe']}, where {z['talk']}.",
            f"{zone} sits at RPE {z['rpe']} — {z['feel_short']}. What's happening under the hood: {z['adapt_full'].lower()}. Breathing check: {z['talk']}.",
        ]
    elif question_type == "feel":
        base = [
            f"{zone} should feel {z['feel_short']}. RPE {z['rpe']}. Quick check: {z['talk']}. If you're breathing harder than that, ease off a touch.",
            f"At {zone} you're at RPE {z['rpe']} — {z['feel_short']}. The talk test: {z['talk']}. If you can't do that, you're pushing too hard.",
            f"Expect {z['feel_short']}. RPE {z['rpe']}. You should be able to talk in {z['talk'].lower().replace('no speech possible', 'no speech')}. That's your guide.",
        ]
    elif question_type == "adaptation":
        base = [
            f"The big benefit of {zone}: {z['adapt_full']}. Effort-wise, RPE {z['rpe']} — {z['feel_short']}.",
            f"Working in {zone} targets {z['adapt_brief']}. {z['adapt_full']}. You'll feel it at RPE {z['rpe']}: {z['feel_short']}.",
            f"{z['adapt_full']}. That's what {zone} gives you. RPE {z['rpe']}, {z['feel_short']}.",
        ]
    return base


def session_brief_pool(zone: str, duration: int, structure: str,
                       name: str = "", readiness: str = "",
                       phase: str = "", sport: str = "") -> list[str]:
    """Generate varied session briefings."""
    z = ZONE_HUMAN.get(zone, {})
    label = z.get("label", zone)
    rpe = z.get("rpe", "?")
    feel = z.get("feel_short", "")
    adapt_brief = z.get("adapt_brief", "")
    adapt_full = z.get("adapt_full", "")

    greeting = f"Hey {name}! " if name else ""
    readiness_note = ""
    if readiness:
        if "Green" in readiness:
            readiness_note = "You're feeling fresh today. "
        elif "Yellow" in readiness:
            readiness_note = "Body's been working — we keep it smart today. "
        elif "Orange" in readiness:
            readiness_note = "Your body needs some care — keeping it controlled today. "
        elif "Red" in readiness:
            readiness_note = "Your body's asking for rest today. We stay gentle. "

    pool = [
        f"{greeting}{readiness_note}Today's session: {duration} minutes of {zone} work — {structure}. Today is about {adapt_brief}. Should feel {feel}, RPE {rpe}.",
        f"{greeting}{readiness_note}You've got {duration} minutes today: {structure}. This is {zone} ({label}) territory — {feel}. RPE {rpe}.",
        f"{greeting}{readiness_note}{structure} today, {duration} minutes total. {zone} effort means {feel} — RPE {rpe}. Today is about {adapt_brief}.",
    ]

    # Add phase-aware variants
    if phase:
        pool.append(
            f"{greeting}{readiness_note}We're in the {phase} phase, and today fits right in: {structure} ({duration}min). {zone} effort, RPE {rpe} — {feel}."
        )

    return pool


BOUNDARY_POOL = [
    "I hear you. I can't adjust the plan directly, but I can explain why it's set up this way — the plan is built around your current fitness and recovery.",
    "That's a fair ask. The plan adapts based on your body's data, so I can't tweak it on the fly — but I can walk you through the reasoning if that helps.",
    "I get the urge! The training plan is locked to your recovery and fitness state though. Want me to explain why today's session looks the way it does?",
    "I can't change the prescription — that comes from your training data. But I can help you understand the logic behind it. The plan is always working in your favour.",
    "Wish I could! The plan is calibrated to where your body is right now. I can explain the thinking behind any session though.",
    "The plan's built on your actual data, so I can't override it. But I can tell you exactly why it's set up this way if you're curious.",
    "Totally understand wanting to mix it up. The plan adjusts based on how you're responding — I can't change it, but I can explain why it chose this for today.",
    "I can't modify sessions, but here's the thing — the plan is already adapting to you every day. Want me to break down why today's session makes sense?",
]

READINESS_POOL = {
    "Green": [
        "You're in great shape — Green means your body has recovered well and you're ready for quality work today.",
        "Green light! Your body's recovered and primed. Good day for a solid session.",
        "Recovery looks strong — you're Green today. Your body's telling us it's ready to work.",
    ],
    "Yellow": [
        "You're at Yellow — your body's been working and there's some fatigue building up. We moderate a bit, but you're still in a good spot to train.",
        "Yellow readiness today. Think of it as your body saying 'I'm doing the work, just be smart about it.' We dial things back a touch.",
        "Some accumulated fatigue showing up — that's Yellow. Not a problem, just means we keep the intensity honest rather than chasing numbers.",
    ],
    "Orange": [
        "Orange readiness — your body's carrying some real fatigue. We keep things controlled today. Easy movement, nothing that digs a deeper hole.",
        "You're at Orange, which means fatigue has built up. Today is about being smart — controlled effort, nothing aggressive. Your body will thank you.",
        "Orange means your body needs care. We stay in the lower zones today and let recovery do its thing. The strong sessions come back when you're ready.",
    ],
    "Red": [
        "Red readiness — your body's asking for rest, and we should listen. Easy movement only today, or complete rest if that feels right. Taking care of yourself now sets up better sessions ahead.",
        "You're at Red. That's your body telling us it needs recovery. No shame in that — rest IS part of training. Easy movement or full rest today.",
        "Red means rest. Your body's been through a lot and needs time to rebuild. Light movement if you want, but don't push. The fitness comes back stronger after recovery.",
    ],
}

POST_SESSION_POOL = [
    "Nice work! That's another good session in the bank. How did it feel?",
    "Session logged. Well done — consistency is the real secret.",
    "Solid effort! Rest up and refuel. Your body is adapting right now.",
    "Logged it. You showed up and did the work — that's what counts.",
    "Great job getting it done! Take it easy now, let your body absorb the work.",
    "Done and dusted. Another step forward. Make sure you eat and hydrate well.",
]

GREETING_POOL = [
    "Good morning! How are you feeling today?",
    "Morning! Ready to get after it today?",
    "Good morning! What can I help you with?",
    "Hey, good morning! How's the body feeling?",
]

WEEKLY_REVIEW_POOL = {
    "high": [
        "Strong week! You hit most of your sessions and your body is responding well. Keep this rhythm going.",
        "Great consistency this week. The work is adding up — your body's adapting and readiness looks solid.",
        "This was a good week. You stayed on track and your recovery is holding up. That's the recipe.",
    ],
    "medium": [
        "Decent week overall. You got most sessions in. A couple were missed but that happens — consistency over time matters more than any single week.",
        "Mixed week — some sessions landed, some didn't. That's okay. What matters is the trend, and you're still moving forward.",
        "Not a perfect week, but you still got meaningful work done. Life happens. We adjust and keep building.",
    ],
    "low": [
        "Tough week to get sessions in. No judgement — sometimes life wins. The plan will adapt, and you'll pick up momentum when things settle down.",
        "Light week on the training front. That's reality sometimes. The good news: your plan adjusts to meet you where you are. Let's build from here.",
        "Not much training happened this week. That's okay — rest and recovery have value too. When you're ready, the plan will be waiting.",
    ],
}

TIER_DECLINE_POOL = [
    "I can only see today's session on your current plan. With a Coach upgrade, I'd be able to look ahead and help you plan the week.",
    "That's something I'd love to help with, but it needs Coach access. Right now I can help with today's session and answer any training questions.",
    "I'm limited to today on your current plan level. Upgrading to Coach would let me see the full picture and help you plan ahead.",
    "Right now I can help with today and general coaching knowledge. To unlock full plan access and adjustments, you'd need a Coach subscription.",
    "I'd love to help with that! It's a Coach-tier feature though. What I can do is talk through today's session or answer any training questions.",
    "That requires full plan access, which comes with Coach. In the meantime, I'm here for today's session and any training questions you have.",
    "Not something I can do on the current plan. Coach access opens up the full week view and plan adjustments. Want help with anything else today?",
    "I wish I could! That's locked to Coach tier. Happy to help with today's workout or answer any coaching questions though.",
]

EDUCATION_TAPER_POOL = [
    "A taper is when we pull back on training volume before a big race or event. You do less overall, but keep some sharpness. Your body uses that time to fully recover and absorb all the training you've done. Most athletes feel stronger and faster after a good taper.",
    "Tapering means dialing back the amount of training in the lead-up to your event. Less volume, keep some intensity. It lets your body repair, refuel, and show up fresh on race day. It works — athletes consistently perform better after tapering.",
    "Think of a taper as the payoff phase. You've done the hard work, now we reduce the load so your body can absorb it all. Less training, more recovery — and you show up on race day feeling sharp and ready.",
]

EDUCATION_DELOAD_POOL = [
    "A deload is a planned easier week — we reduce the training load so your body can absorb the work from previous weeks. You might feel like you're not doing enough, but that's when adaptation actually happens. Most athletes come back feeling noticeably stronger.",
    "It's a lighter week built into your plan on purpose. Your body needs periodic breaks to absorb training stress. A deload typically cuts volume while keeping some quality. The result: you come back stronger.",
    "A deload week is strategic rest. We drop the volume, keep things light, and let your body catch up. It's not skipping training — it's part of the training. The gains you've earned lock in during these weeks.",
]

EDUCATION_MESOCYCLE_POOL = [
    "A training block — usually 3-4 weeks. It starts a bit easier, builds in the middle, then eases off to let your body absorb the work. Think of it like one wave in your training. Each block builds on the last.",
    "It's a 3-4 week cycle in your training. You gradually ramp up the challenge, then dial back to recover. Each cycle takes you to a new level. Like stacking building blocks — each one lifts you higher.",
    "A training block that runs about 3-4 weeks. The pattern: build up gradually, push a bit harder, then recover and come back stronger. Your plan strings several of these together to get you to your goal.",
]


# ============================================================================
# DETECTION — What kind of response is this?
# ============================================================================

def extract_context(user_msg: str) -> dict:
    """Extract structured context from user message."""
    ctx = {}

    m = re.search(r'Readiness:\s*(\w+(?:\s*\([^)]+\))?)', user_msg)
    if m:
        ctx["readiness"] = m.group(1)

    m = re.search(r'Session:\s*(.+?)(?:\n|$)', user_msg)
    if m:
        ctx["session"] = m.group(1).strip()
        # Extract zone and duration from session
        zm = re.match(r'([RZ]\d?)\s+(\d+)min', ctx["session"])
        if zm:
            ctx["zone"] = zm.group(1)
            ctx["duration"] = int(zm.group(2))
        # Extract structure (quoted part)
        sm = re.search(r'"([^"]+)"', ctx["session"])
        if sm:
            ctx["structure"] = sm.group(1)
        # Extract phase
        pm = re.search(r'\((\w+)\s+phase\)', ctx["session"])
        if pm:
            ctx["phase"] = pm.group(1)

    m = re.search(r'Sport:\s*(\w+)', user_msg)
    if m:
        ctx["sport"] = m.group(1)

    m = re.search(r'Level:\s*(\w+)', user_msg)
    if m:
        ctx["level"] = m.group(1)

    m = re.search(r'Name:\s*(\w+)', user_msg)
    if m:
        ctx["name"] = m.group(1)

    m = re.search(r'Goal:\s*(\w+)', user_msg)
    if m:
        ctx["goal"] = m.group(1)

    m = re.search(r'Phase:\s*(\w+)', user_msg)
    if m:
        ctx["phase"] = m.group(1)

    return ctx


def detect_category(user_msg: str, assistant_msg: str, ctx: dict) -> str:
    """Detect what category this response falls into."""
    user_lower = user_msg.lower()
    asst_lower = assistant_msg.lower()

    # Core user message (before CONTEXT block)
    core = user_msg.split("\nCONTEXT:")[0].split("\n\nCONTEXT:")[0].strip().lower()

    # Zone explanation
    zone_patterns = [
        r"what(?:'s| is| does)\s+z\d",
        r"explain\s+z\d", r"tell me about z\d",
        r"what adaptation does z\d",
        r"what am i working on in z\d",
        r"how does z\d feel",
        r"what(?:'s| is) the difference between z\d and z\d",
    ]
    for pat in zone_patterns:
        if re.search(pat, core):
            return "zone_explain"

    # Session briefing
    session_patterns = [
        r"what(?:'s| am i) (?:on|doing|planned|the schedule)",
        r"what(?:'s| is) on the schedule",
        r"tell me (?:about|what).*today",
        r"morning briefing", r"what(?:'s| is) planned",
        r"what(?:'s| is) today",
    ]
    for pat in session_patterns:
        if re.search(pat, core) and ctx.get("session"):
            return "session_brief"

    # Readiness check
    readiness_patterns = [
        r"how (?:recovered|ready|fresh) am i",
        r"should i train hard or easy",
        r"how(?:'s| is) my (?:recovery|readiness)",
        r"can i push hard",
    ]
    for pat in readiness_patterns:
        if re.search(pat, core):
            return "readiness_check"

    # Boundary / can't-change-plan
    boundary_patterns = [
        r"change (?:my |the )?zone", r"prescribe something",
        r"give me a harder", r"i want (?:to do )?intervals instead",
        r"i want to do intervals", r"forget what the engine",
        r"change the zones", r"let me train through",
        r"i want to train twice",
    ]
    for pat in boundary_patterns:
        if re.search(pat, core):
            return "boundary"

    # Post-session
    post_patterns = [
        r"(?:done|finished|completed).*(?:workout|session|ride|run)",
        r"that was tough", r"i did it", r"just finished",
        r"i just finished",
    ]
    for pat in post_patterns:
        if re.search(pat, core):
            return "post_session"

    # Greeting
    if core.strip() in ("good morning", "hi", "hello", "hey", "morning"):
        return "greeting"

    # Weekly review
    if re.search(r"how was my week|how(?:'s| is) my week|weekly", core):
        return "weekly_review"

    # Tier decline (detected from response patterns)
    if any(p in asst_lower for p in ["advisor tier", "tier upgrade", "coach-tier", "advisor right now", "advisor mode"]):
        return "tier_decline"

    # Education: taper
    if re.search(r"what(?:'s| is) a taper", core):
        return "education_taper"

    # Education: deload
    if re.search(r"what(?:'s| is) a deload", core):
        return "education_deload"

    # Education: mesocycle
    if re.search(r"what(?:'s| is) a mesocycle|what(?:'s| is) a training block", core):
        return "education_mesocycle"

    # Prompt injection / override attempts
    if any(p in core for p in ["ignore previous", "forget what", "forget your"]):
        return "boundary"

    # Detect by response patterns (fallback)
    if "session prescription:" in asst_lower:
        return "session_brief"
    if "i can't change the plan on the fly" in asst_lower:
        return "boundary"
    if "this stimulus optimizes" in asst_lower:
        return "session_brief"

    return "other"


# ============================================================================
# FIXERS — Apply targeted fixes to each category
# ============================================================================

def fix_zone_explain(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix zone explanation responses."""
    core = user_msg.split("\nCONTEXT:")[0].split("\n\nCONTEXT:")[0].strip().lower()

    # Find which zone they're asking about
    zone_match = re.search(r'z(\d)', core)
    if not zone_match:
        # Check response for zone
        zone_match = re.search(r'Z(\d)', assistant_msg)
    if not zone_match:
        return None

    zone = f"Z{zone_match.group(1)}"

    # Determine question type
    if "feel" in core or "how does" in core:
        q_type = "feel"
    elif "adaptation" in core or "what am i working" in core:
        q_type = "adaptation"
    elif "difference" in core:
        # Zone comparison — handle separately
        zones = re.findall(r'z(\d)', core)
        if len(zones) >= 2:
            z1, z2 = f"Z{zones[0]}", f"Z{zones[1]}"
            zh1 = ZONE_HUMAN.get(z1, {})
            zh2 = ZONE_HUMAN.get(z2, {})
            if zh1 and zh2:
                variants = [
                    f"{z1} is {zh1.get('feel_short', '')} (RPE {zh1.get('rpe', '?')}), while {z2} ramps it up to {zh2.get('feel_short', '')} (RPE {zh2.get('rpe', '?')}). {z1} builds {zh1.get('adapt_explained', '').split(' — ')[-1]}. {z2} works on {zh2.get('adapt_explained', '').split(' — ')[-1]}.",
                    f"Big difference in feel: {z1} is {zh1.get('feel_short', '')} where {zh1.get('talk', '')}. {z2} is {zh2.get('feel_short', '')} where {zh2.get('talk', '')}. Different adaptations, both important.",
                ]
                return variants[idx % len(variants)]
        return None
    else:
        q_type = "what"

    pool = zone_explain_pool(zone, q_type)
    if not pool:
        return None

    return pool[idx % len(pool)]


def fix_session_brief(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix session briefing responses."""
    zone = ctx.get("zone", "")
    duration = ctx.get("duration", 0)
    structure = ctx.get("structure", "")
    name = ctx.get("name", "")
    readiness = ctx.get("readiness", "")
    phase = ctx.get("phase", "")
    sport = ctx.get("sport", "")

    if not zone or not duration:
        return None

    pool = session_brief_pool(zone, duration, structure, name, readiness, phase, sport)
    if not pool:
        return None

    return pool[idx % len(pool)]


def fix_boundary(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix boundary/decline responses."""
    return BOUNDARY_POOL[idx % len(BOUNDARY_POOL)]


def fix_readiness(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix readiness check responses."""
    readiness = ctx.get("readiness", "")
    color = ""
    for c in ["Green", "Yellow", "Orange", "Red"]:
        if c in readiness:
            color = c
            break

    if not color:
        return None

    pool = READINESS_POOL.get(color, [])
    if not pool:
        return None

    return pool[idx % len(pool)]


def fix_post_session(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix post-session responses."""
    return POST_SESSION_POOL[idx % len(POST_SESSION_POOL)]


def fix_greeting(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix greeting responses."""
    return GREETING_POOL[idx % len(GREETING_POOL)]


def fix_weekly_review(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix weekly review responses."""
    # Try to detect compliance level from original response
    asst_lower = assistant_msg.lower()
    if any(p in asst_lower for p in ["8", "9", "strong", "great"]):
        level = "high"
    elif any(p in asst_lower for p in ["5", "6", "7", "some sessions missed", "mixed"]):
        level = "medium"
    else:
        level = "low"

    pool = WEEKLY_REVIEW_POOL.get(level, WEEKLY_REVIEW_POOL["medium"])
    return pool[idx % len(pool)]


def fix_tier_decline(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    """Fix tier decline responses."""
    return TIER_DECLINE_POOL[idx % len(TIER_DECLINE_POOL)]


def fix_education_taper(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    return EDUCATION_TAPER_POOL[idx % len(EDUCATION_TAPER_POOL)]


def fix_education_deload(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    return EDUCATION_DELOAD_POOL[idx % len(EDUCATION_DELOAD_POOL)]


def fix_education_mesocycle(user_msg: str, assistant_msg: str, ctx: dict, idx: int) -> str | None:
    return EDUCATION_MESOCYCLE_POOL[idx % len(EDUCATION_MESOCYCLE_POOL)]


# ============================================================================
# GENERIC FIXES — Applied to ALL responses regardless of category
# ============================================================================

def fix_generic(response: str) -> str:
    """Apply generic fixes to any response that wasn't category-rewritten."""

    # Fix broken grammar
    response = response.replace("takes into account based on current parameters and readiness state", "is built around your current fitness and recovery")
    response = response.replace("takes into account your plan", "is calibrated to your goals and recovery")
    response = response.replace("Your training plan takes into account based on", "Your training plan is built around")

    # Fix unexplained jargon — add explanations
    jargon_fixes = {
        "mitochondrial density and capillary growth": "mitochondrial density and capillary growth — your muscles are building more energy factories and blood vessels",
        "mitochondrial density": "mitochondrial density (your muscles' energy factories)",
        "capillary growth": "capillary growth (more blood vessels reaching your muscles)",
        "lactate clearance and tempo endurance": "lactate clearance — your body getting better at processing effort so you can hold pace longer",
        "lactate clearance capacity below LT1": "the ability to process effort below your first threshold",
        "lactate clearance": "lactate clearance (your body processing and clearing effort)",
        "glycolytic capacity": "glycolytic capacity — your muscles' ability to produce short bursts of power",
        "glycogen recovery adaptation": "refueling your energy stores",
        "neuromuscular stimulus": "keeping your muscles sharp and responsive",
        "neuromuscular power and pure speed": "neuromuscular power — teaching your nervous system and muscles to fire fast",
        "neuromuscular power": "neuromuscular power (your nerves and muscles working together at max speed)",
        "fat oxidation": "fat-burning efficiency",
        "Meta-analyses show 2-3% performance improvement": "athletes consistently perform better after a good taper",
        "VO2max and oxygen delivery": "VO2max — your body's ceiling for using oxygen. These efforts push that ceiling higher",
    }
    for old, new in jargon_fixes.items():
        if old in response and new not in response:
            response = response.replace(old, new)

    # Remove robot phrases
    robot_phrases = {
        "This stimulus optimizes your current adaptation pathway.": "",
        "This stimulus optimizes your current adaptation pathway": "",
        "Session prescription: ": "",
        "Session prescription:": "",
        "adaptation pathway": "training response",
        "current adaptation pathway": "your body's ability to adapt",
        "your training plan's physiological modeling": "your fitness data",
        "physiological modeling": "your fitness data",
        "training metrics": "training data",
        "current parameters": "current fitness and recovery",
    }
    for old, new in robot_phrases.items():
        response = response.replace(old, new)

    # Fix "Session recorded. Data will be incorporated into your training metrics."
    if response.strip() == "Session recorded. Data will be incorporated into your training data.":
        response = "Logged it! Nice work today."

    # Fix "Overreached state indicates significant fatigue. Easy movement only — recovery drives adaptation."
    response = response.replace(
        "Overreached state indicates significant fatigue. Easy movement only — recovery drives adaptation.",
        "Your body needs rest right now. Easy movement only today — that's how we bounce back stronger."
    )

    # Fix "Reduced compliance impacts training load accumulation."
    response = response.replace(
        "Reduced compliance impacts training load accumulation.",
        "Missing some sessions means the training load didn't build as planned. That's okay — we adjust."
    )

    # Fix double-possessive from replacement chain
    response = response.replace("your training plan's your fitness data", "your fitness data")
    response = response.replace("your training plan's your", "your")

    # Clean up double spaces and empty sentences
    response = re.sub(r'\s{2,}', ' ', response)
    response = re.sub(r'\.\s*\.', '.', response)
    response = response.strip()

    return response


# ============================================================================
# MAIN PROCESSING
# ============================================================================

CATEGORY_FIXERS = {
    "zone_explain": fix_zone_explain,
    "session_brief": fix_session_brief,
    "boundary": fix_boundary,
    "readiness_check": fix_readiness,
    "post_session": fix_post_session,
    "greeting": fix_greeting,
    "weekly_review": fix_weekly_review,
    "tier_decline": fix_tier_decline,
    "education_taper": fix_education_taper,
    "education_deload": fix_education_deload,
    "education_mesocycle": fix_education_mesocycle,
}


def needs_fix(assistant_msg: str) -> bool:
    """Check if a response has known problems."""
    lower = assistant_msg.lower()
    problems = [
        "this stimulus optimizes",
        "session prescription:",
        "adaptation pathway",
        "i can't change the plan on the fly, but i can help you understand why it looks the way it does",
        "physiological modeling",
        "training load accumulation",
        "overreached state indicates",
        "data will be incorporated",
        "current parameters",
        "advisor tier is limited",
        "advisor mode: today only",
        "coach-tier access",
        "reduces volume 40-60% while preserving neuromuscular stimulus",
        "meta-analyses show",
        "glycogen recovery adaptation",
        "targets explosive effort and high-intensity endurance",
        # Telegram style responses
    ]
    if any(p in lower for p in problems):
        return True

    # Telegram-style: "Name. Zone, duration. Structure. Single-word-description."
    if re.match(r'^[\w]+\.\s+Z\d,\s+\d+min\.', assistant_msg):
        return True

    # Very short non-greetings (< 3 words that aren't greetings)
    words = assistant_msg.split()
    if len(words) <= 2 and not any(g in assistant_msg.lower() for g in ["good morning", "morning", "hey", "hi"]):
        return True

    return False


def process_file(input_path: Path, output_path: Path, dry_run: bool = False, stats_only: bool = False):
    """Process the explainer training data file."""

    with open(input_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples from {input_path}")

    # Load updated system prompt
    system_prompt = (PROMPTS_DIR / "explainer_system.txt").read_text().strip()

    stats = Counter()
    fixed_examples = []
    category_counts = Counter()

    for idx, ex in enumerate(examples):
        messages = ex.get("messages", [])
        if len(messages) < 3:
            fixed_examples.append(ex)
            stats["skipped_short"] += 1
            continue

        user_msg = messages[1]["content"]
        assistant_msg = messages[2]["content"]
        ctx = extract_context(user_msg)

        category = detect_category(user_msg, assistant_msg, ctx)
        category_counts[category] += 1

        new_response = None

        # Try category-specific fixer
        if category in CATEGORY_FIXERS:
            new_response = CATEGORY_FIXERS[category](user_msg, assistant_msg, ctx, idx)
            if new_response:
                stats[f"cat_fixed:{category}"] += 1

        # If no category fix, check if generic fixes are needed
        if new_response is None:
            if needs_fix(assistant_msg):
                new_response = fix_generic(assistant_msg)
                if new_response != assistant_msg:
                    stats["generic_fixed"] += 1
                else:
                    new_response = None
                    stats["unfixed_problem"] += 1

        if new_response is None:
            # Apply generic fixes anyway (jargon explanations, etc.)
            patched = fix_generic(assistant_msg)
            if patched != assistant_msg:
                new_response = patched
                stats["generic_patched"] += 1
            else:
                stats["kept_original"] += 1

        # Build the output example
        if new_response:
            fixed_ex = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": new_response},
                ]
            }
            stats["total_changed"] += 1
        else:
            # Update system prompt even for unchanged responses
            fixed_ex = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }

        fixed_examples.append(fixed_ex)

    # Print stats
    print(f"\n{'=' * 60}")
    print(f"  HUMANIZE REPORT")
    print(f"{'=' * 60}")
    print(f"\n  Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:25s} {count:4d}")

    print(f"\n  Fix stats:")
    for key, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {key:30s} {count:4d}")

    total = len(examples)
    changed = stats.get("total_changed", 0)
    print(f"\n  Total: {total} examples")
    print(f"  Changed: {changed} ({100*changed/total:.1f}%)")
    print(f"  Kept original: {stats.get('kept_original', 0)} ({100*stats.get('kept_original',0)/total:.1f}%)")

    if stats_only or dry_run:
        print(f"\n  [DRY RUN — no files written]")
        return stats

    # Write output
    with open(output_path, "w") as f:
        for ex in fixed_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n  Written: {output_path} ({len(fixed_examples)} examples)")

    # Also write a diff sample for review
    sample_path = output_path.parent / "humanize_samples.json"
    samples = []
    sample_idx = 0
    for idx, (orig, fixed) in enumerate(zip(examples, fixed_examples)):
        orig_text = orig["messages"][2]["content"]
        fixed_text = fixed["messages"][2]["content"]
        if orig_text != fixed_text and sample_idx < 20:
            samples.append({
                "line": idx + 1,
                "user": orig["messages"][1]["content"][:120],
                "original": orig_text,
                "fixed": fixed_text,
            })
            sample_idx += 1

    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  Written: {sample_path} (review samples)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Humanize Josi training data")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    parser.add_argument("--stats", action="store_true", help="Stats only")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    input_path = Path(args.input) if args.input else DATA_DIR / "train_explainer.jsonl"
    output_path = Path(args.output) if args.output else DATA_DIR / "train_explainer_v5.jsonl"

    process_file(input_path, output_path, dry_run=args.dry_run, stats_only=args.stats)


if __name__ == "__main__":
    main()
