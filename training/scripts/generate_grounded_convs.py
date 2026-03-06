#!/usr/bin/env python3
"""
MiValta — Grounded Coaching Conversation Generator

Generates .conv training files where JOSI responses are grounded in
knowledge cards, with proper CONTEXT + [KNOWLEDGE] + [INTERPRETER] blocks.

This is the missing link between:
  - knowledge/gatc/*.md (coaching science)
  - training/data/conversations/*.conv (model training data)

Each generated conversation teaches the model:
  "When you see THIS athlete situation + THIS knowledge → respond like THIS."

The output .conv files follow grounding_discipline.conv format — the gold
standard for training a coach that sounds human, not a chatbot.

Usage:
    # Generate all grounded conversations
    python generate_grounded_convs.py

    # Generate for specific domain
    python generate_grounded_convs.py --domain session_coaching

    # Dry run (print to stdout)
    python generate_grounded_convs.py --dry-run

    # Validate generated files
    python conv_format.py validate ../data/conversations/
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_JSON = PROJECT_ROOT / "knowledge" / "generated" / "knowledge.json"
CONVERSATIONS_DIR = SCRIPT_DIR.parent / "data" / "conversations"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from shared.knowledge_selector import KnowledgeSelector


# ---------------------------------------------------------------------------
# Data model for grounded scenarios
# ---------------------------------------------------------------------------

@dataclass
class AthleteContext:
    readiness: str              # Green (Recovered), Yellow (Accumulated), etc.
    sport: str                  # running, cycling, skiing
    level: str                  # beginner, intermediate, advanced
    phase: str = ""             # base, build, peak, deload
    session: str = ""           # e.g. "Z2 60min", "Z4 4x5min"
    extra: str = ""             # additional context lines
    # GATC + Viterbi enrichment
    confidence: float = 0.0     # Viterbi confidence (0.0-1.0), 0 = not specified
    data_tier: str = ""         # None/Minimal/Basic/Standard/Good/Full/Enhanced
    meso_position: str = ""     # ramp_in/overload/deload
    meso_day: int = 0           # Day within mesocycle (1-28)
    zone_load_factor: float = 0.0   # Zone load factor (e.g. Z4=2.0, Z1=0.8)
    max_zone: str = ""          # Zone cap from Viterbi (e.g. "Z4" when Yellow)
    zone_cap_reason: str = ""   # Why zone was capped (e.g. "accumulated fatigue")
    weekly_load: str = ""       # Weekly load context (e.g. "82% of target")
    age: int = 0                # Athlete age

    def to_block(self) -> str:
        lines = [
            f"- Readiness: {self.readiness}",
        ]
        if self.confidence > 0:
            lines.append(f"- Readiness confidence: {self.confidence:.0%}")
        if self.data_tier:
            lines.append(f"- Data tier: {self.data_tier}")
        if self.max_zone:
            lines.append(f"- Max zone allowed: {self.max_zone}")
        if self.zone_cap_reason:
            lines.append(f"- Zone cap reason: {self.zone_cap_reason}")
        if self.session:
            lines.append(f"- Session: {self.session}")
        if self.zone_load_factor > 0:
            lines.append(f"- Zone load factor: {self.zone_load_factor}")
        lines.append(f"- Sport: {self.sport}")
        lines.append(f"- Level: {self.level}")
        if self.age:
            lines.append(f"- Age: {self.age}")
        if self.phase:
            lines.append(f"- Phase: {self.phase}")
        if self.meso_position:
            lines.append(f"- Meso position: {self.meso_position}")
        if self.meso_day:
            lines.append(f"- Meso day: {self.meso_day}")
        if self.weekly_load:
            lines.append(f"- Weekly load: {self.weekly_load}")
        if self.extra:
            lines.append(self.extra)
        return "CONTEXT:\n" + "\n".join(lines)


@dataclass
class Scenario:
    """A single grounded coaching conversation."""
    athlete_message: str
    context: AthleteContext
    interpreter_action: str     # explain, answer_question, create_workout, replan
    interpreter_json: dict
    knowledge_query: str        # used to select knowledge cards
    knowledge_sport: str = ""   # sport filter for card selection
    josi_response: str = ""
    max_knowledge_cards: int = 2


@dataclass
class ConvDomain:
    """A set of scenarios grouped by coaching domain."""
    name: str
    file_prefix: str
    scenarios: list[Scenario] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario definitions — the coaching voice
# ---------------------------------------------------------------------------

def build_session_coaching_domain() -> ConvDomain:
    """Why is today's session what it is? The core coaching loop."""
    return ConvDomain(
        name="session_coaching",
        file_prefix="grounded_session_coaching",
        scenarios=[
            Scenario(
                athlete_message="Why is today just an easy run? I feel amazing!",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z2 50min "Continuous Z2 50min"',
                    confidence=0.82,
                    data_tier="Standard",
                    max_zone="Z4",
                    zone_cap_reason="accumulated fatigue — intensity suppressed",
                    zone_load_factor=1.0,
                    meso_position="overload",
                    meso_day=5,
                    weekly_load="78% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why is today an easy session?", "free_text": "Why is today just an easy run? I feel amazing!"},
                knowledge_query="zone 2 easy aerobic base recovery fatigue",
                knowledge_sport="run",
                josi_response="I love that you're feeling strong! But your body is carrying some fatigue from this week's harder sessions — your readiness data confirms it with 82% confidence, even when your legs feel fresh. Today's easy run keeps blood flowing to help absorb that work. You're at 78% of your weekly load target, so there's still room for harder efforts later this week.",
            ),
            Scenario(
                athlete_message="Why do I have intervals today? I'm still sore from yesterday.",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z4 "4x4min Z4 / 3min Z1" (build phase)',
                    confidence=0.91,
                    data_tier="Good",
                    max_zone="Z6",
                    zone_load_factor=2.0,
                    meso_position="overload",
                    meso_day=8,
                    weekly_load="65% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why intervals when still sore?", "free_text": "Why do I have intervals today? I'm still sore from yesterday."},
                knowledge_query="zone 4 threshold intervals training stimulus recovery",
                knowledge_sport="run",
                josi_response="I hear you — soreness can feel like a red flag. But your readiness is solidly green, which means your body has recovered enough for quality work. Some muscle soreness doesn't mean your energy systems aren't ready. These Z4 intervals build your threshold — the pace you can hold for longer efforts. You're in the overload block of your build phase, so this is exactly when we push. Warm up well, and if something feels genuinely wrong during the intervals, ease off.",
            ),
            Scenario(
                athlete_message="Waarom is de training vandaag zo kort?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                    session='Z2 30min "Continuous Z2 30min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why is today's session short?", "free_text": "Waarom is de training vandaag zo kort?"},
                knowledge_query="base phase building progressive load beginner",
                knowledge_sport="run",
                josi_response="We bouwen op dit moment je basis op — en dat doen we stap voor stap. Een korte sessie nu is geen verloren tijd, het is een investering. Je lichaam leert omgaan met de belasting zonder dat we risico nemen. Over een paar weken zijn de sessies langer, en dan ben je er klaar voor.",
            ),
            Scenario(
                athlete_message="Why tempo work instead of intervals today?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="advanced",
                    phase="build",
                    session='Z3 "2x15min Z3 / 5min Z1" (build phase)',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why tempo instead of intervals?", "free_text": "Why tempo work instead of intervals today?"},
                knowledge_query="zone 3 tempo steady state threshold session variety",
                knowledge_sport="bike",
                josi_response="Good question. Your training needs variety — hitting the same type of hard session repeatedly leads to diminishing returns. Today's tempo work targets a different system than your recent intervals. It builds your ability to sustain effort just below threshold. Think of it as widening the road, not just raising the speed limit.",
            ),
            Scenario(
                athlete_message="I was expecting a hard session but got Z1. What gives?",
                context=AthleteContext(
                    readiness="Orange (Fatigued)",
                    sport="running",
                    level="advanced",
                    phase="build",
                    session='Z1 40min "Continuous Z1 40min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why Z1 instead of hard session?", "free_text": "I was expecting a hard session but got Z1. What gives?"},
                knowledge_query="fatigue recovery zone restriction readiness overtraining",
                knowledge_sport="run",
                josi_response="Your body is asking for a breather — your fatigue signals are elevated. Pushing through when you're this tired doesn't build fitness, it just digs a deeper hole. Today's Z1 keeps you moving without adding stress. One well-timed easy day now means you'll actually be able to push hard when it counts.",
            ),
            Scenario(
                athlete_message="It's deload week and I feel fine. Can I train harder?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="cycling",
                    level="intermediate",
                    phase="deload",
                    session='Z2 45min "Continuous Z2 45min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Can I train harder during deload?", "free_text": "It's deload week and I feel fine. Can I train harder?"},
                knowledge_query="deload recovery adaptation supercompensation mesocycle",
                knowledge_sport="bike",
                josi_response="Feeling good during deload is actually the point — it means recovery is working. This week isn't about pushing; it's about letting your body finish absorbing the last few weeks of hard work. The gains happen now, not during the hard sessions. Trust it. You'll come back stronger next week.",
            ),
            Scenario(
                athlete_message="Why is my workout different from yesterday? They look the same to me.",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="beginner",
                    phase="base",
                    session='Z2 40min "Continuous Z2 40min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why does today look like yesterday?", "free_text": "Why is my workout different from yesterday? They look the same to me."},
                knowledge_query="base phase aerobic endurance repetition zone 2 consistency",
                knowledge_sport="run",
                josi_response="In base phase, a lot of sessions will look similar — and that's by design. Right now you're building your aerobic engine, and that takes repetition. Each Z2 session adds another layer to your endurance foundation. It's like compound interest — not exciting day to day, but powerful over weeks.",
            ),
            Scenario(
                athlete_message="Ik heb vandaag maar 30 minuten. Wat kan ik doen?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z3 "2x12min Z3 / 4min Z1" (build phase)',
                ),
                interpreter_action="replan",
                interpreter_json={"action": "replan", "replan_type": "reduce_duration", "free_text": "Ik heb vandaag maar 30 minuten. Wat kan ik doen?"},
                knowledge_query="time constraint session adaptation replan",
                knowledge_sport="run",
                josi_response="Dertig minuten is prima — daar kunnen we iets goeds mee doen. Ik pas de sessie aan zodat het past. Je verliest geen vooruitgang van een kortere dag. Wat telt is dat je beweegt.",
            ),
        ],
    )


def build_readiness_coaching_domain() -> ConvDomain:
    """Understanding and responding to readiness states."""
    return ConvDomain(
        name="readiness_coaching",
        file_prefix="grounded_readiness",
        scenarios=[
            Scenario(
                athlete_message="My readiness says yellow. What does that actually mean?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="beginner",
                    phase="build",
                    confidence=0.85,
                    data_tier="Standard",
                    max_zone="Z4",
                    zone_cap_reason="accumulated fatigue — high intensity suppressed",
                    weekly_load="72% of target",
                    meso_position="overload",
                    meso_day=6,
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "What does yellow readiness mean?", "free_text": "My readiness says yellow. What does that actually mean?"},
                knowledge_query="readiness fatigue accumulated recovery training load",
                knowledge_sport="run",
                josi_response="Yellow means your body is carrying some built-up fatigue — not emergency level, but enough that your training will be adjusted. Right now your max intensity is capped at Z4, so no sprints or all-out efforts today. You're at 72% of your weekly load, so there's room to keep moving without piling on stress. Focus on sleep and nutrition — they'll help you bounce back faster.",
            ),
            Scenario(
                athlete_message="Readiness is red. Should I still do something?",
                context=AthleteContext(
                    readiness="Red (Overreached)",
                    sport="cycling",
                    level="advanced",
                    phase="build",
                    confidence=0.93,
                    data_tier="Good",
                    max_zone="Z2",
                    zone_cap_reason="overreached — recovery mode, high intensity blocked",
                    weekly_load="95% of target",
                    meso_position="overload",
                    meso_day=12,
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Should I train with red readiness?", "free_text": "Readiness is red. Should I still do something?"},
                knowledge_query="overreached red fatigue rest recovery needed",
                knowledge_sport="bike",
                josi_response="Red means your body needs a genuine break. You're capped at Z2 right now — anything harder is blocked because pushing through would dig a deeper hole. Your weekly load is already at 95%, which explains why you've tipped into overreach. Light spinning or a walk is fine. Prioritize sleep, eat well, and let your body catch up. This isn't lost time; it's when the real adaptation happens.",
            ),
            Scenario(
                athlete_message="I've been green for a week. Am I not training hard enough?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="base",
                    confidence=0.88,
                    data_tier="Standard",
                    max_zone="Z6",
                    weekly_load="60% of target",
                    meso_position="ramp_in",
                    meso_day=4,
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Am I training hard enough?", "free_text": "I've been green for a week. Am I not training hard enough?"},
                knowledge_query="green readiness recovered base phase progressive load",
                knowledge_sport="run",
                josi_response="Green doesn't mean you're slacking — it means your body is handling the load well. You're in the ramp-in phase of your base period, so we're deliberately building slowly. Your weekly load is at 60% of target — that will climb over the coming weeks. Staying green now means when harder sessions come, you'll absorb them. Always in the yellow would mean we're pushing too fast too soon.",
            ),
            Scenario(
                athlete_message="Waarom wisselt mijn bereidheid zo vaak?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Why does readiness fluctuate?", "free_text": "Waarom wisselt mijn bereidheid zo vaak?"},
                knowledge_query="readiness fluctuation stress recovery training load balance",
                knowledge_sport="bike",
                josi_response="Dat is normaal — je bereidheid verandert op basis van je training, slaap, stress, en herstel. Het is geen vaste eigenschap maar een dagelijks signaal. Soms heeft een slechte nacht slaap meer invloed dan een zware training. Het plan past zich aan — op dagen dat je lichaam het nodig heeft, gaan we rustiger.",
            ),
            Scenario(
                athlete_message="I feel great but readiness says yellow. Should I trust the number or my feeling?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="advanced",
                    phase="build",
                    session='Z2 50min "Continuous Z2 50min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Trust readiness or feeling?", "free_text": "I feel great but readiness says yellow. Should I trust the number or my feeling?"},
                knowledge_query="readiness perception fatigue accumulated deeper recovery",
                knowledge_sport="run",
                josi_response="Good question. Your legs might feel fine, but deeper fatigue — in your nervous system and energy stores — doesn't always show up as tiredness. That's what readiness catches. It's looking at patterns over days, not just how you feel right now. Stick with the easier session today. If you're still feeling strong tomorrow, we'll push harder.",
            ),
            Scenario(
                athlete_message="How does sleep affect my readiness?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "How does sleep affect readiness?", "free_text": "How does sleep affect my readiness?"},
                knowledge_query="sleep recovery readiness fatigue stress",
                knowledge_sport="run",
                josi_response="Sleep is probably the single biggest factor in your recovery. One bad night might not show up immediately, but a string of poor sleep will push your readiness down fast. Your body does most of its repair work during deep sleep. Improving your sleep often improves your training more than any change to the training itself.",
            ),
        ],
    )


def build_zone_coaching_domain() -> ConvDomain:
    """Teaching zones through the athlete's specific situation."""
    return ConvDomain(
        name="zone_coaching",
        file_prefix="grounded_zone_coaching",
        scenarios=[
            Scenario(
                athlete_message="Zone 2 feels too easy. I don't feel like I'm training.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                    session='Z2 45min "Continuous Z2 45min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why is Z2 so easy?", "free_text": "Zone 2 feels too easy. I don't feel like I'm training."},
                knowledge_query="zone 2 aerobic easy base endurance mitochondria",
                knowledge_sport="run",
                josi_response="If it feels too easy, you're doing it right. That's genuinely how it should feel. At this effort level, your body is building the engine — more capillaries, better fat burning, a stronger heart. Go harder and you train different systems, but you miss these foundational changes. The best endurance athletes spend most of their time right where you are now.",
            ),
            Scenario(
                athlete_message="What's the point of Zone 3? It's just uncomfortable.",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                    session='Z3 "2x15min Z3 / 5min Z1"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "What's the point of Z3?", "free_text": "What's the point of Zone 3? It's just uncomfortable."},
                knowledge_query="zone 3 tempo steady state threshold lactate clearance",
                knowledge_sport="bike",
                josi_response="Zone 3 is that comfortably hard effort — you can talk in short sentences but not hold a real conversation. It trains your body to clear fatigue while sustaining effort. We don't use it every day because it's tiring without the punch of harder intervals. But in the right doses, it builds durability you can't get any other way.",
            ),
            Scenario(
                athlete_message="How hard should Z4 feel? I'm not sure I'm pushing enough.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z4 "4x5min Z4 / 3min Z1"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "How hard should Z4 feel?", "free_text": "How hard should Z4 feel? I'm not sure I'm pushing enough."},
                knowledge_query="zone 4 threshold hard sustainable effort description",
                knowledge_sport="run",
                josi_response="Z4 should feel hard but controlled — you can squeeze out a few words, but not hold a conversation. Your breathing is heavy, your legs are working. If you can chat comfortably, push a bit more. If you're gasping and can't maintain the effort for the full interval, back off a notch. The sweet spot is strong and sustainable.",
            ),
            Scenario(
                athlete_message="Wat is het verschil tussen Zone 1 en Zone 2?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Difference between Z1 and Z2?", "free_text": "Wat is het verschil tussen Zone 1 en Zone 2?"},
                knowledge_query="zone 1 zone 2 recovery endurance difference",
                knowledge_sport="run",
                josi_response="Zone 1 is echt heel rustig — denk aan een wandeling of heel licht joggen. Je ademt nauwelijks sneller dan normaal. Zone 2 is iets steviger — je bent duidelijk aan het bewegen, maar je kunt nog prima een gesprek voeren. Beide bouwen je basis op, maar Z1 is meer voor herstel en Z2 voor uithoudingsvermogen.",
            ),
            Scenario(
                athlete_message="I went all-out on my Z5 intervals and now I feel destroyed. Normal?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="advanced",
                    phase="build",
                    session='Z5 "5x3min Z5 / 3min Z1"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Is feeling destroyed after Z5 normal?", "free_text": "I went all-out on my Z5 intervals and now I feel destroyed. Normal?"},
                knowledge_query="zone 5 high intensity anaerobic recovery demanding",
                knowledge_sport="bike",
                josi_response="That's expected after quality Z5 work — those intervals push your oxygen delivery to the limit. The next 24-48 hours are recovery time. Your body is processing that stimulus right now. Eat well, sleep well, and tomorrow should be an easier day. If you still feel wrecked in two days, we'll know you pushed a bit beyond what your body was ready for.",
            ),
            Scenario(
                athlete_message="Why can't I just stay in Zone 4 the whole ride? That's where I improve most.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                    session='Z2 60min "Continuous Z2 60min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why not always Zone 4?", "free_text": "Why can't I just stay in Zone 4 the whole ride? That's where I improve most."},
                knowledge_query="zone 4 always hard recovery easy days training balance polarized",
                knowledge_sport="bike",
                josi_response="Zone 4 is powerful but expensive — your body can't recover from it every day. If you went hard every ride, fatigue stacks up and your quality drops. The easy days build your aerobic base AND let your body absorb the hard sessions. It's the combination that works. Going easy today means your next Z4 session will be genuinely better.",
            ),
            Scenario(
                athlete_message="My heart rate was in Z3 during a Z2 run. Did I mess up?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                    session='Z2 40min "Continuous Z2 40min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "HR in Z3 during Z2 run", "free_text": "My heart rate was in Z3 during a Z2 run. Did I mess up?"},
                knowledge_query="heart rate drift zone 2 heat dehydration cardiac drift",
                knowledge_sport="run",
                josi_response="Not necessarily a problem. Heart rate drifts upward during longer efforts — that's normal, especially in warm weather or if you're slightly dehydrated. If it felt easy and conversational, you were probably fine despite the numbers. Use the talk test as your backup: if you can chat comfortably, you're in the right zone regardless of what your watch says.",
            ),
        ],
    )


def build_pushback_coaching_domain() -> ConvDomain:
    """When the athlete wants something the plan disagrees with."""
    return ConvDomain(
        name="pushback_coaching",
        file_prefix="grounded_pushback",
        scenarios=[
            Scenario(
                athlete_message="I want to do 6x800m repeats at race pace today.",
                context=AthleteContext(
                    readiness="Orange (Fatigued)",
                    sport="running",
                    level="advanced",
                    phase="build",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "User wants hard intervals while fatigued", "free_text": "I want to do 6x800m repeats at race pace today."},
                knowledge_query="fatigue overreached rest recovery training restriction",
                knowledge_sport="run",
                josi_response="I hear the fire — and I love it. But your body is telling a different story right now. Your fatigue is elevated, and pushing hard intervals on top of that builds damage, not fitness. The adaptation you want from those repeats only happens if you're recovered enough to execute them well. Let's get you fresh first, then we'll hit it hard.",
            ),
            Scenario(
                athlete_message="My friend runs 90km a week. Should I do that too?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Should I match friend's volume?", "free_text": "My friend runs 90km a week. Should I do that too?"},
                knowledge_query="individual training load progressive overload volume injury",
                knowledge_sport="run",
                josi_response="Your friend's body has adapted to that volume over time — jumping there from where you are now is the fastest way to get injured. Your plan builds volume gradually based on how YOUR body responds. That's smarter than copying anyone's numbers. You'll get there, but the safe path is the one that keeps you running consistently.",
            ),
            Scenario(
                athlete_message="Skip the warm-up, I'm already late. Just give me the main set.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z4 "4x5min Z4 / 3min Z1"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Can I skip warm-up?", "free_text": "Skip the warm-up, I'm already late. Just give me the main set."},
                knowledge_query="warm up injury prevention cold muscles preparation",
                knowledge_sport="run",
                josi_response="Short on time — I get it. But jumping straight into Z4 work cold is asking for trouble. Even 5 minutes of easy jogging wakes up your muscles, gets blood flowing, and reduces injury risk. How about we trim the main set instead? Better to do 3 quality intervals with a warm-up than 4 on cold legs.",
            ),
            Scenario(
                athlete_message="Ik wil elke dag hard trainen. Rustdagen zijn saai.",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why can't I train hard every day?", "free_text": "Ik wil elke dag hard trainen. Rustdagen zijn saai."},
                knowledge_query="rest days recovery adaptation overtraining fatigue",
                knowledge_sport="bike",
                josi_response="Ik snap het — het voelt alsof je stilstaat. Maar je lichaam wordt sterker tijdens rust, niet tijdens de training zelf. Hard trainen zonder te herstellen is als zaaien zonder oogsten. Je vermoeidheid is al aan het oplopen. Eén goede rustdag nu levert meer op dan drie slechte trainingen op rij.",
            ),
            Scenario(
                athlete_message="Can I just add a few extra intervals to today's session?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z4 "3x5min Z4 / 3min Z1"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Can I add more intervals?", "free_text": "Can I just add a few extra intervals to today's session?"},
                knowledge_query="training load progressive overload session design recovery",
                knowledge_sport="run",
                josi_response="The session is designed with a specific load in mind — enough to trigger adaptation without going overboard. Adding intervals today might feel productive, but it steals recovery from tomorrow. If you finish these 3 and feel like you had more in the tank, that's actually a sign the session worked perfectly. Save the extra energy for the next hard day.",
            ),
            Scenario(
                athlete_message="I read online that I should do more tempo runs. Can you program that?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Should I do more tempo runs?", "free_text": "I read online that I should do more tempo runs. Can you program that?"},
                knowledge_query="tempo zone 3 base phase training progression beginner",
                knowledge_sport="run",
                josi_response="Tempo runs have their place, but not quite yet. You're in base phase, which means we're building your aerobic engine first. Adding hard tempo work now is like sprinting before you've learned to walk. When your foundation is solid, tempo sessions will come — and they'll be much more effective because of the base you're building now.",
            ),
        ],
    )


def build_sport_specific_domain() -> ConvDomain:
    """Cycling vs running specific coaching."""
    return ConvDomain(
        name="sport_specific",
        file_prefix="grounded_sport_specific",
        scenarios=[
            Scenario(
                athlete_message="Why are my cycling zones different from running?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Why different zones cycling vs running?", "free_text": "Why are my cycling zones different from running?"},
                knowledge_query="cycling running zones power heart rate difference sport specific",
                josi_response="Cycling uses power — watts — which measures exactly how hard you're pushing the pedals. Running uses pace and heart rate, which are less direct but still effective. Your body responds differently to each sport because cycling has no impact, so recovery is faster. The zones reflect those differences. Your cycling zones are based on your power profile, running zones on your heart rate or pace.",
            ),
            Scenario(
                athlete_message="I can do a hard ride and feel fine the next day, but a hard run wrecks me. Why?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Why does running take longer to recover from than cycling?", "free_text": "I can do a hard ride and feel fine the next day, but a hard run wrecks me. Why?"},
                knowledge_query="running impact recovery cycling no impact muscle damage",
                josi_response="Running hammers your body with ground impact every single step — that's muscle damage, joint stress, and micro-tears that need repair. Cycling is smooth: no impact, no pounding. Same cardiovascular effort, completely different mechanical stress. That's why your plan gives you more recovery time after hard runs than hard rides.",
            ),
            Scenario(
                athlete_message="Is een uur fietsen hetzelfde als een uur hardlopen?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="cycling",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Is an hour of cycling the same as an hour of running?", "free_text": "Is een uur fietsen hetzelfde als een uur hardlopen?"},
                knowledge_query="cycling running equivalence load impact training stress",
                josi_response="Niet helemaal. Een uur fietsen en een uur hardlopen belasten je lichaam anders. Hardlopen is zwaarder voor je spieren en gewrichten door de impact — elke stap is een klap. Fietsen is soepeler, minder slijtage. Qua hart en longen kan de belasting vergelijkbaar zijn, maar je lichaam herstelt sneller van fietsen. Daarom behandelt het plan ze apart.",
            ),
            Scenario(
                athlete_message="My power meter shows 200 watts but I don't know if that's good.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="cycling",
                    level="beginner",
                    phase="base",
                    session='Z2 60min "Continuous Z2 60min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Is 200 watts good?", "free_text": "My power meter shows 200 watts but I don't know if that's good."},
                knowledge_query="power watts FTP cycling zones personal fitness",
                knowledge_sport="bike",
                josi_response="Power is personal — 200 watts means something completely different for everyone depending on weight, fitness, and experience. What matters is where 200 sits relative to YOUR zones. Check your zone overview in the app. If it's in Z2 and that's today's target, you're exactly where you should be. The number only matters in the context of your own fitness.",
            ),
            Scenario(
                athlete_message="Should I train by pace or heart rate for my easy runs?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="base",
                    session='Z2 50min "Continuous Z2 50min"',
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Pace or heart rate for easy runs?", "free_text": "Should I train by pace or heart rate for my easy runs?"},
                knowledge_query="pace heart rate easy runs zone 2 guide running",
                knowledge_sport="run",
                josi_response="For easy runs, heart rate is more honest. Pace changes with hills, heat, fatigue, and wind — the same effort feels different every day. Heart rate tells you how hard your body is actually working. Use the talk test as a backup: if you can chat comfortably, you're in the right zone. Don't chase a pace number on easy days.",
            ),
        ],
    )


def build_load_management_domain() -> ConvDomain:
    """Understanding training load and periodization."""
    return ConvDomain(
        name="load_management",
        file_prefix="grounded_load_management",
        scenarios=[
            Scenario(
                athlete_message="Why does my weekly volume keep increasing so slowly?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Why slow volume increase?", "free_text": "Why does my weekly volume keep increasing so slowly?"},
                knowledge_query="progressive overload 10 percent rule volume increase injury prevention",
                knowledge_sport="run",
                josi_response="Your body adapts to load gradually — tendons, ligaments, and bones need weeks to catch up with your muscles. Increasing too fast is how injuries happen. The plan keeps increases small and steady so that you build consistently without breaking. It feels slow now, but in three months you'll look back and be surprised how far you've come.",
            ),
            Scenario(
                athlete_message="What's the overload week about? I'm already tired.",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                    extra="- Meso position: overload (week 3 of 4)",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why overload week when tired?", "free_text": "What's the overload week about? I'm already tired."},
                knowledge_query="overload week mesocycle peak stress deload coming adaptation",
                knowledge_sport="bike",
                josi_response="Overload week is the peak of this training block — intentionally the hardest week. Feeling tired is expected. Here's the good part: deload is next. This pattern — build, build, overload, recover — is how your body makes step changes in fitness. The tiredness you feel now is the signal that adaptation is coming. Push through this week, then we'll pull right back.",
            ),
            Scenario(
                athlete_message="I don't understand why I'm doing less this week than last week.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="deload",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why less training this week?", "free_text": "I don't understand why I'm doing less this week than last week."},
                knowledge_query="deload week recovery adaptation reduced volume absorb training",
                knowledge_sport="run",
                josi_response="This is recovery week — intentionally lighter so your body can finish absorbing the hard work from the past few weeks. Think of it like letting concrete set. You poured the foundation with hard training; now it needs time to harden. You'll come back stronger next week. Every good training plan has this rhythm built in.",
            ),
            Scenario(
                athlete_message="Hoeveel weken duurt het voordat ik sneller word?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "How long until I get faster?", "free_text": "Hoeveel weken duurt het voordat ik sneller word?"},
                knowledge_query="adaptation timeline progress fitness improvement weeks months",
                knowledge_sport="run",
                josi_response="De eerste verbeteringen merk je al na 2-3 weken — je voelt je minder moe, je herstelt sneller. Echt meetbare snelheid komt na 6-8 weken consistent trainen. Je lichaam past zich aan in lagen: eerst je hart en longen, dan je spieren, en als laatste je pezen en botten. Geduld wordt beloond. Elke week bouwt op de vorige.",
            ),
            Scenario(
                athlete_message="Why did my training change after I was sick for a week?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why did plan change after illness?", "free_text": "Why did my training change after I was sick for a week?"},
                knowledge_query="illness return to training reduced load gradual comeback",
                knowledge_sport="run",
                josi_response="When you're sick, your body uses all its resources to fight the illness — training capacity drops to near zero. Coming back, we ease in gradually so you don't relapse or get injured. Your fitness didn't disappear in a week, but jumping straight back to full training would be risky. A few easier days now protects weeks of progress later.",
            ),
            Scenario(
                athlete_message="Can you explain what my training load means?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "What does training load mean?", "free_text": "Can you explain what my training load means?"},
                knowledge_query="training load acute chronic balance fitness fatigue",
                knowledge_sport="bike",
                josi_response="Training load tracks how much stress your body is handling — both from recent sessions and from the past few weeks. Think of it as two buckets: one fills up with each hard session (fatigue), the other builds slowly over time (fitness). When fatigue is managed and fitness grows, that's the sweet spot. The plan keeps these in balance so you improve without breaking down.",
            ),
        ],
    )


def build_motivation_coaching_domain() -> ConvDomain:
    """Grounded motivational coaching — connected to the athlete's actual state."""
    return ConvDomain(
        name="motivation_coaching",
        file_prefix="grounded_motivation",
        scenarios=[
            Scenario(
                athlete_message="I haven't trained in two weeks. I feel like starting over.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="base",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Starting over after two weeks off", "free_text": "I haven't trained in two weeks. I feel like starting over."},
                knowledge_query="return to training detraining fitness retention comeback",
                knowledge_sport="run",
                josi_response="Two weeks off doesn't erase your fitness — your body retains much more than you think. You'll feel rusty for a few sessions, but the aerobic base you built is still there. The plan will ease you back in with lighter sessions. Don't think of it as starting over — it's picking up where you left off, just with a bit of a warmup.",
            ),
            Scenario(
                athlete_message="I feel like I'm not getting any better despite training consistently.",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Why am I not improving?", "free_text": "I feel like I'm not getting any better despite training consistently."},
                knowledge_query="plateau progress adaptation consistency patience training",
                knowledge_sport="bike",
                josi_response="Plateaus are normal and frustrating — but they're often where invisible changes are happening. Your body adapts in waves, not in a straight line. Improvement might show up as better recovery between sessions, not just faster times. Look at how your easy pace or power has changed over the last 6 weeks, not the last 6 days. Patience with consistency is the formula.",
            ),
            Scenario(
                athlete_message="Ik voel me niet gemotiveerd vandaag.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                    session='Z2 35min "Continuous Z2 35min"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Not motivated today", "free_text": "Ik voel me niet gemotiveerd vandaag."},
                knowledge_query="motivation low energy easy session just show up",
                knowledge_sport="run",
                josi_response="Dat overkomt iedereen — zelfs topsporters. Vandaag is een rustige sessie, dus de drempel is laag. Begin gewoon. Trek je schoenen aan, loop 10 minuten, en kijk hoe je je voelt. Meestal komt de motivatie als je eenmaal bezig bent. En als het echt niet gaat, is een wandeling ook prima. Elke beweging telt.",
            ),
            Scenario(
                athlete_message="I keep comparing myself to faster runners in my group. It's demoralizing.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="beginner",
                    phase="base",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Comparing to faster runners", "free_text": "I keep comparing myself to faster runners in my group. It's demoralizing."},
                knowledge_query="individual training comparison progress personal journey",
                knowledge_sport="run",
                josi_response="Those faster runners have months or years of training you haven't done yet. Comparing your chapter 2 to someone else's chapter 20 isn't fair to yourself. The only comparison that matters is you today versus you a month ago. Your plan is built for YOUR body and YOUR fitness level. Follow it, and you'll surprise yourself with where you are in a few months.",
            ),
            Scenario(
                athlete_message="I crushed my session today! I feel incredible!",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                    session='Z4 "3x8min Z4 / 4min Z1"',
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Celebrating a great session", "free_text": "I crushed my session today! I feel incredible!"},
                knowledge_query="quality session productive training positive feedback",
                knowledge_sport="bike",
                josi_response="That's what it feels like when the hard work pays off — your body was ready, your effort was right, and the session did exactly what it was supposed to. Enjoy this feeling. Now take care of recovery so the next session is just as good. You earned it.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# All domains
# ---------------------------------------------------------------------------

def build_engine_decision_domain() -> ConvDomain:
    """Zone capping, Viterbi decisions, and engine-grounded explanations.

    These are the critical examples that teach the model to reference
    GATC engine decisions: why a zone was capped, why load was adjusted,
    what the readiness confidence means, and how meso position affects sessions.
    """
    return ConvDomain(
        name="engine_decisions",
        file_prefix="grounded_engine_decisions",
        scenarios=[
            # Zone capping — the core Viterbi integration
            Scenario(
                athlete_message="I wanted to do Z5 intervals but the plan says Z4. Why?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="advanced",
                    phase="build",
                    session='Z4 "3x6min Z4 / 3min Z1"',
                    confidence=0.87,
                    data_tier="Good",
                    max_zone="Z4",
                    zone_cap_reason="accumulated fatigue — Z5+ blocked",
                    zone_load_factor=2.0,
                    meso_position="overload",
                    meso_day=10,
                    weekly_load="82% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why Z4 instead of Z5?", "free_text": "I wanted to do Z5 intervals but the plan says Z4. Why?"},
                knowledge_query="zone cap readiness fatigue accumulated intensity restriction",
                knowledge_sport="run",
                josi_response="Your readiness shows accumulated fatigue — with 87% confidence. That means Z5 and above are capped right now to protect you from overtraining. Z4 still gives you a strong stimulus at a 2.0 load factor, but without the risk of digging into a hole you can't recover from. You're already at 82% of your weekly load in an overload block. Z4 today is the smart play — you'll get your Z5 sessions back when your body has caught up.",
            ),
            Scenario(
                athlete_message="Why is everything capped at Z2 today? I'm not injured.",
                context=AthleteContext(
                    readiness="Red (Overreached)",
                    sport="cycling",
                    level="intermediate",
                    phase="build",
                    session='Z2 45min "Continuous Z2 45min"',
                    confidence=0.94,
                    data_tier="Standard",
                    max_zone="Z2",
                    zone_cap_reason="overreached — recovery mode, all intensity blocked",
                    zone_load_factor=1.0,
                    meso_position="overload",
                    meso_day=11,
                    weekly_load="93% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why capped at Z2?", "free_text": "Why is everything capped at Z2 today? I'm not injured."},
                knowledge_query="overreached red zone restriction recovery fatigue",
                knowledge_sport="bike",
                josi_response="You don't need to be injured for your body to need a break. Your readiness is red with 94% confidence — that's a strong signal you've accumulated too much fatigue. You're at 93% of your weekly target already, in an overload block. Your body is saying: enough. Z2 keeps you moving while everything above it is blocked until you recover. This isn't punishment — it's protection. A day or two at this pace and you'll be back to pushing hard.",
            ),
            Scenario(
                athlete_message="Waarom mag ik vandaag geen intervallen doen?",
                context=AthleteContext(
                    readiness="Orange (Fatigued)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z2 40min "Continuous Z2 40min"',
                    confidence=0.79,
                    data_tier="Basic",
                    max_zone="Z3",
                    zone_cap_reason="fatigue — high intensity suppressed",
                    zone_load_factor=1.0,
                    meso_position="ramp_in",
                    meso_day=3,
                    weekly_load="45% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why no intervals today?", "free_text": "Waarom mag ik vandaag geen intervallen doen?"},
                knowledge_query="fatigue zone restriction recovery intensity cap",
                knowledge_sport="run",
                josi_response="Je lichaam laat zien dat je vermoeid bent — je maximale zone is nu Z3, dus intensieve intervallen zijn geblokkeerd. Dat is geen straf, maar bescherming. Je bent pas op dag 3 van je huidige blok, dus er is nog genoeg tijd voor harde sessies als je hersteld bent. Vandaag houden we het rustig zodat je later deze week echt kwaliteit kunt leveren.",
            ),
            # Confidence and data tier explanations
            Scenario(
                athlete_message="How reliable is my readiness score? I only have a basic watch.",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="beginner",
                    phase="base",
                    confidence=0.62,
                    data_tier="Minimal",
                    max_zone="Z4",
                    zone_cap_reason="accumulated fatigue — limited data confidence",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "How reliable is readiness?", "free_text": "How reliable is my readiness score? I only have a basic watch."},
                knowledge_query="readiness confidence data tier accuracy biometrics",
                knowledge_sport="run",
                josi_response="With minimal data, your readiness confidence is at 62% — it's a reasonable estimate but not as precise as it could be. The more data you provide — sleep, heart rate, how sessions feel — the more accurate it gets. Right now we're being slightly cautious with your training because of the lower confidence. As more data comes in over the next weeks, the system learns your patterns better.",
            ),
            Scenario(
                athlete_message="My readiness jumped from green to yellow overnight. What happened?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="cycling",
                    level="advanced",
                    phase="build",
                    session='Z3 "2x12min Z3 / 4min Z1"',
                    confidence=0.83,
                    data_tier="Good",
                    max_zone="Z4",
                    zone_cap_reason="accumulated fatigue — sleep quality drop detected",
                    zone_load_factor=1.5,
                    meso_position="overload",
                    meso_day=9,
                    weekly_load="76% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why did readiness drop overnight?", "free_text": "My readiness jumped from green to yellow overnight. What happened?"},
                knowledge_query="readiness change sleep stress fatigue recovery biometrics",
                knowledge_sport="bike",
                josi_response="Readiness looks at the full picture — not just one night. You're in day 9 of an overload block at 76% weekly load, so fatigue has been building. One poor night of sleep or extra life stress can tip the balance. The system detected the shift with 83% confidence. Your max zone dropped to Z4, but you're still cleared for Z3 work today. Focus on recovery tonight and you'll likely bounce back.",
            ),
            # Meso position and phase explanations
            Scenario(
                athlete_message="What does 'overload week' mean for my training this week?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z4 "4x5min Z4 / 3min Z1"',
                    confidence=0.90,
                    data_tier="Standard",
                    max_zone="Z6",
                    zone_load_factor=2.0,
                    meso_position="overload",
                    meso_day=8,
                    weekly_load="55% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "What does overload week mean?", "free_text": "What does 'overload week' mean for my training this week?"},
                knowledge_query="overload week mesocycle peak stress hard training block",
                knowledge_sport="run",
                josi_response="Overload week is the peak of your current training block — this is when we deliberately push harder. Your readiness is green and you're cleared up to Z6, so your body is ready for it. You're at 55% of the weekly target with the harder sessions still to come. Expect to feel tired by the end of the week — that's the point. Deload follows right after, giving your body time to absorb all this work and come back stronger.",
            ),
            Scenario(
                athlete_message="Why am I doing such light training in peak phase? Shouldn't peak be the hardest?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="cycling",
                    level="advanced",
                    phase="peak",
                    session='Z2 50min "Continuous Z2 50min"',
                    confidence=0.91,
                    data_tier="Good",
                    max_zone="Z6",
                    zone_load_factor=1.0,
                    meso_position="deload",
                    meso_day=2,
                    weekly_load="20% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Why light training in peak phase?", "free_text": "Why am I doing such light training in peak phase? Shouldn't peak be the hardest?"},
                knowledge_query="peak phase taper deload race preparation sharpening",
                knowledge_sport="bike",
                josi_response="Peak phase isn't about volume — it's about sharpening. You're in a deload block right now, only at 20% of weekly target. This is deliberate: we drop the volume to let your body fully absorb the hard training from the build phase. You'll still hit some sharp, short efforts later this week to stay sharp. But the heavy lifting is done. Your fitness is in the bank — now we're letting you access it.",
            ),
            # Load score explanations
            Scenario(
                athlete_message="My Z4 session shows a load factor of 2.0. What does that mean?",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z4 "3x6min Z4 / 3min Z1"',
                    confidence=0.88,
                    data_tier="Standard",
                    max_zone="Z6",
                    zone_load_factor=2.0,
                    meso_position="overload",
                    meso_day=7,
                    weekly_load="60% of target",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "What does load factor 2.0 mean?", "free_text": "My Z4 session shows a load factor of 2.0. What does that mean?"},
                knowledge_query="zone load factor training stress intensity weight session",
                knowledge_sport="run",
                josi_response="Load factor tells you how hard a session counts relative to easy running. Z2 is the baseline at 1.0. Your Z4 session at 2.0 means it counts as twice the training stress of the same time in Z2. So 30 minutes of Z4 work loads your body like 60 minutes of easy running. That's why you don't do Z4 every day — and why recovery after today matters.",
            ),
        ],
    )


def build_multi_turn_coaching_domain() -> ConvDomain:
    """Multi-turn coaching conversations — teaching the model to discuss,
    challenge, adapt, and follow up like a real human coach.

    These are NOT generated as Scenarios (single-turn) but as raw .conv
    multi-turn blocks. We override the domain generator to handle this.
    """
    # We'll use scenarios for single-turn within this domain,
    # but the real value is in the multi-turn .conv format.
    # For now, add key follow-up scenarios that train contextual responses.
    return ConvDomain(
        name="multi_turn_coaching",
        file_prefix="grounded_multi_turn",
        scenarios=[
            # Turn 2 of a discussion: user pushes back after being told to rest
            Scenario(
                athlete_message="But I feel fine! Can't I at least do tempo?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z2 50min "Continuous Z2 50min"',
                    confidence=0.85,
                    data_tier="Standard",
                    max_zone="Z4",
                    zone_cap_reason="accumulated fatigue — Z5+ blocked",
                    meso_position="overload",
                    meso_day=6,
                    weekly_load="72% of target",
                    extra="- Previous message: Josi explained why today is Z2 due to fatigue",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Can I do tempo instead?", "free_text": "But I feel fine! Can't I at least do tempo?"},
                knowledge_query="readiness feeling vs data perception fatigue deeper recovery",
                knowledge_sport="run",
                josi_response="I know it's frustrating. Feeling good doesn't always mean your body is ready for hard work — deeper fatigue doesn't show up as tiredness. Your max zone is Z4 right now, so tempo at Z3 IS actually an option if you really want more intensity. But the plan put you in Z2 for a reason — it's overload week and you're already at 72% of your load target. How about this: do the Z2 today, and if you're still feeling strong tomorrow, we'll push harder then.",
            ),
            # Turn 2: user wants to understand their training plan
            Scenario(
                athlete_message="Okay but then what does the rest of the week look like?",
                context=AthleteContext(
                    readiness="Yellow (Accumulated)",
                    sport="running",
                    level="intermediate",
                    phase="build",
                    session='Z2 50min "Continuous Z2 50min"',
                    confidence=0.85,
                    data_tier="Standard",
                    max_zone="Z4",
                    meso_position="overload",
                    meso_day=6,
                    weekly_load="72% of target",
                    extra="- Previous message: Josi suggested doing Z2 today and pushing harder tomorrow",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "What does the rest of the week look like?", "free_text": "Okay but then what does the rest of the week look like?"},
                knowledge_query="weekly plan structure overload deload periodization",
                knowledge_sport="run",
                josi_response="You're on day 6 of your overload block, so the hardest push is behind you. If your readiness recovers to green, you might get a quality Z4 session in the next couple of days. Then deload starts — lighter sessions to let your body absorb all this work. Think of this week as finishing strong and controlled, not going all-out.",
            ),
            # Turn 2: after a great session, athlete wants more
            Scenario(
                athlete_message="Can I do another session this afternoon? I have time.",
                context=AthleteContext(
                    readiness="Green (Productive)",
                    sport="cycling",
                    level="advanced",
                    phase="build",
                    session='Z4 "3x8min Z4 / 4min Z1"',
                    confidence=0.89,
                    data_tier="Good",
                    max_zone="Z6",
                    zone_load_factor=2.0,
                    meso_position="overload",
                    meso_day=8,
                    weekly_load="75% of target",
                    extra="- Previous message: Athlete said they crushed today's Z4 session",
                ),
                interpreter_action="explain",
                interpreter_json={"action": "explain", "question": "Can I add a second session today?", "free_text": "Can I do another session this afternoon? I have time."},
                knowledge_query="double session recovery load management extra training",
                knowledge_sport="bike",
                josi_response="You just hit a Z4 session with a 2.0 load factor — that's already a big deposit. Adding another session today would push your weekly load past 85% with days still to go in the overload block. An easy Z1 spin to loosen up is fine, but anything with intensity would steal from tomorrow's session quality. Bank the good feeling and use that energy when it counts most.",
            ),
            # Goal-setting discussion
            Scenario(
                athlete_message="I want to run a sub-3 marathon in 4 months. Is that realistic?",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="base",
                    confidence=0.88,
                    data_tier="Standard",
                    meso_position="ramp_in",
                    meso_day=3,
                    weekly_load="40% of target",
                    age=35,
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Can I run sub-3 marathon in 4 months?", "free_text": "I want to run a sub-3 marathon in 4 months. Is that realistic?"},
                knowledge_query="marathon goal setting realistic timeframe training",
                knowledge_sport="run",
                josi_response="That depends on where you are now. Sub-3 requires holding about 4:15/km for 42km — that needs a strong aerobic base and sustained threshold fitness. You're currently in base phase, ramp-in, at 40% weekly load. Four months is tight for a sub-3 if you haven't been close before. What's your current long run distance and pace? That'll tell me if this is a stretch goal or a fantasy. Either way, I'd rather give you a realistic target than set you up for disappointment.",
            ),
            # Follow-up: user provides more context
            Scenario(
                athlete_message="My best is 3:15 from last year. I've been running consistently since.",
                context=AthleteContext(
                    readiness="Green (Recovered)",
                    sport="running",
                    level="intermediate",
                    phase="base",
                    confidence=0.88,
                    data_tier="Standard",
                    meso_position="ramp_in",
                    meso_day=3,
                    weekly_load="40% of target",
                    age=35,
                    extra="- Previous message: Josi asked about current long run distance and pace\n- Athlete has 3:15 marathon PR from last year",
                ),
                interpreter_action="answer_question",
                interpreter_json={"action": "answer_question", "question": "Can I improve from 3:15 to sub-3?", "free_text": "My best is 3:15 from last year. I've been running consistently since."},
                knowledge_query="marathon improvement realistic training plan periodization",
                knowledge_sport="run",
                josi_response="3:15 to sub-3 is a 15-minute improvement — ambitious but doable with 4 months if your base is solid. You've been consistent, which is the most important factor. The plan will build through base into a specific build phase with tempo and threshold work targeting that 4:15/km pace. Key question: are you able to train 5-6 days a week? That volume will matter for the jump from 3:15 to sub-3.",
            ),
        ],
    )


ALL_DOMAIN_BUILDERS = [
    build_session_coaching_domain,
    build_readiness_coaching_domain,
    build_zone_coaching_domain,
    build_pushback_coaching_domain,
    build_sport_specific_domain,
    build_load_management_domain,
    build_motivation_coaching_domain,
    build_engine_decision_domain,
    build_multi_turn_coaching_domain,
]


# ---------------------------------------------------------------------------
# Generator: scenarios → .conv files
# ---------------------------------------------------------------------------

def format_scenario_as_conv(scenario: Scenario, selector: KnowledgeSelector) -> str:
    """Format a single scenario as a .conv conversation block."""
    # Build user message with all blocks
    parts = [scenario.athlete_message]

    # CONTEXT block
    parts.append("")
    parts.append(scenario.context.to_block())

    # [KNOWLEDGE] block — select from real knowledge cards
    cards = selector.select(
        user_message=scenario.knowledge_query,
        action=scenario.interpreter_action,
        sport=scenario.knowledge_sport or None,
        max_cards=scenario.max_knowledge_cards,
    )
    if cards:
        knowledge_lines = ["[KNOWLEDGE]"]
        for card in cards:
            # Truncate long cards to keep training examples focused
            content = card["content"]
            if len(content) > 500:
                content = content[:500].rsplit(". ", 1)[0] + "."
            knowledge_lines.append(content)
        parts.append("")
        parts.append("\n\n".join(knowledge_lines))

    # [INTERPRETER] block
    interp_json = json.dumps(scenario.interpreter_json, ensure_ascii=False)
    parts.append("")
    parts.append(f"[INTERPRETER]\n{interp_json}")

    user_content = "\n".join(parts)

    # Build the conversation block
    lines = [
        "USER:",
        user_content,
        "",
        "JOSI:",
        scenario.josi_response,
    ]
    return "\n".join(lines)


def generate_domain_conv(domain: ConvDomain, selector: KnowledgeSelector) -> str:
    """Generate a complete .conv file for a domain."""
    header = f"# domain: {domain.name}\n"

    blocks = []
    for scenario in domain.scenarios:
        block = format_scenario_as_conv(scenario, selector)
        blocks.append(block)

    return header + "\n" + "\n\n---\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded coaching conversations from knowledge cards"
    )
    parser.add_argument("--domain", help="Generate only this domain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout instead of writing files")
    parser.add_argument("--output-dir", type=Path, default=CONVERSATIONS_DIR,
                        help=f"Output directory (default: {CONVERSATIONS_DIR})")
    args = parser.parse_args()

    # Load knowledge selector
    if not KNOWLEDGE_JSON.exists():
        print(f"ERROR: {KNOWLEDGE_JSON} not found")
        print(f"Run: python knowledge/scripts/export_knowledge_json.py")
        sys.exit(1)

    selector = KnowledgeSelector.from_json(str(KNOWLEDGE_JSON))
    print(f"Loaded {len(selector.entries)} knowledge entries")

    # Build domains
    domains = [builder() for builder in ALL_DOMAIN_BUILDERS]

    if args.domain:
        domains = [d for d in domains if d.name == args.domain]
        if not domains:
            print(f"ERROR: Unknown domain '{args.domain}'")
            print(f"Available: {', '.join(d.name for d in [b() for b in ALL_DOMAIN_BUILDERS])}")
            sys.exit(1)

    # Generate
    total_scenarios = 0
    for domain in domains:
        conv_text = generate_domain_conv(domain, selector)
        total_scenarios += len(domain.scenarios)

        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"  {domain.file_prefix}.conv ({len(domain.scenarios)} scenarios)")
            print(f"{'='*60}\n")
            print(conv_text)
        else:
            output_path = args.output_dir / f"{domain.file_prefix}.conv"
            output_path.write_text(conv_text, encoding="utf-8")
            print(f"  {output_path.name}: {len(domain.scenarios)} scenarios")

    print(f"\nGenerated {total_scenarios} grounded scenarios across {len(domains)} domains")

    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"  1. Review and polish JOSI responses")
        print(f"  2. Validate: python conv_format.py validate {args.output_dir}")
        print(f"  3. Prepare data: python prepare_v6_data.py --update-prompts --inject-knowledge")
        print(f"  4. Fine-tune: python finetune_qwen3.py train --mode unified")


if __name__ == "__main__":
    main()
