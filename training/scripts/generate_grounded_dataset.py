#!/usr/bin/env python3
"""
MiValta Josi — Knowledge-Grounded Training Data Generator

Unlike the generic generator, this script creates training examples that are
DEEPLY grounded in the 18 GATC knowledge cards. Every example traces back
to specific tables, thresholds, and rules in the source of truth.

Architecture:
    1. Parse all 18 knowledge cards → structured tables + rules + Josi text
    2. For each card axis, generate scenarios that exercise specific rules
    3. Claude generates paired examples (interpreter + coach) grounded in the rules
    4. Validate every example against the source of truth
    5. Output validated JSONL for finetune_qwen3.py

The key insight: we don't just use JOSI sections — we use the ALG tables
to create scenarios where the coach response MUST reflect specific thresholds,
rules, and constraints from the cards.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    # Preview examples from each card axis
    python generate_grounded_dataset.py --preview

    # Generate full grounded dataset (uses Claude API)
    python generate_grounded_dataset.py --run --count 2000

    # Generate for specific axis
    python generate_grounded_dataset.py --run --count 500 --axis physiology

    # Validate existing data against knowledge cards
    python generate_grounded_dataset.py --validate training/data/claude_generated_coach.jsonl

    # Dry run: show prompts without calling API
    python generate_grounded_dataset.py --dry-run --count 10
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
OUTPUT_DIR = DATA_DIR / "grounded"

# Import the card parser
sys.path.insert(0, str(SCRIPT_DIR))
from knowledge_card_parser import (
    load_all_cards,
    KnowledgeCard,
    Table,
    get_zone_physiology,
    get_energy_systems,
    get_zone_to_system_map,
    get_load_factors,
    get_readiness_gates,
    get_goal_archetypes,
    get_meso_dance_slices,
    get_feasibility_tiers,
    get_modifiers,
)


# ---------------------------------------------------------------------------
# Load system prompts
# ---------------------------------------------------------------------------
def load_system_prompt(mode: str) -> str:
    path = PROMPTS_DIR / f"josi_v6_{mode}.txt"
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"System prompt not found: {path}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPORTS = ["running", "cycling"]
SPORT_ENUM = {"running": "run", "cycling": "bike"}
LEVELS = ["beginner", "intermediate", "advanced", "elite"]
AGES = [22, 28, 35, 42, 50, 58, 65, 72]
READINESS_STATES = [
    ("green", "Recovered"),
    ("green", "Productive"),
    ("amber", "Accumulated"),
    ("amber", "Fatigued"),
    ("red", "Overreached"),
]
PHASES = ["base", "build", "peak", "taper"]
MESO_POSITIONS = ["ramp_in", "overload", "deload"]
LANGUAGES = ["english", "dutch"]

BANNED_JARGON = [
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "vo2 max", "VO2max",
    "lactate threshold", "ftp", "threshold power",
    "anaerobic capacity", "mitochondrial density",
    "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "gatc", "ewma", "tss", "ctl", "atl", "tsb",
]


# ---------------------------------------------------------------------------
# Grounded Scenario — each scenario traces to specific card rules
# ---------------------------------------------------------------------------
@dataclass
class GroundedScenario:
    """A training scenario grounded in specific knowledge card rules."""
    axis: str                          # physiology, load, session, periodization, etc.
    card_ids: list[str]                # which cards are exercised
    rule_id: str                       # specific rule/table being tested
    rule_description: str              # what the rule says (from the card)
    scenario_type: str                 # question, explanation, guardrail, grounding_test

    # Athlete context
    sport: str = "running"
    level: str = "intermediate"
    age: int = 35
    readiness: tuple[str, str] = ("green", "Productive")
    phase: str = "build"
    meso_position: str = "overload"
    language: str = "english"

    # Session context (optional)
    session_zone: str | None = None
    session_desc: str | None = None

    # Knowledge to inject
    knowledge_block: str = ""
    josi_guidance: str = ""

    # Expected behavior constraints
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    expected_action: str = "answer_question"

    def context_block(self) -> str:
        lines = [
            f"- Readiness: {self.readiness[0]} ({self.readiness[1]})",
            f"- Sport: {self.sport}",
            f"- Level: {self.level}",
            f"- Age: {self.age}",
            f"- Phase: {self.phase}",
            f"- Meso position: {self.meso_position}",
        ]
        if self.session_zone and self.session_desc:
            lines.insert(1, f"- Session: {self.session_zone} \"{self.session_desc}\"")
        return "\n".join(lines)

    def id(self) -> str:
        key = f"{self.axis}|{self.rule_id}|{self.sport}|{self.level}|{self.readiness}|{self.phase}|{self.language}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scenario Generators — one per axis
# ---------------------------------------------------------------------------

def _pick(rng: random.Random, lst: list) -> Any:
    return rng.choice(lst)


def generate_physiology_scenarios(cards: dict[str, KnowledgeCard], rng: random.Random, count: int) -> list[GroundedScenario]:
    """Generate scenarios from zone_physiology, energy_systems, zone_anchors."""
    scenarios = []
    zp_card = cards.get("zone_physiology")
    es_card = cards.get("energy_systems")
    za_card = cards.get("zone_anchors")

    if not zp_card or not es_card:
        return scenarios

    zones = ["R", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8"]

    for _ in range(count):
        zone = _pick(rng, zones)
        sport = _pick(rng, SPORTS)
        level = _pick(rng, LEVELS)
        language = "dutch" if rng.random() < 0.2 else "english"

        # Zone physiology question
        zone_data = None
        for table in zp_card.alg_tables.values():
            row = table.lookup("zone", zone)
            if row:
                zone_data = row
                break

        josi_text = zp_card.all_josi_text()

        # What must the coach mention?
        must_mention = [zone]
        if zone_data:
            if zone_data.get("primary_adaptation"):
                must_mention.append(zone_data["primary_adaptation"])

        # Energy system for this zone
        es_system = None
        zone_map = get_zone_to_system_map(cards)
        if zone_map:
            zrow = zone_map.lookup("zone", zone)
            if zrow:
                es_system = zrow.get("primary_system")

        knowledge = f"Zone {zone} physiology:\n"
        if zone_data:
            for k, v in zone_data.items():
                knowledge += f"  {k}: {v}\n"
        if es_system:
            knowledge += f"\nPrimary energy system: {es_system}\n"

        scenarios.append(GroundedScenario(
            axis="physiology",
            card_ids=["zone_physiology", "energy_systems"],
            rule_id=f"zone_{zone}_physiology",
            rule_description=f"Zone {zone} targets {zone_data.get('primary_adaptation', 'unknown') if zone_data else 'unknown'}",
            scenario_type="question",
            sport=sport,
            level=level,
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            language=language,
            knowledge_block=knowledge,
            josi_guidance=josi_text[:500],
            must_mention=must_mention,
            expected_action="answer_question",
        ))

    # Zone anchoring scenarios (grounding tests — coach must NOT invent numbers)
    if za_card:
        for _ in range(max(1, count // 5)):
            scenarios.append(GroundedScenario(
                axis="physiology",
                card_ids=["zone_anchors"],
                rule_id="zone_anchor_grounding",
                rule_description="Coach must NOT invent HR/pace/power numbers — direct to zone settings",
                scenario_type="grounding_test",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block="Zone anchoring is personal. Zones are calibrated from athlete's threshold test. Never prescribe specific HR/pace/power values.",
                josi_guidance=za_card.all_josi_text()[:500],
                must_not_mention=["bpm", "watts", "W ", "/km", "/mi"],
                expected_action="answer_question",
            ))

    return scenarios[:count]


def generate_load_scenarios(cards: dict[str, KnowledgeCard], rng: random.Random, count: int) -> list[GroundedScenario]:
    """Generate scenarios from load_monitoring, training_load_model, monotony_policy, meso_dance_policy."""
    scenarios = []

    lm_card = cards.get("load_monitoring")
    tlm_card = cards.get("training_load_model")
    mono_card = cards.get("monotony_policy")
    meso_card = cards.get("meso_dance_policy")

    # Readiness explanation scenarios
    if lm_card:
        readiness_josi = lm_card.all_josi_text()
        for _ in range(count // 3):
            readiness = _pick(rng, READINESS_STATES)
            color = readiness[0]

            # What zones are allowed at this readiness?
            gates = get_readiness_gates(cards)
            allowed_info = ""
            if gates:
                for row in gates.rows:
                    if row.get("readiness_state") == color:
                        block = row.get("zone_block", "")
                        allowed = row.get("allowed", "")
                        allowed_info += f"  {block}: {'allowed' if allowed == 'T' else 'blocked'}\n"

            knowledge = f"Readiness: {color} ({readiness[1]})\n"
            knowledge += f"Meaning: athlete is in {readiness[1]} state\n"
            if allowed_info:
                knowledge += f"Zone permissions:\n{allowed_info}"

            must_mention = [color]
            if color == "red":
                must_mention.extend(["rest", "recover"])

            scenarios.append(GroundedScenario(
                axis="load",
                card_ids=["load_monitoring"],
                rule_id=f"readiness_{color}_{readiness[1]}",
                rule_description=f"Readiness {color} means {readiness[1]} — explain what zones are available",
                scenario_type="explanation",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=readiness,
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=readiness_josi[:500],
                must_mention=must_mention,
                expected_action="explain",
            ))

    # MesoDance scenarios
    if meso_card:
        meso_josi = meso_card.all_josi_text()
        for _ in range(count // 3):
            position = _pick(rng, MESO_POSITIONS)

            knowledge = f"Meso position: {position}\n"
            if position == "ramp_in":
                knowledge += "Building load gradually — foundation before intensity.\n"
                knowledge += "Intensity capped, volume building progressively.\n"
            elif position == "overload":
                knowledge += "Peak training stress — expect to feel challenged.\n"
                knowledge += "Highest volume and intensity of the block.\n"
            elif position == "deload":
                knowledge += "Reduced load to allow adaptations from earlier sessions.\n"
                knowledge += "Volume drops 40-60%, intensity maintained at ~80%.\n"

            scenarios.append(GroundedScenario(
                axis="load",
                card_ids=["meso_dance_policy"],
                rule_id=f"meso_position_{position}",
                rule_description=f"MesoDance position: {position}",
                scenario_type="explanation",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=position,
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=meso_josi[:500],
                must_mention=[position.replace("_", " ")],
                expected_action="explain",
            ))

    # Monotony scenarios
    if mono_card:
        for _ in range(count // 6):
            scenarios.append(GroundedScenario(
                axis="load",
                card_ids=["monotony_policy"],
                rule_id="monotony_variety",
                rule_description="Training variety prevents repetitive stress — variety is added when monotony index is high",
                scenario_type="explanation",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block="Variety prevents repetitive stress. Different session types stimulate different adaptations.",
                josi_guidance=mono_card.all_josi_text()[:500],
                expected_action="explain",
            ))

    # Training load model scenarios
    if tlm_card:
        for _ in range(count // 6):
            scenarios.append(GroundedScenario(
                axis="load",
                card_ids=["training_load_model"],
                rule_id="load_monitoring_thresholds",
                rule_description="Load model tracks acute vs chronic training load to prevent overtraining",
                scenario_type="question",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block="The system tracks how much you've trained recently vs your longer-term average. This helps prevent doing too much too quickly.",
                josi_guidance=tlm_card.all_josi_text()[:500],
                must_not_mention=["ACWR", "EWMA", "CTL", "ATL", "TSB"],
                expected_action="answer_question",
            ))

    return scenarios[:count]


def generate_session_scenarios(cards: dict[str, KnowledgeCard], rng: random.Random, count: int) -> list[GroundedScenario]:
    """Generate scenarios from session_rules, session_variety_policy."""
    scenarios = []
    sr_card = cards.get("session_rules")
    sv_card = cards.get("session_variety_policy")

    SESSIONS = {
        "R": ["Recovery spin 20min", "Easy walk 30min"],
        "Z1": ["Continuous Z1 45min", "Continuous Z1 60min"],
        "Z2": ["Continuous Z2 60min", "Continuous Z2 45min", "Continuous Z2 75min", "Long Z2 90min"],
        "Z3": ["2x15min Z3 / 5min Z1", "3x10min Z3 / 3min Z1"],
        "Z4": ["4x5min Z4 / 3min Z1", "3x8min Z4 / 4min Z1"],
        "Z5": ["5x3min Z5 / 3min Z1", "4x4min Z5 / 4min Z1"],
        "Z6": ["6x30s Z6 / 90s Z1", "8x20s Z6 / 60s Z1"],
    }

    # Session explanation scenarios (why this session today?)
    if sr_card:
        for _ in range(count // 2):
            zone = _pick(rng, list(SESSIONS.keys()))
            session = _pick(rng, SESSIONS[zone])
            readiness = _pick(rng, READINESS_STATES)

            # Check if zone is allowed at this readiness
            color = readiness[0]
            zone_cap = {"green": 8, "amber": 5, "red": 2}
            zone_num = int(zone[1]) if zone[1:].isdigit() else 0
            zone_allowed = zone_num <= zone_cap.get(color, 0)

            load_factors = get_load_factors(cards)
            lf = load_factors.get(zone, 1.0)

            knowledge = f"Session: {zone} \"{session}\"\n"
            knowledge += f"Zone load factor: {lf} (Z2 = 1.0 baseline)\n"
            knowledge += f"Readiness: {color} — {'zone allowed' if zone_allowed else 'zone may be capped'}\n"

            phase = _pick(rng, PHASES)
            position = _pick(rng, MESO_POSITIONS)
            knowledge += f"Phase: {phase}, Position: {position}\n"

            scenarios.append(GroundedScenario(
                axis="session",
                card_ids=["session_rules", "energy_systems"],
                rule_id=f"session_explain_{zone}",
                rule_description=f"Session {zone} with load factor {lf} during {phase}/{position}",
                scenario_type="explanation",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=readiness,
                phase=phase,
                meso_position=position,
                session_zone=zone,
                session_desc=session,
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=sr_card.all_josi_text()[:500] if sr_card else "",
                must_mention=[zone],
                expected_action="explain",
            ))

    # Session variety scenarios
    if sv_card:
        for _ in range(count // 4):
            scenarios.append(GroundedScenario(
                axis="session",
                card_ids=["session_variety_policy"],
                rule_id="session_variety",
                rule_description="Quality sessions don't repeat within a slice — variety prevents staleness",
                scenario_type="explanation",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block="Session variety: quality workouts don't repeat in the same training block. This keeps your body adapting.",
                josi_guidance=sv_card.all_josi_text()[:500],
                expected_action="explain",
            ))

    # Replan / skip scenarios
    for _ in range(count // 4):
        zone = _pick(rng, list(SESSIONS.keys()))
        session = _pick(rng, SESSIONS[zone])
        replan_type = _pick(rng, ["skip_today", "reduce_intensity", "illness", "reschedule"])

        scenarios.append(GroundedScenario(
            axis="session",
            card_ids=["session_rules", "load_monitoring"],
            rule_id=f"replan_{replan_type}",
            rule_description=f"Athlete wants to {replan_type} — handle with empathy and safety",
            scenario_type="replan",
            sport=_pick(rng, SPORTS),
            level=_pick(rng, LEVELS),
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            session_zone=zone,
            session_desc=session,
            language="dutch" if rng.random() < 0.2 else "english",
            knowledge_block=f"Replan type: {replan_type}. The engine will adjust the plan accordingly.",
            expected_action="replan" if replan_type != "illness" else "replan",
        ))

    return scenarios[:count]


def generate_periodization_scenarios(cards: dict[str, KnowledgeCard], rng: random.Random, count: int) -> list[GroundedScenario]:
    """Generate scenarios from periodization, goal_demands, feasibility_policy, pack_composition."""
    scenarios = []

    per_card = cards.get("periodization")
    gd_card = cards.get("goal_demands")
    fp_card = cards.get("feasibility_policy")

    # Phase explanation scenarios
    if per_card:
        for _ in range(count // 3):
            phase = _pick(rng, PHASES)
            knowledge = f"Training phase: {phase}\n"
            if phase == "base":
                knowledge += "Focus on aerobic foundation. 80%+ of training in Z1-Z2. Build volume gradually."
            elif phase == "build":
                knowledge += "Adding intensity. More Z3-Z4 work. Volume stabilizes or increases slightly."
            elif phase == "peak":
                knowledge += "Race-specific intensity. Z4-Z5 emphasis. Volume may decrease slightly."
            elif phase == "taper":
                knowledge += "Volume drops 40-60%. Intensity maintained. Body absorbs fitness gains."

            scenarios.append(GroundedScenario(
                axis="periodization",
                card_ids=["periodization"],
                rule_id=f"phase_{phase}",
                rule_description=f"Training phase: {phase}",
                scenario_type="question",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=phase,
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=per_card.all_josi_text()[:500],
                expected_action="answer_question",
            ))

    # Goal demands scenarios
    if gd_card:
        goal_types = ["marathon", "half_marathon", "10k", "5k", "century", "gran_fondo", "triathlon"]
        for _ in range(count // 3):
            goal = _pick(rng, goal_types)
            knowledge = f"Goal: {goal}\n"
            knowledge += "Each goal has specific physiological demands and minimum preparation time.\n"
            knowledge += "The engine calculates achievability based on your current fitness, experience, and available time."

            scenarios.append(GroundedScenario(
                axis="periodization",
                card_ids=["goal_demands"],
                rule_id=f"goal_{goal}",
                rule_description=f"Goal demands for {goal}",
                scenario_type="question",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=gd_card.all_josi_text()[:500],
                expected_action="answer_question",
            ))

    # Feasibility tier scenarios
    if fp_card:
        tiers = [("A", "Ideal — all conditions met"), ("B", "Good — minor adjustments"),
                 ("C", "Adapted — modified for constraints"), ("D", "Minimal — heavily constrained")]
        for _ in range(count // 3):
            tier, desc = _pick(rng, tiers)
            scenarios.append(GroundedScenario(
                axis="periodization",
                card_ids=["feasibility_policy"],
                rule_id=f"feasibility_tier_{tier}",
                rule_description=f"Feasibility tier {tier}: {desc}",
                scenario_type="explanation",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=f"Feasibility: Tier {tier} — {desc}\nLower tiers aren't failures. They're intelligent adaptations to your constraints.",
                josi_guidance=fp_card.all_josi_text()[:500],
                expected_action="explain",
            ))

    return scenarios[:count]


def generate_modifier_scenarios(cards: dict[str, KnowledgeCard], rng: random.Random, count: int) -> list[GroundedScenario]:
    """Generate scenarios from modifiers, modifiers_running, modifiers_cycling."""
    scenarios = []

    mod_card = cards.get("modifiers")
    run_card = cards.get("modifiers_running")
    cyc_card = cards.get("modifiers_cycling")

    # Age modifier scenarios
    if mod_card:
        age_groups = [(22, "young"), (42, "masters_40"), (55, "masters_50"), (65, "senior"), (75, "elderly")]
        for _ in range(count // 3):
            age, group = _pick(rng, age_groups)
            knowledge = f"Age: {age} ({group})\n"
            knowledge += "Age affects recovery time, not training capability.\n"
            knowledge += "Older athletes need more recovery time between hard sessions.\n"
            knowledge += "Warmup and rest periods increase with age."

            scenarios.append(GroundedScenario(
                axis="modifiers",
                card_ids=["modifiers"],
                rule_id=f"age_modifier_{group}",
                rule_description=f"Age {age} ({group}): recovery mult increases, warmup increases",
                scenario_type="question",
                sport=_pick(rng, SPORTS),
                level=_pick(rng, LEVELS),
                age=age,
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=mod_card.all_josi_text()[:500],
                expected_action="answer_question",
            ))

    # Level modifier scenarios
    if mod_card:
        for _ in range(count // 3):
            level = _pick(rng, LEVELS)
            knowledge = f"Level: {level}\n"
            if level == "beginner":
                knowledge += "Beginners: zones unlock progressively over mesos. Max Z4 initially. More recovery needed."
            elif level == "intermediate":
                knowledge += "Intermediate: all zones available after first meso. Standard recovery."
            elif level == "advanced":
                knowledge += "Advanced: full zone access. Can handle higher training load."
            elif level == "elite":
                knowledge += "Elite: maximum training tolerance. Full zone access from day 1."

            scenarios.append(GroundedScenario(
                axis="modifiers",
                card_ids=["modifiers"],
                rule_id=f"level_modifier_{level}",
                rule_description=f"Level: {level} — zone access and recovery adjustments",
                scenario_type="question",
                sport=_pick(rng, SPORTS),
                level=level,
                age=_pick(rng, AGES),
                readiness=_pick(rng, READINESS_STATES),
                phase=_pick(rng, PHASES),
                meso_position=_pick(rng, MESO_POSITIONS),
                language="dutch" if rng.random() < 0.2 else "english",
                knowledge_block=knowledge,
                josi_guidance=mod_card.all_josi_text()[:500],
                expected_action="answer_question",
            ))

    # Sport-specific scenarios
    for _ in range(count // 3):
        sport = _pick(rng, SPORTS)
        card = run_card if sport == "running" else cyc_card
        if not card:
            continue

        knowledge = f"Sport: {sport}\n"
        if sport == "running":
            knowledge += "Running is high-impact. Recovery multiplier: 1.10x. Gradual volume increase essential."
        else:
            knowledge += "Cycling is low-impact. Recovery multiplier: 1.00x. Higher volume tolerance."

        scenarios.append(GroundedScenario(
            axis="modifiers",
            card_ids=[f"modifiers_{sport}"],
            rule_id=f"sport_modifier_{sport}",
            rule_description=f"Sport-specific: {sport}",
            scenario_type="question",
            sport=sport,
            level=_pick(rng, LEVELS),
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            language="dutch" if rng.random() < 0.2 else "english",
            knowledge_block=knowledge,
            josi_guidance=card.all_josi_text()[:500],
            expected_action="answer_question",
        ))

    return scenarios[:count]


def generate_expression_scenarios(cards: dict[str, KnowledgeCard], rng: random.Random, count: int) -> list[GroundedScenario]:
    """Generate scenarios from josi_explanations — coaching voice and guardrails."""
    scenarios = []
    je_card = cards.get("josi_explanations")
    if not je_card:
        return scenarios

    # Stimulus vs cost ratio scenarios
    ratio_bands = [
        ("efficient", "cost/stimulus < 0.85", "Strong adaptive signal with limited fatigue"),
        ("balanced", "0.85 <= cost/stimulus <= 1.15", "Stimulus matches the cost"),
        ("taxing", "1.15 < cost/stimulus <= 1.40", "Costs more than it builds — intentional for durability"),
        ("heavy", "cost/stimulus > 1.40", "Significant fatigue investment for long-term gains"),
    ]
    for _ in range(count // 4):
        band, cond, explanation = _pick(rng, ratio_bands)
        zone = _pick(rng, ["Z2", "Z3", "Z4", "Z5"])
        session = f"Continuous {zone} 60min" if zone in ["Z2", "Z3"] else f"4x5min {zone} / 3min Z1"

        scenarios.append(GroundedScenario(
            axis="expression",
            card_ids=["josi_explanations"],
            rule_id=f"stim_cost_{band}",
            rule_description=f"Stimulus/cost ratio: {band} — {explanation}",
            scenario_type="explanation",
            sport=_pick(rng, SPORTS),
            level=_pick(rng, LEVELS),
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            session_zone=zone,
            session_desc=session,
            language="dutch" if rng.random() < 0.2 else "english",
            knowledge_block=f"Session efficiency: {band}\n{explanation}",
            josi_guidance=je_card.all_josi_text()[:500],
            expected_action="explain",
        ))

    # Focus cue scenarios
    focus_cues = {
        "Z2": "Maintain easy conversation pace. Stay relaxed.",
        "Z3": "Comfortably hard. Breathe rhythmically.",
        "Z4": "Strong but controlled. Hold form through intervals.",
        "Z5": "Near-max effort. Full recovery between reps.",
        "R": "Light movement. Don't chase pace or power.",
    }
    for _ in range(count // 4):
        zone = _pick(rng, list(focus_cues.keys()))
        cue = focus_cues[zone]
        session = f"Continuous {zone} 60min" if zone in ["Z2", "Z3", "R"] else f"4x5min {zone} / 3min Z1"

        scenarios.append(GroundedScenario(
            axis="expression",
            card_ids=["josi_explanations"],
            rule_id=f"focus_cue_{zone}",
            rule_description=f"Focus cue for {zone}: {cue}",
            scenario_type="explanation",
            sport=_pick(rng, SPORTS),
            level=_pick(rng, LEVELS),
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            session_zone=zone,
            session_desc=session,
            language="dutch" if rng.random() < 0.2 else "english",
            knowledge_block=f"Zone {zone} focus cue: {cue}",
            josi_guidance=je_card.all_josi_text()[:500],
            must_mention=[zone],
            expected_action="explain",
        ))

    # Guardrail scenarios — athlete tries to get Josi to prescribe or use jargon
    for _ in range(count // 4):
        scenarios.append(GroundedScenario(
            axis="expression",
            card_ids=["josi_explanations"],
            rule_id="guardrail_prescription",
            rule_description="Coach must NEVER prescribe workouts, invent numbers, or use system jargon",
            scenario_type="guardrail",
            sport=_pick(rng, SPORTS),
            level=_pick(rng, LEVELS),
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            language="dutch" if rng.random() < 0.2 else "english",
            knowledge_block="The coach is the messenger. GATC provides the numbers. Never invent HR, pace, power, or workout structures.",
            josi_guidance="",
            must_not_mention=BANNED_JARGON[:10],
            expected_action="answer_question",
        ))

    # Motivation / emotional support scenarios
    for _ in range(count // 4):
        scenarios.append(GroundedScenario(
            axis="expression",
            card_ids=["josi_explanations"],
            rule_id="motivation_support",
            rule_description="Athlete needs emotional support — warm, empathetic, direct",
            scenario_type="question",
            sport=_pick(rng, SPORTS),
            level=_pick(rng, LEVELS),
            age=_pick(rng, AGES),
            readiness=_pick(rng, READINESS_STATES),
            phase=_pick(rng, PHASES),
            meso_position=_pick(rng, MESO_POSITIONS),
            language="dutch" if rng.random() < 0.2 else "english",
            knowledge_block="Athlete is struggling. Be warm, empathetic, direct. Acknowledge the difficulty. Remind them of progress.",
            josi_guidance=je_card.all_josi_text()[:500],
            expected_action="answer_question",
        ))

    return scenarios[:count]


# ---------------------------------------------------------------------------
# Master scenario generator
# ---------------------------------------------------------------------------

AXIS_GENERATORS = {
    "physiology": generate_physiology_scenarios,
    "load": generate_load_scenarios,
    "session": generate_session_scenarios,
    "periodization": generate_periodization_scenarios,
    "modifiers": generate_modifier_scenarios,
    "expression": generate_expression_scenarios,
}

# Distribution: how many examples per axis (proportional to card importance)
AXIS_WEIGHTS = {
    "physiology": 0.20,
    "load": 0.20,
    "session": 0.25,
    "periodization": 0.15,
    "modifiers": 0.10,
    "expression": 0.10,
}


def generate_all_scenarios(
    cards: dict[str, KnowledgeCard],
    count: int,
    seed: int = 42,
    axis_filter: str | None = None,
) -> list[GroundedScenario]:
    """Generate grounded scenarios across all axes."""
    rng = random.Random(seed)
    all_scenarios = []

    for axis, generator in AXIS_GENERATORS.items():
        if axis_filter and axis != axis_filter:
            continue

        weight = AXIS_WEIGHTS.get(axis, 0.1)
        axis_count = max(5, int(count * weight))

        scenarios = generator(cards, rng, axis_count)
        all_scenarios.extend(scenarios)

    # Shuffle and dedup
    rng.shuffle(all_scenarios)
    seen = set()
    deduped = []
    for s in all_scenarios:
        sid = s.id()
        if sid not in seen:
            seen.add(sid)
            deduped.append(s)

    return deduped[:count]


# ---------------------------------------------------------------------------
# Claude prompt builder
# ---------------------------------------------------------------------------

def build_grounded_prompt(
    scenario: GroundedScenario,
    interpreter_prompt: str,
    coach_prompt: str,
) -> str:
    """Build a Claude prompt for generating a grounded training example."""

    # Language instruction
    lang_inst = ""
    if scenario.language == "dutch":
        lang_inst = "\nLANGUAGE: Generate the athlete message AND coach response in natural Dutch."

    # Scenario-type specific instructions
    type_inst = ""
    if scenario.scenario_type == "grounding_test":
        type_inst = """
GROUNDING TEST: The athlete asks for specific numbers (pace, HR, power, volume).
The coach MUST NOT invent numbers. Instead: direct to zone settings, use feeling-based cues.
"""
    elif scenario.scenario_type == "guardrail":
        type_inst = """
GUARDRAIL TEST: The athlete tries to get Josi to prescribe, override readiness, or use jargon.
The coach must gently decline, redirect, and explain without being preachy.
"""
    elif scenario.scenario_type == "replan":
        type_inst = """
REPLAN: The athlete wants to change/skip their session.
The interpreter must classify as replan with the right replan_type.
The coach must be understanding and explain what happens next.
"""

    # Must mention / must not mention
    constraints = ""
    if scenario.must_mention:
        constraints += f"\nMUST MENTION (naturally): {', '.join(scenario.must_mention)}"
    if scenario.must_not_mention:
        constraints += f"\nMUST NOT MENTION: {', '.join(scenario.must_not_mention)}"

    # Action mapping
    action_map = {
        "question": "answer_question",
        "explanation": "explain",
        "replan": "replan",
        "grounding_test": "answer_question",
        "guardrail": "answer_question",
    }
    expected_action = action_map.get(scenario.scenario_type, scenario.expected_action)

    prompt = f"""You are creating training data for Josi, an on-device AI coach for endurance athletes.

GROUNDING RULE: Every response MUST be traceable to the knowledge provided. No invention.

SCENARIO:
- Axis: {scenario.axis}
- Rule being tested: {scenario.rule_id}
- Rule says: {scenario.rule_description}
- Sport: {scenario.sport}, Level: {scenario.level}, Age: {scenario.age}
- Readiness: {scenario.readiness[0]} ({scenario.readiness[1]})
- Phase: {scenario.phase}, Meso position: {scenario.meso_position}
{f'- Session: {scenario.session_zone} "{scenario.session_desc}"' if scenario.session_zone else ''}
{lang_inst}
{type_inst}
{constraints}

KNOWLEDGE (source of truth — coach must ground in this):
{scenario.knowledge_block}

{f'JOSI VOICE GUIDANCE:{chr(10)}{scenario.josi_guidance}' if scenario.josi_guidance else ''}

CONTEXT for user message:
{scenario.context_block()}

EXPECTED INTERPRETER ACTION: {expected_action}

COACH SYSTEM PROMPT:
{coach_prompt}

---

Generate ONE training example. Output EXACTLY this JSON:

{{
  "athlete_message": "natural athlete message",
  "interpreter_json": {{valid GATCRequest JSON with action="{expected_action}"}},
  "coach_response": "warm coaching response grounded in KNOWLEDGE (60-150 words)",
  "knowledge_used": "which rule/table was applied",
  "grounding_check": "how the response traces to the knowledge"
}}

QUALITY:
1. Athlete message: natural, varied, sometimes casual/emotional
2. Interpreter JSON: valid GATCRequest, free_text matches athlete message
3. Coach response: grounded in knowledge, warm, under 150 words, no jargon
4. NEVER invent HR/pace/power/workout structures
5. NEVER use: {', '.join(BANNED_JARGON[:8])}...
6. End with statement/encouragement, not a question

Output ONLY the JSON. No markdown fences."""

    return prompt


# ---------------------------------------------------------------------------
# Validation against knowledge cards
# ---------------------------------------------------------------------------

def validate_grounded_example(
    example: dict,
    scenario: GroundedScenario,
) -> list[str]:
    """Validate a generated example against its grounding scenario."""
    issues = []
    coach = example.get("coach_response", "")

    # Check must_mention
    for term in scenario.must_mention:
        if term.lower() not in coach.lower():
            issues.append(f"Missing required mention: '{term}'")

    # Check must_not_mention
    for term in scenario.must_not_mention:
        if term.lower() in coach.lower():
            issues.append(f"Forbidden mention found: '{term}'")

    # Check banned jargon
    for term in BANNED_JARGON:
        if term.lower() in coach.lower():
            issues.append(f"Jargon: '{term}'")

    # Length check
    words = len(coach.split())
    if words < 25:
        issues.append(f"Too short: {words} words")
    if words > 250:
        issues.append(f"Too long: {words} words")

    # System reference check
    system_refs = ["knowledge card", "knowledge block", "interpreter", "gatc", "viterbi",
                   "the system says", "the algorithm", "my programming"]
    for ref in system_refs:
        if ref in coach.lower():
            issues.append(f"System leak: '{ref}'")

    # Grounding discipline: no invented numbers
    hr_pat = re.compile(r'\b\d{2,3}\s*(?:bpm|beats)', re.IGNORECASE)
    pace_pat = re.compile(r'\b\d:\d{2}\s*/\s*(?:km|mi)', re.IGNORECASE)
    power_pat = re.compile(r'\b\d{2,3}\s*(?:watts?|W)\b', re.IGNORECASE)

    if hr_pat.search(coach):
        issues.append("Grounding: invented HR value")
    if pace_pat.search(coach):
        issues.append("Grounding: invented pace value")
    if power_pat.search(coach):
        issues.append("Grounding: invented power value")

    # Validate interpreter JSON
    interp = example.get("interpreter_json", {})
    if "action" not in interp:
        issues.append("Interpreter: missing action")
    if "free_text" not in interp:
        issues.append("Interpreter: missing free_text")

    return issues


# ---------------------------------------------------------------------------
# Build training format (ChatML JSONL)
# ---------------------------------------------------------------------------

def build_training_examples(
    example: dict,
    scenario: GroundedScenario,
    interpreter_prompt: str,
    coach_prompt: str,
) -> tuple[dict, dict | None]:
    """Convert a grounded example to interpreter + coach training format."""

    athlete_msg = example["athlete_message"]
    interp_json = example["interpreter_json"]
    coach_response = example.get("coach_response", "")

    # Build user content with context
    user_with_context = athlete_msg + f"\n\nCONTEXT:\n{scenario.context_block()}"

    # Interpreter example
    interpreter_example = {
        "messages": [
            {"role": "system", "content": interpreter_prompt},
            {"role": "user", "content": user_with_context},
            {"role": "assistant", "content": json.dumps(interp_json, ensure_ascii=False)},
        ],
        "metadata": {
            "axis": scenario.axis,
            "rule_id": scenario.rule_id,
            "card_ids": scenario.card_ids,
        },
    }

    # Coach example (if response present)
    coach_example = None
    if coach_response:
        coach_user = user_with_context
        if scenario.knowledge_block:
            coach_user += f"\n\n[KNOWLEDGE]\n{scenario.knowledge_block}"
        coach_user += f"\n\n[INTERPRETER]\n{json.dumps(interp_json, ensure_ascii=False)}"

        coach_example = {
            "messages": [
                {"role": "system", "content": coach_prompt},
                {"role": "user", "content": coach_user},
                {"role": "assistant", "content": coach_response},
            ],
            "metadata": {
                "axis": scenario.axis,
                "rule_id": scenario.rule_id,
                "card_ids": scenario.card_ids,
                "grounding_check": example.get("grounding_check", ""),
            },
        }

    return interpreter_example, coach_example


# ---------------------------------------------------------------------------
# Claude API interaction
# ---------------------------------------------------------------------------

async def generate_one(
    client,
    scenario: GroundedScenario,
    interpreter_prompt: str,
    coach_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 2,
) -> tuple[dict | None, list[str]]:
    """Generate one grounded training example using Claude."""

    prompt = build_grounded_prompt(scenario, interpreter_prompt, coach_prompt)

    for attempt in range(max_retries + 1):
        try:
            message = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=1200,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                if attempt < max_retries:
                    continue
                return None, ["Could not parse JSON"]

            result = json.loads(json_match.group())

            # Validate
            required = ["athlete_message", "interpreter_json"]
            missing = [f for f in required if f not in result]
            if missing:
                if attempt < max_retries:
                    continue
                return None, [f"Missing: {missing}"]

            # Fix free_text
            if "interpreter_json" in result:
                result["interpreter_json"]["free_text"] = result["athlete_message"]

            # Validate grounding
            issues = validate_grounded_example(result, scenario)

            # Retry on jargon
            if any("Jargon" in i for i in issues) and attempt < max_retries:
                continue

            return result, issues

        except json.JSONDecodeError:
            if attempt < max_retries:
                continue
            return None, ["JSON decode error"]
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            return None, [f"API error: {e}"]

    return None, ["Max retries exceeded"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_generation(args):
    """Main generation loop."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load knowledge cards
    cards = load_all_cards()
    print(f"Loaded {len(cards)} knowledge cards")

    # Load system prompts
    interpreter_prompt = load_system_prompt("interpreter")
    coach_prompt = load_system_prompt("coach")

    # Generate scenarios
    scenarios = generate_all_scenarios(
        cards, args.count, seed=args.seed, axis_filter=args.axis
    )
    print(f"Generated {len(scenarios)} grounded scenarios")

    # Axis distribution
    axis_counts = {}
    for s in scenarios:
        axis_counts[s.axis] = axis_counts.get(s.axis, 0) + 1
    for axis, count in sorted(axis_counts.items()):
        print(f"  {axis}: {count}")

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_interpreter = OUTPUT_DIR / f"grounded_interpreter_{timestamp}.jsonl"
    out_coach = OUTPUT_DIR / f"grounded_coach_{timestamp}.jsonl"
    out_combined = OUTPUT_DIR / f"grounded_combined_{timestamp}.jsonl"
    out_meta = OUTPUT_DIR / f"grounded_meta_{timestamp}.json"

    # Stats
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "with_issues": 0,
        "issues_breakdown": {},
        "axis_counts": {},
    }

    # Semaphore for concurrent API calls
    sem = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0}

    async def process_one(scenario):
        async with sem:
            result, issues = await generate_one(
                client, scenario, interpreter_prompt, coach_prompt,
                model=args.model,
            )
            progress["done"] += 1
            if progress["done"] % 25 == 0:
                print(f"  Progress: {progress['done']}/{len(scenarios)}")
            return scenario, result, issues

    # Run all
    tasks = [process_one(s) for s in scenarios]
    results = await asyncio.gather(*tasks)

    interpreter_examples = []
    coach_examples = []

    for scenario, result, issues in results:
        stats["total"] += 1

        if result is None:
            stats["failed"] += 1
            continue

        if issues:
            stats["with_issues"] += 1
            # Skip examples with hard errors
            hard_errors = [i for i in issues if "Jargon" in i or "System leak" in i or "invented" in i.lower()]
            if hard_errors:
                stats["failed"] += 1
                for i in hard_errors:
                    stats["issues_breakdown"][i] = stats["issues_breakdown"].get(i, 0) + 1
                continue

        stats["success"] += 1
        stats["axis_counts"][scenario.axis] = stats["axis_counts"].get(scenario.axis, 0) + 1

        interp_ex, coach_ex = build_training_examples(
            result, scenario, interpreter_prompt, coach_prompt
        )
        interpreter_examples.append(interp_ex)
        if coach_ex:
            coach_examples.append(coach_ex)

        for i in issues:
            stats["issues_breakdown"][i] = stats["issues_breakdown"].get(i, 0) + 1

    # Write output
    with open(out_interpreter, "w") as f:
        for ex in interpreter_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_coach, "w") as f:
        for ex in coach_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_combined, "w") as f:
        for ex in interpreter_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in coach_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_meta, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  GROUNDED DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total scenarios: {stats['total']}")
    print(f"  Successful: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  With warnings: {stats['with_issues']}")
    print(f"\n  Interpreter examples: {len(interpreter_examples)}")
    print(f"  Coach examples: {len(coach_examples)}")
    print(f"\n  Output:")
    print(f"    {out_interpreter}")
    print(f"    {out_coach}")
    print(f"    {out_combined}")
    print(f"\n  By axis:")
    for axis, count in sorted(stats["axis_counts"].items()):
        print(f"    {axis}: {count}")

    if stats["issues_breakdown"]:
        print(f"\n  Issues:")
        for issue, count in sorted(stats["issues_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {issue}: {count}")


def preview(args):
    """Preview scenarios without calling the API."""
    cards = load_all_cards()
    scenarios = generate_all_scenarios(cards, args.count or 30, axis_filter=args.axis)

    print(f"Preview: {len(scenarios)} grounded scenarios\n")

    for i, s in enumerate(scenarios[:args.count or 30]):
        print(f"--- Scenario {i+1} ---")
        print(f"  Axis: {s.axis}")
        print(f"  Cards: {s.card_ids}")
        print(f"  Rule: {s.rule_id}")
        print(f"  Rule says: {s.rule_description}")
        print(f"  Type: {s.scenario_type}")
        print(f"  Context: {s.sport}, {s.level}, age={s.age}, {s.readiness[0]}, {s.phase}/{s.meso_position}")
        if s.session_zone:
            print(f"  Session: {s.session_zone} \"{s.session_desc}\"")
        print(f"  Language: {s.language}")
        if s.must_mention:
            print(f"  Must mention: {s.must_mention}")
        if s.must_not_mention:
            print(f"  Must NOT mention: {s.must_not_mention}")
        print(f"  Knowledge: {s.knowledge_block[:100]}...")
        print()


def dry_run(args):
    """Show prompts without calling the API."""
    cards = load_all_cards()
    interpreter_prompt = load_system_prompt("interpreter")
    coach_prompt = load_system_prompt("coach")

    scenarios = generate_all_scenarios(cards, args.count or 5, axis_filter=args.axis)

    for i, s in enumerate(scenarios[:args.count or 5]):
        prompt = build_grounded_prompt(s, interpreter_prompt, coach_prompt)
        print(f"\n{'='*60}")
        print(f"SCENARIO {i+1}: {s.axis} / {s.rule_id}")
        print(f"{'='*60}")
        print(prompt[:2000])
        print("...")


def main():
    parser = argparse.ArgumentParser(
        description="Generate knowledge-grounded training data for Josi"
    )
    parser.add_argument("--preview", action="store_true",
                        help="Preview scenarios without API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without API calls")
    parser.add_argument("--run", action="store_true",
                        help="Run generation with Claude API")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of examples to generate")
    parser.add_argument("--axis", type=str, default=None,
                        choices=list(AXIS_GENERATORS.keys()),
                        help="Generate for specific axis only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model for generation")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent API calls")

    args = parser.parse_args()

    if args.preview:
        preview(args)
    elif args.dry_run:
        dry_run(args)
    elif args.run:
        asyncio.run(run_generation(args))
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python generate_grounded_dataset.py --preview")
        print("  python generate_grounded_dataset.py --dry-run --count 5")
        print("  python generate_grounded_dataset.py --run --count 2000")


if __name__ == "__main__":
    main()
