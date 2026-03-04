#!/usr/bin/env python3
"""
MiValta Josi — Claude-Powered Training Data Generator (Knowledge Distillation)

Uses Claude (Anthropic API) as an expert teacher to generate high-quality
training data for the on-device Josi LLM. Claude deeply understands the
science-based knowledge cards and generates diverse, grounded examples
that the local model (Qwen3-8B) can learn from.

Architecture:
    Claude reads 18 knowledge cards → generates paired training examples
    → interpreter (GATCRequest JSON) + coach (warm text) → JSONL output
    → finetune_qwen3.py trains the local model

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    # Preview 5 examples (dry run, no file output)
    python generate_claude_dataset.py --preview 5

    # Generate 2000 new examples (1000 interpreter + 1000 coach)
    python generate_claude_dataset.py --run --count 1000

    # Generate coach-only examples (e.g., to augment weak areas)
    python generate_claude_dataset.py --run --count 500 --focus coach

    # Generate with higher Dutch ratio
    python generate_claude_dataset.py --run --count 1000 --dutch-ratio 0.25

    # Resume from checkpoint if interrupted
    python generate_claude_dataset.py --run --count 1000 --resume
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
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge" / "gatc"
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"

OUTPUT_INTERPRETER = DATA_DIR / "claude_generated_interpreter.jsonl"
OUTPUT_COACH = DATA_DIR / "claude_generated_coach.jsonl"
OUTPUT_COMBINED = DATA_DIR / "claude_generated_combined.jsonl"
CHECKPOINT_FILE = DATA_DIR / ".claude_generation_checkpoint.jsonl"

# ---------------------------------------------------------------------------
# Load system prompts
# ---------------------------------------------------------------------------
def load_system_prompt(mode: str) -> str:
    """Load the v6 system prompt for interpreter or coach mode."""
    path = PROMPTS_DIR / f"josi_v6_{mode}.txt"
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"System prompt not found: {path}")


# ---------------------------------------------------------------------------
# Load all knowledge cards
# ---------------------------------------------------------------------------
def load_knowledge_cards() -> dict[str, str]:
    """Load all 18 knowledge cards as {concept_id: full_markdown}."""
    cards = {}
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = md_file.read_text()
        concept_id = md_file.stem
        cards[concept_id] = content
    return cards


def extract_josi_sections(card_content: str) -> str:
    """Extract JOSI: sections from a knowledge card (plain-language coaching content)."""
    sections = []
    in_josi = False
    current = []

    for line in card_content.split("\n"):
        if line.startswith("## JOSI:") or line.startswith("<!-- JOSI:"):
            in_josi = True
            current = [line]
        elif in_josi and (line.startswith("## ") or line.startswith("<!-- ") or line.startswith("---")):
            if current:
                sections.append("\n".join(current))
            in_josi = "JOSI:" in line
            current = [line] if in_josi else []
        elif in_josi:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return "\n\n".join(sections) if sections else ""


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
SPORTS = ["running", "cycling"]
SPORT_ENUM = {"running": "run", "cycling": "bike"}
LEVELS = ["beginner", "intermediate", "advanced"]
READINESS = [
    ("Green", "Recovered"),
    ("Green", "Productive"),
    ("Yellow", "Accumulated"),
    ("Orange", "Fatigued"),
    ("Red", "Overreached"),
]
PHASES = ["base", "build", "peak", "taper", "recovery"]
SESSIONS = {
    "R": ["Recovery spin 20min", "Easy walk 30min"],
    "Z1": ["Continuous Z1 45min", "Continuous Z1 60min"],
    "Z2": ["Continuous Z2 60min", "Continuous Z2 45min", "Continuous Z2 75min"],
    "Z3": ["2x15min Z3 / 5min Z1", "3x10min Z3 / 3min Z1"],
    "Z4": ["4x5min Z4 / 3min Z1", "3x8min Z4 / 4min Z1"],
    "Z5": ["5x3min Z5 / 3min Z1", "4x4min Z5 / 4min Z1"],
    "Z6": ["6x30s Z6 / 90s Z1", "8x20s Z6 / 60s Z1"],
}

# Zone access by readiness (from load_monitoring.md)
ZONE_CAP = {
    "Green": ["R", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"],
    "Yellow": ["R", "Z1", "Z2", "Z3", "Z4"],
    "Orange": ["R", "Z1", "Z2", "Z3"],
    "Red": ["R", "Z1", "Z2"],
}

# Topics mapped to relevant knowledge cards
TOPICS = [
    {
        "topic": "zone_explanation",
        "description": "Athlete asks about training zones, what they mean, how they feel",
        "cards": ["zone_physiology"],
        "action": "answer_question",
    },
    {
        "topic": "session_explanation",
        "description": "Athlete asks why today's session is what it is",
        "cards": ["zone_physiology", "session_rules"],
        "action": "explain",
        "needs_session": True,
    },
    {
        "topic": "readiness_explanation",
        "description": "Athlete asks about their readiness color/status",
        "cards": ["load_monitoring"],
        "action": "explain",
    },
    {
        "topic": "recovery_question",
        "description": "Athlete asks about recovery, rest days, fatigue",
        "cards": ["load_monitoring", "energy_systems"],
        "action": "answer_question",
    },
    {
        "topic": "periodization_question",
        "description": "Athlete asks about training phases, base building, peaking",
        "cards": ["periodization"],
        "action": "answer_question",
    },
    {
        "topic": "create_workout",
        "description": "Athlete wants a workout created",
        "cards": ["session_rules", "zone_physiology"],
        "action": "create_workout",
    },
    {
        "topic": "replan_skip",
        "description": "Athlete wants to skip or modify today's session",
        "cards": ["load_monitoring", "session_rules"],
        "action": "replan",
    },
    {
        "topic": "replan_illness",
        "description": "Athlete is sick and needs plan adjustment",
        "cards": ["load_monitoring"],
        "action": "replan",
    },
    {
        "topic": "motivation",
        "description": "Athlete struggling with motivation or confidence",
        "cards": ["josi_explanations"],
        "action": "answer_question",
    },
    {
        "topic": "intensity_question",
        "description": "Athlete asks about effort, pacing, heart rate zones",
        "cards": ["zone_physiology", "zone_anchors"],
        "action": "answer_question",
    },
    {
        "topic": "weekly_overview",
        "description": "Athlete asks about their week or training structure",
        "cards": ["periodization", "session_rules"],
        "action": "explain",
    },
    {
        "topic": "sport_modifier",
        "description": "Athlete asks sport-specific questions (running impact, cycling heat)",
        "cards": ["modifiers_running", "modifiers_cycling", "modifiers"],
        "action": "answer_question",
    },
    {
        "topic": "goal_demands",
        "description": "Athlete asks about race preparation, event demands",
        "cards": ["goal_demands", "periodization"],
        "action": "answer_question",
    },
    {
        "topic": "encouragement_after_session",
        "description": "Athlete reports how a session went, needs feedback",
        "cards": ["zone_physiology", "load_monitoring"],
        "action": "explain",
        "needs_session": True,
    },
    {
        "topic": "clarify_ambiguous",
        "description": "Athlete message is too vague to act on",
        "cards": [],
        "action": "clarify",
    },
    {
        "topic": "guardrail_refusal",
        "description": "Athlete tries to get Josi to prescribe, override readiness, or use jargon",
        "cards": ["load_monitoring"],
        "action": "explain",
        "is_guardrail": True,
    },
]

# Banned jargon (from coach system prompt)
BANNED_JARGON = [
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "vo2 max", "VO2max",
    "lactate threshold", "ftp", "threshold power",
    "anaerobic capacity", "mitochondrial density", "capillary growth",
    "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "gatc", "ewma", "tss", "ctl", "atl", "tsb",
]


@dataclass
class Scenario:
    """A single training scenario."""
    topic: dict
    sport: str
    level: str
    readiness: tuple[str, str]  # (color, state)
    phase: str
    session_zone: str | None = None
    session_desc: str | None = None
    language: str = "english"
    memory: list[str] = field(default_factory=list)

    def context_block(self) -> str:
        """Build the CONTEXT block for the user message."""
        lines = [
            f"- Readiness: {self.readiness[0]} ({self.readiness[1]})",
            f"- Sport: {self.sport}",
            f"- Level: {self.level}",
            f"- Phase: {self.phase}",
        ]
        if self.session_zone and self.session_desc:
            lines.insert(1, f"- Session: {self.session_zone} \"{self.session_desc}\" ({self.phase} phase)")
        if self.memory:
            lines.append(f"- Memory: {'; '.join(self.memory)}")
        return "\n".join(lines)

    def id(self) -> str:
        """Unique hash for deduplication."""
        key = f"{self.topic['topic']}|{self.sport}|{self.level}|{self.readiness}|{self.phase}|{self.session_zone}|{self.language}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------
MEMORIES = [
    ["prefers morning training"],
    ["primary sport: running", "trains 4 days per week"],
    ["recovering from knee injury 3 months ago"],
    ["has a half marathon in 8 weeks"],
    ["prefers cycling outdoors over indoor trainer"],
    ["works shifts, schedule varies weekly"],
    ["training for first 10K race"],
    ["experienced runner, 5+ years"],
    [],  # no memory (common case)
    [],
    [],
]


def generate_scenarios(count: int, dutch_ratio: float = 0.2, seed: int = 42) -> list[Scenario]:
    """Generate a diverse set of training scenarios."""
    rng = random.Random(seed)
    scenarios = []
    seen = set()

    while len(scenarios) < count:
        topic = rng.choice(TOPICS)
        sport = rng.choice(SPORTS)
        level = rng.choice(LEVELS)
        readiness = rng.choice(READINESS)
        phase = rng.choice(PHASES)
        language = "dutch" if rng.random() < dutch_ratio else "english"
        memory = rng.choice(MEMORIES)

        # Pick a valid session for this readiness
        session_zone = None
        session_desc = None
        if topic.get("needs_session"):
            available_zones = ZONE_CAP[readiness[0]]
            session_zone = rng.choice(available_zones)
            session_desc = rng.choice(SESSIONS.get(session_zone, [f"Continuous {session_zone} 45min"]))

        scenario = Scenario(
            topic=topic,
            sport=sport,
            level=level,
            readiness=readiness,
            phase=phase,
            session_zone=session_zone,
            session_desc=session_desc,
            language=language,
            memory=memory,
        )

        # Dedup
        sid = scenario.id()
        if sid not in seen:
            seen.add(sid)
            scenarios.append(scenario)

    return scenarios


# ---------------------------------------------------------------------------
# Claude generation prompts
# ---------------------------------------------------------------------------

def build_paired_generation_prompt(
    scenario: Scenario,
    knowledge_cards: dict[str, str],
    interpreter_prompt: str,
    coach_prompt: str,
) -> str:
    """Build the prompt for Claude to generate a paired training example."""

    # Collect relevant knowledge content
    knowledge_content = []
    for card_id in scenario.topic.get("cards", []):
        if card_id in knowledge_cards:
            josi_sections = extract_josi_sections(knowledge_cards[card_id])
            if josi_sections:
                knowledge_content.append(f"### {card_id}\n{josi_sections}")
            else:
                # Use first 500 chars of the card if no JOSI sections
                knowledge_content.append(f"### {card_id}\n{knowledge_cards[card_id][:500]}")

    knowledge_text = "\n\n".join(knowledge_content) if knowledge_content else "(no specific knowledge for this topic)"

    # Language instruction
    lang_inst = ""
    if scenario.language == "dutch":
        lang_inst = """
LANGUAGE: Generate the athlete message in Dutch. The coach response must also be in Dutch.
Use natural, warm Dutch — not translated English. Use Dutch idioms and phrasing.
Examples of Dutch athlete messages: "Waarom moet ik vandaag rustig doen?", "Ik voel me moe, kan ik beter overslaan?", "Wat is Zone 2 precies?"
"""

    # Guardrail instruction
    guardrail_inst = ""
    if scenario.topic.get("is_guardrail"):
        guardrail_inst = """
GUARDRAIL SCENARIO: Generate an athlete message that tries to:
- Get Josi to prescribe a specific workout ("Give me 5x1km at 4:00 pace")
- Override their readiness ("I know I'm red but I want to do intervals anyway")
- Ask about internal system details ("What algorithm do you use?")
- Use technical jargon ("What's my CTL/ATL ratio?")

The interpreter should classify this as explain/answer_question (NOT create_workout).
The coach response should gently decline, redirect, and explain the boundary without being preachy.
"""

    # Action-specific instructions
    action = scenario.topic["action"]
    action_inst = ""
    if action == "create_workout":
        action_inst = f"""ACTION: create_workout
The athlete wants a workout. The interpreter JSON must include:
- action: "create_workout"
- sport: "{SPORT_ENUM[scenario.sport]}"
- free_text: (the athlete's original message)
- Optionally: time_available_min (if mentioned), goal, constraints.fatigue_hint
The coach response should acknowledge the request warmly and say they'll set something up.
"""
    elif action == "replan":
        replan_type = "illness" if "illness" in scenario.topic["topic"] else random.choice(
            ["skip_today", "reduce_intensity", "reschedule"]
        )
        action_inst = f"""ACTION: replan
The athlete wants to change their plan. The interpreter JSON must include:
- action: "replan"
- replan_type: "{replan_type}"
- free_text: (the athlete's original message)
The coach response should be understanding, validate their decision, and explain what happens next.
"""
    elif action == "explain":
        action_inst = """ACTION: explain
The athlete asks about THEIR specific situation (session, readiness, week).
The interpreter JSON must include: action: "explain", question: (extracted question), free_text
The coach response must reference the specific context (readiness color, session details, phase).
"""
    elif action == "answer_question":
        action_inst = """ACTION: answer_question
The athlete asks a general coaching/education question.
The interpreter JSON must include: action: "answer_question", question: (extracted question), free_text
The coach response must ground in the [KNOWLEDGE] provided — rephrase naturally, don't quote.
"""
    elif action == "clarify":
        action_inst = """ACTION: clarify
The athlete's message is ambiguous. The interpreter JSON must include:
- action: "clarify"
- missing: [list of what's needed, e.g. "sport"]
- clarify_message: (a friendly question to ask, in the athlete's language)
- free_text: (the athlete's original message)
The coach is NOT called for clarify actions (interpreter only).
"""

    # Session context if present
    session_ctx = ""
    if scenario.session_zone and scenario.session_desc:
        session_ctx = f"\nToday's session: {scenario.session_zone} \"{scenario.session_desc}\" ({scenario.phase} phase)"

    prompt = f"""You are creating training data for Josi, a mobile AI coaching assistant for endurance athletes.
Josi runs 100% on-device (no internet). Your job: generate realistic, high-quality training examples.

SCENARIO:
- Sport: {scenario.sport}
- Level: {scenario.level}
- Readiness: {scenario.readiness[0]} ({scenario.readiness[1]})
- Phase: {scenario.phase}{session_ctx}
- Topic: {scenario.topic['description']}
{f"- Memory: {'; '.join(scenario.memory)}" if scenario.memory else ""}
{lang_inst}
{guardrail_inst}
{action_inst}

KNOWLEDGE CARDS (coaching science the model has access to):
{knowledge_text}

INTERPRETER SYSTEM PROMPT:
{interpreter_prompt[:500]}...

COACH SYSTEM PROMPT:
{coach_prompt}

---

Generate ONE training example. Output EXACTLY this JSON structure:

{{
  "athlete_message": "the realistic athlete message",
  "interpreter_json": {{the GATCRequest JSON object}},
  "coach_response": "the warm coaching response (60-150 words)",
  "knowledge_used": "brief summary of which knowledge was applied"
}}

QUALITY REQUIREMENTS:
1. Athlete message must be natural, varied — real people don't write perfectly. Include typos occasionally, casual language, emotions.
2. Interpreter JSON must be valid GATCRequest: action + free_text always required. Include only relevant fields.
3. Coach response must:
   - Ground in the knowledge cards provided (rephrase, don't quote)
   - React to the specific context (readiness, session, sport, level)
   - Use plain language — NEVER use: {', '.join(BANNED_JARGON[:10])}...
   - Be warm and direct — no corporate speak, no filler
   - Answer first, then explain
   - Under 150 words for simple topics, up to 200 for complex
   - End with statement/encouragement, NOT a question
   - NEVER mention knowledge cards, interpreters, or internal systems
4. free_text in interpreter JSON must exactly match the athlete message.

Output ONLY the JSON object. No markdown fences. No explanation."""

    return prompt


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def check_jargon(text: str) -> list[str]:
    """Check for banned jargon in text."""
    violations = []
    text_lower = text.lower()
    for term in BANNED_JARGON:
        if term.lower() in text_lower:
            violations.append(term)
    return violations


def validate_interpreter_json(data: dict) -> list[str]:
    """Validate interpreter JSON against GATCRequest schema."""
    issues = []
    if "action" not in data:
        issues.append("Missing 'action' field")
        return issues
    if "free_text" not in data:
        issues.append("Missing 'free_text' field")

    action = data["action"]
    valid_actions = ["create_workout", "replan", "explain", "answer_question", "clarify"]
    if action not in valid_actions:
        issues.append(f"Invalid action: {action}")

    if action == "create_workout" and "sport" not in data:
        issues.append("create_workout requires 'sport'")
    if action == "replan" and "replan_type" not in data:
        issues.append("replan requires 'replan_type'")
    if action == "clarify" and "missing" not in data:
        issues.append("clarify requires 'missing'")
    if action in ("explain", "answer_question") and "question" not in data:
        issues.append(f"{action} requires 'question'")

    # Check enum values
    if "sport" in data:
        valid_sports = ["run", "bike", "ski", "skate", "strength", "other"]
        if data["sport"] not in valid_sports:
            issues.append(f"Invalid sport: {data['sport']}")
    if "replan_type" in data:
        valid_replans = ["skip_today", "swap_days", "reschedule", "reduce_intensity",
                         "illness", "travel", "goal_change"]
        if data["replan_type"] not in valid_replans:
            issues.append(f"Invalid replan_type: {data['replan_type']}")

    return issues


def validate_coach_response(response: str, scenario: Scenario) -> list[str]:
    """Validate coach response quality."""
    issues = []

    # Jargon check
    jargon = check_jargon(response)
    if jargon:
        issues.append(f"Jargon: {', '.join(jargon)}")

    # Length check
    words = len(response.split())
    if words < 25:
        issues.append(f"Too short: {words} words")
    if words > 250:
        issues.append(f"Too long: {words} words")

    # Generic opener check
    generic = ["great question", "that's a great", "absolutely!", "of course!",
               "certainly!", "definitely!", "sure thing!"]
    for g in generic:
        if response.lower().startswith(g):
            issues.append(f"Generic opener: '{g}'")
            break

    # System reference check
    system_refs = ["knowledge card", "knowledge block", "interpreter", "gatc engine",
                   "the system", "the algorithm", "my programming"]
    for ref in system_refs:
        if ref in response.lower():
            issues.append(f"System reference: '{ref}'")

    # Readiness acknowledgment for Red
    if scenario.readiness[0] == "Red":
        fatigue_words = ["fatigue", "tired", "rest", "recover", "easy", "light",
                         "careful", "signal", "back off", "moe", "rust", "herstel",
                         "overreached", "accumulated"]
        if not any(w in response.lower() for w in fatigue_words):
            issues.append("Red readiness not acknowledged")

    return issues


# ---------------------------------------------------------------------------
# Claude API interaction
# ---------------------------------------------------------------------------

async def generate_one_example(
    client,
    scenario: Scenario,
    knowledge_cards: dict[str, str],
    interpreter_prompt: str,
    coach_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 2,
) -> tuple[dict | None, list[str]]:
    """Generate one paired training example using Claude."""

    prompt = build_paired_generation_prompt(
        scenario, knowledge_cards, interpreter_prompt, coach_prompt
    )

    for attempt in range(max_retries + 1):
        try:
            message = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=1000,
                temperature=0.8,  # Higher temp for diversity
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            # Parse JSON response
            # Try to extract JSON from the response (Claude sometimes wraps it)
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                if attempt < max_retries:
                    continue
                return None, ["Could not parse JSON from response"]

            result = json.loads(json_match.group())

            # Validate required fields
            issues = []
            if "athlete_message" not in result:
                issues.append("Missing athlete_message")
            if "interpreter_json" not in result:
                issues.append("Missing interpreter_json")
            if scenario.topic["action"] != "clarify" and "coach_response" not in result:
                issues.append("Missing coach_response")

            if issues:
                if attempt < max_retries:
                    continue
                return None, issues

            # Validate interpreter JSON
            interp_issues = validate_interpreter_json(result["interpreter_json"])
            if interp_issues:
                issues.extend(interp_issues)

            # Validate coach response (if present)
            if "coach_response" in result and scenario.topic["action"] != "clarify":
                coach_issues = validate_coach_response(result["coach_response"], scenario)
                if coach_issues:
                    # Retry on jargon violations
                    if any("Jargon" in i for i in coach_issues) and attempt < max_retries:
                        prompt += f"\n\nCRITICAL: Remove all banned jargon. Use plain language."
                        continue
                    issues.extend(coach_issues)

            # Fix free_text to match athlete message
            if "athlete_message" in result and "interpreter_json" in result:
                result["interpreter_json"]["free_text"] = result["athlete_message"]

            return result, issues

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                continue
            return None, [f"JSON decode error: {e}"]
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
                continue
            return None, [f"API error: {e}"]

    return None, ["Max retries exceeded"]


# ---------------------------------------------------------------------------
# Build training examples in the correct format
# ---------------------------------------------------------------------------

def build_interpreter_example(
    athlete_message: str,
    interpreter_json: dict,
    context_block: str,
    interpreter_prompt: str,
) -> dict:
    """Build an interpreter training example in ChatML format."""
    user_content = athlete_message
    if context_block:
        user_content += f"\n\nCONTEXT:\n{context_block}"

    return {
        "messages": [
            {"role": "system", "content": interpreter_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(interpreter_json, ensure_ascii=False)},
        ]
    }


def build_coach_example(
    athlete_message: str,
    coach_response: str,
    context_block: str,
    knowledge_text: str,
    interpreter_json: dict,
    coach_prompt: str,
) -> dict:
    """Build a coach training example in ChatML format."""
    user_content = athlete_message
    if context_block:
        user_content += f"\n\nCONTEXT:\n{context_block}"
    if knowledge_text:
        user_content += f"\n\n[KNOWLEDGE]\n{knowledge_text}"
    user_content += f"\n\n[INTERPRETER]\n{json.dumps(interpreter_json, ensure_ascii=False)}"

    return {
        "messages": [
            {"role": "system", "content": coach_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": coach_response},
        ]
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

async def run_generation(args):
    """Main generation loop."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load resources
    print("Loading resources...")
    knowledge_cards = load_knowledge_cards()
    print(f"  Knowledge cards: {len(knowledge_cards)}")

    interpreter_prompt = load_system_prompt("interpreter")
    coach_prompt = load_system_prompt("coach")
    print(f"  Interpreter prompt: {len(interpreter_prompt)} chars")
    print(f"  Coach prompt: {len(coach_prompt)} chars")

    # Generate scenarios
    print(f"\nGenerating {args.count} scenarios...")
    scenarios = generate_scenarios(
        count=args.count,
        dutch_ratio=args.dutch_ratio,
        seed=args.seed,
    )
    print(f"  Generated: {len(scenarios)} unique scenarios")

    # Filter by focus
    if args.focus == "interpreter":
        # Keep all actions
        pass
    elif args.focus == "coach":
        # Skip clarify (coach isn't called for clarify)
        scenarios = [s for s in scenarios if s.topic["action"] != "clarify"]
        print(f"  After coach filter: {len(scenarios)} scenarios")

    # Show distribution
    topic_counts = {}
    for s in scenarios:
        t = s.topic["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print("\n  Topic distribution:")
    for t, c in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    lang_counts = {"english": 0, "dutch": 0}
    for s in scenarios:
        lang_counts[s.language] += 1
    print(f"\n  Language: EN={lang_counts['english']} NL={lang_counts['dutch']}")

    # Load checkpoint if resuming
    completed_ids = set()
    checkpoint_data = []
    if args.resume and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                entry = json.loads(line.strip())
                completed_ids.add(entry["scenario_id"])
                checkpoint_data.append(entry)
        print(f"\n  Resumed: {len(completed_ids)} already completed")

    # Preview mode
    if args.preview > 0:
        print(f"\n{'='*60}")
        print(f"  PREVIEW MODE — generating {args.preview} examples")
        print(f"{'='*60}")

        random.shuffle(scenarios)
        for i, scenario in enumerate(scenarios[:args.preview]):
            print(f"\n--- Example {i+1} ---")
            print(f"Topic: {scenario.topic['topic']}")
            print(f"Sport: {scenario.sport} | Level: {scenario.level}")
            print(f"Readiness: {scenario.readiness[0]} ({scenario.readiness[1]})")
            print(f"Phase: {scenario.phase} | Language: {scenario.language}")
            if scenario.session_zone:
                print(f"Session: {scenario.session_zone} \"{scenario.session_desc}\"")

            result, issues = await generate_one_example(
                client, scenario, knowledge_cards,
                interpreter_prompt, coach_prompt,
                model=args.model,
            )

            if result:
                print(f"\nAthlete: {result['athlete_message']}")
                print(f"\nInterpreter JSON:")
                print(f"  {json.dumps(result['interpreter_json'], indent=2)}")
                if "coach_response" in result:
                    print(f"\nCoach response ({len(result['coach_response'].split())} words):")
                    print(f"  {result['coach_response']}")
                if result.get("knowledge_used"):
                    print(f"\nKnowledge used: {result['knowledge_used']}")
                status = "PASS" if not issues else f"ISSUES: {issues}"
                print(f"\nQuality: {status}")
            else:
                print(f"\nFAILED: {issues}")
        return

    # Full generation
    print(f"\n{'='*60}")
    print(f"  GENERATING {len(scenarios)} paired examples")
    print(f"  Model: {args.model}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"{'='*60}")

    interpreter_examples = []
    coach_examples = []
    stats = {
        "total": 0, "success": 0, "failed": 0,
        "interp_valid": 0, "coach_valid": 0,
        "jargon_violations": 0, "skipped_checkpoint": 0,
    }
    start_time = time.time()

    # Process in batches
    batch_size = args.batch_size
    for batch_start in range(0, len(scenarios), batch_size):
        batch = scenarios[batch_start:batch_start + batch_size]

        # Filter already completed
        batch_todo = []
        for scenario in batch:
            if scenario.id() in completed_ids:
                stats["skipped_checkpoint"] += 1
                continue
            batch_todo.append(scenario)

        if not batch_todo:
            continue

        # Run batch concurrently
        sem = asyncio.Semaphore(args.concurrency)

        async def process_one(s):
            async with sem:
                return await generate_one_example(
                    client, s, knowledge_cards,
                    interpreter_prompt, coach_prompt,
                    model=args.model,
                )

        results = await asyncio.gather(
            *[process_one(s) for s in batch_todo],
            return_exceptions=True,
        )

        # Process results
        for scenario, result_or_err in zip(batch_todo, results):
            stats["total"] += 1

            if isinstance(result_or_err, Exception):
                stats["failed"] += 1
                continue

            result, issues = result_or_err
            if result is None:
                stats["failed"] += 1
                continue

            stats["success"] += 1

            # Track jargon
            if any("Jargon" in str(i) for i in issues):
                stats["jargon_violations"] += 1

            # Build knowledge text for coach example
            knowledge_text = ""
            for card_id in scenario.topic.get("cards", []):
                if card_id in knowledge_cards:
                    josi = extract_josi_sections(knowledge_cards[card_id])
                    if josi:
                        knowledge_text += josi[:300] + "\n"

            # Build interpreter example
            interp_ex = build_interpreter_example(
                athlete_message=result["athlete_message"],
                interpreter_json=result["interpreter_json"],
                context_block=scenario.context_block(),
                interpreter_prompt=interpreter_prompt,
            )
            interpreter_examples.append(interp_ex)

            interp_issues = validate_interpreter_json(result["interpreter_json"])
            if not interp_issues:
                stats["interp_valid"] += 1

            # Build coach example (if not clarify)
            if "coach_response" in result and scenario.topic["action"] != "clarify":
                coach_ex = build_coach_example(
                    athlete_message=result["athlete_message"],
                    coach_response=result["coach_response"],
                    context_block=scenario.context_block(),
                    knowledge_text=knowledge_text,
                    interpreter_json=result["interpreter_json"],
                    coach_prompt=coach_prompt,
                )
                coach_examples.append(coach_ex)

                coach_issues = validate_coach_response(result["coach_response"], scenario)
                if not coach_issues:
                    stats["coach_valid"] += 1

            # Checkpoint
            with open(CHECKPOINT_FILE, "a") as f:
                f.write(json.dumps({
                    "scenario_id": scenario.id(),
                    "success": True,
                    "issues": issues,
                }) + "\n")

        # Progress
        elapsed = time.time() - start_time
        done = stats["total"]
        total = len(scenarios) - stats["skipped_checkpoint"]
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {rate:.1f}/s  ETA: {eta/60:.0f}min  "
              f"ok={stats['success']} fail={stats['failed']} "
              f"jargon={stats['jargon_violations']}")

    # Save output
    print(f"\n{'='*60}")
    print(f"  SAVING RESULTS")
    print(f"{'='*60}")

    # Add any checkpointed data from previous runs
    # (The checkpoint only tracks what was done, actual examples are in memory)

    # Save interpreter examples
    if interpreter_examples:
        with open(OUTPUT_INTERPRETER, "w") as f:
            for ex in interpreter_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Interpreter: {len(interpreter_examples)} examples → {OUTPUT_INTERPRETER.name}")

    # Save coach examples
    if coach_examples:
        with open(OUTPUT_COACH, "w") as f:
            for ex in coach_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Coach: {len(coach_examples)} examples → {OUTPUT_COACH.name}")

    # Save combined (shuffled)
    combined = interpreter_examples + coach_examples
    random.shuffle(combined)
    if combined:
        with open(OUTPUT_COMBINED, "w") as f:
            for ex in combined:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Combined: {len(combined)} examples → {OUTPUT_COMBINED.name}")

    # Stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Generated: {stats['success']}/{stats['total']} examples")
    print(f"  Failed: {stats['failed']}")
    print(f"  Interpreter valid: {stats['interp_valid']}/{len(interpreter_examples)}")
    print(f"  Coach valid: {stats['coach_valid']}/{len(coach_examples)}")
    print(f"  Jargon violations: {stats['jargon_violations']}")
    if stats["skipped_checkpoint"] > 0:
        print(f"  Skipped (checkpoint): {stats['skipped_checkpoint']}")

    print(f"\n  Output files:")
    print(f"    {OUTPUT_INTERPRETER.name}  ({len(interpreter_examples)} interpreter examples)")
    print(f"    {OUTPUT_COACH.name}  ({len(coach_examples)} coach examples)")
    print(f"    {OUTPUT_COMBINED.name}  ({len(combined)} total)")

    print(f"\n  Next steps:")
    print(f"    1. Review: python generate_claude_dataset.py --preview 10")
    print(f"    2. Merge with existing data:")
    print(f"       cat {OUTPUT_INTERPRETER.name} >> train_interpreter.jsonl")
    print(f"       cat {OUTPUT_COACH.name} >> train_explainer_sequential.jsonl")
    print(f"    3. Rebuild unified: python prepare_v6_data.py --update-prompts --inject-knowledge")
    print(f"    4. Retrain: python finetune_qwen3.py train --mode unified")

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for Josi using Claude (knowledge distillation)"
    )
    parser.add_argument("--preview", type=int, default=0,
                        help="Preview N random examples (dry run)")
    parser.add_argument("--run", action="store_true",
                        help="Run full generation")
    parser.add_argument("--count", type=int, default=1000,
                        help="Number of paired examples to generate (default: 1000)")
    parser.add_argument("--focus", choices=["both", "interpreter", "coach"], default="both",
                        help="Which example type to generate (default: both)")
    parser.add_argument("--dutch-ratio", type=float, default=0.2,
                        help="Fraction of Dutch examples (default: 0.2)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Examples per batch (default: 10)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Concurrent API calls per batch (default: 5)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use (default: claude-sonnet-4-20250514)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if interrupted")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    if not args.run and args.preview == 0:
        parser.print_help()
        print("\n  Quick start:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    python generate_claude_dataset.py --preview 5")
        print("    python generate_claude_dataset.py --run --count 1000")
        sys.exit(0)

    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
