#!/usr/bin/env python3
"""
MiValta Josi v5 — Real-World Simulation Script

Runs the full dual-model pipeline (Interpreter → Router → Explainer) with
realistic athlete scenarios. Uses llama-cpp-python for CPU/GPU inference
on GGUF models.

Architecture:
  - LLMs handle ONLY classification + entity extraction + phrasing
  - All policy, triage, and routing is deterministic code
  - InjuryTriage state machine handles pain/injury flows
  - Conversation state resolves short follow-ups ("ok", "so?", "why?")
  - ENGINE_PAYLOAD gives the explainer structured data to verbalize

Pipeline:
  1. User message + STATE JSON → Interpreter → GATCRequest JSON
  2. Post-processor: validate, extract entities, detect injury
  3. Deterministic router: injury triage, clarify repair, short-msg resolution
  4. Build ENGINE_PAYLOAD with the decision
  5. Explainer verbalizes the payload → Dialogue Governor → tone check

Usage:
    python scripts/simulate.py --interactive
    python scripts/simulate.py --scenarios
    python scripts/simulate.py --download --scenarios
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from datetime import date

# Add shared/ to path for post-processors
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from gatc_postprocessor import postprocess_gatc_request, parse_gatc_response
from dialogue_governor import govern_dialogue
from athlete_state_store import AthleteStateStore
from cross_domain_rules import evaluate_rules
from context_assembler import enrich_payload, update_state_from_turn

# Try importing llm_intent_parser for explainer output (LLMIntent JSON)
try:
    from llm_intent_parser import parse_llm_intent
    HAS_INTENT_PARSER = True
except ImportError:
    HAS_INTENT_PARSER = False


# =============================================================================
# TONE GUARDRAILS — block hostile/snarky explainer output
# =============================================================================

_HOSTILE_PATTERNS = [
    r"you'?re just trying", r"you'?re not asking", r"magic pill",
    r"no clue", r"you expect me to", r"hoping for the best",
    r"that fixes everything", r"you'?re just .{0,30} and hoping",
    r"what do you want from me", r"i can'?t help you if",
    r"you need to figure", r"not my problem", r"i'?m not your",
    r"stop wasting", r"pay attention", r"i already told you",
    r"as i said before",
]

_HOSTILE_RES = [re.compile(p, re.IGNORECASE) for p in _HOSTILE_PATTERNS]

_TONE_FALLBACK = (
    "I hear you. Could you tell me a bit more about what's going on "
    "so I can help?"
)


def check_tone(text: str) -> str:
    """Block hostile/snarky explainer output and replace with safe fallback."""
    for r in _HOSTILE_RES:
        if r.search(text):
            return _TONE_FALLBACK
    return text


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_SERVER = "http://144.76.62.249/models"
DEFAULT_MODEL_DIR = REPO_ROOT / "training" / "models" / "gguf"

INTERPRETER_FILENAME = "josi-v5-interpreter-q4_k_m.gguf"
EXPLAINER_FILENAME = "josi-v5-explainer-q4_k_m.gguf"

# Qwen 2.5 ChatML template
CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"

# Inference parameters
INTERPRETER_PARAMS = {
    "n_predict": 150,
    "temperature": 0.45,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# Explainer gets higher repeat_penalty to reduce cross-turn repetition
EXPLAINER_PARAMS = {
    "n_predict": 150,
    "temperature": 0.5,
    "top_p": 0.9,
    "repeat_penalty": 1.4,
}


# =============================================================================
# INJURY TRIAGE STATE MACHINE — deterministic, not LLM
# =============================================================================

# Regex patterns for extracting pain info from user messages
_PAIN_AREA_PATTERNS = {
    "knee": r"\bknee\b",
    "elbow": r"\belbow\b",
    "ankle": r"\bankle\b",
    "hip": r"\bhip\b",
    "back": r"\b(?:back|lower\s*back|upper\s*back|spine)\b",
    "shoulder": r"\bshoulder\b",
    "shin": r"\bshin\b",
    "calf": r"\bcalf|calves\b",
    "foot": r"\bfoot|feet|heel|arch\b",
    "wrist": r"\bwrist\b",
    "neck": r"\bneck\b",
}

_SIDE_PATTERNS = {
    "left": r"\bleft\b",
    "right": r"\bright\b",
    "both": r"\bboth\b",
}

_TRIGGER_PATTERNS = {
    "hard_effort": r"\bhard|fast|sprint|intense|interval|hill\b",
    "running": r"\brun|running|jog\b",
    "cycling": r"\bbike|cycling|ride|handlebar|steering\b",
    "gripping": r"\bgrip|gripping|holding|handlebar|steering\b",
    "rest": r"\brest|sitting|lying|sleep|at rest\b",
}

_SEVERITY_PATTERN = re.compile(r"\b(\d{1,2})\s*(?:/\s*10|out\s*of\s*10)\b")
_DURATION_PATTERN = re.compile(r"\b(\d+)\s*(?:day|week|month)s?\b")

# Medical red flags — these ALWAYS bypass triage and go to emergency
_RED_FLAG_KEYWORDS = [
    "chest pain", "chest", "heart", "dizzy", "dizziness", "faint",
    "fainting", "blacked out", "black out", "can't breathe",
    "breathing problem", "numb", "numbness", "tingling",
    "lost consciousness",
]


def extract_pain_info(message: str) -> dict:
    """Extract structured pain data from a user message. Deterministic."""
    lower = message.lower()
    info = {}

    for area, pattern in _PAIN_AREA_PATTERNS.items():
        if re.search(pattern, lower):
            info["pain_area"] = area
            break

    for side, pattern in _SIDE_PATTERNS.items():
        if re.search(pattern, lower):
            info["side"] = side
            break

    for trigger, pattern in _TRIGGER_PATTERNS.items():
        if re.search(pattern, lower):
            info["trigger"] = trigger
            break

    sev_match = _SEVERITY_PATTERN.search(lower)
    if sev_match:
        val = int(sev_match.group(1))
        if 0 <= val <= 10:
            info["severity_0_10"] = val

    dur_match = _DURATION_PATTERN.search(lower)
    if dur_match:
        info["duration_days"] = int(dur_match.group(1))
        if "week" in lower:
            info["duration_days"] *= 7
        elif "month" in lower:
            info["duration_days"] *= 30

    if any(kw in lower for kw in ["swelling", "swollen", "puffy"]):
        info["swelling"] = True
    if any(kw in lower for kw in ["no swelling", "not swollen"]):
        info["swelling"] = False

    return info


def has_medical_red_flag(message: str) -> bool:
    """Check if message contains medical emergency keywords."""
    lower = message.lower()
    return any(kw in lower for kw in _RED_FLAG_KEYWORDS)


def message_mentions_pain(message: str) -> bool:
    """Check if message discusses pain/injury."""
    lower = message.lower()
    return any(kw in lower for kw in [
        "pain", "hurt", "hurts", "ache", "sore", "injury", "injured",
    ])


class InjuryTriage:
    """Deterministic injury triage state machine.

    Collects pain data step by step, screens for red flags, then
    produces a recommendation. The LLM is NOT involved in triage logic.
    """

    # Fields we need before making a recommendation
    REQUIRED = ["pain_area", "severity_0_10", "trigger"]

    # Deterministic triage questions for each missing field
    QUESTIONS = {
        "pain_area": "Where exactly is the pain — knee, elbow, back, shoulder, or somewhere else?",
        "severity_0_10": "On a scale of 0-10, how bad is the pain during activity?",
        "trigger": "What triggers it most — easy effort, hard effort, specific movements, or does it hurt at rest too?",
        "duration_days": "How long has this been going on — days, weeks?",
        "swelling": "Is there any swelling or visible changes around the area?",
    }

    def __init__(self):
        self.data: dict = {}
        self.stage = "collecting"  # collecting | recommending | referred

    def update(self, new_data: dict):
        """Merge newly extracted pain data."""
        for k, v in new_data.items():
            if v is not None:
                self.data[k] = v

    def is_red_flag(self) -> bool:
        return self.data.get("red_flag", False)

    def get_missing(self) -> list[str]:
        return [f for f in self.REQUIRED if f not in self.data]

    def next_question(self) -> str | None:
        """Return the next deterministic triage question, or None if complete."""
        if self.is_red_flag():
            return None  # Emergency — don't triage, refer immediately

        missing = self.get_missing()
        if missing:
            return self.QUESTIONS[missing[0]]

        # Required fields complete — check optional
        if "duration_days" not in self.data:
            return self.QUESTIONS["duration_days"]
        if "swelling" not in self.data:
            return self.QUESTIONS["swelling"]

        # All collected — ready to recommend
        self.stage = "recommending"
        return None

    def get_recommendation(self) -> str:
        """Generate deterministic recommendation from collected data."""
        severity = self.data.get("severity_0_10", 5)
        pain_area = self.data.get("pain_area", "that area")
        trigger = self.data.get("trigger", "activity")
        duration = self.data.get("duration_days", 0)
        swelling = self.data.get("swelling", False)

        if severity >= 7 or swelling or duration > 14:
            self.stage = "referred"
            return (
                f"Your {pain_area} pain needs professional attention. "
                f"{'Swelling plus high ' if swelling else 'High '}"
                f"pain during {trigger} "
                f"{'over two weeks ' if duration > 14 else ''}"
                f"means we should get it checked before pushing through. "
                f"See a physio or sports doctor. "
                f"I'll adjust your plan to avoid aggravating it."
            )
        elif severity >= 4:
            return (
                f"Your {pain_area} is telling you something. "
                f"Pain at {severity}/10 during {trigger} means we should "
                f"back off intensity there. We can modify your sessions — "
                f"lower impact, avoid the movements that trigger it. "
                f"If it doesn't improve in a week, get it checked."
            )
        else:
            return (
                f"Mild {pain_area} discomfort ({severity}/10) is common. "
                f"Keep training but stay below the pain threshold. "
                f"Warm up well, and if it worsens during a session, stop. "
                f"Your plan can adapt around this."
            )

    def to_dict(self) -> dict:
        return {"stage": self.stage, "data": self.data}


# =============================================================================
# CONVERSATION STATE — tracks history, topic, triage, pending questions
# =============================================================================

# Short acknowledgements that are not real questions
_ACK_WORDS = {
    "ok", "okay", "k", "sure", "yes", "yeah", "yep", "alright",
    "right", "cool", "fine", "thanks", "thank you", "cheers",
    "hm", "hmm", "ah", "oh",
}

# Short follow-ups that refer to previous context
_FOLLOWUP_WORDS = {
    "so", "so?", "and?", "then?", "why?", "how?", "what?",
    "really?", "is that all?", "anything else?", "just did",
}


class ConversationState:
    """Tracks conversation history, topic, triage state, and pending questions."""

    def __init__(self, max_turns: int = 6, athlete_id: str = "default"):
        self.max_turns = max_turns
        self.history: list[dict] = []
        self.last_topic: str | None = None
        self.last_user_problem: str | None = None
        self.last_action: str | None = None
        self.pending_question: str | None = None  # What Josi last asked
        self.triage: InjuryTriage | None = None    # Active injury triage
        self.athlete_store = AthleteStateStore(athlete_id)  # Cross-domain state

    def add_user(self, message: str):
        self.history.append({"role": "user", "message": message})
        self._trim()
        self._extract_topic(message)

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "message": message})
        self._trim()

    def _trim(self):
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def _extract_topic(self, message: str):
        lower = message.lower()
        topic_keywords = {
            "sleep": ["sleep", "insomnia", "bad night", "wake up", "slept badly",
                       "can't sleep", "not sleeping"],
            "fatigue": ["tired", "exhausted", "fatigue", "fatigued", "no energy",
                        "drained", "worn out"],
            "injury": ["hurt", "pain", "sore", "injury", "injured", "ache"],
            "motivation": ["motivation", "motivated", "bored", "boring", "dread",
                           "quit", "give up", "point", "ruined"],
            "nutrition": ["eat", "food", "diet", "nutrition", "fuel", "hydration"],
            "workout": ["workout", "session", "training"],
            "recovery": ["recovery", "rest", "rest day", "off day"],
            "zones": ["zone", "zones", "z2", "z3", "z4", "z5", "heart rate"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in lower for kw in keywords):
                self.last_topic = topic
                self.last_user_problem = message
                return
        # Don't clear topic for short follow-ups

    def format_state_json(self) -> str:
        """Format conversation state as JSON for the interpreter."""
        state = {
            "last_topic": self.last_topic,
            "last_action": self.last_action,
            "pending_question": self.pending_question,
        }
        if self.triage:
            state["triage"] = self.triage.to_dict()
        return json.dumps(state)

    def format_history_block(self) -> str:
        """Format recent history for flat injection into interpreter prompt."""
        if not self.history:
            return ""
        lines = []
        for turn in self.history[:-1]:
            role = turn["role"]
            msg = turn["message"]
            if len(msg) > 120:
                msg = msg[:117] + "..."
            lines.append(f"  [{role}]: {msg}")
        if not lines:
            return ""
        return "\nHISTORY:\n" + "\n".join(lines)

    def get_clean_history(self) -> list[dict]:
        """Return history with system placeholders filtered out."""
        return [
            turn for turn in self.history
            if not (turn["role"] == "assistant" and "GATC Engine" in turn["message"])
        ]


# =============================================================================
# SHORT-MESSAGE RESOLUTION — deterministic, not LLM
# =============================================================================

def resolve_short_message(message: str, state: ConversationState) -> str | None:
    """Resolve short/ambiguous messages using conversation state.

    Returns a rewritten message if resolvable, or None to keep original.
    """
    stripped = message.strip().lower().rstrip("?!. ")
    words = stripped.split()

    if len(words) > 6:
        return None  # Not a short message

    # Pure acknowledgement with no pending question → prompt for intent
    if stripped in _ACK_WORDS and not state.pending_question:
        return None  # Let it through, router will handle

    # Short follow-up referencing previous context
    if stripped in _FOLLOWUP_WORDS or stripped in _ACK_WORDS:
        if state.pending_question:
            # Rewrite using pending question context
            return f"{message} (regarding: {state.pending_question})"
        elif state.last_topic:
            return f"{message} (topic: {state.last_topic})"

    return None


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

INTERPRETER_SYSTEM_PROMPT = """\
You are Josi's Interpreter — you translate athlete messages into structured GATC requests.

TASK: Read the user message, athlete CONTEXT, and conversation STATE. Output ONLY valid JSON. No markdown fences. No explanation.

ACTIONS (action field):
- create_workout: user wants a new workout → extract sport, time, goal, constraints
- replan: user wants to change/skip/swap a session → extract replan_type
- explain: user asks about THEIR specific session, readiness, or plan in CONTEXT → extract question
- answer_question: general coaching/knowledge question → extract question
- injury_triage: user mentions pain, injury, or soreness → extract what you can
- clarify: intent is completely ambiguous → include clarify_message (max 20 words)

STATE HANDLING:
- STATE.last_topic tells you what the conversation is about
- STATE.pending_question is what Josi last asked — use it to interpret short replies
- STATE.triage shows active injury triage data already collected
- If user message is short ("ok", "why?", "so?") and STATE has context, use it

WHEN TO USE injury_triage:
- Any mention of pain, hurt, ache, soreness, injury in any body part
- Do NOT use clarify for pain — use injury_triage instead
- Extract: pain_area, side, trigger, severity if mentioned

ILLNESS vs INJURY vs MEDICAL EMERGENCY:
- "I'm sick", "flu", "cold" → action="replan", replan_type="illness"
- "knee hurts", "elbow pain", "back is sore" → action="injury_triage"
- "chest pain", "dizziness", "heart racing", "can't breathe" → action="injury_triage" with red_flag=true

ENUMS:
- sport: "run", "bike", "ski", "skate", "strength", "other"
- replan_type: "skip_today", "swap_days", "reschedule", "reduce_intensity", "illness", "travel", "goal_change"
- goal: "endurance", "threshold", "vo2", "recovery", "strength", "race_prep"

RULES:
- ALWAYS include free_text with the original user message
- NEVER output markdown fences — raw JSON only
- NEVER invent workouts, zones, durations, paces, or power numbers
- Infer sport from CONTEXT when available"""

EXPLAINER_SYSTEM_PROMPT = """\
You are Josi, MiValta's AI coaching assistant. You generate the final user-facing text.

You will receive an ENGINE_PAYLOAD with the decision already made by the system.
Your job is to VERBALIZE that payload warmly and concisely. Do not override it.

PERSONALITY:
- Empathetic and warm — you genuinely care about the athlete
- Direct and honest — no filler, no corporate speak
- You ARE the coach — never recommend other apps, coaches, or services

RULES:
- Verbalize what ENGINE_PAYLOAD contains. Do not invent beyond it.
- If payload has a "message" field, use it as the core of your response.
- If payload has a "question" field, end with that question.
- Keep responses under 80 words. 3-5 lines max.
- Ask at most ONE question (only if payload includes one).
- NEVER repeat what you said in previous turns.
- If the athlete seems frustrated, acknowledge it warmly first.

CROSS-DOMAIN CONTEXT (if present in payload):
- "cross_domain" contains coaching insights from connecting injury, load, readiness, and history data
- Weave these insights naturally into your response — like a coach who sees the full picture
- Prioritize the first cross_domain message (highest priority)
- Do NOT list them mechanically — integrate them into your coaching advice
- Example: Instead of "Rule says reduce load", say "Your knee is still bothering you and your training has been ramping up — let's ease off this week."

SAFETY:
- If payload says "red_flag": stop training, see a doctor. Be direct.

BOUNDARIES:
- NEVER prescribe, create, or modify training yourself
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems (algorithm, gatc, ewma, tss, cross_domain, rules, etc.)
- NEVER use jargon: periodization, mesocycle, vo2max, lactate threshold, ftp
- NEVER output JSON, [INTERPRETER], ENGINE_PAYLOAD, or internal terms

OUTPUT: Plain coaching text only. No JSON. No markdown."""


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

def download_model(filename: str, dest_dir: Path) -> Path:
    """Download a GGUF model from the nginx server."""
    dest = dest_dir / filename
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Already downloaded: {filename} ({size_mb:.0f} MB)")
        return dest

    url = f"{MODEL_SERVER}/{filename}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {filename}...")
    print(f"    URL: {url}")

    start = time.time()

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    elapsed = time.time() - start
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\n    Done: {size_mb:.0f} MB in {elapsed:.1f}s")
    return dest


def download_models(model_dir: Path) -> tuple[Path, Path]:
    """Download both models."""
    print("\n  Downloading models from server...")
    interp = download_model(INTERPRETER_FILENAME, model_dir)
    expl = download_model(EXPLAINER_FILENAME, model_dir)
    return interp, expl


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class JosiEngine:
    """Dual-model inference engine using llama-cpp-python."""

    def __init__(self, interpreter_path: str, explainer_path: str,
                 n_ctx: int = 4096):
        from llama_cpp import Llama

        print("\n  Loading interpreter model...")
        t0 = time.time()
        self.interpreter = Llama(
            model_path=interpreter_path,
            n_ctx=n_ctx,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        print(f"    Loaded in {time.time() - t0:.1f}s")

        print("  Loading explainer model...")
        t0 = time.time()
        self.explainer_model = Llama(
            model_path=explainer_path,
            n_ctx=n_ctx,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        print(f"    Loaded in {time.time() - t0:.1f}s")

    def _format_chatml(self, system_prompt: str, user_message: str) -> str:
        """Format a prompt using Qwen 2.5 ChatML template."""
        return (
            f"{CHATML_START}system\n{system_prompt}{CHATML_END}\n"
            f"{CHATML_START}user\n{user_message}{CHATML_END}\n"
            f"{CHATML_START}assistant\n"
        )

    def _format_chatml_multiturn(self, system_prompt: str,
                                  history: list[dict],
                                  user_message: str) -> str:
        """Format a multi-turn prompt using proper ChatML turns."""
        parts = [f"{CHATML_START}system\n{system_prompt}{CHATML_END}"]
        for turn in history:
            role = turn["role"]
            msg = turn["message"]
            if len(msg) > 150:
                msg = msg[:147] + "..."
            parts.append(f"{CHATML_START}{role}\n{msg}{CHATML_END}")
        parts.append(f"{CHATML_START}user\n{user_message}{CHATML_END}")
        parts.append(f"{CHATML_START}assistant\n")
        return "\n".join(parts)

    def run_interpreter(self, user_message: str) -> tuple[str, dict | None]:
        """Run the interpreter model → GATCRequest JSON."""
        prompt = self._format_chatml(INTERPRETER_SYSTEM_PROMPT, user_message)

        t0 = time.time()
        result = self.interpreter(
            prompt,
            max_tokens=INTERPRETER_PARAMS["n_predict"],
            temperature=INTERPRETER_PARAMS["temperature"],
            top_p=INTERPRETER_PARAMS["top_p"],
            repeat_penalty=INTERPRETER_PARAMS["repeat_penalty"],
            stop=[CHATML_END, "<|endoftext|>"],
        )
        elapsed = time.time() - t0

        raw = result["choices"][0]["text"].strip()
        tokens = result["usage"]["completion_tokens"]
        print(f"    Interpreter: {tokens} tokens in {elapsed:.2f}s "
              f"({tokens/elapsed:.0f} tok/s)")

        parsed = parse_gatc_response(raw)
        if parsed:
            parsed = postprocess_gatc_request(parsed, user_message)

        return raw, parsed

    def run_explainer(self, user_message: str,
                      engine_payload: dict,
                      history: list[dict] | None = None) -> str:
        """Run the explainer model with ENGINE_PAYLOAD.

        The explainer verbalizes what the deterministic engine decided.
        It does NOT make policy decisions.
        """
        # Build current turn: user message + payload
        payload_str = json.dumps(engine_payload, indent=None)
        current_msg = f"{user_message}\n\nENGINE_PAYLOAD: {payload_str}"

        if history and len(history) > 1:
            prompt = self._format_chatml_multiturn(
                EXPLAINER_SYSTEM_PROMPT, history[:-1], current_msg
            )
        else:
            prompt = self._format_chatml(EXPLAINER_SYSTEM_PROMPT, current_msg)

        t0 = time.time()
        result = self.explainer_model(
            prompt,
            max_tokens=EXPLAINER_PARAMS["n_predict"],
            temperature=EXPLAINER_PARAMS["temperature"],
            top_p=EXPLAINER_PARAMS["top_p"],
            repeat_penalty=EXPLAINER_PARAMS["repeat_penalty"],
            stop=[CHATML_END, "<|endoftext|>"],
        )
        elapsed = time.time() - t0

        raw = result["choices"][0]["text"].strip()
        tokens = result["usage"]["completion_tokens"]
        print(f"    Explainer:   {tokens} tokens in {elapsed:.2f}s "
              f"({tokens/elapsed:.0f} tok/s)")

        # If explainer outputs JSON, extract message field
        if HAS_INTENT_PARSER and raw.strip().startswith("{"):
            intent = parse_llm_intent(raw)
            if intent:
                intent = govern_dialogue(intent)
                raw = intent.get("message", raw)

        return check_tone(raw)

    # -----------------------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------------------

    def simulate(self, user_message: str, context: dict | None = None,
                 state: ConversationState | None = None) -> dict:
        """Run the full pipeline: classify → validate → route → respond."""

        # Track in conversation state
        if state:
            state.add_user(user_message)

        # --- Build CONTEXT block ---
        full_message = user_message
        if context:
            ctx_lines = []
            if "readiness" in context:
                ctx_lines.append(f"- Readiness: {context['readiness']}")
            if "sport" in context:
                ctx_lines.append(f"- Sport: {context['sport']}")
            if "level" in context:
                ctx_lines.append(f"- Level: {context['level']}")
            if "session" in context:
                ctx_lines.append(f"- Session: {context['session']}")
            if "phase" in context:
                ctx_lines.append(f"- Phase: {context['phase']}")
            if ctx_lines:
                full_message += "\n\nCONTEXT:\n" + "\n".join(ctx_lines)

        # --- Build interpreter message with STATE + HISTORY ---
        interp_message = full_message
        if state:
            interp_message += f"\n\n<STATE>\n{state.format_state_json()}\n</STATE>"
            history_block = state.format_history_block()
            if history_block:
                interp_message += history_block

        # --- Short-message resolution ---
        rewritten = None
        if state:
            rewritten = resolve_short_message(user_message, state)
            if rewritten:
                # Replace the user message portion in the interpreter input
                interp_message = interp_message.replace(user_message, rewritten, 1)

        # =====================================================================
        # Step 1: Interpreter — classify + extract
        # =====================================================================
        interp_raw, interp_parsed = self.run_interpreter(interp_message)
        action = interp_parsed.get("action", "unknown") if interp_parsed else "parse_error"

        # =====================================================================
        # Step 2: Deterministic validation + routing
        # =====================================================================

        # --- Detect injury from ANY action if message mentions pain ---
        is_injury = message_mentions_pain(user_message)
        if state and state.triage:
            is_injury = True  # Active triage continues

        # Force injury_triage for pain messages, regardless of interpreter
        if is_injury and action not in ("injury_triage",):
            # Only override if it's not already a sensible action
            if action in ("clarify", "unknown", "parse_error"):
                action = "injury_triage"
            elif action == "answer_question" and state and state.last_topic == "injury":
                action = "injury_triage"

        # --- Medical red flag check (always, before anything else) ---
        if has_medical_red_flag(user_message):
            action = "injury_triage"
            if not state:
                pass  # Handled below
            else:
                if not state.triage:
                    state.triage = InjuryTriage()
                state.triage.data["red_flag"] = True

        # --- Clarify loop-breaker ---
        if action == "clarify" and state and state.last_action == "clarify":
            if state.last_topic == "injury":
                action = "injury_triage"
            else:
                action = "answer_question"
                if interp_parsed:
                    interp_parsed["action"] = "answer_question"
                    interp_parsed["question"] = user_message

        # =====================================================================
        # Step 3: Build response (deterministic + explainer)
        # =====================================================================

        engine_payload = {}
        final_response = None
        explainer_text = None
        needs_explainer = False

        # ---- INJURY TRIAGE (fully deterministic) ----
        if action == "injury_triage":
            if state:
                if not state.triage:
                    state.triage = InjuryTriage()

                # Extract pain info from current message
                pain_info = extract_pain_info(user_message)
                state.triage.update(pain_info)

                # Also extract from interpreter if it provided triage fields
                if interp_parsed and "injury_triage" in interp_parsed:
                    state.triage.update(interp_parsed["injury_triage"])

                # Red flag → emergency message
                if state.triage.is_red_flag():
                    final_response = (
                        "Stop training immediately and see a doctor. "
                        "What you're describing could be serious — "
                        "don't push through this. Get checked first."
                    )
                    engine_payload = {
                        "action": "injury_triage",
                        "red_flag": True,
                        "message": final_response,
                    }
                else:
                    # Get next triage question or recommendation
                    next_q = state.triage.next_question()
                    if next_q:
                        final_response = next_q
                        state.pending_question = next_q
                        engine_payload = {
                            "action": "injury_triage",
                            "stage": "collecting",
                            "collected": state.triage.data,
                            "question": next_q,
                        }
                    else:
                        # Triage complete → recommendation
                        recommendation = state.triage.get_recommendation()
                        engine_payload = {
                            "action": "injury_triage",
                            "stage": "recommending",
                            "collected": state.triage.data,
                            "message": recommendation,
                        }
                        # Use explainer to verbalize warmly
                        needs_explainer = True
            else:
                # No state (single-message mode) — basic response
                final_response = (
                    "Pain during training needs attention. "
                    "Can you tell me where it hurts and how bad it is (0-10)?"
                )

        # ---- CREATE WORKOUT ----
        elif action == "create_workout":
            needs_explainer = True
            sport = interp_parsed.get("sport", "") if interp_parsed else ""
            time_min = ""
            if interp_parsed:
                time_min = interp_parsed.get("time_available_min") or interp_parsed.get("time", "")
            goal = interp_parsed.get("goal", "") if interp_parsed else ""
            engine_payload = {
                "action": "create_workout",
                "sport": sport,
                "time_min": time_min,
                "goal": goal,
                "message": f"Building a {time_min}min {sport} session focused on {goal}.",
            }

        # ---- REPLAN ----
        elif action == "replan":
            needs_explainer = True
            replan_type = interp_parsed.get("replan_type", "general") if interp_parsed else "general"
            if replan_type == "illness":
                msg = "Rest is the priority right now. Your plan will adjust — focus on getting better."
            elif replan_type == "skip_today":
                msg = "No problem — life happens. Your plan will adapt around today."
            else:
                msg = "Got it. Your plan will adjust to accommodate this change."
            engine_payload = {
                "action": "replan",
                "replan_type": replan_type,
                "message": msg,
            }

        # ---- EXPLAIN / ANSWER_QUESTION ----
        elif action in ("explain", "answer_question"):
            needs_explainer = True
            question = user_message
            if interp_parsed and interp_parsed.get("question"):
                question = interp_parsed["question"]
            engine_payload = {
                "action": action,
                "topic": state.last_topic if state else None,
                "question": question,
                "message": f"Answer the athlete's question: {question}",
            }

        # ---- CLARIFY ----
        elif action == "clarify":
            if interp_parsed:
                clarify_msg = interp_parsed.get("clarify_message")
                if not clarify_msg:
                    missing = interp_parsed.get("missing", [])
                    if "medical_clearance" in missing:
                        clarify_msg = (
                            "That sounds like it could be serious. "
                            "Please stop training and see a doctor before continuing."
                        )
                    elif "sport" in missing:
                        clarify_msg = (
                            "What sport would you like to train? "
                            "Running, cycling, strength, or something else?"
                        )
                    elif "pain_location" in missing:
                        clarify_msg = "Where exactly does it hurt, and when did it start?"
                    else:
                        clarify_msg = "Can you give me a bit more detail so I can help?"
                final_response = clarify_msg
                if state:
                    state.pending_question = clarify_msg
            else:
                final_response = "Can you give me a bit more detail so I can help?"

        # ---- ACK (acknowledgement without clear intent) ----
        elif action == "ack" or (action in ("unknown", "parse_error") and not is_injury):
            final_response = "What would you like to do — a workout, plan change, or have a question?"
        else:
            final_response = "What would you like to do — a workout, plan change, or have a question?"

        # =====================================================================
        # Step 4: Cross-Domain Enrichment — fire rules, enrich payload
        # =====================================================================

        if state and engine_payload:
            # Enrich payload with cross-domain intelligence
            engine_payload = enrich_payload(
                engine_payload=engine_payload,
                store=state.athlete_store,
                interp_parsed=interp_parsed,
                conversation_state=state,
            )

            # Log fired rules
            if engine_payload.get("cross_domain"):
                rules = engine_payload.get("cross_domain_rules", [])
                print(f"    Cross-domain: {len(rules)} rule(s) fired: {rules}")

        # =====================================================================
        # Step 5: Explainer (if needed) — verbalizes ENGINE_PAYLOAD
        # =====================================================================

        if needs_explainer and engine_payload:
            clean_history = state.get_clean_history() if state else None
            explainer_text = self.run_explainer(
                full_message,
                engine_payload=engine_payload,
                history=clean_history,
            )
            final_response = explainer_text

        # =====================================================================
        # Step 6: Track state + update cross-domain store
        # =====================================================================

        if state:
            state.last_action = action
            if final_response:
                state.add_assistant(final_response)
            # Clear triage if we moved past it
            if state.triage and state.triage.stage == "referred":
                state.triage = None
            # Clear pending question if this turn answered something
            if action not in ("clarify", "injury_triage"):
                state.pending_question = None

            # Update cross-domain state store from this turn
            triage_data = None
            if state.triage:
                triage_data = state.triage.data

            readiness_level = None
            readiness_state_str = None
            if context:
                readiness_str = context.get("readiness", "")
                # Parse "Green (Recovered)" → level="Green", state="Recovered"
                if readiness_str:
                    parts = readiness_str.split("(")
                    readiness_level = parts[0].strip()
                    if len(parts) > 1:
                        readiness_state_str = parts[1].rstrip(")")

            update_state_from_turn(
                store=state.athlete_store,
                interp_parsed=interp_parsed,
                triage_data=triage_data,
                user_message=user_message,
                readiness_level=readiness_level,
                readiness_state=readiness_state_str,
            )

        return {
            "user_message": user_message,
            "full_message": full_message,
            "interpreter_raw": interp_raw,
            "interpreter_parsed": interp_parsed,
            "action": action,
            "needs_explainer": needs_explainer,
            "explainer_text": explainer_text,
            "final_response": final_response,
            "engine_payload": engine_payload,
        }


# =============================================================================
# TEST SCENARIOS
# =============================================================================

SCENARIOS = [
    {
        "name": "Workout request with sport + time",
        "message": "Give me a 45-minute run",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "intermediate",
        },
        "expect_action": "create_workout",
    },
    {
        "name": "Explain today's session",
        "message": "What am I doing today?",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "intermediate",
            "session": "Z2 60min 'Easy aerobic' (base phase)",
        },
        "expect_action": "explain",
    },
    {
        "name": "General knowledge question",
        "message": "What is Zone 2 and why does everyone talk about it?",
        "context": {
            "readiness": "Yellow (Productive)",
            "sport": "cycling",
            "level": "beginner",
        },
        "expect_action": "answer_question",
    },
    {
        "name": "Fatigue / tired athlete",
        "message": "I'm exhausted but I don't want to skip my workout",
        "context": {
            "readiness": "Yellow (Productive)",
            "sport": "running",
            "level": "intermediate",
        },
        "expect_action": "explain",
    },
    {
        "name": "Skip today (replan)",
        "message": "I need to skip today, I have a work meeting",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "intermediate",
            "session": "Z4 45min 'Threshold intervals'",
        },
        "expect_action": "replan",
    },
    {
        "name": "Illness (should be replan, NOT medical)",
        "message": "I have the flu, should I still train?",
        "context": {
            "readiness": "Orange (Accumulated)",
            "sport": "running",
            "level": "intermediate",
        },
        "expect_action": "replan",
    },
    {
        "name": "Medical red flag (chest pain)",
        "message": "I'm having chest pain when I run",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "intermediate",
        },
        "expect_action": "injury_triage",
    },
    {
        "name": "Ambiguous request (no sport, no context)",
        "message": "Give me a workout",
        "context": None,
        "expect_action": "clarify",
    },
    {
        "name": "Motivation / encouragement",
        "message": "I missed a week of training. Have I ruined everything?",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "beginner",
        },
        "expect_action": "answer_question",
    },
    {
        "name": "Unrealistic goal pushback",
        "message": "I want to run a marathon in 3 weeks. I ran once last month.",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "beginner",
        },
        "expect_action": "answer_question",
    },
    {
        "name": "Knee pain (injury triage, NOT medical emergency)",
        "message": "Can I do a workout if I have pain in my knee?",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "running",
            "level": "intermediate",
        },
        "expect_action": "injury_triage",
    },
    {
        "name": "Elbow pain from cycling grip",
        "message": "My elbow hurts when I grip the handlebars on my bike",
        "context": {
            "readiness": "Green (Recovered)",
            "sport": "cycling",
            "level": "intermediate",
        },
        "expect_action": "injury_triage",
    },
]


# =============================================================================
# DISPLAY
# =============================================================================

def print_header():
    print("=" * 64)
    print("  MiValta Josi v5 — Real-World Simulation")
    print(f"  Date: {date.today()}")
    print("=" * 64)


def print_result(result: dict, scenario_name: str = "", expect_action: str = ""):
    """Pretty-print a simulation result."""
    print(f"\n{'─' * 64}")
    if scenario_name:
        print(f"  Scenario: {scenario_name}")
    print(f"  Athlete:  \"{result['user_message']}\"")
    print(f"{'─' * 64}")

    action = result["action"]
    parsed = result["interpreter_parsed"]
    action_match = ""
    if expect_action:
        ok = action == expect_action
        action_match = f"  {'PASS' if ok else 'FAIL'} (expected: {expect_action})"

    print(f"\n  [Interpreter] action={action}{action_match}")
    if parsed:
        display = {k: v for k, v in parsed.items()
                   if k not in ("free_text",) and v is not None}
        print(f"    {json.dumps(display, indent=None)}")
    else:
        print(f"    Raw: {result['interpreter_raw'][:200]}")

    # Show engine payload if present
    payload = result.get("engine_payload")
    if payload:
        # Show base payload (without cross-domain for cleaner display)
        display_payload = {k: v for k, v in payload.items()
                          if k not in ("cross_domain", "cross_domain_actions",
                                       "cross_domain_rules", "athlete_context")}
        print(f"\n  [Engine] {json.dumps(display_payload, indent=None)[:200]}")

        # Show cross-domain insights separately
        if payload.get("cross_domain"):
            print(f"\n  [Cross-Domain Intelligence]")
            for msg in payload["cross_domain"]:
                print(f"    → {msg[:120]}")

    if result["needs_explainer"]:
        print(f"\n  [Router] → Explainer needed (action={action})")
    else:
        print(f"\n  [Router] → Deterministic response (action={action})")

    print(f"\n  [Josi says]:")
    response = result["final_response"] or ""
    words = response.split()
    line = "    "
    for w in words:
        if len(line) + len(w) + 1 > 68:
            print(line)
            line = "    " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        print(line)

    return action == expect_action if expect_action else None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MiValta Josi v5 — Real-World Simulation"
    )

    parser.add_argument("--interpreter", type=str,
                        help="Path to interpreter GGUF model")
    parser.add_argument("--explainer", type=str,
                        help="Path to explainer GGUF model")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
                        help=f"Directory for model files (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--download", action="store_true",
                        help="Download models from server before running")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scenarios", action="store_true",
                      help="Run built-in test scenarios")
    mode.add_argument("--interactive", action="store_true",
                      help="Interactive chat mode")
    mode.add_argument("--message", type=str,
                      help="Run a single message")

    parser.add_argument("--sport", type=str, default="running",
                        help="Default sport for --message/--interactive (default: running)")
    parser.add_argument("--readiness", type=str, default="Green (Recovered)",
                        help="Default readiness (default: Green (Recovered))")
    parser.add_argument("--level", type=str, default="intermediate",
                        help="Default level (default: intermediate)")

    args = parser.parse_args()

    print_header()

    # Resolve model paths
    model_dir = Path(args.model_dir)
    if args.download:
        interp_path, expl_path = download_models(model_dir)
    else:
        interp_path = Path(args.interpreter) if args.interpreter else model_dir / INTERPRETER_FILENAME
        expl_path = Path(args.explainer) if args.explainer else model_dir / EXPLAINER_FILENAME

    if not interp_path.exists():
        print(f"\n  ERROR: Interpreter model not found: {interp_path}")
        print(f"  Run with --download to fetch from server, or provide --interpreter path")
        sys.exit(1)
    if not expl_path.exists():
        print(f"\n  ERROR: Explainer model not found: {expl_path}")
        print(f"  Run with --download to fetch from server, or provide --explainer path")
        sys.exit(1)

    # Load models
    try:
        engine = JosiEngine(str(interp_path), str(expl_path))
    except ImportError:
        print("\n  ERROR: llama-cpp-python not installed.")
        print("  Install with: pip install llama-cpp-python")
        sys.exit(1)

    # ─── Scenarios mode ───
    if args.scenarios:
        print(f"\n  Running {len(SCENARIOS)} scenarios...\n")
        passed = 0
        total = 0

        for scenario in SCENARIOS:
            result = engine.simulate(
                scenario["message"],
                context=scenario.get("context"),
            )
            ok = print_result(
                result,
                scenario_name=scenario["name"],
                expect_action=scenario.get("expect_action", ""),
            )
            if ok is not None:
                total += 1
                if ok:
                    passed += 1

        print(f"\n{'=' * 64}")
        print(f"  Results: {passed}/{total} action matches")
        print(f"{'=' * 64}")

    # ─── Interactive mode ───
    elif args.interactive:
        default_ctx = {
            "readiness": args.readiness,
            "sport": args.sport,
            "level": args.level,
        }
        conv_state = ConversationState(max_turns=6, athlete_id="interactive_user")

        print(f"\n  Interactive mode — type 'quit' to exit")
        print(f"  Context: {args.sport}, {args.readiness}, {args.level}")
        print(f"  Cross-domain knowledge: ENABLED")
        print(f"  Commands: /sport, /readiness, /level, /reset, /state (show cross-domain)\n")

        while True:
            try:
                user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "/quit"):
                break
            if user_input.startswith("/sport "):
                default_ctx["sport"] = user_input[7:].strip()
                print(f"    → Sport set to: {default_ctx['sport']}")
                continue
            if user_input.startswith("/readiness "):
                default_ctx["readiness"] = user_input[11:].strip()
                print(f"    → Readiness set to: {default_ctx['readiness']}")
                continue
            if user_input.startswith("/level "):
                default_ctx["level"] = user_input[7:].strip()
                print(f"    → Level set to: {default_ctx['level']}")
                continue
            if user_input.lower() in ("/reset", "/clear"):
                conv_state = ConversationState(max_turns=6, athlete_id="interactive_user")
                print(f"    → Conversation history and cross-domain state cleared")
                continue
            if user_input.lower() == "/state":
                store = conv_state.athlete_store
                cross = store.serialize_for_prompt()
                if cross:
                    print(f"    {cross}")
                else:
                    print(f"    → No cross-domain state yet (interact to build it)")
                if store.active_injuries():
                    for inj in store.active_injuries():
                        print(f"    Injury: {inj['area']} ({inj['severity']}/10) "
                              f"x{inj.get('occurrences', 1)} [{inj['status']}]")
                if store.correlations:
                    for c in store.correlations:
                        print(f"    Correlation: {c['trigger']} → {c['result']} "
                              f"(x{c['occurrences']}, conf={c['confidence']:.1f})")
                if store.preferences:
                    print(f"    Preferences: {json.dumps(store.preferences)}")
                continue

            result = engine.simulate(user_input, context=default_ctx, state=conv_state)
            if conv_state.last_topic:
                print(f"    [topic: {conv_state.last_topic}]")
            if conv_state.triage:
                triage = conv_state.triage
                collected = list(triage.data.keys())
                print(f"    [triage: stage={triage.stage}, collected={collected}]")
            # Show cross-domain rules that fired
            if result.get("engine_payload", {}).get("cross_domain"):
                rules = result["engine_payload"].get("cross_domain_rules", [])
                print(f"    [cross-domain: {', '.join(rules)}]")
            print_result(result)

    # ─── Single message ───
    elif args.message:
        ctx = {
            "readiness": args.readiness,
            "sport": args.sport,
            "level": args.level,
        }
        result = engine.simulate(args.message, context=ctx)
        print_result(result)


if __name__ == "__main__":
    main()
