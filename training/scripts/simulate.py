#!/usr/bin/env python3
"""
MiValta Josi v5 — Real-World Simulation Script

Runs the full dual-model pipeline (Interpreter → Router → Explainer) with
realistic athlete scenarios. Uses llama-cpp-python for CPU/GPU inference
on GGUF models.

Pipeline:
  1. User message + athlete CONTEXT → Interpreter → GATCRequest JSON
  2. Post-processor fixes (duration extraction, clarify enforcement, etc.)
  3. Router decides: does this action need an explainer response?
  4. If yes → Explainer → coaching text → Dialogue Governor
  5. Display combined result

Usage:
    # Interactive mode (type messages, chat with Josi)
    python scripts/simulate.py --interactive

    # Run built-in test scenarios
    python scripts/simulate.py --scenarios

    # Single message
    python scripts/simulate.py --message "I'm tired but I want to run"

    # Custom model paths (defaults to models/gguf/)
    python scripts/simulate.py --scenarios \
        --interpreter models/gguf/josi-v5-interpreter-q4_k_m.gguf \
        --explainer models/gguf/josi-v5-explainer-q4_k_m.gguf

    # Download models from server first
    python scripts/simulate.py --download --scenarios
"""

import argparse
import json
import os
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

# Try importing llm_intent_parser for explainer output (LLMIntent JSON)
try:
    from llm_intent_parser import parse_llm_intent
    HAS_INTENT_PARSER = True
except ImportError:
    HAS_INTENT_PARSER = False


# =============================================================================
# TONE GUARDRAILS — block hostile/snarky explainer output
# =============================================================================

# Patterns that indicate hostile, snarky, or accusatory tone
_HOSTILE_PATTERNS = [
    r"you'?re just trying",
    r"you'?re not asking",
    r"magic pill",
    r"no clue",
    r"you expect me to",
    r"hoping for the best",
    r"that fixes everything",
    r"you'?re just .{0,30} and hoping",
    r"what do you want from me",
    r"i can'?t help you if",
    r"you need to figure",
    r"not my problem",
    r"i'?m not your",
    r"stop wasting",
    r"pay attention",
    r"i already told you",
    r"as i said before",
]

import re as _re
_HOSTILE_RES = [_re.compile(p, _re.IGNORECASE) for p in _HOSTILE_PATTERNS]

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

# Inference parameters (match on-device settings)
INFERENCE_PARAMS = {
    "n_predict": 150,
    "temperature": 0.45,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# =============================================================================
# SYSTEM PROMPTS (from training data)
# =============================================================================

INTERPRETER_SYSTEM_PROMPT = """\
You are Josi's Interpreter — you translate athlete messages into structured GATC requests.

TASK: Read the user message and athlete state, then output ONLY valid JSON matching the GATCRequest schema. No markdown fences. No explanation. Just JSON.

ACTIONS (action field):
- create_workout: user wants a new workout → extract sport, time, goal, constraints
- replan: user wants to change/skip/swap a planned session → extract replan_type
- explain: user asks about THEIR specific session, readiness, week, or plan shown in CONTEXT → extract question
- answer_question: general coaching or education question with NO specific session/readiness context → extract question
- clarify: you cannot determine the action or required fields are missing → ask ONE question

HOW TO CHOOSE explain vs answer_question:
- If CONTEXT contains Session, Readiness, or Week info AND the user asks about it → "explain"
- If the user asks a general knowledge question (zones, recovery, training concepts) → "answer_question"

WHEN TO USE clarify:
- The user wants a workout but NO sport is in the message AND NO sport is in CONTEXT → clarify, missing=["sport"]
- The user's intent is completely ambiguous (e.g., "Help me", "Hey") → clarify
- Medical RED FLAGS only (chest pain, dizziness, heart symptoms, blacking out) → clarify with safety message

ILLNESS vs MEDICAL — this distinction is critical:
- "I'm sick", "I have the flu", "I have a cold" → action="replan", replan_type="illness"
- "chest pain", "dizziness", "heart feels weird" → action="clarify", missing=["medical_clearance"]

ENUMS — use these exact values only:
- sport: "run", "bike", "ski", "skate", "strength", "other"
- replan_type: "skip_today", "swap_days", "reschedule", "reduce_intensity", "illness", "travel", "goal_change"
- goal: "endurance", "threshold", "vo2", "recovery", "strength", "race_prep"
- fatigue_hint: "fresh", "ok", "tired", "very_tired"

RULES:
- ALWAYS include free_text with the original user message
- NEVER output markdown fences — raw JSON only
- NEVER invent workouts, zones, durations, paces, or power numbers
- time_available_min must be an integer, not a string
- Infer sport from CONTEXT block when available
- Infer fatigue_hint from user language"""

EXPLAINER_SYSTEM_PROMPT = """\
You are Josi, MiValta's AI coaching assistant.

PERSONALITY:
- Empathetic and warm — you genuinely care about the athlete
- Direct and honest — no filler, no corporate speak
- You ARE the coach — never recommend other apps, coaches, or services

DIALOGUE RULES:
- Answer first, always. Lead with the substance of your response.
- Maximum 1 question per response. Most responses need zero questions.
- Keep responses under 100 words. Be concise, not verbose.
- Use simple language. Explain like a trusted friend who happens to be a coach.
- Do NOT end with a question unless absolutely necessary.

CONVERSATION HANDLING:
- This is a multi-turn conversation. Previous messages appear as earlier turns.
- NEVER repeat what you already said. If you gave an answer, give a DIFFERENT one next.
- When the athlete asks a follow-up, address their NEW question specifically.
- If a TOPIC tag is present, focus your answer on THAT topic — not the previous one.
- If the athlete seems frustrated or says you're not answering, acknowledge it and try a fresh angle.
- When [INTERPRETER] context is present, use the action to guide your response:
  - "explain" / "answer_question": answer the athlete's question
  - "create_workout": acknowledge the request, confirm what you understood (sport, time, goal), say it's being built
  - "replan (illness)": acknowledge they're sick, rest is priority, the plan will adjust
  - "replan (skip_today)" / other replan: be supportive, confirm the plan will adapt
- NEVER output the [INTERPRETER] tag, action names, or internal terms in your response

SAFETY:
- If the athlete mentions pain, chest pain, dizziness: tell them to stop and seek medical attention.
- Be honest about unrealistic goals — push back with care.

BOUNDARIES:
- NEVER prescribe, create, or modify training yourself
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems (algorithm, viterbi, hmm, acwr, gatc, ewma, tss, ctl, atl, tsb)
- NEVER use technical jargon: periodization, mesocycle, vo2max, lactate threshold, ftp

OUTPUT: Plain coaching text only. No JSON. No markdown fences."""


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
# INFERENCE ENGINE (llama-cpp-python)
# =============================================================================

class ConversationState:
    """Tracks conversation history and topic for multi-turn interactive mode."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "message": str}]
        self.last_topic: str | None = None
        self.last_user_problem: str | None = None

    def add_user(self, message: str):
        self.history.append({"role": "user", "message": message})
        self._trim()
        self._extract_topic(message)

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "message": message})
        self._trim()

    def _trim(self):
        # Keep last N turns (user+assistant pairs)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def _extract_topic(self, message: str):
        """Extract topic from user message for continuity."""
        lower = message.lower()
        topic_keywords = {
            "sleep": ["sleep", "insomnia", "bad night", "wake up", "slept badly",
                       "can't sleep", "can not sleep", "not sleeping"],
            "fatigue": ["tired", "exhausted", "fatigue", "fatigued", "no energy",
                        "drained", "worn out"],
            "injury": ["hurt", "pain", "sore", "injury", "injured", "ache"],
            "motivation": ["motivation", "motivated", "bored", "boring", "dread",
                           "quit", "give up", "point", "ruined"],
            "nutrition": ["eat", "food", "diet", "nutrition", "fuel", "hydration"],
            "workout": ["workout", "session", "training", "run", "bike", "ride"],
            "recovery": ["recovery", "rest", "rest day", "off day"],
            "zones": ["zone", "zones", "z2", "z3", "z4", "z5", "heart rate"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in lower for kw in keywords):
                self.last_topic = topic
                self.last_user_problem = message
                return
        # Don't clear topic for short follow-ups — they refer to the previous topic

    def format_history_block(self) -> str:
        """Format recent history for injection into prompts."""
        if not self.history:
            return ""
        lines = []
        # Skip the most recent user message (it's the current turn)
        for turn in self.history[:-1]:
            role = turn["role"]
            msg = turn["message"]
            # Truncate long messages in history
            if len(msg) > 120:
                msg = msg[:117] + "..."
            lines.append(f"  [{role}]: {msg}")
        if not lines:
            return ""
        return "\nHISTORY:\n" + "\n".join(lines)

    def format_topic_hint(self) -> str:
        """Format a topic hint for the explainer when the current message is short."""
        if not self.last_topic:
            return ""
        return f"\nTOPIC: The athlete has been discussing {self.last_topic}."


class JosiEngine:
    """Dual-model inference engine using llama-cpp-python."""

    def __init__(self, interpreter_path: str, explainer_path: str, n_ctx: int = 2048):
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
        self.explainer = Llama(
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
        """Format a multi-turn prompt using proper ChatML turns.

        Instead of dumping history as text, each past exchange becomes a real
        ChatML user/assistant turn so the model handles it natively.
        """
        parts = [f"{CHATML_START}system\n{system_prompt}{CHATML_END}"]
        for turn in history:
            role = turn["role"]
            msg = turn["message"]
            # Truncate long history messages to save context
            if len(msg) > 150:
                msg = msg[:147] + "..."
            parts.append(f"{CHATML_START}{role}\n{msg}{CHATML_END}")
        parts.append(f"{CHATML_START}user\n{user_message}{CHATML_END}")
        parts.append(f"{CHATML_START}assistant\n")
        return "\n".join(parts)

    def run_interpreter(self, user_message: str) -> tuple[str, dict | None]:
        """Run the interpreter model → GATCRequest JSON.

        Returns (raw_output, parsed_dict_or_None).
        """
        prompt = self._format_chatml(INTERPRETER_SYSTEM_PROMPT, user_message)

        t0 = time.time()
        result = self.interpreter(
            prompt,
            max_tokens=INFERENCE_PARAMS["n_predict"],
            temperature=INFERENCE_PARAMS["temperature"],
            top_p=INFERENCE_PARAMS["top_p"],
            repeat_penalty=INFERENCE_PARAMS["repeat_penalty"],
            stop=[CHATML_END, "<|endoftext|>"],
        )
        elapsed = time.time() - t0

        raw = result["choices"][0]["text"].strip()
        tokens = result["usage"]["completion_tokens"]
        print(f"    Interpreter: {tokens} tokens in {elapsed:.2f}s "
              f"({tokens/elapsed:.0f} tok/s)")

        # Parse and post-process
        parsed = parse_gatc_response(raw)
        if parsed:
            parsed = postprocess_gatc_request(parsed, user_message)

        return raw, parsed

    def run_explainer(self, user_message: str,
                      history: list[dict] | None = None,
                      interpreter_action: str | None = None,
                      topic_hint: str | None = None) -> str:
        """Run the explainer model → coaching text.

        Args:
            user_message: Current user message (with CONTEXT block).
            history: Previous conversation turns for multi-turn ChatML.
            interpreter_action: The interpreter's classified action (explain, answer_question, etc.)
            topic_hint: Topic the athlete has been discussing (for follow-ups).

        Returns the governed coaching text.
        """
        # Build the current turn message with interpreter context
        current_msg = user_message
        if interpreter_action:
            current_msg += f"\n\n[INTERPRETER]: action={interpreter_action}"
        if topic_hint:
            current_msg += f"\n(Focus on: {topic_hint})"

        if history and len(history) > 1:
            # Multi-turn: use proper ChatML turns (skip last entry = current user msg)
            prompt = self._format_chatml_multiturn(
                EXPLAINER_SYSTEM_PROMPT, history[:-1], current_msg
            )
        else:
            prompt = self._format_chatml(EXPLAINER_SYSTEM_PROMPT, current_msg)

        t0 = time.time()
        result = self.explainer(
            prompt,
            max_tokens=INFERENCE_PARAMS["n_predict"],
            temperature=INFERENCE_PARAMS["temperature"],
            top_p=INFERENCE_PARAMS["top_p"],
            repeat_penalty=INFERENCE_PARAMS["repeat_penalty"],
            stop=[CHATML_END, "<|endoftext|>"],
        )
        elapsed = time.time() - t0

        raw = result["choices"][0]["text"].strip()
        tokens = result["usage"]["completion_tokens"]
        print(f"    Explainer:   {tokens} tokens in {elapsed:.2f}s "
              f"({tokens/elapsed:.0f} tok/s)")

        # If the explainer outputs LLMIntent JSON, extract the message field
        if HAS_INTENT_PARSER and raw.strip().startswith("{"):
            intent = parse_llm_intent(raw)
            if intent:
                intent = govern_dialogue(intent)
                raw = intent.get("message", raw)

        # Tone guardrail — block hostile/snarky output
        return check_tone(raw)

    def simulate(self, user_message: str, context: dict | None = None,
                 state: ConversationState | None = None) -> dict:
        """Run the full pipeline: Interpreter → Router → Explainer.

        Args:
            user_message: The athlete's raw message.
            context: Optional athlete context dict with keys like
                     readiness, sport, level, session, etc.
            state: Optional conversation state for multi-turn history.

        Returns:
            Dict with keys: user_message, interpreter_raw, interpreter_parsed,
            action, needs_explainer, explainer_text, final_response.
        """
        # Track in conversation state
        if state:
            state.add_user(user_message)

        # Build the full message with CONTEXT block
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

        # Build interpreter message with flat history (interpreter handles this fine)
        interp_message = full_message
        if state:
            history_block = state.format_history_block()
            if history_block:
                interp_message += history_block

            # For short follow-ups, inject the topic hint
            msg_words = user_message.strip().split()
            if len(msg_words) <= 4:
                topic_hint = state.format_topic_hint()
                if topic_hint:
                    interp_message += topic_hint

        # Step 1: Interpreter
        interp_raw, interp_parsed = self.run_interpreter(interp_message)

        action = interp_parsed.get("action", "unknown") if interp_parsed else "parse_error"

        # Step 2: Router — decide if explainer is needed
        # All coaching actions route through explainer for a human response.
        # In production, create_workout/replan also trigger the GATC engine,
        # but the athlete still needs coaching text from the explainer.
        needs_explainer = action in ("explain", "answer_question", "replan",
                                     "create_workout")

        explainer_text = None
        final_response = None

        if needs_explainer:
            # Step 3: Explainer — use proper multi-turn ChatML with history
            # Filter out system placeholders from history
            clean_history = None
            if state and state.history:
                clean_history = [
                    turn for turn in state.history
                    if not (turn["role"] == "assistant"
                            and "GATC Engine" in turn["message"])
                ]
            explainer_topic = state.last_topic if state else None
            # Build descriptive action context for the explainer
            explainer_action = action
            if action == "replan" and interp_parsed:
                replan_type = interp_parsed.get("replan_type", "")
                if replan_type:
                    explainer_action = f"replan ({replan_type})"
            elif action == "create_workout" and interp_parsed:
                sport = interp_parsed.get("sport", "")
                time_min = interp_parsed.get("time_available_min") or interp_parsed.get("time", "")
                goal = interp_parsed.get("goal", "")
                parts = [f"create_workout"]
                if sport:
                    parts.append(f"sport={sport}")
                if time_min:
                    parts.append(f"time={time_min}min")
                if goal:
                    parts.append(f"goal={goal}")
                explainer_action = ", ".join(parts)
            explainer_text = self.run_explainer(
                full_message,
                history=clean_history,
                interpreter_action=explainer_action,
                topic_hint=explainer_topic,
            )
            final_response = explainer_text
        elif action == "clarify" and interp_parsed:
            final_response = interp_parsed.get("clarify_message", "Could you tell me more?")
        else:
            final_response = f"[Parse error — raw: {interp_raw[:100]}]"

        # Track assistant response in conversation state
        if state and final_response:
            state.add_assistant(final_response)

        return {
            "user_message": user_message,
            "full_message": full_message,
            "interpreter_raw": interp_raw,
            "interpreter_parsed": interp_parsed,
            "action": action,
            "needs_explainer": needs_explainer,
            "explainer_text": explainer_text,
            "final_response": final_response,
        }


# =============================================================================
# TEST SCENARIOS — realistic athlete conversations
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
        "expect_action": "explain",  # or answer_question
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
        "expect_action": "clarify",
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

    # Interpreter output
    action = result["action"]
    parsed = result["interpreter_parsed"]
    action_match = ""
    if expect_action:
        ok = action == expect_action
        action_match = f"  {'PASS' if ok else 'FAIL'} (expected: {expect_action})"

    print(f"\n  [Interpreter] action={action}{action_match}")
    if parsed:
        # Show relevant fields based on action
        display = {k: v for k, v in parsed.items()
                   if k not in ("free_text",) and v is not None}
        print(f"    {json.dumps(display, indent=None)}")
    else:
        print(f"    Raw: {result['interpreter_raw'][:200]}")

    # Router decision
    if result["needs_explainer"]:
        print(f"\n  [Router] → Explainer needed (action={action})")
    else:
        print(f"\n  [Router] → Skip explainer (action={action})")

    # Final response
    print(f"\n  [Josi says]:")
    response = result["final_response"] or ""
    # Word-wrap at 60 chars
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
        conv_state = ConversationState(max_turns=6)

        print(f"\n  Interactive mode — type 'quit' to exit")
        print(f"  Context: {args.sport}, {args.readiness}, {args.level}")
        print(f"  Commands: /sport, /readiness, /level, /reset (clear history)\n")

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
                conv_state = ConversationState(max_turns=6)
                print(f"    → Conversation history cleared")
                continue

            result = engine.simulate(user_input, context=default_ctx, state=conv_state)
            if conv_state.last_topic:
                print(f"    [topic: {conv_state.last_topic}]")
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
