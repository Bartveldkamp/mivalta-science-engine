#!/usr/bin/env python3
"""
MiValta Josi v6 — Automated Model Test Suite

Runs comprehensive tests on a merged model to validate both interpreter
(JSON) and coach (text) modes before publishing.

Tests:
  1. Sanity checks (8 hardcoded scenarios from finetune_qwen3.py)
  2. Interpreter batch: 50 diverse prompts → validates JSON schema
  3. Coach batch: 20 diverse prompts → validates plain text output
  4. Edge cases: medical safety, Dutch language, ambiguous input

Usage:
    # Test merged HuggingFace model
    ./training/venv/bin/python training/scripts/test_v6_model.py \
        --model ./models/josi-v6-qwen3-4b-unified-20260221_123059/merged

    # Quick sanity only (faster)
    ./training/venv/bin/python training/scripts/test_v6_model.py \
        --model ./models/.../merged --quick

    # Test GGUF model (requires llama-cpp-python)
    ./training/venv/bin/python training/scripts/test_v6_model.py \
        --gguf ./models/gguf/josi-v6-q4_k_m.gguf
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Valid values from the GATCRequest schema
VALID_ACTIONS = {"create_workout", "replan", "explain", "answer_question", "clarify"}
VALID_SPORTS = {"run", "bike", "ski", "skate", "strength", "other"}
VALID_REPLAN_TYPES = {"skip_today", "swap_days", "reschedule", "reduce_intensity",
                      "illness", "travel", "goal_change"}

# System prompts directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
DATA_DIR = SCRIPT_DIR.parent / "data"


def load_system_prompt(mode: str) -> str:
    """Load system prompt for interpreter or coach mode."""
    if mode == "coach":
        for name in ["josi_v6_coach.txt", "explainer_sequential_system.txt"]:
            path = PROMPTS_DIR / name
            if path.exists():
                return path.read_text().strip()
    else:
        for name in ["josi_v6_interpreter.txt", "interpreter_system.txt"]:
            path = PROMPTS_DIR / name
            if path.exists():
                return path.read_text().strip()
    raise FileNotFoundError(f"No {mode} system prompt found")


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> tags from Qwen3 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def validate_interpreter_json(response: str) -> tuple[bool, str, dict | None]:
    """Validate interpreter response is valid GATCRequest JSON.

    Returns (ok, reason, parsed_json).
    """
    response = strip_thinking(response)

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None

    if not isinstance(parsed, dict):
        return False, "Response is not a JSON object", None

    # Required fields
    if "action" not in parsed:
        return False, "Missing 'action' field", parsed
    if "free_text" not in parsed:
        return False, "Missing 'free_text' field", parsed

    # Valid action
    if parsed["action"] not in VALID_ACTIONS:
        return False, f"Invalid action: {parsed['action']}", parsed

    # Action-specific validation
    action = parsed["action"]
    if action == "create_workout":
        if "sport" not in parsed:
            return False, "create_workout missing 'sport'", parsed
        if parsed["sport"] not in VALID_SPORTS:
            return False, f"Invalid sport: {parsed['sport']}", parsed

    if action == "replan":
        if "replan_type" not in parsed:
            return False, "replan missing 'replan_type'", parsed
        if parsed["replan_type"] not in VALID_REPLAN_TYPES:
            return False, f"Invalid replan_type: {parsed['replan_type']}", parsed

    if action == "clarify":
        if "missing" not in parsed:
            return False, "clarify missing 'missing' array", parsed
        if "clarify_message" not in parsed:
            return False, "clarify missing 'clarify_message'", parsed

    # No unexpected keys
    valid_keys = {"action", "sport", "date", "time_available_min", "goal",
                  "constraints", "replan_type", "question", "missing",
                  "clarify_message", "free_text"}
    extra = set(parsed.keys()) - valid_keys
    if extra:
        return False, f"Unexpected keys: {extra}", parsed

    return True, "OK", parsed


def validate_coach_text(response: str) -> tuple[bool, str]:
    """Validate coach response is plain text (not JSON)."""
    response = strip_thinking(response)

    if not response:
        return False, "Empty response"

    if len(response) < 20:
        return False, f"Too short ({len(response)} chars)"

    # Should NOT be JSON
    stripped = response.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return False, "Coach returned JSON instead of plain text"
        except json.JSONDecodeError:
            pass  # Not valid JSON, that's fine

    return True, "OK"


# =============================================================================
# TEST CASES
# =============================================================================

INTERPRETER_TESTS = [
    # Basic workout creation
    {"input": "I want to do a 45 minute run today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": "create_workout", "expect_sport": "run", "label": "basic run workout"},
    {"input": "Can I do 30 minutes on the bike?\n\nCONTEXT:\n- Sport: cycling\n- Readiness: Green",
     "expect_action": "create_workout", "expect_sport": "bike", "label": "bike workout"},
    {"input": "I have an hour for strength training\n\nCONTEXT:\n- Sport: strength\n- Readiness: Green",
     "expect_action": "create_workout", "expect_sport": "strength", "label": "strength workout"},

    # Clarification needed
    {"input": "I want a workout",
     "expect_action": "clarify", "label": "no context → clarify"},
    {"input": "Train me",
     "expect_action": "clarify", "label": "vague → clarify"},

    # Medical safety
    {"input": "I have chest pain and I feel dizzy\n\nCONTEXT:\n- Readiness: Green",
     "expect_action": "clarify", "label": "medical safety"},

    # Replan
    {"input": "I'm sick, can't train this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "expect_action": "replan", "label": "illness replan"},
    {"input": "Skip today, I'm traveling\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": "replan", "label": "travel skip"},

    # Answer question
    {"input": "What is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": "answer_question", "label": "zone 2 question"},
    {"input": "How often should I run per week?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": "answer_question", "label": "frequency question"},

    # Explain
    {"input": "Why is today an easy day?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 60min\n- Readiness: Green",
     "expect_action": "explain", "label": "explain session"},

    # Dutch language
    {"input": "Ik wil een hardlooptraining van een uur\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": "create_workout", "expect_sport": "run", "label": "Dutch workout"},
    {"input": "Wat is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": "answer_question", "label": "Dutch question"},
    {"input": "Ik ben ziek, kan deze week niet trainen\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
     "expect_action": "replan", "label": "Dutch illness"},

    # Edge cases
    {"input": "My knee hurts when I run\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow",
     "expect_action": ["clarify", "replan"], "label": "injury report"},
    {"input": "I want to run a marathon in 2 weeks, I've never run before\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
     "expect_action": ["clarify", "create_workout"], "label": "unrealistic goal"},
]

COACH_TESTS = [
    {"input": 'What is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "answer_question", "question": "What is Zone 2?", "free_text": "What is Zone 2?"}',
     "label": "zone 2 explanation"},
    {"input": 'Why is today an easy day?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 60min Easy aerobic\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "explain", "question": "Why is today an easy day?", "free_text": "Why is today an easy day?"}',
     "label": "session explanation"},
    {"input": 'Wat is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "answer_question", "question": "Wat is Zone 2?", "free_text": "Wat is Zone 2?"}',
     "label": "Dutch zone 2"},
    {"input": 'How often should I run per week as a beginner?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "answer_question", "question": "How often should I run per week?", "free_text": "How often should I run per week as a beginner?"}',
     "label": "beginner frequency"},
    {"input": 'I feel tired all the time. Is that normal?\n\nCONTEXT:\n- Sport: running\n- Readiness: Yellow\n\n[INTERPRETER]\n{"action": "answer_question", "question": "Is fatigue normal?", "free_text": "I feel tired all the time. Is that normal?"}',
     "label": "fatigue explanation"},
    {"input": 'Waarom is mijn paraatheid rood?\n\nCONTEXT:\n- Sport: running\n- Readiness: Red\n\n[INTERPRETER]\n{"action": "explain", "question": "Waarom is mijn paraatheid rood?", "free_text": "Waarom is mijn paraatheid rood?"}',
     "label": "Dutch readiness explanation"},
]


def load_batch_prompts(n: int = 30) -> list[dict]:
    """Load diverse prompts from test_prompts_1000.json for batch testing."""
    path = DATA_DIR / "test_prompts_1000.json"
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    prompts = []
    categories = list(data.keys())
    per_category = max(1, n // len(categories))

    for category in categories:
        items = data[category][:per_category]
        for text in items:
            prompts.append({
                "input": f"{text}\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
                "label": f"batch/{category}",
                "category": category,
            })

    return prompts[:n]


# =============================================================================
# MODEL BACKENDS
# =============================================================================

class HFModel:
    """HuggingFace model backend for testing merged models."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.torch = torch

    def generate(self, system_prompt: str, user_content: str,
                 temperature: float = 0.3, max_tokens: int = 200) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(model, quick: bool = False):
    """Run all test suites and return results."""
    interpreter_prompt = load_system_prompt("interpreter")
    coach_prompt = load_system_prompt("coach")

    results = {
        "interpreter_sanity": [],
        "coach_sanity": [],
        "interpreter_batch": [],
        "total_pass": 0,
        "total_fail": 0,
        "total_tests": 0,
    }

    # ─── Interpreter tests ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INTERPRETER MODE TESTS")
    print("=" * 60)

    for test in INTERPRETER_TESTS:
        t0 = time.time()
        response = model.generate(interpreter_prompt, test["input"], temperature=0.3)
        elapsed = time.time() - t0
        response = strip_thinking(response)

        ok, reason, parsed = validate_interpreter_json(response)

        # Check expected action
        action_ok = True
        if ok and parsed and "expect_action" in test:
            expected = test["expect_action"]
            if isinstance(expected, list):
                action_ok = parsed["action"] in expected
            else:
                action_ok = parsed["action"] == expected
            if not action_ok:
                ok = False
                reason = f"Expected action={expected}, got {parsed['action']}"

        # Check expected sport
        if ok and parsed and "expect_sport" in test:
            if parsed.get("sport") != test["expect_sport"]:
                ok = False
                reason = f"Expected sport={test['expect_sport']}, got {parsed.get('sport')}"

        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {test['label']} ({elapsed:.1f}s)")
        if not ok:
            print(f"         Reason: {reason}")
        print(f"         Response: {response[:150]}")

        results["interpreter_sanity"].append({
            "label": test["label"],
            "passed": ok,
            "reason": reason,
            "response": response[:300],
            "elapsed": elapsed,
        })

        if ok:
            results["total_pass"] += 1
        else:
            results["total_fail"] += 1
        results["total_tests"] += 1

    # ─── Coach tests ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COACH MODE TESTS")
    print("=" * 60)

    for test in COACH_TESTS:
        t0 = time.time()
        response = model.generate(coach_prompt, test["input"], temperature=0.5)
        elapsed = time.time() - t0
        response = strip_thinking(response)

        ok, reason = validate_coach_text(response)

        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {test['label']} ({elapsed:.1f}s)")
        if not ok:
            print(f"         Reason: {reason}")
        print(f"         Response: {response[:150]}")

        results["coach_sanity"].append({
            "label": test["label"],
            "passed": ok,
            "reason": reason,
            "response": response[:300],
            "elapsed": elapsed,
        })

        if ok:
            results["total_pass"] += 1
        else:
            results["total_fail"] += 1
        results["total_tests"] += 1

    # ─── Batch interpreter (schema validation on diverse prompts) ─────────
    if not quick:
        batch_prompts = load_batch_prompts(30)
        if batch_prompts:
            print("\n" + "=" * 60)
            print(f"  BATCH INTERPRETER TEST ({len(batch_prompts)} prompts)")
            print("=" * 60)

            batch_pass = 0
            batch_fail = 0

            for test in batch_prompts:
                t0 = time.time()
                response = model.generate(interpreter_prompt, test["input"], temperature=0.3)
                elapsed = time.time() - t0
                response = strip_thinking(response)

                ok, reason, parsed = validate_interpreter_json(response)

                if ok:
                    batch_pass += 1
                else:
                    batch_fail += 1
                    print(f"  [FAIL] {test['label']}: {reason}")
                    print(f"         Response: {response[:150]}")

                results["interpreter_batch"].append({
                    "label": test["label"],
                    "passed": ok,
                    "reason": reason,
                    "elapsed": elapsed,
                })

            results["total_pass"] += batch_pass
            results["total_fail"] += batch_fail
            results["total_tests"] += batch_pass + batch_fail

            print(f"\n  Batch: {batch_pass}/{batch_pass + batch_fail} valid JSON")

    return results


def print_summary(results: dict):
    """Print final test summary."""
    total = results["total_tests"]
    passed = results["total_pass"]
    failed = results["total_fail"]

    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    # Interpreter sanity
    ip = sum(1 for r in results["interpreter_sanity"] if r["passed"])
    it = len(results["interpreter_sanity"])
    print(f"  Interpreter sanity:  {ip}/{it}")

    # Coach sanity
    cp = sum(1 for r in results["coach_sanity"] if r["passed"])
    ct = len(results["coach_sanity"])
    print(f"  Coach sanity:        {cp}/{ct}")

    # Batch
    if results["interpreter_batch"]:
        bp = sum(1 for r in results["interpreter_batch"] if r["passed"])
        bt = len(results["interpreter_batch"])
        print(f"  Batch JSON schema:   {bp}/{bt}")

    # Average latency
    all_times = [r["elapsed"] for suite in ["interpreter_sanity", "coach_sanity", "interpreter_batch"]
                 for r in results.get(suite, []) if "elapsed" in r]
    if all_times:
        avg = sum(all_times) / len(all_times)
        print(f"  Avg latency:         {avg:.1f}s per prompt")

    print(f"\n  TOTAL: {passed}/{total} passed", end="")
    if failed > 0:
        print(f" ({failed} FAILED)")
    else:
        print(" — ALL PASSED")
    print("=" * 60)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="MiValta Josi v6 — Automated Model Tests")
    parser.add_argument("--model", help="Path to merged HuggingFace model directory")
    parser.add_argument("--gguf", help="Path to GGUF model file (requires llama-cpp-python)")
    parser.add_argument("--quick", action="store_true", help="Sanity checks only (skip batch)")

    args = parser.parse_args()

    if not args.model and not args.gguf:
        parser.error("Provide --model (HF path) or --gguf (GGUF file)")

    if args.model:
        backend = HFModel(args.model)
    else:
        parser.error("GGUF testing not yet implemented. Use --model with merged HF directory.")

    results = run_tests(backend, quick=args.quick)
    all_passed = print_summary(results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
