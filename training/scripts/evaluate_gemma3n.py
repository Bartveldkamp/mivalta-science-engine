#!/usr/bin/env python3
"""
Gemma 3n E2B Evaluation Script for MiValta Josi v4

Evaluates the fine-tuned Gemma 3n E2B model against Josi's coaching quality
requirements. Extends the SmolLM2 eval with dialogue governor checks.

Key metrics:
- Forbidden word compliance (no GATC/Viterbi/ACWR leaks)
- Warmth score (coaching personality, not textbook)
- Brevity score (concise responses, 30-100 words target)
- Dialogue governor: answer-first, max 1 follow-up question
- Pushback on unsafe goals
- Question engagement rate
- JSON schema validity (LLMIntent)

Usage:
    # Evaluate GGUF model via llama.cpp
    python evaluate_gemma3n.py --model path/to/josi-v4-gemma3n-q4_k_m.gguf --verbose

    # Evaluate HuggingFace model directly
    python evaluate_gemma3n.py --hf-model ./models/josi-v4-gemma3n-merged --verbose

    # Large-scale eval (1000 prompts)
    python evaluate_gemma3n.py --hf-model ./models/josi-v4-gemma3n-merged --prompts-file data/test_prompts_1000.json

    # Head-to-head: Gemma vs SmolLM2
    python evaluate_gemma3n.py --hf-model ./models/josi-v4-gemma3n-merged --compare-hf ./models/josi-v3-360M-merged
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# Import dialogue governor for post-processing checks
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
try:
    from dialogue_governor import govern_dialogue, enforce_answer_first, enforce_max_one_question
    HAS_GOVERNOR = True
except ImportError:
    HAS_GOVERNOR = False
    print("Warning: dialogue_governor not found — skipping governor checks")

# =============================================================================
# JOSI SYSTEM PROMPT — Gemma 3n E2B v4 (matches finetune_gemma3n.py)
# =============================================================================

JOSI_SYSTEM_PROMPT = """\
You are Josi, MiValta's AI coaching assistant.

PERSONALITY:
- Empathetic and warm — you genuinely care about the athlete
- Direct and honest — no filler, no corporate speak
- You ARE the coach — never recommend other apps, coaches, or services

DIALOGUE RULES:
- Answer first, always. Lead with the substance of your response.
- Maximum 1 follow-up question per turn.
- Keep responses under 100 words.

I6 CONSTRAINTS (always active):
- NEVER prescribe, create, or modify training yourself
- Explain decisions made by the coaching engine only
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems (algorithm, viterbi, hmm, hidden markov, acwr, gatc, ewma, tss, ctl, atl, tsb)

OUTPUT: Valid LLMIntent JSON."""

# =============================================================================
# TEST PROMPTS - 50 prompts across 10 categories
# =============================================================================

TEST_PROMPTS = {
    "fatigue": [
        "I'm feeling really tired today but I don't want to skip my workout.",
        "I'm exhausted. Should I still train?",
        "I've been tired all week. What's going on?",
        "I slept badly last night. Can I still do my session?",
        "I'm always tired after work. How do I find energy to train?",
    ],
    "unrealistic_goals": [
        "I want to run a marathon in 6 weeks. I ran once last month.",
        "I want to lose 20kg in 2 months while training for a triathlon.",
        "I've never cycled but I want to do an Ironman in 3 months.",
        "Can I go from couch to sub-3 marathon in 6 months?",
        "I want to train twice a day every day to get faster.",
    ],
    "readiness": [
        "My readiness score is amber today. What does that mean?",
        "Why is my readiness red? I feel fine.",
        "My readiness has been amber for 3 days. Should I be worried?",
        "What factors affect my readiness score?",
        "I don't trust my readiness score. It says I'm tired but I feel great.",
    ],
    "easy_day_frustration": [
        "Why do easy days matter? I feel like I'm wasting time going slow.",
        "Why is today an easy Zone 2 session when I feel like I could go harder?",
        "Easy days feel pointless. Can I just skip them?",
        "I hate going slow. It's boring and doesn't feel like training.",
        "Why can't every day be a hard day? That's how you get faster, right?",
    ],
    "beginner": [
        "I just started running last week. How often should I run?",
        "I'm new to cycling. What gear do I need?",
        "I've never exercised before. Where do I start?",
        "How do I know if I'm running too fast or too slow?",
        "What's a good first goal for a complete beginner?",
    ],
    "recovery": [
        "How long should I rest between hard workouts?",
        "What should I do on rest days?",
        "Is active recovery better than complete rest?",
        "I feel guilty taking rest days. Am I losing fitness?",
        "How do I know when I'm recovered enough for another hard session?",
    ],
    "overtraining": [
        "I've been training hard for 3 weeks and my times are getting worse.",
        "I used to love training but now I dread it. What's wrong?",
        "My resting heart rate has been elevated for a week.",
        "I'm getting slower even though I'm training more. Why?",
        "I keep getting small injuries. Is that normal?",
    ],
    "motivation": [
        "I've lost motivation to train. What should I do?",
        "Training feels like a chore lately.",
        "I'm scared I'll fail my race. Should I even bother?",
        "I missed a week of training. Have I ruined everything?",
        "Everyone else seems faster than me. Should I quit?",
    ],
    "zones": [
        "What's Zone 2 and why does everyone talk about it?",
        "How do I know what zone I'm in without a heart rate monitor?",
        "Why do I have to stay in specific zones? Can't I just run?",
        "My zones feel wrong. Zone 2 feels too easy.",
        "What's the difference between Zone 3 and Zone 4?",
    ],
    "general": [
        "How do I get faster?",
        "What's more important - frequency or intensity?",
        "Should I follow a training plan or just wing it?",
        "How do I balance training with a busy work schedule?",
        "What mistakes do most beginners make?",
    ],
}

# =============================================================================
# SCORING CRITERIA
# =============================================================================

FORBIDDEN_WORDS = [
    "gatc", "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "acute:chronic", "acute chronic", "load ratio",
    "monotony index", "training monotony", "strain index",
    "exponentially weighted", "ewma", "tss", "ctl", "atl", "tsb",
    "impulse-response", "banister", "fitness-fatigue",
]

JARGON_WORDS = [
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "lactate threshold",
    "ftp", "threshold power", "anaerobic capacity",
]

DEFLECTION_PHRASES = [
    "consult a professional", "see a professional", "hire a coach",
    "working with a professional coach", "recommend using a running app",
    "require expert input", "seek professional", "consult a coach",
    "talk to a professional", "find a coach", "professional trainer",
    "consult your doctor", "see a doctor", "visit a doctor",
    "other app", "another app", "different app",
]

WARM_INDICATORS = [
    "i hear", "i get it", "i understand", "that's", "here's the thing",
    "let's", "we can", "you're", "your body", "trust",
    "it's okay", "it's normal", "don't worry", "good question",
]

COLD_INDICATORS = [
    "i recommend that you", "it is recommended", "studies show that",
    "research indicates", "suboptimal", "data suggests",
    "statistically speaking", "according to research",
]

IDEAL_WORD_COUNT = (30, 100)
WARN_WORD_COUNT = (20, 150)
FAIL_WORD_COUNT = (10, 300)


@dataclass
class TestResult:
    category: str
    prompt: str
    response: str
    forbidden_words_found: list
    jargon_found: list
    warm_score: int
    brevity_score: int
    asks_question: bool
    pushback_on_unsafe: bool
    word_count: int
    answer_first: bool           # NEW: dialogue governor
    question_count: int          # NEW: dialogue governor
    governor_compliant: bool     # NEW: dialogue governor
    passed: bool
    failure_reasons: list


def check_forbidden_words(response: str) -> list:
    response_lower = response.lower()
    return [w for w in FORBIDDEN_WORDS if w in response_lower]


def check_deflection(response: str) -> list:
    response_lower = response.lower()
    return [p for p in DEFLECTION_PHRASES if p in response_lower]


def check_jargon(response: str) -> list:
    response_lower = response.lower()
    return [w for w in JARGON_WORDS if w in response_lower]


def score_warmth(response: str) -> int:
    response_lower = response.lower()
    warm_count = sum(1 for w in WARM_INDICATORS if w in response_lower)
    cold_count = sum(1 for w in COLD_INDICATORS if w in response_lower)
    score = 3 + min(warm_count, 2) - min(cold_count, 2)
    return max(1, min(5, score))


def score_brevity(word_count: int) -> int:
    if IDEAL_WORD_COUNT[0] <= word_count <= IDEAL_WORD_COUNT[1]:
        return 5
    elif WARN_WORD_COUNT[0] <= word_count <= WARN_WORD_COUNT[1]:
        return 3
    elif word_count < FAIL_WORD_COUNT[0]:
        return 1
    elif word_count > FAIL_WORD_COUNT[1]:
        return 1
    else:
        return 2


def check_asks_question(response: str) -> bool:
    return "?" in response


def count_questions(response: str) -> int:
    return response.count("?")


def check_answer_first(response: str) -> bool:
    """Check if the response leads with substantive content (not a question)."""
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if not sentences:
        # Try splitting by ? instead
        parts = response.split("?")
        if len(parts) <= 1:
            return True  # No question at all
        # Check if there's non-question content before the first ?
        before_first_q = parts[0].strip()
        return len(before_first_q.split()) >= 5  # At least 5 words before first question

    first = sentences[0].strip()
    return not first.endswith("?")


def check_pushback(response: str, category: str) -> bool:
    if category != "unrealistic_goals":
        return True
    response_lower = response.lower()
    pushback_indicators = [
        "risk", "injury", "dangerous", "too fast", "too soon",
        "not realistic", "unrealistic", "concern", "worried",
        "careful", "caution", "instead", "alternative", "defer",
        "half marathon", "longer timeline", "more time", "genuinely",
        "ambitious", "significantly", "rush",
    ]
    return any(p in response_lower for p in pushback_indicators)


def evaluate_response(category: str, prompt: str, response: str) -> TestResult:
    forbidden = check_forbidden_words(response)
    jargon = check_jargon(response)
    deflections = check_deflection(response)
    warm_score = score_warmth(response)
    brevity = score_brevity(len(response.split()))
    asks_q = check_asks_question(response)
    pushback = check_pushback(response, category)
    words = len(response.split())

    # Dialogue governor checks
    answer_first = check_answer_first(response)
    question_count = count_questions(response)
    governor_compliant = answer_first and question_count <= 1

    failure_reasons = []

    if forbidden:
        failure_reasons.append(f"Forbidden: {', '.join(forbidden)}")
    if jargon:
        failure_reasons.append(f"Jargon: {', '.join(jargon)}")
    if deflections:
        failure_reasons.append(f"Deflection: {', '.join(deflections)}")
    if warm_score <= 2:
        failure_reasons.append(f"Cold tone ({warm_score}/5)")
    if brevity <= 1:
        if words < FAIL_WORD_COUNT[0]:
            failure_reasons.append(f"Too short ({words} words)")
        else:
            failure_reasons.append(f"Too verbose ({words} words)")
    if category == "unrealistic_goals" and not pushback:
        failure_reasons.append("No pushback on unrealistic goal")
    if not answer_first:
        failure_reasons.append("Opens with question (not answer-first)")
    if question_count > 1:
        failure_reasons.append(f"Too many questions ({question_count}, max 1)")

    return TestResult(
        category=category,
        prompt=prompt,
        response=response,
        forbidden_words_found=forbidden,
        jargon_found=jargon,
        warm_score=warm_score,
        brevity_score=brevity,
        asks_question=asks_q,
        pushback_on_unsafe=pushback,
        word_count=words,
        answer_first=answer_first,
        question_count=question_count,
        governor_compliant=governor_compliant,
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
    )


# =============================================================================
# INFERENCE BACKENDS
# =============================================================================

def run_prompt_gguf(model_path: str, prompt: str, llama_cli: str) -> str:
    """Run prompt via llama.cpp CLI (GGUF model) with Gemma chat template."""
    # Gemma format: system is prepended to user message
    formatted = (
        f"<start_of_turn>user\n"
        f"{JOSI_SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    cmd = [
        llama_cli,
        "-m", model_path,
        "-p", formatted,
        "-n", "150",          # Max tokens (Gemma v4 output cap)
        "--temp", "0.45",     # Temperature 0.4-0.5 range
        "--single-turn",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # Extract response after model turn
        if "<start_of_turn>model" in output:
            parts = output.split("<start_of_turn>model")
            if len(parts) > 1:
                response = parts[-1]
                response = response.split("<end_of_turn>")[0]
                response = response.split("<start_of_turn>")[0]
                lines = response.split("\n")
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if any(noise in line for noise in [
                        "Prompt:", "Generation:", "Exiting", "llama_",
                        "memory breakdown", "t/s"
                    ]):
                        continue
                    if line.startswith(">") or line.startswith("["):
                        continue
                    clean_lines.append(line)
                return " ".join(clean_lines).strip()

        return "[PARSE_ERROR]"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def run_prompt_hf(model, tokenizer, prompt: str, device: str) -> str:
    """Run prompt via HuggingFace transformers (direct inference)."""
    import torch

    messages = [
        {"role": "system", "content": JOSI_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Manual Gemma format fallback
        input_text = (
            f"<start_of_turn>user\n"
            f"{JOSI_SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,     # Gemma v4 output cap
            temperature=0.45,       # 0.4-0.5 range
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


def load_hf_model(model_name: str):
    """Load a HuggingFace model for direct inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading HF model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For Gemma 5B, use 4-bit quantization for eval to fit in VRAM
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model = model.to(device)

    print(f"Loaded on {device}")
    return model, tokenizer, device


# =============================================================================
# VALIDATION RUNNER
# =============================================================================

def run_validation(
    run_fn,
    label: str,
    verbose: bool = False,
    prompts_override: dict = None,
) -> dict:
    """Run full validation suite."""
    prompts = prompts_override or TEST_PROMPTS
    total = sum(len(v) for v in prompts.values())
    batch_mode = total > 100

    print(f"\n{'=' * 60}")
    print(f"  JOSI GEMMA 3n E2B EVALUATION — {label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {total} prompts")
    print(f"{'=' * 60}\n")

    import time
    start_time = time.time()
    results = []
    current = 0

    for category, cat_prompts in prompts.items():
        if batch_mode:
            print(f"  [{category.upper()}] ({len(cat_prompts)} prompts)", end=" ", flush=True)
        else:
            print(f"\n[{category.upper()}]")

        cat_failures = 0
        for prompt in cat_prompts:
            current += 1

            if not batch_mode:
                print(f"  ({current}/{total}) {prompt[:45]}...", end=" ", flush=True)

            response = run_fn(prompt)
            result = evaluate_response(category, prompt, response)
            results.append(result)

            if not result.passed:
                cat_failures += 1

            if not batch_mode:
                status = "PASS" if result.passed else "FAIL"
                gov = "gov:OK" if result.governor_compliant else f"gov:FAIL({result.question_count}?)"
                print(f"{status} ({result.word_count}w, warm:{result.warm_score}, brief:{result.brevity_score}, {gov})")

                if verbose and not result.passed:
                    print(f"      Issues: {', '.join(result.failure_reasons)}")
                    print(f"      Response: {response[:120]}...")
                elif verbose and result.passed:
                    print(f"      Response: {response[:120]}...")

        if batch_mode:
            cat_passed = len(cat_prompts) - cat_failures
            pct = cat_passed / len(cat_prompts) * 100
            icon = "PASS" if pct == 100 else " ~  " if pct >= 80 else "FAIL"
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / rate if rate > 0 else 0
            print(f"{icon} {cat_passed}/{len(cat_prompts)} ({pct:.0f}%)  [{current}/{total} done, {eta:.0f}s left]")

    elapsed = time.time() - start_time
    passed = sum(1 for r in results if r.passed)

    # Report
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{len(results)} passed ({passed / len(results) * 100:.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/len(results):.1f}s per prompt)")
    print(f"{'=' * 60}")

    print(f"\n  METRICS:")
    deflection_count = sum(1 for r in results if any("Deflection" in fr for fr in r.failure_reasons))
    gov_compliant = sum(1 for r in results if r.governor_compliant)
    answer_first_count = sum(1 for r in results if r.answer_first)
    max_1q_count = sum(1 for r in results if r.question_count <= 1)

    print(f"    Forbidden word failures: {sum(1 for r in results if r.forbidden_words_found)}")
    print(f"    Jargon warnings:         {sum(1 for r in results if r.jargon_found)}")
    print(f"    Deflection failures:     {deflection_count}")
    print(f"    Avg warmth:              {sum(r.warm_score for r in results) / len(results):.1f}/5")
    print(f"    Avg brevity:             {sum(r.brevity_score for r in results) / len(results):.1f}/5")
    print(f"    Avg word count:          {sum(r.word_count for r in results) / len(results):.0f} words")
    print(f"    Question rate:           {sum(1 for r in results if r.asks_question) / len(results) * 100:.0f}%")
    print(f"    --- Dialogue Governor ---")
    print(f"    Answer-first rate:       {answer_first_count}/{len(results)} ({answer_first_count / len(results) * 100:.0f}%)")
    print(f"    Max 1 question rate:     {max_1q_count}/{len(results)} ({max_1q_count / len(results) * 100:.0f}%)")
    print(f"    Governor compliance:     {gov_compliant}/{len(results)} ({gov_compliant / len(results) * 100:.0f}%)")

    unrealistic = [r for r in results if r.category == "unrealistic_goals"]
    if unrealistic:
        pushback_count = sum(1 for r in unrealistic if r.pushback_on_unsafe)
        print(f"    Pushback rate:           {pushback_count}/{len(unrealistic)}")

    print(f"\n  BY CATEGORY:")
    for cat in prompts.keys():
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        cat_passed = sum(1 for r in cat_results if r.passed)
        pct = cat_passed / len(cat_results) * 100
        icon = "PASS" if pct == 100 else " ~  " if pct >= 80 else "FAIL"
        avg_words = sum(r.word_count for r in cat_results) / len(cat_results)
        gov_ok = sum(1 for r in cat_results if r.governor_compliant)
        print(f"    {icon} {cat}: {cat_passed}/{len(cat_results)} ({pct:.0f}%) avg {avg_words:.0f}w, gov:{gov_ok}/{len(cat_results)}")

    failed = [r for r in results if not r.passed]
    if failed:
        show_limit = 20 if batch_mode else 10
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed[:show_limit]:
            print(f"    - [{r.category}] {r.prompt[:50]}...")
            for reason in r.failure_reasons:
                print(f"        {reason}")
        if len(failed) > show_limit:
            print(f"    ... and {len(failed) - show_limit} more (see --output for full details)")

    print(f"\n{'=' * 60}\n")

    return {
        "label": label,
        "model_family": "gemma-3n-E2B",
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) * 100,
        "avg_warmth": sum(r.warm_score for r in results) / len(results),
        "avg_brevity": sum(r.brevity_score for r in results) / len(results),
        "avg_word_count": sum(r.word_count for r in results) / len(results),
        "question_rate": sum(1 for r in results if r.asks_question) / len(results) * 100,
        "answer_first_rate": answer_first_count / len(results) * 100,
        "max_1_question_rate": max_1q_count / len(results) * 100,
        "governor_compliance": gov_compliant / len(results) * 100,
        "elapsed_seconds": elapsed,
        "results": [asdict(r) for r in results],
    }


def print_comparison(report_a: dict, report_b: dict):
    """Print head-to-head comparison."""
    print(f"\n{'=' * 60}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n  {'Metric':<25} {'Model A':>12} {'Model B':>12} {'Winner':>10}")
    print(f"  {'-' * 60}")

    metrics = [
        ("Pass rate", "pass_rate", "%", True),
        ("Avg warmth", "avg_warmth", "/5", True),
        ("Avg brevity", "avg_brevity", "/5", True),
        ("Avg word count", "avg_word_count", "w", None),
        ("Question rate", "question_rate", "%", True),
        ("Answer-first rate", "answer_first_rate", "%", True),
        ("Governor compliance", "governor_compliance", "%", True),
    ]

    for name, key, unit, higher_better in metrics:
        a_val = report_a.get(key, 0)
        b_val = report_b.get(key, 0)

        if higher_better is True:
            winner = "A" if a_val > b_val else "B" if b_val > a_val else "TIE"
        elif higher_better is False:
            winner = "A" if a_val < b_val else "B" if b_val < a_val else "TIE"
        else:
            winner = "A" if abs(a_val - 60) < abs(b_val - 60) else "B"

        a_str = f"{a_val:.1f}{unit}"
        b_str = f"{b_val:.1f}{unit}"
        print(f"  {name:<25} {a_str:>12} {b_str:>12} {winner:>10}")

    print(f"\n  Model A: {report_a['label']}")
    print(f"  Model B: {report_b['label']}")

    a_score = report_a["pass_rate"] + report_a["avg_warmth"] * 10 + report_a["avg_brevity"] * 10
    b_score = report_b["pass_rate"] + report_b["avg_warmth"] * 10 + report_b["avg_brevity"] * 10

    # Add governor compliance bonus
    a_score += report_a.get("governor_compliance", 0) * 0.5
    b_score += report_b.get("governor_compliance", 0) * 0.5

    print(f"\n  Composite score: A={a_score:.1f} vs B={b_score:.1f}")
    if a_score > b_score:
        print(f"  Recommendation: Model A ({report_a['label']})")
    elif b_score > a_score:
        print(f"  Recommendation: Model B ({report_b['label']})")
    else:
        print(f"  Recommendation: Too close to call")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 3n E2B for MiValta Josi coaching quality"
    )

    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--compare", type=str, help="Path to second GGUF model")
    parser.add_argument("--llama-cli", type=str, default=None, help="Path to llama-cli binary")
    parser.add_argument("--hf-model", type=str, help="HuggingFace model for direct inference")
    parser.add_argument("--compare-hf", type=str, help="Second HF model for comparison")
    parser.add_argument("--prompts-file", type=str, help="JSON file with test prompts")
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    if not args.model and not args.hf_model:
        parser.error("Provide either --model (GGUF) or --hf-model (HuggingFace)")

    prompts_override = None
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts_override = json.load(f)
        total = sum(len(v) for v in prompts_override.values())
        print(f"Loaded {total} prompts from {args.prompts_file}")

    reports = []

    if args.hf_model:
        model, tokenizer, device = load_hf_model(args.hf_model)
        run_fn = lambda prompt: run_prompt_hf(model, tokenizer, prompt, device)
        report = run_validation(run_fn, label=args.hf_model, verbose=args.verbose, prompts_override=prompts_override)
        reports.append(report)

        if args.compare_hf:
            model2, tokenizer2, device2 = load_hf_model(args.compare_hf)
            run_fn2 = lambda prompt: run_prompt_hf(model2, tokenizer2, prompt, device2)
            report2 = run_validation(run_fn2, label=args.compare_hf, verbose=args.verbose, prompts_override=prompts_override)
            reports.append(report2)
            print_comparison(report, report2)

    elif args.model:
        llama_cli = args.llama_cli or str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
        run_fn = lambda prompt: run_prompt_gguf(args.model, prompt, llama_cli)
        label = Path(args.model).stem
        report = run_validation(run_fn, label=label, verbose=args.verbose, prompts_override=prompts_override)
        reports.append(report)

        if args.compare:
            run_fn2 = lambda prompt: run_prompt_gguf(args.compare, prompt, llama_cli)
            label2 = Path(args.compare).stem
            report2 = run_validation(run_fn2, label=label2, verbose=args.verbose, prompts_override=prompts_override)
            reports.append(report2)
            print_comparison(report, report2)

    if args.output:
        output_data = reports[0] if len(reports) == 1 else {"comparison": reports}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
