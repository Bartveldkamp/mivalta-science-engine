#!/usr/bin/env python3
"""
SmolLM2 Evaluation Script for MiValta Josi

Evaluates SmolLM2 (360M or 1.7B) against Josi's coaching quality requirements.
Adapted from validate_josi.py with SmolLM2 ChatML prompt format.

Supports two modes:
1. Single model evaluation (same as validate_josi.py but for SmolLM2)
2. Head-to-head comparison (360M vs 1.7B, or SmolLM2 vs Mistral)

Key metrics:
- Forbidden word compliance (no GATC/Viterbi/ACWR leaks)
- Warmth score (coaching personality, not textbook)
- Brevity score (Mistral pain point — penalizes verbose responses)
- Pushback on unsafe goals
- Question engagement rate

Usage:
    # Single model evaluation
    python evaluate_smollm2.py --model path/to/model.gguf --verbose

    # Head-to-head comparison
    python evaluate_smollm2.py --model path/to/smollm2-1.7b.gguf --compare path/to/smollm2-360m.gguf

    # Transformers-based evaluation (no GGUF needed, uses HF model directly)
    python evaluate_smollm2.py --hf-model HuggingFaceTB/SmolLM2-1.7B-Instruct --verbose

    # Compare base instruct models before fine-tuning
    python evaluate_smollm2.py --hf-model HuggingFaceTB/SmolLM2-1.7B-Instruct --compare-hf HuggingFaceTB/SmolLM2-360M-Instruct
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# =============================================================================
# JOSI SYSTEM PROMPT — same as in finetune_smollm2.py
# =============================================================================

JOSI_SYSTEM_PROMPT = (
    "You are Josi, a friendly and knowledgeable sports coaching assistant for MiValta. "
    "You communicate training decisions made by the coaching engine. "
    "Rules: Keep responses under 80 words. Be warm and conversational. "
    "Use simple language, not textbook explanations. Ask follow-up questions. "
    "Never invent training rules — only explain what the engine decided. "
    "Never mention algorithms, GATC, Viterbi, ACWR, or internal systems."
)

# =============================================================================
# TEST PROMPTS - 50 prompts across 10 categories (same as validate_josi.py)
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

# Brevity thresholds (key difference from Mistral validation)
IDEAL_WORD_COUNT = (30, 100)  # Target: 30-100 words per response
WARN_WORD_COUNT = (20, 150)   # Acceptable but not ideal
FAIL_WORD_COUNT = (10, 300)   # Hard limits


@dataclass
class TestResult:
    category: str
    prompt: str
    response: str
    forbidden_words_found: list
    jargon_found: list
    warm_score: int
    brevity_score: int  # 1-5 scale, NEW metric
    asks_question: bool
    pushback_on_unsafe: bool
    word_count: int
    passed: bool
    failure_reasons: list


def check_forbidden_words(response: str) -> list:
    response_lower = response.lower()
    return [w for w in FORBIDDEN_WORDS if w in response_lower]


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
    """Score response brevity (1-5). Higher = better for Josi's coaching style."""
    if IDEAL_WORD_COUNT[0] <= word_count <= IDEAL_WORD_COUNT[1]:
        return 5  # Perfect length
    elif WARN_WORD_COUNT[0] <= word_count <= WARN_WORD_COUNT[1]:
        return 3  # Acceptable
    elif word_count < FAIL_WORD_COUNT[0]:
        return 1  # Too short — probably broken
    elif word_count > FAIL_WORD_COUNT[1]:
        return 1  # Too verbose — Mistral problem
    else:
        return 2  # Borderline


def check_asks_question(response: str) -> bool:
    return "?" in response


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
    warm_score = score_warmth(response)
    brevity = score_brevity(len(response.split()))
    asks_q = check_asks_question(response)
    pushback = check_pushback(response, category)
    words = len(response.split())

    failure_reasons = []

    if forbidden:
        failure_reasons.append(f"Forbidden: {', '.join(forbidden)}")
    if jargon:
        failure_reasons.append(f"Jargon: {', '.join(jargon)}")
    if warm_score <= 2:
        failure_reasons.append(f"Cold tone ({warm_score}/5)")
    if brevity <= 1:
        if words < FAIL_WORD_COUNT[0]:
            failure_reasons.append(f"Too short ({words} words)")
        else:
            failure_reasons.append(f"Too verbose ({words} words) — Josi should be concise")
    if category == "unrealistic_goals" and not pushback:
        failure_reasons.append("No pushback on unrealistic goal")

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
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
    )


# =============================================================================
# INFERENCE BACKENDS
# =============================================================================

def run_prompt_gguf(model_path: str, prompt: str, llama_cli: str) -> str:
    """Run prompt via llama.cpp CLI (GGUF model)."""
    # SmolLM2 ChatML format
    formatted = (
        f"<|im_start|>system\n{JOSI_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    cmd = [
        llama_cli,
        "-m", model_path,
        "-p", formatted,
        "-n", "120",  # Max tokens — keep Josi concise
        "--temp", "0.7",
        "--single-turn",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # Extract response after assistant tag
        if "<|im_start|>assistant" in output:
            parts = output.split("<|im_start|>assistant")
            if len(parts) > 1:
                response = parts[-1]
                # Clean up end tokens and llama.cpp noise
                response = response.split("<|im_end|>")[0]
                response = response.split("<|im_start|>")[0]
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
    """Run prompt via HuggingFace transformers (direct model inference)."""
    import torch

    messages = [
        {"role": "system", "content": JOSI_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,  # Keep Josi concise
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip input)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


def load_hf_model(model_name: str):
    """Load a HuggingFace model for direct inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
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
) -> dict:
    """Run full validation suite using provided inference function."""
    print(f"\n{'=' * 60}")
    print(f"  JOSI SMOLLM2 EVALUATION — {label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}\n")

    results = []
    total = sum(len(v) for v in TEST_PROMPTS.values())
    current = 0

    for category, prompts in TEST_PROMPTS.items():
        print(f"\n[{category.upper()}]")

        for prompt in prompts:
            current += 1
            print(f"  ({current}/{total}) {prompt[:45]}...", end=" ", flush=True)

            response = run_fn(prompt)
            result = evaluate_response(category, prompt, response)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"{status} ({result.word_count}w, warm:{result.warm_score}, brief:{result.brevity_score})")

            if verbose and not result.passed:
                print(f"      Issues: {', '.join(result.failure_reasons)}")
                print(f"      Response: {response[:120]}...")
            elif verbose and result.passed:
                print(f"      Response: {response[:120]}...")

    # Report
    passed = sum(1 for r in results if r.passed)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{len(results)} passed ({passed / len(results) * 100:.1f}%)")
    print(f"{'=' * 60}")

    print(f"\n  METRICS:")
    print(f"    Forbidden word failures: {sum(1 for r in results if r.forbidden_words_found)}")
    print(f"    Jargon warnings:         {sum(1 for r in results if r.jargon_found)}")
    print(f"    Avg warmth:              {sum(r.warm_score for r in results) / len(results):.1f}/5")
    print(f"    Avg brevity:             {sum(r.brevity_score for r in results) / len(results):.1f}/5")
    print(f"    Avg word count:          {sum(r.word_count for r in results) / len(results):.0f} words")
    print(f"    Question rate:           {sum(1 for r in results if r.asks_question) / len(results) * 100:.0f}%")

    unrealistic = [r for r in results if r.category == "unrealistic_goals"]
    pushback_count = sum(1 for r in unrealistic if r.pushback_on_unsafe)
    print(f"    Pushback rate:           {pushback_count}/{len(unrealistic)}")

    print(f"\n  BY CATEGORY:")
    for cat in TEST_PROMPTS.keys():
        cat_results = [r for r in results if r.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        pct = cat_passed / len(cat_results) * 100
        icon = "PASS" if pct == 100 else " ~  " if pct >= 60 else "FAIL"
        avg_words = sum(r.word_count for r in cat_results) / len(cat_results)
        print(f"    {icon} {cat}: {cat_passed}/{len(cat_results)} ({pct:.0f}%) avg {avg_words:.0f}w")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed[:10]:
            print(f"    - [{r.category}] {r.prompt[:35]}...")
            for reason in r.failure_reasons:
                print(f"        {reason}")

    print(f"\n{'=' * 60}\n")

    return {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) * 100,
        "avg_warmth": sum(r.warm_score for r in results) / len(results),
        "avg_brevity": sum(r.brevity_score for r in results) / len(results),
        "avg_word_count": sum(r.word_count for r in results) / len(results),
        "question_rate": sum(1 for r in results if r.asks_question) / len(results) * 100,
        "results": [asdict(r) for r in results],
    }


def print_comparison(report_a: dict, report_b: dict):
    """Print head-to-head comparison of two model evaluations."""
    print(f"\n{'=' * 60}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n  {'Metric':<25} {'Model A':>12} {'Model B':>12} {'Winner':>10}")
    print(f"  {'-' * 60}")

    metrics = [
        ("Pass rate", "pass_rate", "%", True),
        ("Avg warmth", "avg_warmth", "/5", True),
        ("Avg brevity", "avg_brevity", "/5", True),
        ("Avg word count", "avg_word_count", "w", None),  # Lower can be better
        ("Question rate", "question_rate", "%", True),
    ]

    for name, key, unit, higher_better in metrics:
        a_val = report_a[key]
        b_val = report_b[key]

        if higher_better is True:
            winner = "A" if a_val > b_val else "B" if b_val > a_val else "TIE"
        elif higher_better is False:
            winner = "A" if a_val < b_val else "B" if b_val < a_val else "TIE"
        else:
            # Word count: closer to 60 (ideal) is better
            winner = "A" if abs(a_val - 60) < abs(b_val - 60) else "B"

        a_str = f"{a_val:.1f}{unit}"
        b_str = f"{b_val:.1f}{unit}"
        print(f"  {name:<25} {a_str:>12} {b_str:>12} {winner:>10}")

    print(f"\n  Model A: {report_a['label']}")
    print(f"  Model B: {report_b['label']}")

    # Overall recommendation
    a_score = report_a["pass_rate"] + report_a["avg_warmth"] * 10 + report_a["avg_brevity"] * 10
    b_score = report_b["pass_rate"] + report_b["avg_warmth"] * 10 + report_b["avg_brevity"] * 10

    print(f"\n  Composite score: A={a_score:.1f} vs B={b_score:.1f}")
    if a_score > b_score:
        print(f"  Recommendation: Model A ({report_a['label']})")
    elif b_score > a_score:
        print(f"  Recommendation: Model B ({report_b['label']})")
    else:
        print(f"  Recommendation: Too close to call — check qualitative responses")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolLM2 for MiValta Josi coaching quality"
    )

    # GGUF mode
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--compare", type=str, help="Path to second GGUF model for comparison")
    parser.add_argument("--llama-cli", type=str, default=None, help="Path to llama-cli binary")

    # HuggingFace mode
    parser.add_argument("--hf-model", type=str, help="HuggingFace model name for direct inference")
    parser.add_argument("--compare-hf", type=str, help="Second HF model for comparison")

    # Options
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    if not args.model and not args.hf_model:
        parser.error("Provide either --model (GGUF) or --hf-model (HuggingFace)")

    reports = []

    if args.hf_model:
        # HuggingFace direct inference mode
        model, tokenizer, device = load_hf_model(args.hf_model)
        run_fn = lambda prompt: run_prompt_hf(model, tokenizer, prompt, device)
        report = run_validation(run_fn, label=args.hf_model, verbose=args.verbose)
        reports.append(report)

        if args.compare_hf:
            model2, tokenizer2, device2 = load_hf_model(args.compare_hf)
            run_fn2 = lambda prompt: run_prompt_hf(model2, tokenizer2, prompt, device2)
            report2 = run_validation(run_fn2, label=args.compare_hf, verbose=args.verbose)
            reports.append(report2)
            print_comparison(report, report2)

    elif args.model:
        # GGUF/llama.cpp mode
        llama_cli = args.llama_cli or str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
        run_fn = lambda prompt: run_prompt_gguf(args.model, prompt, llama_cli)
        label = Path(args.model).stem
        report = run_validation(run_fn, label=label, verbose=args.verbose)
        reports.append(report)

        if args.compare:
            run_fn2 = lambda prompt: run_prompt_gguf(args.compare, prompt, llama_cli)
            label2 = Path(args.compare).stem
            report2 = run_validation(run_fn2, label=label2, verbose=args.verbose)
            reports.append(report2)
            print_comparison(report, report2)

    # Save reports
    if args.output:
        output_data = reports[0] if len(reports) == 1 else {"comparison": reports}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
